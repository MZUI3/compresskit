"""Physically remove the output channels that structured pruning zeroed, so the model gets smaller and faster.

``prune_filters`` only zeroes channels; ``remove_pruned_channels`` deletes them: the rows of the producing
Conv / Linear (and its bias), the matching BatchNorm entries, and the input columns of the layer that consumes
them. One traced forward pass decides what is safe. A channel is removed only when the producer's output flows
through channel-preserving modules (BatchNorm, activations, pooling, dropout, Flatten) into exactly one Conv
(``groups=1``) or Linear. Tensors with several consumers (residual additions, concatenations), grouped
convolutions and functional ops on the way are left as they are, still zeroed.

A zeroed channel is not silent: its bias and BatchNorm shift pass through the activation as a constant that the
consumer multiplies by its weights. With ``compensate_bias`` (the default) that constant, measured on
``example``, is folded into the consumer's bias before the columns are dropped, so the removal is exact for
Linear consumers and for convolutions without padding, and differs only at the padded border otherwise.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .._common import ConvNd, batch_input, is_mha_out_proj, model_device, to_device
from ..quantization.fold import BN_TYPES, Trace, trace_chains
from ..registry import register

PASS_THROUGH = (
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Mish, nn.ELU, nn.Sigmoid, nn.Tanh, nn.Hardswish,
    nn.Hardsigmoid, nn.Hardtanh, nn.Softplus, nn.Identity,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, nn.Upsample,
)

Chain = Tuple[List[nn.Module], nn.Module, int]


def _follow(channels: int, out_id: int, trace: Trace, consumers: Dict[int, list]) -> Optional[Chain]:
    """Walk from a producer's output to the single Conv / Linear that consumes it. Returns the BatchNorms on
    the way, the consumer and the number of flattened values per channel, or None when removal is unsafe."""
    cur, bns, block = out_id, [], 1
    for _ in range(32):
        if trace.uses.get(cur, 0) != 1:
            return None
        if any(not isinstance(t, PASS_THROUGH) for t in trace.transparent.get(cur, [])):
            return None
        nxt = consumers.get(cur, [])
        if len(nxt) != 1:
            return None
        mod, nout = nxt[0]
        if isinstance(mod, BN_TYPES):
            if mod.num_features != channels:
                return None
            bns.append(mod)
            cur = nout
            continue
        if isinstance(mod, nn.Flatten):
            shape = trace.shapes.get(cur)
            if mod.start_dim != 1 or shape is None or len(shape) < 2 or shape[1] != channels:
                return None
            block = math.prod(shape[2:]) if len(shape) > 2 else 1
            cur = nout
            continue
        if isinstance(mod, PASS_THROUGH):
            cur = nout
            continue
        if isinstance(mod, nn.Linear):
            return (bns, mod, block) if mod.in_features == channels * block else None
        if isinstance(mod, ConvNd):
            ok = mod.groups == 1 and mod.in_channels == channels and block == 1
            return (bns, mod, 1) if ok else None
        return None
    return None


@torch.no_grad()
def _consumer_input_means(model: nn.Module, example) -> Dict[int, torch.Tensor]:
    """``id(layer) -> per-input-channel mean`` of what every Linear / Conv receives on ``example``."""
    means: Dict[int, torch.Tensor] = {}
    hooks = []

    def pre(m, inp):
        x = inp[0] if inp else None
        if isinstance(x, torch.Tensor) and x.dim() >= 2:
            means[id(m)] = x.detach().float().mean([0] + list(range(2, x.dim())))

    for m in model.modules():
        if isinstance(m, (nn.Linear, ConvNd)):
            hooks.append(m.register_forward_pre_hook(pre))
    was_training = model.training
    model.eval()
    try:
        model(to_device(batch_input(example), model_device(model)))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return means


def _slice(p: Optional[torch.Tensor], idx: torch.Tensor, dim: int) -> None:
    if p is not None:
        p.data = p.data.index_select(dim, idx.to(p.device)).clone()


def _shrink_producer(m: nn.Module, keep: torch.Tensor, old: int) -> None:
    _slice(m.weight, keep, 0)
    _slice(m.bias, keep, 0)
    if isinstance(m, ConvNd):
        m.out_channels = len(keep)
    else:
        m.out_features = len(keep)
    wq = getattr(m, "wq", None)  # per-channel weight quantizer state, if the layer is quantized
    if wq is not None:
        for name in ("scale", "zero_point"):
            t = getattr(wq, name, None)
            if isinstance(t, torch.Tensor) and t.numel() == old:
                setattr(wq, name, t[keep.to(t.device)].clone())
        for name in ("s", "beta"):
            t = getattr(wq, name, None)
            if isinstance(t, torch.Tensor) and t.numel() == old:
                _slice(t, keep, 0)


def _shrink_bn(bn: nn.Module, keep: torch.Tensor) -> None:
    _slice(bn.weight, keep, 0)
    _slice(bn.bias, keep, 0)
    for name in ("running_mean", "running_var"):
        buf = getattr(bn, name, None)
        if buf is not None:
            setattr(bn, name, buf[keep.to(buf.device)].clone())
    bn.num_features = len(keep)


def _linear_cols(idx: torch.Tensor, block: int) -> torch.Tensor:
    return (idx[:, None] * block + torch.arange(block, device=idx.device)).flatten()


def _compensate(mod: nn.Module, dead: torch.Tensor, block: int, mean: Optional[torch.Tensor]) -> None:
    """Fold the constant input of the removed channels into the consumer's bias."""
    if mean is None or dead.numel() == 0:
        return
    w = mod.weight.detach()
    mean = mean.to(w.device)
    if isinstance(mod, nn.Linear):
        cols = _linear_cols(dead.to(w.device), block)
        delta = w[:, cols].float() @ mean[cols]
    else:
        k = w[0, 0].numel()
        delta = (w[:, dead.to(w.device)].flatten(1).float() * mean[dead.to(w.device)].repeat_interleave(k)).sum(1)
    if mod.bias is None:
        mod.bias = nn.Parameter(torch.zeros(w.shape[0], device=w.device, dtype=w.dtype))
    mod.bias.data += delta.to(mod.bias.dtype)


def _shrink_consumer(mod: nn.Module, keep: torch.Tensor, block: int) -> None:
    if isinstance(mod, nn.Linear):
        cols = _linear_cols(keep, block)
        _slice(mod.weight, cols, 1)
        mod.in_features = len(cols)
    else:
        _slice(mod.weight, keep, 1)
        mod.in_channels = len(keep)


@register("shrink", kind="other")
@torch.no_grad()
def remove_pruned_channels(model: nn.Module, example, *, min_keep: int = 1,
                           compensate_bias: bool = True) -> Dict[str, int]:
    """Delete the all-zero output channels of Conv / Linear layers (after ``prune_filters``), in place.

    ``example`` is one input batch used to trace the model (and, with ``compensate_bias``, to measure the
    constant each removed channel fed forward, which is folded into the consumer's bias). Returns
    ``{layer name: channels removed}``. Layers whose output is shared (residual blocks, concatenations) or
    consumed by something other than a single Conv / Linear are skipped and keep their zeros. At least
    ``min_keep`` channels stay in every layer; the final classifier's outputs are never removed.

    Do this after training: parameter shapes change, so optimizers and ``MaskSet`` objects built before are
    stale (detach the masks first). Weight-only quantized layers are not shrunk; quantize after shrinking."""
    names = {id(m): n for n, m in model.named_modules()}
    trace = trace_chains(model, example)
    means = _consumer_input_means(model, example) if compensate_bias else {}
    consumers: Dict[int, list] = {}
    for m, in_id, out_id in trace.calls:
        consumers.setdefault(in_id, []).append((m, out_id))
    removed: Dict[str, int] = {}
    for m, _, out_id in trace.calls:
        if not isinstance(m, (nn.Linear, ConvNd)) or is_mha_out_proj(m):
            continue
        if isinstance(m, ConvNd) and m.groups != 1:
            continue
        alive = m.weight.flatten(1).any(1)
        dead_count = int((~alive).sum())
        if dead_count == 0 or int(alive.sum()) < max(min_keep, 1):
            continue
        chain = _follow(alive.numel(), out_id, trace, consumers)
        if chain is None:
            continue
        bns, consumer, block = chain
        keep = alive.nonzero().flatten()
        dead = (~alive).nonzero().flatten()
        _compensate(consumer, dead, block, means.get(id(consumer)))
        _shrink_producer(m, keep, alive.numel())
        for bn in bns:
            _shrink_bn(bn, keep)
        _shrink_consumer(consumer, keep, block)
        removed[names.get(id(m), "?")] = dead_count
    return removed

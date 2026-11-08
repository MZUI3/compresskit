"""Physically remove the output channels that structured pruning zeroed, so the model gets smaller and faster.

``prune_filters`` only zeroes channels; ``remove_pruned_channels`` deletes them: the rows of the producing
Conv / Linear (and its bias), the matching BatchNorm entries, and the input columns of the layer that consumes
them. One traced forward pass decides what is safe. A channel is removed only when the producer's output flows
through channel-preserving modules (BatchNorm, activations, pooling, dropout, Flatten) into exactly one Conv
(``groups=1``) or Linear. Tensors with several consumers (residual additions, concatenations), grouped
convolutions and functional ops on the way are left as they are, still zeroed.

Removing a channel also drops the constant that its bias / BatchNorm shift fed into the next layer, so the
output changes slightly: fine-tune afterwards, as with every structured-pruning tool.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .._common import ConvNd, is_mha_out_proj
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


def _shrink_consumer(mod: nn.Module, keep: torch.Tensor, block: int) -> None:
    if isinstance(mod, nn.Linear):
        cols = (keep[:, None] * block + torch.arange(block, device=keep.device)).flatten()
        _slice(mod.weight, cols, 1)
        mod.in_features = len(cols)
    else:
        _slice(mod.weight, keep, 1)
        mod.in_channels = len(keep)


@register("shrink", kind="other")
@torch.no_grad()
def remove_pruned_channels(model: nn.Module, example, *, min_keep: int = 1) -> Dict[str, int]:
    """Delete the all-zero output channels of Conv / Linear layers (after ``prune_filters``), in place.

    ``example`` is one input batch used to trace the model. Returns ``{layer name: channels removed}``. Layers
    whose output is shared (residual blocks, concatenations) or consumed by something other than a single Conv /
    Linear are skipped and keep their zeros. At least ``min_keep`` channels stay in every layer; the final
    classifier's outputs are never removed.

    Do this after training: parameter shapes change, so optimizers and ``MaskSet`` objects built before are stale
    (detach the masks first). Weight-only quantized layers are not shrunk; quantize after shrinking."""
    names = {id(m): n for n, m in model.named_modules()}
    trace = trace_chains(model, example)
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
        dead = int((~alive).sum())
        if dead == 0 or int(alive.sum()) < max(min_keep, 1):
            continue
        chain = _follow(alive.numel(), out_id, trace, consumers)
        if chain is None:
            continue
        bns, consumer, block = chain
        keep = alive.nonzero().flatten()
        _shrink_producer(m, keep, alive.numel())
        for bn in bns:
            _shrink_bn(bn, keep)
        _shrink_consumer(consumer, keep, block)
        removed[names.get(id(m), "?")] = dead
    return removed

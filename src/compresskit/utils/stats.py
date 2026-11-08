"""What a model costs: parameters, zeros, storage bytes and MACs."""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .._common import batch_input, format_table, is_mha_out_proj, model_device, to_device, weight_of, weight_tensors
from ..quantization.layers import quant_layers
from ..quantization.weight_only import WEIGHT_ONLY_TYPES

SKIP_SUFFIXES = ("num_batches_tracked",)


def _weight_only_modules(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, WEIGHT_ONLY_TYPES)]


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters. The integer weights of weight-only layers live in buffers and are counted too
    (unless ``trainable_only``)."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
    if not trainable_only:
        n += sum(m.rows * m.cols for m in _weight_only_modules(model))
    return n


@torch.no_grad()
def count_nonzero(model: nn.Module) -> int:
    n = sum(int(p.count_nonzero()) for p in model.parameters())
    return n + sum(int(m.dequantize().count_nonzero()) for m in _weight_only_modules(model))


@torch.no_grad()
def weight_sparsity(model: nn.Module) -> float:
    """Fraction of exactly-zero values in the prunable weights (Conv, Linear, MultiheadAttention, and the
    dequantized weights of weight-only layers)."""
    total = zeros = 0
    tensors = [w for _, _, w in weight_tensors(model)] + [m.dequantize() for m in _weight_only_modules(model)]
    for w in tensors:
        total += w.numel()
        zeros += int((w == 0).sum())
    return zeros / max(total, 1)


def state_dict_bytes(model: nn.Module) -> int:
    """Bytes of every tensor in ``model.state_dict()``: roughly what ``torch.save`` writes."""
    return sum(t.numel() * t.element_size() for n, t in model.state_dict().items()
               if torch.is_tensor(t) and not n.endswith(SKIP_SUFFIXES))


def _weight_bits(model: nn.Module) -> Dict[str, int]:
    out = {}
    for name, m in quant_layers(model):
        bits = getattr(m.wq, "bits", 32)
        if bits < 32 and getattr(m.wq, "mode", "off") == "quant":
            out[f"{name}.weight" if name else "weight"] = bits
    return out


@torch.no_grad()
def estimated_bytes(model: nn.Module, sparse: bool = True) -> int:
    """Storage a deployment would need.

    Fake-quantized weights count at their quantizer's bit width, weight-only layers at the size of their packed
    buffers, everything else at its dtype size. With ``sparse`` a tensor with zeros is counted as its non-zeros
    plus a 1-bit mask when that is smaller than the dense tensor."""
    bits_of = _weight_bits(model)
    total_bits = 0
    for name, t in model.state_dict().items():
        if not torch.is_tensor(t) or name.endswith(SKIP_SUFFIXES):
            continue
        n = t.numel()
        bits = bits_of.get(name, t.element_size() * 8)
        if sparse and t.is_floating_point() and n > 0:
            nnz = int(t.count_nonzero())
            if nnz < n:
                total_bits += min(n * bits, nnz * bits + n)
                continue
        total_bits += n * bits
    return (total_bits + 7) // 8


def _density(m: nn.Module) -> float:
    w = weight_of(m)
    if w is None and hasattr(m, "dequantize") and not hasattr(m, "weight"):
        w = m.dequantize()
    if w is None:
        return 1.0
    return float((w != 0).float().mean())


def _is_conv_like(m: nn.Module) -> bool:
    return all(hasattr(m, a) for a in ("kernel_size", "in_channels", "out_channels", "groups"))


def _is_linear_like(m: nn.Module) -> bool:
    return hasattr(m, "in_features") and hasattr(m, "out_features") and not is_mha_out_proj(m)


@torch.no_grad()
def count_macs(model: nn.Module, example) -> Dict[str, float]:
    """Multiply-accumulates of one forward pass, per sample.

    Counted with forward hooks on Conv, Linear and MultiheadAttention layers (quantized and weight-only versions
    included), at the shapes the model actually sees. ``sparse_macs`` scales each layer by the density of its
    weight: the work a sparse kernel could skip, not what dense hardware does."""
    dense = [0.0]
    sparse = [0.0]
    hooks = []

    def conv_hook(m, inp, out):
        k = m.in_channels // m.groups * math.prod(m.kernel_size)
        macs = out.numel() * k
        dense[0] += macs
        sparse[0] += macs * _density(m)

    def linear_hook(m, inp, out):
        macs = out.numel() * m.in_features
        dense[0] += macs
        sparse[0] += macs * _density(m)

    def mha_hook(m, inp, out):
        x = inp[0]
        if x.dim() == 3:
            b, n = (x.shape[0], x.shape[1]) if m.batch_first else (x.shape[1], x.shape[0])
        else:
            b, n = 1, x.shape[0]
        e = x.shape[-1]
        tokens = b * n
        proj = tokens * e * 3 * e + tokens * e * e
        att = 2 * b * n * n * e
        dense[0] += proj + att
        sparse[0] += proj * _density(m) + att

    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))
        elif _is_conv_like(m):
            hooks.append(m.register_forward_hook(conv_hook))
        elif _is_linear_like(m):
            hooks.append(m.register_forward_hook(linear_hook))
    x = to_device(batch_input(example), model_device(model))
    batch = x.shape[0] if isinstance(x, torch.Tensor) and x.dim() > 0 else 1
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return {"macs": dense[0] / batch, "sparse_macs": sparse[0] / batch}


def summary(model: nn.Module, example=None, *, sparse: bool = True) -> Dict[str, float]:
    """Parameters, non-zeros, weight sparsity, state_dict and estimated storage (MB), and with ``example`` the
    MACs per sample."""
    out: Dict[str, float] = {
        "params": count_params(model),
        "nonzero_params": count_nonzero(model),
        "weight_sparsity": weight_sparsity(model),
        "state_dict_mb": state_dict_bytes(model) / 2 ** 20,
        "estimated_mb": estimated_bytes(model, sparse) / 2 ** 20,
    }
    if example is not None:
        out.update(count_macs(model, example))
    return out


def _fmt(key: str, v) -> str:
    if key in ("params", "nonzero_params"):
        return f"{v / 1e6:.3f}M" if v >= 1e6 else f"{v / 1e3:.1f}K"
    if key in ("macs", "sparse_macs"):
        return f"{v / 1e6:.2f}M"
    if key == "weight_sparsity":
        return f"{100 * v:.1f}%"
    if key.endswith("_mb"):
        return f"{v:.3f}"
    if key.endswith("_ms"):
        return f"{v:.3f}"
    return str(v)


def print_summary(model: nn.Module, example=None, **kwargs) -> None:
    s = summary(model, example, **kwargs)
    print(format_table(["metric", "value"], [(k, _fmt(k, v)) for k, v in s.items()]))


def compare(models: Dict[str, nn.Module], example=None, *, latency: bool = False, sparse: bool = True,
            **latency_kwargs) -> str:
    """A markdown table comparing several models, e.g. ``compare({"fp32": m, "pruned": p}, example=x)``.
    With ``latency`` a ``latency_ms`` column is measured too (keyword arguments go to ``latency_ms``)."""
    from .benchmark import latency_ms

    rows = []
    keys: Optional[list] = None
    for name, m in models.items():
        s = summary(m, example, sparse=sparse)
        if latency and example is not None:
            s["latency_ms"] = latency_ms(m, example, **latency_kwargs)
        keys = keys or list(s)
        rows.append([name] + [_fmt(k, s[k]) for k in keys])
    return format_table(["model"] + (keys or []), rows)

"""Weight-only integer quantization with real low-bit storage.

Weights are stored as packed unsigned integers (2, 4 or 8 bits) with one float scale (and zero point) per
output channel or per group of ``group_size`` input values, and dequantized on the fly in ``forward``.
Storage shrinks by 32/bits; compute stays in float, so the model runs on any device. This is the weight-only
scheme used for int8/int4 LLM checkpoints, here with plain round-to-nearest.
"""
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._common import ConvNd, Names, copy_model, is_mha_out_proj, replace_module, select_layers
from ..registry import register

BITS = (2, 4, 8)


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``q`` (rows x cols, integer codes in [0, 2^bits)) into uint8 along the last dim, 8/bits per byte."""
    vpb = 8 // bits
    rows, cols = q.shape
    pad = (-cols) % vpb
    if pad:
        q = F.pad(q, (0, pad))
    q = q.to(torch.uint8).reshape(rows, -1, vpb)
    packed = torch.zeros(q.shape[:2], dtype=torch.uint8, device=q.device)
    for i in range(vpb):
        packed |= q[:, :, i] << (bits * i)
    return packed


def unpack(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    vpb = 8 // bits
    mask = (1 << bits) - 1
    parts = [(packed >> (bits * i)) & mask for i in range(vpb)]
    return torch.stack(parts, -1).reshape(packed.shape[0], -1)[:, :cols]


@torch.no_grad()
def quantize_matrix(w: torch.Tensor, bits: int, group_size: Optional[int] = None,
                    symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight (rows x cols) per row, or per group of ``group_size`` values along the columns.

    Returns ``(packed, scale, zero_point)``: uint8 codes, ``scale`` of shape (rows, groups) and ``zero_point``
    of the same shape (asymmetric) or a scalar ``2^(bits-1)`` (symmetric: signed codes stored with an offset).
    ``dequantize_matrix`` inverts it. Exact zeros stay exactly zero."""
    if bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}, got {bits}")
    rows, cols = w.shape
    g = cols if group_size is None else int(group_size)
    if cols % g:
        raise ValueError(f"cols={cols} is not divisible by group_size={g}")
    wg = w.detach().float().reshape(rows, cols // g, g)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(wg / scale), -qmax, qmax) + 2 ** (bits - 1)
        zp = torch.tensor(float(2 ** (bits - 1)), device=w.device)
    else:
        qmax = 2 ** bits - 1
        lo = wg.amin(-1, keepdim=True).clamp_max(0.0)
        hi = wg.amax(-1, keepdim=True).clamp_min(0.0)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        zp = torch.round(-lo / scale).clamp(0, qmax)
        q = torch.clamp(torch.round(wg / scale) + zp, 0, qmax)
        zp = zp.squeeze(-1)
    packed = pack(q.reshape(rows, cols), bits)
    return packed, scale.squeeze(-1).to(w.dtype), zp.to(w.dtype)


def dequantize_matrix(packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int, cols: int,
                      group_size: Optional[int] = None) -> torch.Tensor:
    rows = packed.shape[0]
    g = cols if group_size is None else int(group_size)
    q = unpack(packed, bits, cols).to(scale.dtype).reshape(rows, cols // g, g)
    zp = zero_point.reshape(rows, cols // g, 1) if zero_point.dim() == 2 else zero_point
    s = scale.reshape(rows, cols // g, 1)
    return ((q - zp) * s).reshape(rows, cols)


class _WeightOnly(nn.Module):
    """Shared storage and dequantization of the weight-only layers."""

    def _init_storage(self, rows: int, cols: int, bits: int, group_size: Optional[int], symmetric: bool,
                      dtype, device) -> None:
        if bits not in BITS:
            raise ValueError(f"bits must be one of {BITS}, got {bits}")
        g = cols if group_size is None else int(group_size)
        if cols % g:
            raise ValueError(f"input size {cols} is not divisible by group_size={g}")
        self.bits, self.group_size, self.symmetric = bits, group_size, symmetric
        self.rows, self.cols = rows, cols
        groups = cols // g
        self.register_buffer("qweight", torch.zeros(rows, math.ceil(cols / (8 // bits)), dtype=torch.uint8,
                                                    device=device))
        self.register_buffer("scale", torch.ones(rows, groups, dtype=dtype, device=device))
        if symmetric:
            self.register_buffer("zero_point", torch.tensor(float(2 ** (bits - 1)), dtype=dtype, device=device))
        else:
            self.register_buffer("zero_point", torch.zeros(rows, groups, dtype=dtype, device=device))

    @torch.no_grad()
    def quantize_(self, weight2d: torch.Tensor) -> None:
        packed, scale, zp = quantize_matrix(weight2d, self.bits, self.group_size, self.symmetric)
        self.qweight.copy_(packed)
        self.scale.copy_(scale)
        self.zero_point.copy_(zp)

    def dequantize(self) -> torch.Tensor:
        """The (rows x cols) float weight matrix."""
        return dequantize_matrix(self.qweight, self.scale, self.zero_point, self.bits, self.cols, self.group_size)

    def _storage_repr(self) -> str:
        return f"bits={self.bits}, group_size={self.group_size}, symmetric={self.symmetric}"


class WeightOnlyLinear(_WeightOnly):
    """``nn.Linear`` with int2/4/8 weights in packed uint8 storage, dequantized in ``forward``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, bits: int = 8,
                 group_size: Optional[int] = None, symmetric: bool = True, device=None, dtype=torch.float32):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self._init_storage(out_features, in_features, bits, group_size, symmetric, dtype, device)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 8, group_size: Optional[int] = None,
                   symmetric: bool = True) -> "WeightOnlyLinear":
        w = linear.weight.detach()
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, bits=bits,
                group_size=group_size, symmetric=symmetric, device=w.device, dtype=w.dtype)
        m.quantize_(w)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.detach())
        return m

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, {self._storage_repr()}")


class WeightOnlyConv(_WeightOnly):
    """Conv1d/2d/3d with int2/4/8 weights (zero padding only). The weight (out, in/groups, *k) is quantized as
    an (out, in/groups * prod(k)) matrix, so groups run along the flattened kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 groups: int = 1, bias: bool = True, *, bits: int = 8, group_size: Optional[int] = None,
                 symmetric: bool = True, device=None, dtype=torch.float32):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        nd = len(self.kernel_size)
        self._conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[nd]
        cols = in_channels // groups * math.prod(self.kernel_size)
        self._init_storage(out_channels, cols, bits, group_size, symmetric, dtype, device)
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_float(cls, conv: nn.Module, bits: int = 8, group_size: Optional[int] = None,
                   symmetric: bool = True) -> "WeightOnlyConv":
        if conv.padding_mode != "zeros":
            raise ValueError(f"padding_mode={conv.padding_mode!r} is not supported, only 'zeros'")
        w = conv.weight.detach()
        m = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                conv.groups, conv.bias is not None, bits=bits, group_size=group_size, symmetric=symmetric,
                device=w.device, dtype=w.dtype)
        m.quantize_(w.flatten(1))
        if conv.bias is not None:
            m.bias.data.copy_(conv.bias.detach())
        return m

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize().reshape(self.out_channels, self.in_channels // self.groups, *self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(x, self.weight.to(x.dtype), self.bias, self.stride, self.padding, self.dilation,
                          self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, groups={self.groups}, bias={self.bias is not None}, "
                f"{self._storage_repr()}")


WEIGHT_ONLY_TYPES = (WeightOnlyLinear, WeightOnlyConv)


@register("weight_only", kind="quant")
def quantize_weight_only(model: nn.Module, bits: int = 8, group_size: Optional[int] = None, *,
                         symmetric: Optional[bool] = None, layers=(nn.Linear,), exclude: Names = (),
                         include: Names = None, inplace: bool = False) -> nn.Module:
    """Replace Linear (and, if listed in ``layers``, Conv) layers with weight-only quantized versions.

    Integer weights live in packed uint8 buffers (real storage reduction: 4x for int8, 8x for int4) and are
    dequantized in ``forward``; activations stay in float, so the model runs anywhere, with no speed-up.

    Args:
        bits: 8, 4 or 2.
        group_size: values per scale along the input dimension (128 is common for int4); ``None`` = one scale
            per output channel. Layers whose input size is not divisible fall back to per-channel.
        symmetric: signed symmetric grid; defaults to ``True`` for 8 bits and ``False`` (asymmetric with zero
            points, the usual int4 choice) below.
        layers: module types to convert, e.g. ``(nn.Linear, nn.Conv2d)``.
        exclude / include: layers to skip / restrict to (``"first"``, ``"last"``, names or prefixes).
    """
    if symmetric is None:
        symmetric = bits >= 8
    model = copy_model(model, inplace)
    for name, m in select_layers(model, tuple(layers), exclude, include):
        if isinstance(m, WEIGHT_ONLY_TYPES) or is_mha_out_proj(m):
            continue
        cols = m.weight[0].numel()
        g = group_size
        if g is not None and cols % g:
            warnings.warn(f"{name}: input size {cols} not divisible by group_size={g}; using per-channel scales")
            g = None
        if isinstance(m, nn.Linear):
            new = WeightOnlyLinear.from_float(m, bits, g, symmetric)
        elif isinstance(m, ConvNd):
            new = WeightOnlyConv.from_float(m, bits, g, symmetric)
        else:
            continue
        replace_module(model, name, new)
    return model

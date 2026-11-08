"""Range observers: they turn the values they see into a clipping range ``(lo, hi)``.

- ``minmax``: running min and max.
- ``ema``: exponential moving average of per-batch min and max (Jacob et al., 2018).
- ``percentile``: the ``percentile`` and ``100 - percentile`` quantiles of a sample of the values.
- ``mse``: the clipping range with the smallest quantization MSE (OMSE, Choukroun et al., 2019), searched over
  ``grid`` fractions of the min/max range.
- ``kl``: TensorRT's entropy calibration (Migacz, 2017): the clipping threshold that minimises the KL divergence
  between the histogram of |x| and its quantized version. Symmetric only.

Per-channel observers reduce over every axis except ``ch_axis``; activations are observed per tensor.
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch

KINDS = ("minmax", "ema", "percentile", "mse", "kl")
MAX_SAMPLES = 1 << 18


def range_to_qparams(lo: torch.Tensor, hi: torch.Tensor, qmin: int, qmax: int,
                     symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale and zero point of the integer grid [qmin, qmax] covering [lo, hi] (zero is always representable)."""
    lo = torch.minimum(lo, torch.zeros_like(lo))
    hi = torch.maximum(hi, torch.zeros_like(hi))
    if symmetric:
        bound = torch.maximum(-lo, hi)
        scale = bound / ((qmax - qmin) / 2)
        zp = torch.zeros_like(scale)
    else:
        scale = (hi - lo) / (qmax - qmin)
        zp = torch.round(qmin - lo / scale.clamp_min(1e-12)).clamp(qmin, qmax)
    return scale.clamp_min(1e-8), zp


def quant_error(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, qmin: int, qmax: int,
                symmetric: bool) -> torch.Tensor:
    """Per-row MSE of uniform quantization of ``x`` (rows x values) with ranges ``lo`` / ``hi`` (rows)."""
    scale, zp = range_to_qparams(lo, hi, qmin, qmax, symmetric)
    q = torch.clamp(torch.round(x / scale[:, None]) + zp[:, None], qmin, qmax)
    return ((q - zp[:, None]) * scale[:, None] - x).pow(2).mean(1)


def _rows(x: torch.Tensor, ch_axis: Optional[int]) -> torch.Tensor:
    if ch_axis is None:
        return x.reshape(1, -1)
    return x.transpose(0, ch_axis).reshape(x.shape[ch_axis], -1)


class Observer:
    def __init__(self, kind: str = "minmax", ch_axis: Optional[int] = None, percentile: float = 99.99,
                 grid: int = 80, momentum: float = 0.9):
        if kind not in KINDS:
            raise ValueError(f"unknown observer {kind!r}; choose from {KINDS}")
        self.kind = kind
        self.ch_axis = ch_axis
        self.percentile = percentile
        self.grid = grid
        self.momentum = momentum
        self.lo: Optional[torch.Tensor] = None
        self.hi: Optional[torch.Tensor] = None
        self.samples = []
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        rows = _rows(x, self.ch_axis)
        lo, hi = rows.min(1).values, rows.max(1).values
        if self.lo is None:
            self.lo, self.hi = lo, hi
        elif self.kind == "ema":
            m = self.momentum
            self.lo, self.hi = m * self.lo + (1 - m) * lo, m * self.hi + (1 - m) * hi
        else:
            self.lo, self.hi = torch.minimum(self.lo, lo), torch.maximum(self.hi, hi)
        if self.kind in ("percentile", "mse", "kl"):
            if self.ch_axis is None:
                flat = rows.flatten()
                keep = max(1, MAX_SAMPLES // 16)
                if flat.numel() > keep:
                    flat = flat[torch.randint(0, flat.numel(), (keep,), device=flat.device)]
                self.samples.append(flat)
                total = sum(s.numel() for s in self.samples)
                if total > MAX_SAMPLES:
                    allv = torch.cat(self.samples)
                    self.samples = [allv[torch.randperm(allv.numel(), device=allv.device)[:MAX_SAMPLES]]]
            else:
                self.samples = [rows]  # weights: one tensor, seen once
        self.count += 1

    @torch.no_grad()
    def compute(self, qmin: int, qmax: int, symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.lo is None:
            raise RuntimeError("the observer saw no data")
        if self.kind in ("minmax", "ema"):
            return self.lo, self.hi
        if self.ch_axis is None:
            data = torch.cat(self.samples).reshape(1, -1)
        else:
            data = torch.cat(self.samples, 1)
        if self.kind == "percentile":
            q = torch.tensor([1 - self.percentile / 100, self.percentile / 100], device=data.device)
            vals = torch.quantile(data[:, : 1 << 24], q, dim=1)
            return vals[0], vals[1]
        if self.kind == "mse":
            best_err = torch.full((data.shape[0],), float("inf"), device=data.device)
            best_lo, best_hi = self.lo.clone(), self.hi.clone()
            for i in range(1, self.grid + 1):
                f = i / self.grid
                lo, hi = self.lo * f, self.hi * f
                err = quant_error(data, lo, hi, qmin, qmax, symmetric)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_lo = torch.where(better, lo, best_lo)
                best_hi = torch.where(better, hi, best_hi)
            return best_lo, best_hi
        levels = (qmax - qmin + 1) // 2
        t = kl_threshold(data.abs().flatten().cpu().numpy(), levels)
        t = torch.tensor([t], device=data.device, dtype=torch.float32)
        return -t, t


def kl_threshold(values: np.ndarray, levels: int = 128, bins: int = 2048) -> float:
    """TensorRT entropy calibration on |x|.

    Exact zeros (half of a ReLU output) are left out: every grid represents them exactly, and as a spike in the
    first bin they would make every wider threshold look worse once that bin is merged with its neighbours."""
    values = values[values > 0]
    if values.size == 0:
        return 1e-8
    hist, edges = np.histogram(values, bins=bins, range=(0, float(values.max()) + 1e-12))
    hist = hist.astype(np.float64)
    best_kl, best_i = math.inf, bins
    for i in range(levels, bins + 1, 8 if bins > 1024 else 1):
        ref = hist[:i].copy()
        ref[i - 1] += hist[i:].sum()
        h = hist[:i]
        bounds = np.linspace(0, i, levels + 1).astype(np.int64)
        starts = bounds[:-1]
        nonempty = (h > 0).astype(np.float64)
        sums = np.add.reduceat(h, starts)
        counts = np.add.reduceat(nonempty, starts)
        avg = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        q = np.repeat(avg, np.diff(bounds)) * nonempty
        p = ref / ref.sum()
        qs = q.sum()
        if qs == 0:
            continue
        q = q / qs
        mask = p > 0
        kl = np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12)))
        if kl < best_kl:
            best_kl, best_i = kl, i
    return float(edges[best_i])

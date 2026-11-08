"""Reading sparsity off a model without a MaskSet."""
from typing import Dict

import torch
import torch.nn as nn

from .._common import Names, format_table, weight_tensors


@torch.no_grad()
def sparsity_report(model: nn.Module, exclude: Names = (), include: Names = None) -> Dict[str, float]:
    """Fraction of exactly-zero weights per prunable layer, plus ``"total"``; read from the weights themselves."""
    out: Dict[str, float] = {}
    total = zeros = 0
    for name, _, w in weight_tensors(model, exclude, include):
        z = int((w == 0).sum())
        out[name] = z / max(w.numel(), 1)
        total += w.numel()
        zeros += z
    out["total"] = zeros / max(total, 1)
    return out


def format_sparsity(report: Dict[str, float]) -> str:
    rows = [(n, f"{100 * s:.1f}%") for n, s in report.items()]
    return format_table(["weight", "sparsity"], rows)

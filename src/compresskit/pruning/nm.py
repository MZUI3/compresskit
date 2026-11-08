"""N:M semi-structured sparsity (2:4 by default).

In every group of M consecutive weights along the input dimension, the N with the largest magnitude are kept.
For a conv kernel (out, in, *k) the groups run along ``in`` at each kernel position, as in NVIDIA's ASP.
Ampere and newer GPUs run 2:4 fp16/bf16 Linear layers on sparse tensor cores
(``torch.sparse.to_sparse_semi_structured``); on other hardware the pattern only saves storage.
"""
from typing import Optional

import torch
import torch.nn as nn

from .._common import Names
from ..registry import register
from .masks import MaskSet


def nm_mask(w: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """The N:M mask of a weight (True = kept). Weights whose input dimension is not a multiple of ``m`` get an
    all-True mask (left dense)."""
    if n > m:
        raise ValueError(f"n={n} must not exceed m={m}")
    if w.dim() > 2:
        moved = w.movedim(1, -1)  # (out, *spatial, in)
        flat = moved.reshape(-1, w.shape[1])
    else:
        moved = None
        flat = w.reshape(w.shape[0], -1)
    cols = flat.shape[1]
    if cols % m:
        return torch.ones_like(w, dtype=torch.bool)
    groups = flat.detach().abs().reshape(-1, m)
    idx = groups.topk(n, dim=1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, idx, True).reshape(flat.shape)
    if moved is not None:
        return mask.reshape(moved.shape).movedim(-1, 1).contiguous()
    return mask.reshape(w.shape)


@register("nm", kind="prune")
def prune_nm(model: nn.Module, n: int = 2, m: int = 4, *, exclude: Names = (), include: Names = None,
             masks: Optional[MaskSet] = None) -> MaskSet:
    """N:M sparsity (Mishra et al., 2021): keep the ``n`` largest of every ``m`` consecutive input weights.

    The result is exactly ``1 - n/m`` sparse in every layer whose input dimension is a multiple of ``m``.
    Weights already pruned stay pruned (the pattern is applied on top of the existing masks)."""
    if masks is None:
        masks = MaskSet(model, exclude, include)
    elif masks.model is not model:
        raise ValueError("the given masks belong to a different model")
    for name, w in masks.targets.items():
        masks.set_mask(name, nm_mask(w, n, m) & masks.masks[name])
    return masks

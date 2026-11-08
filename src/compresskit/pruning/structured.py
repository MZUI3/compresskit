"""Structured pruning: whole output channels (conv filters, linear rows) are zeroed.

The masks keep the tensor shapes, so the model runs as before and dense hardware does not get faster; what you
get is a regular zero pattern. To physically remove the channels and shrink the layers that consume them, run a
dependency-graph tool such as Torch-Pruning on the masked model.
"""
from typing import Optional

import torch
import torch.nn as nn

from .._common import ConvNd, Names
from ..registry import register
from .masks import MaskSet

STRUCTURED_LAYERS = (nn.Linear, ConvNd)


def channel_scores(w: torch.Tensor, criterion: str = "l2") -> torch.Tensor:
    """One importance score per output channel (dim 0) of a weight: its L1 or L2 norm, or a random number."""
    flat = w.detach().flatten(1).float()
    if criterion == "l1":
        return flat.abs().sum(1)
    if criterion == "l2":
        return flat.norm(dim=1)
    if criterion == "random":
        return torch.rand(flat.shape[0], device=w.device)
    raise ValueError(f"criterion must be 'l1', 'l2' or 'random', got {criterion!r}")


def _expand(row: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return row.view(-1, *([1] * (w.dim() - 1))).expand_as(w)


@register("filters", kind="prune")
def prune_filters(model: nn.Module, ratio: float, criterion: str = "l2", scope: str = "layer", *,
                  exclude: Names = (), include: Names = None, masks: Optional[MaskSet] = None,
                  min_keep: int = 1) -> MaskSet:
    """Norm-based filter pruning (Li et al., 2017): zero the ``ratio`` of output channels with the smallest norm.

    Applies to Conv and Linear layers (``nn.MultiheadAttention`` is skipped). ``scope="layer"`` removes the same
    fraction from every layer; ``"global"`` ranks the raw norms of all layers together, so layers with small
    weights lose more channels. Channels that are already fully pruned stay pruned; at least ``min_keep``
    channels per layer are kept."""
    if masks is None:
        masks = MaskSet(model, exclude, include, layers=STRUCTURED_LAYERS)
    elif masks.model is not model:
        raise ValueError("the given masks belong to a different model")
    ratio = float(min(max(ratio, 0.0), 1.0))
    neg_inf = -float("inf")
    targets = {n: w for n, w in masks.targets.items()
               if w.dim() >= 2 and not isinstance(masks.modules[n], nn.MultiheadAttention)}
    scores = {}
    for n, w in targets.items():
        s = channel_scores(w, criterion)
        alive = masks.masks[n].flatten(1).any(1)
        s[~alive] = neg_inf
        scores[n] = s
    if scope == "layer":
        for n, s in scores.items():
            keep = min(max(int(round(s.numel() * (1 - ratio))), min_keep), s.numel())
            row = torch.zeros_like(s, dtype=torch.bool)
            row[torch.topk(s, keep).indices] = True
            masks.set_mask(n, _expand(row, targets[n]) & masks.masks[n])
    elif scope == "global":
        device = next(iter(scores.values())).device
        allscores = torch.cat([s.to(device) for s in scores.values()])
        keep = max(int(round(allscores.numel() * (1 - ratio))), 1)
        threshold = torch.topk(allscores, keep).values.min()
        for n, s in scores.items():
            row = (s >= threshold.to(s.device)) & (s > neg_inf)
            if int(row.sum()) < min_keep:
                row[torch.topk(s, min(min_keep, s.numel())).indices] = True
            masks.set_mask(n, _expand(row, targets[n]) & masks.masks[n])
    else:
        raise ValueError(f"scope must be 'layer' or 'global', got {scope!r}")
    return masks

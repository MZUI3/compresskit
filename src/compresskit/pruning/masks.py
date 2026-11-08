"""Boolean weight masks: what every pruning function returns."""
from typing import Callable, Dict, Iterator

import torch
import torch.nn as nn

from .._common import PRUNABLE, Names, format_table, weight_tensors

Scores = Dict[str, torch.Tensor]
Scorer = Callable[[str, torch.Tensor], torch.Tensor]


class MaskSet:
    """One boolean mask per prunable weight of a model (``True`` = kept).

    Every ``prune_*`` function returns a ``MaskSet`` and leaves the pruned weights at zero in the model. To keep
    them at zero while training, call ``attach()`` once: gradients of pruned weights are zeroed and the masks are
    re-applied before every forward pass, so momentum and weight decay cannot revive them. ``detach()`` removes
    those hooks. ``state_dict()`` / ``load_state_dict()`` save and restore the masks.

    Args:
        model: the model to prune; the masks reference its parameters directly.
        exclude: layers left dense: ``"first"``, ``"last"``, module names or name prefixes.
        include: when given, only these layers (names or prefixes) are prunable.
        layers: module types that count as prunable (default: Conv, Linear, MultiheadAttention).
    """

    def __init__(self, model: nn.Module, exclude: Names = (), include: Names = None, layers=PRUNABLE):
        self.model = model
        self.targets: Dict[str, torch.Tensor] = {}
        self.modules: Dict[str, nn.Module] = {}
        for name, module, w in weight_tensors(model, exclude, include, layers):
            self.targets[name] = w
            self.modules[name] = module
        if not self.targets:
            raise ValueError("no prunable layers found (Conv, Linear or MultiheadAttention)")
        self.masks: Dict[str, torch.Tensor] = {n: torch.ones_like(w, dtype=torch.bool)
                                               for n, w in self.targets.items()}
        self._handles: list = []

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self.masks)

    def __iter__(self) -> Iterator[str]:
        return iter(self.masks)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.masks[name]

    def __repr__(self) -> str:
        return f"MaskSet({len(self)} layers, sparsity={100 * self.sparsity():.1f}%, attached={self.attached})"

    @property
    def numel(self) -> int:
        return sum(m.numel() for m in self.masks.values())

    def nonzero(self) -> int:
        return sum(int(m.sum()) for m in self.masks.values())

    def sparsity(self) -> float:
        """Fraction of masked (zero) weights over all prunable weights."""
        return 1.0 - self.nonzero() / max(self.numel, 1)

    def layer_sparsity(self) -> Dict[str, float]:
        return {n: 1.0 - float(m.float().mean()) for n, m in self.masks.items()}

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    # ------------------------------------------------------------------ applying
    @torch.no_grad()
    def apply(self) -> None:
        """Multiply every weight by its mask (pruned weights become zero again)."""
        for n, w in self.targets.items():
            w.mul_(self.masks[n])

    @torch.no_grad()
    def set_mask(self, name: str, mask: torch.Tensor) -> None:
        w = self.targets[name]
        if tuple(mask.shape) != tuple(w.shape):
            raise ValueError(f"mask for {name} has shape {tuple(mask.shape)}, the weight has {tuple(w.shape)}")
        self.masks[name] = mask.to(device=w.device, dtype=torch.bool)
        w.mul_(self.masks[name])

    def scores(self, fn: Scorer) -> Scores:
        """``{name: fn(name, weight)}`` over the prunable weights."""
        return {n: fn(n, w) for n, w in self.targets.items()}

    @torch.no_grad()
    def prune_by_scores(self, scores: Scores, sparsity: float, scope: str = "global",
                        only_unpruned: bool = True) -> "MaskSet":
        """Keep the highest-scoring weights so that ``sparsity`` of the scored weights are zero.

        ``scope="global"`` ranks all layers together with one threshold; ``"layer"`` prunes every layer to the
        same sparsity. With ``only_unpruned`` already-pruned weights stay pruned (iterative pruning). At least one
        weight per layer is kept. Layers missing from ``scores`` are untouched. Returns self."""
        sparsity = float(min(max(sparsity, 0.0), 1.0))
        unknown = set(scores) - set(self.masks)
        if unknown:
            raise KeyError(f"scores for unknown weights: {sorted(unknown)}")
        neg_inf = -float("inf")
        prepared: Dict[str, torch.Tensor] = {}
        for n, s in scores.items():
            s = s.detach().to(self.masks[n].device).float().clone()
            if only_unpruned:
                s[~self.masks[n]] = neg_inf
            prepared[n] = s
        if scope == "layer":
            for n, s in prepared.items():
                k = max(int(round(s.numel() * (1 - sparsity))), 1)
                idx = torch.topk(s.flatten(), k).indices
                mask = torch.zeros(s.numel(), dtype=torch.bool, device=s.device)
                mask[idx] = True
                mask = mask.view_as(s) & (s > neg_inf)
                if not mask.any():
                    mask.view(-1)[s.flatten().argmax()] = True
                self.masks[n] = mask
        elif scope == "global":
            device = next(iter(prepared.values())).device
            allscores = torch.cat([s.flatten().to(device) for s in prepared.values()])
            k = max(int(round(allscores.numel() * (1 - sparsity))), 1)
            threshold = torch.topk(allscores, k).values.min()
            for n, s in prepared.items():
                mask = (s >= threshold.to(s.device)) & (s > neg_inf)
                if not mask.any():
                    mask.view(-1)[s.flatten().argmax()] = True
                self.masks[n] = mask
        else:
            raise ValueError(f"scope must be 'global' or 'layer', got {scope!r}")
        self.apply()
        return self

    # ------------------------------------------------------------------ training
    def attach(self) -> "MaskSet":
        """Keep pruned weights at zero during training: gradients are masked and the masks are re-applied
        before every forward pass of the model. Returns self."""
        if self._handles:
            return self
        for name, w in self.targets.items():
            if w.requires_grad:
                self._handles.append(w.register_hook(self._grad_hook(name)))
        self._handles.append(self.model.register_forward_pre_hook(self._pre_forward))
        return self

    def _grad_hook(self, name: str):
        def hook(grad):
            return grad * self.masks[name]

        return hook

    def _pre_forward(self, module, inputs):
        self.apply()

    def detach(self) -> "MaskSet":
        """Remove the training hooks. The weights stay as they are (zeros included)."""
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: m.clone() for n, m in self.masks.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> "MaskSet":
        missing = set(self.masks) - set(state)
        if missing:
            raise KeyError(f"masks missing for {sorted(missing)}")
        for n, m in state.items():
            if n in self.masks:
                self.set_mask(n, m)
        return self

    def report(self) -> str:
        """A markdown table of per-layer sparsity."""
        rows = [(n, "x".join(str(d) for d in self.targets[n].shape), f"{100 * s:.1f}%")
                for n, s in self.layer_sparsity().items()]
        rows.append(("total", str(self.numel), f"{100 * self.sparsity():.1f}%"))
        return format_table(["weight", "shape", "sparsity"], rows)

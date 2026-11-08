"""Gradual pruning while training (Zhu & Gupta, 2017)."""
from typing import Optional, Union

import torch.nn as nn

from .._common import Names
from .masks import MaskSet, Scorer
from .unstructured import magnitude_scores


def cubic_schedule(step: int, total_steps: int, final: float, initial: float = 0.0, start: float = 0.0,
                   end: float = 0.8) -> float:
    """Target sparsity at ``step``: s_t = s_f + (s_i - s_f) * (1 - (t - t0) / (t1 - t0))^3 between ``start``
    and ``end`` (fractions of ``total_steps``), ``initial`` before and ``final`` after."""
    t0, t1 = start * total_steps, end * total_steps
    if step <= t0:
        return initial
    if step >= t1:
        return final
    frac = (step - t0) / max(t1 - t0, 1e-12)
    return final + (initial - final) * (1 - frac) ** 3


class GradualPruner:
    """Raise sparsity from ``initial_sparsity`` to ``final_sparsity`` on the cubic schedule while you train.

        pruner = GradualPruner(model, final_sparsity=0.8, total_steps=len(loader) * epochs)
        for epoch in range(epochs):
            for x, y in loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pruner.step()
        masks = pruner.finish()

    The masks are attached for the whole run (gradients masked, weights re-masked before every forward).
    Every ``update_every`` steps the target sparsity is recomputed and the lowest-scoring unpruned weights are
    removed; ``criterion`` is ``"magnitude"`` or a callable ``(name, weight) -> scores``. ``finish()`` prunes to
    ``final_sparsity`` if the schedule did not reach it (short runs) and returns the ``MaskSet``, still attached.
    """

    def __init__(self, model: nn.Module, final_sparsity: float, total_steps: int, *,
                 initial_sparsity: float = 0.0, start: float = 0.0, end: float = 0.8, update_every: int = 100,
                 scope: str = "global", criterion: Union[str, Scorer] = "magnitude", exclude: Names = (),
                 include: Names = None, masks: Optional[MaskSet] = None):
        self.masks = masks if masks is not None else MaskSet(model, exclude, include)
        self.masks.attach()
        self.final = float(final_sparsity)
        self.initial = float(initial_sparsity)
        self.total_steps = int(total_steps)
        self.start, self.end = start, end
        self.update_every = max(int(update_every), 1)
        self.scope = scope
        self.scorer: Scorer = magnitude_scores if criterion == "magnitude" else criterion  # type: ignore
        if not callable(self.scorer):
            raise ValueError("criterion must be 'magnitude' or a callable (name, weight) -> scores")
        self.step_count = 0
        if self.initial > 0:
            self._prune_to(self.initial)

    def target(self, step: Optional[int] = None) -> float:
        return cubic_schedule(self.step_count if step is None else step, self.total_steps, self.final,
                              self.initial, self.start, self.end)

    def _prune_to(self, target: float) -> None:
        if target > self.masks.sparsity() + 1e-6:
            self.masks.prune_by_scores(self.masks.scores(self.scorer), target, self.scope)

    def step(self) -> float:
        """Call once per optimizer step. Returns the current target sparsity."""
        self.step_count += 1
        end_step = int(round(self.end * self.total_steps))
        if self.step_count % self.update_every == 0 or self.step_count == end_step:
            self._prune_to(self.target())
        return self.target()

    def finish(self) -> MaskSet:
        self._prune_to(self.final)
        return self.masks

    def __repr__(self) -> str:
        return (f"GradualPruner(step={self.step_count}/{self.total_steps}, target={100 * self.target():.1f}%, "
                f"current={100 * self.masks.sparsity():.1f}%)")

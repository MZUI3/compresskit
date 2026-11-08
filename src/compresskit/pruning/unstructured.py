"""Unstructured pruning: individual weights are zeroed, shapes stay the same.

Dense hardware does not run these models faster; the gain is storage (non-zeros plus a mask) and the work a
sparse kernel could skip. See ``prune_nm`` for the 2:4 pattern GPUs do accelerate and ``prune_filters`` for
structured pruning.
"""
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from .._common import Names, default_loss, model_device, take, to_device
from ..registry import register
from .masks import MaskSet, Scores

LossFn = Callable[[nn.Module, object], torch.Tensor]


def _masks(model: nn.Module, masks: Optional[MaskSet], exclude: Names, include: Names) -> MaskSet:
    if masks is None:
        return MaskSet(model, exclude, include)
    if masks.model is not model:
        raise ValueError("the given masks belong to a different model")
    return masks


def magnitude_scores(name: str, w: torch.Tensor) -> torch.Tensor:
    return w.detach().abs()


@register("magnitude", kind="prune")
def prune_magnitude(model: nn.Module, sparsity: float, scope: str = "global", *, exclude: Names = (),
                    include: Names = None, masks: Optional[MaskSet] = None) -> MaskSet:
    """Magnitude pruning (Han et al., 2015): zero the weights with the smallest |w|.

    Args:
        model: any ``nn.Module``; its Conv / Linear / MultiheadAttention weights are pruned in place.
        sparsity: fraction of the prunable weights to zero, 0..1.
        scope: ``"global"`` ranks all layers together (one threshold), ``"layer"`` prunes every layer to
            ``sparsity``.
        exclude / include: layers to skip / restrict to: ``"first"``, ``"last"``, module names or prefixes.
        masks: continue from an existing ``MaskSet`` (iterative pruning); pruned weights stay pruned.
    Returns:
        the ``MaskSet``. Call ``.attach()`` before fine-tuning to keep the zeros.
    """
    masks = _masks(model, masks, exclude, include)
    return masks.prune_by_scores(masks.scores(magnitude_scores), sparsity, scope)


@register("random", kind="prune")
def prune_random(model: nn.Module, sparsity: float, scope: str = "global", *, seed: Optional[int] = None,
                 exclude: Names = (), include: Names = None, masks: Optional[MaskSet] = None) -> MaskSet:
    """Random pruning: the baseline every importance criterion should beat."""
    masks = _masks(model, masks, exclude, include)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=model_device(model)).manual_seed(seed)
    scores = masks.scores(lambda n, w: torch.rand(w.shape, device=w.device, generator=gen))
    return masks.prune_by_scores(scores, sparsity, scope)


def prune_by_scores(model: nn.Module, scores: Scores, sparsity: float, scope: str = "global", *,
                    exclude: Names = (), include: Names = None, masks: Optional[MaskSet] = None) -> MaskSet:
    """Prune with your own importance scores: ``{parameter name: tensor shaped like the weight}``, higher = keep.
    Parameter names are ``"<module>.weight"`` (see ``MaskSet(model).targets``)."""
    masks = _masks(model, masks, exclude, include)
    return masks.prune_by_scores(scores, sparsity, scope)


def gradient_scores(model: nn.Module, batches: Iterable, criterion: str = "snip", *,
                    loss_fn: Optional[LossFn] = None, num_batches: Optional[int] = None,
                    masks: Optional[MaskSet] = None, exclude: Names = (), include: Names = None) -> Scores:
    """Importance scores that need gradients, averaged over ``batches``.

    - ``"snip"``: |w * dL/dw| (Lee et al., 2019; also the first-order Taylor saliency of removing w).
    - ``"fisher"``: 0.5 * w^2 * E[(dL/dw)^2]: Optimal Brain Damage (LeCun et al., 1990) with the empirical
      Fisher diagonal as the Hessian diagonal.

    ``loss_fn(model, batch)`` returns the loss; the default is cross-entropy on an ``(inputs, targets)`` batch.
    Batches are moved to the model's device. BatchNorm runs in eval mode so the scores do not depend on batch
    composition."""
    if criterion not in ("snip", "fisher"):
        raise ValueError(f"criterion must be 'snip' or 'fisher', got {criterion!r}")
    masks = _masks(model, masks, exclude, include)
    loss_fn = loss_fn or default_loss
    device = model_device(model)
    acc = {n: torch.zeros_like(w) for n, w in masks.targets.items()}
    frozen = [w for w in masks.targets.values() if not w.requires_grad]
    for w in frozen:
        w.requires_grad_(True)
    was_training = model.training
    model.eval()
    count = 0
    try:
        with torch.enable_grad():
            for batch in take(batches, num_batches):
                model.zero_grad(set_to_none=True)
                loss = loss_fn(model, to_device(batch, device))
                loss.backward()
                for n, w in masks.targets.items():
                    if w.grad is None:
                        continue
                    g = w.grad.detach()
                    if criterion == "snip":
                        acc[n] += (w.detach() * g).abs()
                    else:
                        acc[n] += g * g
                count += 1
    finally:
        model.zero_grad(set_to_none=True)
        for w in frozen:
            w.requires_grad_(False)
        model.train(was_training)
    if count == 0:
        raise ValueError("batches is empty")
    if criterion == "snip":
        return {n: a / count for n, a in acc.items()}
    return {n: 0.5 * a / count * masks.targets[n].detach() ** 2 for n, a in acc.items()}


@register("snip", kind="prune")
def prune_snip(model: nn.Module, batches: Iterable, sparsity: float, scope: str = "global", *,
               loss_fn: Optional[LossFn] = None, num_batches: Optional[int] = None, exclude: Names = (),
               include: Names = None, masks: Optional[MaskSet] = None) -> MaskSet:
    """SNIP (Lee et al., 2019): prune by |w * dL/dw| measured on a few batches.

    Works at initialisation (prune, then train from scratch) and on trained models. ``batches`` yields
    ``(inputs, targets)`` unless ``loss_fn(model, batch)`` is given."""
    masks = _masks(model, masks, exclude, include)
    scores = gradient_scores(model, batches, "snip", loss_fn=loss_fn, num_batches=num_batches, masks=masks)
    return masks.prune_by_scores(scores, sparsity, scope)


@register("obd", kind="prune")
def prune_obd(model: nn.Module, batches: Iterable, sparsity: float, scope: str = "global", *,
              loss_fn: Optional[LossFn] = None, num_batches: Optional[int] = None, exclude: Names = (),
              include: Names = None, masks: Optional[MaskSet] = None) -> MaskSet:
    """Optimal Brain Damage (LeCun et al., 1990): saliency 0.5 * H_ii * w_i^2 with the empirical Fisher
    diagonal standing in for the Hessian diagonal."""
    masks = _masks(model, masks, exclude, include)
    scores = gradient_scores(model, batches, "fisher", loss_fn=loss_fn, num_batches=num_batches, masks=masks)
    return masks.prune_by_scores(scores, sparsity, scope)

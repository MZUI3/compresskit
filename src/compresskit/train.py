"""A small training loop for when you do not have one: fine-tune after pruning, train a QAT model, evaluate.

    masks = ck.prune_magnitude(model, 0.8)
    ck.finetune(model, train_loader, epochs=3, lr=1e-3, masks=masks, val_loader=val_loader)

Batches are ``(inputs, targets)`` tuples with a cross-entropy loss unless you pass ``loss_fn(model, batch)``.
If you already train with PyTorch Lightning, use the callbacks in ``compresskit.lightning`` instead.
"""
import time
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import batch_input, default_loss, model_device, to_device
from .pruning.gradual import GradualPruner
from .pruning.masks import MaskSet
from .quantization.layers import quantizers
from .quantization.qat import model_parameters, quantizer_parameters
from .registry import register

LossFn = Callable[[nn.Module, object], torch.Tensor]
MetricFn = Callable[[nn.Module, object], float]


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, *, loss_fn: Optional[LossFn] = None,
             metric: Optional[MetricFn] = None, num_batches: Optional[int] = None) -> Dict[str, float]:
    """Mean loss (and top-1 accuracy) over ``loader``.

    By default batches are ``(inputs, targets)`` and the loss is cross-entropy; ``loss_fn(model, batch)``
    replaces it, in which case accuracy is reported only if ``metric(model, batch)`` (a per-batch mean) is
    given, under the key ``"metric"``."""
    was_training = model.training
    model.eval()
    device = model_device(model)
    sums = {"loss": 0.0, "accuracy": 0.0, "metric": 0.0}
    count = 0
    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        batch = to_device(batch, device)
        bs = len(batch_input(batch))
        if loss_fn is None:
            x, y = batch[0], batch[1]
            out = model(x)
            sums["loss"] += float(F.cross_entropy(out, y)) * bs
            sums["accuracy"] += float((out.argmax(-1) == y).float().sum())
        else:
            sums["loss"] += float(loss_fn(model, batch)) * bs
            if metric is not None:
                sums["metric"] += float(metric(model, batch)) * bs
        count += bs
    model.train(was_training)
    if count == 0:
        raise ValueError("loader is empty")
    out = {"loss": sums["loss"] / count}
    if loss_fn is None:
        out["accuracy"] = sums["accuracy"] / count
    elif metric is not None:
        out["metric"] = sums["metric"] / count
    return out


def _make_optimizer(model: nn.Module, optimizer: Union[str, torch.optim.Optimizer], lr: float, momentum: float,
                    weight_decay: float, quantizer_lr: Optional[float]) -> torch.optim.Optimizer:
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    qparams = [p for p in quantizer_parameters(model) if p.requires_grad]
    main = [p for p in (model_parameters(model) if qparams else model.parameters()) if p.requires_grad]
    groups = [{"params": main}]
    if qparams:  # learned step sizes: their own learning rate, no weight decay
        groups.append({"params": qparams, "lr": lr if quantizer_lr is None else quantizer_lr, "weight_decay": 0.0})
    name = optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'sgd', 'adam', 'adamw' or a torch.optim.Optimizer, got {optimizer!r}")


@register("finetune", kind="other")
def finetune(model: nn.Module, train_loader: Iterable, epochs: int = 1, *, lr: float = 1e-3,
             optimizer: Union[str, torch.optim.Optimizer] = "sgd", momentum: float = 0.9,
             weight_decay: float = 0.0, scheduler: Optional[str] = "cosine", loss_fn: Optional[LossFn] = None,
             masks: Optional[MaskSet] = None, pruner: Optional[GradualPruner] = None,
             val_loader: Optional[Iterable] = None, metric: Optional[MetricFn] = None,
             quantizer_lr: Optional[float] = None, grad_clip: Optional[float] = None,
             max_steps: Optional[int] = None, log: Optional[Callable] = print) -> List[Dict[str, float]]:
    """Train ``model`` on ``train_loader`` for ``epochs`` epochs with sensible defaults, in place.

    - ``masks``: a ``MaskSet`` to keep attached while training (pruned weights stay zero); detached afterwards
      unless it was attached before.
    - ``pruner``: a ``GradualPruner`` stepped once per optimizer step and finished at the end.
    - QAT models: lazily-initialised quantizers get one forward pass first, and learned step sizes (LSQ) go in
      their own parameter group with ``quantizer_lr`` and no weight decay.
    - ``optimizer``: ``"sgd"`` (momentum), ``"adam"``, ``"adamw"`` or a ready ``torch.optim.Optimizer``;
      ``scheduler="cosine"`` anneals the learning rate per step over the whole run, ``None`` keeps it constant.
    - ``loss_fn(model, batch)`` replaces the default cross-entropy on ``(inputs, targets)`` batches.

    Leaves the model in eval mode. Returns one dict per epoch (``train_loss``, ``lr``, ``time_s``, validation
    metrics, ``sparsity``)."""
    device = model_device(model)
    loss_fn = loss_fn or default_loss
    if any(getattr(q, "mode", "") == "init" for _, q in quantizers(model)):
        with torch.no_grad():  # lazily-initialised quantizers must exist before the optimizer is built
            was_training = model.training
            model.eval()
            model(batch_input(to_device(next(iter(train_loader)), device)))
            model.train(was_training)
    opt = _make_optimizer(model, optimizer, lr, momentum, weight_decay, quantizer_lr)
    steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
    total = epochs * steps_per_epoch if steps_per_epoch else None
    if max_steps is not None:
        total = min(total, max_steps) if total else max_steps
    sched = None
    if scheduler == "cosine" and total:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    elif scheduler not in (None, "cosine"):
        raise ValueError(f"scheduler must be 'cosine' or None, got {scheduler!r}")
    was_attached = masks.attached if masks is not None else False
    if masks is not None:
        masks.attach()
    history: List[Dict[str, float]] = []
    step = 0
    done = False
    try:
        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()
            loss_sum, n = 0.0, 0
            for batch in train_loader:
                batch = to_device(batch, device)
                loss = loss_fn(model, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                if pruner is not None:
                    pruner.step()
                loss_sum += float(loss)
                n += 1
                step += 1
                if max_steps is not None and step >= max_steps:
                    done = True
                    break
            rec: Dict[str, float] = {"epoch": epoch + 1, "train_loss": loss_sum / max(n, 1),
                                     "lr": opt.param_groups[0]["lr"], "time_s": time.perf_counter() - t0}
            if val_loader is not None:
                val = evaluate(model, val_loader, loss_fn=None if loss_fn is default_loss else loss_fn, metric=metric)
                rec.update({f"val_{k}": v for k, v in val.items()})
            active = pruner.masks if pruner is not None else masks
            if active is not None:
                rec["sparsity"] = active.sparsity()
            history.append(rec)
            if log is not None:
                log("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if done:
                break
    finally:
        if pruner is not None:
            pruner.finish()
        if masks is not None and not was_attached:
            masks.detach()
        model.eval()
    return history

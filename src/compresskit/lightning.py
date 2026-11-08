"""PyTorch Lightning callbacks: prune, keep masks fixed, or run QAT by adding a callback to the ``Trainer``.

    import lightning as L
    import compresskit.lightning as ckl

    trainer = L.Trainer(max_epochs=10, callbacks=[
        ckl.GradualPruning(final_sparsity=0.8, module_attr="model"),   # 0 -> 80% sparsity on the cubic schedule
        ckl.QAT(w_bits=8, a_bits=8, module_attr="model"),              # quantization-aware training, finalised at the end
    ])
    trainer.fit(lit_module, datamodule)

``module_attr`` names the network inside your ``LightningModule`` (``self.model`` -> ``"model"``); with ``None``
the whole LightningModule is compressed, which is fine when it only holds Linear / Conv layers you want
compressed. Works with ``lightning`` and with the older ``pytorch_lightning`` package.
"""
import math
from typing import Callable, Iterable, Optional, Union

import torch.nn as nn

try:
    from lightning.pytorch import Callback
except ImportError:  # pragma: no cover - depends on which package is installed
    try:
        from pytorch_lightning import Callback
    except ImportError as exc:
        raise ImportError("compresskit.lightning needs PyTorch Lightning: pip install lightning") from exc

from ._common import Names
from .pruning.gradual import GradualPruner
from .pruning.masks import MaskSet
from .quantization.qat import finalize_qat, prepare_qat, quantizer_parameters
from .registry import get

__all__ = ["Pruning", "GradualPruning", "QAT"]


def _target(pl_module: nn.Module, attr: Optional[str]) -> nn.Module:
    return getattr(pl_module, attr) if attr else pl_module


class Pruning(Callback):
    """Prune once (at the start of training, or at ``at_epoch``) and keep the masks fixed for the rest of fit.

    ``prune`` is a registered method name used as ``fn(model, amount, **kwargs)`` (``"magnitude"``,
    ``"random"``, ``"filters"``; ``"nm"`` takes ``n`` / ``m`` in ``kwargs`` instead of ``amount``), or a
    callable ``model -> MaskSet`` for anything else, e.g. ``lambda m: ck.prune_snip(m, batches, 0.8)``.
    Pass ``masks=`` instead to attach a ``MaskSet`` you already have. Logs ``sparsity`` every epoch; the masks
    are available as ``callback.masks`` and are detached when fit ends."""

    def __init__(self, prune: Union[str, Callable[[nn.Module], MaskSet], None] = None,
                 amount: Optional[float] = None, *, at_epoch: int = 0, masks: Optional[MaskSet] = None,
                 module_attr: Optional[str] = None, log: bool = True, **kwargs):
        super().__init__()
        if prune is None and masks is None:
            raise ValueError("give prune (a method name or a callable) or masks")
        if isinstance(prune, str) and prune != "nm" and amount is None:
            raise ValueError("amount (sparsity / ratio) is required with a method name")
        self.prune, self.amount, self.kwargs = prune, amount, kwargs
        self.at_epoch, self.masks, self.module_attr, self.log_sparsity = at_epoch, masks, module_attr, log

    def _run(self, model: nn.Module) -> None:
        if self.masks is None:
            if callable(self.prune):
                self.masks = self.prune(model)
            elif self.prune == "nm":
                self.masks = get("nm").fn(model, **self.kwargs)
            else:
                self.masks = get(self.prune).fn(model, self.amount, **self.kwargs)
        self.masks.attach()

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.at_epoch and (self.masks is None or not self.masks.attached):
            self._run(_target(pl_module, self.module_attr))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.log_sparsity and self.masks is not None:
            pl_module.log("sparsity", self.masks.sparsity(), on_epoch=True)

    def on_fit_end(self, trainer, pl_module):
        if self.masks is not None:
            self.masks.detach()


class GradualPruning(Callback):
    """Gradual magnitude pruning over the whole fit (Zhu & Gupta, 2017).

    Sparsity follows the cubic schedule from ``initial_sparsity`` to ``final_sparsity`` between fractions
    ``start`` and ``end`` of the optimizer steps (``trainer.estimated_stepping_batches``), updating every
    ``update_every`` steps; the masks stay attached so pruned weights stay zero, and are detached when training
    ends. ``criterion`` is ``"magnitude"`` or a callable ``(name, weight) -> scores``. Logs ``sparsity`` and
    ``target_sparsity`` every epoch; the final ``MaskSet`` is ``callback.masks``."""

    def __init__(self, final_sparsity: float, *, initial_sparsity: float = 0.0, start: float = 0.0,
                 end: float = 0.8, update_every: int = 100, scope: str = "global",
                 criterion: Union[str, Callable] = "magnitude", exclude: Names = (), include: Names = None,
                 module_attr: Optional[str] = None, log: bool = True):
        super().__init__()
        self.final, self.initial, self.start, self.end = final_sparsity, initial_sparsity, start, end
        self.update_every, self.scope, self.criterion = update_every, scope, criterion
        self.exclude, self.include, self.module_attr, self.log_sparsity = exclude, include, module_attr, log
        self.pruner: Optional[GradualPruner] = None
        self.masks: Optional[MaskSet] = None

    def on_train_start(self, trainer, pl_module):
        total = trainer.estimated_stepping_batches
        if total is None or not math.isfinite(total):
            raise ValueError("GradualPruning needs a sized train dataloader or Trainer(max_steps=...)")
        self.pruner = GradualPruner(_target(pl_module, self.module_attr), self.final, int(total),
                                    initial_sparsity=self.initial, start=self.start, end=self.end,
                                    update_every=self.update_every, scope=self.scope, criterion=self.criterion,
                                    exclude=self.exclude, include=self.include)
        self.masks = self.pruner.masks

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.pruner is not None and (batch_idx + 1) % max(trainer.accumulate_grad_batches, 1) == 0:
            self.pruner.step()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.log_sparsity and self.pruner is not None:
            pl_module.log("sparsity", self.pruner.masks.sparsity(), on_epoch=True)
            pl_module.log("target_sparsity", self.pruner.target(), on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        if self.pruner is not None:
            self.pruner.finish()
            self.pruner.masks.detach()


class QAT(Callback):
    """Quantization-aware training inside a Lightning fit.

    In ``setup`` the network gets fake quantizers (``prepare_qat``: ``method`` ``"ste"``, ``"lsq"`` or
    ``"lsq+"``). Activation ranges are calibrated on ``calib_batches`` if given, else on ``num_calib_batches``
    taken from the train dataloader (datamodule or LightningModule) when it is reachable, else on the first
    training batch. Learned step sizes that the optimizer does not know yet are added to it as their own
    parameter group (``quantizer_lr``, no weight decay). When fit ends, ``finalize_qat`` freezes the
    quantizers (``finalize=False`` to skip)."""

    def __init__(self, w_bits: int = 8, a_bits: int = 8, *, method: str = "ste",
                 calib_batches: Optional[Iterable] = None, num_calib_batches: int = 8,
                 quantizer_lr: Optional[float] = None, exclude: Names = (), include: Names = None,
                 module_attr: Optional[str] = None, fold_bn: bool = False, finalize: bool = True):
        super().__init__()
        self.w_bits, self.a_bits, self.method = w_bits, a_bits, method
        self.calib_batches, self.num_calib_batches, self.quantizer_lr = calib_batches, num_calib_batches, quantizer_lr
        self.exclude, self.include, self.module_attr = exclude, include, module_attr
        self.fold_bn, self.finalize = fold_bn, finalize
        self._prepared = False

    def _loader_batches(self, trainer, pl_module) -> Optional[list]:
        try:
            dm = getattr(trainer, "datamodule", None)
            loader = dm.train_dataloader() if dm is not None else pl_module.train_dataloader()
            return [b for _, b in zip(range(self.num_calib_batches), loader)]
        except Exception:  # no dataloader reachable at setup time: calibrate on the first training batch
            return None

    def setup(self, trainer, pl_module, stage=None):
        if self._prepared or stage not in (None, "fit"):
            return
        calib = self.calib_batches if self.calib_batches is not None else self._loader_batches(trainer, pl_module)
        prepare_qat(_target(pl_module, self.module_attr), calib, w_bits=self.w_bits, a_bits=self.a_bits,
                    method=self.method, exclude=self.exclude, include=self.include,
                    num_batches=self.num_calib_batches, fold_bn=self.fold_bn, inplace=True)
        self._prepared = True

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        qparams = quantizer_parameters(_target(pl_module, self.module_attr))
        if not qparams:
            return
        known = {id(p) for g in optimizer.param_groups for p in g["params"]}
        new = [p for p in qparams if id(p) not in known]
        if new:
            lr = optimizer.param_groups[0]["lr"] if self.quantizer_lr is None else self.quantizer_lr
            optimizer.add_param_group({"params": new, "lr": lr, "weight_decay": 0.0})

    def on_fit_end(self, trainer, pl_module):
        if self.finalize and self._prepared:
            finalize_qat(_target(pl_module, self.module_attr))

"""Quantization-aware training: fake quantizers with the straight-through estimator, trained with the network."""
from typing import Iterable, List, Optional

import torch.nn as nn

from .._common import Names, batch_input, copy_model, take
from ..registry import register
from .fold import fold_bn as _fold_bn
from .layers import attach_quantizers, calibrate, init_weight_quantizers, quant_layers, quantizers, set_mode
from .quantizer import DynamicWeightQuantizer, LSQQuantizer, NoQuant, Quantizer

METHODS = ("ste", "lsq", "lsq+")


@register("qat", kind="quant")
def prepare_qat(model: nn.Module, calib_batches: Optional[Iterable] = None, *, w_bits: int = 8, a_bits: int = 8,
                method: str = "ste", w_per_channel: bool = True, fold_bn: bool = False, exclude: Names = (),
                include: Names = None, num_batches: Optional[int] = None, inplace: bool = False) -> nn.Module:
    """Insert trainable fake quantizers. Train the model as usual, then call ``finalize_qat``.

    ``method``:
      - ``"ste"``: fixed uniform grids. Weight ranges follow the weights every step, activation ranges follow an
        EMA of the batch min/max (Jacob et al., 2018).
      - ``"lsq"``: Learned Step Size Quantization (Esser et al., 2020): the step sizes are parameters.
      - ``"lsq+"``: LSQ with learnable activation offsets (Bhalgat et al., 2020).

    With ``calib_batches`` the activation quantizers are calibrated now; otherwise they initialise on the first
    forward pass. Build the optimizer after that point so LSQ's step sizes exist, and give
    ``quantizer_parameters(model)`` their own param group (no weight decay). BatchNorm is kept by default so it
    keeps adapting during training; ``fold_bn=True`` folds it first (needs ``calib_batches``).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    model = copy_model(model, inplace)
    batches = take(calib_batches, num_batches) if calib_batches is not None else []
    if fold_bn:
        if not batches:
            raise ValueError("fold_bn=True needs calib_batches (one batch is enough)")
        _fold_bn(model, batch_input(batches[0]))

    def wq(name, layer):
        if w_bits >= 32:
            return NoQuant()
        if method == "ste":
            return DynamicWeightQuantizer(w_bits, per_channel=w_per_channel)
        return LSQQuantizer(w_bits, symmetric=True, per_channel=w_per_channel)

    def aq(name, layer):
        if a_bits >= 32:
            return NoQuant()
        if method == "ste":
            return Quantizer(a_bits, symmetric=False, per_channel=False, observer="minmax", ema_training=True)
        if method == "lsq":
            return LSQQuantizer(a_bits, symmetric=True, auto_signed=True)
        return LSQQuantizer(a_bits, symmetric=False, learn_offset=True)

    attach_quantizers(model, exclude=exclude, include=include, weight_quantizer=wq, act_quantizer=aq)
    init_weight_quantizers(model)
    if batches:
        calibrate(model, batches)
    else:
        set_mode(model, "init", which="act")
    return model


def quantizer_parameters(model: nn.Module) -> List[nn.Parameter]:
    """The learnable parameters of the quantizers (LSQ step sizes and offsets)."""
    out: List[nn.Parameter] = []
    for _, q in quantizers(model):
        out.extend(q.parameters(recurse=False))
    return out


def model_parameters(model: nn.Module) -> List[nn.Parameter]:
    """All parameters except the quantizers' (the complement of ``quantizer_parameters``)."""
    qids = {id(p) for p in quantizer_parameters(model)}
    return [p for p in model.parameters() if id(p) not in qids]


def finalize_qat(model: nn.Module, inplace: bool = True) -> nn.Module:
    """After training: freeze the quantizers and switch the model to eval.

    Weight ranges are recomputed from the final weights, learned step sizes are copied into ``scale`` /
    ``zero_point`` and EMA tracking stops. The result behaves like a PTQ model: evaluate it, ``export_onnx`` it or
    ``strip_quantizers`` it."""
    model = copy_model(model, inplace)
    model.eval()
    for _, layer in quant_layers(model):
        if isinstance(layer.wq, DynamicWeightQuantizer):
            layer.wq.init_from(layer.weight)
    for _, q in quantizers(model):
        if hasattr(q, "sync_buffers"):
            q.sync_buffers()
        if isinstance(q, Quantizer):
            q.ema_training = False
    return model

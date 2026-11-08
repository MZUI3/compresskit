"""Post-training quantization: fold BatchNorm, attach fake quantizers, calibrate on a few batches."""
from typing import Iterable, Optional

import torch.nn as nn

from .._common import Names, batch_input, copy_model, take
from ..registry import register
from .fold import fold_bn as _fold_bn
from .layers import attach_quantizers, calibrate, init_weight_quantizers, set_mode


@register("ptq", kind="quant")
def quantize_ptq(model: nn.Module, calib_batches: Optional[Iterable] = None, *, w_bits: int = 8, a_bits: int = 8,
                 w_observer: str = "minmax", a_observer: str = "minmax", w_per_channel: bool = True,
                 w_symmetric: bool = True, a_symmetric: bool = False, fold_bn: bool = True, exclude: Names = (),
                 include: Names = None, num_batches: Optional[int] = None, percentile: float = 99.99,
                 example=None, inplace: bool = False) -> nn.Module:
    """Post-training quantization with simulated (fake) quantization of weights and layer inputs.

    BatchNorm is folded into the preceding Conv / Linear, every Linear / Conv becomes a quantized layer, weight
    ranges are set from the weights and activation ranges are calibrated on ``calib_batches``. Without
    ``calib_batches`` the activation quantizers calibrate themselves on the first batch that runs through the
    model. The result runs on the model's device (CPU or GPU) and reports the accuracy an int8 deployment would
    get; ``export_onnx`` turns it into a Q/DQ graph and ``strip_quantizers`` gives plain layers back.

    Args:
        model: a trained model (BatchNorm running statistics are used for folding).
        calib_batches: inputs or ``(inputs, ...)`` tuples, typically 8-32 batches of training data.
        w_bits / a_bits: bit widths; 32 leaves weights / activations in float (``a_bits=32`` = weight-only).
        w_observer / a_observer: ``"minmax"``, ``"percentile"``, ``"mse"`` or ``"kl"``.
        w_per_channel: one scale per output channel for weights (default) or one per tensor.
        w_symmetric / a_symmetric: symmetric grids; activations default to asymmetric (suits ReLU outputs).
        fold_bn: fold BatchNorm first (needs ``calib_batches`` or ``example``).
        exclude / include: layers to keep in float / restrict to (``"first"``, ``"last"``, names or prefixes).
        num_batches: cap on the batches taken from ``calib_batches``.
        percentile: for the percentile observer.
        example: one input batch for BatchNorm folding when ``calib_batches`` is not given.
        inplace: modify ``model`` instead of a deep copy.
    """
    model = copy_model(model, inplace)
    was_training = model.training
    model.eval()
    batches = take(calib_batches, num_batches) if calib_batches is not None else []
    if fold_bn:
        sample = example if example is not None else (batch_input(batches[0]) if batches else None)
        if sample is None:
            raise ValueError("fold_bn needs calib_batches or example (one batch is enough); or pass fold_bn=False")
        _fold_bn(model, sample)
    attach_quantizers(model, w_bits, a_bits, w_observer=w_observer, a_observer=a_observer,
                      w_per_channel=w_per_channel, w_symmetric=w_symmetric, a_symmetric=a_symmetric,
                      exclude=exclude, include=include, percentile=percentile)
    init_weight_quantizers(model)
    if batches:
        calibrate(model, batches)
    else:
        set_mode(model, "init", which="act")
    model.train(was_training)
    return model

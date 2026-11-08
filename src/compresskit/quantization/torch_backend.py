"""Thin wrappers over ``torch.ao.quantization`` for real int8 kernels on the CPU.

PyTorch's eager-mode quantization is deprecated in favour of ``torchao``, but it still ships and it is the
quickest way to get genuine int8 speed-ups for Linear/LSTM-heavy models on x86 CPUs. The wrapper silences the
deprecation warnings and picks a working engine.
"""
import warnings
from typing import Optional

import torch
import torch.nn as nn

from .._common import copy_model
from ..registry import register


def _ao():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import torch.ao.quantization as ao
        except ImportError as exc:  # pragma: no cover - depends on the torch build
            raise ImportError("torch.ao.quantization is not available in this PyTorch build; "
                              "use quantize_weight_only or quantize_ptq instead") from exc
    return ao


def set_engine(engine: Optional[str] = None) -> str:
    """Select the quantized engine: the given one, or the first supported of fbgemm, x86, onednn, qnnpack."""
    supported = list(torch.backends.quantized.supported_engines)
    if engine is None:
        for candidate in ("fbgemm", "x86", "onednn", "qnnpack"):
            if candidate in supported:
                engine = candidate
                break
    if engine is None or engine not in supported:
        raise RuntimeError(f"no usable quantized engine (requested {engine!r}, supported {supported})")
    torch.backends.quantized.engine = engine
    return engine


@register("dynamic_int8", kind="quant")
def quantize_dynamic_int8(model: nn.Module, layers=(nn.Linear, nn.LSTM, nn.GRU), *, engine: Optional[str] = None,
                          inplace: bool = False) -> nn.Module:
    """Dynamic int8 quantization: int8 weights, activations quantized on the fly, real int8 kernels, CPU only.

    The returned model lives on the CPU in eval mode. Best for Linear/LSTM-heavy models (transformers, RNNs);
    convolutions are left in float."""
    ao = _ao()
    set_engine(engine)
    model = copy_model(model, inplace).cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ao.quantize_dynamic(model, set(layers), dtype=torch.qint8, inplace=True)

"""Quantized Linear / Conv layers (fake quantization of the weight and of the layer input) and the functions
that install, calibrate, inspect and remove them.

A quantized layer is the original module's class with a different ``forward``: it keeps the parameters and the
state_dict keys (``fc.weight`` stays ``fc.weight``) and gains two sub-modules, ``wq`` for the weight and ``aq``
for the input. Residual additions, normalisation and activations stay in float: this is the usual
"weights and layer inputs" placement.
"""
import copy
from contextlib import contextmanager
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._common import (QUANTIZABLE, Names, batch_input, copy_model, is_mha_out_proj, model_device,
                       replace_module, select_layers, take, to_device)
from .quantizer import NoQuant, Quantizer

QuantizerFactory = Callable[[str, nn.Module], nn.Module]


def _clone_into(src: nn.Module, cls: type, drop: Iterable[str] = ()) -> nn.Module:
    """A module of class ``cls`` that shares the parameters, buffers and attributes of ``src``."""
    m = cls.__new__(cls)
    nn.Module.__init__(m)
    for k, v in src.__dict__.items():
        if k in ("_parameters", "_buffers", "_modules"):
            continue
        m.__dict__[k] = copy.copy(v) if isinstance(v, (dict, list, set)) else v
    m._parameters = type(src._parameters)(src._parameters)
    m._buffers = type(src._buffers)(src._buffers)
    m._modules = type(src._modules)((k, v) for k, v in src._modules.items() if k not in drop)
    return m


class _Quantized:
    """Mixin for the quantized layers: ``wq`` quantizes the weight, ``aq`` the layer input."""

    wq: nn.Module
    aq: nn.Module
    trace_as_leaf = True  # the tracer in ``fold.py`` treats the layer as one unit despite its sub-modules

    @classmethod
    def from_float(cls, m: nn.Module, wq: nn.Module, aq: nn.Module):
        q = _clone_into(m, cls)
        q.wq = wq
        q.aq = aq
        return q

    def to_float(self, bake: bool = True) -> nn.Module:
        """The plain layer again. With ``bake`` the weight becomes its quantized-dequantized value."""
        m = _clone_into(self, _BASE[type(self)], drop=("wq", "aq"))
        if bake and getattr(self.wq, "mode", "off") == "quant":
            with torch.no_grad():
                m.weight.copy_(self.wq(self.weight))
        return m


class QLinear(_Quantized, nn.Linear):
    def forward(self, x):
        return F.linear(self.aq(x), self.wq(self.weight), self.bias)


class QConv1d(_Quantized, nn.Conv1d):
    def forward(self, x):
        return self._conv_forward(self.aq(x), self.wq(self.weight), self.bias)


class QConv2d(_Quantized, nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(self.aq(x), self.wq(self.weight), self.bias)


class QConv3d(_Quantized, nn.Conv3d):
    def forward(self, x):
        return self._conv_forward(self.aq(x), self.wq(self.weight), self.bias)


QUANT_TYPES = (QLinear, QConv1d, QConv2d, QConv3d)
_BASE = {QLinear: nn.Linear, QConv1d: nn.Conv1d, QConv2d: nn.Conv2d, QConv3d: nn.Conv3d}
_QCLASS = [(nn.Linear, QLinear), (nn.Conv1d, QConv1d), (nn.Conv2d, QConv2d), (nn.Conv3d, QConv3d)]


def _qclass_for(m: nn.Module):
    for base, q in _QCLASS:
        if isinstance(m, base):
            return q
    return None


def default_weight_quantizer(bits: int = 8, per_channel: bool = True, observer: str = "minmax",
                             symmetric: bool = True) -> QuantizerFactory:
    def make(name, layer):
        if bits >= 32:
            return NoQuant()
        return Quantizer(bits, symmetric=symmetric, per_channel=per_channel, ch_axis=0, observer=observer)

    return make


def default_act_quantizer(bits: int = 8, symmetric: bool = False, observer: str = "minmax",
                          **observer_kwargs) -> QuantizerFactory:
    def make(name, layer):
        if bits >= 32:
            return NoQuant()
        return Quantizer(bits, symmetric=symmetric, per_channel=False, observer=observer, **observer_kwargs)

    return make


def attach_quantizers(model: nn.Module, w_bits: int = 8, a_bits: int = 8, *, w_observer: str = "minmax",
                      a_observer: str = "minmax", w_per_channel: bool = True, w_symmetric: bool = True,
                      a_symmetric: bool = False, exclude: Names = (), include: Names = None,
                      weight_quantizer: Optional[QuantizerFactory] = None,
                      act_quantizer: Optional[QuantizerFactory] = None, **observer_kwargs) -> List[str]:
    """Replace every Linear / Conv (under ``include``, not under ``exclude``) with its quantized twin, in place.

    The new layers keep the original parameters and state_dict keys and gain ``wq`` (weight) and ``aq`` (input)
    quantizers, created in ``"off"`` mode: call ``init_weight_quantizers`` and ``calibrate`` (or ``set_mode``)
    next. ``weight_quantizer(name, layer)`` / ``act_quantizer(name, layer)`` override the default quantizers.
    Returns the names of the converted layers."""
    wq_factory = weight_quantizer or default_weight_quantizer(w_bits, w_per_channel, w_observer, w_symmetric)
    aq_factory = act_quantizer or default_act_quantizer(a_bits, a_symmetric, a_observer, **observer_kwargs)
    names = []
    for name, m in select_layers(model, QUANTIZABLE, exclude, include):
        if isinstance(m, QUANT_TYPES) or is_mha_out_proj(m):
            continue
        qcls = _qclass_for(m)
        if qcls is None:
            continue
        device = m.weight.device
        q = qcls.from_float(m, wq_factory(name, m).to(device), aq_factory(name, m).to(device))
        replace_module(model, name, q)
        names.append(name)
    return names


def quant_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """``(name, layer)`` of every quantized Linear / Conv in the model."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QUANT_TYPES)]


def quantizers(model: nn.Module, which: str = "all") -> List[Tuple[str, nn.Module]]:
    """``(name, quantizer)`` of the weight and/or activation quantizers (``which``: all / weight / act)."""
    out = []
    for n, m in quant_layers(model):
        if which in ("all", "weight") and not isinstance(m.wq, NoQuant):
            out.append((f"{n}.wq", m.wq))
        if which in ("all", "act") and not isinstance(m.aq, NoQuant):
            out.append((f"{n}.aq", m.aq))
    return out


@torch.no_grad()
def init_weight_quantizers(model: nn.Module) -> None:
    """Set every weight quantizer's range from its weight."""
    for _, m in quant_layers(model):
        m.wq.init_from(m.weight)


@torch.no_grad()
def calibrate(model: nn.Module, batches: Iterable, *, num_batches: Optional[int] = None, which: str = "act",
              forward: Optional[Callable] = None) -> int:
    """Run ``batches`` through the model with the quantizers observing, then fix their ranges.

    ``batches`` yields input tensors or ``(inputs, ...)`` tuples, moved to the model's device.
    ``forward(model, batch)`` overrides how a batch is fed to the model. Returns the number of batches used."""
    qs = [q for _, q in quantizers(model, which)]
    for q in qs:
        q.observe()
    device = model_device(model)
    was_training = model.training
    model.eval()
    count = 0
    for batch in take(batches, num_batches):
        batch = to_device(batch, device)
        if forward is not None:
            forward(model, batch)
        else:
            model(batch_input(batch))
        count += 1
    for q in qs:
        if getattr(q, "mode", "off") == "observe":
            if q.observer is not None and q.observer.lo is not None:
                q.finish()
            else:
                q.mode = "off"
    model.train(was_training)
    if count == 0:
        raise ValueError("the calibration batches are empty")
    return count


def set_mode(model: nn.Module, mode: str, which: str = "all") -> None:
    """Switch quantizers to ``"off"``, ``"observe"``, ``"init"`` (calibrate on the next batch) or ``"quant"``."""
    for _, q in quantizers(model, which):
        if mode == "observe":
            q.observe()
        elif mode == "init":
            q.lazy_init()
        elif mode in ("off", "quant"):
            q.mode = mode
        else:
            raise ValueError(f"unknown mode {mode!r}")


def strip_quantizers(model: nn.Module, bake: bool = True, inplace: bool = False) -> nn.Module:
    """Turn quantized layers back into plain ``nn.Linear`` / ``nn.Conv``.

    With ``bake`` (default) the weights keep their quantized values, so the float model computes what the
    fake-quantized one did for the weights; activation quantization is dropped. Useful before handing the model
    to tools that do not know these modules."""
    model = copy_model(model, inplace)
    for name, m in quant_layers(model):
        replace_module(model, name, m.to_float(bake))
    return model


def quant_summary(model: nn.Module) -> dict:
    layers = quant_layers(model)
    return {
        "quantized_layers": len(layers),
        "weight_bits": sorted({getattr(m.wq, "bits", 32) for _, m in layers}),
        "act_bits": sorted({getattr(m.aq, "bits", 32) for _, m in layers}),
        "layers": [n for n, _ in layers],
    }


@contextmanager
def export_mode(model: nn.Module):
    """Inside the block, 8-bit quantizers emit ``torch.fake_quantize_per_*_affine`` (ONNX exporters turn them
    into QuantizeLinear/DequantizeLinear). Learned quantizers sync their buffers first."""
    qs = [m for m in model.modules() if isinstance(m, Quantizer)]
    for q in qs:
        if hasattr(q, "sync_buffers"):
            q.sync_buffers()
        q.export = True
    try:
        yield model
    finally:
        for q in qs:
            q.export = False


def export_onnx(model: nn.Module, example, path: str, *, opset: int = 17, input_names=("input",),
                output_names=("output",), **kwargs) -> str:
    """Export a (fake-)quantized model to ONNX with QuantizeLinear/DequantizeLinear nodes (needs ``onnx``).

    Uses the TorchScript exporter (``dynamo=False``), where 8-bit fake quantizers map to Q/DQ; ONNX Runtime then
    reproduces the fake-quantized model to about 1e-2. Pass ``dynamo=True`` for the newer exporter (needs
    ``onnxscript``). Extra keyword arguments go to ``torch.onnx.export``. Runs on a CPU copy of the model."""
    import inspect

    m = copy.deepcopy(model).cpu().eval()
    x = batch_input(example)
    x = x.cpu() if isinstance(x, torch.Tensor) else x
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs.setdefault("dynamo", False)
    with export_mode(m), torch.no_grad():
        torch.onnx.export(m, (x,), path, opset_version=opset, input_names=list(input_names),
                          output_names=list(output_names), **kwargs)
    return path

"""BatchNorm folding and the forward-pass tracer it is built on.

A deployed int8 model runs folded weights, so post-training quantization folds every BatchNorm into the
Conv / Linear that feeds it before quantizing. ``trace_chains`` runs one forward pass and records, for every
leaf-module call, which module produced its input, the tensor shapes, and whether that tensor has any other
consumer; folding (and channel removal in ``compresskit.pruning.shrink``) touches only pairs whose intermediate
tensor has a single consumer, so residual branches are left alone.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from .._common import ConvNd, batch_input, model_device, to_device

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
FOLDABLE = (nn.Linear, ConvNd)


@dataclass
class Trace:
    """One traced forward pass.

    ``calls``: ``(module, input tensor id, output tensor id)`` per leaf-module call, in execution order.
    ``producer``: tensor id -> the module that returned it. ``uses``: tensor id -> number of consumers (modules
    and torch functions outside leaf modules). ``transparent``: tensor id -> modules that returned their input
    itself (in-place activations, Identity). ``shapes``: tensor id -> shape. ``tensors`` keeps the traced tensors
    alive so the ids stay valid."""

    calls: List[Tuple[nn.Module, int, int]] = field(default_factory=list)
    producer: Dict[int, nn.Module] = field(default_factory=dict)
    uses: Dict[int, int] = field(default_factory=dict)
    transparent: Dict[int, List[nn.Module]] = field(default_factory=dict)
    shapes: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    tensors: list = field(default_factory=list)


class _Recorder(TorchFunctionMode):
    """Counts how many torch functions consume each tensor outside leaf modules (residual adds, concats)."""

    def __init__(self, uses: Dict[int, int]):
        super().__init__()
        self.uses = uses
        self.depth = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.depth == 0:
            for a in list(args) + list(kwargs.values()):
                for t in (a if isinstance(a, (list, tuple)) else [a]):
                    if isinstance(t, torch.Tensor):
                        self.uses[id(t)] = self.uses.get(id(t), 0) + 1
        return func(*args, **kwargs)


@torch.no_grad()
def trace_chains(model: nn.Module, example) -> Trace:
    """Run ``example`` (a tensor or an ``(inputs, ...)`` batch) through ``model`` once and record the data flow
    between leaf modules. The model is put in eval mode for the pass and restored afterwards."""
    trace = Trace()
    rec = _Recorder(trace.uses)
    hooks = []

    def pre(m, inp):
        rec.depth += 1

    def post(m, inp, out):
        # Bookkeeping runs while depth is still > 0: reading ``.shape`` dispatches through the recorder too
        # and must not be counted as a use of the tensor.
        top = rec.depth == 1
        if top and inp and isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
            if out is inp[0]:
                trace.transparent.setdefault(id(out), []).append(m)
            else:
                trace.uses[id(inp[0])] = trace.uses.get(id(inp[0]), 0) + 1
                trace.producer[id(out)] = m
                trace.calls.append((m, id(inp[0]), id(out)))
                trace.shapes[id(inp[0])] = tuple(inp[0].shape)
                trace.shapes[id(out)] = tuple(out.shape)
                trace.tensors.extend([inp[0], out])
        rec.depth -= 1

    for m in model.modules():
        # Leaves are modules without children, plus modules that ask to be traced as one unit although they
        # hold sub-modules (quantized layers carry their quantizers as children).
        if m is not model and (len(list(m.children())) == 0 or getattr(m, "trace_as_leaf", False)):
            hooks.append(m.register_forward_pre_hook(pre))
            hooks.append(m.register_forward_hook(post))
    was_training = model.training
    model.eval()
    try:
        with rec:
            out = model(to_device(batch_input(example), model_device(model)))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    if isinstance(out, torch.Tensor):
        trace.uses[id(out)] = trace.uses.get(id(out), 0) + 1
        trace.tensors.append(out)
    return trace


def _replace(model: nn.Module, target: nn.Module, new: nn.Module) -> None:
    for _, m in model.named_modules():
        for k, child in list(m._modules.items()):
            if child is target:
                m._modules[k] = new


@torch.no_grad()
def fold_bn(model: nn.Module, example) -> int:
    """Fold every BatchNorm into the Linear / Conv that feeds it, in place; the BatchNorm becomes ``nn.Identity``.

    Only pairs whose intermediate tensor has no other consumer are folded. Uses the running statistics, so
    call it on a trained model. ``example`` is one input batch (a tensor or an ``(inputs, ...)`` tuple), needed
    to trace which layer feeds which. Returns the number of folded pairs."""
    trace = trace_chains(model, example)
    folded = 0
    for m, in_id, _ in trace.calls:
        if not isinstance(m, BN_TYPES) or m.running_var is None:
            continue
        prev = trace.producer.get(in_id)
        if not isinstance(prev, FOLDABLE) or trace.uses.get(in_id, 0) != 1:
            continue
        if prev.weight.shape[0] != m.num_features:
            continue
        std = (m.running_var + m.eps).sqrt()
        gamma = m.weight if m.weight is not None else torch.ones_like(std)
        beta = m.bias if m.bias is not None else torch.zeros_like(std)
        factor = gamma / std
        prev.weight.mul_(factor.view(-1, *([1] * (prev.weight.dim() - 1))))
        bias = prev.bias if prev.bias is not None else torch.zeros_like(m.running_mean)
        prev.bias = nn.Parameter((bias - m.running_mean) * factor + beta)
        _replace(model, m, nn.Identity())
        folded += 1
    return folded

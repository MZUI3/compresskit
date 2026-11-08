"""BatchNorm folding.

A deployed int8 model runs folded weights, so post-training quantization folds every BatchNorm into the
Conv / Linear that feeds it before quantizing. ``trace_chains`` runs one forward pass and records, for every
leaf-module call, which module produced its input and whether that tensor has any other consumer; folding
touches only pairs whose intermediate tensor has a single consumer, so residual branches are left alone.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from .._common import ConvNd, batch_input, model_device, to_device

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
FOLDABLE = (nn.Linear, ConvNd)


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
def trace_chains(model: nn.Module, example) -> Tuple[List[Tuple[nn.Module, int, int]], Dict[int, nn.Module],
                                                     Dict[int, int], Dict[int, List[nn.Module]], list]:
    """One traced forward pass.

    Returns ``(calls, producer, uses, transparent, tensors)``: ``calls`` = ``(module, input id, output id)`` per
    leaf-module call in execution order, ``producer`` = {tensor id: module that returned it}, ``uses`` =
    {tensor id: number of consumers}, ``transparent`` = {tensor id: modules that returned their input itself}
    (in-place activations, Identity), ``tensors`` keeps the traced tensors alive so the ids stay valid."""
    calls, producer, uses, keep, transparent = [], {}, {}, [], {}
    rec = _Recorder(uses)
    hooks = []

    def pre(m, inp):
        rec.depth += 1

    def post(m, inp, out):
        rec.depth -= 1
        if rec.depth == 0 and inp and isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
            if out is inp[0]:
                transparent.setdefault(id(out), []).append(m)
                return
            uses[id(inp[0])] = uses.get(id(inp[0]), 0) + 1
            producer[id(out)] = m
            calls.append((m, id(inp[0]), id(out)))
            keep.extend([inp[0], out])

    for m in model.modules():
        if m is not model and len(list(m.children())) == 0:
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
        uses[id(out)] = uses.get(id(out), 0) + 1
        keep.append(out)
    return calls, producer, uses, transparent, keep


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
    calls, producer, uses, _, _keep = trace_chains(model, example)
    folded = 0
    for m, in_id, _ in calls:
        if not isinstance(m, BN_TYPES) or m.running_var is None:
            continue
        prev = producer.get(in_id)
        if not isinstance(prev, FOLDABLE) or uses.get(in_id, 0) != 1:
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

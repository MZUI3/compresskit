"""Run several methods in sequence by name.

    result = compress(model, [
        ("magnitude", {"sparsity": 0.5}),
        ("ptq", {"calib_batches": calib, "w_bits": 8, "a_bits": 8}),
    ])
    result.model, result.masks, result.steps
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch.nn as nn

from ._common import copy_model
from .pruning.masks import MaskSet
from .registry import get

Step = Union[str, Tuple[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class CompressResult:
    model: nn.Module
    masks: Optional[MaskSet] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)


def _parse(step: Step) -> Tuple[str, Dict[str, Any]]:
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict):
        d = dict(step)
        return d.pop("method"), d
    name, kwargs = step
    return name, dict(kwargs or {})


def compress(model: nn.Module, steps: Iterable[Step], *, inplace: bool = False,
             verbose: bool = False) -> CompressResult:
    """Apply registered methods in order to (a copy of) ``model``.

    Each step is a name, ``(name, kwargs)`` or ``{"method": name, **kwargs}``. Pruning steps share one ``MaskSet``
    (each continues from the previous masks); quantization steps run in place on the working copy. Returns the
    model, the masks (``None`` if nothing was pruned) and one dict per step."""
    model = copy_model(model, inplace)
    result = CompressResult(model)
    for step in steps:
        name, kwargs = _parse(step)
        method = get(name)
        if method.kind == "prune":
            if result.masks is not None and "masks" not in kwargs:
                kwargs["masks"] = result.masks
            result.masks = method.fn(result.model, **kwargs)
            info = {"method": method.name, "sparsity": result.masks.sparsity()}
        elif method.kind == "quant":
            kwargs.setdefault("inplace", True)
            result.model = method.fn(result.model, **kwargs)
            info = {"method": method.name}
        else:
            out = method.fn(result.model, **kwargs)
            if isinstance(out, nn.Module):
                result.model = out
            info = {"method": method.name}
        result.steps.append(info)
        if verbose:
            print(info)
    return result

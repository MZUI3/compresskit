"""Helpers shared by the pruning and quantization modules."""
import copy
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.nn as nn

ConvNd = nn.modules.conv._ConvNd
LayerTypes = Tuple[Type[nn.Module], ...]
QUANTIZABLE: LayerTypes = (nn.Linear, ConvNd)
PRUNABLE: LayerTypes = (nn.Linear, ConvNd, nn.MultiheadAttention)
Names = Union[str, Sequence[str], None]


def is_mha_out_proj(m: nn.Module) -> bool:
    """``nn.MultiheadAttention.out_proj``: its ``forward`` is never called, the attention runs functionally."""
    return type(m).__name__ == "NonDynamicallyQuantizableLinear"


def layer_names(model: nn.Module, types: LayerTypes = QUANTIZABLE) -> List[str]:
    """Names of the modules of the given types, in registration order."""
    return [n for n, m in model.named_modules() if isinstance(m, types)]


def resolve_names(model: nn.Module, items: Names, types: LayerTypes = QUANTIZABLE) -> Set[str]:
    """Expand ``items`` into module names.

    ``"first"`` is the first layer of the given types, ``"last"`` the last ``nn.Linear`` (the classifier), and
    anything else a module name or a prefix: ``"backbone.layer1"`` matches ``backbone.layer1`` and everything
    below it. An entry that matches nothing raises ``KeyError`` (it is almost always a typo).
    """
    if items is None:
        return set()
    if isinstance(items, str):
        items = (items,)
    names = layer_names(model, types)
    out: Set[str] = set()
    for item in items:
        if item == "first":
            out.update(names[:1])
        elif item == "last":
            linears = layer_names(model, (nn.Linear,))
            out.update(linears[-1:] or names[-1:])
        else:
            matched = [n for n in names if n == item or n.startswith(item + ".")]
            if not matched:
                raise KeyError(f"'{item}' matches no layer of type {tuple(t.__name__ for t in types)}")
            out.update(matched)
    return out


def select_layers(model: nn.Module, types: LayerTypes = QUANTIZABLE, exclude: Names = (),
                  include: Names = None) -> List[Tuple[str, nn.Module]]:
    """``(name, module)`` of the layers to compress: of ``types``, under ``include`` (everything when None)
    and not under ``exclude``."""
    skip = resolve_names(model, exclude, types)
    keep = resolve_names(model, include, types) if include is not None else None
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, types) or name in skip:
            continue
        if keep is not None and name not in keep:
            continue
        out.append((name, m))
    return out


def weight_of(m: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(m, nn.MultiheadAttention):
        return m.in_proj_weight
    w = getattr(m, "weight", None)
    return w if isinstance(w, torch.Tensor) else None


def weight_tensors(model: nn.Module, exclude: Names = (), include: Names = None,
                   types: LayerTypes = PRUNABLE) -> List[Tuple[str, nn.Module, torch.Tensor]]:
    """``(parameter name, module, weight)`` of every prunable weight. For ``nn.MultiheadAttention`` that is
    ``in_proj_weight``; its ``out_proj`` is an ``nn.Linear`` and is listed on its own."""
    out = []
    for name, m in select_layers(model, types, exclude, include):
        w = weight_of(m)
        if w is None:
            continue
        attr = "in_proj_weight" if isinstance(m, nn.MultiheadAttention) else "weight"
        out.append((f"{name}.{attr}" if name else attr, m, w))
    return out


def get_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parent_name, _, attr = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attr


def replace_module(model: nn.Module, name: str, new: nn.Module) -> None:
    if name == "":
        raise ValueError("cannot replace the root module itself; wrap the layer in an nn.Sequential")
    parent, attr = get_parent(model, name)
    setattr(parent, attr, new)


def model_device(model: nn.Module) -> torch.device:
    for t in model.parameters():
        return t.device
    for t in model.buffers():
        return t.device
    return torch.device("cpu")


def batch_input(batch: Any) -> Any:
    """The model input of a batch: a tensor, or the first element of a tuple/list."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def take(batches: Iterable, n: Optional[int] = None) -> list:
    """Materialise up to ``n`` batches (all of them when ``n`` is None)."""
    out = []
    for i, b in enumerate(batches):
        if n is not None and i >= n:
            break
        out.append(b)
    return out


def copy_model(model: nn.Module, inplace: bool) -> nn.Module:
    return model if inplace else copy.deepcopy(model)


def default_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    """Cross-entropy on an ``(inputs, targets)`` batch: the default loss of the gradient-based methods."""
    x, y = batch[0], batch[1]
    return nn.functional.cross_entropy(model(x), y)


def format_table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """A markdown table."""
    cols = [str(h) for h in header]
    srows = [[str(c) for c in r] for r in rows]
    widths = [max(len(x) for x in col) for col in zip(cols, *srows)] if srows else [len(c) for c in cols]

    def line(r):
        return "| " + " | ".join(x.ljust(w) for x, w in zip(r, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(cols), sep] + [line(r) for r in srows])

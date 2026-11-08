"""A name -> function registry.

Methods can be picked by string (``compress(model, [("magnitude", {...})])``) and users can add their own:

    from compresskit import register, MaskSet

    @register("my_prune", kind="prune")
    def prune_mine(model, sparsity, *, masks=None, **kw):
        masks = masks or MaskSet(model)
        return masks.prune_by_scores(masks.scores(lambda n, w: w.abs()), sparsity)
"""
import difflib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

KINDS = ("prune", "quant", "other")


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    fn: Callable
    summary: str = ""

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


_REGISTRY: Dict[str, Method] = {}


def _key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _first_line(doc: Optional[str]) -> str:
    for line in (doc or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def register(name: str, kind: str = "other", summary: str = ""):
    """Decorator that registers ``fn`` under ``name``.

    A ``prune`` function takes ``(model, ..., masks=None)`` and returns a ``MaskSet``; a ``quant`` function takes
    ``(model, ..., inplace=False)`` and returns the model. Anything else is ``other``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    def deco(fn):
        key = _key(name)
        if key in _REGISTRY and _REGISTRY[key].fn is not fn:
            raise KeyError(f"'{key}' is already registered")
        _REGISTRY[key] = Method(key, kind, fn, summary or _first_line(fn.__doc__))
        return fn

    return deco


def get(name: str) -> Method:
    """The registered method, with a hint on typos."""
    key = _key(name)
    if key not in _REGISTRY:
        close = difflib.get_close_matches(key, list(_REGISTRY), n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown method '{name}'.{hint} Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available(kind: Optional[str] = None) -> List[str]:
    return sorted(k for k, m in _REGISTRY.items() if kind is None or m.kind == kind)


def describe(kind: Optional[str] = None) -> str:
    """One line per registered method: name, kind, summary."""
    from ._common import format_table

    rows = [(m.name, m.kind, m.summary) for m in sorted(_REGISTRY.values(), key=lambda m: (m.kind, m.name))
            if kind is None or m.kind == kind]
    return format_table(["method", "kind", "summary"], rows)

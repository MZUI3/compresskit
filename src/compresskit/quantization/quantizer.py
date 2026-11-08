"""Fake quantizers: uniform affine quantization with the straight-through estimator.

``Quantizer`` computes ``x_q = (clamp(round(x / s) + z, qmin, qmax) - z) * s``. Signed symmetric quantization
uses the restricted range [-(2^(b-1) - 1), 2^(b-1) - 1]; asymmetric quantization uses [0, 2^b - 1] with a
zero point. ``round`` passes gradients straight through (Bengio et al., 2013).
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .observers import Observer, range_to_qparams

MODES = ("off", "observe", "init", "quant")


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """round() in the forward pass, identity in the backward pass."""
    return (x.round() - x).detach() + x


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """``x`` in the forward pass, gradient multiplied by ``scale`` in the backward pass."""
    return (x - x * scale).detach() + x * scale


def qrange(bits: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


class Quantizer(nn.Module):
    """A uniform affine fake quantizer with an observer for calibration.

    ``mode`` is one of ``"off"`` (pass x through), ``"observe"`` (feed x to the observer, pass it through),
    ``"init"`` (calibrate from the first tensor seen, then quantize) and ``"quant"`` (fake-quantize with the
    current ``scale`` / ``zero_point``). With ``ema_training`` the range follows an exponential moving average
    of the batch min/max while the module is in training mode (QAT of Jacob et al., 2018). With ``export`` set,
    8-bit quantizers emit ``torch.fake_quantize_per_*_affine``, which ONNX exporters turn into
    QuantizeLinear/DequantizeLinear.

    Args:
        bits: bit width (>= 2).
        symmetric: signed symmetric grid around zero, or asymmetric grid with a zero point.
        per_channel: one scale per index of ``ch_axis`` (weights) or one per tensor (activations).
        observer: ``"minmax"``, ``"ema"``, ``"percentile"``, ``"mse"`` or ``"kl"``.
        observer_kwargs: ``percentile=99.99``, ``grid=80`` for the matching observers.
    """

    exportable = True

    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = False, ch_axis: int = 0,
                 observer: str = "minmax", ema_training: bool = False, momentum: float = 0.9, **observer_kwargs):
        super().__init__()
        if bits < 2:
            raise ValueError("bits must be at least 2")
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.per_channel = bool(per_channel)
        self.ch_axis = ch_axis
        self.observer_kind = observer
        self.observer_kwargs = observer_kwargs
        self.ema_training = ema_training
        self.momentum = momentum
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        self.mode = "off"
        self.observer: Optional[Observer] = None
        self.export = False
        self._ema: Optional[Observer] = None

    @property
    def qmin(self) -> int:
        return qrange(self.bits, self.symmetric)[0]

    @property
    def qmax(self) -> int:
        return qrange(self.bits, self.symmetric)[1]

    def extra_repr(self) -> str:
        return (f"bits={self.bits}, {'symmetric' if self.symmetric else 'asymmetric'}, "
                f"{'per_channel' if self.per_channel else 'per_tensor'}, observer={self.observer_kind}, "
                f"mode={self.mode}")

    # ------------------------------------------------------------------ calibration
    def reset_observer(self) -> None:
        self.observer = Observer(self.observer_kind, self.ch_axis if self.per_channel else None,
                                 **self.observer_kwargs)

    def observe(self) -> None:
        """Start collecting statistics; ``finish()`` turns them into a range."""
        self.reset_observer()
        self.mode = "observe"

    def lazy_init(self) -> None:
        """Calibrate from the first tensor that passes through, then quantize."""
        self.mode = "init"

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        """Set the range from one tensor (e.g. a weight) and switch to ``"quant"``."""
        self.reset_observer()
        self.observer.update(x)
        self.finish()

    @torch.no_grad()
    def finish(self) -> None:
        if self.observer is None or self.observer.lo is None:
            raise RuntimeError("the quantizer saw no data; run some batches through the model first")
        lo, hi = self.observer.compute(self.qmin, self.qmax, self.symmetric)
        self.set_range(lo, hi)
        self.mode = "quant"

    @torch.no_grad()
    def set_range(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        scale, zp = range_to_qparams(lo.float(), hi.float(), self.qmin, self.qmax, self.symmetric)
        self.scale = scale.to(self.scale.device)
        self.zero_point = zp.to(self.scale.device)

    def range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The representable range ``(lo, hi)``, one entry per channel (or one for the tensor)."""
        lo = (self.qmin - self.zero_point) * self.scale
        hi = (self.qmax - self.zero_point) * self.scale
        return lo.flatten(), hi.flatten()

    # ------------------------------------------------------------------ quantization
    def _shape(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 1:
            return t
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        return t.view(shape)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        s = self._shape(x, self.scale).abs().clamp_min(1e-8).to(x.dtype)
        z = self._shape(x, self.zero_point).to(x.dtype)
        q = torch.clamp(round_ste(x / s) + z, self.qmin, self.qmax)
        return (q - z) * s

    def export_quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.exportable or self.bits != 8:
            raise RuntimeError(f"{type(self).__name__} with {self.bits} bits has no int8 Q/DQ equivalent")
        qmin, qmax = (-128, 127) if self.symmetric else (0, 255)
        if self.scale.numel() > 1:
            return torch.fake_quantize_per_channel_affine(x, self.scale.float(), self.zero_point.int(),
                                                          self.ch_axis, qmin, qmax)
        return torch.fake_quantize_per_tensor_affine(x, float(self.scale), int(self.zero_point), qmin, qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.mode == "observe":
            self.observer.update(x)
            return x
        if self.mode == "init":
            self.init_from(x)
        if self.ema_training and self.training:
            self._ema_update(x)
        if self.export:
            return self.export_quantize(x)
        return self.quantize(x)

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor) -> None:
        if self._ema is None:
            self._ema = Observer("ema", self.ch_axis if self.per_channel else None, momentum=self.momentum)
            self._ema.lo, self._ema.hi = self.range()
        self._ema.update(x)
        self.set_range(self._ema.lo, self._ema.hi)

    # ------------------------------------------------------------------ persistence
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ("scale", "zero_point"):
            full = prefix + key
            if full in state_dict:
                cur = getattr(self, key)
                if tuple(cur.shape) != tuple(state_dict[full].shape):
                    setattr(self, key, torch.empty(state_dict[full].shape, dtype=cur.dtype, device=cur.device))
                self.mode = "quant"
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class NoQuant(nn.Module):
    """A float pass-through with the quantizer interface (used for ``bits >= 32``)."""

    bits = 32
    mode = "off"
    exportable = True
    export = False

    def observe(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def lazy_init(self) -> None:
        pass

    def init_from(self, x: torch.Tensor) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DynamicWeightQuantizer(Quantizer):
    """A weight quantizer whose symmetric min/max range follows the weight: recomputed on every forward pass
    in training mode, frozen in eval mode. The usual weight quantizer for QAT with the straight-through
    estimator."""

    def __init__(self, bits: int = 8, per_channel: bool = True, ch_axis: int = 0):
        super().__init__(bits, symmetric=True, per_channel=per_channel, ch_axis=ch_axis, observer="minmax")
        self.mode = "init"

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.training and self.mode == "quant":
            self.init_from(w)
        return super().forward(w)


class LSQQuantizer(Quantizer):
    """Learned Step Size Quantization (Esser et al., 2020).

    The step size ``s`` is a parameter trained with the network, with its gradient scaled by 1/sqrt(N * Qp) and
    initialised to 2 * mean|x| / sqrt(Qp). With ``learn_offset`` an offset ``beta`` is learned as well (LSQ+,
    Bhalgat et al., 2020) and both start from the MSE-optimal range. With ``auto_signed`` the quantizer picks a
    signed or unsigned grid from the calibration data (ReLU outputs become unsigned). Build the optimizer after
    the quantizer is initialised (calibration or one forward pass) so ``s`` exists; ``quantizer_parameters``
    collects these parameters for their own param group."""

    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = False, ch_axis: int = 0,
                 learn_offset: bool = False, auto_signed: bool = False):
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, ch_axis=ch_axis,
                         observer="mse" if learn_offset else "minmax")
        self.learn_offset = learn_offset
        self.auto_signed = auto_signed
        self.s: Optional[nn.Parameter] = None
        self.beta: Optional[nn.Parameter] = None
        self._abs_sum = 0.0
        self._n_seen = 0

    def observe(self) -> None:
        super().observe()
        self._abs_sum, self._n_seen = 0.0, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "observe":
            self._abs_sum = self._abs_sum + float(x.detach().abs().mean())
            self._n_seen += 1
        return super().forward(x)

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        self.reset_observer()
        self.observer.update(x)
        if self.per_channel:
            rows = x.detach().transpose(0, self.ch_axis).reshape(x.shape[self.ch_axis], -1)
            self._abs_sum, self._n_seen = rows.abs().mean(1), 1
        else:
            self._abs_sum, self._n_seen = x.detach().abs().mean(), 1
        self.finish()

    @torch.no_grad()
    def finish(self) -> None:
        if self.observer is None or self.observer.lo is None:
            raise RuntimeError("the quantizer saw no data; run some batches through the model first")
        if self.auto_signed:
            self.symmetric = bool(self.observer.lo.min() < 0)
        lo, hi = self.observer.compute(self.qmin, self.qmax, self.symmetric)
        self.set_range(lo, hi)
        if self.learn_offset:
            s = self.scale.clone()
            beta = (-self.zero_point * self.scale).clone()  # x = s * q + beta
        else:
            mean_abs = torch.as_tensor(self._abs_sum / max(self._n_seen, 1), device=self.scale.device,
                                       dtype=torch.float32).reshape(-1)
            s = (2 * mean_abs / math.sqrt(self.qmax)).clamp_min(1e-8)
            beta = torch.zeros_like(s)
        self.s = nn.Parameter(s)
        self.beta = nn.Parameter(beta) if self.learn_offset else None
        self.mode = "quant"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.s is None:
            raise RuntimeError("LSQ quantizer is not initialised: calibrate or run one forward pass first")
        n = x.numel() // x.shape[self.ch_axis] if self.per_channel else x.numel()
        g = 1.0 / math.sqrt(max(n, 1) * self.qmax)
        s = self._shape(x, grad_scale(self.s, g)).abs().clamp_min(1e-8)
        if self.learn_offset:
            b = self._shape(x, grad_scale(self.beta, g))
            return torch.clamp(round_ste((x - b) / s), self.qmin, self.qmax) * s + b
        return torch.clamp(round_ste(x / s), self.qmin, self.qmax) * s

    @torch.no_grad()
    def sync_buffers(self) -> None:
        """Copy the learned step (and offset) into ``scale`` / ``zero_point`` for export and inspection."""
        if self.s is None:
            return
        self.scale = self.s.detach().abs().clamp_min(1e-8).clone()
        if self.learn_offset:
            self.zero_point = torch.round(-self.beta.detach() / self.scale).clamp(self.qmin, self.qmax)
        else:
            self.zero_point = torch.zeros_like(self.scale)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ("s", "beta"):
            full = prefix + key
            if full in state_dict and getattr(self, key) is None:
                setattr(self, key, nn.Parameter(torch.empty_like(state_dict[full], device=self.scale.device)))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

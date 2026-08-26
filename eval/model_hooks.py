"""Residual-stream capture and intervention hooks for open-weight decoder models.

The helpers in this module deliberately operate on a decoder block's output rather than on a
model-specific attention or MLP implementation.  They support the common Hugging Face layouts used
by Llama/Qwen, GPT-2, and GPT-NeoX while keeping the scientific intervention explicit:

* steering adds a fixed, train-fold-derived direction;
* directional erasure removes only that direction's component;
* patching copies either a complete state or one directional component from a source run.

No helper accepts an evaluation label.  Item-specific label-signed steering would be an
orchestration policy, not evidence that the unperturbed model naturally uses a representation.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from typing import Any, Callable, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - exercised in the lightweight CI environment
    torch = None


HiddenTransform = Callable[[Any], Any]


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "Residual-stream interventions require the optional 'full' dependencies: "
            "install torch with `pip install -e '.[full]'`."
        )


def _nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            raise AttributeError(path)
        current = getattr(current, name)
    return current


def resolve_decoder_layers(model: Any) -> Sequence[Any]:
    """Return the ordered decoder blocks for common Hugging Face causal-LM layouts."""

    candidates = (
        "model.layers",          # Llama, Mistral, Qwen2/Qwen3
        "transformer.h",         # GPT-2/GPT-J style
        "gpt_neox.layers",       # GPT-NeoX/Pythia
        "model.decoder.layers",  # some encoder-decoder implementations
    )
    for path in candidates:
        try:
            layers = _nested_attr(model, path)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
            return layers
    raise ValueError(
        "Could not locate decoder layers. Supported layouts: "
        + ", ".join(candidates)
    )


def _hidden_from_output(output: Any):
    _require_torch()
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Decoder block output must be a hidden-state tensor or a tuple/list whose first "
        "element is that tensor."
    )


def _replace_hidden(output: Any, hidden: Any):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported decoder block output type: {type(output)!r}")


def _updated_token_states(hidden: Any, token_index: int | slice, update: HiddenTransform):
    _require_torch()
    if hidden.ndim != 3:
        raise ValueError(
            f"expected hidden states shaped (batch, sequence, width), got {tuple(hidden.shape)}"
        )
    changed = hidden.clone()
    changed[:, token_index, :] = update(hidden[:, token_index, :])
    return changed


def _as_direction(direction: Any, hidden: Any):
    _require_torch()
    value = torch.as_tensor(direction, device=hidden.device, dtype=hidden.dtype)
    if value.ndim != 1 or value.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"direction must have shape ({hidden.shape[-1]},), got {tuple(value.shape)}"
        )
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) == 0.0:
        raise ValueError("direction must be finite and non-zero")
    return value, norm


def steering_transform(
    direction: Any,
    alpha: float,
    *,
    token_index: int | slice = -1,
    dose_scale: float = 1.0,
) -> HiddenTransform:
    """Create ``h <- h + alpha * dose_scale * unit(direction)``.

    Normalizing inside the hook makes the intervention invariant to arbitrary
    rescaling of the supplied vector. ``dose_scale`` is the separately locked
    residual-stream unit (normally the train-fold residual feature RMS).
    Positive and negative alpha values estimate the signed derivative of the
    positive-vs-negative answer-logit margin. The evaluation label must not be
    used to choose the sign.
    """

    alpha = float(alpha)
    dose_scale = float(dose_scale)
    if not math.isfinite(dose_scale) or dose_scale <= 0.0:
        raise ValueError("steering dose_scale must be finite and positive")

    def transform(hidden):
        if alpha == 0.0:
            return hidden
        value, norm = _as_direction(direction, hidden)
        unit = value / norm
        return _updated_token_states(
            hidden,
            token_index,
            lambda selected: selected + alpha * dose_scale * unit,
        )

    return transform


def directional_erasure_transform(
    direction: Any,
    *,
    token_index: int | slice = -1,
    center: Any | None = None,
    strength: float = 1.0,
) -> HiddenTransform:
    """Remove a selected direction's component while preserving its orthogonal complement.

    This is a single-direction intervention, not a claim that all information about the concept has
    been erased.  Use a train-fold-derived center and direction; compare against
    covariance-matched random directions and record collateral behavior.
    """

    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("erasure strength must be in [0, 1]")

    def transform(hidden):
        if strength == 0.0:
            return hidden
        value, norm = _as_direction(direction, hidden)
        unit = value / norm
        offset = (
            torch.zeros_like(unit)
            if center is None
            else torch.as_tensor(center, device=hidden.device, dtype=hidden.dtype)
        )
        if offset.shape != unit.shape:
            raise ValueError(f"center must have shape {tuple(unit.shape)}, got {tuple(offset.shape)}")

        def erase(selected):
            centered = selected - offset
            coefficient = torch.sum(centered * unit, dim=-1, keepdim=True)
            return selected - strength * coefficient * unit

        return _updated_token_states(hidden, token_index, erase)

    return transform


def patch_transform(
    source_activation: Any,
    *,
    token_index: int | slice = -1,
    direction: Any | None = None,
    center: Any | None = None,
    strength: float = 1.0,
) -> HiddenTransform:
    """Patch a source activation into a target run.

    With ``direction=None`` this interpolates the complete token state.  With a direction it copies
    only the source's component along that train-fold-derived direction, leaving the target's
    orthogonal component unchanged.  The latter is the preferred content-subspace intervention
    when paired biological counterfactuals are available.
    """

    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("patch strength must be in [0, 1]")

    def transform(hidden):
        if strength == 0.0:
            return hidden
        source = torch.as_tensor(
            source_activation,
            device=hidden.device,
            dtype=hidden.dtype,
        )

        def patch(selected):
            candidate = source
            if (
                candidate.ndim == selected.ndim - 1
                and candidate.ndim >= 2
                and candidate.shape[0] == selected.shape[0]
            ):
                # A per-batch source shaped (batch, width) patches a token slice
                # shaped (batch, selected_sequence, width).
                candidate = candidate.unsqueeze(1)
            while candidate.ndim < selected.ndim:
                candidate = candidate.unsqueeze(0)
            if candidate.shape[-1] != selected.shape[-1]:
                raise ValueError(
                    "source activation width does not match target hidden-state width"
                )
            candidate = torch.broadcast_to(candidate, selected.shape)
            if direction is None:
                if strength == 1.0:
                    # Complete replacement is an exact copy operation.  The
                    # equivalent interpolation can incur float32 roundoff.
                    return candidate
                return selected + strength * (candidate - selected)

            value, norm = _as_direction(direction, hidden)
            unit = value / norm
            offset = (
                torch.zeros_like(unit)
                if center is None
                else torch.as_tensor(center, device=hidden.device, dtype=hidden.dtype)
            )
            if offset.shape != unit.shape:
                raise ValueError(
                    f"center must have shape {tuple(unit.shape)}, got {tuple(offset.shape)}"
                )
            target_coefficient = torch.sum(
                (selected - offset) * unit,
                dim=-1,
                keepdim=True,
            )
            source_coefficient = torch.sum(
                (candidate - offset) * unit,
                dim=-1,
                keepdim=True,
            )
            return selected + strength * (source_coefficient - target_coefficient) * unit

        return _updated_token_states(hidden, token_index, patch)

    return transform


def compose_transforms(*transforms: HiddenTransform) -> HiddenTransform:
    """Compose interventions in the declared order."""

    def transform(hidden):
        value = hidden
        for operation in transforms:
            value = operation(value)
        return value

    return transform


class ResidualStreamIntervention(AbstractContextManager):
    """Install one temporary decoder-block output intervention and always remove it."""

    def __init__(self, model: Any, layer_index: int, transform: HiddenTransform):
        self.model = model
        self.layer_index = int(layer_index)
        self.transform = transform
        self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _hook(self, _module, _inputs, output):
        hidden = _hidden_from_output(output)
        return _replace_hidden(output, self.transform(hidden))

    def __enter__(self):
        if self.active:
            raise RuntimeError("intervention is already active")
        layers = resolve_decoder_layers(self.model)
        if not -len(layers) <= self.layer_index < len(layers):
            raise IndexError(
                f"layer index {self.layer_index} is outside decoder with {len(layers)} layers"
            )
        self._handle = layers[self.layer_index].register_forward_hook(self._hook)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class ResidualStreamCapture(AbstractContextManager):
    """Capture selected residual states without modifying the forward pass."""

    def __init__(
        self,
        model: Any,
        layer_index: int,
        *,
        token_index: int | slice = -1,
        detach: bool = True,
    ):
        self.model = model
        self.layer_index = int(layer_index)
        self.token_index = token_index
        self.detach = bool(detach)
        self.values: list[Any] = []
        self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _hook(self, _module, _inputs, output):
        hidden = _hidden_from_output(output)
        value = hidden[:, self.token_index, :]
        if self.detach:
            value = value.detach().clone()
        self.values.append(value)
        return None

    def __enter__(self):
        if self.active:
            raise RuntimeError("capture is already active")
        layers = resolve_decoder_layers(self.model)
        if not -len(layers) <= self.layer_index < len(layers):
            raise IndexError(
                f"layer index {self.layer_index} is outside decoder with {len(layers)} layers"
            )
        self._handle = layers[self.layer_index].register_forward_hook(self._hook)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

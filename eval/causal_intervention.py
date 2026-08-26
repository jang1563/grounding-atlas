"""Confirmatory residual-stream interventions for the causal GroundBench track.

This module measures signed answer-logit effects.  It does not turn probe accuracy into a causal
claim.  A fixed train-fold-derived direction can establish causal availability when steering or
patching changes the output beyond controls.  Evidence of natural use additionally requires
targeted erasure, rescue, specificity, and a naturally successful unperturbed condition.

The intervention functions never accept evaluation labels.  Labels enter only when records are
assembled after model execution for paired outcome analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from .model_hooks import (
        ResidualStreamCapture,
        ResidualStreamIntervention,
        compose_transforms,
        directional_erasure_transform,
        patch_transform,
        steering_transform,
    )
except ImportError:  # direct execution from eval/
    from model_hooks import (
        ResidualStreamCapture,
        ResidualStreamIntervention,
        compose_transforms,
        directional_erasure_transform,
        patch_transform,
        steering_transform,
    )

try:
    import torch
except ImportError:  # pragma: no cover - lightweight CI need not install torch
    torch = None


CAUSAL_ARTIFACT_SCHEMA = 4
FORBIDDEN_SIGN_SOURCES = {
    "evaluation_label",
    "held_out_label",
    "item_target_label",
    "target_label",
    "test_label",
}
ALLOWED_SIGN_SOURCES = {
    "predeclared_positive_class",
    "train_fold_labels",
}
EXECUTION_RECEIPT_SCHEMA = 3
EXECUTION_TRACE_SCHEMA = 2
EXECUTION_MEASUREMENT_SCHEMA = 1
EXECUTION_RECEIPT_FIELDS = (
    "item_id",
    "split_group_id",
    "intervention_pair_id",
    "condition",
    "alpha",
    "direction_kind",
    "direction_id",
    "applied_direction_sha256",
    "answer_logit_margin",
    "executed_layer_index",
    "executed_token_position",
    "positive_answer_token_ids",
    "negative_answer_token_ids",
    "execution_trace_schema_version",
    "execution_trace_kind",
    "execution_trace_item_id",
    "execution_input_ids",
    "execution_attention_mask",
    "execution_input_sha256",
    "execution_context_sha256",
    "steering_dose_scale",
    "erasure_strength",
    "erasure_center_sha256",
    "patch_strength",
    "source_condition",
    "source_activation_sha256",
    "recipient_activation_sha256",
    "source_item_id",
    "source_intervention_pair_id",
    "collateral_kl",
    "unrelated_margin_change",
    "direction_norm_relative_error",
    "projected_variance_relative_error",
    "content_token_count_absolute_difference",
    "content_embedding_cosine_distance",
    "source_content_sha256",
    "recipient_content_sha256",
    "test_label_used_for_intervention",
)
LABEL_FREE_EXECUTION_FIELDS = frozenset(EXECUTION_RECEIPT_FIELDS)
RECEIPTED_EXECUTION_FIELDS = frozenset(
    {
        *EXECUTION_RECEIPT_FIELDS,
        "execution_receipt",
        "execution_receipt_sha256",
    }
)
INTERVENTION_FIELD_GROUPS = {
    "direction": {"applied_direction_sha256"},
    "steering": {"alpha", "steering_dose_scale"},
    "matching": {
        "direction_norm_relative_error",
        "projected_variance_relative_error",
    },
    "erasure": {"erasure_strength", "erasure_center_sha256"},
    "patch": {
        "patch_strength",
        "source_condition",
        "source_activation_sha256",
        "recipient_activation_sha256",
        "source_item_id",
        "source_intervention_pair_id",
    },
    "content": {
        "content_token_count_absolute_difference",
        "content_embedding_cosine_distance",
        "source_content_sha256",
        "recipient_content_sha256",
    },
}
SCORED_RECORD_FIELDS = {
    *EXECUTION_RECEIPT_FIELDS,
    "target_label",
    "correct_answer_margin",
    "execution_receipt",
    "execution_receipt_sha256",
}


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _id_token(value: Any) -> str:
    if value is None or isinstance(value, (list, dict)):
        raise TypeError("IDs must be non-null JSON scalars")
    return (
        f"{type(value).__name__}:"
        + json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def validate_binary_label(
    value: Any,
    name: str = "target label",
) -> int:
    """Validate one canonical binary integer without lossy coercion."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) not in {0, 1}
    ):
        raise ValueError(f"{name} must be an integer in {{0, 1}}")
    return int(value)


def validate_condition_record_schema(
    family: str,
    row: Mapping[str, Any],
) -> None:
    """Enforce direction kind and exact null/non-null fields for one causal cell."""

    condition = str(row.get("condition", ""))
    direction_kind = str(row.get("direction_kind", ""))
    required_groups: set[str]
    expected_direction_kind: str | None = None
    if family == "steering":
        if condition == "baseline_unhooked":
            expected_direction_kind = "baseline"
            required_groups = set()
        else:
            expected_direction_kind = next(
                (
                    prefix
                    for prefix in (
                        "target",
                        "random",
                        "shuffled",
                        "surface",
                    )
                    if condition.startswith(f"{prefix}-")
                ),
                None,
            )
            if expected_direction_kind is None:
                raise ValueError(
                    f"condition {condition!r} is not allowed for steering"
                )
            required_groups = {"direction", "steering"}
            if expected_direction_kind == "random":
                required_groups.add("matching")
            direction_id = str(row.get("direction_id", ""))
            alpha_prefix = (
                f"{direction_id}-"
                if expected_direction_kind == "random"
                else f"{expected_direction_kind}-"
            )
            if not condition.startswith(alpha_prefix):
                raise ValueError(
                    "steering condition name differs from its direction_id"
                )
            try:
                encoded_alpha = float(condition[len(alpha_prefix) :])
                observed_alpha = float(row["alpha"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "steering condition must encode its numeric alpha"
                ) from exc
            if (
                not np.isfinite(encoded_alpha)
                or not np.isfinite(observed_alpha)
                or not np.isclose(
                    encoded_alpha,
                    observed_alpha,
                    atol=1e-12,
                    rtol=1e-12,
                )
            ):
                raise ValueError(
                    "steering condition name and row alpha differ"
                )
    elif family == "erasure_rescue":
        schema = {
            "baseline": ("baseline", set()),
            "erased_target": ("target", {"direction", "erasure"}),
            "erased_random": (
                "random",
                {"direction", "matching", "erasure"},
            ),
            "rescued_target": (
                "target",
                {"direction", "erasure", "patch"},
            ),
        }
        if condition not in schema:
            raise ValueError(
                f"condition {condition!r} is not allowed for erasure_rescue"
            )
        expected_direction_kind, required_groups = schema[condition]
    elif family == "routing_patch":
        schema = {
            "native": ("baseline", set()),
            "elicited": ("baseline", set()),
            "no_content_no_route": ("baseline", set()),
            "route_success_to_native": (
                "target",
                {"direction", "patch", "content"},
            ),
            "route_native_to_success": (
                "target",
                {"direction", "patch", "content"},
            ),
            "route_random_to_native": (
                "random",
                {"direction", "matching", "patch", "content"},
            ),
            "route_random_to_success": (
                "random",
                {"direction", "matching", "patch", "content"},
            ),
            "route_shuffled_to_native": (
                "shuffled",
                {"direction", "patch", "content"},
            ),
            "route_shuffled_to_success": (
                "shuffled",
                {"direction", "patch", "content"},
            ),
            "route_only_no_content": (
                "target",
                {"direction", "patch", "content"},
            ),
            "content_and_route": (
                "target",
                {"direction", "patch", "content"},
            ),
        }
        if condition not in schema:
            raise ValueError(
                f"condition {condition!r} is not allowed for routing_patch"
            )
        expected_direction_kind, required_groups = schema[condition]
    else:
        return

    if direction_kind != expected_direction_kind:
        raise ValueError(
            f"condition {condition!r} requires direction_kind="
            f"{expected_direction_kind!r}"
        )
    required_fields = set().union(
        *(
            INTERVENTION_FIELD_GROUPS[group]
            for group in required_groups
        ),
        set(),
    )
    conditional_fields = set().union(
        *INTERVENTION_FIELD_GROUPS.values()
    )
    missing = sorted(
        field for field in required_fields if row.get(field) is None
    )
    forbidden = sorted(
        field
        for field in conditional_fields - required_fields
        if row.get(field) is not None
    )
    if missing:
        raise ValueError(
            f"condition {condition!r} requires non-null fields: {missing}"
        )
    if forbidden:
        raise ValueError(
            f"condition {condition!r} requires null fields: {forbidden}"
        )


def validate_scored_record_schema(row: Mapping[str, Any]) -> None:
    """Require one uniform scored-row schema with no free-form label channel."""

    if set(row) != SCORED_RECORD_FIELDS:
        missing = sorted(SCORED_RECORD_FIELDS - set(row))
        extras = sorted(set(row) - SCORED_RECORD_FIELDS)
        raise ValueError(
            "causal scored record must use the exact schema; "
            f"missing={missing}, extras={extras}"
        )


def _token_ids(values: int | Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, bool):
        raise ValueError(f"{name} token IDs must be integers")
    if isinstance(values, (int, np.integer)):
        result = (int(values),)
    else:
        raw_values = tuple(values)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for value in raw_values
        ):
            raise ValueError(f"{name} token IDs must be integers")
        result = tuple(int(value) for value in raw_values)
    if not result:
        raise ValueError(f"{name} token set cannot be empty")
    if min(result) < 0:
        raise ValueError(f"{name} token IDs must be non-negative")
    if result != tuple(sorted(set(result))):
        raise ValueError(
            f"{name} token IDs must be unique and sorted canonically"
        )
    return result


def _numpy_logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(maximum, axis=axis)
        + np.log(np.exp(values - maximum).sum(axis=axis))
    )


def answer_logit_margin(
    logits: Any,
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
):
    """Return ``logsumexp(positive logits) - logsumexp(negative logits)``.

    Inputs may be shaped ``(..., vocabulary)`` or ``(batch, sequence, vocabulary)``.  For a
    three-dimensional causal-LM output, the final sequence position is used.  The result stays a
    torch tensor when the input is a tensor, otherwise it is a NumPy array/scalar.
    """

    positive = _token_ids(positive_token_ids, "positive")
    negative = _token_ids(negative_token_ids, "negative")
    if set(positive) & set(negative):
        raise ValueError("positive and negative answer token sets must be disjoint")

    if torch is not None and torch.is_tensor(logits):
        values = logits[:, -1, :] if logits.ndim == 3 else logits
        if values.ndim < 1:
            raise ValueError("logits must include a vocabulary dimension")
        pos = torch.logsumexp(values[..., list(positive)], dim=-1)
        neg = torch.logsumexp(values[..., list(negative)], dim=-1)
        return pos - neg

    values = np.asarray(logits, dtype=float)
    values = values[:, -1, :] if values.ndim == 3 else values
    if values.ndim < 1:
        raise ValueError("logits must include a vocabulary dimension")
    pos = _numpy_logsumexp(values[..., list(positive)], axis=-1)
    neg = _numpy_logsumexp(values[..., list(negative)], axis=-1)
    return pos - neg


def _as_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def activation_sha256(value: Any) -> str:
    """Hash one activation with a stable float32 little-endian canonicalization."""

    array = np.asarray(_as_numpy(value), dtype=np.dtype("<f4"), order="C")
    if not np.isfinite(array).all():
        raise ValueError("activation contains non-finite values")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "canonicalization": "float32-le-c-order-v1",
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def activation_row_sha256(value: Any) -> list[str]:
    """Hash each batch row independently for source/recipient provenance."""

    array = _as_numpy(value)
    if array.ndim == 0:
        raise ValueError("activation must include a feature dimension")
    rows = array[None, ...] if array.ndim == 1 else array
    return [activation_sha256(row) for row in rows]


def _execution_input_vector(
    values: Sequence[int],
    name: str,
    *,
    binary: bool = False,
) -> list[int]:
    raw = list(values)
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError(f"{name} must contain integers")
    result = [int(value) for value in raw]
    if binary:
        if any(value not in {0, 1} for value in result):
            raise ValueError(f"{name} must contain only 0/1 values")
    elif min(result) < 0:
        raise ValueError(f"{name} must contain non-negative token IDs")
    return result


def execution_input_sha256(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
) -> str:
    """Hash one exact tokenized model input used by a causal measurement."""

    ids = _execution_input_vector(input_ids, "execution_input_ids")
    mask = _execution_input_vector(
        attention_mask,
        "execution_attention_mask",
        binary=True,
    )
    if len(ids) != len(mask):
        raise ValueError(
            "execution input IDs and attention mask must have equal length"
        )
    if not any(mask):
        raise ValueError("execution attention mask must retain at least one token")
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": EXECUTION_TRACE_SCHEMA,
                "input_ids": ids,
                "attention_mask": mask,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_execution_traces(
    *,
    item_ids: Sequence[Any],
    layer_index: int,
    token_position: int,
    positive_answer_token_ids: int | Sequence[int],
    negative_answer_token_ids: int | Sequence[int],
    execution_input_ids: Sequence[Sequence[int]] | Sequence[int],
    execution_attention_mask: Sequence[Sequence[int]] | Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Create identity-bound traces from actual measurement-helper arguments."""

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, (int, np.integer))
        or int(layer_index) < 0
    ):
        raise ValueError("execution trace layer_index must be a non-negative integer")
    if isinstance(token_position, bool) or not isinstance(
        token_position,
        (int, np.integer),
    ):
        raise ValueError("execution trace token_position must be an integer")
    positive = list(_token_ids(positive_answer_token_ids, "positive_answer"))
    negative = list(_token_ids(negative_answer_token_ids, "negative_answer"))
    raw_input_rows = list(execution_input_ids)
    if not raw_input_rows:
        raise ValueError("execution_input_ids cannot be empty")
    if isinstance(raw_input_rows[0], (int, np.integer)) and not isinstance(
        raw_input_rows[0],
        bool,
    ):
        input_rows = [raw_input_rows]
    else:
        input_rows = [list(row) for row in raw_input_rows]
    if execution_attention_mask is None:
        mask_rows = [[1] * len(row) for row in input_rows]
    else:
        raw_mask_rows = list(execution_attention_mask)
        if raw_mask_rows and isinstance(
            raw_mask_rows[0],
            (int, np.integer),
        ) and not isinstance(raw_mask_rows[0], bool):
            mask_rows = [raw_mask_rows]
        else:
            mask_rows = [list(row) for row in raw_mask_rows]
    if len(input_rows) != len(mask_rows):
        raise ValueError(
            "execution input and attention-mask batches must have equal length"
        )
    trace_item_ids = list(item_ids)
    if len(trace_item_ids) != len(input_rows):
        raise ValueError(
            "execution item IDs and tokenized-input batches must have equal length"
        )
    item_tokens = [_id_token(item_id) for item_id in trace_item_ids]
    if len(item_tokens) != len(set(item_tokens)):
        raise ValueError("execution trace item IDs must be unique")
    traces = []
    for item_id, input_row, mask_row in zip(
        trace_item_ids,
        input_rows,
        mask_rows,
        strict=True,
    ):
        ids = _execution_input_vector(input_row, "execution_input_ids")
        mask = _execution_input_vector(
            mask_row,
            "execution_attention_mask",
            binary=True,
        )
        traces.append(
            {
                "schema_version": EXECUTION_TRACE_SCHEMA,
                "trace_kind": "measurement_helper_arguments_v1",
                "item_id": item_id,
                "executed_layer_index": int(layer_index),
                "executed_token_position": int(token_position),
                "positive_answer_token_ids": positive,
                "negative_answer_token_ids": negative,
                "execution_input_ids": ids,
                "execution_attention_mask": mask,
                "execution_input_sha256": execution_input_sha256(ids, mask),
            }
        )
    return traces


def validate_execution_trace(
    trace: Mapping[str, Any],
    contract: InterventionContract | None = None,
) -> dict[str, Any]:
    """Validate a measurement-emitted trace and optionally bind it to a contract."""

    fields = {
        "schema_version",
        "trace_kind",
        "item_id",
        "executed_layer_index",
        "executed_token_position",
        "positive_answer_token_ids",
        "negative_answer_token_ids",
        "execution_input_ids",
        "execution_attention_mask",
        "execution_input_sha256",
    }
    if not isinstance(trace, Mapping) or set(trace) != fields:
        raise ValueError("execution trace must use the exact measurement schema")
    if trace.get("schema_version") != EXECUTION_TRACE_SCHEMA or trace.get(
        "trace_kind"
    ) != "measurement_helper_arguments_v1":
        raise ValueError("execution trace schema or kind is invalid")
    rebuilt = build_execution_traces(
        item_ids=[trace["item_id"]],
        layer_index=trace["executed_layer_index"],
        token_position=trace["executed_token_position"],
        positive_answer_token_ids=trace["positive_answer_token_ids"],
        negative_answer_token_ids=trace["negative_answer_token_ids"],
        execution_input_ids=trace["execution_input_ids"],
        execution_attention_mask=trace["execution_attention_mask"],
    )[0]
    if dict(trace) != rebuilt:
        raise ValueError("execution trace differs from its recomputed input trace")
    if contract is not None:
        if rebuilt["executed_layer_index"] != int(contract.layer_index):
            raise ValueError("executed layer differs from the intervention contract")
        if rebuilt["executed_token_position"] != int(contract.token_position):
            raise ValueError(
                "executed token position differs from the intervention contract"
            )
        if rebuilt["positive_answer_token_ids"] != list(
            _token_ids(contract.positive_answer_token_ids, "positive_answer")
        ) or rebuilt["negative_answer_token_ids"] != list(
            _token_ids(contract.negative_answer_token_ids, "negative_answer")
        ):
            raise ValueError("executed answer-token sets differ from the contract")
    return rebuilt


def build_execution_measurements(
    execution_traces: Sequence[Mapping[str, Any]],
    answer_logit_margins: Sequence[float],
) -> list[dict[str, Any]]:
    """Bind each measured margin to the trace identity that produced its row."""

    traces = [validate_execution_trace(trace) for trace in execution_traces]
    margins = np.asarray(answer_logit_margins, dtype=float).reshape(-1)
    if len(traces) != len(margins):
        raise ValueError(
            "execution traces and measured margins must have equal length"
        )
    if not np.isfinite(margins).all():
        raise ValueError("execution measurements contain non-finite margins")
    return [
        {
            "schema_version": EXECUTION_MEASUREMENT_SCHEMA,
            "item_id": trace["item_id"],
            "answer_logit_margin": float(margin),
            "execution_trace": trace,
        }
        for trace, margin in zip(traces, margins, strict=True)
    ]


def validate_execution_measurement(
    measurement: Mapping[str, Any],
    contract: InterventionContract | None = None,
) -> dict[str, Any]:
    """Validate one exact identity-bound margin/trace measurement record."""

    fields = {
        "schema_version",
        "item_id",
        "answer_logit_margin",
        "execution_trace",
    }
    if not isinstance(measurement, Mapping) or set(measurement) != fields:
        raise ValueError(
            "execution measurement must use the exact identity-bound schema"
        )
    if measurement.get("schema_version") != EXECUTION_MEASUREMENT_SCHEMA:
        raise ValueError("execution measurement schema is invalid")
    trace = validate_execution_trace(
        measurement["execution_trace"],
        contract,
    )
    if _id_token(measurement["item_id"]) != _id_token(trace["item_id"]):
        raise ValueError(
            "execution measurement item differs from its execution trace"
        )
    try:
        margin = float(measurement["answer_logit_margin"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "execution measurement margin must be finite"
        ) from exc
    if not np.isfinite(margin):
        raise ValueError("execution measurement margin must be finite")
    return {
        "schema_version": EXECUTION_MEASUREMENT_SCHEMA,
        "item_id": measurement["item_id"],
        "answer_logit_margin": margin,
        "execution_trace": trace,
    }


def direction_sha256(value: Any) -> str:
    """Hash the effective unit direction while preserving its global sign."""

    array = _as_numpy(value).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and non-zero")
    return activation_sha256(array / norm)


def execution_receipt_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact label-free execution fields bound to one output row."""

    payload = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        **{field: row.get(field) for field in EXECUTION_RECEIPT_FIELDS},
    }
    if payload["test_label_used_for_intervention"] is not False:
        raise ValueError("execution receipts require a test-label-free intervention")
    margin = float(payload["answer_logit_margin"])
    if not np.isfinite(margin):
        raise ValueError("execution receipt answer margin must be finite")
    payload["answer_logit_margin"] = margin
    if not _valid_sha256(payload["execution_context_sha256"]):
        raise ValueError("execution receipt context checksum is invalid")
    trace = validate_execution_trace(
        {
            "schema_version": payload["execution_trace_schema_version"],
            "trace_kind": payload["execution_trace_kind"],
            "item_id": payload["execution_trace_item_id"],
            "executed_layer_index": payload["executed_layer_index"],
            "executed_token_position": payload["executed_token_position"],
            "positive_answer_token_ids": payload[
                "positive_answer_token_ids"
            ],
            "negative_answer_token_ids": payload[
                "negative_answer_token_ids"
            ],
            "execution_input_ids": payload["execution_input_ids"],
            "execution_attention_mask": payload[
                "execution_attention_mask"
            ],
            "execution_input_sha256": payload["execution_input_sha256"],
        }
    )
    if _id_token(trace["item_id"]) != _id_token(payload["item_id"]):
        raise ValueError(
            "execution trace item identity differs from the receipt item"
        )
    payload["positive_answer_token_ids"] = trace[
        "positive_answer_token_ids"
    ]
    payload["negative_answer_token_ids"] = trace[
        "negative_answer_token_ids"
    ]
    payload["execution_input_ids"] = trace["execution_input_ids"]
    payload["execution_attention_mask"] = trace[
        "execution_attention_mask"
    ]
    for field in (
        "collateral_kl",
        "unrelated_margin_change",
        "direction_norm_relative_error",
        "projected_variance_relative_error",
        "content_token_count_absolute_difference",
        "content_embedding_cosine_distance",
    ):
        value = payload[field]
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"execution receipt {field} must be finite")
    return payload


def execution_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash one canonical label-free execution receipt."""

    expected = {"schema_version", *EXECUTION_RECEIPT_FIELDS}
    if set(receipt) != expected or receipt.get("schema_version") != (
        EXECUTION_RECEIPT_SCHEMA
    ):
        raise ValueError("execution receipt schema is invalid")
    return hashlib.sha256(
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def bind_execution_receipts(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach immutable label-free execution receipts before outcome scoring."""

    bound = []
    for raw_row in records:
        row = dict(raw_row)
        if set(row) != LABEL_FREE_EXECUTION_FIELDS:
            missing = sorted(LABEL_FREE_EXECUTION_FIELDS - set(row))
            extras = sorted(set(row) - LABEL_FREE_EXECUTION_FIELDS)
            raise ValueError(
                "receipt binding requires the exact label-free execution schema; "
                f"missing={missing}, extras={extras}"
            )
        receipt = execution_receipt_payload(row)
        row["execution_receipt"] = receipt
        row["execution_receipt_sha256"] = execution_receipt_sha256(receipt)
        bound.append(row)
    return bound


def _run_margin(
    forward_logits: Callable[..., Any],
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
    *,
    model_inputs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    logits = (
        forward_logits()
        if model_inputs is None
        else forward_logits(model_inputs)
    )
    return _as_numpy(
        answer_logit_margin(
            logits,
            positive_token_ids,
            negative_token_ids,
        )
    ).reshape(-1)


@dataclass(frozen=True)
class InterventionContract:
    """Locked intervention metadata written into every causal artifact."""

    task_id: str
    model_revision: str
    fold_id: str
    layer_index: int
    token_position: int
    direction_artifact_sha256: str
    decoder_evaluation_artifact_sha256: str
    applied_direction_sha256: str
    direction_center_sha256: str
    erasure_strength: float
    prompt_protocol_sha256: str
    decoding_config_sha256: str
    execution_code_sha256: str
    positive_answer_token_ids: tuple[int, ...]
    negative_answer_token_ids: tuple[int, ...]
    hook_site_kind: str
    steering_dose_scale: float = 1.0
    sign_source: str = "train_fold_labels"
    label_signed_per_item: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id.strip()
            or not isinstance(self.model_revision, str)
            or not self.model_revision.strip()
        ):
            raise ValueError(
                "intervention contract task_id/model_revision must be non-empty strings"
            )
        try:
            _id_token(self.fold_id)
        except TypeError as exc:
            raise ValueError(
                "intervention contract fold_id must be a non-null JSON scalar"
            ) from exc
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, (int, np.integer))
            or int(self.layer_index) < 0
        ):
            raise ValueError(
                "intervention contract layer_index must be a non-negative integer"
            )
        if isinstance(self.token_position, bool) or not isinstance(
            self.token_position,
            (int, np.integer),
        ):
            raise ValueError(
                "intervention contract token_position must be an integer"
            )
        missing = [
            name
            for name in (
                "task_id",
                "model_revision",
                "fold_id",
                "direction_artifact_sha256",
                "decoder_evaluation_artifact_sha256",
                "applied_direction_sha256",
                "direction_center_sha256",
                "prompt_protocol_sha256",
                "decoding_config_sha256",
                "execution_code_sha256",
            )
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(f"intervention contract has empty fields: {missing}")
        if self.sign_source in FORBIDDEN_SIGN_SOURCES:
            raise ValueError(
                f"evaluation-label-derived sign source {self.sign_source!r} is forbidden"
            )
        if self.sign_source not in ALLOWED_SIGN_SOURCES:
            raise ValueError(
                f"unknown sign source {self.sign_source!r}; allowed: "
                f"{sorted(ALLOWED_SIGN_SOURCES)}"
            )
        if self.label_signed_per_item:
            raise ValueError(
                "label-signed per-item intervention is not a causal-use test; use one fixed "
                "positive-class direction and analyze the signed answer-logit derivative"
            )
        for field in (
            "direction_artifact_sha256",
            "decoder_evaluation_artifact_sha256",
            "applied_direction_sha256",
            "direction_center_sha256",
            "prompt_protocol_sha256",
            "decoding_config_sha256",
            "execution_code_sha256",
        ):
            raw_digest = str(getattr(self, field))
            digest = raw_digest.lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ) or raw_digest != digest:
                raise ValueError(
                    f"{field} must be a lowercase 64-character SHA-256 digest"
                )
        if (
            isinstance(self.erasure_strength, bool)
            or not np.isfinite(float(self.erasure_strength))
            or not 0.0 < float(self.erasure_strength) <= 1.0
        ):
            raise ValueError("erasure_strength must be in (0, 1]")
        if (
            isinstance(self.steering_dose_scale, bool)
            or not np.isfinite(float(self.steering_dose_scale))
            or float(self.steering_dose_scale) <= 0.0
        ):
            raise ValueError("steering_dose_scale must be finite and positive")
        positive_tokens = _token_ids(
            self.positive_answer_token_ids,
            "positive_answer",
        )
        negative_tokens = _token_ids(
            self.negative_answer_token_ids,
            "negative_answer",
        )
        if set(positive_tokens) & set(negative_tokens):
            raise ValueError("contract answer token sets must be disjoint")
        if self.hook_site_kind != "decoder_block_output":
            raise ValueError(
                "confirmatory interventions currently require decoder_block_output hooks"
            )


def execution_context_sha256(contract: InterventionContract) -> str:
    """Hash every outcome-relevant, label-free execution setting."""

    contract.validate()
    payload = {
        "task_id": contract.task_id,
        "model_revision": contract.model_revision,
        "fold_id": contract.fold_id,
        "layer_index": int(contract.layer_index),
        "token_position": int(contract.token_position),
        "hook_site_kind": contract.hook_site_kind,
        "prompt_protocol_sha256": contract.prompt_protocol_sha256,
        "decoding_config_sha256": contract.decoding_config_sha256,
        "execution_code_sha256": contract.execution_code_sha256,
        "positive_answer_token_ids": list(
            _token_ids(
                contract.positive_answer_token_ids,
                "positive_answer",
            )
        ),
        "negative_answer_token_ids": list(
            _token_ids(
                contract.negative_answer_token_ids,
                "negative_answer",
            )
        ),
        "direction_artifact_sha256": contract.direction_artifact_sha256,
        "applied_direction_sha256": contract.applied_direction_sha256,
        "direction_center_sha256": contract.direction_center_sha256,
        "steering_dose_scale": float(contract.steering_dose_scale),
        "erasure_strength": float(contract.erasure_strength),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_execution_row_context(
    row: Mapping[str, Any],
    contract: InterventionContract,
) -> None:
    """Require a row to bind the actual hook site, answer tokens, and run context."""

    validate_execution_trace(
        {
            "schema_version": row.get("execution_trace_schema_version"),
            "trace_kind": row.get("execution_trace_kind"),
            "item_id": row.get("execution_trace_item_id"),
            "executed_layer_index": row.get("executed_layer_index"),
            "executed_token_position": row.get("executed_token_position"),
            "positive_answer_token_ids": row.get(
                "positive_answer_token_ids"
            ),
            "negative_answer_token_ids": row.get(
                "negative_answer_token_ids"
            ),
            "execution_input_ids": row.get("execution_input_ids"),
            "execution_attention_mask": row.get(
                "execution_attention_mask"
            ),
            "execution_input_sha256": row.get("execution_input_sha256"),
        },
        contract,
    )
    if _id_token(row.get("execution_trace_item_id")) != _id_token(
        row.get("item_id")
    ):
        raise ValueError(
            "execution trace item identity differs from the causal row"
        )
    if row.get("execution_context_sha256") != execution_context_sha256(
        contract
    ):
        raise ValueError("execution-context checksum differs from the contract")


def validate_train_test_firewall(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    *,
    train_groups: Iterable[str] | None = None,
    test_groups: Iterable[str] | None = None,
    split_group_scope: str,
    confirmatory: bool = True,
) -> dict[str, Any]:
    """Validate entity and biological dependency separation for a causal fold."""

    train_id_values = list(train_ids)
    test_id_values = list(test_ids)
    train_id_set = {_id_token(value) for value in train_id_values}
    test_id_set = {_id_token(value) for value in test_id_values}
    if len(train_id_set) != len(train_id_values) or len(test_id_set) != len(
        test_id_values
    ):
        raise ValueError("train/test entity IDs must be unique within each split")
    overlap = sorted(train_id_set & test_id_set)
    if overlap:
        raise ValueError(f"train/test entity overlap: {overlap[:5]}")
    if not train_id_set or not test_id_set:
        raise ValueError("train and test entity sets must both be non-empty")

    proxy_scopes = {
        "",
        "entity_proxy_nonconfirmatory",
        "snapshot_entity_proxy",
        "unavailable",
    }
    train_group_set = None
    test_group_set = None
    if train_groups is not None or test_groups is not None:
        if train_groups is None or test_groups is None:
            raise ValueError("train_groups and test_groups must be supplied together")
        train_group_values = list(train_groups)
        test_group_values = list(test_groups)
        if len(train_group_values) != len(train_id_values) or len(
            test_group_values
        ) != len(test_id_values):
            raise ValueError(
                "train/test group vectors must align one-to-one with entity IDs"
            )
        if any(value is None for value in [*train_group_values, *test_group_values]):
            raise ValueError("train/test split-group IDs cannot be null")
        train_group_set = {_id_token(value) for value in train_group_values}
        test_group_set = {_id_token(value) for value in test_group_values}
        group_overlap = sorted(train_group_set & test_group_set)
        if group_overlap:
            raise ValueError(f"train/test split-group overlap: {group_overlap[:5]}")
    elif confirmatory:
        raise ValueError("confirmatory causal folds require biological split groups")

    if confirmatory and split_group_scope in proxy_scopes:
        raise ValueError(
            f"split_group_scope={split_group_scope!r} is not confirmatory biological grouping"
        )
    return {
        "train_entities": len(train_id_set),
        "test_entities": len(test_id_set),
        "train_groups": None if train_group_set is None else len(train_group_set),
        "test_groups": None if test_group_set is None else len(test_group_set),
        "split_group_scope": split_group_scope,
        "confirmatory": bool(confirmatory),
    }


def _measurement_execution_traces(
    *,
    execution_item_ids: Sequence[Any] | None,
    layer_index: int,
    token_position: int | slice,
    positive_answer_token_ids: int | Sequence[int],
    negative_answer_token_ids: int | Sequence[int],
    execution_input_ids: Sequence[Sequence[int]] | Sequence[int] | None,
    execution_attention_mask: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ),
    expected_batch_size: int | None = None,
) -> list[dict[str, Any]] | None:
    if execution_input_ids is None:
        if execution_attention_mask is not None or execution_item_ids is not None:
            raise ValueError(
                "execution item IDs and attention masks require execution_input_ids"
            )
        return None
    if execution_item_ids is None:
        raise ValueError(
            "execution_input_ids require identity-bound execution_item_ids"
        )
    if isinstance(token_position, slice):
        raise ValueError(
            "confirmatory execution traces require one integer token position"
        )
    traces = build_execution_traces(
        item_ids=execution_item_ids,
        layer_index=layer_index,
        token_position=token_position,
        positive_answer_token_ids=positive_answer_token_ids,
        negative_answer_token_ids=negative_answer_token_ids,
        execution_input_ids=execution_input_ids,
        execution_attention_mask=execution_attention_mask,
    )
    if (
        expected_batch_size is not None
        and len(traces) != expected_batch_size
    ):
        raise ValueError(
            "execution input trace count differs from the measured batch size"
        )
    return traces


def _model_inputs_from_execution_traces(
    traces: Sequence[Mapping[str, Any]] | None,
) -> dict[str, list[list[int]]] | None:
    if traces is None:
        return None
    return {
        "input_ids": [
            list(trace["execution_input_ids"]) for trace in traces
        ],
        "attention_mask": [
            list(trace["execution_attention_mask"]) for trace in traces
        ],
    }


def _execution_measurements_or_none(
    traces: Sequence[Mapping[str, Any]] | None,
    margins: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]] | None:
    if traces is None:
        return None
    return build_execution_measurements(traces, margins)


def _validate_source_execution_traces(
    traces: Sequence[Mapping[str, Any]] | None,
    source_activation: Any,
    *,
    layer_index: int,
    token_position: int | slice,
    positive_answer_token_ids: int | Sequence[int],
    negative_answer_token_ids: int | Sequence[int],
) -> list[dict[str, Any]]:
    """Bind a precomputed donor activation to its identity-bearing source run."""

    if traces is None:
        raise ValueError(
            "precomputed source activations require source execution traces"
        )
    if isinstance(token_position, slice):
        raise ValueError(
            "scientific source traces require one integer token position"
        )
    validated = [validate_execution_trace(trace) for trace in traces]
    source_array = _as_numpy(source_activation)
    if source_array.ndim == 0:
        raise ValueError("source activation must include a feature dimension")
    source_count = 1 if source_array.ndim == 1 else int(source_array.shape[0])
    if len(validated) != source_count:
        raise ValueError(
            "source execution trace count differs from source activation rows"
        )
    item_tokens = [_id_token(trace["item_id"]) for trace in validated]
    if len(item_tokens) != len(set(item_tokens)):
        raise ValueError("source execution traces contain duplicate item IDs")
    expected_positive = list(
        _token_ids(positive_answer_token_ids, "positive_answer")
    )
    expected_negative = list(
        _token_ids(negative_answer_token_ids, "negative_answer")
    )
    for trace in validated:
        if (
            trace["executed_layer_index"] != int(layer_index)
            or trace["executed_token_position"] != int(token_position)
            or trace["positive_answer_token_ids"] != expected_positive
            or trace["negative_answer_token_ids"] != expected_negative
        ):
            raise ValueError(
                "source execution trace differs from the patch measurement site"
            )
    return validated


def capture_source_activations(
    model: Any,
    forward_logits: Callable[..., Any],
    *,
    layer_index: int,
    item_ids: Sequence[Any],
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
    execution_input_ids: Sequence[Sequence[int]] | Sequence[int],
    execution_attention_mask: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
    token_index: int = -1,
) -> dict[str, Any]:
    """Capture donor states together with the exact identity-bound source run."""

    traces = _measurement_execution_traces(
        execution_item_ids=item_ids,
        layer_index=layer_index,
        token_position=token_index,
        positive_answer_token_ids=positive_token_ids,
        negative_answer_token_ids=negative_token_ids,
        execution_input_ids=execution_input_ids,
        execution_attention_mask=execution_attention_mask,
    )
    model_inputs = _model_inputs_from_execution_traces(traces)
    with ResidualStreamCapture(
        model,
        layer_index,
        token_index=token_index,
    ) as capture:
        _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )
    if len(capture.values) != 1:
        raise RuntimeError(
            f"expected one captured residual state, observed {len(capture.values)}"
        )
    activations = capture.values[0]
    checksums = activation_row_sha256(activations)
    if traces is None or len(traces) != len(checksums):
        raise ValueError(
            "source execution trace count differs from captured activation rows"
        )
    return {
        "source_activations": activations,
        "source_activation_sha256": checksums,
        "source_execution_traces": traces,
        "test_label_used_for_intervention": False,
    }


def steering_margin_sweep(
    model: Any,
    forward_logits: Callable[..., Any],
    *,
    layer_index: int,
    direction: Any,
    alphas: Sequence[float],
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
    token_index: int | slice = -1,
    alpha_zero_atol: float = 1e-6,
    dose_scale: float = 1.0,
    execution_item_ids: Sequence[Any] | None = None,
    execution_input_ids: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
    execution_attention_mask: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
) -> dict[str, Any]:
    """Measure the fixed-direction signed answer-logit response over a locked alpha sweep."""

    if not alphas:
        raise ValueError("alphas cannot be empty")
    alpha_values = [float(alpha) for alpha in alphas]
    if len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be unique")

    execution_traces = _measurement_execution_traces(
        execution_item_ids=execution_item_ids,
        layer_index=layer_index,
        token_position=token_index,
        positive_answer_token_ids=positive_token_ids,
        negative_answer_token_ids=negative_token_ids,
        execution_input_ids=execution_input_ids,
        execution_attention_mask=execution_attention_mask,
    )
    model_inputs = _model_inputs_from_execution_traces(execution_traces)
    baseline = _run_margin(
        forward_logits,
        positive_token_ids,
        negative_token_ids,
        model_inputs=model_inputs,
    )
    if execution_traces is not None and len(execution_traces) != len(
        baseline
    ):
        raise ValueError(
            "execution input trace count differs from the measured batch size"
        )
    responses: dict[str, list[float]] = {}
    zero_difference = None
    for alpha in alpha_values:
        transform = steering_transform(
            direction,
            alpha,
            token_index=token_index,
            dose_scale=dose_scale,
        )
        with ResidualStreamIntervention(model, layer_index, transform):
            margin = _run_margin(
                forward_logits,
                positive_token_ids,
                negative_token_ids,
                model_inputs=model_inputs,
            )
        if margin.shape != baseline.shape:
            raise ValueError("intervened and baseline batches have different sizes")
        responses[format(alpha, ".17g")] = margin.tolist()
        if alpha == 0.0:
            zero_difference = float(np.max(np.abs(margin - baseline)))
            if zero_difference > alpha_zero_atol:
                raise RuntimeError(
                    f"alpha=0 changed the answer-logit margin by {zero_difference:.3g}"
                )
    return {
        "baseline_margin": baseline.tolist(),
        "alpha_margin": responses,
        "alphas": alpha_values,
        "alpha_zero_max_abs_difference": zero_difference,
        "estimand": "d_positive_vs_negative_answer_logit_margin_d_alpha",
        "applied_direction_sha256": direction_sha256(direction),
        "steering_dose_scale": float(dose_scale),
        "execution_traces": execution_traces,
        "baseline_measurements": _execution_measurements_or_none(
            execution_traces,
            baseline,
        ),
        "alpha_measurements": {
            alpha: _execution_measurements_or_none(
                execution_traces,
                margin,
            )
            for alpha, margin in responses.items()
        },
        "test_label_used_for_intervention": False,
    }


def erasure_rescue_margins(
    model: Any,
    forward_logits: Callable[..., Any],
    *,
    layer_index: int,
    direction: Any,
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
    token_index: int | slice = -1,
    center: Any | None = None,
    erasure_strength: float = 1.0,
    rescue_strength: float = 1.0,
    rescue_source_activation: Any | None = None,
    rescue_source_execution_traces: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    execution_item_ids: Sequence[Any] | None = None,
    execution_input_ids: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
    execution_attention_mask: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
) -> dict[str, Any]:
    """Measure baseline, directional erasure, and directional rescue margins.

    When ``rescue_source_activation`` is omitted, the same-run pre-erasure component is added back.
    That is an engineering identity control only.  A scientific rescue must provide a held-out,
    preregistered source activation (normally a paired biological counterfactual).
    """

    execution_traces = _measurement_execution_traces(
        execution_item_ids=execution_item_ids,
        layer_index=layer_index,
        token_position=token_index,
        positive_answer_token_ids=positive_token_ids,
        negative_answer_token_ids=negative_token_ids,
        execution_input_ids=execution_input_ids,
        execution_attention_mask=execution_attention_mask,
    )
    model_inputs = _model_inputs_from_execution_traces(execution_traces)
    with ResidualStreamCapture(
        model,
        layer_index,
        token_index=token_index,
    ) as capture:
        baseline = _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )
    if len(capture.values) != 1:
        raise RuntimeError(
            f"expected one captured residual state, observed {len(capture.values)}"
        )
    baseline_activation = capture.values[0]
    if execution_traces is not None and len(execution_traces) != len(
        baseline
    ):
        raise ValueError(
            "execution input trace count differs from the measured batch size"
        )
    if rescue_source_activation is None:
        if rescue_source_execution_traces is not None:
            raise ValueError(
                "source execution traces require a precomputed rescue activation"
            )
        source = baseline_activation
        source_execution_traces = execution_traces
        rescue_kind = "same_run_identity_control"
    else:
        source = rescue_source_activation
        source_execution_traces = _validate_source_execution_traces(
            rescue_source_execution_traces,
            source,
            layer_index=layer_index,
            token_position=token_index,
            positive_answer_token_ids=positive_token_ids,
            negative_answer_token_ids=negative_token_ids,
        )
        rescue_kind = "held_out_counterfactual_source"

    erasure = directional_erasure_transform(
        direction,
        token_index=token_index,
        center=center,
        strength=erasure_strength,
    )
    with ResidualStreamIntervention(model, layer_index, erasure):
        erased = _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )

    rescue = compose_transforms(
        erasure,
        patch_transform(
            source,
            token_index=token_index,
            direction=direction,
            center=center,
            strength=rescue_strength,
        ),
    )
    with ResidualStreamIntervention(model, layer_index, rescue):
        rescued = _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )

    return {
        "baseline_margin": baseline.tolist(),
        "erased_margin": erased.tolist(),
        "rescued_margin": rescued.tolist(),
        "rescue_kind": rescue_kind,
        "source_activation_sha256": activation_row_sha256(source),
        "recipient_activation_sha256": activation_row_sha256(
            baseline_activation
        ),
        "applied_direction_sha256": direction_sha256(direction),
        "erasure_strength": float(erasure_strength),
        "patch_strength": float(rescue_strength),
        "rescue_strength": float(rescue_strength),
        "execution_traces": execution_traces,
        "baseline_measurements": _execution_measurements_or_none(
            execution_traces,
            baseline,
        ),
        "erased_measurements": _execution_measurements_or_none(
            execution_traces,
            erased,
        ),
        "rescued_measurements": _execution_measurements_or_none(
            execution_traces,
            rescued,
        ),
        "source_execution_traces": source_execution_traces,
        "confirmatory_source_trace_available": bool(
            rescue_source_activation is not None
            and source_execution_traces is not None
        ),
        "erasure_center_sha256": activation_sha256(
            (
                np.zeros_like(_as_numpy(direction).reshape(-1))
                if center is None
                else _as_numpy(center).reshape(-1)
            )
        ),
        "test_label_used_for_intervention": False,
    }


def patch_margins(
    model: Any,
    forward_logits: Callable[..., Any],
    *,
    layer_index: int,
    source_activation: Any,
    positive_token_ids: int | Sequence[int],
    negative_token_ids: int | Sequence[int],
    token_index: int | slice = -1,
    direction: Any | None = None,
    center: Any | None = None,
    strength: float = 1.0,
    source_execution_traces: Sequence[Mapping[str, Any]] | None = None,
    execution_item_ids: Sequence[Any] | None = None,
    execution_input_ids: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
    execution_attention_mask: (
        Sequence[Sequence[int]] | Sequence[int] | None
    ) = None,
) -> dict[str, Any]:
    """Measure a base run and a full-state or directional source-to-target patch."""

    execution_traces = _measurement_execution_traces(
        execution_item_ids=execution_item_ids,
        layer_index=layer_index,
        token_position=token_index,
        positive_answer_token_ids=positive_token_ids,
        negative_answer_token_ids=negative_token_ids,
        execution_input_ids=execution_input_ids,
        execution_attention_mask=execution_attention_mask,
    )
    validated_source_execution_traces = _validate_source_execution_traces(
        source_execution_traces,
        source_activation,
        layer_index=layer_index,
        token_position=token_index,
        positive_answer_token_ids=positive_token_ids,
        negative_answer_token_ids=negative_token_ids,
    )
    model_inputs = _model_inputs_from_execution_traces(execution_traces)
    with ResidualStreamCapture(
        model,
        layer_index,
        token_index=token_index,
    ) as capture:
        baseline = _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )
    if len(capture.values) != 1:
        raise RuntimeError(
            f"expected one captured residual state, observed {len(capture.values)}"
        )
    recipient_activation = capture.values[0]
    if execution_traces is not None and len(execution_traces) != len(
        baseline
    ):
        raise ValueError(
            "execution input trace count differs from the measured batch size"
        )
    transform = patch_transform(
        source_activation,
        token_index=token_index,
        direction=direction,
        center=center,
        strength=strength,
    )
    with ResidualStreamIntervention(model, layer_index, transform):
        patched = _run_margin(
            forward_logits,
            positive_token_ids,
            negative_token_ids,
            model_inputs=model_inputs,
        )
    return {
        "baseline_margin": baseline.tolist(),
        "patched_margin": patched.tolist(),
        "source_activation_sha256": activation_row_sha256(source_activation),
        "recipient_activation_sha256": activation_row_sha256(
            recipient_activation
        ),
        "applied_direction_sha256": (
            None if direction is None else direction_sha256(direction)
        ),
        "patch_strength": float(strength),
        "execution_traces": execution_traces,
        "baseline_measurements": _execution_measurements_or_none(
            execution_traces,
            baseline,
        ),
        "patched_measurements": _execution_measurements_or_none(
            execution_traces,
            patched,
        ),
        "source_execution_traces": validated_source_execution_traces,
        "confirmatory_source_trace_available": True,
        "erasure_center_sha256": (
            None
            if direction is None
            else activation_sha256(
                (
                    np.zeros_like(_as_numpy(direction).reshape(-1))
                    if center is None
                    else _as_numpy(center).reshape(-1)
                )
            )
        ),
        "test_label_used_for_intervention": False,
    }


def execution_records(
    *,
    contract: InterventionContract,
    item_ids: Sequence[str],
    split_group_ids: Sequence[str | None],
    condition_measurements: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    alphas: Mapping[str, float | None] | None = None,
    direction_kinds: Mapping[str, str] | None = None,
    direction_ids: Mapping[str, str] | None = None,
    applied_direction_sha256: Mapping[str, str | None] | None = None,
    steering_dose_scales: Mapping[str, float] | None = None,
    intervention_pair_ids: Sequence[str | None] | None = None,
    collateral_kl: Mapping[str, Sequence[float]] | None = None,
    unrelated_margin_change: Mapping[str, Sequence[float]] | None = None,
    direction_norm_relative_error: Mapping[str, Sequence[float]] | None = None,
    projected_variance_relative_error: Mapping[str, Sequence[float]] | None = None,
    erasure_strengths: Mapping[str, float] | None = None,
    erasure_center_sha256: Mapping[str, str | None] | None = None,
    patch_strengths: Mapping[str, float] | None = None,
    source_conditions: Mapping[str, str] | None = None,
    source_activation_sha256: Mapping[str, Sequence[str]] | None = None,
    recipient_activation_sha256: Mapping[str, Sequence[str]] | None = None,
    source_item_ids: Mapping[str, Sequence[str]] | None = None,
    source_intervention_pair_ids: (
        Mapping[str, Sequence[str | None]] | None
    ) = None,
    content_token_count_absolute_difference: (
        Mapping[str, Sequence[int | float]] | None
    ) = None,
    content_embedding_cosine_distance: (
        Mapping[str, Sequence[float]] | None
    ) = None,
    source_content_sha256: Mapping[str, Sequence[str]] | None = None,
    recipient_content_sha256: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Freeze paired intervention outputs without accepting target labels."""

    contract.validate()
    context_sha256 = execution_context_sha256(contract)
    ids = list(item_ids)
    groups = list(split_group_ids)
    for value in ids:
        _id_token(value)
    for value in groups:
        if value is not None:
            _id_token(value)
    pair_ids = (
        [None] * len(ids)
        if intervention_pair_ids is None
        else [
            value
            for value in intervention_pair_ids
        ]
    )
    if not (len(ids) == len(groups) == len(pair_ids)):
        raise ValueError(
            "item IDs, split groups, and intervention pairs must have equal length"
        )
    if len(ids) != len({_id_token(value) for value in ids}):
        raise ValueError("item IDs must be unique")
    rows = []
    for condition, raw_measurements in condition_measurements.items():
        measurements = list(raw_measurements)
        if len(measurements) != len(ids):
            raise ValueError(
                f"condition {condition!r} has {len(measurements)} execution "
                f"measurements for {len(ids)} items"
            )
        measurement_by_item = {}
        for raw_measurement in measurements:
            measurement = validate_execution_measurement(
                raw_measurement,
                contract,
            )
            measurement_item_token = _id_token(measurement["item_id"])
            if measurement_item_token in measurement_by_item:
                raise ValueError(
                    f"condition {condition!r} has duplicate execution-measurement "
                    "item IDs"
                )
            measurement_by_item[measurement_item_token] = measurement
        expected_item_tokens = {_id_token(item_id) for item_id in ids}
        if set(measurement_by_item) != expected_item_tokens:
            raise ValueError(
                f"condition {condition!r} execution measurements must cover the exact "
                "item identity set"
            )
        diagnostic_values = {}
        for diagnostic_name, diagnostic_mapping in (
            ("collateral_kl", collateral_kl),
            ("unrelated_margin_change", unrelated_margin_change),
            (
                "direction_norm_relative_error",
                direction_norm_relative_error,
            ),
            (
                "projected_variance_relative_error",
                projected_variance_relative_error,
            ),
        ):
            if diagnostic_mapping is None or condition not in diagnostic_mapping:
                diagnostic_values[diagnostic_name] = [None] * len(ids)
                continue
            diagnostic = np.asarray(
                diagnostic_mapping[condition],
                dtype=float,
            ).reshape(-1)
            if len(diagnostic) != len(ids) or not np.isfinite(diagnostic).all():
                raise ValueError(
                    f"condition {condition!r} has invalid {diagnostic_name} values"
                )
            diagnostic_values[diagnostic_name] = diagnostic.tolist()
        content_values = {}
        for metric_name, metric_mapping in (
            (
                "content_token_count_absolute_difference",
                content_token_count_absolute_difference,
            ),
            (
                "content_embedding_cosine_distance",
                content_embedding_cosine_distance,
            ),
        ):
            if metric_mapping is None or condition not in metric_mapping:
                content_values[metric_name] = [None] * len(ids)
                continue
            metric = np.asarray(
                metric_mapping[condition],
                dtype=float,
            ).reshape(-1)
            if (
                len(metric) != len(ids)
                or not np.isfinite(metric).all()
                or np.min(metric, initial=0.0) < 0.0
            ):
                raise ValueError(
                    f"condition {condition!r} has invalid {metric_name} values"
                )
            content_values[metric_name] = metric.tolist()
        source_checksums = (
            [None] * len(ids)
            if source_activation_sha256 is None
            or condition not in source_activation_sha256
            else [str(value) for value in source_activation_sha256[condition]]
        )
        if len(source_checksums) != len(ids):
            raise ValueError(
                f"condition {condition!r} has invalid source activation checksums"
            )
        source_content_checksums = (
            [None] * len(ids)
            if source_content_sha256 is None
            or condition not in source_content_sha256
            else [str(value) for value in source_content_sha256[condition]]
        )
        recipient_content_checksums = (
            [None] * len(ids)
            if recipient_content_sha256 is None
            or condition not in recipient_content_sha256
            else [
                str(value)
                for value in recipient_content_sha256[condition]
            ]
        )
        if not (
            len(source_content_checksums)
            == len(recipient_content_checksums)
            == len(ids)
        ):
            raise ValueError(
                f"condition {condition!r} has invalid content checksums"
            )
        recipient_checksums = (
            [None] * len(ids)
            if recipient_activation_sha256 is None
            or condition not in recipient_activation_sha256
            else [
                str(value)
                for value in recipient_activation_sha256[condition]
            ]
        )
        source_items = (
            [None] * len(ids)
            if source_item_ids is None or condition not in source_item_ids
            else list(source_item_ids[condition])
        )
        source_pair_ids = (
            [None] * len(ids)
            if source_intervention_pair_ids is None
            or condition not in source_intervention_pair_ids
            else [
                value
                for value in source_intervention_pair_ids[condition]
            ]
        )
        if not (
            len(recipient_checksums)
            == len(source_items)
            == len(source_pair_ids)
            == len(ids)
        ):
            raise ValueError(
                f"condition {condition!r} has invalid source/recipient provenance"
            )
        for index, item_id in enumerate(ids):
            measurement = measurement_by_item[_id_token(item_id)]
            margin = measurement["answer_logit_margin"]
            execution_trace = measurement["execution_trace"]
            rows.append(
                {
                    "item_id": item_id,
                    "split_group_id": groups[index],
                    "intervention_pair_id": pair_ids[index],
                    "condition": str(condition),
                    "alpha": (
                        None
                        if alphas is None or alphas.get(condition) is None
                        else float(alphas[condition])
                    ),
                    "direction_kind": (
                        "target"
                        if direction_kinds is None
                        else str(direction_kinds.get(condition, "target"))
                    ),
                    "direction_id": (
                        (
                            "target"
                            if direction_kinds is None
                            else str(direction_kinds.get(condition, "target"))
                        )
                        if direction_ids is None
                        else str(
                            direction_ids.get(
                                condition,
                                (
                                    "target"
                                    if direction_kinds is None
                                    else direction_kinds.get(condition, "target")
                                ),
                            )
                        )
                    ),
                    "applied_direction_sha256": (
                        None
                        if applied_direction_sha256 is None
                        or condition not in applied_direction_sha256
                        else applied_direction_sha256[condition]
                    ),
                    "steering_dose_scale": (
                        None
                        if steering_dose_scales is None
                        or condition not in steering_dose_scales
                        else float(steering_dose_scales[condition])
                    ),
                    "answer_logit_margin": margin,
                    "executed_layer_index": execution_trace[
                        "executed_layer_index"
                    ],
                    "executed_token_position": execution_trace[
                        "executed_token_position"
                    ],
                    "positive_answer_token_ids": execution_trace[
                        "positive_answer_token_ids"
                    ],
                    "negative_answer_token_ids": execution_trace[
                        "negative_answer_token_ids"
                    ],
                    "execution_trace_schema_version": execution_trace[
                        "schema_version"
                    ],
                    "execution_trace_kind": execution_trace["trace_kind"],
                    "execution_trace_item_id": execution_trace["item_id"],
                    "execution_input_ids": execution_trace[
                        "execution_input_ids"
                    ],
                    "execution_attention_mask": execution_trace[
                        "execution_attention_mask"
                    ],
                    "execution_input_sha256": execution_trace[
                        "execution_input_sha256"
                    ],
                    "execution_context_sha256": context_sha256,
                    "collateral_kl": diagnostic_values["collateral_kl"][index],
                    "unrelated_margin_change": diagnostic_values[
                        "unrelated_margin_change"
                    ][index],
                    "direction_norm_relative_error": diagnostic_values[
                        "direction_norm_relative_error"
                    ][index],
                    "projected_variance_relative_error": diagnostic_values[
                        "projected_variance_relative_error"
                    ][index],
                    "erasure_strength": (
                        None
                        if erasure_strengths is None
                        or condition not in erasure_strengths
                        else float(erasure_strengths[condition])
                    ),
                    "erasure_center_sha256": (
                        None
                        if erasure_center_sha256 is None
                        or condition not in erasure_center_sha256
                        else erasure_center_sha256[condition]
                    ),
                    "patch_strength": (
                        None
                        if patch_strengths is None
                        or condition not in patch_strengths
                        else float(patch_strengths[condition])
                    ),
                    "source_condition": (
                        None
                        if source_conditions is None
                        or condition not in source_conditions
                        else str(source_conditions[condition])
                    ),
                    "source_activation_sha256": source_checksums[index],
                    "recipient_activation_sha256": recipient_checksums[index],
                    "source_item_id": source_items[index],
                    "source_intervention_pair_id": source_pair_ids[index],
                    "content_token_count_absolute_difference": content_values[
                        "content_token_count_absolute_difference"
                    ][index],
                    "content_embedding_cosine_distance": content_values[
                        "content_embedding_cosine_distance"
                    ][index],
                    "source_content_sha256": source_content_checksums[index],
                    "recipient_content_sha256": (
                        recipient_content_checksums[index]
                    ),
                    "test_label_used_for_intervention": False,
                }
            )
    return bind_execution_receipts(rows)


def intervention_records(
    *,
    execution_rows: Sequence[Mapping[str, Any]],
    labeled_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach outcome labels to already-frozen execution rows.

    ``labeled_items`` is identity-bound rather than positional. It contains exactly
    ``item_id`` and ``target_label`` and must cover the execution item set once.
    """

    label_by_item: dict[str, int] = {}
    for item in labeled_items:
        if set(item) != {"item_id", "target_label"}:
            raise ValueError(
                "labeled_items must contain exactly item_id and target_label"
            )
        token = _id_token(item["item_id"])
        if token in label_by_item:
            raise ValueError("labeled_items contain duplicate item IDs")
        label = validate_binary_label(item["target_label"], "target label")
        label_by_item[token] = label

    for raw_row in execution_rows:
        if set(raw_row) != RECEIPTED_EXECUTION_FIELDS:
            missing = sorted(RECEIPTED_EXECUTION_FIELDS - set(raw_row))
            extras = sorted(set(raw_row) - RECEIPTED_EXECUTION_FIELDS)
            raise ValueError(
                "label attachment requires the exact receipted execution schema; "
                f"missing={missing}, extras={extras}"
            )
    execution_item_tokens = {
        _id_token(row["item_id"]) for row in execution_rows
    }
    if execution_item_tokens != set(label_by_item):
        raise ValueError(
            "labeled_items must cover the exact frozen execution item set"
        )

    rows = []
    for raw_row in execution_rows:
        row = dict(raw_row)
        receipt = row.get("execution_receipt")
        expected_receipt = execution_receipt_payload(row)
        if receipt != expected_receipt or row.get(
            "execution_receipt_sha256"
        ) != execution_receipt_sha256(expected_receipt):
            raise ValueError("execution row changed before labels were attached")
        label = label_by_item[_id_token(row["item_id"])]
        margin = float(row["answer_logit_margin"])
        row["target_label"] = label
        row["correct_answer_margin"] = (2 * label - 1) * margin
        rows.append(row)
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direction_from_frozen_decoder(
    artifact_or_path: Mapping[str, Any] | str | Path,
    *,
    require_hookable: bool = True,
) -> dict[str, Any]:
    """Convert a frozen standardized linear decoder into a raw residual-stream direction.

    For a standardized probe score ``w @ ((h - mean) / scale) + b``, the corresponding raw-space
    direction is ``w / scale``.  The returned vector has unit Euclidean norm; a dimensionless alpha
    can be converted to residual units with ``residual_feature_rms``.  This conversion does not
    establish that the direction is unique or naturally used.
    """

    if isinstance(artifact_or_path, Mapping):
        artifact = dict(artifact_or_path)
        try:
            from .probe_common import frozen_decoder_artifact_sha256
        except ImportError:
            from probe_common import frozen_decoder_artifact_sha256

        artifact_sha = frozen_decoder_artifact_sha256(artifact)
        artifact_source = "in_memory_canonical_json"
    else:
        artifact_path = Path(artifact_or_path)
        with open(artifact_path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        artifact_sha = file_sha256(artifact_path)
        artifact_source = str(artifact_path)

    try:
        from .probe_common import validate_frozen_decoder_artifact
    except ImportError:
        from probe_common import validate_frozen_decoder_artifact

    validate_frozen_decoder_artifact(artifact)
    activation_site = artifact["activation_site"]
    if require_hookable and activation_site["decoder_block_index"] is None:
        raise ValueError(
            "frozen decoder activation site is not mapped to a decoder-block output; "
            "export it with explicit hidden-state semantics before causal intervention"
        )
    model = artifact["model"]
    scaler = model["scaler"]
    decoder = model["decoder"]
    coefficient = np.asarray(decoder["coef"], dtype=float)[0]
    scale = (
        np.ones_like(coefficient)
        if not scaler["with_std"]
        else np.asarray(scaler["scale"], dtype=float)
    )
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("frozen decoder contains invalid scaler scale")
    raw_direction = coefficient / scale
    norm = float(np.linalg.norm(raw_direction))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("frozen decoder has a zero or non-finite raw-space direction")
    center = (
        np.zeros_like(raw_direction)
        if not scaler["with_mean"]
        else np.asarray(scaler["mean"], dtype=float)
    )
    residual_feature_rms = float(np.sqrt(np.mean(np.square(scale))))
    return {
        "direction": raw_direction / norm,
        "center": center,
        "raw_direction_norm": norm,
        "residual_feature_rms": residual_feature_rms,
        "selected_hidden_state_index": int(artifact["selected_layer"]),
        "decoder_block_index": activation_site["decoder_block_index"],
        "activation_site_semantics": activation_site["semantics"],
        "fold_id": artifact["fold_id"],
        "artifact_sha256": artifact_sha,
        "artifact_source": artifact_source,
        "positive_class": int(decoder["positive_class"]),
        "applied_direction_sha256": direction_sha256(raw_direction),
        "direction_center_sha256": activation_sha256(center),
        "normalization": (
            "raw_probe_gradient_unit_l2; dimensionless doses should be multiplied by "
            "residual_feature_rms"
        ),
    }


def write_causal_artifact(
    path: str | Path,
    *,
    contract: InterventionContract,
    split_group_scope: str,
    records: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    intervention_family: str,
    design: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
    frozen_decoder_artifact: Mapping[str, Any] | None = None,
    frozen_decoder_evaluation_artifact: Mapping[str, Any] | None = None,
    content_equivalence_manifest: Mapping[str, Any] | None = None,
    activation_capture_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and write one immutable-input causal measurement artifact."""

    contract.validate()
    if not records:
        raise ValueError("causal artifact cannot be empty")
    if intervention_family not in {
        "steering",
        "erasure_rescue",
        "content_patch",
        "routing_patch",
    }:
        raise ValueError(f"unknown intervention family {intervention_family!r}")
    checksum_by_direction_id = {}
    random_checksums = {}
    for row in records:
        if row.get("test_label_used_for_intervention") is not False:
            raise ValueError("every record must attest that its test label was not used")
        validate_scored_record_schema(row)
        validate_condition_record_schema(intervention_family, row)
        validate_execution_row_context(row, contract)
        receipt = row.get("execution_receipt")
        expected_receipt = execution_receipt_payload(row)
        if receipt != expected_receipt or row.get(
            "execution_receipt_sha256"
        ) != execution_receipt_sha256(expected_receipt):
            raise ValueError(
                "record fields differ from the label-free execution receipt"
            )
        direction_kind = str(row.get("direction_kind", ""))
        if (
            intervention_family == "steering"
            and direction_kind
            in {"target", "random", "shuffled", "surface"}
        ):
            try:
                observed_dose_scale = float(row["steering_dose_scale"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "steering records require the executed dose scale"
                ) from exc
            if not np.isclose(
                observed_dose_scale,
                float(contract.steering_dose_scale),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(
                    "steering record dose scale differs from the contract"
                )
        applied_direction = row.get("applied_direction_sha256")
        if (
            direction_kind == "target"
            and applied_direction != contract.applied_direction_sha256
        ):
            raise ValueError(
                "target record applied-direction checksum differs from the contract"
            )
        if direction_kind in {"random", "shuffled", "surface"} and not _valid_sha256(
            applied_direction
        ):
            raise ValueError(
                "every intervention control record requires an applied-direction checksum"
            )
        if direction_kind in {"target", "random", "shuffled", "surface"}:
            direction_id = str(row.get("direction_id", "")).strip()
            if not direction_id:
                raise ValueError("intervention record direction_id is missing")
            previous = checksum_by_direction_id.setdefault(
                direction_id,
                str(applied_direction),
            )
            if previous != applied_direction:
                raise ValueError(
                    "one direction_id maps to multiple applied-direction checksums"
                )
            if direction_kind == "random":
                random_checksums[direction_id] = str(applied_direction)
        if "item_id" not in row or "condition" not in row:
            raise ValueError("every record requires item_id and condition")
        label = validate_binary_label(
            row.get("target_label"),
            "record target_label",
        )
        try:
            answer_margin = float(row["answer_logit_margin"])
            correct_margin = float(row["correct_answer_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every record requires finite answer and correct-answer margins"
            ) from exc
        expected = (2 * int(label) - 1) * answer_margin
        if (
            not np.isfinite(answer_margin)
            or not np.isfinite(correct_margin)
            or not np.isclose(
                correct_margin,
                expected,
                atol=1e-12,
                rtol=1e-12,
            )
        ):
            raise ValueError(
                "correct_answer_margin must be derived after execution from target_label "
                "and answer_logit_margin"
            )
    if len(set(random_checksums.values())) != len(random_checksums):
        raise ValueError(
            "distinct random direction IDs must use distinct applied-direction checksums"
        )
    if len(set(checksum_by_direction_id.values())) != len(
        checksum_by_direction_id
    ):
        raise ValueError(
            "distinct intervention direction IDs must use distinct checksums"
        )
    embedded_decoder = (
        None
        if frozen_decoder_artifact is None
        else dict(frozen_decoder_artifact)
    )
    if embedded_decoder is not None:
        try:
            from .probe_common import (
                frozen_decoder_artifact_sha256,
                validate_frozen_decoder_artifact,
            )
        except ImportError:
            from probe_common import (
                frozen_decoder_artifact_sha256,
                validate_frozen_decoder_artifact,
            )

        validate_frozen_decoder_artifact(embedded_decoder)
        if (
            frozen_decoder_artifact_sha256(embedded_decoder)
            != contract.direction_artifact_sha256
        ):
            raise ValueError(
                "embedded frozen decoder checksum does not match the intervention contract"
            )
        direction_metadata = direction_from_frozen_decoder(embedded_decoder)
        if (
            direction_metadata["applied_direction_sha256"]
            != contract.applied_direction_sha256
        ):
            raise ValueError(
                "embedded frozen decoder direction does not match the applied direction"
            )
        if (
            direction_metadata["direction_center_sha256"]
            != contract.direction_center_sha256
        ):
            raise ValueError(
                "embedded frozen decoder center does not match the intervention contract"
            )
        if not np.isclose(
            direction_metadata["residual_feature_rms"],
            float(contract.steering_dose_scale),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(
                "embedded frozen decoder dose scale does not match the intervention contract"
            )
    embedded_evaluation = (
        None
        if frozen_decoder_evaluation_artifact is None
        else dict(frozen_decoder_evaluation_artifact)
    )
    if embedded_evaluation is not None:
        if embedded_decoder is None:
            raise ValueError(
                "frozen decoder evaluation requires its train-only decoder artifact"
            )
        try:
            from .probe_common import (
                frozen_decoder_evaluation_artifact_sha256,
                recompute_frozen_decoder_evaluation,
            )
        except ImportError:
            from probe_common import (
                frozen_decoder_evaluation_artifact_sha256,
                recompute_frozen_decoder_evaluation,
            )

        recompute_frozen_decoder_evaluation(
            embedded_evaluation,
            embedded_decoder,
        )
        if (
            frozen_decoder_evaluation_artifact_sha256(
                embedded_evaluation,
                embedded_decoder,
            )
            != contract.decoder_evaluation_artifact_sha256
        ):
            raise ValueError(
                "frozen decoder evaluation checksum does not match the contract"
            )

    contract_record = asdict(contract)
    contract_record["layer_index"] = int(contract.layer_index)
    contract_record["token_position"] = int(contract.token_position)
    contract_record["positive_answer_token_ids"] = list(
        _token_ids(
            contract.positive_answer_token_ids,
            "positive_answer",
        )
    )
    contract_record["negative_answer_token_ids"] = list(
        _token_ids(
            contract.negative_answer_token_ids,
            "negative_answer",
        )
    )
    artifact = {
        "schema_version": CAUSAL_ARTIFACT_SCHEMA,
        "claim_scope": (
            "intervention measurement; causal availability requires controls, and natural use "
            "requires erasure plus rescue/mediation"
        ),
        "intervention_family": intervention_family,
        "contract": contract_record,
        "split_group_scope": str(split_group_scope),
        "provenance": dict(provenance),
        "design": dict(design or {}),
        "frozen_decoder_artifact": embedded_decoder,
        "frozen_decoder_evaluation_artifact": embedded_evaluation,
        "content_equivalence_manifest": (
            None
            if content_equivalence_manifest is None
            else dict(content_equivalence_manifest)
        ),
        "activation_capture_manifest": (
            None
            if activation_capture_manifest is None
            else dict(activation_capture_manifest)
        ),
        "notes": list(notes or []),
        "records": [dict(row) for row in records],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(
            artifact,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return artifact

"""Zero-target-label transfer contracts built on the frozen decoder machinery.

The fit stage may consume source labels. The prediction stage has no target-label
argument and freezes exact target features and probabilities for five required
channels. Target labels are attached by identity only in the evaluation stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from . import probe_common
except ImportError:  # direct execution from eval/
    import probe_common


TRANSFER_SCHEMA_VERSION = 1
SOURCE_SELECTION_SCHEMA_VERSION = 1
SOURCE_SELECTION_ARTIFACT_TYPE = "groundbench.source_only_layer_selection"
TRANSFER_FIT_ARTIFACT_TYPE = "groundbench.zero_target_label_transfer_fit"
TRANSFER_PREDICTION_ARTIFACT_TYPE = "groundbench.zero_target_label_transfer_prediction"
TRANSFER_EVALUATION_ARTIFACT_TYPE = "groundbench.zero_target_label_transfer_evaluation"
TRANSFER_FREEZE_RECEIPT_ARTIFACT_TYPE = "groundbench.zero_target_label_transfer_freeze_receipt"
TRANSFER_FREEZE_COMMITMENT_ARTIFACT_TYPE = "groundbench.zero_target_label_transfer_freeze_commitment"

REQUIRED_CHANNELS = (
    "selected_layer",
    "layer0",
    "raw_input",
    "surface",
    "shuffled_label",
)
INPUT_CHANNELS = REQUIRED_CHANNELS[:-1]
CONTROL_CHANNELS = REQUIRED_CHANNELS[1:]

DMS_RELEASE_BINDING_FIELDS = {
    "dataset_id",
    "preregistration_sha256",
    "input_manifest_sha256",
    "replicate_manifest_sha256",
    "outcome_manifest_sha256",
    "transfer_outcome_rows_sha256",
    "target_contract_sha256",
    "family_map_sha256",
    "outcome_items_sha256",
}
ITEM_FIELD_ORDER = (
    "item_id",
    "group_id",
    "protein_id",
    "wt_sequence_sha256",
    "variant_id",
    "intervention_pair_id",
)
ITEM_FIELDS = set(ITEM_FIELD_ORDER)
LABELED_ITEM_FIELDS = ITEM_FIELDS | {"target_label"}
OUTCOME_ITEM_FIELDS = ITEM_FIELDS | {"target_label", "oriented_effect"}
CHANNEL_INPUT_FIELDS = {"item_ids", "group_ids", "features"}
PILOT_ADJUDICATION_REASONS = [
    "EXTERNAL_DMS_RELEASE_VERIFICATION_REQUIRED",
    "WT_TO_MUTANT_PREDICTION_DELTA_NOT_IMPLEMENTED",
    "ONLY_ONE_SHUFFLED_LABEL_CONTROL",
    "REPLICATION_OR_CONFIRMATORY_SUPPORT_ABSENT",
]
SOURCE_SELECTION_PROVENANCE_FIELDS = {
    "source_task_id",
    "model_revision",
    "prompt_protocol_sha256",
    "model_forward_code_sha256",
    "split_group_scope",
    "group_id_source",
    "producer",
}
FIT_PROVENANCE_FIELDS = {
    "source_task_id",
    "target_task_id",
    "model_revision",
    "prompt_protocol_sha256",
    "model_forward_code_sha256",
    "split_group_scope",
    "group_id_source",
    "producer",
}
EXTRACTION_PROVENANCE_FIELDS = {
    "target_task_id",
    "model_revision",
    "prompt_protocol_sha256",
    "model_forward_code_sha256",
    "producer",
}


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _id_token(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(
            value,
            (str, int, float),
        )
    ):
        raise TypeError("item and group IDs must be non-null non-boolean JSON scalars")
    if isinstance(value, float) and not np.isfinite(value):
        raise TypeError("item and group IDs must be finite JSON scalars")
    return f"{type(value).__name__}:" + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _strict_binary_label(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) not in {0, 1}:
        raise ValueError(f"{field} must be the literal integer 0 or 1")
    return int(value)


def _json_safe(value: Any, field: str = "value") -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item, field) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item(), field)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field} dictionary keys must be strings")
        return {key: _json_safe(item, f"{field}.{key}") for key, item in sorted(value.items())}
    raise TypeError(f"{field} contains unsupported JSON value {type(value).__name__}")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _matrix_sha256(values: Any) -> str:
    matrix = np.asarray(values, dtype=np.dtype("<f8"), order="C")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("channel features must be a finite 2D matrix")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "canonicalization": "float64-le-c-order-v1",
                "shape": list(matrix.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _validate_dataset_descriptor(
    descriptor: Mapping[str, Any],
    field: str,
) -> dict[str, str]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != (DMS_RELEASE_BINDING_FIELDS):
        raise ValueError(f"{field} must use the exact DMS release-binding schema")
    dataset_id = descriptor.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"{field}.dataset_id must be a non-empty string")
    result = {"dataset_id": dataset_id}
    for digest_field in sorted(DMS_RELEASE_BINDING_FIELDS - {"dataset_id"}):
        value = descriptor.get(digest_field)
        if not _valid_sha256(value):
            raise ValueError(f"{field}.{digest_field} must be a lowercase SHA-256 digest")
        result[digest_field] = value
    return result


def _dataset_binding_sha256(descriptor: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(_validate_dataset_descriptor(descriptor, "dataset"))


def _release_bindings_sha256(
    source_dataset: Mapping[str, Any],
    target_dataset: Mapping[str, Any],
) -> str:
    return _canonical_json_sha256(
        {
            "source": _validate_dataset_descriptor(
                source_dataset,
                "source_dataset",
            ),
            "target": _validate_dataset_descriptor(
                target_dataset,
                "target_dataset",
            ),
        }
    )


def _validate_release_pair(
    source_dataset: Mapping[str, Any],
    target_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    target_items: Sequence[Mapping[str, Any]],
) -> None:
    if (
        source_dataset["dataset_id"] == target_dataset["dataset_id"]
        or source_dataset["input_manifest_sha256"] == target_dataset["input_manifest_sha256"]
    ):
        raise ValueError("source and target DMS releases must be independently identified")
    if source_dataset["family_map_sha256"] != target_dataset["family_map_sha256"]:
        raise ValueError("source and target must share the committed global family map")
    for role, descriptor, items in (
        ("source", source_dataset, source_items),
        ("target", target_dataset, target_items),
    ):
        observed = _item_order_sha256(items)
        if descriptor["outcome_items_sha256"] != observed:
            raise ValueError(f"{role} outcome_items_sha256 differs from exact item identity rows")


def _normalise_item_record(
    item: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != ITEM_FIELDS:
        raise ValueError(f"{field} must contain exactly {sorted(ITEM_FIELDS)}")
    result = {}
    for identity_field in ITEM_FIELD_ORDER:
        value = item[identity_field]
        if identity_field == "wt_sequence_sha256":
            if not _valid_sha256(value):
                raise ValueError(f"{field}.wt_sequence_sha256 must be a SHA-256 digest")
        else:
            _id_token(value)
        result[identity_field] = _json_safe(
            value,
            f"{field}.{identity_field}",
        )
    return result


def _item_identity_tokens(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_id_token(item[field]) for field in ITEM_FIELD_ORDER)


def _normalise_items(
    items: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of item records")
    result = []
    seen = set()
    for index, item in enumerate(items):
        row = _normalise_item_record(
            item,
            field=f"{field}[{index}]",
        )
        item_token = _id_token(row["item_id"])
        if item_token in seen:
            raise ValueError(f"{field} contains duplicate item IDs")
        seen.add(item_token)
        result.append(row)
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result


def _ordered_labels(
    labeled_items: Sequence[Mapping[str, Any]],
    expected_items: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> np.ndarray:
    label_by_item: dict[str, tuple[tuple[str, ...], int]] = {}
    for index, item in enumerate(labeled_items):
        if not isinstance(item, Mapping) or set(item) != LABELED_ITEM_FIELDS:
            raise ValueError(f"{field} records must contain the exact transfer identity fields and target_label")
        identity = _normalise_item_record(
            {key: item[key] for key in ITEM_FIELD_ORDER},
            field=f"{field}[{index}]",
        )
        item_token = _id_token(identity["item_id"])
        if item_token in label_by_item:
            raise ValueError(f"{field} contains duplicate item IDs")
        label_by_item[item_token] = (
            _item_identity_tokens(identity),
            _strict_binary_label(item["target_label"], f"{field}.target_label"),
        )
    expected_tokens = {_id_token(item["item_id"]) for item in expected_items}
    if set(label_by_item) != expected_tokens:
        raise ValueError(f"{field} must cover the exact expected item set")
    labels = []
    for item in expected_items:
        identity_tokens, label = label_by_item[_id_token(item["item_id"])]
        if identity_tokens != _item_identity_tokens(item):
            raise ValueError(f"{field} identity differs from the frozen item map")
        labels.append(label)
    values = np.asarray(labels, dtype=int)
    if set(values.tolist()) != {0, 1}:
        raise ValueError(f"{field} must contain both binary classes")
    return values


def _strict_oriented_effect(value: Any, field: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite numeric value")
    return float(value)


def _ordered_outcomes(
    outcome_items: Sequence[Mapping[str, Any]],
    expected_items: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    outcome_by_item: dict[str, tuple[tuple[str, ...], int, float]] = {}
    for index, item in enumerate(outcome_items):
        if not isinstance(item, Mapping) or set(item) != OUTCOME_ITEM_FIELDS:
            raise ValueError(
                f"{field} records must contain the exact transfer identity fields, target_label, and oriented_effect"
            )
        identity = _normalise_item_record(
            {key: item[key] for key in ITEM_FIELD_ORDER},
            field=f"{field}[{index}]",
        )
        item_token = _id_token(identity["item_id"])
        if item_token in outcome_by_item:
            raise ValueError(f"{field} contains duplicate item IDs")
        outcome_by_item[item_token] = (
            _item_identity_tokens(identity),
            _strict_binary_label(item["target_label"], f"{field}.target_label"),
            _strict_oriented_effect(
                item["oriented_effect"],
                f"{field}.oriented_effect",
            ),
        )
    expected_tokens = {_id_token(item["item_id"]) for item in expected_items}
    if set(outcome_by_item) != expected_tokens:
        raise ValueError(f"{field} must cover the exact expected item set")
    labels = []
    effects = []
    for item in expected_items:
        identity_tokens, label, effect = outcome_by_item[_id_token(item["item_id"])]
        if identity_tokens != _item_identity_tokens(item):
            raise ValueError(f"{field} identity differs from the frozen item map")
        labels.append(label)
        effects.append(effect)
    label_values = np.asarray(labels, dtype=int)
    if set(label_values.tolist()) != {0, 1}:
        raise ValueError(f"{field} must contain both binary classes")
    return label_values, np.asarray(effects, dtype=float)


def _item_order_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_json_sha256([{field: item[field] for field in ITEM_FIELD_ORDER} for item in items])


def _group_map_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_json_sha256(
        sorted(
            (
                _id_token(item["item_id"]),
                _id_token(item["group_id"]),
            )
            for item in items
        )
    )


def transfer_outcome_items_sha256(
    items: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the exact ordered, value-free identity rows of one DMS outcome set."""

    rows = _normalise_items(items, field="outcome_items")
    return _item_order_sha256(rows)


def _outcome_rows_sha256(
    items: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_json_sha256(
        [
            {
                **{field: item[field] for field in ITEM_FIELD_ORDER},
                "target_label": int(item["target_label"]),
                "oriented_effect": float(item["oriented_effect"]),
            }
            for item in items
        ]
    )


def _source_labeled_rows_sha256(
    items: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_json_sha256(
        [
            {
                **{field: item[field] for field in ITEM_FIELD_ORDER},
                "target_label": int(item["target_label"]),
            }
            for item in items
        ]
    )


def _labels_sha256(
    items: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> str:
    return _canonical_json_sha256(
        [
            {
                **{field: item[field] for field in ITEM_FIELD_ORDER},
                "target_label": int(label),
            }
            for item, label in zip(items, labels, strict=True)
        ]
    )


def _validate_disjoint_source_target(
    source_items: Sequence[Mapping[str, Any]],
    target_items: Sequence[Mapping[str, Any]],
) -> None:
    identity_fields = (
        ("item_id", "item IDs"),
        ("group_id", "biological groups"),
        ("protein_id", "protein IDs"),
        ("wt_sequence_sha256", "WT sequence identities"),
        ("variant_id", "variant IDs"),
        ("intervention_pair_id", "intervention-pair IDs"),
    )
    for field, label in identity_fields:
        source_values = {_id_token(item[field]) for item in source_items}
        target_values = {_id_token(item[field]) for item in target_items}
        if source_values & target_values:
            raise ValueError(f"source and target {label} must be globally disjoint")


def _validate_channel_input(
    payload: Mapping[str, Any],
    expected_items: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> np.ndarray:
    if not isinstance(payload, Mapping) or set(payload) != CHANNEL_INPUT_FIELDS:
        raise ValueError(f"{field} must contain exactly item_ids, group_ids, and features")
    item_ids = list(payload["item_ids"])
    group_ids = list(payload["group_ids"])
    expected_ids = [item["item_id"] for item in expected_items]
    expected_groups = [item["group_id"] for item in expected_items]
    if [_id_token(value) for value in item_ids] != [_id_token(value) for value in expected_ids]:
        raise ValueError(f"{field} item row order differs from the frozen item map")
    if [_id_token(value) for value in group_ids] != [_id_token(value) for value in expected_groups]:
        raise ValueError(f"{field} group row order differs from the frozen item map")
    matrix = np.asarray(payload["features"], dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(expected_items) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError(f"{field} features must be a finite aligned 2D matrix")
    return matrix


def _validate_fit_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or set(provenance) != (FIT_PROVENANCE_FIELDS):
        raise ValueError("fit provenance must use the exact transfer schema")
    result = dict(provenance)
    for field in (
        "source_task_id",
        "target_task_id",
        "model_revision",
        "split_group_scope",
        "group_id_source",
        "producer",
    ):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"fit provenance {field} is missing")
    for field in ("prompt_protocol_sha256", "model_forward_code_sha256"):
        if not _valid_sha256(result.get(field)):
            raise ValueError(f"fit provenance {field} must be a SHA-256 digest")
    if result["producer"] != "eval.latent_transfer":
        raise ValueError("fit provenance producer must be eval.latent_transfer")
    return result


def _validate_source_selection_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or set(provenance) != (SOURCE_SELECTION_PROVENANCE_FIELDS):
        raise ValueError("source-selection provenance must use the exact source-only schema")
    result = dict(provenance)
    for field in (
        "source_task_id",
        "model_revision",
        "split_group_scope",
        "group_id_source",
        "producer",
    ):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"source-selection provenance {field} is missing")
    for field in ("prompt_protocol_sha256", "model_forward_code_sha256"):
        if not _valid_sha256(result.get(field)):
            raise ValueError(f"source-selection provenance {field} must be a SHA-256 digest")
    if result["producer"] != "eval.latent_transfer":
        raise ValueError("source-selection provenance producer must be eval.latent_transfer")
    return result


def _validate_extraction_provenance(
    provenance: Mapping[str, Any],
    fit_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or set(provenance) != (EXTRACTION_PROVENANCE_FIELDS):
        raise ValueError("prediction extraction provenance must use the exact transfer schema")
    result = dict(provenance)
    for field in ("target_task_id", "model_revision", "producer"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"prediction extraction provenance {field} is missing")
    for field in ("prompt_protocol_sha256", "model_forward_code_sha256"):
        if not _valid_sha256(result.get(field)):
            raise ValueError(f"prediction extraction provenance {field} must be a SHA-256 digest")
    for field in (
        "target_task_id",
        "model_revision",
        "prompt_protocol_sha256",
        "model_forward_code_sha256",
    ):
        if result[field] != fit_provenance[field]:
            raise ValueError(f"prediction extraction provenance {field} differs from the fit")
    if result["producer"] != "eval.latent_transfer":
        raise ValueError("prediction extraction provenance producer must be eval.latent_transfer")
    return result


def _deterministic_shuffled_labels(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("shuffle_seed must be an integer")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(labels))
    shuffled = labels[permutation]
    if np.array_equal(shuffled, labels):
        zero_index = int(np.flatnonzero(labels == 0)[0])
        one_index = int(np.flatnonzero(labels == 1)[0])
        permutation[zero_index], permutation[one_index] = (
            permutation[one_index],
            permutation[zero_index],
        )
        shuffled = labels[permutation]
    if np.array_equal(shuffled, labels):
        raise RuntimeError("failed to construct a non-identity shuffled-label control")
    return shuffled, permutation.astype(int).tolist()


def _normalise_analysis_lock(
    n_boot: Any,
    seed: Any,
    confidence_level: Any,
) -> dict[str, int | float]:
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, int)
        or n_boot < 2
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("transfer evaluation analysis lock is invalid")
    return {
        "n_boot": n_boot,
        "seed": seed,
        "confidence_level": float(confidence_level),
    }


def _spearman_rank_correlation(
    probabilities: np.ndarray,
    oriented_effects: np.ndarray,
) -> float | None:
    if len(probabilities) < 2 or np.ptp(probabilities) == 0.0 or np.ptp(oriented_effects) == 0.0:
        return None
    correlation = float(
        np.corrcoef(
            rankdata(probabilities, method="average"),
            rankdata(oriented_effects, method="average"),
        )[0, 1]
    )
    return correlation if np.isfinite(correlation) else None


def _effect_monotonicity_metrics(
    items: Sequence[Mapping[str, Any]],
    oriented_effects: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    n_boot: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    group_tokens = [_id_token(item["group_id"]) for item in items]
    unique_groups = sorted(set(group_tokens))
    group_indices = {
        group: np.asarray(
            [index for index, observed_group in enumerate(group_tokens) if observed_group == group],
            dtype=int,
        )
        for group in unique_groups
    }
    observed = {
        channel: _spearman_rank_correlation(
            np.asarray(probabilities[channel], dtype=float),
            oriented_effects,
        )
        for channel in REQUIRED_CHANNELS
    }
    bootstrap_values: dict[str, list[float]] = {channel: [] for channel in REQUIRED_CHANNELS}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        for channel in REQUIRED_CHANNELS:
            correlation = _spearman_rank_correlation(
                np.asarray(probabilities[channel], dtype=float)[indices],
                oriented_effects[indices],
            )
            if correlation is not None:
                bootstrap_values[channel].append(correlation)
    tail = (1.0 - confidence_level) / 2.0
    channels = {}
    for channel in REQUIRED_CHANNELS:
        values = bootstrap_values[channel]
        channels[channel] = {
            "spearman": observed[channel],
            "confidence_interval": (
                [
                    float(np.quantile(values, tail)),
                    float(np.quantile(values, 1.0 - tail)),
                ]
                if values
                else None
            ),
            "n_valid_resamples": len(values),
        }
    return {
        "metric_status": (
            "metric_reported_not_adjudicated" if observed["selected_layer"] is not None else "not_estimable"
        ),
        "metric": "effect_monotonicity_spearman",
        "orientation": ("higher_class_1_probability_should_track_higher_oriented_effect"),
        "n_items": len(items),
        "n_groups": len(unique_groups),
        "channels": channels,
        "bootstrap": {
            "method": "biological_group_percentile",
            "n_resamples": n_boot,
            "seed": seed,
            "confidence_level": confidence_level,
        },
    }


def _pilot_adjudication() -> dict[str, Any]:
    return {
        "status": "NOT_ADJUDICATED",
        "claim_scope": "schema_v1_pilot_only",
        "reasons": list(PILOT_ADJUDICATION_REASONS),
    }


def _channel_activation_site(channel: str, selected_layer: int) -> dict[str, Any]:
    if channel == "selected_layer":
        return {
            "hidden_state_index": selected_layer,
            "decoder_block_index": selected_layer - 1,
            "semantics": "hf_hidden_states_decoder_block_output",
        }
    if channel == "layer0":
        return {
            "hidden_state_index": 0,
            "decoder_block_index": None,
            "semantics": "hf_pre_decoder_embedding",
        }
    if channel == "shuffled_label":
        return {
            "hidden_state_index": selected_layer,
            "decoder_block_index": selected_layer - 1,
            "semantics": "selected_hidden_state_shuffled_source_labels_control",
        }
    return {
        "hidden_state_index": 0,
        "decoder_block_index": None,
        "semantics": f"{channel}_control_not_model_hidden_state",
    }


def _channel_layer(channel: str, selected_layer: int) -> int:
    return selected_layer if channel in {"selected_layer", "shuffled_label"} else 0


def _normalise_selection_layers(
    layer_channels: Mapping[int, Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[int, np.ndarray]]:
    if not isinstance(layer_channels, Mapping):
        raise TypeError("source layer channels must be a mapping")
    layers = []
    for layer in layer_channels:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or int(layer) < 1:
            raise ValueError("source layer channel keys must be positive integer layers")
        layers.append(int(layer))
    if len(set(layers)) != len(layers) or len(layers) < 2:
        raise ValueError("source-only layer selection requires at least two unique layers")
    layers.sort()
    matrices = {
        layer: _validate_channel_input(
            layer_channels[layer],
            source_items,
            field=f"source_layer_channels.{layer}",
        )
        for layer in layers
    }
    return layers, matrices


def _source_group_cv_splits(
    items: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    n_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("source-selection n_splits must be an integer >= 2")
    group_tokens = np.asarray(
        [_id_token(item["group_id"]) for item in items],
        dtype=object,
    )
    if n_splits > len(set(group_tokens.tolist())):
        raise ValueError("source-selection n_splits exceeds the number of source groups")
    splitter = GroupKFold(n_splits=n_splits)
    splits = [
        (np.asarray(train, dtype=int), np.asarray(validation, dtype=int))
        for train, validation in splitter.split(
            np.zeros(len(items), dtype=float),
            labels,
            group_tokens,
        )
    ]
    fold_records = []
    for fold_index, (train, validation) in enumerate(splits):
        if set(labels[train].tolist()) != {0, 1}:
            raise ValueError("every source-selection training fold must contain both classes")
        fold_records.append(
            {
                "fold_index": fold_index,
                "train_item_ids": [items[index]["item_id"] for index in train],
                "validation_item_ids": [items[index]["item_id"] for index in validation],
                "train_group_ids": [items[index]["group_id"] for index in train],
                "validation_group_ids": [items[index]["group_id"] for index in validation],
            }
        )
    return splits, _canonical_json_sha256(fold_records)


def _selection_summary_from_layer_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    means = [float(record["mean_auroc"]) for record in records]
    best_position = int(np.argmax(np.asarray(means, dtype=float)))
    best_record = records[best_position]
    threshold = float(best_record["mean_auroc"] - best_record["standard_error"])
    band = [int(record["layer"]) for record in records if float(record["mean_auroc"]) >= threshold]
    return {
        "best_mean_layer": int(best_record["layer"]),
        "best_mean_auroc": float(best_record["mean_auroc"]),
        "best_standard_error": float(best_record["standard_error"]),
        "one_standard_error_threshold": threshold,
        "one_standard_error_band": band,
        "selected_layer": min(band),
    }


def _compute_source_selection_artifact(
    *,
    source_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    source_labeled_items: Sequence[Mapping[str, Any]],
    layer_channels: Mapping[int, Mapping[str, Any]],
    n_splits: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    source_descriptor = _validate_dataset_descriptor(
        source_dataset,
        "source_dataset",
    )
    source_rows = _normalise_items(source_items, field="source_items")
    if source_descriptor["outcome_items_sha256"] != _item_order_sha256(source_rows):
        raise ValueError("source outcome_items_sha256 differs from selection item identities")
    labels = _ordered_labels(
        source_labeled_items,
        source_rows,
        field="source_labeled_items",
    )
    source_labeled_rows = [
        {**item, "target_label": int(label)} for item, label in zip(source_rows, labels, strict=True)
    ]
    if source_descriptor["transfer_outcome_rows_sha256"] != (_source_labeled_rows_sha256(source_labeled_rows)):
        raise ValueError("source transfer outcome-row binding differs from selection labels")
    layers, matrices = _normalise_selection_layers(
        layer_channels,
        source_rows,
    )
    selection_provenance = _validate_source_selection_provenance(provenance)
    splits, fold_assignment_sha256 = _source_group_cv_splits(
        source_rows,
        labels,
        n_splits,
    )
    layer_records = []
    for layer in layers:
        fold_aurocs: list[float | None] = []
        for train, validation in splits:
            classifier = probe_common.balanced_lr().fit(
                matrices[layer][train],
                labels[train],
            )
            probabilities = classifier.predict_proba(matrices[layer][validation])[:, 1]
            fold_aurocs.append(
                float(roc_auc_score(labels[validation], probabilities))
                if set(labels[validation].tolist()) == {0, 1}
                else None
            )
        valid_scores = np.asarray(
            [score for score in fold_aurocs if score is not None],
            dtype=float,
        )
        if len(valid_scores) < 2:
            raise ValueError("source-selection requires at least two two-class validation folds")
        layer_records.append(
            {
                "layer": layer,
                "feature_count": int(matrices[layer].shape[1]),
                "source_activation_sha256": _matrix_sha256(matrices[layer]),
                "fold_aurocs": fold_aurocs,
                "mean_auroc": float(valid_scores.mean()),
                "standard_error": float(valid_scores.std(ddof=1) / np.sqrt(len(valid_scores))),
            }
        )
    return {
        "schema_version": SOURCE_SELECTION_SCHEMA_VERSION,
        "artifact_type": SOURCE_SELECTION_ARTIFACT_TYPE,
        "label_access_policy": "source_labels_only",
        "source_dataset": source_descriptor,
        "items": source_rows,
        "bindings": {
            "source_item_order_sha256": _item_order_sha256(source_rows),
            "source_group_map_sha256": _group_map_sha256(source_rows),
            "source_labels_sha256": _labels_sha256(source_rows, labels),
            "fold_assignment_sha256": fold_assignment_sha256,
        },
        "cv_lock": {
            "splitter": "GroupKFold",
            "n_splits": n_splits,
            "scoring": "roc_auc",
            "classifier": ("standard_scaler_plus_balanced_l2_logistic_regression_c1"),
            "selection_rule": "one_standard_error_lowest_layer_v1",
        },
        "candidate_layer_order": layers,
        "layers": layer_records,
        "selection": _selection_summary_from_layer_records(layer_records),
        "provenance": selection_provenance,
    }


def _same_finite_float(observed: Any, expected: float) -> bool:
    return bool(
        not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and np.isfinite(float(observed))
        and np.isclose(
            float(observed),
            float(expected),
            rtol=0.0,
            atol=1e-15,
        )
    )


def _validate_source_selection_schema(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "artifact_type",
        "label_access_policy",
        "source_dataset",
        "items",
        "bindings",
        "cv_lock",
        "candidate_layer_order",
        "layers",
        "selection",
        "provenance",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected:
        raise ValueError("source-selection artifact must use the exact schema")
    if (
        artifact.get("schema_version") != SOURCE_SELECTION_SCHEMA_VERSION
        or artifact.get("artifact_type") != SOURCE_SELECTION_ARTIFACT_TYPE
        or artifact.get("label_access_policy") != "source_labels_only"
    ):
        raise ValueError("unsupported source-selection artifact")
    source_dataset = _validate_dataset_descriptor(
        artifact["source_dataset"],
        "source_selection.source_dataset",
    )
    source_items = _normalise_items(
        artifact["items"],
        field="source_selection.items",
    )
    if source_dataset["outcome_items_sha256"] != _item_order_sha256(source_items):
        raise ValueError("source-selection outcome_items_sha256 differs from item identities")
    bindings = artifact.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source_item_order_sha256",
        "source_group_map_sha256",
        "source_labels_sha256",
        "fold_assignment_sha256",
    }:
        raise ValueError("source-selection bindings have an invalid schema")
    if (
        bindings.get("source_item_order_sha256") != _item_order_sha256(source_items)
        or bindings.get("source_group_map_sha256") != _group_map_sha256(source_items)
        or not _valid_sha256(bindings.get("source_labels_sha256"))
        or not _valid_sha256(bindings.get("fold_assignment_sha256"))
    ):
        raise ValueError("source-selection bindings are invalid")
    cv_lock = artifact.get("cv_lock")
    if not isinstance(cv_lock, Mapping) or set(cv_lock) != {
        "splitter",
        "n_splits",
        "scoring",
        "classifier",
        "selection_rule",
    }:
        raise ValueError("source-selection CV lock has an invalid schema")
    if (
        cv_lock.get("splitter") != "GroupKFold"
        or isinstance(cv_lock.get("n_splits"), bool)
        or not isinstance(cv_lock.get("n_splits"), int)
        or cv_lock["n_splits"] < 2
        or cv_lock.get("scoring") != "roc_auc"
        or cv_lock.get("classifier") != "standard_scaler_plus_balanced_l2_logistic_regression_c1"
        or cv_lock.get("selection_rule") != "one_standard_error_lowest_layer_v1"
    ):
        raise ValueError("source-selection CV lock is invalid")
    layer_order = artifact.get("candidate_layer_order")
    if (
        not isinstance(layer_order, list)
        or len(layer_order) < 2
        or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 1 for layer in layer_order)
        or layer_order != sorted(set(layer_order))
    ):
        raise ValueError("source-selection candidate layer order is invalid")
    records = artifact.get("layers")
    if not isinstance(records, list) or len(records) != len(layer_order):
        raise ValueError("source-selection layer records are incomplete")
    validated_records = []
    for expected_layer, record in zip(layer_order, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "layer",
            "feature_count",
            "source_activation_sha256",
            "fold_aurocs",
            "mean_auroc",
            "standard_error",
        }:
            raise ValueError("source-selection layer record has an invalid schema")
        scores = record.get("fold_aurocs")
        if (
            record.get("layer") != expected_layer
            or isinstance(record.get("feature_count"), bool)
            or not isinstance(record.get("feature_count"), int)
            or record["feature_count"] < 1
            or not _valid_sha256(record.get("source_activation_sha256"))
            or not isinstance(scores, list)
            or len(scores) != cv_lock["n_splits"]
        ):
            raise ValueError("source-selection layer metadata is invalid")
        valid_scores = []
        for score in scores:
            if score is None:
                continue
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not np.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError("source-selection fold AUROC is invalid")
            valid_scores.append(float(score))
        if len(valid_scores) < 2:
            raise ValueError("source-selection layer lacks two-class validation folds")
        expected_mean = float(np.mean(valid_scores))
        expected_se = float(np.std(valid_scores, ddof=1) / np.sqrt(len(valid_scores)))
        if not _same_finite_float(
            record.get("mean_auroc"),
            expected_mean,
        ) or not _same_finite_float(
            record.get("standard_error"),
            expected_se,
        ):
            raise ValueError("source-selection layer summary differs from folds")
        validated_records.append(dict(record))
    expected_selection = _selection_summary_from_layer_records(validated_records)
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != set(expected_selection):
        raise ValueError("source-selection decision has an invalid schema")
    for field, expected_value in expected_selection.items():
        observed = selection.get(field)
        if isinstance(expected_value, float):
            if not _same_finite_float(observed, expected_value):
                raise ValueError("source-selection decision differs from locked rule")
        elif observed != expected_value:
            raise ValueError("source-selection decision differs from locked rule")
    provenance = _validate_source_selection_provenance(artifact.get("provenance"))
    json.dumps(
        _json_safe(dict(artifact), "source_selection_artifact"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        "source_dataset": source_dataset,
        "source_items": source_items,
        "selected_layer": expected_selection["selected_layer"],
        "provenance": provenance,
    }


def build_source_selection_artifact(
    *,
    source_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    source_labeled_items: Sequence[Mapping[str, Any]],
    layer_channels: Mapping[int, Mapping[str, Any]],
    n_splits: int = 5,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one layer by deterministic grouped CV using source data only."""

    artifact = _compute_source_selection_artifact(
        source_dataset=source_dataset,
        source_items=source_items,
        source_labeled_items=source_labeled_items,
        layer_channels=layer_channels,
        n_splits=n_splits,
        provenance=provenance,
    )
    _validate_source_selection_schema(artifact)
    return artifact


def validate_source_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    source_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    source_labeled_items: Sequence[Mapping[str, Any]],
    layer_channels: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the complete source-only selection from its bound inputs."""

    values = _validate_source_selection_schema(artifact)
    expected = _compute_source_selection_artifact(
        source_dataset=source_dataset,
        source_items=source_items,
        source_labeled_items=source_labeled_items,
        layer_channels=layer_channels,
        n_splits=artifact["cv_lock"]["n_splits"],
        provenance=artifact["provenance"],
    )
    if dict(artifact) != expected:
        raise ValueError("source-selection artifact differs from source-only recomputation")
    return values


def source_selection_artifact_sha256(
    artifact: Mapping[str, Any],
) -> str:
    _validate_source_selection_schema(artifact)
    return hashlib.sha256(_artifact_bytes(artifact)).hexdigest()


def build_transfer_fit_artifact(
    *,
    source_dataset: Mapping[str, Any],
    target_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    target_items: Sequence[Mapping[str, Any]],
    source_labeled_items: Sequence[Mapping[str, Any]],
    source_channels: Mapping[str, Mapping[str, Any]],
    source_layer_channels: Mapping[int, Mapping[str, Any]],
    selected_layer: int,
    source_selection: Mapping[str, Any],
    shuffle_seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit all transfer channels with source labels and zero target labels."""

    source_descriptor = _validate_dataset_descriptor(
        source_dataset,
        "source_dataset",
    )
    target_descriptor = _validate_dataset_descriptor(
        target_dataset,
        "target_dataset",
    )
    source_rows = _normalise_items(source_items, field="source_items")
    target_rows = _normalise_items(target_items, field="target_items")
    _validate_disjoint_source_target(source_rows, target_rows)
    _validate_release_pair(
        source_descriptor,
        target_descriptor,
        source_rows,
        target_rows,
    )
    labels = _ordered_labels(
        source_labeled_items,
        source_rows,
        field="source_labeled_items",
    )
    source_labeled_rows = [
        {
            **item,
            "target_label": int(label),
        }
        for item, label in zip(source_rows, labels, strict=True)
    ]
    if source_descriptor["transfer_outcome_rows_sha256"] != (_source_labeled_rows_sha256(source_labeled_rows)):
        raise ValueError("source transfer_outcome_rows_sha256 differs from source labeled rows")
    if isinstance(selected_layer, bool) or not isinstance(selected_layer, (int, np.integer)) or int(selected_layer) < 1:
        raise ValueError("selected_layer must be a positive integer")
    selected_layer = int(selected_layer)
    selection_values = validate_source_selection_artifact(
        source_selection,
        source_dataset=source_descriptor,
        source_items=source_rows,
        source_labeled_items=source_labeled_items,
        layer_channels=source_layer_channels,
    )
    if selected_layer != selection_values["selected_layer"]:
        raise ValueError("selected_layer differs from the locked source-only selection")
    fit_provenance = _validate_fit_provenance(provenance)
    for field in SOURCE_SELECTION_PROVENANCE_FIELDS - {"producer"}:
        if field in fit_provenance and selection_values["provenance"][field] != fit_provenance[field]:
            raise ValueError(f"source-selection provenance {field} differs from the fit")
    if set(source_channels) != set(INPUT_CHANNELS):
        raise ValueError(f"source_channels must contain exactly {list(INPUT_CHANNELS)}")
    source_matrices = {
        channel: _validate_channel_input(
            source_channels[channel],
            source_rows,
            field=f"source_channels.{channel}",
        )
        for channel in INPUT_CHANNELS
    }
    candidate_layers, candidate_matrices = _normalise_selection_layers(
        source_layer_channels,
        source_rows,
    )
    if candidate_layers != source_selection["candidate_layer_order"]:
        raise ValueError("source selection candidate-layer order differs from inputs")
    selected_record = next(record for record in source_selection["layers"] if record["layer"] == selected_layer)
    if _matrix_sha256(source_matrices["selected_layer"]) != selected_record["source_activation_sha256"]:
        raise ValueError("selected-layer fit features differ from the source selection")
    shuffled_labels, permutation = _deterministic_shuffled_labels(
        labels,
        shuffle_seed,
    )
    source_matrices["shuffled_label"] = source_matrices["selected_layer"]

    source_ids = [item["item_id"] for item in source_rows]
    source_groups = [item["group_id"] for item in source_rows]
    target_ids = [item["item_id"] for item in target_rows]
    target_groups = [item["group_id"] for item in target_rows]
    source_labels_sha256 = source_selection["bindings"]["source_labels_sha256"]
    channels = {}
    for channel in REQUIRED_CHANNELS:
        channel_labels = shuffled_labels if channel == "shuffled_label" else labels
        classifier = probe_common.balanced_lr().fit(
            source_matrices[channel],
            channel_labels,
        )
        decoder = probe_common.build_frozen_decoder_artifact(
            classifier,
            source_matrices[channel],
            channel_labels,
            selected_layer=_channel_layer(channel, selected_layer),
            fold_id="source_all_to_independent_target",
            train_ids=source_ids,
            test_ids=target_ids,
            train_group_ids=source_groups,
            test_group_ids=target_groups,
            provenance={
                "task_id": fit_provenance["source_task_id"],
                "model_revision": fit_provenance["model_revision"],
                "source_dataset_sha256": _dataset_binding_sha256(source_descriptor),
                "split_group_scope": fit_provenance["split_group_scope"],
                "prompt_protocol_sha256": fit_provenance["prompt_protocol_sha256"],
                "model_forward_code_sha256": fit_provenance["model_forward_code_sha256"],
                "producer": "eval.latent_transfer",
                "role": channel,
                "group_id_source": fit_provenance["group_id_source"],
            },
            activation_site=_channel_activation_site(channel, selected_layer),
        )
        channels[channel] = {
            "feature_role": channel,
            "source_feature_sha256": _matrix_sha256(source_matrices[channel]),
            "source_feature_matrix": _json_safe(
                source_matrices[channel],
                f"source_channels.{channel}.features",
            ),
            "source_labels_sha256": _labels_sha256(
                source_rows,
                channel_labels,
            ),
            "decoder_artifact_sha256": (probe_common.frozen_decoder_artifact_sha256(decoder)),
            "decoder_artifact": decoder,
        }

    artifact = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_type": TRANSFER_FIT_ARTIFACT_TYPE,
        "source_dataset": source_descriptor,
        "target_dataset": target_descriptor,
        "release_bindings_sha256": _release_bindings_sha256(
            source_descriptor,
            target_descriptor,
        ),
        "release_binding_assurance": ("external_dms_release_verification_required_pilot_v1"),
        "label_access_policy": "source_labels_only_target_labels_unavailable",
        "source_selection": {
            "artifact_sha256": source_selection_artifact_sha256(source_selection),
            "artifact": dict(source_selection),
            "candidate_layer_feature_matrices": [
                {
                    "layer": layer,
                    "source_feature_sha256": _matrix_sha256(candidate_matrices[layer]),
                    "source_feature_matrix": _json_safe(
                        candidate_matrices[layer],
                        f"source_layer_channels.{layer}.features",
                    ),
                }
                for layer in candidate_layers
            ],
        },
        "shuffle_seed": int(shuffle_seed),
        "shuffled_label_permutation": permutation,
        "items": {
            "source": source_rows,
            "target": target_rows,
        },
        "source_labeled_items": source_labeled_rows,
        "channel_order": list(REQUIRED_CHANNELS),
        "channels": channels,
        "provenance": fit_provenance,
    }
    validate_transfer_fit_artifact(artifact)
    return artifact


def validate_transfer_fit_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a source-only multi-channel transfer fit artifact."""

    expected = {
        "schema_version",
        "artifact_type",
        "source_dataset",
        "target_dataset",
        "release_bindings_sha256",
        "release_binding_assurance",
        "label_access_policy",
        "source_selection",
        "shuffle_seed",
        "shuffled_label_permutation",
        "items",
        "source_labeled_items",
        "channel_order",
        "channels",
        "provenance",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected:
        raise ValueError("transfer fit artifact must use the exact schema")
    if (
        artifact.get("schema_version") != TRANSFER_SCHEMA_VERSION
        or artifact.get("artifact_type") != TRANSFER_FIT_ARTIFACT_TYPE
    ):
        raise ValueError("unsupported transfer fit artifact")
    source_dataset = _validate_dataset_descriptor(
        artifact["source_dataset"],
        "source_dataset",
    )
    target_dataset = _validate_dataset_descriptor(
        artifact["target_dataset"],
        "target_dataset",
    )
    if artifact.get("release_bindings_sha256") != (_release_bindings_sha256(source_dataset, target_dataset)):
        raise ValueError("transfer fit DMS release-binding checksum differs")
    if artifact.get("release_binding_assurance") != ("external_dms_release_verification_required_pilot_v1"):
        raise ValueError("transfer fit release-binding assurance is invalid")
    if artifact.get("label_access_policy") != ("source_labels_only_target_labels_unavailable"):
        raise ValueError("transfer fit label-access policy is invalid")
    items = artifact.get("items")
    if not isinstance(items, Mapping) or set(items) != {"source", "target"}:
        raise ValueError("transfer fit items must contain source and target")
    source_items = _normalise_items(items["source"], field="items.source")
    target_items = _normalise_items(items["target"], field="items.target")
    _validate_disjoint_source_target(source_items, target_items)
    _validate_release_pair(
        source_dataset,
        target_dataset,
        source_items,
        target_items,
    )
    source_labels = _ordered_labels(
        artifact.get("source_labeled_items"),
        source_items,
        field="source_labeled_items",
    )
    expected_labeled_items = [
        {
            **item,
            "target_label": int(label),
        }
        for item, label in zip(source_items, source_labels, strict=True)
    ]
    if artifact.get("source_labeled_items") != expected_labeled_items:
        raise ValueError("serialized source labeled records are not in exact item order")
    if source_dataset["transfer_outcome_rows_sha256"] != (_source_labeled_rows_sha256(expected_labeled_items)):
        raise ValueError("source transfer outcome-row binding differs from labeled records")
    provenance = _validate_fit_provenance(artifact["provenance"])

    selection = artifact.get("source_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "artifact_sha256",
        "artifact",
        "candidate_layer_feature_matrices",
    }:
        raise ValueError("transfer source-selection record has an invalid schema")
    selection_artifact = selection.get("artifact")
    selection_values = _validate_source_selection_schema(selection_artifact)
    if selection.get("artifact_sha256") != source_selection_artifact_sha256(selection_artifact):
        raise ValueError("transfer source-selection artifact checksum differs")
    if selection_values["source_dataset"] != source_dataset:
        raise ValueError("transfer source-selection dataset differs from the fit")
    if [_item_identity_tokens(item) for item in selection_values["source_items"]] != [
        _item_identity_tokens(item) for item in source_items
    ]:
        raise ValueError("transfer source-selection item map differs from the fit")
    candidate_records = selection.get("candidate_layer_feature_matrices")
    candidate_order = selection_artifact["candidate_layer_order"]
    if not isinstance(candidate_records, list) or len(candidate_records) != len(candidate_order):
        raise ValueError("transfer source-selection candidate matrices are incomplete")
    selection_layer_records = {record["layer"]: record for record in selection_artifact["layers"]}
    embedded_layer_channels = {}
    for expected_layer, record in zip(
        candidate_order,
        candidate_records,
        strict=True,
    ):
        if not isinstance(record, Mapping) or set(record) != {
            "layer",
            "source_feature_sha256",
            "source_feature_matrix",
        }:
            raise ValueError("transfer source-selection candidate matrix schema is invalid")
        matrix = np.asarray(record.get("source_feature_matrix"), dtype=float)
        if (
            record.get("layer") != expected_layer
            or matrix.ndim != 2
            or len(matrix) != len(source_items)
            or matrix.shape[1] < 1
            or not np.isfinite(matrix).all()
            or record.get("source_feature_sha256") != _matrix_sha256(matrix)
            or record["source_feature_sha256"] != selection_layer_records[expected_layer]["source_activation_sha256"]
        ):
            raise ValueError("transfer source-selection candidate matrix differs")
        embedded_layer_channels[expected_layer] = {
            "item_ids": [item["item_id"] for item in source_items],
            "group_ids": [item["group_id"] for item in source_items],
            "features": matrix,
        }
    selection_values = validate_source_selection_artifact(
        selection_artifact,
        source_dataset=source_dataset,
        source_items=source_items,
        source_labeled_items=expected_labeled_items,
        layer_channels=embedded_layer_channels,
    )
    for field in SOURCE_SELECTION_PROVENANCE_FIELDS - {"producer"}:
        if field in provenance and selection_values["provenance"][field] != provenance[field]:
            raise ValueError(f"transfer source-selection provenance {field} differs")
    selected_layer = selection_values["selected_layer"]
    selected_layer_record = next(record for record in selection_artifact["layers"] if record["layer"] == selected_layer)
    if not _valid_sha256(selected_layer_record.get("source_activation_sha256")):
        raise ValueError("transfer source-selection record is invalid")
    if isinstance(artifact.get("shuffle_seed"), bool) or not isinstance(artifact.get("shuffle_seed"), int):
        raise ValueError("transfer shuffle_seed must be an integer")
    permutation = artifact.get("shuffled_label_permutation")
    if (
        not isinstance(permutation, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in permutation)
        or sorted(permutation) != list(range(len(source_items)))
    ):
        raise ValueError("shuffled-label permutation is invalid")
    shuffled_labels, expected_permutation = _deterministic_shuffled_labels(
        source_labels,
        artifact["shuffle_seed"],
    )
    if permutation != expected_permutation:
        raise ValueError("shuffled-label permutation differs from the locked source labels")
    if artifact.get("channel_order") != list(REQUIRED_CHANNELS):
        raise ValueError("transfer channel order is not canonical")
    channels = artifact.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != set(REQUIRED_CHANNELS):
        raise ValueError("transfer fit is missing required channels")

    source_ids = [item["item_id"] for item in source_items]
    source_groups = [item["group_id"] for item in source_items]
    target_ids = [item["item_id"] for item in target_items]
    target_groups = [item["group_id"] for item in target_items]
    source_matrices = {}
    for channel in REQUIRED_CHANNELS:
        record = channels[channel]
        record_fields = {
            "feature_role",
            "source_feature_sha256",
            "source_feature_matrix",
            "source_labels_sha256",
            "decoder_artifact_sha256",
            "decoder_artifact",
        }
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError(f"transfer fit channel {channel!r} has an invalid schema")
        if (
            record.get("feature_role") != channel
            or not _valid_sha256(record.get("source_feature_sha256"))
            or not _valid_sha256(record.get("source_labels_sha256"))
        ):
            raise ValueError(f"transfer fit channel {channel!r} metadata is invalid")
        matrix = np.asarray(record.get("source_feature_matrix"), dtype=float)
        if (
            matrix.ndim != 2
            or len(matrix) != len(source_items)
            or matrix.shape[1] < 1
            or not np.isfinite(matrix).all()
            or record["source_feature_sha256"] != _matrix_sha256(matrix)
        ):
            raise ValueError(f"transfer fit channel {channel!r} source matrix differs")
        source_matrices[channel] = matrix
        channel_labels = shuffled_labels if channel == "shuffled_label" else source_labels
        if record["source_labels_sha256"] != _labels_sha256(
            source_items,
            channel_labels,
        ):
            raise ValueError(f"transfer fit channel {channel!r} source labels differ")
        decoder = record.get("decoder_artifact")
        probe_common.validate_frozen_decoder_artifact(decoder)
        if record.get("decoder_artifact_sha256") != (probe_common.frozen_decoder_artifact_sha256(decoder)):
            raise ValueError(f"transfer fit channel {channel!r} decoder checksum differs")
        split = decoder["split"]
        for observed, expected_values, field in (
            (split["train_ids"], source_ids, "train IDs"),
            (split["train_group_ids"], source_groups, "train groups"),
            (split["test_ids"], target_ids, "target IDs"),
            (split["test_group_ids"], target_groups, "target groups"),
        ):
            if [_id_token(value) for value in observed] != [_id_token(value) for value in expected_values]:
                raise ValueError(f"transfer fit channel {channel!r} {field} differ")
        expected_site = _channel_activation_site(channel, selected_layer)
        if (
            decoder["selected_layer"] != _channel_layer(channel, selected_layer)
            or decoder["activation_site"] != expected_site
        ):
            raise ValueError(f"transfer fit channel {channel!r} feature site differs")
        training_provenance = decoder["training"]["provenance"]
        expected_training = {
            "task_id": provenance["source_task_id"],
            "model_revision": provenance["model_revision"],
            "source_dataset_sha256": _dataset_binding_sha256(source_dataset),
            "split_group_scope": provenance["split_group_scope"],
            "prompt_protocol_sha256": provenance["prompt_protocol_sha256"],
            "model_forward_code_sha256": provenance["model_forward_code_sha256"],
            "producer": "eval.latent_transfer",
            "role": channel,
            "group_id_source": provenance["group_id_source"],
        }
        if training_provenance != expected_training:
            raise ValueError(f"transfer fit channel {channel!r} provenance differs")
        replay_classifier = probe_common.balanced_lr().fit(
            matrix,
            channel_labels,
        )
        replay_decoder = probe_common.build_frozen_decoder_artifact(
            replay_classifier,
            matrix,
            channel_labels,
            selected_layer=_channel_layer(channel, selected_layer),
            fold_id="source_all_to_independent_target",
            train_ids=source_ids,
            test_ids=target_ids,
            train_group_ids=source_groups,
            test_group_ids=target_groups,
            provenance=expected_training,
            activation_site=_channel_activation_site(
                channel,
                selected_layer,
            ),
        )
        if decoder != replay_decoder:
            raise ValueError(f"transfer fit channel {channel!r} decoder differs from replay")
        if record["decoder_artifact_sha256"] != (probe_common.frozen_decoder_artifact_sha256(replay_decoder)):
            raise ValueError(f"transfer fit channel {channel!r} replay checksum differs")
    if channels["selected_layer"]["source_feature_sha256"] != channels["shuffled_label"]["source_feature_sha256"]:
        raise ValueError("shuffled-label control must use the selected-layer source features")
    if not np.array_equal(
        source_matrices["selected_layer"],
        source_matrices["shuffled_label"],
    ):
        raise ValueError("shuffled-label control must embed selected-layer source features")
    if channels["selected_layer"]["source_feature_sha256"] != (selected_layer_record["source_activation_sha256"]):
        raise ValueError("transfer selected-layer features differ from source selection")
    source_labels_sha256 = selection_artifact["bindings"]["source_labels_sha256"]
    for channel in INPUT_CHANNELS:
        if channels[channel]["source_labels_sha256"] != source_labels_sha256:
            raise ValueError(f"transfer fit channel {channel!r} changed source labels")
    if channels["shuffled_label"]["source_labels_sha256"] == (source_labels_sha256):
        raise ValueError("shuffled-label control did not change source labels")
    json.dumps(
        _json_safe(dict(artifact), "transfer_fit_artifact"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        "source_items": source_items,
        "target_items": target_items,
        "selected_layer": selected_layer,
    }


def transfer_fit_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    validate_transfer_fit_artifact(artifact)
    return hashlib.sha256(_artifact_bytes(artifact)).hexdigest()


def build_transfer_prediction_artifact(
    fit_artifact: Mapping[str, Any],
    target_channels: Mapping[str, Mapping[str, Any]],
    *,
    extraction_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze target features and predictions without a target-label argument."""

    fit_values = validate_transfer_fit_artifact(fit_artifact)
    target_items = fit_values["target_items"]
    if set(target_channels) != set(INPUT_CHANNELS):
        raise ValueError(f"target_channels must contain exactly {list(INPUT_CHANNELS)}")
    target_matrices = {
        channel: _validate_channel_input(
            target_channels[channel],
            target_items,
            field=f"target_channels.{channel}",
        )
        for channel in INPUT_CHANNELS
    }
    target_matrices["shuffled_label"] = target_matrices["selected_layer"]
    provenance = _validate_extraction_provenance(
        extraction_provenance,
        fit_artifact["provenance"],
    )
    target_ids = [item["item_id"] for item in target_items]
    target_groups = [item["group_id"] for item in target_items]
    channels = {}
    for channel in REQUIRED_CHANNELS:
        decoder_record = fit_artifact["channels"][channel]
        matrix = target_matrices[channel]
        probabilities = probe_common.predict_with_frozen_decoder(
            decoder_record["decoder_artifact"],
            matrix,
            target_ids=target_ids,
            target_group_ids=target_groups,
        )
        channels[channel] = {
            "feature_source_channel": ("selected_layer" if channel == "shuffled_label" else channel),
            "feature_matrix_sha256": _matrix_sha256(matrix),
            "feature_matrix": _json_safe(
                matrix,
                f"target_channels.{channel}.features",
            ),
            "decoder_artifact_sha256": decoder_record["decoder_artifact_sha256"],
            "probabilities": probabilities.tolist(),
        }
    artifact = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_type": TRANSFER_PREDICTION_ARTIFACT_TYPE,
        "fit_artifact_sha256": transfer_fit_artifact_sha256(fit_artifact),
        "source_dataset": dict(fit_artifact["source_dataset"]),
        "target_dataset": dict(fit_artifact["target_dataset"]),
        "label_access_policy": ("target_labels_unavailable_until_prediction_artifact_frozen"),
        "items": list(target_items),
        "channel_order": list(REQUIRED_CHANNELS),
        "channels": channels,
        "extraction_provenance": provenance,
    }
    validate_transfer_prediction_artifact(artifact, fit_artifact)
    return artifact


def validate_transfer_prediction_artifact(
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Recompute every label-free target prediction from embedded features."""

    fit_values = validate_transfer_fit_artifact(fit_artifact)
    expected = {
        "schema_version",
        "artifact_type",
        "fit_artifact_sha256",
        "source_dataset",
        "target_dataset",
        "label_access_policy",
        "items",
        "channel_order",
        "channels",
        "extraction_provenance",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected:
        raise ValueError("transfer prediction artifact must use the exact schema")
    if (
        artifact.get("schema_version") != TRANSFER_SCHEMA_VERSION
        or artifact.get("artifact_type") != TRANSFER_PREDICTION_ARTIFACT_TYPE
    ):
        raise ValueError("unsupported transfer prediction artifact")
    if artifact.get("fit_artifact_sha256") != (transfer_fit_artifact_sha256(fit_artifact)):
        raise ValueError("transfer prediction fit-artifact checksum differs")
    if (
        artifact.get("source_dataset") != fit_artifact["source_dataset"]
        or artifact.get("target_dataset") != fit_artifact["target_dataset"]
    ):
        raise ValueError("transfer prediction dataset identity differs from the fit")
    if artifact.get("label_access_policy") != ("target_labels_unavailable_until_prediction_artifact_frozen"):
        raise ValueError("transfer prediction label-access policy is invalid")
    target_items = _normalise_items(
        artifact.get("items"),
        field="prediction.items",
    )
    if [_item_identity_tokens(item) for item in target_items] != [
        _item_identity_tokens(item) for item in fit_values["target_items"]
    ]:
        raise ValueError("transfer prediction identity row order differs from the fit")
    if artifact.get("channel_order") != list(REQUIRED_CHANNELS):
        raise ValueError("transfer prediction channel order is not canonical")
    channels = artifact.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != set(REQUIRED_CHANNELS):
        raise ValueError("transfer prediction is missing required channels")
    _validate_extraction_provenance(
        artifact.get("extraction_provenance"),
        fit_artifact["provenance"],
    )
    target_ids = [item["item_id"] for item in target_items]
    target_groups = [item["group_id"] for item in target_items]
    recomputed = {}
    matrices = {}
    for channel in REQUIRED_CHANNELS:
        record = channels[channel]
        record_fields = {
            "feature_source_channel",
            "feature_matrix_sha256",
            "feature_matrix",
            "decoder_artifact_sha256",
            "probabilities",
        }
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError(f"transfer prediction channel {channel!r} has an invalid schema")
        expected_source = "selected_layer" if channel == "shuffled_label" else channel
        if record.get("feature_source_channel") != expected_source:
            raise ValueError(f"transfer prediction channel {channel!r} feature source differs")
        matrix = np.asarray(record.get("feature_matrix"), dtype=float)
        if (
            matrix.ndim != 2
            or len(matrix) != len(target_items)
            or not np.isfinite(matrix).all()
            or record.get("feature_matrix_sha256") != _matrix_sha256(matrix)
        ):
            raise ValueError(f"transfer prediction channel {channel!r} feature matrix differs")
        fit_channel = fit_artifact["channels"][channel]
        if record.get("decoder_artifact_sha256") != fit_channel["decoder_artifact_sha256"]:
            raise ValueError(f"transfer prediction channel {channel!r} decoder checksum differs")
        expected_probability = probe_common.predict_with_frozen_decoder(
            fit_channel["decoder_artifact"],
            matrix,
            target_ids=target_ids,
            target_group_ids=target_groups,
        )
        reported = np.asarray(record.get("probabilities"), dtype=float)
        if (
            reported.shape != expected_probability.shape
            or not np.isfinite(reported).all()
            or np.min(reported) < 0.0
            or np.max(reported) > 1.0
            or not np.allclose(
                reported,
                expected_probability,
                atol=1e-15,
                rtol=1e-15,
            )
        ):
            raise ValueError(f"transfer prediction channel {channel!r} probabilities differ")
        matrices[channel] = matrix
        recomputed[channel] = expected_probability
    if not np.array_equal(
        matrices["selected_layer"],
        matrices["shuffled_label"],
    ):
        raise ValueError("shuffled-label prediction must reuse selected-layer target features")
    json.dumps(
        _json_safe(dict(artifact), "transfer_prediction_artifact"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return recomputed


def transfer_prediction_artifact_sha256(
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> str:
    validate_transfer_prediction_artifact(artifact, fit_artifact)
    return hashlib.sha256(_artifact_bytes(artifact)).hexdigest()


def _validate_freeze_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("freeze receipt timestamp must be a UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("freeze receipt timestamp must use YYYY-MM-DDTHH:MM:SSZ") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("freeze receipt timestamp is not canonical")
    return value


def _exclusive_durable_write(
    path: str | os.PathLike[str],
    payload: bytes,
) -> str:
    target = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(target)
        except OSError:
            pass
        raise
    return target


def _read_regular_file_no_follow(
    path: str | os.PathLike[str],
    *,
    field: str,
) -> tuple[str, bytes]:
    try:
        target = os.path.abspath(os.fspath(path))
    except TypeError as exc:
        raise ValueError(f"{field} must be a filesystem path") from exc
    try:
        link_status = os.lstat(target)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(link_status.st_mode):
        raise ValueError(f"{field} must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ValueError(f"{field} could not be opened safely") from exc
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return target, payload


def _freeze_commitment_path(receipt_path: str) -> str:
    return f"{receipt_path}.commit.json"


def _freeze_commitment_artifact(
    receipt_path: str,
    receipt_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_type": TRANSFER_FREEZE_COMMITMENT_ARTIFACT_TYPE,
        "assurance": ("detached_local_commitment_not_remote_attestation"),
        "receipt_path": receipt_path,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_size_bytes": len(receipt_bytes),
    }


def _validate_freeze_commitment(
    receipt_path: str,
    receipt_bytes: bytes,
) -> None:
    commitment_path = _freeze_commitment_path(receipt_path)
    _, commitment_bytes = _read_regular_file_no_follow(
        commitment_path,
        field="prediction freeze receipt commitment",
    )
    try:
        commitment = json.loads(commitment_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction freeze receipt commitment is not canonical JSON") from exc
    expected = _freeze_commitment_artifact(receipt_path, receipt_bytes)
    if commitment != expected or commitment_bytes != _artifact_bytes(expected):
        raise ValueError("prediction freeze receipt differs from its durable commitment")


def validate_transfer_prediction_freeze_receipt(
    receipt_path: str | os.PathLike[str],
    prediction_artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a local tamper-evident receipt; this is not remote attestation."""

    committed_path, receipt_bytes = _read_regular_file_no_follow(
        receipt_path,
        field="prediction freeze receipt",
    )
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction freeze receipt is not canonical JSON") from exc
    expected = {
        "schema_version",
        "artifact_type",
        "assurance",
        "committed_receipt_path",
        "fit_artifact_sha256",
        "prediction_artifact_sha256",
        "prediction_size_bytes",
        "frozen_prediction_path",
        "target_outcome_items_sha256",
        "target_transfer_outcome_rows_sha256",
        "created_at_utc",
        "producer",
        "nonce",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise ValueError("prediction freeze receipt must use the exact schema")
    if (
        receipt.get("schema_version") != TRANSFER_SCHEMA_VERSION
        or receipt.get("artifact_type") != TRANSFER_FREEZE_RECEIPT_ARTIFACT_TYPE
        or receipt.get("assurance") != "tamper_evident_local_receipt_not_remote_attestation"
    ):
        raise ValueError("unsupported prediction freeze receipt")
    if receipt.get("committed_receipt_path") != committed_path:
        raise ValueError("freeze receipt is not at its committed filesystem path")
    expected_fit_sha256 = transfer_fit_artifact_sha256(fit_artifact)
    expected_prediction_sha256 = transfer_prediction_artifact_sha256(
        prediction_artifact,
        fit_artifact,
    )
    if receipt.get("fit_artifact_sha256") != expected_fit_sha256:
        raise ValueError("freeze receipt fit checksum differs")
    if receipt.get("prediction_artifact_sha256") != (expected_prediction_sha256):
        raise ValueError("freeze receipt prediction checksum differs")
    prediction_size = receipt.get("prediction_size_bytes")
    if isinstance(prediction_size, bool) or not isinstance(prediction_size, int) or prediction_size < 1:
        raise ValueError("freeze receipt prediction byte size is invalid")
    frozen_path = receipt.get("frozen_prediction_path")
    if (
        not isinstance(frozen_path, str)
        or not os.path.isabs(frozen_path)
        or os.path.abspath(frozen_path) != frozen_path
    ):
        raise ValueError("freeze receipt frozen_prediction_path must be absolute")
    validated_frozen_path, frozen_bytes = _read_regular_file_no_follow(
        frozen_path,
        field="frozen prediction file",
    )
    if validated_frozen_path != frozen_path:
        raise ValueError("freeze receipt frozen prediction path differs")
    expected_bytes = _artifact_bytes(prediction_artifact)
    if (
        len(frozen_bytes) != prediction_size
        or len(frozen_bytes) != len(expected_bytes)
        or hashlib.sha256(frozen_bytes).hexdigest() != expected_prediction_sha256
        or frozen_bytes != expected_bytes
    ):
        raise ValueError("frozen prediction bytes differ from the freeze receipt")
    if receipt.get("target_outcome_items_sha256") != fit_artifact["target_dataset"]["outcome_items_sha256"]:
        raise ValueError("freeze receipt target outcome-items checksum differs")
    if (
        receipt.get("target_transfer_outcome_rows_sha256")
        != fit_artifact["target_dataset"]["transfer_outcome_rows_sha256"]
    ):
        raise ValueError("freeze receipt transfer outcome-row checksum differs")
    _validate_freeze_timestamp(receipt.get("created_at_utc"))
    if receipt.get("producer") != "eval.latent_transfer":
        raise ValueError("freeze receipt producer must be eval.latent_transfer")
    nonce = receipt.get("nonce")
    if not isinstance(nonce, str) or len(nonce) < 16 or not nonce.strip():
        raise ValueError("freeze receipt nonce must be a non-empty string of length >= 16")
    json.dumps(
        _json_safe(dict(receipt), "prediction_freeze_receipt"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if receipt_bytes != _artifact_bytes(receipt):
        raise ValueError("prediction freeze receipt bytes are not canonical")
    _validate_freeze_commitment(committed_path, receipt_bytes)
    return dict(receipt)


def transfer_prediction_freeze_receipt_sha256(
    receipt_path: str | os.PathLike[str],
    prediction_artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> str:
    validate_transfer_prediction_freeze_receipt(
        receipt_path,
        prediction_artifact,
        fit_artifact,
    )
    _, receipt_bytes = _read_regular_file_no_follow(
        receipt_path,
        field="prediction freeze receipt",
    )
    return hashlib.sha256(receipt_bytes).hexdigest()


def create_transfer_prediction_freeze_receipt(
    receipt_path: str | os.PathLike[str],
    prediction_artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
    *,
    prediction_path: str | os.PathLike[str],
    created_at_utc: str,
    nonce: str,
    producer: str = "eval.latent_transfer",
) -> str:
    """Durably exclusive-create a local tamper-evident prediction receipt."""

    validate_transfer_prediction_artifact(prediction_artifact, fit_artifact)
    prediction_bytes = _artifact_bytes(prediction_artifact)
    frozen_prediction_path = os.path.abspath(os.fspath(prediction_path))
    target = os.path.abspath(os.fspath(receipt_path))
    receipt = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_type": TRANSFER_FREEZE_RECEIPT_ARTIFACT_TYPE,
        "assurance": ("tamper_evident_local_receipt_not_remote_attestation"),
        "committed_receipt_path": target,
        "fit_artifact_sha256": transfer_fit_artifact_sha256(fit_artifact),
        "prediction_artifact_sha256": (
            transfer_prediction_artifact_sha256(
                prediction_artifact,
                fit_artifact,
            )
        ),
        "prediction_size_bytes": len(prediction_bytes),
        "frozen_prediction_path": frozen_prediction_path,
        "target_outcome_items_sha256": fit_artifact["target_dataset"]["outcome_items_sha256"],
        "target_transfer_outcome_rows_sha256": fit_artifact["target_dataset"]["transfer_outcome_rows_sha256"],
        "created_at_utc": created_at_utc,
        "producer": producer,
        "nonce": nonce,
    }
    _exclusive_durable_write(frozen_prediction_path, prediction_bytes)
    receipt_created = False
    commitment_created = False
    commitment_path = _freeze_commitment_path(target)
    try:
        receipt_bytes = _artifact_bytes(receipt)
        _exclusive_durable_write(target, receipt_bytes)
        receipt_created = True
        commitment = _freeze_commitment_artifact(target, receipt_bytes)
        _exclusive_durable_write(
            commitment_path,
            _artifact_bytes(commitment),
        )
        commitment_created = True
        validate_transfer_prediction_freeze_receipt(
            target,
            prediction_artifact,
            fit_artifact,
        )
    except Exception:
        if commitment_created:
            try:
                os.unlink(commitment_path)
            except OSError:
                pass
        if receipt_created:
            try:
                os.unlink(target)
            except OSError:
                pass
        try:
            os.unlink(frozen_prediction_path)
        except OSError:
            pass
        raise
    return target


def build_transfer_evaluation_artifact(
    prediction_artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
    target_outcome_items: Sequence[Mapping[str, Any]],
    *,
    freeze_receipt_path: str | os.PathLike[str],
    n_boot: int = 2000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Attach identity-bound labels and oriented effects after prediction freezes."""

    probabilities = validate_transfer_prediction_artifact(
        prediction_artifact,
        fit_artifact,
    )
    prediction_artifact_sha256 = hashlib.sha256(_artifact_bytes(prediction_artifact)).hexdigest()
    validated_receipt = validate_transfer_prediction_freeze_receipt(
        freeze_receipt_path,
        prediction_artifact,
        fit_artifact,
    )
    committed_receipt_path = validated_receipt["committed_receipt_path"]
    _, committed_receipt_bytes = _read_regular_file_no_follow(
        committed_receipt_path,
        field="prediction freeze receipt",
    )
    freeze_receipt_sha256 = hashlib.sha256(committed_receipt_bytes).hexdigest()
    analysis_lock = _normalise_analysis_lock(
        n_boot,
        seed,
        confidence_level,
    )
    target_items = _normalise_items(
        prediction_artifact["items"],
        field="prediction.items",
    )
    labels, oriented_effects = _ordered_outcomes(
        target_outcome_items,
        target_items,
        field="target_outcome_items",
    )
    items = [
        {
            **item,
            "target_label": int(label),
            "oriented_effect": float(effect),
        }
        for item, label, effect in zip(
            target_items,
            labels,
            oriented_effects,
            strict=True,
        )
    ]
    outcome_items_sha256 = _item_order_sha256(target_items)
    if outcome_items_sha256 != fit_artifact["target_dataset"]["outcome_items_sha256"]:
        raise ValueError("target outcome item identities differ from the fit precommitment")
    transfer_outcome_rows_sha256 = _outcome_rows_sha256(items)
    if transfer_outcome_rows_sha256 != fit_artifact["target_dataset"]["transfer_outcome_rows_sha256"]:
        raise ValueError("target outcome rows differ from the sealed transfer-row binding")
    comparisons = {}
    for control in CONTROL_CHANNELS:
        records = [
            {
                "item_id": item["item_id"],
                "group_id": item["group_id"],
                "target_label": item["target_label"],
                "selected_probability": float(probabilities["selected_layer"][index]),
                "reference_probability": float(probabilities[control][index]),
            }
            for index, item in enumerate(items)
        ]
        comparisons[control] = probe_common.frozen_decoder_signal_metrics(
            records,
            n_boot=n_boot,
            seed=seed,
            confidence_level=confidence_level,
        )
    effect_monotonicity = _effect_monotonicity_metrics(
        items,
        oriented_effects,
        probabilities,
        n_boot=n_boot,
        seed=seed,
        confidence_level=confidence_level,
    )
    artifact = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_type": TRANSFER_EVALUATION_ARTIFACT_TYPE,
        "fit_artifact_sha256": transfer_fit_artifact_sha256(fit_artifact),
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_artifact": dict(prediction_artifact),
        "freeze_receipt_sha256": freeze_receipt_sha256,
        "freeze_receipt_path": committed_receipt_path,
        "target_dataset": dict(fit_artifact["target_dataset"]),
        "outcome_items_sha256": outcome_items_sha256,
        "transfer_outcome_rows_sha256": transfer_outcome_rows_sha256,
        "items": items,
        "analysis_lock": analysis_lock,
        "comparisons": comparisons,
        "effect_monotonicity": effect_monotonicity,
        "adjudication": _pilot_adjudication(),
    }
    recompute_transfer_evaluation(artifact, fit_artifact)
    return artifact


def recompute_transfer_evaluation(
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate labels and recompute every selected-minus-control metric."""

    expected = {
        "schema_version",
        "artifact_type",
        "fit_artifact_sha256",
        "prediction_artifact_sha256",
        "prediction_artifact",
        "freeze_receipt_sha256",
        "freeze_receipt_path",
        "target_dataset",
        "outcome_items_sha256",
        "transfer_outcome_rows_sha256",
        "items",
        "analysis_lock",
        "comparisons",
        "effect_monotonicity",
        "adjudication",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected:
        raise ValueError("transfer evaluation artifact must use the exact schema")
    if (
        artifact.get("schema_version") != TRANSFER_SCHEMA_VERSION
        or artifact.get("artifact_type") != TRANSFER_EVALUATION_ARTIFACT_TYPE
    ):
        raise ValueError("unsupported transfer evaluation artifact")
    if artifact.get("fit_artifact_sha256") != (transfer_fit_artifact_sha256(fit_artifact)):
        raise ValueError("transfer evaluation fit-artifact checksum differs")
    prediction = artifact.get("prediction_artifact")
    probabilities = validate_transfer_prediction_artifact(
        prediction,
        fit_artifact,
    )
    if artifact.get("prediction_artifact_sha256") != (transfer_prediction_artifact_sha256(prediction, fit_artifact)):
        raise ValueError("transfer evaluation prediction checksum differs")
    receipt_path = artifact.get("freeze_receipt_path")
    validated_receipt = validate_transfer_prediction_freeze_receipt(
        receipt_path,
        prediction,
        fit_artifact,
    )
    if validated_receipt["committed_receipt_path"] != receipt_path:
        raise ValueError("transfer evaluation freeze-receipt path differs")
    _, receipt_bytes = _read_regular_file_no_follow(
        receipt_path,
        field="prediction freeze receipt",
    )
    if artifact.get("freeze_receipt_sha256") != hashlib.sha256(receipt_bytes).hexdigest():
        raise ValueError("transfer evaluation freeze-receipt checksum differs")
    if artifact.get("target_dataset") != fit_artifact["target_dataset"]:
        raise ValueError("transfer evaluation target dataset differs from the fit")
    target_items = _normalise_items(
        prediction["items"],
        field="prediction.items",
    )
    outcome_items = artifact.get("items")
    labels, oriented_effects = _ordered_outcomes(
        outcome_items,
        target_items,
        field="evaluation.items",
    )
    if [
        (
            _item_identity_tokens(item),
            int(item["target_label"]),
            float(item["oriented_effect"]),
        )
        for item in outcome_items
    ] != [
        (
            _item_identity_tokens(item),
            int(label),
            float(effect),
        )
        for item, label, effect in zip(
            target_items,
            labels,
            oriented_effects,
            strict=True,
        )
    ]:
        raise ValueError("transfer evaluation item order differs from prediction")
    expected_outcome_items_sha256 = _item_order_sha256(target_items)
    if (
        artifact.get("outcome_items_sha256") != expected_outcome_items_sha256
        or expected_outcome_items_sha256 != fit_artifact["target_dataset"]["outcome_items_sha256"]
    ):
        raise ValueError("transfer evaluation outcome item precommitment differs")
    expected_outcome_rows_sha256 = _outcome_rows_sha256(outcome_items)
    if (
        artifact.get("transfer_outcome_rows_sha256") != expected_outcome_rows_sha256
        or expected_outcome_rows_sha256 != fit_artifact["target_dataset"]["transfer_outcome_rows_sha256"]
    ):
        raise ValueError("transfer evaluation sealed outcome rows differ")
    lock = artifact.get("analysis_lock")
    if not isinstance(lock, Mapping) or set(lock) != {
        "n_boot",
        "seed",
        "confidence_level",
    }:
        raise ValueError("transfer evaluation analysis lock is invalid")
    normalised_lock = _normalise_analysis_lock(
        lock["n_boot"],
        lock["seed"],
        lock["confidence_level"],
    )
    if dict(lock) != normalised_lock:
        raise ValueError("transfer evaluation analysis lock is not canonical")
    n_boot = normalised_lock["n_boot"]
    seed = normalised_lock["seed"]
    confidence_level = normalised_lock["confidence_level"]
    comparisons = artifact.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(CONTROL_CHANNELS):
        raise ValueError("transfer evaluation comparisons are incomplete")
    recomputed = {}
    for control in CONTROL_CHANNELS:
        records = [
            {
                "item_id": item["item_id"],
                "group_id": item["group_id"],
                "target_label": int(labels[index]),
                "selected_probability": float(probabilities["selected_layer"][index]),
                "reference_probability": float(probabilities[control][index]),
            }
            for index, item in enumerate(target_items)
        ]
        recomputed[control] = probe_common.frozen_decoder_signal_metrics(
            records,
            n_boot=n_boot,
            seed=seed,
            confidence_level=confidence_level,
        )
    if dict(comparisons) != recomputed:
        raise ValueError("reported transfer evaluation metrics differ from frozen predictions")
    effect_monotonicity = _effect_monotonicity_metrics(
        target_items,
        oriented_effects,
        probabilities,
        n_boot=n_boot,
        seed=seed,
        confidence_level=confidence_level,
    )
    if artifact.get("effect_monotonicity") != effect_monotonicity:
        raise ValueError("reported effect monotonicity differs from frozen outcomes")
    if artifact.get("adjudication") != _pilot_adjudication():
        raise ValueError("schema-v1 transfer evaluation must remain NOT_ADJUDICATED")
    return recomputed


def transfer_evaluation_artifact_sha256(
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> str:
    recompute_transfer_evaluation(artifact, fit_artifact)
    return hashlib.sha256(_artifact_bytes(artifact)).hexdigest()


def _write_artifact(path: str | os.PathLike[str], artifact: Mapping[str, Any]) -> str:
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(_artifact_bytes(artifact))
    return os.fspath(path)


def write_source_selection_artifact(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    *,
    source_dataset: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    source_labeled_items: Sequence[Mapping[str, Any]],
    layer_channels: Mapping[int, Mapping[str, Any]],
) -> str:
    """Write a source-only selection artifact after full input recomputation."""

    validate_source_selection_artifact(
        artifact,
        source_dataset=source_dataset,
        source_items=source_items,
        source_labeled_items=source_labeled_items,
        layer_channels=layer_channels,
    )
    return _write_artifact(path, artifact)


def write_transfer_fit_artifact(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
) -> str:
    validate_transfer_fit_artifact(artifact)
    return _write_artifact(path, artifact)


def write_transfer_prediction_artifact(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> str:
    validate_transfer_prediction_artifact(artifact, fit_artifact)
    return _write_artifact(path, artifact)


def write_transfer_evaluation_artifact(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    fit_artifact: Mapping[str, Any],
) -> str:
    recompute_transfer_evaluation(artifact, fit_artifact)
    return _write_artifact(path, artifact)

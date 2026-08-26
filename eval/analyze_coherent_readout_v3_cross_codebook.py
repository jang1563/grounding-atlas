"""Analyze the staged v3 cross-codebook projected-content experiment.

This module never executes the model.  It reconstructs frozen runner artifacts,
aggregates at the symbolic-world dependency unit, and applies preregistered gates.
All scientific bootstrap comparisons reuse one set of world resamples.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import run_coherent_readout_v3_cross_codebook as runner
except ImportError:  # direct execution from eval/
    try:
        import run_coherent_readout_v3_cross_codebook as runner
    except ImportError:  # runner may be landing concurrently during preregistration
        runner = None  # type: ignore[assignment]


FIT_ANALYSIS_SCHEMA = "coherent-readout-v3-cross-codebook-fit-analysis-v1"
BASIS_LOCK_SCHEMA = "coherent-readout-v3-cross-codebook-basis-lock-v1"
LOCALIZATION_BASELINE_ANALYSIS_SCHEMA = "coherent-readout-v3-cross-codebook-localization-baseline-analysis-v1"
LOCALIZATION_ENTRY_SCHEMA = "coherent-readout-v3-cross-codebook-localization-entry-v1"
LOCALIZATION_ANALYSIS_SCHEMA = "coherent-readout-v3-cross-codebook-localization-analysis-v1"
LAYER_LOCK_SCHEMA = "coherent-readout-v3-cross-codebook-layer-lock-v1"
HOLDOUT_BASELINE_ANALYSIS_SCHEMA = "coherent-readout-v3-cross-codebook-holdout-baseline-analysis-v1"
HOLDOUT_ENTRY_SCHEMA = "coherent-readout-v3-cross-codebook-holdout-entry-v1"
FINAL_ANALYSIS_SCHEMA = "coherent-readout-v3-cross-codebook-final-analysis-v1"
ANALYSIS_MANIFEST_SCHEMA = "coherent-readout-v3-analysis-manifest-v1"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260804
ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "docs" / "COHERENT_READOUT_V3_CROSS_CODEBOOK_ANTICOPY_PREREG.md"
PREREGISTRATION_SHA256 = "98a31bd903744fa054e696ecb421c07da78557d8f8a51ab00fb471514776f949"
LAYER_GRID = (8, 12, 16, 20, 24)
X_TOKEN_ID = 55
Y_TOKEN_ID = 56
MODEL_VOCAB_SIZE = 151_936
PRIMARY_CONDITION = "content_anticopy"
SPECIFICITY_CONTROLS = (
    "content_same",
    "answer_anticopy",
    "codebook_anticopy",
    "distractor_anticopy",
    "query_anticopy",
    "order_anticopy",
    "full_anticopy",
)
SIMULTANEOUS_NULLS = tuple(f"null_{index}_anticopy" for index in range(4))
NATURAL_USE_CONDITIONS = (
    "content_erase",
    "content_rescue_same",
    "content_rescue_opposite",
    "null_0_erase",
    "null_1_erase",
    "null_2_erase",
    "null_3_erase",
)
ENGINEERING_CONDITIONS = ("identity", "full_text_counterfactual")

FINAL_STATUSES = (
    "NO_REPLICATED_PROJECTED_CONTENT_RECOMPOSITION",
    "NONSPECIFIC_PROJECTED_TRANSFER_REPLICATED",
    "CONTENT_RECOMPOSITION_SUPPORTED_NATURAL_USE_NOT_ESTABLISHED",
    "CONTENT_RECOMPOSITION_AND_PARTIAL_NATURAL_USE_SUPPORTED",
)
STOP_STATUSES = (
    "FIT_STOP_BASIS_LOCK_INVALID",
    "LOCALIZATION_BASELINE_STOP_NOT_ADMITTED",
    "LOCALIZATION_STOP_ENGINEERING_INVALID",
    "LOCALIZATION_STOP_NO_PREREGISTERED_LAYER",
    "HOLDOUT_BASELINE_STOP_NOT_ADMITTED",
    "FINAL_STOP_ENGINEERING_INVALID",
)

CLAIM_BOUNDARIES = {
    "supported_scope": ("synthetic_symbolic_projected_content_recomposition_in_one_locked_model"),
    "biology_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "activation_gap_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
    "natural_use_requires_separate_tier": True,
    "selected_token_patched_states": "independently_reconstructed_from_sidecars",
    "full_vocabulary_diagnostics": "hash_committed_runner_attestation",
    "runtime_hook_events": "runner_attestation_checked_against_persisted_states",
}

RESULT_ROOT = (
    runner.RESULT_ROOT
    if runner is not None
    else ROOT / "results" / "benchmark" / "single_cell" / "coherent_readout_v3_cross_codebook" / "qwen2.5-1.5b-instruct"
)
FIT_ANALYSIS_PATH = RESULT_ROOT / "fit_basis_analysis.json"
LOCALIZATION_BASELINE_ANALYSIS_PATH = RESULT_ROOT / "localization_baseline_analysis.json"
LOCALIZATION_ENTRY_PATH = (
    getattr(runner, "DEFAULT_LOCALIZATION_ENTRY", RESULT_ROOT / "localization_entry.json")
    if runner is not None
    else RESULT_ROOT / "localization_entry.json"
)
LOCALIZATION_ANALYSIS_PATH = RESULT_ROOT / "localization_analysis.json"
HOLDOUT_BASELINE_ANALYSIS_PATH = RESULT_ROOT / "holdout_baseline_analysis.json"
FINAL_ANALYSIS_PATH = RESULT_ROOT / "analysis.json"
FINAL_MARKDOWN_PATH = RESULT_ROOT / "analysis.md"
FINAL_MANIFEST_PATH = RESULT_ROOT / "analysis_manifest.json"


class CrossCodebookAnalysisError(ValueError):
    """Raised when an input artifact or preregistered analysis contract fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fit_only_intercept(activations: np.ndarray) -> np.ndarray:
    array = np.asarray(activations)
    if array.shape != (512, 5, 1536) or not np.isfinite(array).all():
        raise CrossCodebookAnalysisError("fit activations must be finite with shape (512,5,1536)")
    return np.mean(array.astype(np.float64), axis=0).astype("<f4")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrossCodebookAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CrossCodebookAnalysisError(f"{label} must be finite")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossCodebookAnalysisError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise CrossCodebookAnalysisError("JSON artifact must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise CrossCodebookAnalysisError(f"cannot read JSONL artifact: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise CrossCodebookAnalysisError("JSONL records must be objects")
    return values


def _validate_sha256(value: Any, label: str, *, allow_to_freeze: bool = False) -> str:
    if allow_to_freeze and value == "TO_FREEZE":
        return str(value)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CrossCodebookAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def target_oriented_margin(*, native_answer: str, x_logit: Any, y_logit: Any) -> float:
    """Return exactly z_counterfactual - z_native for an X/Y recipient."""

    x = _finite(x_logit, "x_logit")
    y = _finite(y_logit, "y_logit")
    if native_answer == "X":
        return y - x
    if native_answer == "Y":
        return x - y
    raise CrossCodebookAnalysisError("native answer must be X or Y")


def _validate_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "x_logit",
        "y_logit",
        "x_minus_y_margin",
        "full_vocab_logsumexp",
        "label_probability_mass",
        "greedy_token_id",
        "greedy_logit",
        "maximum_token_ids",
        "maximum_tie_count",
        "full_vocab_logits_sha256",
    }
    if set(value) != expected:
        raise CrossCodebookAnalysisError("diagnostic schema changed")
    x = _finite(value["x_logit"], "x_logit")
    y = _finite(value["y_logit"], "y_logit")
    margin = _finite(value["x_minus_y_margin"], "x_minus_y_margin")
    logsumexp = _finite(value["full_vocab_logsumexp"], "full_vocab_logsumexp")
    mass = _finite(value["label_probability_mass"], "label_probability_mass")
    greedy_logit = _finite(value["greedy_logit"], "greedy_logit")
    if abs(margin - (x - y)) > 1e-7:
        raise CrossCodebookAnalysisError("stored X-minus-Y margin is inconsistent")
    if not 0.0 <= mass <= 1.0:
        raise CrossCodebookAnalysisError("label probability mass is outside [0,1]")
    expected_mass = math.exp(float(np.logaddexp(x, y)) - logsumexp)
    if abs(mass - expected_mass) > 1e-12:
        raise CrossCodebookAnalysisError("label probability mass is inconsistent")
    maximum_ids = value["maximum_token_ids"]
    tie_count = value["maximum_tie_count"]
    greedy_token_id = value["greedy_token_id"]
    if (
        not isinstance(maximum_ids, list)
        or not maximum_ids
        or any(isinstance(token, bool) or not isinstance(token, int) for token in maximum_ids)
        or any(token < 0 or token >= MODEL_VOCAB_SIZE for token in maximum_ids)
        or maximum_ids != sorted(set(maximum_ids))
        or isinstance(tie_count, bool)
        or not isinstance(tie_count, int)
        or tie_count != len(maximum_ids)
        or isinstance(greedy_token_id, bool)
        or not isinstance(greedy_token_id, int)
        or greedy_token_id != maximum_ids[0]
    ):
        raise CrossCodebookAnalysisError("global-maximum diagnostics are inconsistent")
    if logsumexp < greedy_logit or greedy_logit < max(x, y):
        raise CrossCodebookAnalysisError("global maximum or log-sum-exp is inconsistent")
    if (X_TOKEN_ID in maximum_ids) != (abs(greedy_logit - x) <= 1e-7):
        raise CrossCodebookAnalysisError("maximal X membership is inconsistent")
    if (Y_TOKEN_ID in maximum_ids) != (abs(greedy_logit - y) <= 1e-7):
        raise CrossCodebookAnalysisError("maximal Y membership is inconsistent")
    _validate_sha256(value["full_vocab_logits_sha256"], "full-vocabulary digest")
    return dict(value)


def _diagnostic_equivalence(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 1e-4,
    require_global_max_preserved: bool,
) -> dict[str, Any]:
    left = _validate_diagnostics(first)
    right = _validate_diagnostics(second)
    x_difference = abs(left["x_logit"] - right["x_logit"])
    y_difference = abs(left["y_logit"] - right["y_logit"])
    global_max_preserved = bool(
        left["maximum_token_ids"] == right["maximum_token_ids"]
        and left["maximum_tie_count"] == right["maximum_tie_count"]
    )
    passed = bool(
        x_difference <= atol and y_difference <= atol and (global_max_preserved or not require_global_max_preserved)
    )
    return {
        "atol": atol,
        "absolute_x_logit_difference": x_difference,
        "absolute_y_logit_difference": y_difference,
        "global_max_preserved": global_max_preserved,
        "global_max_preservation_required": require_global_max_preserved,
        "pass": passed,
    }


def _strict_final_diagnostic_engineering(
    diagnostics_by_condition: Mapping[str, Mapping[str, Any]],
    *,
    trace_projector_and_sidecar_pass: bool,
) -> dict[str, Any]:
    expected = {
        "recipient_baseline",
        "identity",
        PRIMARY_CONDITION,
        "content_same",
        "content_rescue_same",
        "content_rescue_opposite",
    }
    if set(diagnostics_by_condition) != expected:
        raise CrossCodebookAnalysisError("strict engineering diagnostic set changed")
    if not isinstance(trace_projector_and_sidecar_pass, bool):
        raise CrossCodebookAnalysisError("trace/projector gate must be Boolean")
    identity = _diagnostic_equivalence(
        diagnostics_by_condition["recipient_baseline"],
        diagnostics_by_condition["identity"],
        require_global_max_preserved=True,
    )
    rescue_same = _diagnostic_equivalence(
        diagnostics_by_condition["content_same"],
        diagnostics_by_condition["content_rescue_same"],
        require_global_max_preserved=False,
    )
    rescue_opposite = _diagnostic_equivalence(
        diagnostics_by_condition[PRIMARY_CONDITION],
        diagnostics_by_condition["content_rescue_opposite"],
        require_global_max_preserved=False,
    )
    return {
        "identity": identity,
        "rescue_same_equals_content_same": rescue_same,
        "rescue_opposite_equals_content_anticopy": rescue_opposite,
        "trace_projector_and_sidecar_pass": trace_projector_and_sidecar_pass,
        "pass": bool(
            identity["pass"] and rescue_same["pass"] and rescue_opposite["pass"] and trace_projector_and_sidecar_pass
        ),
    }


def _engineering_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "identity_absolute_x_logit_error",
        "identity_absolute_y_logit_error",
        "identity_global_max_preserved",
        "rescue_same_absolute_x_logit_error",
        "rescue_same_absolute_y_logit_error",
        "rescue_opposite_absolute_x_logit_error",
        "rescue_opposite_absolute_y_logit_error",
        "trace_projector_and_sidecar_pass",
        "source_reference_pass",
        "patched_activation_hash_pass",
        "fit_only_center_pass",
    }
    locked = dict(payload)
    if set(locked) != expected:
        raise CrossCodebookAnalysisError("engineering receipt payload changed")
    numeric_fields = {field for field in expected if field.endswith("_logit_error")}
    for field in numeric_fields:
        locked[field] = _finite(locked[field], field)
        if locked[field] < 0.0:
            raise CrossCodebookAnalysisError("engineering error must be nonnegative")
    for field in expected - numeric_fields:
        if not isinstance(locked[field], bool):
            raise CrossCodebookAnalysisError("engineering receipt pass fields must be Boolean")
    passed = bool(
        all(locked[field] <= 1e-4 for field in numeric_fields)
        and all(locked[field] for field in expected - numeric_fields)
    )
    return {
        "schema_version": "coherent-readout-v3-engineering-receipt-v1",
        "payload": locked,
        "payload_canonical_sha256": canonical_sha256(locked),
        "pass": passed,
    }


def _validate_engineering_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "payload",
        "payload_canonical_sha256",
        "pass",
    }:
        raise CrossCodebookAnalysisError("engineering receipt schema changed")
    if value["schema_version"] != "coherent-readout-v3-engineering-receipt-v1":
        raise CrossCodebookAnalysisError("engineering receipt version changed")
    if not isinstance(value["payload"], Mapping):
        raise CrossCodebookAnalysisError("engineering receipt payload is invalid")
    expected = _engineering_receipt(value["payload"])
    if dict(value) != expected:
        raise CrossCodebookAnalysisError("engineering receipt does not reconstruct")
    return dict(value)


def _world_means(cell_values: Mapping[str, float], cell_to_world: Mapping[str, str]) -> dict[str, float]:
    if set(cell_values) != set(cell_to_world):
        raise CrossCodebookAnalysisError("cell values and world mapping differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for cell_id, value in cell_values.items():
        grouped[cell_to_world[cell_id]].append(_finite(value, f"value[{cell_id}]"))
    if not grouped:
        raise CrossCodebookAnalysisError("world aggregation is empty")
    return {world_id: float(np.mean(values)) for world_id, values in sorted(grouped.items())}


def _behavioral_admission(
    rows: Sequence[Mapping[str, Any]],
    recipient_pairs: Sequence[Mapping[str, Any]],
    *,
    expected_worlds: int,
) -> dict[str, Any]:
    expected_row_keys = {
        "cell_id",
        "world_id",
        "queried_property",
        "codebook_id",
        "native_answer",
        "diagnostics",
    }
    observed = []
    for raw in rows:
        row = dict(raw)
        if set(row) != expected_row_keys:
            raise CrossCodebookAnalysisError("behavioral baseline schema changed")
        if row["native_answer"] not in {"X", "Y"}:
            raise CrossCodebookAnalysisError("baseline native answer is invalid")
        if row["queried_property"] not in {"P", "Q"} or row["codebook_id"] not in {"identity", "swapped"}:
            raise CrossCodebookAnalysisError("baseline factor level is invalid")
        expected_native = "X" if (row["queried_property"] == "P") == (row["codebook_id"] == "identity") else "Y"
        if row["native_answer"] != expected_native:
            raise CrossCodebookAnalysisError("native answer violates p-by-m codebook")
        row["diagnostics"] = _validate_diagnostics(row["diagnostics"])
        observed.append(row)
    expected_rows = expected_worlds * 32
    if len(observed) != expected_rows:
        raise CrossCodebookAnalysisError("behavioral baseline row count changed")
    by_cell = {row["cell_id"]: row for row in observed}
    if len(by_cell) != expected_rows:
        raise CrossCodebookAnalysisError("behavioral cell IDs are duplicated")
    world_counts = Counter(row["world_id"] for row in observed)
    if len(world_counts) != expected_worlds or set(world_counts.values()) != {32}:
        raise CrossCodebookAnalysisError("behavioral world coverage changed")

    def expected_token(row: Mapping[str, Any]) -> int:
        return X_TOKEN_ID if row["native_answer"] == "X" else Y_TOKEN_ID

    correct = {row["cell_id"]: row["diagnostics"]["maximum_token_ids"] == [expected_token(row)] for row in observed}
    accuracy_by_property_codebook = {}
    for queried_property in ("P", "Q"):
        for codebook in ("identity", "swapped"):
            members = [
                row
                for row in observed
                if row["queried_property"] == queried_property and row["codebook_id"] == codebook
            ]
            if len(members) != expected_worlds * 8:
                raise CrossCodebookAnalysisError("p-by-m behavioral cell count changed")
            key = f"{queried_property}__{codebook}"
            accuracy_by_property_codebook[key] = float(np.mean([correct[row["cell_id"]] for row in members]))

    expected_pair_keys = {
        "recipient_cell_id",
        "counterfactual_cell_id",
        "world_id",
        "queried_property",
        "codebook_id",
    }
    gap_cells = {}
    pair_cell_to_world = {}
    counterfactual_correct: dict[str, list[bool]] = defaultdict(list)
    for raw_pair in recipient_pairs:
        pair = dict(raw_pair)
        if set(pair) != expected_pair_keys:
            raise CrossCodebookAnalysisError("behavioral recipient-pair schema changed")
        recipient = by_cell.get(pair["recipient_cell_id"])
        counterfactual = by_cell.get(pair["counterfactual_cell_id"])
        if recipient is None or counterfactual is None:
            raise CrossCodebookAnalysisError("behavioral recipient source does not resolve")
        if (
            recipient["world_id"] != pair["world_id"]
            or counterfactual["world_id"] != pair["world_id"]
            or recipient["queried_property"] != pair["queried_property"]
            or recipient["codebook_id"] != pair["codebook_id"]
            or counterfactual["queried_property"] != ("Q" if recipient["queried_property"] == "P" else "P")
            or counterfactual["codebook_id"] != recipient["codebook_id"]
            or counterfactual["native_answer"] == recipient["native_answer"]
        ):
            raise CrossCodebookAnalysisError("behavioral recipient source identity changed")
        recipient_margin = target_oriented_margin(
            native_answer=recipient["native_answer"],
            x_logit=recipient["diagnostics"]["x_logit"],
            y_logit=recipient["diagnostics"]["y_logit"],
        )
        counterfactual_margin = target_oriented_margin(
            native_answer=recipient["native_answer"],
            x_logit=counterfactual["diagnostics"]["x_logit"],
            y_logit=counterfactual["diagnostics"]["y_logit"],
        )
        recipient_cell_id = pair["recipient_cell_id"]
        gap_cells[recipient_cell_id] = counterfactual_margin - recipient_margin
        pair_cell_to_world[recipient_cell_id] = pair["world_id"]
        stratum = _stratum_id(pair)
        counterfactual_correct[stratum].append(
            counterfactual["diagnostics"]["maximum_token_ids"] == [expected_token(counterfactual)]
        )
    if len(recipient_pairs) != expected_worlds * 8 or len(gap_cells) != len(recipient_pairs):
        raise CrossCodebookAnalysisError("behavioral recipient coverage changed")
    expected_counterfactual_strata = {
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    }
    if set(counterfactual_correct) != expected_counterfactual_strata or any(
        len(values) != expected_worlds * 2 for values in counterfactual_correct.values()
    ):
        raise CrossCodebookAnalysisError("counterfactual stratum coverage changed")
    gap_worlds = _world_means(gap_cells, pair_cell_to_world)
    bootstrap_indices = _common_bootstrap_indices(expected_worlds)
    gap = _effect_summary(gap_worlds, gap_worlds, bootstrap_indices=bootstrap_indices)
    overall_accuracy = float(np.mean(list(correct.values())))
    counterfactual_accuracy = {
        stratum: float(np.mean(values)) for stratum, values in sorted(counterfactual_correct.items())
    }
    no_ties = all(row["diagnostics"]["maximum_tie_count"] == 1 for row in observed)
    unique_xy_rate = float(
        np.mean([row["diagnostics"]["maximum_token_ids"] in ([X_TOKEN_ID], [Y_TOKEN_ID]) for row in observed])
    )
    mean_label_mass = float(np.mean([row["diagnostics"]["label_probability_mass"] for row in observed]))
    gates = {
        "native_accuracy_overall": overall_accuracy >= 0.95,
        "native_accuracy_each_p_by_m": all(value >= 0.90 for value in accuracy_by_property_codebook.values()),
        "counterfactual_accuracy_each_transition_by_codebook": all(
            value >= 0.95 for value in counterfactual_accuracy.values()
        ),
        "no_global_argmax_ties": no_ties,
        "unique_global_argmax_in_xy": unique_xy_rate >= 0.95,
        "mean_xy_probability_mass": mean_label_mass >= 0.95,
        "G_mean_positive": gap["mean"] > 0.0,
        "G_bootstrap_lower_positive": gap["bootstrap_95"]["lower_95"] > 0.0,
        "all_leave_one_world_out_G_positive": all(value > 0.0 for value in gap["leave_one_world_out_means"]),
    }
    return {
        "n_cells": len(observed),
        "n_worlds": expected_worlds,
        "native_accuracy": overall_accuracy,
        "native_accuracy_by_queried_property_and_codebook": (accuracy_by_property_codebook),
        "counterfactual_accuracy_by_transition_and_codebook": (counterfactual_accuracy),
        "no_global_argmax_ties": no_ties,
        "unique_global_argmax_in_xy_rate": unique_xy_rate,
        "mean_xy_probability_mass": mean_label_mass,
        "G": gap,
        "gates": {**gates, "pass": all(gates.values())},
    }


def _common_bootstrap_indices(
    n_worlds: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if n_worlds <= 0 or draws <= 0:
        raise CrossCodebookAnalysisError("bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(0, n_worlds, size=(draws, n_worlds))


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise CrossCodebookAnalysisError("bootstrap distribution is invalid")
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _effect_summary(
    world_values: Mapping[str, float],
    gap_world_values: Mapping[str, float],
    *,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    if list(world_values) != list(gap_world_values):
        raise CrossCodebookAnalysisError("effect and G world order differs")
    worlds = list(world_values)
    values = np.asarray([world_values[world] for world in worlds], dtype=float)
    gaps = np.asarray([gap_world_values[world] for world in worlds], dtype=float)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(gaps).all()
        or bootstrap_indices.ndim != 2
        or bootstrap_indices.shape[1] != len(values)
    ):
        raise CrossCodebookAnalysisError("effect summary arrays are invalid")
    leave_one_out = np.asarray(
        [np.delete(values, index).mean() for index in range(len(values))],
        dtype=float,
    )
    leave_one_out_g = np.asarray(
        [np.delete(gaps, index).mean() for index in range(len(gaps))],
        dtype=float,
    )
    mean = float(values.mean())
    gap_mean = float(gaps.mean())
    ratio_defined = bool(gap_mean > 0.0 and np.all(leave_one_out_g > 0.0))
    bootstrap_values = values[bootstrap_indices].mean(axis=1)
    bootstrap_gaps = gaps[bootstrap_indices].mean(axis=1)
    bootstrap_ratio_valid = bool(ratio_defined and np.isfinite(bootstrap_gaps).all() and np.all(bootstrap_gaps != 0.0))
    bootstrap_ratio = bootstrap_values / bootstrap_gaps if bootstrap_ratio_valid else None
    return {
        "world_values": dict(world_values),
        "gap_world_values": dict(gap_world_values),
        "n_worlds": len(worlds),
        "mean": mean,
        "median": float(np.median(values)),
        "gap_mean": gap_mean,
        "positive_world_count": int(np.sum(values > 0.0)),
        "positive_world_fraction": float(np.mean(values > 0.0)),
        "leave_one_world_out_means": leave_one_out.tolist(),
        "leave_one_world_out_gap_means": leave_one_out_g.tolist(),
        "ratio_defined": ratio_defined,
        "mean_over_G": float(mean / gap_mean) if ratio_defined else None,
        "leave_one_world_out_ratios": ((leave_one_out / leave_one_out_g).tolist() if ratio_defined else None),
        "bootstrap_95": _percentile_interval(bootstrap_values),
        "bootstrap_G_95": _percentile_interval(bootstrap_gaps),
        "bootstrap_ratio_valid": bootstrap_ratio_valid,
        "bootstrap_ratio_95": (_percentile_interval(bootstrap_ratio) if bootstrap_ratio is not None else None),
    }


def _rate_summary(world_rates: Mapping[str, float], *, bootstrap_indices: np.ndarray) -> dict[str, Any]:
    rates = np.asarray(list(world_rates.values()), dtype=float)
    if (
        not len(rates)
        or not np.isfinite(rates).all()
        or np.any((rates < 0.0) | (rates > 1.0))
        or bootstrap_indices.shape[1] != len(rates)
    ):
        raise CrossCodebookAnalysisError("world rate input is invalid")
    samples = rates[bootstrap_indices].mean(axis=1)
    leave_one_out = [float(np.delete(rates, index).mean()) for index in range(len(rates))]
    return {
        "world_values": dict(world_rates),
        "n_worlds": len(rates),
        "point": float(rates.mean()),
        "median_world_rate": float(np.median(rates)),
        "leave_one_world_out_rates": leave_one_out,
        "cluster_bootstrap_95": _percentile_interval(samples),
    }


def _normalized_final_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_row_keys = {
        "cell_id",
        "world_id",
        "queried_property",
        "codebook_id",
        "recipient_margin",
        "counterfactual_margin",
        "recipient_label_probability_mass",
        "primary_label_probability_mass",
        "primary_global_argmax_is_counterfactual",
        "primary_unique_global_argmax_in_xy",
        "engineering_receipt",
        "condition_margins",
    }
    required_conditions = {
        PRIMARY_CONDITION,
        *SPECIFICITY_CONTROLS,
        *SIMULTANEOUS_NULLS,
        *NATURAL_USE_CONDITIONS,
        *ENGINEERING_CONDITIONS,
    }
    observed: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if set(row) != expected_row_keys:
            raise CrossCodebookAnalysisError("normalized final-row schema changed")
        for key in ("cell_id", "world_id"):
            if not isinstance(row[key], str) or not row[key]:
                raise CrossCodebookAnalysisError(f"{key} is invalid")
        if row["queried_property"] not in {"P", "Q"}:
            raise CrossCodebookAnalysisError("queried property is invalid")
        if row["codebook_id"] not in {"identity", "swapped"}:
            raise CrossCodebookAnalysisError("codebook is invalid")
        row["recipient_margin"] = _finite(row["recipient_margin"], "recipient_margin")
        row["counterfactual_margin"] = _finite(row["counterfactual_margin"], "counterfactual_margin")
        for mass_field in (
            "recipient_label_probability_mass",
            "primary_label_probability_mass",
        ):
            row[mass_field] = _finite(row[mass_field], mass_field)
            if not 0.0 <= row[mass_field] <= 1.0:
                raise CrossCodebookAnalysisError(f"{mass_field} is outside [0,1]")
        for bool_field in (
            "primary_global_argmax_is_counterfactual",
            "primary_unique_global_argmax_in_xy",
        ):
            if not isinstance(row[bool_field], bool):
                raise CrossCodebookAnalysisError(f"{bool_field} must be Boolean")
        if row["primary_global_argmax_is_counterfactual"] and not row["primary_unique_global_argmax_in_xy"]:
            raise CrossCodebookAnalysisError("a counterfactual global argmax must be a unique X/Y argmax")
        margins = row["condition_margins"]
        if not isinstance(margins, Mapping) or set(margins) != required_conditions:
            raise CrossCodebookAnalysisError("final intervention conditions changed")
        row["condition_margins"] = {
            condition: _finite(value, f"condition[{condition}]") for condition, value in margins.items()
        }
        primary_margin = row["condition_margins"][PRIMARY_CONDITION]
        if row["primary_unique_global_argmax_in_xy"] and primary_margin == 0.0:
            raise CrossCodebookAnalysisError("a unique X/Y global maximum cannot have a zero X/Y margin")
        if row["primary_unique_global_argmax_in_xy"] and (
            row["primary_global_argmax_is_counterfactual"] != (primary_margin > 0.0)
        ):
            raise CrossCodebookAnalysisError("primary global-argmax claim disagrees with its X/Y margin")
        receipt = row["engineering_receipt"]
        if not isinstance(receipt, Mapping):
            raise CrossCodebookAnalysisError("engineering receipt is invalid")
        row["engineering_receipt"] = _validate_engineering_receipt(receipt)
        observed.append(row)
    if not observed or len({row["cell_id"] for row in observed}) != len(observed):
        raise CrossCodebookAnalysisError("final cell IDs are empty or duplicated")
    return sorted(observed, key=lambda row: row["cell_id"])


def _stratum_id(row: Mapping[str, Any]) -> str:
    transition = "P_to_Q" if row["queried_property"] == "P" else "Q_to_P"
    return f"{transition}__{row['codebook_id']}"


def _final_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = _normalized_final_rows(rows)
    cell_to_world = {row["cell_id"]: row["world_id"] for row in normalized}
    worlds = sorted(set(cell_to_world.values()))
    world_cell_counts = Counter(cell_to_world.values())
    if len(worlds) != 32 or set(world_cell_counts.values()) != {8}:
        raise CrossCodebookAnalysisError("final analysis requires 32 holdout worlds and eight recipients per world")
    bootstrap_indices = _common_bootstrap_indices(len(worlds))
    recipient = {row["cell_id"]: row["recipient_margin"] for row in normalized}
    gap_cells = {row["cell_id"]: row["counterfactual_margin"] - row["recipient_margin"] for row in normalized}
    gap_worlds = _world_means(gap_cells, cell_to_world)
    primary_cells = {
        row["cell_id"]: row["condition_margins"][PRIMARY_CONDITION] - row["recipient_margin"] for row in normalized
    }
    primary_worlds = _world_means(primary_cells, cell_to_world)
    primary = _effect_summary(primary_worlds, gap_worlds, bootstrap_indices=bootstrap_indices)

    stratum_primary: dict[str, Any] = {}
    for stratum in (
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    ):
        members = [row for row in normalized if _stratum_id(row) == stratum]
        mapping = {row["cell_id"]: row["world_id"] for row in members}
        if Counter(mapping.values()) != {world: 2 for world in worlds}:
            raise CrossCodebookAnalysisError("each final stratum must contribute two recipients per world")
        stratum_effect = {row["cell_id"]: primary_cells[row["cell_id"]] for row in members}
        stratum_gap = {row["cell_id"]: gap_cells[row["cell_id"]] for row in members}
        effect_worlds = _world_means(stratum_effect, mapping)
        gap_by_world = _world_means(stratum_gap, mapping)
        if list(effect_worlds) != worlds:
            raise CrossCodebookAnalysisError("a final stratum omits a holdout world")
        stratum_primary[stratum] = _effect_summary(
            effect_worlds,
            gap_by_world,
            bootstrap_indices=bootstrap_indices,
        )

    specificity = {}
    for condition in SPECIFICITY_CONTROLS:
        difference_cells = {
            row["cell_id"]: row["condition_margins"][PRIMARY_CONDITION] - row["condition_margins"][condition]
            for row in normalized
        }
        specificity[condition] = _effect_summary(
            _world_means(difference_cells, cell_to_world),
            gap_worlds,
            bootstrap_indices=bootstrap_indices,
        )
    simultaneous_cells = {
        row["cell_id"]: row["condition_margins"][PRIMARY_CONDITION]
        - max(row["condition_margins"][name] for name in SIMULTANEOUS_NULLS)
        for row in normalized
    }
    simultaneous = _effect_summary(
        _world_means(simultaneous_cells, cell_to_world),
        gap_worlds,
        bootstrap_indices=bootstrap_indices,
    )
    flip_cells = {row["cell_id"]: float(row["primary_global_argmax_is_counterfactual"]) for row in normalized}
    target_flip = _rate_summary(
        _world_means(flip_cells, cell_to_world),
        bootstrap_indices=bootstrap_indices,
    )
    within_xy_flip_cells = {
        row["cell_id"]: float(row["condition_margins"][PRIMARY_CONDITION] > 0.0) for row in normalized
    }
    within_xy_flip = _rate_summary(
        _world_means(within_xy_flip_cells, cell_to_world),
        bootstrap_indices=bootstrap_indices,
    )
    unique_xy_cells = {row["cell_id"]: float(row["primary_unique_global_argmax_in_xy"]) for row in normalized}
    unique_xy = _rate_summary(
        _world_means(unique_xy_cells, cell_to_world),
        bootstrap_indices=bootstrap_indices,
    )
    label_mass_loss_cells = {
        row["cell_id"]: row["recipient_label_probability_mass"] - row["primary_label_probability_mass"]
        for row in normalized
    }
    label_mass_loss_worlds = _world_means(label_mass_loss_cells, cell_to_world)
    label_mass_loss_values = np.asarray(list(label_mass_loss_worlds.values()), dtype=float)
    label_mass_loss_bootstrap = label_mass_loss_values[bootstrap_indices].mean(axis=1)
    label_mass_loss = {
        "world_values": label_mass_loss_worlds,
        "n_worlds": len(label_mass_loss_worlds),
        "point": float(label_mass_loss_values.mean()),
        "median_world_value": float(np.median(label_mass_loss_values)),
        "positive_world_count": int(np.sum(label_mass_loss_values > 0.0)),
        "leave_one_world_out_values": [
            float(np.delete(label_mass_loss_values, index).mean()) for index in range(len(label_mass_loss_values))
        ],
        "cluster_bootstrap_95": _percentile_interval(label_mass_loss_bootstrap),
    }
    identity_errors = {
        row["cell_id"]: abs(row["condition_margins"]["identity"] - row["recipient_margin"]) for row in normalized
    }
    rescue_same_equivalence_errors = {
        row["cell_id"]: abs(row["condition_margins"]["content_rescue_same"] - row["condition_margins"]["content_same"])
        for row in normalized
    }
    rescue_opposite_equivalence_errors = {
        row["cell_id"]: abs(
            row["condition_margins"]["content_rescue_opposite"] - row["condition_margins"][PRIMARY_CONDITION]
        )
        for row in normalized
    }
    strict_artifact_engineering_pass = all(row["engineering_receipt"]["pass"] for row in normalized)
    engineering_receipt_sha256 = canonical_sha256(
        [
            {
                "cell_id": row["cell_id"],
                "receipt": row["engineering_receipt"],
            }
            for row in normalized
        ]
    )

    erasure_damage_cells = {
        row["cell_id"]: row["condition_margins"]["content_erase"] - row["recipient_margin"] for row in normalized
    }
    rescue_cells = {
        row["cell_id"]: row["condition_margins"]["content_erase"] - row["condition_margins"]["content_rescue_same"]
        for row in normalized
    }
    alternative_rescue_cells = {
        row["cell_id"]: row["condition_margins"]["content_erase"] - row["condition_margins"]["content_rescue_opposite"]
        for row in normalized
    }
    erasure_damage_worlds = _world_means(erasure_damage_cells, cell_to_world)
    rescue_worlds = _world_means(rescue_cells, cell_to_world)
    erasure_damage = _effect_summary(
        erasure_damage_worlds,
        gap_worlds,
        bootstrap_indices=bootstrap_indices,
    )
    rescue = _effect_summary(
        rescue_worlds,
        gap_worlds,
        bootstrap_indices=bootstrap_indices,
    )
    rescue_over_damage = _effect_summary(
        rescue_worlds,
        erasure_damage_worlds,
        bootstrap_indices=bootstrap_indices,
    )
    alternative_rescue_superiority_cells = {
        cell_id: rescue_cells[cell_id] - alternative_rescue_cells[cell_id] for cell_id in rescue_cells
    }
    alternative_rescue_superiority = _effect_summary(
        _world_means(alternative_rescue_superiority_cells, cell_to_world),
        gap_worlds,
        bootstrap_indices=bootstrap_indices,
    )
    null_erasure_superiority = {}
    for index in range(4):
        condition = f"null_{index}_erase"
        null_damage = {
            row["cell_id"]: row["condition_margins"][condition] - recipient[row["cell_id"]] for row in normalized
        }
        difference = {cell_id: erasure_damage_cells[cell_id] - null_damage[cell_id] for cell_id in erasure_damage_cells}
        null_erasure_superiority[condition] = _effect_summary(
            _world_means(difference, cell_to_world),
            gap_worlds,
            bootstrap_indices=bootstrap_indices,
        )
    full_text_counterfactual_cells = {
        row["cell_id"]: row["condition_margins"]["full_text_counterfactual"] - row["recipient_margin"]
        for row in normalized
    }

    return {
        "bootstrap": {
            "dependency_unit": "symbolic_world",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "common_draws_for_all_comparisons": True,
            "index_matrix_sha256": f32_sha256(bootstrap_indices.astype("<f4")),
        },
        "n_cells": len(normalized),
        "n_worlds": len(worlds),
        "G": _effect_summary(gap_worlds, gap_worlds, bootstrap_indices=bootstrap_indices),
        "primary": primary,
        "primary_strata": stratum_primary,
        "primary_channel": {
            "counterfactual_global_argmax_rate": target_flip,
            "within_xy_counterfactual_margin_positive_rate_descriptive": (within_xy_flip),
            "unique_global_argmax_in_xy_rate": unique_xy,
            "label_probability_mass_loss": label_mass_loss,
        },
        "identity_engineering": {
            "tolerance": 1e-4,
            "maximum_absolute_target_margin_error": max(identity_errors.values()),
            "rescue_same_equals_content_same_maximum_margin_error": max(rescue_same_equivalence_errors.values()),
            "rescue_opposite_equals_content_anticopy_maximum_margin_error": max(
                rescue_opposite_equivalence_errors.values()
            ),
            "duplicate_rescue_executions_are_not_independent_evidence": True,
            "strict_raw_xy_global_max_trace_and_projector_validation_pass": (strict_artifact_engineering_pass),
            "engineering_receipt_registry_canonical_sha256": (engineering_receipt_sha256),
        },
        "full_text_counterfactual_positive_control": _effect_summary(
            _world_means(full_text_counterfactual_cells, cell_to_world),
            gap_worlds,
            bootstrap_indices=bootstrap_indices,
        ),
        "specificity": specificity,
        "simultaneous_max_null": simultaneous,
        "natural_use": {
            "c_erasure_damage": erasure_damage,
            "same_content_opposite_answer_rescue": rescue,
            "restoration_fraction": rescue_over_damage,
            "superiority_to_opposite_content_same_answer_rescue": (alternative_rescue_superiority),
            "superiority_to_null_erasures": null_erasure_superiority,
        },
    }


def _effect_gate(summary: Mapping[str, Any], *, minimum_ratio: float) -> bool:
    interval = summary.get("bootstrap_95")
    return bool(
        summary.get("ratio_defined") is True
        and summary.get("mean_over_G") is not None
        and summary["mean_over_G"] >= minimum_ratio
        and isinstance(interval, Mapping)
        and interval.get("lower_95") is not None
        and interval["lower_95"] > 0.0
    )


def _final_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected_strata = {
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    }
    if set(metrics.get("primary_strata", {})) != expected_strata:
        raise CrossCodebookAnalysisError("final primary stratum registry changed")
    if set(metrics.get("specificity", {})) != set(SPECIFICITY_CONTROLS):
        raise CrossCodebookAnalysisError("final specificity registry changed")
    natural_registry = metrics.get("natural_use", {})
    if not isinstance(natural_registry, Mapping) or set(natural_registry.get("superiority_to_null_erasures", {})) != {
        f"null_{index}_erase" for index in range(4)
    }:
        raise CrossCodebookAnalysisError("final null-erasure registry changed")
    primary = metrics["primary"]
    primary_ratio_interval = primary.get("bootstrap_ratio_95")
    channel = metrics["primary_channel"]
    channel_pass = bool(
        channel["unique_global_argmax_in_xy_rate"]["point"] >= 0.95
        and channel["label_probability_mass_loss"]["point"] <= 0.02
    )
    identity_pass = bool(
        metrics["identity_engineering"]["maximum_absolute_target_margin_error"]
        <= metrics["identity_engineering"]["tolerance"]
    )
    rescue_equivalence_pass = bool(
        metrics["identity_engineering"]["rescue_same_equals_content_same_maximum_margin_error"]
        <= metrics["identity_engineering"]["tolerance"]
        and metrics["identity_engineering"]["rescue_opposite_equals_content_anticopy_maximum_margin_error"]
        <= metrics["identity_engineering"]["tolerance"]
    )
    strict_artifact_pass = bool(
        metrics["identity_engineering"]["strict_raw_xy_global_max_trace_and_projector_validation_pass"]
    )
    engineering_pass = bool(channel_pass and identity_pass and rescue_equivalence_pass and strict_artifact_pass)
    primary_pass = bool(
        primary.get("ratio_defined") is True
        and primary.get("mean_over_G") is not None
        and primary["mean_over_G"] >= 0.30
        and isinstance(primary_ratio_interval, Mapping)
        and primary_ratio_interval.get("lower_95") is not None
        and primary_ratio_interval["lower_95"] > 0.20
        and primary["bootstrap_95"]["lower_95"] > 0.0
        and primary["positive_world_fraction"] >= 0.75
        and all(
            summary["n_worlds"] == 32 and summary["positive_world_count"] >= 24
            for summary in metrics["primary_strata"].values()
        )
        and channel["counterfactual_global_argmax_rate"]["cluster_bootstrap_95"]["lower_95"] > 0.50
    )
    specificity_components = {
        name: _effect_gate(summary, minimum_ratio=0.20) for name, summary in metrics["specificity"].items()
    }
    simultaneous_pass = _effect_gate(metrics["simultaneous_max_null"], minimum_ratio=0.20)
    specificity_pass = bool(primary_pass and all(specificity_components.values()) and simultaneous_pass)

    natural = metrics["natural_use"]
    damage_pass = _effect_gate(natural["c_erasure_damage"], minimum_ratio=0.10)
    rescue_pass = _effect_gate(natural["same_content_opposite_answer_rescue"], minimum_ratio=0.10)
    restoration = natural["restoration_fraction"]
    restoration_pass = bool(
        restoration.get("ratio_defined") is True
        and restoration.get("bootstrap_ratio_valid") is True
        and restoration.get("mean_over_G") is not None
        and restoration["mean_over_G"] >= 0.70
    )
    alternative_rescue_pass = bool(
        natural["superiority_to_opposite_content_same_answer_rescue"]["bootstrap_95"]["lower_95"] > 0.0
    )
    null_erasure_components = {
        name: summary["bootstrap_95"]["lower_95"] > 0.0
        for name, summary in natural["superiority_to_null_erasures"].items()
    }
    natural_use_pass = bool(
        specificity_pass
        and damage_pass
        and rescue_pass
        and restoration_pass
        and alternative_rescue_pass
        and all(null_erasure_components.values())
    )
    return {
        "engineering": {
            "identity_pass": identity_pass,
            "rescue_equivalence_pass": rescue_equivalence_pass,
            "strict_artifact_pass": strict_artifact_pass,
            "channel_pass": channel_pass,
            "unique_global_argmax_in_xy_rate": ">=0.95",
            "mean_label_probability_mass_loss": "<=0.02",
            "pass": engineering_pass,
        },
        "primary": {
            "pass": primary_pass,
            "thresholds": {
                "F_over_G_point": ">=0.30",
                "bootstrap_lower_F_over_G": ">0.20",
                "bootstrap_lower_F": ">0",
                "positive_world_fraction_overall_and_each_stratum": ">=0.75",
                "bootstrap_lower_counterfactual_target_flip": ">0.50",
            },
        },
        "specificity": {
            "components": specificity_components,
            "simultaneous_primary_minus_max_null": simultaneous_pass,
            "pass": specificity_pass,
        },
        "natural_use": {
            "c_erasure_damage": damage_pass,
            "same_content_opposite_answer_rescue": rescue_pass,
            "restoration_at_least_70_percent": restoration_pass,
            "superior_to_opposite_content_same_answer_rescue": (alternative_rescue_pass),
            "superior_to_each_null_erasure": null_erasure_components,
            "pass": natural_use_pass,
        },
    }


def _final_status(
    *,
    primary_pass: bool,
    specificity_pass: bool,
    natural_use_pass: bool,
    engineering_pass: bool,
) -> str:
    if not engineering_pass:
        return "FINAL_STOP_ENGINEERING_INVALID"
    if not primary_pass:
        return "NO_REPLICATED_PROJECTED_CONTENT_RECOMPOSITION"
    if not specificity_pass:
        return "NONSPECIFIC_PROJECTED_TRANSFER_REPLICATED"
    if not natural_use_pass:
        return "CONTENT_RECOMPOSITION_SUPPORTED_NATURAL_USE_NOT_ESTABLISHED"
    return "CONTENT_RECOMPOSITION_AND_PARTIAL_NATURAL_USE_SUPPORTED"


def _select_localization_layer(
    decisions: Mapping[int, Mapping[str, Any]], *, engineering_pass: bool
) -> tuple[int | None, str]:
    if set(decisions) != set(LAYER_GRID):
        raise CrossCodebookAnalysisError("localization layer grid changed")
    if not engineering_pass:
        return None, "LOCALIZATION_STOP_ENGINEERING_INVALID"
    validated = {}
    expected_keys = {
        "causal_anticopy_pass",
        "control_components",
        "max_four_null_means_pass",
        "control_separation_pass",
        "identity_pass",
        "pass",
    }
    for layer, raw in decisions.items():
        decision = dict(raw)
        if set(decision) != expected_keys:
            raise CrossCodebookAnalysisError("localization decision schema changed")
        components = decision["control_components"]
        if (
            not isinstance(components, Mapping)
            or set(components) != set(SPECIFICITY_CONTROLS)
            or any(not isinstance(value, bool) for value in components.values())
        ):
            raise CrossCodebookAnalysisError("localization control decision registry changed")
        scalar_fields = expected_keys - {"control_components"}
        if any(not isinstance(decision[field], bool) for field in scalar_fields):
            raise CrossCodebookAnalysisError("localization decisions must be Boolean")
        recomputed_control = bool(all(components.values()) and decision["max_four_null_means_pass"])
        recomputed_pass = bool(decision["causal_anticopy_pass"] and recomputed_control and decision["identity_pass"])
        if decision["control_separation_pass"] != recomputed_control or decision["pass"] != recomputed_pass:
            raise CrossCodebookAnalysisError("localization decision does not recompute")
        validated[layer] = decision
    passing = [layer for layer in LAYER_GRID if validated[layer]["pass"] is True]
    if not passing:
        return None, "LOCALIZATION_STOP_NO_PREREGISTERED_LAYER"
    return min(passing), "LOCALIZATION_LAYER_LOCKED_HOLDOUT_BASELINE_AUTHORIZED"


def _localization_metrics(rows: Sequence[Mapping[str, Any]], *, layer: int) -> dict[str, Any]:
    if layer not in LAYER_GRID:
        raise CrossCodebookAnalysisError("localization layer is outside the grid")
    expected_keys = {
        "cell_id",
        "world_id",
        "queried_property",
        "codebook_id",
        "recipient_margin",
        "counterfactual_margin",
        "condition_margins",
    }
    required_conditions = {
        PRIMARY_CONDITION,
        *SPECIFICITY_CONTROLS,
        *SIMULTANEOUS_NULLS,
        *ENGINEERING_CONDITIONS,
    }
    normalized = []
    for raw in rows:
        row = dict(raw)
        if set(row) != expected_keys:
            raise CrossCodebookAnalysisError("localization row schema changed")
        if row["queried_property"] not in {"P", "Q"} or row["codebook_id"] not in {"identity", "swapped"}:
            raise CrossCodebookAnalysisError("localization stratum is invalid")
        row["recipient_margin"] = _finite(row["recipient_margin"], "recipient_margin")
        row["counterfactual_margin"] = _finite(row["counterfactual_margin"], "counterfactual_margin")
        margins = row["condition_margins"]
        if not isinstance(margins, Mapping) or set(margins) != required_conditions:
            raise CrossCodebookAnalysisError("localization conditions changed")
        row["condition_margins"] = {
            condition: _finite(value, f"condition[{condition}]") for condition, value in margins.items()
        }
        normalized.append(row)
    if len(normalized) != 64 or len({row["cell_id"] for row in normalized}) != 64:
        raise CrossCodebookAnalysisError("localization requires 64 unique recipients")
    cell_to_world = {row["cell_id"]: row["world_id"] for row in normalized}
    world_counts = Counter(cell_to_world.values())
    if len(world_counts) != 8 or set(world_counts.values()) != {8}:
        raise CrossCodebookAnalysisError("localization requires eight worlds and eight recipients per world")
    bootstrap_indices = _common_bootstrap_indices(8)
    gap_cells = {row["cell_id"]: row["counterfactual_margin"] - row["recipient_margin"] for row in normalized}
    primary_cells = {
        row["cell_id"]: row["condition_margins"][PRIMARY_CONDITION] - row["recipient_margin"] for row in normalized
    }
    gap_worlds = _world_means(gap_cells, cell_to_world)
    primary_worlds = _world_means(primary_cells, cell_to_world)
    primary = _effect_summary(primary_worlds, gap_worlds, bootstrap_indices=bootstrap_indices)
    strata = {}
    for stratum in (
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    ):
        members = [row for row in normalized if _stratum_id(row) == stratum]
        if len(members) != 16:
            raise CrossCodebookAnalysisError("localization stratum count changed")
        if Counter(row["world_id"] for row in members) != {world_id: 2 for world_id in sorted(world_counts)}:
            raise CrossCodebookAnalysisError("localization strata require two recipients per world")
        strata[stratum] = float(np.mean([primary_cells[row["cell_id"]] for row in members]))
    specificity = {}
    for condition in SPECIFICITY_CONTROLS:
        difference = {
            row["cell_id"]: row["condition_margins"][PRIMARY_CONDITION] - row["condition_margins"][condition]
            for row in normalized
        }
        specificity[condition] = _effect_summary(
            _world_means(difference, cell_to_world),
            gap_worlds,
            bootstrap_indices=bootstrap_indices,
        )
    null_summaries = {}
    for condition in SIMULTANEOUS_NULLS:
        null_effect = {
            row["cell_id"]: row["condition_margins"][condition] - row["recipient_margin"] for row in normalized
        }
        null_summaries[condition] = _effect_summary(
            _world_means(null_effect, cell_to_world),
            gap_worlds,
            bootstrap_indices=bootstrap_indices,
        )
    identity_errors = [abs(row["condition_margins"]["identity"] - row["recipient_margin"]) for row in normalized]
    return {
        "layer": layer,
        "n_worlds": 8,
        "n_recipients": 64,
        "G": _effect_summary(gap_worlds, gap_worlds, bootstrap_indices=bootstrap_indices),
        "primary": primary,
        "primary_stratum_means": strata,
        "specificity": specificity,
        "nulls": null_summaries,
        "identity_maximum_absolute_margin_error": max(identity_errors),
    }


def _localization_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected_strata = {
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    }
    if set(metrics.get("primary_stratum_means", {})) != expected_strata:
        raise CrossCodebookAnalysisError("localization stratum registry changed")
    if set(metrics.get("specificity", {})) != set(SPECIFICITY_CONTROLS):
        raise CrossCodebookAnalysisError("localization specificity registry changed")
    if set(metrics.get("nulls", {})) != set(SIMULTANEOUS_NULLS):
        raise CrossCodebookAnalysisError("localization null registry changed")
    primary = metrics["primary"]
    loo_ratios = primary.get("leave_one_world_out_ratios")
    causal_anticopy_pass = bool(
        primary.get("ratio_defined") is True
        and primary.get("mean_over_G") is not None
        and primary["mean_over_G"] >= 0.30
        and isinstance(loo_ratios, list)
        and loo_ratios
        and min(loo_ratios) >= 0.20
        and primary["positive_world_count"] >= 6
        and all(value > 0.0 for value in metrics["primary_stratum_means"].values())
    )
    control_components = {}
    for condition, summary in metrics["specificity"].items():
        control_components[condition] = bool(
            summary.get("ratio_defined") is True
            and summary.get("mean_over_G") is not None
            and summary["mean_over_G"] >= 0.20
            and summary["leave_one_world_out_means"]
            and min(summary["leave_one_world_out_means"]) > 0.0
        )
    primary_loo = primary["leave_one_world_out_means"]
    null_means = [summary["mean"] for summary in metrics["nulls"].values()]
    null_loo = [summary["leave_one_world_out_means"] for summary in metrics["nulls"].values()]
    max_null_pass = bool(
        primary["mean"] > max(null_means)
        and all(
            primary_loo[index] > max(null_values[index] for null_values in null_loo)
            for index in range(len(primary_loo))
        )
    )
    identity_pass = bool(metrics["identity_maximum_absolute_margin_error"] <= 1e-4)
    control_separation_pass = bool(all(control_components.values()) and max_null_pass)
    return {
        "causal_anticopy_pass": causal_anticopy_pass,
        "control_components": control_components,
        "max_four_null_means_pass": max_null_pass,
        "control_separation_pass": control_separation_pass,
        "identity_pass": identity_pass,
        "pass": bool(causal_anticopy_pass and control_separation_pass and identity_pass),
    }


def _load_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    try:
        plan, design = runner._load_frozen_plan()
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    if not isinstance(plan, dict) or not isinstance(design, dict):
        raise CrossCodebookAnalysisError("runner returned an invalid frozen plan")
    return plan, design


def _validate_phase_manifest(
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    phase: str,
    expected_count: int,
    expected_activation_shape: tuple[int, ...] | None,
    expected_patched_activation_shape: tuple[int, ...] | None,
    selected_layer: int | None,
    prerequisite_bindings: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray | None,
    np.ndarray | None,
]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    try:
        manifest, records, activations = runner._validate_execution_manifest(
            phase,
            expected_count=expected_count,
            expected_activation_shape=expected_activation_shape,
            expected_patched_activation_shape=expected_patched_activation_shape,
        )
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    expected_manifest_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "design_file_sha256",
        "plan_manifest_file_sha256",
        "preregistration_sha256",
        "runner_sha256",
        "analyzer_sha256",
        "attempt",
        "records",
        "activations",
        "patched_activations",
        "prerequisite_bindings",
        "selected_layer",
        "model_calls",
        "generation_used",
        "biological_model_calls",
        "partial_resume_allowed",
    }
    if set(manifest) != expected_manifest_keys:
        raise CrossCodebookAnalysisError(f"{phase} manifest field registry changed")
    analyzer_binding = design.get("locks", {}).get("analyzer", {})
    if (
        manifest["call_plan_sha256"] != plan.get("call_plan_sha256")
        or manifest["analyzer_sha256"] != analyzer_binding.get("sha256")
        or manifest["selected_layer"] != selected_layer
        or manifest["prerequisite_bindings"] != dict(prerequisite_bindings)
    ):
        raise CrossCodebookAnalysisError(f"{phase} manifest authority changed")
    paths = runner.PHASE_PATHS[phase]
    attempt_binding = manifest["attempt"]
    if not isinstance(attempt_binding, Mapping) or set(attempt_binding) != {
        "path",
        "file_sha256",
    }:
        raise CrossCodebookAnalysisError(f"{phase} attempt binding changed")
    attempt_path = paths["attempt"]
    if attempt_binding["path"] != str(attempt_path) or attempt_binding["file_sha256"] != file_sha256(attempt_path):
        raise CrossCodebookAnalysisError(f"{phase} attempt receipt is unbound")
    attempt = _load_json(attempt_path)
    expected_attempt = {
        "schema_version": runner.ATTEMPT_SCHEMA,
        "status": "EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(runner.DEFAULT_PLAN_MANIFEST),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "runner_sha256": file_sha256(Path(runner.__file__)),
        "analyzer_sha256": analyzer_binding.get("sha256"),
        "prerequisite_bindings": dict(prerequisite_bindings),
        "model_calls_before_attempt": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }
    if attempt != expected_attempt:
        raise CrossCodebookAnalysisError(f"{phase} attempt receipt changed")
    patched_activations = None
    if expected_patched_activation_shape is not None:
        patched_path = paths.get("patched_activations")
        if not isinstance(patched_path, Path):
            raise CrossCodebookAnalysisError(f"{phase} patched-activation path is missing")
        try:
            patched_activations = runner._load_activation_sidecar(patched_path, expected_patched_activation_shape)
        except (runner.CrossCodebookRunnerError, OSError, ValueError) as error:
            raise CrossCodebookAnalysisError(str(error)) from error
        receipt = manifest.get("patched_activations")
        expected_map = {
            record["record_id"]: {
                "patched_activation_row": record["patched_activation_row"],
                "patched_activation_sha256": record["patched_activation_sha256"],
            }
            for record in records
        }
        if not isinstance(receipt, Mapping) or receipt.get("logical_id_map") != expected_map:
            raise CrossCodebookAnalysisError(f"{phase} patched activation logical map changed")
    return manifest, records, activations, patched_activations


def _activation_block(array: np.ndarray, row: int) -> np.ndarray:
    if array.ndim == 3:
        return np.ascontiguousarray(array[row], dtype="<f4")
    if array.ndim == 2:
        return np.ascontiguousarray(array[row][None, :], dtype="<f4")
    raise CrossCodebookAnalysisError("baseline activation sidecar rank changed")


def _validate_baseline_records(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    *,
    phase: str,
    role: str,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    try:
        templates = runner._phase_baseline_templates(plan, role)
    except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    if len(records) != len(templates) or activations.shape[0] != len(records):
        raise CrossCodebookAnalysisError("baseline record/template counts differ")
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    prompts = {prompt["prompt_id"]: prompt for prompt in plan["prompts"]}
    if len(cells) != len(plan["cell_registry"]) or len(prompts) != len(plan["prompts"]):
        raise CrossCodebookAnalysisError("plan cell or prompt IDs are duplicated")
    expected_layers = list(layers)
    observed: list[dict[str, Any]] = []
    for index, (raw, template) in enumerate(zip(records, templates, strict=True)):
        row = dict(raw)
        if row.get("cell_id") != template["cell_id"]:
            raise CrossCodebookAnalysisError("baseline record order changed")
        cell = cells.get(template["cell_id"])
        prompt = prompts.get(template["prompt_id"])
        if cell is None or prompt is None or cell.get("role") != role:
            raise CrossCodebookAnalysisError("baseline plan identity does not resolve")
        if (
            row.get("activation_row") != index
            or row.get("phase") != phase
            or row.get("captured_layers") != expected_layers
        ):
            raise CrossCodebookAnalysisError("baseline phase, row, or layer changed")
        if role == "holdout":
            if template.get("capture_layers") is not None or template.get("capture_layer_from_lock") is not True:
                raise CrossCodebookAnalysisError("holdout capture template changed")
        elif template.get("capture_layers") != expected_layers or template.get("capture_layer_from_lock") is not False:
            raise CrossCodebookAnalysisError("grid capture template changed")
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise CrossCodebookAnalysisError("baseline diagnostics are missing")
        diagnostics = _validate_diagnostics(diagnostics)
        trace = row.get("forward_trace")
        expected_trace = {
            "use_cache": False,
            "return_dict": True,
            "generation_used": False,
            "teacher_forced_prompt_forward": True,
            "capture_layers": expected_layers,
            "capture_counts": [1] * len(expected_layers),
            "captures_removed": True,
            "final_attended_token_index": -1,
            "model_calls": 1,
        }
        if trace != expected_trace:
            raise CrossCodebookAnalysisError("baseline forward trace changed")
        block = _activation_block(activations, index)
        try:
            expected = runner._baseline_record(
                plan,
                prompt,
                cell,
                phase=phase,
                activation_row=index,
                layers=expected_layers,
                activations=block,
                diagnostics=diagnostics,
                trace=trace,
            )
        except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
            raise CrossCodebookAnalysisError(str(error)) from error
        if row != expected:
            raise CrossCodebookAnalysisError("baseline record does not reconstruct")
        observed.append(row)
    if len({row["record_id"] for row in observed}) != len(observed):
        raise CrossCodebookAnalysisError("baseline record IDs are duplicated")
    activation_receipt = manifest.get("activations")
    if not isinstance(activation_receipt, Mapping):
        raise CrossCodebookAnalysisError("baseline activation receipt is missing")
    expected_logical_map = {
        row["cell_id"]: {
            "activation_row": row["activation_row"],
            "activation_sha256": row["activation_sha256"],
            "activation_layer_sha256": row["activation_layer_sha256"],
        }
        for row in observed
    }
    if activation_receipt.get("logical_id_map") != expected_logical_map:
        raise CrossCodebookAnalysisError("baseline logical activation map changed")
    return observed


def _behavioral_inputs(
    plan: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "cell_id": row["cell_id"],
            "world_id": row["world_id"],
            "queried_property": row["queried_property"],
            "codebook_id": row["codebook_id"],
            "native_answer": row["native_answer"],
            "diagnostics": row["diagnostics"],
        }
        for row in baselines
    ]
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"] if cell["role"] == role}
    pairs = [
        {
            "recipient_cell_id": cell["cell_id"],
            "counterfactual_cell_id": cell["text_counterfactual_cell_id"],
            "world_id": cell["world_id"],
            "queried_property": cell["queried_property"],
            "codebook_id": cell["codebook_id"],
        }
        for cell in sorted(cells.values(), key=lambda value: value["cell_id"])
        if cell["recipient_selected"] is True
    ]
    return rows, pairs


_PATCH_TRACE_KEYS = {
    "hook_calls",
    "hook_removed",
    "non_target_tokens_unchanged",
    "pre_activation_matches_registered_recipient",
    "post_activation_matches_expected",
    "pre_activation_sha256",
    "post_activation_sha256",
    "expected_activation_sha256",
    "post_expected_l2_error",
    "post_expected_l2_tolerance",
    "displacement_l2",
    "direction_input_norm",
    "projector_symmetry_error",
    "projector_idempotence_error",
    "unit_norm_error",
    "orthogonal_displacement_l2",
    "orthogonal_displacement_tolerance",
    "corresponding_full_displacement_l2",
    "selective_displacement_tolerance",
    "orthogonal_displacement_pass",
    "selective_not_larger_than_full_pass",
    "pre_axis_coefficient",
    "source_axis_coefficient",
    "post_axis_coefficient",
    "expected_axis_coefficient",
    "post_expected_axis_coefficient_error",
    "finite_activations",
    "operation",
    "intervention_kind",
    "layer",
    "token_index",
    "strength",
    "model_calls",
    "generation_used",
    "patched_activation_hash_pass",
    "finite_logits",
    "direction_name",
}


def _close(first: Any, second: float, *, atol: float = 1e-9) -> bool:
    try:
        return abs(_finite(first, "trace numeric value") - second) <= atol
    except CrossCodebookAnalysisError:
        return False


def _validate_patch_trace(
    trace: Mapping[str, Any],
    *,
    layer: int,
    operation: str,
    direction_name: str | None,
    recipient: np.ndarray,
    source: np.ndarray | None,
    direction: np.ndarray | None,
    center: np.ndarray | None,
    patched_activation: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    value = dict(trace)
    if set(value) != _PATCH_TRACE_KEYS:
        raise CrossCodebookAnalysisError("patch hook-trace field registry changed")
    try:
        expected = runner.expected_intervention_activation(
            recipient,
            operation=operation,
            source=source,
            direction=direction,
            center=center,
        )
    except (runner.CrossCodebookRunnerError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    expected_hash = f32_sha256(expected)
    patched = np.ascontiguousarray(patched_activation, dtype="<f4")
    if patched.shape != recipient.shape or not np.isfinite(patched).all():
        raise CrossCodebookAnalysisError("patched activation sidecar row is invalid")
    required_booleans = {
        "hook_removed",
        "non_target_tokens_unchanged",
        "pre_activation_matches_registered_recipient",
        "post_activation_matches_expected",
        "orthogonal_displacement_pass",
        "selective_not_larger_than_full_pass",
        "finite_activations",
        "patched_activation_hash_pass",
        "finite_logits",
    }
    if any(value.get(field) is not True for field in required_booleans):
        raise CrossCodebookAnalysisError("patch hook or numerical Boolean gate failed")
    if (
        value.get("hook_calls") != 1
        or value.get("operation") != operation
        or value.get("intervention_kind") != operation
        or value.get("direction_name") != direction_name
        or value.get("layer") != layer
        or value.get("token_index") != -1
        or value.get("strength") != 1.0
        or value.get("model_calls") != 1
        or value.get("generation_used") is not False
        or value.get("pre_activation_sha256") != f32_sha256(recipient)
        or value.get("post_activation_sha256") != f32_sha256(patched)
        or value.get("expected_activation_sha256") != expected_hash
    ):
        raise CrossCodebookAnalysisError("patch hook trace identity changed")
    _validate_sha256(value.get("post_activation_sha256"), "post activation")
    tolerance = 1e-6 * max(1.0, float(np.linalg.norm(expected.astype(np.float64))))
    expected_error = float(np.linalg.norm(patched.astype(np.float64) - expected.astype(np.float64)))
    error = _finite(value.get("post_expected_l2_error"), "post expected error")
    if (
        error < 0.0
        or not _close(error, expected_error)
        or not _close(value.get("post_expected_l2_tolerance"), tolerance)
        or error > tolerance
    ):
        raise CrossCodebookAnalysisError("patch post-activation tolerance changed")
    observed_displacement = float(np.linalg.norm(patched.astype(np.float64) - recipient.astype(np.float64)))
    displacement = _finite(value.get("displacement_l2"), "patch displacement")
    if displacement < 0.0 or not _close(displacement, observed_displacement):
        raise CrossCodebookAnalysisError("patch displacement does not reconstruct")

    nullable_projector_fields = {
        "direction_input_norm",
        "projector_symmetry_error",
        "projector_idempotence_error",
        "unit_norm_error",
        "orthogonal_displacement_l2",
        "orthogonal_displacement_tolerance",
        "corresponding_full_displacement_l2",
        "selective_displacement_tolerance",
        "pre_axis_coefficient",
        "source_axis_coefficient",
        "post_axis_coefficient",
        "expected_axis_coefficient",
        "post_expected_axis_coefficient_error",
    }
    if direction is None:
        if any(value[field] is not None for field in nullable_projector_fields):
            raise CrossCodebookAnalysisError("full patch unexpectedly reports a projector")
        if f32_sha256(patched) != expected_hash:
            raise CrossCodebookAnalysisError("full patch is not an exact source copy")
        return value, expected

    direction64 = np.asarray(direction, dtype=np.float64)
    unit = direction64 / np.linalg.norm(direction64)
    try:
        projector = runner._projector_diagnostics(direction)
    except (runner.CrossCodebookRunnerError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    for field, expected_value in projector.items():
        if not _close(value.get(field), expected_value):
            raise CrossCodebookAnalysisError("projector diagnostic does not reconstruct")
    if projector["unit_norm_error"] > 1e-6:
        raise CrossCodebookAnalysisError("applied direction is not unit norm")
    origin = np.zeros_like(recipient, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)
    pre_axis = float(unit @ (recipient.astype(np.float64) - origin))
    source_axis = None if source is None else float(unit @ (source.astype(np.float64) - origin))
    expected_axis = float(unit @ (expected.astype(np.float64) - origin))
    if (
        not _close(value.get("pre_axis_coefficient"), pre_axis)
        or (source_axis is None and value.get("source_axis_coefficient") is not None)
        or (source_axis is not None and not _close(value.get("source_axis_coefficient"), source_axis))
        or not _close(value.get("expected_axis_coefficient"), expected_axis)
    ):
        raise CrossCodebookAnalysisError("axis coefficients do not reconstruct")
    expected_post_axis = float(unit @ (patched.astype(np.float64) - origin))
    post_axis = _finite(value.get("post_axis_coefficient"), "post axis coefficient")
    axis_error = _finite(value.get("post_expected_axis_coefficient_error"), "axis coefficient error")
    if (
        not _close(post_axis, expected_post_axis)
        or not _close(axis_error, abs(post_axis - expected_axis))
        or axis_error > error + 1e-9
    ):
        raise CrossCodebookAnalysisError("post-axis coefficient error changed")
    orthogonal_tolerance = 1e-6 * max(1.0, float(np.linalg.norm(recipient.astype(np.float64))))
    delta = patched.astype(np.float64) - recipient.astype(np.float64)
    expected_orthogonal = float(np.linalg.norm(delta - unit * float(unit @ delta)))
    orthogonal = _finite(value.get("orthogonal_displacement_l2"), "orthogonal displacement")
    if (
        not _close(orthogonal, expected_orthogonal)
        or not _close(
            value.get("orthogonal_displacement_tolerance"),
            orthogonal_tolerance,
        )
        or orthogonal > orthogonal_tolerance
    ):
        raise CrossCodebookAnalysisError("orthogonal displacement gate changed")
    reference = source if source is not None else center
    if reference is None:
        raise CrossCodebookAnalysisError("selective reference is missing")
    full_displacement = float(np.linalg.norm(reference.astype(np.float64) - recipient.astype(np.float64)))
    selective_tolerance = 1e-6 * max(1.0, full_displacement)
    if (
        not _close(value.get("corresponding_full_displacement_l2"), full_displacement)
        or not _close(value.get("selective_displacement_tolerance"), selective_tolerance)
        or displacement > full_displacement + selective_tolerance
    ):
        raise CrossCodebookAnalysisError("selective displacement gate changed")
    return value, expected


def _validate_patch_records(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    patched_activations: np.ndarray,
    basis_sidecar: np.ndarray,
    basis_details: Mapping[str, Any],
    *,
    phase: str,
    role: str,
    selected_layer: int | None,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    try:
        templates = runner._phase_patch_templates(plan, role)
    except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    if (
        len(records) != len(templates)
        or patched_activations.shape != (len(templates), 1536)
        or not np.isfinite(patched_activations).all()
    ):
        raise CrossCodebookAnalysisError("patch record/template counts differ")
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    prompts = {prompt["cell_id"]: prompt for prompt in plan["prompts"]}
    baseline_by_cell = {row["cell_id"]: row for row in baselines}
    if len(baseline_by_cell) != len(baselines):
        raise CrossCodebookAnalysisError("baseline cell IDs are duplicated")
    observed: list[dict[str, Any]] = []
    for index, (raw, template) in enumerate(zip(records, templates, strict=True)):
        row = dict(raw)
        if row.get("template_id") != template["template_id"]:
            raise CrossCodebookAnalysisError("patch record order changed")
        layer = template["layer"] if role == "localization" else selected_layer
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in LAYER_GRID:
            raise CrossCodebookAnalysisError("patch layer is not frozen")
        recipient_cell = cells.get(template["recipient_cell_id"])
        recipient_prompt = prompts.get(template["recipient_cell_id"])
        recipient_baseline = baseline_by_cell.get(template["recipient_cell_id"])
        source_baseline = (
            None if template["source_cell_id"] is None else baseline_by_cell.get(template["source_cell_id"])
        )
        if (
            recipient_cell is None
            or recipient_prompt is None
            or recipient_baseline is None
            or recipient_cell.get("role") != role
            or recipient_cell.get("recipient_selected") is not True
            or (template["source_cell_id"] is not None and source_baseline is None)
        ):
            raise CrossCodebookAnalysisError("patch source graph does not resolve")
        try:
            expected_template = runner._patch_template(
                recipient_cell,
                condition=template["condition"],
                layer=template["layer"],
            )
            recipient_activation = runner._activation_for_record(
                recipient_baseline, activations, layer=layer, role=role
            )
            source_activation = (
                None
                if source_baseline is None
                else runner._activation_for_record(source_baseline, activations, layer=layer, role=role)
            )
            fit_center, directions = runner._basis_vectors_at_layer(basis_sidecar, basis_details, layer)
        except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
            raise CrossCodebookAnalysisError(str(error)) from error
        if expected_template != template:
            raise CrossCodebookAnalysisError("patch template does not reconstruct")
        direction_name = template["direction_name"]
        direction = None if direction_name is None else directions[direction_name]
        center = fit_center if template["operation"] in {"erasure", "rescue"} else None
        diagnostics = row.get("diagnostics")
        trace = row.get("hook_trace")
        if not isinstance(diagnostics, Mapping) or not isinstance(trace, Mapping):
            raise CrossCodebookAnalysisError("patch diagnostics or trace is missing")
        diagnostics = _validate_diagnostics(diagnostics)
        trace, expected_activation = _validate_patch_trace(
            trace,
            layer=layer,
            operation=template["operation"],
            direction_name=direction_name,
            recipient=recipient_activation,
            source=source_activation,
            direction=direction,
            center=center,
            patched_activation=patched_activations[index],
        )
        if (
            row.get("patched_activation_row") != index
            or row.get("patched_activation_sha256") != f32_sha256(patched_activations[index])
            or row.get("patched_activation_sha256") != trace["post_activation_sha256"]
            or row.get("expected_activation_sha256") != f32_sha256(expected_activation)
        ):
            raise CrossCodebookAnalysisError("patched activation hashes changed")
        try:
            expected_row = runner._patch_record(
                plan,
                template,
                recipient_cell,
                recipient_prompt,
                phase=phase,
                layer=layer,
                patched_activation_row=index,
                recipient_baseline=recipient_baseline,
                source_baseline=source_baseline,
                recipient_activation=recipient_activation,
                source_activation=source_activation,
                direction=direction,
                center=center,
                diagnostics=diagnostics,
                trace=trace,
            )
        except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
            raise CrossCodebookAnalysisError(str(error)) from error
        if row != expected_row:
            raise CrossCodebookAnalysisError("patch record does not reconstruct")
        observed.append(row)
    if len({row["record_id"] for row in observed}) != len(observed):
        raise CrossCodebookAnalysisError("patch record IDs are duplicated")
    return observed, {
        "trace_projector_and_sidecar_pass": True,
        "source_reference_pass": True,
        "patched_activation_hash_pass": True,
        "fit_only_center_pass": True,
    }


def _validate_fit_basis(
    activations: np.ndarray,
    sidecar: np.ndarray,
    details: Mapping[str, Any],
    calculations: np.ndarray,
) -> dict[str, Any]:
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    expected_registry = [
        "walsh_00_intercept",
        "fit_intercept",
        *[f"walsh_{mask:02d}" for mask in range(1, 32)],
        *[f"direction_{name}" for name in runner.DIRECTION_NAMES],
    ]
    if (
        sidecar.shape != (5, 43, 1536)
        or sidecar.dtype != np.dtype("<f4")
        or not np.isfinite(sidecar).all()
        or details.get("schema_version") != runner.BASIS_SCHEMA
        or details.get("layer_grid") != list(LAYER_GRID)
        or details.get("registry") != expected_registry
        or details.get("sidecar_shape") != [5, 43, 1536]
        or details.get("sidecar_logical_sha256") != f32_sha256(sidecar)
        or calculations.shape != (5, 62, 1536)
        or calculations.dtype != np.dtype("<f8")
        or not np.isfinite(calculations).all()
        or details.get("calculation_sidecar_shape") != [5, 62, 1536]
        or details.get("calculation_sidecar_logical_sha256") != runner.f64_sha256(calculations)
    ):
        raise CrossCodebookAnalysisError("fit basis registry or sidecar changed")
    center = _fit_only_intercept(activations)
    if not np.array_equal(sidecar[:, 0], center) or not np.array_equal(sidecar[:, 1], center):
        raise CrossCodebookAnalysisError("fit-only intercept does not reconstruct")
    layer_details = details.get("layers")
    if not isinstance(layer_details, Mapping) or set(layer_details) != {str(layer) for layer in LAYER_GRID}:
        raise CrossCodebookAnalysisError("fit basis layer registry changed")
    direction_receipts: dict[str, Any] = {}
    for layer_position, layer in enumerate(LAYER_GRID):
        layer_value = layer_details[str(layer)]
        if (
            not isinstance(layer_value, Mapping)
            or layer_value.get("applied_layer_f32_sha256") != f32_sha256(sidecar[layer_position])
            or layer_value.get("calculation_layer_f64_sha256") != runner.f64_sha256(calculations[layer_position])
        ):
            raise CrossCodebookAnalysisError("fit basis layer hash changed")
        directions = layer_value.get("directions")
        if not isinstance(directions, Mapping) or set(directions) != set(runner.DIRECTION_NAMES):
            raise CrossCodebookAnalysisError("fit direction registry changed")
        direction_receipts[str(layer)] = {}
        for name in runner.DIRECTION_NAMES:
            vector = sidecar[layer_position, expected_registry.index(f"direction_{name}")]
            receipt = directions[name]
            if not isinstance(receipt, Mapping):
                raise CrossCodebookAnalysisError("fit direction receipt is invalid")
            residual_ratio = _finite(receipt.get("residual_ratio"), "basis residual ratio")
            residual_norm = _finite(receipt.get("residual_norm"), "basis residual norm")
            raw_norm = _finite(receipt.get("raw_norm"), "basis raw norm")
            rank = receipt.get("retained_rank")
            singular_values = receipt.get("singular_values")
            if (
                residual_ratio <= runner.SVD_RELATIVE_TOLERANCE
                or residual_norm <= 0.0
                or raw_norm < 0.0
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 0
                or not isinstance(singular_values, list)
                or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in singular_values)
                or receipt.get("direction_f32_sha256") != f32_sha256(vector)
                or abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0) > 1e-6
            ):
                raise CrossCodebookAnalysisError("fit direction eligibility changed")
            for digest_field in (
                "raw_f64_sha256",
                "residual_f64_sha256",
            ):
                _validate_sha256(receipt.get(digest_field), digest_field)
            direction_receipts[str(layer)][name] = {
                "retained_rank": rank,
                "residual_ratio": residual_ratio,
                "direction_f32_sha256": f32_sha256(vector),
            }
    return {
        "pass": True,
        "sidecar_shape": list(sidecar.shape),
        "sidecar_logical_sha256": f32_sha256(sidecar),
        "calculation_sidecar_shape": list(calculations.shape),
        "calculation_sidecar_logical_sha256": runner.f64_sha256(calculations),
        "fit_only_intercept_f32_sha256_by_layer": {
            str(layer): f32_sha256(center[index]) for index, layer in enumerate(LAYER_GRID)
        },
        "directions": direction_receipts,
    }


def _baseline_margin(baseline: Mapping[str, Any], *, native_answer: str | None = None) -> float:
    answer = baseline["native_answer"] if native_answer is None else native_answer
    derived = target_oriented_margin(
        native_answer=answer,
        x_logit=baseline["diagnostics"]["x_logit"],
        y_logit=baseline["diagnostics"]["y_logit"],
    )
    if (
        answer == baseline["native_answer"]
        and abs(_finite(baseline.get("target_oriented_margin"), "baseline target margin") - derived) > 1e-7
    ):
        raise CrossCodebookAnalysisError("baseline target-oriented margin changed")
    return derived


def _patch_margin(patch: Mapping[str, Any], *, native_answer: str | None = None) -> float:
    answer = patch["native_answer"] if native_answer is None else native_answer
    derived = target_oriented_margin(
        native_answer=answer,
        x_logit=patch["diagnostics"]["x_logit"],
        y_logit=patch["diagnostics"]["y_logit"],
    )
    if (
        answer == patch["native_answer"]
        and abs(_finite(patch.get("target_oriented_margin"), "patch target margin") - derived) > 1e-7
    ):
        raise CrossCodebookAnalysisError("patch target-oriented margin changed")
    return derived


def _patch_registry(
    patches: Sequence[Mapping[str, Any]], *, include_layer: bool
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in patches:
        row = dict(raw)
        key = (
            (row["recipient_cell_id"], row["layer"], row["condition"])
            if include_layer
            else (row["recipient_cell_id"], row["condition"])
        )
        if key in result:
            raise CrossCodebookAnalysisError("patch condition registry is duplicated")
        result[key] = row
    return result


def _localization_rows_from_artifacts(
    plan: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    patches: Sequence[Mapping[str, Any]],
    *,
    layer: int,
) -> list[dict[str, Any]]:
    baseline_by_cell = {row["cell_id"]: row for row in baselines}
    patch_by_key = _patch_registry(patches, include_layer=True)
    conditions = {
        PRIMARY_CONDITION,
        *SPECIFICITY_CONTROLS,
        *SIMULTANEOUS_NULLS,
        *ENGINEERING_CONDITIONS,
    }
    recipients = sorted(
        (
            cell
            for cell in plan["cell_registry"]
            if cell["role"] == "localization" and cell["recipient_selected"] is True
        ),
        key=lambda value: value["cell_id"],
    )
    rows = []
    for cell in recipients:
        baseline = baseline_by_cell.get(cell["cell_id"])
        counterfactual = baseline_by_cell.get(cell["text_counterfactual_cell_id"])
        if baseline is None or counterfactual is None:
            raise CrossCodebookAnalysisError("localization baseline source is missing")
        condition_rows = {condition: patch_by_key.get((cell["cell_id"], layer, condition)) for condition in conditions}
        if any(value is None for value in condition_rows.values()):
            raise CrossCodebookAnalysisError("localization condition row is missing")
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "world_id": cell["world_id"],
                "queried_property": cell["queried_property"],
                "codebook_id": cell["codebook_id"],
                "recipient_margin": _baseline_margin(baseline),
                "counterfactual_margin": _baseline_margin(counterfactual, native_answer=cell["native_answer"]),
                "condition_margins": {
                    condition: _patch_margin(
                        condition_rows[condition],
                        native_answer=cell["native_answer"],
                    )
                    for condition in conditions
                },
            }
        )
    return rows


def _identity_engineering(
    baselines: Sequence[Mapping[str, Any]],
    patches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_cell = {row["cell_id"]: row for row in baselines}
    comparisons = []
    for patch in patches:
        if patch["condition"] != "identity":
            continue
        baseline = baseline_by_cell.get(patch["recipient_cell_id"])
        if baseline is None:
            raise CrossCodebookAnalysisError("identity recipient baseline is missing")
        comparisons.append(
            _diagnostic_equivalence(
                baseline["diagnostics"],
                patch["diagnostics"],
                require_global_max_preserved=True,
            )
        )
    if not comparisons:
        raise CrossCodebookAnalysisError("identity comparison registry is empty")
    return {
        "n_comparisons": len(comparisons),
        "maximum_absolute_x_logit_error": max(value["absolute_x_logit_difference"] for value in comparisons),
        "maximum_absolute_y_logit_error": max(value["absolute_y_logit_difference"] for value in comparisons),
        "all_global_maxima_preserved": all(value["global_max_preserved"] for value in comparisons),
        "pass": all(value["pass"] for value in comparisons),
    }


def _final_rows_from_artifacts(
    plan: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    patches: Sequence[Mapping[str, Any]],
    artifact_gates: Mapping[str, bool],
) -> list[dict[str, Any]]:
    baseline_by_cell = {row["cell_id"]: row for row in baselines}
    patch_by_key = _patch_registry(patches, include_layer=False)
    required_conditions = {
        PRIMARY_CONDITION,
        *SPECIFICITY_CONTROLS,
        *SIMULTANEOUS_NULLS,
        *NATURAL_USE_CONDITIONS,
        *ENGINEERING_CONDITIONS,
    }
    recipients = sorted(
        (cell for cell in plan["cell_registry"] if cell["role"] == "holdout" and cell["recipient_selected"] is True),
        key=lambda value: value["cell_id"],
    )
    rows = []
    for cell in recipients:
        baseline = baseline_by_cell.get(cell["cell_id"])
        counterfactual = baseline_by_cell.get(cell["text_counterfactual_cell_id"])
        if baseline is None or counterfactual is None:
            raise CrossCodebookAnalysisError("holdout baseline source is missing")
        condition_rows = {
            condition: patch_by_key.get((cell["cell_id"], condition)) for condition in required_conditions
        }
        if any(value is None for value in condition_rows.values()):
            raise CrossCodebookAnalysisError("holdout condition row is missing")
        primary = condition_rows[PRIMARY_CONDITION]
        identity_comparison = _diagnostic_equivalence(
            baseline["diagnostics"],
            condition_rows["identity"]["diagnostics"],
            require_global_max_preserved=True,
        )
        rescue_same_comparison = _diagnostic_equivalence(
            condition_rows["content_same"]["diagnostics"],
            condition_rows["content_rescue_same"]["diagnostics"],
            require_global_max_preserved=False,
        )
        rescue_opposite_comparison = _diagnostic_equivalence(
            primary["diagnostics"],
            condition_rows["content_rescue_opposite"]["diagnostics"],
            require_global_max_preserved=False,
        )
        receipt = _engineering_receipt(
            {
                "identity_absolute_x_logit_error": identity_comparison["absolute_x_logit_difference"],
                "identity_absolute_y_logit_error": identity_comparison["absolute_y_logit_difference"],
                "identity_global_max_preserved": identity_comparison["global_max_preserved"],
                "rescue_same_absolute_x_logit_error": rescue_same_comparison["absolute_x_logit_difference"],
                "rescue_same_absolute_y_logit_error": rescue_same_comparison["absolute_y_logit_difference"],
                "rescue_opposite_absolute_x_logit_error": (rescue_opposite_comparison["absolute_x_logit_difference"]),
                "rescue_opposite_absolute_y_logit_error": (rescue_opposite_comparison["absolute_y_logit_difference"]),
                **dict(artifact_gates),
            }
        )
        target_token = Y_TOKEN_ID if cell["native_answer"] == "X" else X_TOKEN_ID
        maximum_ids = primary["diagnostics"]["maximum_token_ids"]
        unique_xy = bool(maximum_ids in ([X_TOKEN_ID], [Y_TOKEN_ID]))
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "world_id": cell["world_id"],
                "queried_property": cell["queried_property"],
                "codebook_id": cell["codebook_id"],
                "recipient_margin": _baseline_margin(baseline),
                "counterfactual_margin": _baseline_margin(counterfactual, native_answer=cell["native_answer"]),
                "recipient_label_probability_mass": baseline["diagnostics"]["label_probability_mass"],
                "primary_label_probability_mass": primary["diagnostics"]["label_probability_mass"],
                "primary_global_argmax_is_counterfactual": maximum_ids == [target_token],
                "primary_unique_global_argmax_in_xy": unique_xy,
                "engineering_receipt": receipt,
                "condition_margins": {
                    condition: _patch_margin(
                        condition_rows[condition],
                        native_answer=cell["native_answer"],
                    )
                    for condition in required_conditions
                },
            }
        )
    return _normalized_final_rows(rows)


def _write_frozen_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CrossCodebookAnalysisError(f"refusing to overwrite differing frozen analysis artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CrossCodebookAnalysisError(f"stale atomic-write temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_frozen_bytes(path, payload)


def _write_array(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise CrossCodebookAnalysisError("analysis array is non-finite")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _write_frozen_bytes(path, stream.getvalue())


def _write_f64_array(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.isfinite(array).all():
        raise CrossCodebookAnalysisError("analysis float64 array is non-finite")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _write_frozen_bytes(path, stream.getvalue())


def _render_bilingual_markdown(analysis: Mapping[str, Any]) -> str:
    status = analysis["status"]
    selected_layer = analysis.get("selected_layer")
    return "\n".join(
        [
            "# Cross-codebook projected-content analysis",
            "",
            "## English",
            "",
            f"- Status: `{status}`",
            f"- Locked layer: `{selected_layer}`",
            "- Scope: synthetic symbolic evidence in one locked model only.",
            "- Verifiability: selected-token patched states are independently reconstructed; full-vocabulary and runtime-hook evidence remains hash-bound runner attestation.",
            "- This does not establish biology, latent knowledge, an activation gap, "
            "a physical law, or model-family generality.",
            "",
            "## 한국어",
            "",
            f"- 상태: `{status}`",
            f"- 고정 레이어: `{selected_layer}`",
            "- 범위: 하나의 고정 모델에서 수행한 합성 기호 실험에 한정됩니다.",
            "- 검증성: 선택 토큰의 패치 상태는 독립 재구성하지만, 전체 어휘 및 런타임 훅 증거는 해시로 결속된 runner 증언입니다.",
            "- 생물학, 잠재지식, activation gap, 물리 법칙 또는 모델 계열 일반화를 입증하지 않습니다.",
            "",
        ]
    )


def _write_analysis_bundle(
    analysis: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    manifest_path: Path,
    input_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not PREREGISTRATION_PATH.is_file() or file_sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise CrossCodebookAnalysisError("preregistration binding changed")
    _write_json(json_path, analysis)
    markdown = _render_bilingual_markdown(analysis)
    _write_frozen_bytes(markdown_path, markdown.encode("utf-8"))
    manifest = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "status": analysis["status"],
        "claim_boundaries": CLAIM_BOUNDARIES,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "file_sha256": PREREGISTRATION_SHA256,
        },
        "input_bindings": dict(input_bindings or {}),
        "analysis_json": {
            "path": str(json_path),
            "file_sha256": file_sha256(json_path),
            "canonical_sha256": canonical_sha256(dict(analysis)),
        },
        "analysis_markdown": {
            "path": str(markdown_path),
            "file_sha256": file_sha256(markdown_path),
        },
        "analyzer_sha256": file_sha256(Path(__file__)),
        "runner_sha256": (file_sha256(Path(runner.__file__)) if runner is not None else "TO_FREEZE"),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "biological_model_calls": 0,
    }
    _write_json(manifest_path, manifest)
    return manifest


def analyze_fit_basis() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    manifest, raw_records, activations, fit_patched = _validate_phase_manifest(
        plan,
        design,
        phase="fit-baseline",
        expected_count=512,
        expected_activation_shape=(512, 5, 1536),
        expected_patched_activation_shape=None,
        selected_layer=None,
        prerequisite_bindings={},
    )
    if activations is None or fit_patched is not None:
        raise CrossCodebookAnalysisError("fit activation sidecar is missing")
    baselines = _validate_baseline_records(
        plan,
        manifest,
        raw_records,
        activations,
        phase="fit-baseline",
        role="direction_fit",
        layers=LAYER_GRID,
    )
    behavioral_rows, recipient_pairs = _behavioral_inputs(plan, baselines, role="direction_fit")
    admission = _behavioral_admission(behavioral_rows, recipient_pairs, expected_worlds=16)
    admitted = admission["gates"]["pass"] is True
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    fit_manifest_path = runner.PHASE_PATHS["fit-baseline"]["manifest"]
    try:
        sidecar, details, calculations = runner._fit_basis_with_calculations(plan, baselines, activations)
    except (runner.CrossCodebookRunnerError, KeyError, ValueError) as error:
        message = str(error)
        degeneracy_prefixes = (
            "ineligible residual",
            "ineligible null residual",
            "direction is not unit norm",
            "null direction is not unit norm",
        )
        if not message.startswith(degeneracy_prefixes):
            raise CrossCodebookAnalysisError(message) from error
        status = "FIT_STOP_BASIS_LOCK_INVALID"
        basis_engineering = {
            "pass": False,
            "failure": message,
            "all_five_grid_layers_required": True,
            "reduced_layer_grid_forbidden": True,
        }
        analysis = {
            "schema_version": FIT_ANALYSIS_SCHEMA,
            "status": status,
            "claim_authority": "fit_only_basis_and_baseline_admission",
            "claim_boundaries": CLAIM_BOUNDARIES,
            "behavioral_admission": admission,
            "basis_engineering": basis_engineering,
            "call_plan_sha256": plan["call_plan_sha256"],
            "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
            "fit_execution_manifest_file_sha256": file_sha256(fit_manifest_path),
            "fit_records_file_sha256": manifest["records"]["file_sha256"],
            "fit_activations_file_sha256": manifest["activations"]["file_sha256"],
            "basis_sidecar_logical_sha256": None,
            "basis_calculations_logical_sha256": None,
            "basis_details_canonical_sha256": None,
            "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
            "runner_sha256": design["locks"]["runner"]["sha256"],
            "model_forwards_executed_by_analyzer": 0,
            "biological_model_calls": 0,
        }
        _write_json(FIT_ANALYSIS_PATH, analysis)
        lock = {
            "schema_version": BASIS_LOCK_SCHEMA,
            "status": status,
            "call_plan_sha256": plan["call_plan_sha256"],
            "engineering_pass": False,
            "behavioral_admission_pass": admitted,
            "localization_baseline_authorized": False,
            "fit_execution_manifest_file_sha256": file_sha256(fit_manifest_path),
            "fit_analysis": {
                "path": str(FIT_ANALYSIS_PATH),
                "file_sha256": file_sha256(FIT_ANALYSIS_PATH),
                "canonical_sha256": canonical_sha256(analysis),
            },
            "basis_sidecar": None,
            "basis_calculations": None,
            "basis_details": None,
            "failure": message,
            "holdout_data_used": False,
            "biological_execution_allowed": False,
            "biological_model_calls": 0,
        }
        _write_json(runner.DEFAULT_BASIS_LOCK, lock)
        return analysis, lock
    sidecar = np.ascontiguousarray(sidecar, dtype="<f4")
    calculations = np.ascontiguousarray(calculations, dtype="<f8")
    basis_engineering = _validate_fit_basis(activations, sidecar, details, calculations)
    _write_array(runner.DEFAULT_BASIS_SIDECAR, sidecar)
    _write_f64_array(runner.DEFAULT_BASIS_CALCULATIONS, calculations)
    _write_json(runner.DEFAULT_BASIS_DETAILS, details)
    engineering_pass = basis_engineering["pass"] is True
    authorized = admitted and engineering_pass
    status = "FIT_BASIS_LOCKED_LOCALIZATION_AUTHORIZED" if authorized else "FIT_STOP_BASIS_LOCK_INVALID"
    analysis = {
        "schema_version": FIT_ANALYSIS_SCHEMA,
        "status": status,
        "claim_authority": "fit_only_basis_and_baseline_admission",
        "claim_boundaries": CLAIM_BOUNDARIES,
        "behavioral_admission": admission,
        "basis_engineering": basis_engineering,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
        "fit_execution_manifest_file_sha256": file_sha256(fit_manifest_path),
        "fit_records_file_sha256": manifest["records"]["file_sha256"],
        "fit_activations_file_sha256": manifest["activations"]["file_sha256"],
        "basis_sidecar_logical_sha256": f32_sha256(sidecar),
        "basis_calculations_logical_sha256": runner.f64_sha256(calculations),
        "basis_details_canonical_sha256": canonical_sha256(details),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "runner_sha256": design["locks"]["runner"]["sha256"],
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    _write_json(FIT_ANALYSIS_PATH, analysis)
    lock = {
        "schema_version": BASIS_LOCK_SCHEMA,
        "status": status,
        "call_plan_sha256": plan["call_plan_sha256"],
        "engineering_pass": engineering_pass,
        "behavioral_admission_pass": admitted,
        "localization_baseline_authorized": authorized,
        "fit_execution_manifest_file_sha256": file_sha256(fit_manifest_path),
        "fit_analysis": {
            "path": str(FIT_ANALYSIS_PATH),
            "file_sha256": file_sha256(FIT_ANALYSIS_PATH),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "basis_sidecar": {
            "path": str(runner.DEFAULT_BASIS_SIDECAR),
            "file_sha256": file_sha256(runner.DEFAULT_BASIS_SIDECAR),
            "logical_sha256": f32_sha256(sidecar),
            "shape": list(sidecar.shape),
            "dtype": "<f4",
        },
        "basis_calculations": {
            "path": str(runner.DEFAULT_BASIS_CALCULATIONS),
            "file_sha256": file_sha256(runner.DEFAULT_BASIS_CALCULATIONS),
            "logical_sha256": runner.f64_sha256(calculations),
            "shape": list(calculations.shape),
            "dtype": "<f8",
        },
        "basis_details": {
            "path": str(runner.DEFAULT_BASIS_DETAILS),
            "file_sha256": file_sha256(runner.DEFAULT_BASIS_DETAILS),
            "canonical_sha256": canonical_sha256(details),
        },
        "holdout_data_used": False,
        "biological_execution_allowed": False,
        "biological_model_calls": 0,
    }
    _write_json(runner.DEFAULT_BASIS_LOCK, lock)
    return analysis, lock


def analyze_localization_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    _, expected_basis_lock = analyze_fit_basis()
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    if _load_json(runner.DEFAULT_BASIS_LOCK) != expected_basis_lock:
        raise CrossCodebookAnalysisError("fit basis lock does not recompute")
    if expected_basis_lock["localization_baseline_authorized"] is not True:
        raise CrossCodebookAnalysisError("fit basis did not authorize localization")
    try:
        runner._load_basis_artifacts(plan)
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "fit_execution_manifest_file_sha256": file_sha256(runner.PHASE_PATHS["fit-baseline"]["manifest"]),
    }
    manifest, raw_baselines, activations, patched = _validate_phase_manifest(
        plan,
        design,
        phase="localization-baseline",
        expected_count=256,
        expected_activation_shape=(256, 5, 1536),
        expected_patched_activation_shape=None,
        selected_layer=None,
        prerequisite_bindings=prerequisites,
    )
    if activations is None or patched is not None:
        raise CrossCodebookAnalysisError("localization baseline sidecar changed")
    baselines = _validate_baseline_records(
        plan,
        manifest,
        raw_baselines,
        activations,
        phase="localization-baseline",
        role="localization",
        layers=LAYER_GRID,
    )
    behavioral_rows, recipient_pairs = _behavioral_inputs(plan, baselines, role="localization")
    admission = _behavioral_admission(behavioral_rows, recipient_pairs, expected_worlds=8)
    engineering_pass = True
    admitted = admission["gates"]["pass"] is True
    authorized = admitted and engineering_pass
    status = (
        "LOCALIZATION_BASELINE_ADMITTED_PATCH_AUTHORIZED" if authorized else "LOCALIZATION_BASELINE_STOP_NOT_ADMITTED"
    )
    analysis = {
        "schema_version": LOCALIZATION_BASELINE_ANALYSIS_SCHEMA,
        "status": status,
        "claim_authority": "localization_baseline_admission_only",
        "claim_boundaries": CLAIM_BOUNDARIES,
        "behavioral_admission": admission,
        "engineering_pass": engineering_pass,
        "call_plan_sha256": plan["call_plan_sha256"],
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_baseline_records_file_sha256": manifest["records"]["file_sha256"],
        "localization_activations_file_sha256": manifest["activations"]["file_sha256"],
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    _write_json(LOCALIZATION_BASELINE_ANALYSIS_PATH, analysis)
    entry = {
        "schema_version": LOCALIZATION_ENTRY_SCHEMA,
        "status": status,
        "call_plan_sha256": plan["call_plan_sha256"],
        "behavioral_admission_pass": admitted,
        "engineering_pass": engineering_pass,
        "localization_patch_authorized": authorized,
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_baseline_analysis": {
            "path": str(LOCALIZATION_BASELINE_ANALYSIS_PATH),
            "file_sha256": file_sha256(LOCALIZATION_BASELINE_ANALYSIS_PATH),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "biological_execution_allowed": False,
        "biological_model_calls": 0,
    }
    _write_json(LOCALIZATION_ENTRY_PATH, entry)
    return analysis, entry


def analyze_localization() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    baseline_analysis, expected_localization_entry = analyze_localization_baseline()
    _, expected_basis_lock = analyze_fit_basis()
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    observed_basis_lock = _load_json(runner.DEFAULT_BASIS_LOCK)
    if observed_basis_lock != expected_basis_lock:
        raise CrossCodebookAnalysisError("fit basis lock does not recompute")
    if expected_basis_lock["localization_baseline_authorized"] is not True:
        raise CrossCodebookAnalysisError("fit basis did not authorize localization")
    if _load_json(LOCALIZATION_ENTRY_PATH) != expected_localization_entry:
        raise CrossCodebookAnalysisError("localization baseline entry does not recompute")
    if expected_localization_entry["localization_patch_authorized"] is not True:
        raise CrossCodebookAnalysisError("localization baseline did not authorize patching")
    try:
        basis_sidecar, basis_details, _ = runner._load_basis_artifacts(plan)
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error

    baseline_prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "fit_execution_manifest_file_sha256": file_sha256(runner.PHASE_PATHS["fit-baseline"]["manifest"]),
    }
    baseline_manifest, raw_baselines, activations, baseline_patched = _validate_phase_manifest(
        plan,
        design,
        phase="localization-baseline",
        expected_count=256,
        expected_activation_shape=(256, 5, 1536),
        expected_patched_activation_shape=None,
        selected_layer=None,
        prerequisite_bindings=baseline_prerequisites,
    )
    if activations is None or baseline_patched is not None:
        raise CrossCodebookAnalysisError("localization activation sidecar is missing")
    baselines = _validate_baseline_records(
        plan,
        baseline_manifest,
        raw_baselines,
        activations,
        phase="localization-baseline",
        role="localization",
        layers=LAYER_GRID,
    )
    behavioral_rows, recipient_pairs = _behavioral_inputs(plan, baselines, role="localization")
    admission = _behavioral_admission(behavioral_rows, recipient_pairs, expected_worlds=8)
    if admission != baseline_analysis["behavioral_admission"]:
        raise CrossCodebookAnalysisError("localization baseline admission does not recompute")
    patch_prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_baseline_records_file_sha256": baseline_manifest["records"]["file_sha256"],
        "localization_entry_file_sha256": file_sha256(LOCALIZATION_ENTRY_PATH),
    }
    patch_manifest, raw_patches, no_patch_activations, patched_activations = _validate_phase_manifest(
        plan,
        design,
        phase="localization-patch",
        expected_count=4480,
        expected_activation_shape=None,
        expected_patched_activation_shape=(4480, 1536),
        selected_layer=None,
        prerequisite_bindings=patch_prerequisites,
    )
    if no_patch_activations is not None or patched_activations is None:
        raise CrossCodebookAnalysisError("localization patch bound extra activations")
    patches, artifact_gates = _validate_patch_records(
        plan,
        raw_patches,
        baselines,
        activations,
        patched_activations,
        basis_sidecar,
        basis_details,
        phase="localization-patch",
        role="localization",
        selected_layer=None,
    )
    identity = _identity_engineering(baselines, patches)
    engineering_pass = bool(admission["gates"]["pass"] and identity["pass"] and all(artifact_gates.values()))
    layer_metrics = {
        layer: _localization_metrics(
            _localization_rows_from_artifacts(plan, baselines, patches, layer=layer),
            layer=layer,
        )
        for layer in LAYER_GRID
    }
    decisions = {layer: _localization_decision(metrics) for layer, metrics in layer_metrics.items()}
    selected_layer, status = _select_localization_layer(decisions, engineering_pass=engineering_pass)
    analysis = {
        "schema_version": LOCALIZATION_ANALYSIS_SCHEMA,
        "status": status,
        "claim_authority": "earliest_preregistered_grid_layer_only",
        "claim_boundaries": CLAIM_BOUNDARIES,
        "selected_layer": selected_layer,
        "behavioral_admission": admission,
        "identity_engineering": identity,
        "artifact_engineering": artifact_gates,
        "engineering_pass": engineering_pass,
        "layer_decisions": {str(layer): decisions[layer] for layer in LAYER_GRID},
        "layer_metrics": {str(layer): layer_metrics[layer] for layer in LAYER_GRID},
        "selection_rule": "earliest_passing_layer_in_[8,12,16,20,24]",
        "call_plan_sha256": plan["call_plan_sha256"],
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_entry_file_sha256": file_sha256(LOCALIZATION_ENTRY_PATH),
        "localization_patch_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-patch"]["manifest"]
        ),
        "localization_patch_records_file_sha256": patch_manifest["records"]["file_sha256"],
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    _write_json(LOCALIZATION_ANALYSIS_PATH, analysis)
    lock = {
        "schema_version": LAYER_LOCK_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "engineering_pass": engineering_pass,
        "behavioral_admission_pass": admission["gates"]["pass"],
        "holdout_baseline_authorized": selected_layer is not None and engineering_pass,
        "call_plan_sha256": plan["call_plan_sha256"],
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_entry_file_sha256": file_sha256(LOCALIZATION_ENTRY_PATH),
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_patch_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["localization-patch"]["manifest"]
        ),
        "localization_analysis": {
            "path": str(LOCALIZATION_ANALYSIS_PATH),
            "file_sha256": file_sha256(LOCALIZATION_ANALYSIS_PATH),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "layer_grid": list(LAYER_GRID),
        "holdout_patch_authorized": False,
        "biological_execution_allowed": False,
        "biological_model_calls": 0,
    }
    _write_json(runner.DEFAULT_LAYER_LOCK, lock)
    return analysis, lock


def analyze_holdout_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    _, expected_layer_lock = analyze_localization()
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    if _load_json(runner.DEFAULT_LAYER_LOCK) != expected_layer_lock:
        raise CrossCodebookAnalysisError("localization layer lock does not recompute")
    if expected_layer_lock["holdout_baseline_authorized"] is not True:
        raise CrossCodebookAnalysisError("localization did not authorize holdout")
    try:
        runner._load_basis_artifacts(plan)
        selected_layer, _ = runner._load_layer_lock(plan)
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
    }
    manifest, raw_baselines, activations, baseline_patched = _validate_phase_manifest(
        plan,
        design,
        phase="holdout-baseline",
        expected_count=1024,
        expected_activation_shape=(1024, 1536),
        expected_patched_activation_shape=None,
        selected_layer=selected_layer,
        prerequisite_bindings=prerequisites,
    )
    if activations is None or baseline_patched is not None:
        raise CrossCodebookAnalysisError("holdout activation sidecar is missing")
    baselines = _validate_baseline_records(
        plan,
        manifest,
        raw_baselines,
        activations,
        phase="holdout-baseline",
        role="holdout",
        layers=(selected_layer,),
    )
    behavioral_rows, recipient_pairs = _behavioral_inputs(plan, baselines, role="holdout")
    admission = _behavioral_admission(behavioral_rows, recipient_pairs, expected_worlds=32)
    admitted = admission["gates"]["pass"] is True
    status = "HOLDOUT_BASELINE_ADMITTED_PATCH_AUTHORIZED" if admitted else "HOLDOUT_BASELINE_STOP_NOT_ADMITTED"
    analysis = {
        "schema_version": HOLDOUT_BASELINE_ANALYSIS_SCHEMA,
        "status": status,
        "claim_authority": "holdout_baseline_admission_only",
        "claim_boundaries": CLAIM_BOUNDARIES,
        "selected_layer": selected_layer,
        "behavioral_admission": admission,
        "call_plan_sha256": plan["call_plan_sha256"],
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["holdout-baseline"]["manifest"]
        ),
        "holdout_baseline_records_file_sha256": manifest["records"]["file_sha256"],
        "holdout_activations_file_sha256": manifest["activations"]["file_sha256"],
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    _write_json(HOLDOUT_BASELINE_ANALYSIS_PATH, analysis)
    entry = {
        "schema_version": HOLDOUT_ENTRY_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "behavioral_admission_pass": admitted,
        "holdout_patch_authorized": admitted,
        "call_plan_sha256": plan["call_plan_sha256"],
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["holdout-baseline"]["manifest"]
        ),
        "holdout_baseline_analysis": {
            "path": str(HOLDOUT_BASELINE_ANALYSIS_PATH),
            "file_sha256": file_sha256(HOLDOUT_BASELINE_ANALYSIS_PATH),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "biological_execution_allowed": False,
        "biological_model_calls": 0,
    }
    _write_json(runner.DEFAULT_HOLDOUT_ENTRY, entry)
    return analysis, entry


def analyze_final() -> dict[str, Any]:
    plan, design = _load_plan()
    holdout_analysis, expected_entry = analyze_holdout_baseline()
    if runner is None:
        raise CrossCodebookAnalysisError("v3 runner module is unavailable")
    if _load_json(runner.DEFAULT_HOLDOUT_ENTRY) != expected_entry:
        raise CrossCodebookAnalysisError("holdout entry does not recompute")
    if expected_entry["holdout_patch_authorized"] is not True:
        raise CrossCodebookAnalysisError("holdout baseline did not authorize patching")
    try:
        basis_sidecar, basis_details, _ = runner._load_basis_artifacts(plan)
        selected_layer, _ = runner._load_holdout_entry(plan)
    except (runner.CrossCodebookRunnerError, OSError, KeyError, ValueError) as error:
        raise CrossCodebookAnalysisError(str(error)) from error
    baseline_prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
    }
    baseline_manifest, raw_baselines, activations, baseline_patched = _validate_phase_manifest(
        plan,
        design,
        phase="holdout-baseline",
        expected_count=1024,
        expected_activation_shape=(1024, 1536),
        expected_patched_activation_shape=None,
        selected_layer=selected_layer,
        prerequisite_bindings=baseline_prerequisites,
    )
    if activations is None or baseline_patched is not None:
        raise CrossCodebookAnalysisError("holdout activation sidecar is missing")
    baselines = _validate_baseline_records(
        plan,
        baseline_manifest,
        raw_baselines,
        activations,
        phase="holdout-baseline",
        role="holdout",
        layers=(selected_layer,),
    )
    patch_prerequisites = {
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        "holdout_entry_file_sha256": file_sha256(runner.DEFAULT_HOLDOUT_ENTRY),
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["holdout-baseline"]["manifest"]
        ),
        "holdout_baseline_records_file_sha256": baseline_manifest["records"]["file_sha256"],
    }
    patch_manifest, raw_patches, no_patch_activations, patched_activations = _validate_phase_manifest(
        plan,
        design,
        phase="holdout-patch",
        expected_count=5376,
        expected_activation_shape=None,
        expected_patched_activation_shape=(5376, 1536),
        selected_layer=selected_layer,
        prerequisite_bindings=patch_prerequisites,
    )
    if no_patch_activations is not None or patched_activations is None:
        raise CrossCodebookAnalysisError("holdout patch bound extra activations")
    patches, artifact_gates = _validate_patch_records(
        plan,
        raw_patches,
        baselines,
        activations,
        patched_activations,
        basis_sidecar,
        basis_details,
        phase="holdout-patch",
        role="holdout",
        selected_layer=selected_layer,
    )
    rows = _final_rows_from_artifacts(plan, baselines, patches, artifact_gates)
    metrics = _final_metrics(rows)
    gates = _final_gates(metrics)
    holdout_admission_pass = (
        holdout_analysis["behavioral_admission"]["gates"]["pass"] is True
    )
    gates["engineering"]["behavioral_admission_pass"] = holdout_admission_pass
    gates["engineering"]["pass"] = bool(
        gates["engineering"]["pass"] and holdout_admission_pass
    )
    status = _final_status(
        primary_pass=gates["primary"]["pass"],
        specificity_pass=gates["specificity"]["pass"],
        natural_use_pass=gates["natural_use"]["pass"],
        engineering_pass=gates["engineering"]["pass"],
    )
    analysis = {
        "schema_version": FINAL_ANALYSIS_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "maximum_claim": (
            "synthetic symbolic projected-content recomposition in one locked model; "
            "natural-use language only when the natural-use tier passes"
        ),
        "holdout_behavioral_admission": holdout_analysis["behavioral_admission"],
        "artifact_engineering": artifact_gates,
        "metrics": metrics,
        "gates": gates,
        "normalized_row_registry_canonical_sha256": canonical_sha256(rows),
        "n_normalized_recipients": len(rows),
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        "holdout_entry_file_sha256": file_sha256(runner.DEFAULT_HOLDOUT_ENTRY),
        "holdout_patch_execution_manifest_file_sha256": file_sha256(runner.PHASE_PATHS["holdout-patch"]["manifest"]),
        "holdout_patch_records_file_sha256": patch_manifest["records"]["file_sha256"],
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "runner_sha256": design["locks"]["runner"]["sha256"],
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    input_bindings = {
        name: {"path": str(path), "file_sha256": file_sha256(path)}
        for name, path in {
            "design": runner.DEFAULT_DESIGN,
            "plan_manifest": runner.DEFAULT_PLAN_MANIFEST,
            "tokenization_receipt": runner.DEFAULT_TOKENIZATION_RECEIPT,
            "dependency_lock": runner.DEFAULT_DEPENDENCY_LOCK,
            "fit_execution_manifest": runner.PHASE_PATHS["fit-baseline"]["manifest"],
            "fit_records": runner.PHASE_PATHS["fit-baseline"]["records"],
            "fit_activations": runner.PHASE_PATHS["fit-baseline"]["activations"],
            "fit_analysis": FIT_ANALYSIS_PATH,
            "basis_lock": runner.DEFAULT_BASIS_LOCK,
            "basis_sidecar": runner.DEFAULT_BASIS_SIDECAR,
            "basis_calculations": runner.DEFAULT_BASIS_CALCULATIONS,
            "basis_details": runner.DEFAULT_BASIS_DETAILS,
            "localization_baseline_execution_manifest": runner.PHASE_PATHS["localization-baseline"]["manifest"],
            "localization_baseline_records": runner.PHASE_PATHS["localization-baseline"]["records"],
            "localization_activations": runner.PHASE_PATHS["localization-baseline"]["activations"],
            "localization_baseline_analysis": LOCALIZATION_BASELINE_ANALYSIS_PATH,
            "localization_entry": LOCALIZATION_ENTRY_PATH,
            "localization_patch_execution_manifest": runner.PHASE_PATHS["localization-patch"]["manifest"],
            "localization_patch_records": runner.PHASE_PATHS["localization-patch"]["records"],
            "localization_patched_activations": runner.PHASE_PATHS["localization-patch"]["patched_activations"],
            "localization_analysis": LOCALIZATION_ANALYSIS_PATH,
            "layer_lock": runner.DEFAULT_LAYER_LOCK,
            "holdout_baseline_execution_manifest": runner.PHASE_PATHS["holdout-baseline"]["manifest"],
            "holdout_baseline_records": runner.PHASE_PATHS["holdout-baseline"]["records"],
            "holdout_activations": runner.PHASE_PATHS["holdout-baseline"]["activations"],
            "holdout_baseline_analysis": HOLDOUT_BASELINE_ANALYSIS_PATH,
            "holdout_entry": runner.DEFAULT_HOLDOUT_ENTRY,
            "holdout_patch_execution_manifest": runner.PHASE_PATHS["holdout-patch"]["manifest"],
            "holdout_patch_records": runner.PHASE_PATHS["holdout-patch"]["records"],
            "holdout_patched_activations": runner.PHASE_PATHS["holdout-patch"]["patched_activations"],
        }.items()
    }
    _write_analysis_bundle(
        analysis,
        json_path=FINAL_ANALYSIS_PATH,
        markdown_path=FINAL_MARKDOWN_PATH,
        manifest_path=FINAL_MANIFEST_PATH,
        input_bindings=input_bindings,
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "fit_basis",
            "localization_baseline",
            "localization",
            "holdout_baseline",
            "final",
        ),
    )
    args = parser.parse_args()
    actions = {
        "fit_basis": analyze_fit_basis,
        "localization_baseline": analyze_localization_baseline,
        "localization": analyze_localization,
        "holdout_baseline": analyze_holdout_baseline,
        "final": analyze_final,
    }
    result = actions[args.stage]()
    status = result[0]["status"] if isinstance(result, tuple) else result["status"]
    print(json.dumps({"stage": args.stage, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Analyze the preregistered V5 causal TARGET-order/context experiment.

The analyzer never executes a language model.  It independently verifies the
runner's immutable files, reconstructs behavioral and intervention summaries,
and is the sole issuer of stage authorization artifacts.

The strongest permitted conclusion is deliberately narrow: selective causal
mediation by a TARGET-order/preceding-context state at one TARGET-property
token, in one frozen synthetic prompt/model setting.  It does not separate
absolute position from preceding-OTHER interference and cannot support claims about
biology, latent biological knowledge, general activation gaps, or physical
laws.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import run_coherent_readout_v5_positional_activation as runner
except ImportError:  # pragma: no cover - direct execution / concurrent landing
    try:
        import run_coherent_readout_v5_positional_activation as runner
    except ImportError:  # pragma: no cover
        runner = None  # type: ignore[assignment]


FIT_ANALYSIS_SCHEMA = "coherent-readout-v5-positional-fit-analysis-v1"
BASIS_LOCK_SCHEMA = "coherent-readout-v5-positional-basis-lock-v1"
LOCALIZATION_BASELINE_SCHEMA = "coherent-readout-v5-positional-localization-baseline-v1"
LOCALIZATION_BASELINE_ENTRY_SCHEMA = "coherent-readout-v5-positional-localization-baseline-entry-v1"
LOCALIZATION_PATCH_ENTRY_SCHEMA = "coherent-readout-v5-positional-localization-patch-entry-v1"
LOCALIZATION_ANALYSIS_SCHEMA = "coherent-readout-v5-positional-localization-analysis-v1"
LAYER_LOCK_SCHEMA = "coherent-readout-v5-positional-layer-lock-v1"
HOLDOUT_BASELINE_SCHEMA = "coherent-readout-v5-positional-holdout-baseline-v1"
HOLDOUT_BASELINE_ENTRY_SCHEMA = "coherent-readout-v5-positional-holdout-baseline-entry-v1"
HOLDOUT_PATCH_ENTRY_SCHEMA = "coherent-readout-v5-positional-holdout-patch-entry-v1"
FINAL_ANALYSIS_SCHEMA = "coherent-readout-v5-positional-final-analysis-v1"
ANALYSIS_MANIFEST_SCHEMA = "coherent-readout-v5-positional-analysis-manifest-v1"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260803
DOSE_MATCH_TOLERANCE = 1e-5
LAYER_GRID = (12, 16, 20, 24)
FIT_WORLDS = 8
LOCALIZATION_WORLDS = 8
HOLDOUT_WORLDS = 16
FACTOR_NAMES = ("p", "m", "r", "v", "o")
NUISANCE_NAMES = ("p", "m", "r", "v", "pm")
PATCH_CONDITIONS = (
    "positional_rescue",
    "positional_damage",
    "answer_rescue_sham",
    "answer_damage_sham",
    "null_rescue_sham",
    "null_damage_sham",
    "identity",
)

BASIS_THRESHOLDS = {
    "minimum_residual_ratio": 0.10,
    "maximum_pairwise_absolute_cosine": 0.35,
    "minimum_split_half_reliability": 0.25,
    "minimum_leave_one_world_out_reliability": 0.25,
    "unit_norm_tolerance": 1e-5,
}
BEHAVIOR_THRESHOLDS = {
    "retrieval_lookup_overall_accuracy": 0.95,
    "retrieval_lookup_world_accuracy": 0.90,
    "retrieval_lookup_stratum_accuracy": 0.90,
    "target_first_composition_accuracy": 0.95,
    "target_second_error_rate": 0.25,
    "composition_accuracy_gap": 0.10,
}
CAUSAL_THRESHOLDS = {
    "minimum_effect_over_G": 0.30,
    "minimum_specificity_over_G": 0.20,
    "localization_positive_worlds": 6,
    "holdout_positive_worlds": 12,
    "identity_margin_tolerance": 1e-4,
}

FINAL_STATUSES = (
    "V5_NO_REPLICATED_CAUSAL_GAP_CLOSURE",
    "V5_NONSPECIFIC_CAUSAL_GAP_CLOSURE",
    "V5_CAUSAL_GAP_CLOSURE_NATURAL_USE_NOT_ESTABLISHED",
    "V5_CAUSAL_TARGET_ORDER_CONTEXT_ACTIVATION_GAP_SUPPORTED",
)
STOP_STATUSES = (
    "V5_ENGINEERING_INVALID",
    "V5_FIT_COMPONENT_ADMISSION_FAIL",
    "V5_FIT_BASIS_INVALID",
    "V5_LOCALIZATION_COMPONENT_ADMISSION_FAIL",
    "V5_LOCALIZATION_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED",
    "V5_LOCALIZATION_ENGINEERING_INVALID",
    "V5_NO_PREREGISTERED_CAUSAL_LAYER",
    "V5_HOLDOUT_COMPONENT_ADMISSION_FAIL",
    "V5_HOLDOUT_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED",
    "V5_FINAL_ENGINEERING_INVALID",
)
ALL_STATUSES = STOP_STATUSES + FINAL_STATUSES

CLAIM_BOUNDARIES = {
    "supported_scope": (
        "synthetic_model_prompt_target_property_token_and_selected_layer_specific_"
        "causal_target_order_context_state_mediation"
    ),
    "biology_inference": "forbidden",
    "latent_biological_knowledge_inference": "forbidden",
    "general_activation_gap_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
    "absolute_position_vs_preceding_other_interference": "not_separated",
    "natural_gap_is_behaviorally_preexisting": True,
    "intervention_is_selective_rank_one_coordinate_replacement": True,
}

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    getattr(runner, "RESULT_ROOT", None)
    if runner is not None
    else None
) or (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v5_positional_activation"
    / "qwen2.5-1.5b-instruct"
)
FIT_ANALYSIS_PATH = RESULT_ROOT / "fit_analysis.json"
LOCALIZATION_BASELINE_ANALYSIS_PATH = RESULT_ROOT / "localization_baseline_analysis.json"
LOCALIZATION_ANALYSIS_PATH = RESULT_ROOT / "localization_analysis.json"
HOLDOUT_BASELINE_ANALYSIS_PATH = RESULT_ROOT / "holdout_baseline_analysis.json"
FINAL_ANALYSIS_PATH = RESULT_ROOT / "analysis.json"
FINAL_MARKDOWN_PATH = RESULT_ROOT / "analysis.md"
FINAL_MANIFEST_PATH = RESULT_ROOT / "analysis_manifest.json"


class PositionalActivationAnalysisError(ValueError):
    """Raised when a frozen artifact or preregistered analysis contract fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PositionalActivationAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PositionalActivationAnalysisError(f"{label} must be finite")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PositionalActivationAnalysisError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PositionalActivationAnalysisError(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise PositionalActivationAnalysisError(f"cannot read JSONL: {path}") from error
    if not values or any(not isinstance(value, dict) for value in values):
        raise PositionalActivationAnalysisError(f"JSONL is empty or contains non-objects: {path}")
    return values


def _write_frozen_bytes(path: Path, payload: bytes) -> None:
    """Create an analysis artifact without replacing an existing byte sequence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PositionalActivationAnalysisError(f"refusing to overwrite frozen artifact: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_frozen_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _write_array(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise PositionalActivationAnalysisError("refusing to freeze a nonfinite array")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _write_frozen_bytes(path, buffer.getvalue())


def _validate_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PositionalActivationAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _relative_artifact_path(manifest_path: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise PositionalActivationAnalysisError("artifact path is missing")
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def verify_file_binding(manifest_path: Path, binding: Mapping[str, Any]) -> Path:
    """Verify path, byte count, and SHA-256 for one immutable file binding."""

    if not isinstance(binding, Mapping):
        raise PositionalActivationAnalysisError("file binding must be an object")
    if not {"path", "sha256"}.issubset(binding):
        raise PositionalActivationAnalysisError("file binding lacks path or sha256")
    path = _relative_artifact_path(manifest_path, binding["path"])
    if not path.is_file():
        raise PositionalActivationAnalysisError(f"bound artifact does not exist: {path}")
    expected_hash = _validate_hash(binding["sha256"], f"hash for {path}")
    if file_sha256(path) != expected_hash:
        raise PositionalActivationAnalysisError(f"artifact hash mismatch: {path}")
    if "bytes" in binding:
        size = binding["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size != path.stat().st_size:
            raise PositionalActivationAnalysisError(f"artifact byte count mismatch: {path}")
    return path


def verify_array_binding(manifest_path: Path, binding: Mapping[str, Any]) -> np.ndarray:
    """Verify and load a raw NumPy shard, including its declared dtype and shape."""

    path = verify_file_binding(manifest_path, binding)
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PositionalActivationAnalysisError(f"cannot load bound array: {path}") from error
    if "shape" not in binding or "dtype" not in binding:
        raise PositionalActivationAnalysisError("array binding lacks dtype or shape")
    shape = binding["shape"]
    if (
        not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        or list(array.shape) != shape
        or str(array.dtype) != binding["dtype"]
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise PositionalActivationAnalysisError(f"bound array metadata/content mismatch: {path}")
    return array


def verify_phase_manifest(manifest_path: Path) -> dict[str, Any]:
    """Independently verify all files and canonical record commitments in a phase."""

    manifest = _load_json(manifest_path)
    if manifest.get("status") not in {"COMPLETE", "EXECUTION_COMPLETE_IMMUTABLE"}:
        raise PositionalActivationAnalysisError("phase manifest is not complete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or "records" not in artifacts:
        raise PositionalActivationAnalysisError("phase manifest artifact registry is invalid")
    records_path = verify_file_binding(manifest_path, artifacts["records"])
    records = _load_jsonl(records_path)
    if len(records) != manifest.get("record_count"):
        raise PositionalActivationAnalysisError("record count does not reconstruct")
    if len({row.get("record_id") for row in records}) != len(records) or any(
        not isinstance(row.get("record_id"), str) or not row["record_id"] for row in records
    ):
        raise PositionalActivationAnalysisError("record IDs are missing or duplicated")
    expected_records_hash = manifest.get("records_canonical_sha256")
    if expected_records_hash is not None:
        _validate_hash(expected_records_hash, "canonical record hash")
        if canonical_sha256(records) != expected_records_hash:
            raise PositionalActivationAnalysisError("canonical record hash mismatch")
    arrays: dict[str, np.ndarray] = {}
    for name, binding in artifacts.items():
        if name == "records":
            continue
        if not isinstance(binding, Mapping):
            raise PositionalActivationAnalysisError(f"invalid binding for {name}")
        arrays[name] = (
            verify_array_binding(manifest_path, binding)
            if "shape" in binding or "dtype" in binding
            else np.asarray([], dtype=np.float32)
        )
        if not ("shape" in binding or "dtype" in binding):
            verify_file_binding(manifest_path, binding)
    declared_calls = manifest.get("model_calls")
    if isinstance(declared_calls, bool) or not isinstance(declared_calls, int) or declared_calls != len(records):
        raise PositionalActivationAnalysisError("model-call accounting does not equal record count")
    if manifest.get("generation_used") is not False or manifest.get("biological_model_calls") != 0:
        raise PositionalActivationAnalysisError("generation or biological model calls are forbidden")
    return {"manifest": manifest, "records": records, "arrays": arrays, "manifest_path": manifest_path}


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise PositionalActivationAnalysisError(f"{label} vector is invalid")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise PositionalActivationAnalysisError(f"{label} vector has zero/invalid norm")
    return value / norm


def _residualize(vector: np.ndarray, against: Iterable[np.ndarray]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    columns = [np.asarray(column, dtype=np.float64) for column in against]
    if not columns:
        return value.copy()
    matrix = np.column_stack(columns)
    if matrix.ndim != 2 or matrix.shape[0] != value.shape[0] or not np.isfinite(matrix).all():
        raise PositionalActivationAnalysisError("residualization matrix is invalid")
    projection = matrix @ np.linalg.pinv(matrix, rcond=1e-6) @ value
    return value - projection


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(_unit(first, "first") @ _unit(second, "second"))


def _factor_sign(value: Any, label: str) -> int:
    if value in (-1, 1):
        return int(value)
    raise PositionalActivationAnalysisError(f"{label} must be -1 or +1")


def factorial_coefficients(
    activations: np.ndarray,
    factor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Return exact 2^5 Walsh coefficients for one world's 32 cells.

    ``activations`` has shape ``(32, hidden_width)``.  Rows may arrive in any
    order, but their five sign tuples must be the complete, unique factorial.
    """

    array = np.asarray(activations, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 32 or not np.isfinite(array).all():
        raise PositionalActivationAnalysisError("factorial activations must have shape (32,width)")
    if len(factor_rows) != 32:
        raise PositionalActivationAnalysisError("factor row count must be 32")
    signs = np.empty((32, 5), dtype=np.int8)
    tuples: set[tuple[int, ...]] = set()
    for index, row in enumerate(factor_rows):
        sign_tuple = tuple(_factor_sign(row.get(name), name) for name in FACTOR_NAMES)
        if sign_tuple in tuples:
            raise PositionalActivationAnalysisError("factorial sign tuple is duplicated")
        tuples.add(sign_tuple)
        signs[index] = sign_tuple
    expected = {
        (p, m, r, v, o)
        for p in (-1, 1)
        for m in (-1, 1)
        for r in (-1, 1)
        for v in (-1, 1)
        for o in (-1, 1)
    }
    if tuples != expected:
        raise PositionalActivationAnalysisError("factor rows do not form the complete 2^5 design")
    coefficient_signs = {
        "intercept": np.ones(32, dtype=np.int8),
        "p": signs[:, 0],
        "m": signs[:, 1],
        "r": signs[:, 2],
        "v": signs[:, 3],
        "o": signs[:, 4],
        "pm": signs[:, 0] * signs[:, 1],
    }
    return {
        name: np.mean(array * contrast[:, None], axis=0)
        for name, contrast in coefficient_signs.items()
    }


def _aligned_reliability(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_unit = _unit(candidate, "reliability candidate")
    reference_unit = _unit(reference, "reliability reference")
    cosine = float(candidate_unit @ reference_unit)
    return abs(cosine)


def construct_layer_basis(
    world_activations: Mapping[str, np.ndarray],
    world_factor_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Construct positional, answer, and deterministic null directions.

    Each activation block must be ``(32,width)`` for one fit world and one
    layer.  The positional coefficient is the pooled order Walsh coefficient,
    residualized only against the preregistered p, m, r, v, and p*m axes.
    """

    worlds = sorted(world_activations)
    if len(worlds) != FIT_WORLDS or set(world_factor_rows) != set(worlds):
        raise PositionalActivationAnalysisError("basis construction requires exactly eight matching fit worlds")
    coefficients = {
        world: factorial_coefficients(world_activations[world], world_factor_rows[world])
        for world in worlds
    }
    width = {coefficient["o"].shape for coefficient in coefficients.values()}
    if len(width) != 1:
        raise PositionalActivationAnalysisError("hidden width differs between fit worlds")

    pooled = {
        name: np.mean([coefficients[world][name] for world in worlds], axis=0)
        for name in ("p", "m", "r", "v", "o", "pm")
    }
    center = np.mean([coefficients[world]["intercept"] for world in worlds], axis=0)
    positional_raw = pooled["o"]
    positional_residual = _residualize(positional_raw, [pooled[name] for name in NUISANCE_NAMES])
    answer_raw = pooled["pm"]
    answer_residual = _residualize(
        answer_raw,
        [pooled[name] for name in ("p", "m", "r", "v")] + [positional_raw],
    )

    # A fixed nonconstant row of the order-8 Sylvester-Hadamard matrix.  World
    # IDs are sorted before applying it, so this null cannot be data-selected.
    null_signs = np.asarray([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    null_raw = np.mean(
        [sign * coefficients[world]["o"] for sign, world in zip(null_signs, worlds, strict=True)],
        axis=0,
    )
    null_residual = _residualize(
        null_raw,
        [pooled[name] for name in ("p", "m", "r", "v", "pm", "o")],
    )

    raw_and_residual = {
        "positional": (positional_raw, positional_residual),
        "answer": (answer_raw, answer_residual),
        "null": (null_raw, null_residual),
    }
    directions: dict[str, np.ndarray] = {}
    residual_ratios: dict[str, float] = {}
    construction_errors: list[str] = []
    for name, (raw, residual) in raw_and_residual.items():
        raw_norm = float(np.linalg.norm(raw))
        residual_norm = float(np.linalg.norm(residual))
        residual_ratios[name] = residual_norm / raw_norm if raw_norm > 0.0 else 0.0
        try:
            directions[name] = _unit(residual, name)
        except PositionalActivationAnalysisError as error:
            construction_errors.append(str(error))
            directions[name] = np.zeros_like(residual, dtype=np.float64)

    valid_direction_names = [name for name, value in directions.items() if np.linalg.norm(value) > 0.0]
    pairwise_cosines: dict[str, float] = {}
    for first_index, first in enumerate(("positional", "answer", "null")):
        for second in ("positional", "answer", "null")[first_index + 1 :]:
            key = f"{first}__{second}"
            pairwise_cosines[key] = (
                abs(_cosine(directions[first], directions[second]))
                if first in valid_direction_names and second in valid_direction_names
                else 1.0
            )

    def subset_position(world_subset: Sequence[str]) -> np.ndarray:
        subset_pooled = {
            name: np.mean([coefficients[world][name] for world in world_subset], axis=0)
            for name in ("p", "m", "r", "v", "o", "pm")
        }
        return _residualize(
            subset_pooled["o"],
            [subset_pooled[name] for name in NUISANCE_NAMES],
        )

    even_raw = subset_position(worlds[::2])
    odd_raw = subset_position(worlds[1::2])
    try:
        split_half = _aligned_reliability(even_raw, odd_raw)
    except PositionalActivationAnalysisError:
        split_half = 0.0
    loo_reliabilities: dict[str, float] = {}
    for held_out in worlds:
        loo = subset_position([world for world in worlds if world != held_out])
        try:
            loo_reliabilities[held_out] = _aligned_reliability(loo, positional_residual)
        except PositionalActivationAnalysisError:
            loo_reliabilities[held_out] = 0.0

    gates = {
        "finite_nonzero_directions": not construction_errors
        and all(np.isfinite(value).all() and np.linalg.norm(value) > 0.0 for value in directions.values()),
        "unit_norms": all(
            abs(float(np.linalg.norm(value)) - 1.0) <= BASIS_THRESHOLDS["unit_norm_tolerance"]
            for value in directions.values()
        ),
        "residual_ratios": all(
            value >= BASIS_THRESHOLDS["minimum_residual_ratio"] for value in residual_ratios.values()
        ),
        "pairwise_separation": all(
            value <= BASIS_THRESHOLDS["maximum_pairwise_absolute_cosine"]
            for value in pairwise_cosines.values()
        ),
        "split_half_reliability": split_half
        >= BASIS_THRESHOLDS["minimum_split_half_reliability"],
        "leave_one_world_out_reliability": min(loo_reliabilities.values())
        >= BASIS_THRESHOLDS["minimum_leave_one_world_out_reliability"],
    }
    gates["pass"] = all(gates.values())
    return {
        "world_ids": worlds,
        "center": center.astype("<f4"),
        "directions": {name: value.astype("<f4") for name, value in directions.items()},
        "raw_sha256": {name: f32_sha256(pair[0]) for name, pair in raw_and_residual.items()},
        "direction_sha256": {name: f32_sha256(value) for name, value in directions.items()},
        "residual_ratios": residual_ratios,
        "pairwise_absolute_cosines": pairwise_cosines,
        "split_half_absolute_cosine": split_half,
        "leave_one_world_out_absolute_cosines": loo_reliabilities,
        "construction_errors": construction_errors,
        "gates": gates,
    }


def _bootstrap_indices(n_worlds: int, *, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    if n_worlds < 2 or draws <= 0:
        raise PositionalActivationAnalysisError("bootstrap requires at least two worlds and positive draws")
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_worlds, size=(draws, n_worlds))


def _interval(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise PositionalActivationAnalysisError("bootstrap distribution is invalid")
    return {
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
        "interpretation": "finite_registered_world_panel_stability_not_population_generalization",
    }


def effect_summary(
    world_values: Mapping[str, float],
    world_gaps: Mapping[str, float],
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    worlds = sorted(world_values)
    if worlds != sorted(world_gaps) or len(worlds) < 2:
        raise PositionalActivationAnalysisError("effect and G must share at least two worlds")
    values = np.asarray([_finite(world_values[world], world) for world in worlds], dtype=np.float64)
    gaps = np.asarray([_finite(world_gaps[world], world) for world in worlds], dtype=np.float64)
    indices = _bootstrap_indices(len(worlds)) if bootstrap_indices is None else bootstrap_indices
    if indices.ndim != 2 or indices.shape[1] != len(worlds):
        raise PositionalActivationAnalysisError("bootstrap index dimensions changed")
    samples = values[indices].mean(axis=1)
    gap_samples = gaps[indices].mean(axis=1)
    mean = float(values.mean())
    gap_mean = float(gaps.mean())
    ratio_defined = bool(gap_mean > 0.0)
    bootstrap_ratio_valid = bool(ratio_defined and np.all(gap_samples > 0.0))
    ratio_samples = samples / gap_samples if bootstrap_ratio_valid else None
    return {
        "world_values": dict(zip(worlds, values.tolist(), strict=True)),
        "gap_world_values": dict(zip(worlds, gaps.tolist(), strict=True)),
        "n_worlds": len(worlds),
        "mean": mean,
        "gap_mean": gap_mean,
        "positive_world_count": int(np.sum(values > 0.0)),
        "positive_world_fraction": float(np.mean(values > 0.0)),
        "bootstrap_95": _interval(samples),
        "bootstrap_G_95": _interval(gap_samples),
        "mean_over_G": float(mean / gap_mean) if ratio_defined else None,
        "bootstrap_ratio_95": _interval(ratio_samples) if ratio_samples is not None else None,
        "ratio_defined": ratio_defined,
        "bootstrap_ratio_valid": bootstrap_ratio_valid,
    }


def _mean_by_world(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        world = row.get("world_id")
        if not isinstance(world, str) or not world:
            raise PositionalActivationAnalysisError("row world_id is invalid")
        grouped[world].append(_finite(row.get(value_key), value_key))
    if not grouped:
        raise PositionalActivationAnalysisError("world aggregation is empty")
    return {world: float(np.mean(values)) for world, values in sorted(grouped.items())}


def _diagnostic_margin(row: Mapping[str, Any]) -> float:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise PositionalActivationAnalysisError("row diagnostics are missing")
    if "expected_logit" in diagnostics and "distractor_logit" in diagnostics:
        expected = _finite(diagnostics["expected_logit"], "expected_logit")
        distractor = _finite(diagnostics["distractor_logit"], "distractor_logit")
        margin = expected - distractor
        if "expected_minus_distractor_margin" in diagnostics and abs(
            _finite(diagnostics["expected_minus_distractor_margin"], "stored margin") - margin
        ) > 1e-7:
            raise PositionalActivationAnalysisError("stored expected-minus-distractor margin is inconsistent")
        return margin
    if "correct_logit" in diagnostics and "other_logit" in diagnostics:
        return _finite(diagnostics["correct_logit"], "correct_logit") - _finite(
            diagnostics["other_logit"], "other_logit"
        )
    token_logits = diagnostics.get("answer_token_logits")
    correct_id = row.get("expected_token_id")
    other_id = row.get("other_token_id", row.get("distractor_token_id"))
    if isinstance(token_logits, Mapping):
        try:
            return _finite(token_logits[str(correct_id)], "correct token logit") - _finite(
                token_logits[str(other_id)], "other token logit"
            )
        except KeyError as error:
            raise PositionalActivationAnalysisError("answer-token logits are incomplete") from error
    raise PositionalActivationAnalysisError("diagnostics do not expose an oriented two-code margin")


def _diagnostic_correct(row: Mapping[str, Any]) -> bool:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise PositionalActivationAnalysisError("row diagnostics are missing")
    expected = row.get("expected_token_id")
    if "answer_correct" in diagnostics:
        answer_correct = diagnostics["answer_correct"]
        if not isinstance(answer_correct, bool):
            raise PositionalActivationAnalysisError("answer_correct must be Boolean")
        predicted = diagnostics.get("predicted_token_id")
        if answer_correct != (predicted == expected):
            raise PositionalActivationAnalysisError("answer_correct disagrees with predicted token")
        return answer_correct
    maxima = diagnostics.get("maximum_token_ids")
    if isinstance(maxima, list):
        return maxima == [expected]
    greedy = diagnostics.get("greedy_token_id")
    ties = diagnostics.get("maximum_tie_count", 1)
    return greedy == expected and ties == 1


def diagnostics_from_full_vocab(row: np.ndarray, record: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconstruct every persisted next-token diagnostic."""

    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise PositionalActivationAnalysisError("raw full-vocabulary logit row is invalid")
    expected_id = record.get("expected_token_id")
    distractor_id = record.get("distractor_token_id")
    if (
        isinstance(expected_id, bool)
        or not isinstance(expected_id, int)
        or isinstance(distractor_id, bool)
        or not isinstance(distractor_id, int)
        or expected_id == distractor_id
        or not 0 <= expected_id < len(value)
        or not 0 <= distractor_id < len(value)
    ):
        raise PositionalActivationAnalysisError("registered answer-token IDs are invalid")
    expected_answer = record.get("expected_answer")
    distractor_answer = record.get("distractor_answer")
    if not all(isinstance(answer, str) and answer for answer in (expected_answer, distractor_answer)):
        raise PositionalActivationAnalysisError("registered answer labels are invalid")
    expected_logit = float(value[expected_id])
    distractor_logit = float(value[distractor_id])
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    peak = float(value.astype(np.float64).max())
    logsumexp = peak + math.log(float(np.exp(value.astype(np.float64) - peak).sum()))
    label_logsumexp = float(np.logaddexp(expected_logit, distractor_logit))
    if expected_logit > distractor_logit:
        predicted_answer: str | None = expected_answer
        predicted_id: int | None = expected_id
    elif distractor_logit > expected_logit:
        predicted_answer = distractor_answer
        predicted_id = distractor_id
    else:
        predicted_answer = None
        predicted_id = None
    return {
        "expected_logit": expected_logit,
        "distractor_logit": distractor_logit,
        "expected_minus_distractor_margin": expected_logit - distractor_logit,
        "predicted_answer": predicted_answer,
        "predicted_token_id": predicted_id,
        "answer_correct": expected_logit > distractor_logit,
        "answer_tie": expected_logit == distractor_logit,
        "greedy_token_id": maximum_ids[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "full_vocab_logits_sha256": f32_sha256(value),
    }


def validate_record_diagnostics(record: Mapping[str, Any], logits: np.ndarray) -> dict[str, Any]:
    expected = diagnostics_from_full_vocab(logits, record)
    observed = record.get("diagnostics")
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise PositionalActivationAnalysisError("stored diagnostic schema changed")
    for key, value in expected.items():
        candidate = observed[key]
        if isinstance(value, float):
            if not isinstance(candidate, (int, float)) or not math.isclose(
                float(candidate), value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise PositionalActivationAnalysisError(
                    f"stored diagnostic does not independently reconstruct: {key}"
                )
        elif candidate != value:
            raise PositionalActivationAnalysisError(
                f"stored diagnostic does not independently reconstruct: {key}"
            )
    return expected


def _normal_family(value: Any) -> str:
    aliases = {
        "retrieval": "retrieval",
        "property_retrieval": "retrieval",
        "lookup": "lookup",
        "codebook_lookup": "lookup",
        "composition": "composition",
        "joint": "composition",
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise PositionalActivationAnalysisError("record family changed") from error


def _normal_order(value: Any) -> str | None:
    if value is None:
        return None
    aliases = {
        "first": "first",
        "second": "second",
        "target_first": "first",
        "target_second": "second",
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise PositionalActivationAnalysisError("record order changed") from error


def _answer_choices(cell: Mapping[str, Any]) -> tuple[str, str]:
    expected = cell.get("correct_answer", cell.get("native_answer"))
    choices = cell.get("answer_options", cell.get("valid_answers", cell.get("answer_labels")))
    if (
        not isinstance(expected, str)
        or not isinstance(choices, list)
        or len(choices) != 2
        or len(set(choices)) != 2
        or expected not in choices
    ):
        raise PositionalActivationAnalysisError("cell answer contract is invalid")
    return expected, choices[1] if choices[0] == expected else choices[0]


def _raw_row_for_record(
    record: Mapping[str, Any],
    shards: Sequence[np.ndarray],
    *,
    call_index: int,
) -> np.ndarray:
    shard_index = call_index // 64
    row_index = call_index % 64
    if (
        record.get("raw_logits_global_row") != call_index
        or record.get("raw_logits_shard_index") != shard_index
        or record.get("raw_logits_row_in_shard") != row_index
        or not 0 <= shard_index < len(shards)
        or not 0 <= row_index < len(shards[shard_index])
    ):
        raise PositionalActivationAnalysisError("raw-logit row registry changed")
    row = np.ascontiguousarray(shards[shard_index][row_index], dtype="<f4")
    if record.get("raw_logits_row_sha256") != f32_sha256(row):
        raise PositionalActivationAnalysisError("raw-logit row hash changed")
    validate_record_diagnostics(record, row)
    return row


def validate_baseline_records(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    raw_shards: Sequence[np.ndarray],
    *,
    phase: str,
    role: str,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    """Reconstruct baseline identity, logits, activation rows, and traces."""

    expected_count = {"fit": 448, "localization": 448, "holdout": 896}[role]
    if len(records) != expected_count:
        raise PositionalActivationAnalysisError("baseline record count changed")
    cells = {str(row["cell_id"]): row for row in plan.get("cell_registry", [])}
    prompts = {str(row["cell_id"]): row for row in plan.get("prompts", [])}
    templates = sorted(
        (row for row in plan.get("baseline_templates", []) if row.get("role") == role),
        key=lambda row: str(row["cell_id"]),
    )
    if len(templates) != expected_count or len(cells) != len(plan.get("cell_registry", [])):
        raise PositionalActivationAnalysisError("plan baseline registry changed")
    expected_activation_rows = {"fit": 256, "localization": 256, "holdout": 512}[role]
    expected_shape = (
        (expected_activation_rows, len(layers), int(plan["model"]["hidden_width"]))
        if role != "holdout"
        else (expected_activation_rows, int(plan["model"]["hidden_width"]))
    )
    array = np.asarray(activations)
    if array.shape != expected_shape or array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
        raise PositionalActivationAnalysisError("baseline activation sidecar changed")
    observed: list[dict[str, Any]] = []
    next_activation = 0
    for call_index, (raw_record, template) in enumerate(zip(records, templates, strict=True)):
        record = dict(raw_record)
        cell_id = str(template["cell_id"])
        cell = cells.get(cell_id)
        prompt = prompts.get(cell_id)
        if cell is None or prompt is None:
            raise PositionalActivationAnalysisError("baseline cell/prompt does not resolve")
        family = _normal_family(cell.get("family"))
        identity = {
            "schema_version": getattr(runner, "BASELINE_SCHEMA", "coherent-readout-v5-positional-activation-baseline-v1"),
            "phase": phase,
            "role": role,
            "call_index": call_index,
            "world_id": cell["world_id"],
            "cell_id": cell_id,
            "prompt_id": prompt["prompt_id"],
            "call_plan_sha256": plan["call_plan_sha256"],
        }
        expected_answer, distractor_answer = _answer_choices(cell)
        factor_value = dict(cell["factors"]) if isinstance(cell.get("factors"), Mapping) else None
        required_values = {
            **identity,
            "record_id": canonical_sha256(identity),
            "baseline_id": canonical_sha256(identity),
            "template_id": template["template_id"],
            "family": family,
            "stratum_id": cell.get("stratum_id"),
            "intervention_prerequisite": bool(cell.get("intervention_prerequisite", False)),
            "expected_answer": expected_answer,
            "expected_token_id": prompt["expected_token_id"],
            "distractor_answer": distractor_answer,
            "distractor_token_id": prompt["distractor_token_id"],
            "factors": factor_value,
            "factor_levels": cell.get("factor_levels"),
            "order": _normal_order(cell.get("fact_order")),
            "pair_id": cell.get("semantic_pair_id", cell.get("pair_id")),
            "mate_cell_id": cell.get("mate_cell_id"),
            "target_property": cell.get("target_property"),
            "target_token_index": prompt.get("target_property_token_index"),
            "execution_input_sha256": canonical_sha256(
                {
                    "schema_version": "coherent-readout-v5-execution-input-v1",
                    "input_ids": prompt["execution_input_ids"],
                    "attention_mask": prompt["execution_attention_mask"],
                }
            ),
            "captured_layers": list(layers) if family == "composition" else [],
            "runner_sha256": file_sha256(Path(runner.__file__)) if runner is not None else None,
            "preregistration_sha256": (
                file_sha256(runner.FROZEN_PREREGISTRATION) if runner is not None else None
            ),
            "generation_used": False,
            "biological_model_calls": 0,
        }
        if any(record.get(key) != value for key, value in required_values.items()):
            raise PositionalActivationAnalysisError("baseline record does not independently reconstruct")
        _raw_row_for_record(record, raw_shards, call_index=call_index)
        trace = record.get("trace")
        expected_trace = {
            "use_cache": False,
            "return_dict": True,
            "generation_used": False,
            "teacher_forced_prompt_forward": True,
            "capture_layers": list(layers) if family == "composition" else [],
            "capture_counts": [1] * len(layers) if family == "composition" else [],
            "captures_removed": True,
            "hook_site": "resid_post",
            "token_site": "target_property_token" if family == "composition" else None,
            "token_index": prompt.get("target_property_token_index"),
            "model_calls": 1,
        }
        if trace != expected_trace:
            raise PositionalActivationAnalysisError("baseline forward trace changed")
        if family == "composition":
            activation_row = next_activation
            next_activation += 1
            block = array[activation_row]
            block_2d = block[None, :] if block.ndim == 1 else block
            layer_hashes = {
                str(layer): f32_sha256(block_2d[index]) for index, layer in enumerate(layers)
            }
            if (
                record.get("activation_row") != activation_row
                or record.get("activation_sha256") != f32_sha256(block_2d if role == "holdout" else block)
                or record.get("activation_layer_sha256") != layer_hashes
            ):
                raise PositionalActivationAnalysisError("baseline activation binding changed")
        elif any(
            record.get(key) not in (None, {}, [])
            for key in ("activation_row", "activation_sha256", "activation_layer_sha256")
        ):
            raise PositionalActivationAnalysisError("non-composition record binds an activation")
        record["recipient_selected"] = bool(cell.get("recipient_selected", False))
        observed.append(record)
    if next_activation != expected_activation_rows:
        raise PositionalActivationAnalysisError("composition activation coverage changed")
    return observed


def _activation_vector(
    record: Mapping[str, Any], activations: np.ndarray, *, role: str, layer: int
) -> np.ndarray:
    row = record.get("activation_row")
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(activations):
        raise PositionalActivationAnalysisError("baseline activation row is invalid")
    if role == "localization":
        value = activations[row, LAYER_GRID.index(layer)]
    elif role == "holdout":
        value = activations[row]
    else:
        raise PositionalActivationAnalysisError("patch role changed")
    result = np.ascontiguousarray(value, dtype="<f4")
    if record.get("activation_layer_sha256", {}).get(str(layer)) != f32_sha256(result):
        raise PositionalActivationAnalysisError("baseline activation vector hash changed")
    return result


def crossfit_coordinate_separation(
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    basis: np.ndarray,
    *,
    role: str,
    layer: int,
    expected_worlds: int,
) -> dict[str, Any]:
    """Measure the natural TARGET-second minus TARGET-first positional coordinate.

    The axis is fit-only.  Localization and holdout states are never used to
    rotate or sign-select it, so this is a genuine cross-fit natural-use check.
    """

    if role not in {"localization", "holdout"} or layer not in LAYER_GRID:
        raise PositionalActivationAnalysisError("cross-fit coordinate request changed")
    positional = _unit(
        np.asarray(basis[LAYER_GRID.index(layer), 1], dtype=np.float64),
        "fit-only positional direction",
    )
    panel = [
        row
        for row in records
        if row.get("family") == "composition" and row.get("recipient_selected") is True
    ]
    registry: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in panel:
        factors = row.get("factors")
        order = factors.get("o") if isinstance(factors, Mapping) else None
        pair = row.get("pair_id")
        world = row.get("world_id")
        if order not in (-1, 1) or not isinstance(pair, str) or not isinstance(world, str):
            raise PositionalActivationAnalysisError("cross-fit panel identity changed")
        key = (world, pair)
        if int(order) in registry[key]:
            raise PositionalActivationAnalysisError("cross-fit panel order is duplicated")
        registry[key][int(order)] = row
    if (
        len(registry) != expected_worlds * 8
        or any(set(pair) != {-1, 1} for pair in registry.values())
    ):
        raise PositionalActivationAnalysisError("cross-fit panel must have eight complete pairs per world")
    values: list[dict[str, Any]] = []
    for (world, pair), order_rows in sorted(registry.items()):
        first = _activation_vector(order_rows[-1], activations, role=role, layer=layer)
        second = _activation_vector(order_rows[1], activations, role=role, layer=layer)
        coordinate = float(
            positional @ (second.astype(np.float64) - first.astype(np.float64))
        )
        values.append({"world_id": world, "pair_id": pair, "Z": coordinate})
    world_values = _mean_by_world(values, "Z")
    summary = effect_summary(
        world_values,
        world_values,
        bootstrap_indices=_bootstrap_indices(expected_worlds),
    )
    positive_required = 6 if expected_worlds == LOCALIZATION_WORLDS else 12
    gates = {
        "Z_mean_positive": summary["mean"] > 0.0,
        "Z_stability_lower_positive": summary["bootstrap_95"]["lower_95"] > 0.0,
        "Z_positive_worlds": summary["positive_world_count"] >= positive_required,
    }
    gates["pass"] = all(gates.values())
    return {
        "layer": layer,
        "axis_source": "fit_only",
        "orientation": "target_second_minus_target_first_dot_positional_axis",
        "Z": summary,
        "gates": gates,
    }


def expected_projected_patch(
    recipient: np.ndarray,
    source: np.ndarray,
    application_direction: np.ndarray,
    *,
    dose_direction: np.ndarray | None = None,
) -> np.ndarray:
    recipient64 = np.asarray(recipient, dtype=np.float64)
    source64 = np.asarray(source, dtype=np.float64)
    unit = _unit(np.asarray(application_direction, dtype=np.float64), "patch direction")
    dose_unit = _unit(
        np.asarray(
            application_direction if dose_direction is None else dose_direction,
            dtype=np.float64,
        ),
        "dose direction",
    )
    if recipient64.shape != source64.shape or recipient64.shape != unit.shape or unit.shape != dose_unit.shape:
        raise PositionalActivationAnalysisError("patch vector shapes differ")
    dose = float(dose_unit @ (source64 - recipient64))
    return np.ascontiguousarray(
        recipient64 + dose * unit,
        dtype="<f4",
    )


def validate_patch_records(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    patched_activations: np.ndarray,
    raw_shards: Sequence[np.ndarray],
    baseline_records: Sequence[Mapping[str, Any]],
    baseline_activations: np.ndarray,
    basis: np.ndarray,
    *,
    phase: str,
    role: str,
    selected_layer: int | None,
) -> list[dict[str, Any]]:
    """Reconstruct every selective patch from frozen states and axes."""

    expected_count = {"localization": 1568, "holdout": 784}[role]
    if len(records) != expected_count:
        raise PositionalActivationAnalysisError("patch record count changed")
    patched = np.asarray(patched_activations)
    width = int(plan["model"]["hidden_width"])
    if patched.shape != (expected_count, width) or patched.dtype != np.dtype("<f4"):
        raise PositionalActivationAnalysisError("patched activation sidecar changed")
    templates = [row for row in plan["patch_templates"] if row["role"] == role]
    baselines = {row["cell_id"]: row for row in baseline_records}
    prompts = {row["cell_id"]: row for row in plan["prompts"]}
    cells = {row["cell_id"]: row for row in plan["cell_registry"]}
    if len(templates) != expected_count or len(baselines) != len(baseline_records):
        raise PositionalActivationAnalysisError("patch template/baseline registry changed")
    observed: list[dict[str, Any]] = []
    for call_index, (raw_record, template) in enumerate(zip(records, templates, strict=True)):
        record = dict(raw_record)
        layer = template["layer"] if role == "localization" else selected_layer
        if layer not in LAYER_GRID:
            raise PositionalActivationAnalysisError("selected patch layer is invalid")
        recipient = baselines.get(template["recipient_cell_id"])
        source = baselines.get(template["source_cell_id"])
        prompt = prompts.get(template["recipient_cell_id"])
        cell = cells.get(template["recipient_cell_id"])
        if any(value is None for value in (recipient, source, prompt, cell)):
            raise PositionalActivationAnalysisError("patch source graph does not resolve")
        if (
            recipient.get("expected_answer") != source.get("expected_answer")
            or recipient.get("expected_token_id") != source.get("expected_token_id")
            or recipient.get("distractor_token_id") != source.get("distractor_token_id")
        ):
            raise PositionalActivationAnalysisError(
                "patch source/recipient is not a same-answer semantic pair"
            )
        identity = {
            "schema_version": getattr(runner, "PATCH_SCHEMA", "coherent-readout-v5-positional-activation-patch-v1"),
            "phase": phase,
            "role": role,
            "call_index": call_index,
            "world_id": template["world_id"],
            "pair_id": template["pair_id"],
            "template_id": template["template_id"],
            "condition": template["condition"],
            "layer": layer,
            "recipient_cell_id": template["recipient_cell_id"],
            "source_cell_id": template["source_cell_id"],
            "call_plan_sha256": plan["call_plan_sha256"],
        }
        record_id = canonical_sha256(identity)
        required = {
            **identity,
            "record_id": record_id,
            "patch_id": record_id,
            "operation": template["operation"],
            "direction_name": template["direction_name"],
            "recipient_order": template["recipient_order"],
            "source_order": template["source_order"],
            "cell_id": template["recipient_cell_id"],
            "prompt_id": prompt["prompt_id"],
            "family": "composition",
            "stratum_id": cell.get("stratum_id"),
            "expected_answer": prompt["expected_answer"],
            "expected_token_id": prompt["expected_token_id"],
            "distractor_answer": prompt["distractor_answer"],
            "distractor_token_id": prompt["distractor_token_id"],
            "factors": dict(cell["factors"]),
            "order": _normal_order(cell.get("fact_order")),
            "target_token_index": prompt["target_property_token_index"],
            "execution_input_sha256": canonical_sha256(
                {
                    "schema_version": "coherent-readout-v5-execution-input-v1",
                    "input_ids": prompt["execution_input_ids"],
                    "attention_mask": prompt["execution_attention_mask"],
                }
            ),
            "patched_activation_row": call_index,
            "recipient_baseline_id": recipient["baseline_id"],
            "recipient_activation_row": recipient["activation_row"],
            "source_baseline_id": source["baseline_id"],
            "source_activation_row": source["activation_row"],
            "runner_sha256": file_sha256(Path(runner.__file__)) if runner is not None else None,
            "preregistration_sha256": (
                file_sha256(runner.FROZEN_PREREGISTRATION) if runner is not None else None
            ),
            "generation_used": False,
            "biological_model_calls": 0,
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise PositionalActivationAnalysisError("patch record does not independently reconstruct")
        raw_logits = _raw_row_for_record(record, raw_shards, call_index=call_index)
        recipient_vector = _activation_vector(recipient, baseline_activations, role=role, layer=layer)
        source_vector = _activation_vector(source, baseline_activations, role=role, layer=layer)
        direction = np.ascontiguousarray(
            basis[LAYER_GRID.index(layer), ("center", "positional", "answer", "null").index(template["direction_name"])],
            dtype="<f4",
        )
        positional_direction = np.ascontiguousarray(
            basis[LAYER_GRID.index(layer), 1], dtype="<f4"
        )
        registered_dose = float(
            _unit(positional_direction, "positional dose")
            @ (source_vector.astype(np.float64) - recipient_vector.astype(np.float64))
        )
        positional_scalar_d = (
            registered_dose
            if template["recipient_order"] == "second"
            else -registered_dose
        )
        expected_activation = expected_projected_patch(
            recipient_vector,
            source_vector,
            direction,
            dose_direction=positional_direction,
        )
        observed_activation = np.ascontiguousarray(patched[call_index], dtype="<f4")
        tolerance = 1e-6 * max(1.0, float(np.linalg.norm(expected_activation.astype(np.float64))))
        error = float(
            np.linalg.norm(observed_activation.astype(np.float64) - expected_activation.astype(np.float64))
        )
        if (
            record.get("recipient_activation_sha256") != f32_sha256(recipient_vector)
            or record.get("source_activation_sha256") != f32_sha256(source_vector)
            or record.get("applied_direction_sha256") != f32_sha256(direction)
            or record.get("patched_activation_sha256") != f32_sha256(observed_activation)
            or error > tolerance
        ):
            raise PositionalActivationAnalysisError("patched activation does not reconstruct")
        displacement_l2 = float(
            np.linalg.norm(
                observed_activation.astype(np.float64) - recipient_vector.astype(np.float64)
            )
        )
        dose_tolerance = DOSE_MATCH_TOLERANCE * max(1.0, abs(registered_dose))
        if abs(displacement_l2 - abs(registered_dose)) > dose_tolerance:
            raise PositionalActivationAnalysisError(
                "patch displacement is not matched to the positional scalar dose"
            )
        scalar_fields = {
            "positional_scalar_d": positional_scalar_d,
            "applied_signed_scalar": registered_dose,
            "registered_positional_dose": registered_dose,
            "expected_displacement_l2": abs(registered_dose),
            "observed_displacement_l2": displacement_l2,
            "applied_displacement_l2": displacement_l2,
            "displacement_abs_scalar_error": abs(displacement_l2 - abs(registered_dose)),
        }
        if any(
            key not in record
            or not math.isclose(
                _finite(record[key], key), value, rel_tol=1e-7, abs_tol=1e-7
            )
            for key, value in scalar_fields.items()
        ):
            raise PositionalActivationAnalysisError("dose-matched scalar receipt changed")
        trace = record.get("trace")
        if not isinstance(trace, Mapping) or not _trace_engineering_pass(record):
            raise PositionalActivationAnalysisError("patch trace engineering failed")
        trace_exact = {
            "operation": template["operation"],
            "layer": layer,
            "token_index": prompt["target_property_token_index"],
            "strength": 1.0,
            "direction_name": template["direction_name"],
            "pre_activation_sha256": f32_sha256(recipient_vector),
            "post_activation_sha256": f32_sha256(observed_activation),
            "expected_activation_sha256": f32_sha256(expected_activation),
        }
        if any(trace.get(key) != value for key, value in trace_exact.items()):
            raise PositionalActivationAnalysisError("patch trace identity changed")
        direction_unit = _unit(direction, "applied direction")
        displacement = (
            observed_activation.astype(np.float64) - recipient_vector.astype(np.float64)
        )
        orthogonal = displacement - direction_unit * float(direction_unit @ displacement)
        trace_numeric = {
            "post_expected_l2_error": error,
            "post_expected_l2_tolerance": tolerance,
            "orthogonal_displacement_l2": float(np.linalg.norm(orthogonal)),
            "displacement_l2": displacement_l2,
            "positional_scalar_d": positional_scalar_d,
            "applied_signed_scalar": registered_dose,
            "expected_displacement_l2": abs(registered_dose),
            "observed_displacement_l2": displacement_l2,
            "applied_displacement_l2": displacement_l2,
            "displacement_abs_scalar_error": abs(displacement_l2 - abs(registered_dose)),
            "displacement_abs_scalar_tolerance": dose_tolerance,
            "pre_axis_coefficient": float(
                direction_unit @ recipient_vector.astype(np.float64)
            ),
            "source_axis_coefficient": float(
                direction_unit @ source_vector.astype(np.float64)
            ),
            "post_axis_coefficient": float(
                direction_unit @ observed_activation.astype(np.float64)
            ),
        }
        if any(
            key not in trace
            or not math.isclose(
                _finite(trace[key], f"trace {key}"),
                value,
                rel_tol=1e-7,
                abs_tol=1e-7,
            )
            for key, value in trace_numeric.items()
        ):
            raise PositionalActivationAnalysisError("patch trace numerics do not reconstruct")
        dose_key = next(
            (
                key
                for key in ("registered_positional_dose", "positional_scalar_dose")
                if key in trace
            ),
            None,
        )
        if dose_key is None or not math.isclose(
            _finite(trace[dose_key], "registered positional dose"),
            registered_dose,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise PositionalActivationAnalysisError("registered positional dose changed")
        if "applied_displacement_l2" in trace and not math.isclose(
            _finite(trace["applied_displacement_l2"], "applied displacement"),
            displacement_l2,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise PositionalActivationAnalysisError("trace displacement norm changed")
        if abs(_finite(record.get("baseline_margin"), "baseline margin") - _diagnostic_margin(recipient)) > 1e-7:
            raise PositionalActivationAnalysisError("patch baseline margin changed")
        if template["condition"] == "identity":
            baseline_logits = None
            baseline_global = recipient.get("raw_logits_global_row")
            if isinstance(baseline_global, int):
                baseline_shard = baseline_global // 64
                baseline_row = baseline_global % 64
                # Raw baseline shards are not passed here; stored errors are
                # nevertheless independently tied to the two diagnostics and
                # exact self-state activation equality.
                baseline_logits = (baseline_shard, baseline_row)
            if baseline_logits is None or template["recipient_cell_id"] != template["source_cell_id"]:
                raise PositionalActivationAnalysisError("identity source graph changed")
            expected_error = abs(
                float(record["diagnostics"]["expected_logit"])
                - float(recipient["diagnostics"]["expected_logit"])
            )
            distractor_error = abs(
                float(record["diagnostics"]["distractor_logit"])
                - float(recipient["diagnostics"]["distractor_logit"])
            )
            if (
                not math.isclose(trace.get("identity_expected_logit_abs_error", -1), expected_error, abs_tol=1e-12)
                or not math.isclose(
                    trace.get("identity_distractor_logit_abs_error", -1), distractor_error, abs_tol=1e-12
                )
                or trace.get("identity_global_argmax_preserved")
                != (record["diagnostics"]["maximum_token_ids"] == recipient["diagnostics"]["maximum_token_ids"])
            ):
                raise PositionalActivationAnalysisError("identity diagnostic does not reconstruct")
        observed.append(record)
    dose_registry: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in observed:
        if record["condition"] == "identity":
            continue
        direction = "rescue" if "rescue" in record["condition"] else "damage"
        dose_registry[(record["world_id"], record["pair_id"], direction)].append(
            float(record["registered_positional_dose"])
        )
    if any(
        len(values) != 3
        or max(values) - min(values) > 1e-6 * max(1.0, max(abs(value) for value in values))
        for values in dose_registry.values()
    ):
        raise PositionalActivationAnalysisError(
            "positional, answer, and null conditions do not share the exact scalar dose"
        )
    return observed


def behavioral_admission(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    expected_worlds: int,
    prerequisite_record_ids: Sequence[str] | None = None,
    intervention_pair_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply component and paired positional-gap admission gates."""

    expected_by_role = {
        "fit": FIT_WORLDS,
        "localization": LOCALIZATION_WORLDS,
        "holdout": HOLDOUT_WORLDS,
    }
    if role not in expected_by_role or expected_worlds != expected_by_role[role]:
        raise PositionalActivationAnalysisError("behavioral role/world count changed")
    normalized = [dict(row) for row in rows]
    if not normalized or len({row.get("record_id") for row in normalized}) != len(normalized):
        raise PositionalActivationAnalysisError("behavioral records are empty or duplicated")
    worlds = sorted({row.get("world_id") for row in normalized})
    if len(worlds) != expected_worlds or any(not isinstance(world, str) for world in worlds):
        raise PositionalActivationAnalysisError("behavioral world coverage changed")
    family_alias = {
        "retrieval": "retrieval",
        "property_retrieval": "retrieval",
        "lookup": "lookup",
        "codebook_lookup": "lookup",
        "composition": "composition",
    }
    if any(row.get("family") not in family_alias for row in normalized):
        raise PositionalActivationAnalysisError("behavioral family changed")
    for row in normalized:
        row["family"] = family_alias[row["family"]]
        row["correct"] = _diagnostic_correct(row)
        row["margin"] = _diagnostic_margin(row)

    expected_family_counts = {"retrieval": 8, "lookup": 16, "composition": 32}
    for world in worlds:
        observed_counts = Counter(
            row["family"] for row in normalized if row.get("world_id") == world
        )
        if observed_counts != Counter(expected_family_counts):
            raise PositionalActivationAnalysisError("behavioral family counts changed")

    component = [row for row in normalized if row["family"] in {"retrieval", "lookup"}]
    composition = [row for row in normalized if row["family"] == "composition"]
    if not component or not composition:
        raise PositionalActivationAnalysisError("component or composition records are absent")
    family_accuracy = {
        family: float(np.mean([row["correct"] for row in component if row["family"] == family]))
        for family in ("retrieval", "lookup")
    }
    if any(math.isnan(value) for value in family_accuracy.values()):
        raise PositionalActivationAnalysisError("retrieval or lookup family is empty")
    world_accuracy: dict[str, float] = {}
    for world in worlds:
        for family in ("retrieval", "lookup"):
            members = [
                row for row in component if row["world_id"] == world and row["family"] == family
            ]
            if not members:
                raise PositionalActivationAnalysisError("component world/family is empty")
            world_accuracy[f"{world}__{family}"] = float(np.mean([row["correct"] for row in members]))
    stratum_groups: dict[str, list[bool]] = defaultdict(list)
    for row in component:
        factors = row.get("factors")
        if not isinstance(factors, Mapping):
            raise PositionalActivationAnalysisError("component factors are missing")
        names = ("p", "o", "v") if row["family"] == "retrieval" else ("p", "m", "r", "v")
        observed_factor = False
        for name in names:
            level = factors.get(name)
            if level is None:
                continue
            if level not in (-1, 1):
                raise PositionalActivationAnalysisError("component factor level is not signed")
            observed_factor = True
            stratum_groups[f"{row['family']}__{name}={int(level):+d}"].append(row["correct"])
        if not observed_factor:
            raise PositionalActivationAnalysisError("component record has no registered factor stratum")
    stratum_accuracy = {
        stratum: float(np.mean(values)) for stratum, values in sorted(stratum_groups.items())
    }

    by_pair: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in composition:
        pair_id = row.get("pair_id", row.get("semantic_pair_id"))
        order = row.get("factors", {}).get("o") if isinstance(row.get("factors"), Mapping) else row.get("o")
        if order not in (-1, 1):
            order = {"target_first": -1, "target_second": 1}.get(row.get("order"))
        if not isinstance(pair_id, str) or not pair_id or order not in (-1, 1):
            raise PositionalActivationAnalysisError("composition pair/order identity is invalid")
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise PositionalActivationAnalysisError("composition world identity is invalid")
        pair_key = (world_id, pair_id)
        if int(order) in by_pair[pair_key]:
            raise PositionalActivationAnalysisError("composition pair/order is duplicated")
        by_pair[pair_key][int(order)] = row
    if any(set(pair) != {-1, 1} for pair in by_pair.values()):
        raise PositionalActivationAnalysisError("composition pair is incomplete")
    first_rows: list[dict[str, Any]] = []
    second_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    panel_gap_rows: list[dict[str, Any]] = []
    panel_ids = set(intervention_pair_ids or ())
    for (pair_world, pair_id), pair in sorted(by_pair.items()):
        first = pair[-1]  # locked convention: o=-1 means TARGET fact first
        second = pair[1]
        first_factors = dict(first["factors"])
        second_factors = dict(second["factors"])
        first_factors.pop("o", None)
        second_factors.pop("o", None)
        if (
            first["world_id"] != second["world_id"]
            or first["world_id"] != pair_world
            or first.get("expected_answer") != second.get("expected_answer")
            or first.get("expected_token_id") != second.get("expected_token_id")
            or first_factors != second_factors
        ):
            raise PositionalActivationAnalysisError(
                "composition pair is not a same-answer order-only match"
            )
        first_rows.append(first)
        second_rows.append(second)
        gap_row = {
            "world_id": first["world_id"],
            "pair_id": pair_id,
            "gap": first["margin"] - second["margin"],
        }
        gap_rows.append(gap_row)
        is_panel = (
            pair_id in panel_ids
            or f"{pair_world}::{pair_id}" in panel_ids
            or first.get("intervention_pair") is True
            or first.get("recipient_selected") is True
        )
        if is_panel:
            panel_gap_rows.append(gap_row)
    first_accuracy = float(np.mean([row["correct"] for row in first_rows]))
    second_accuracy = float(np.mean([row["correct"] for row in second_rows]))
    panel_counts = Counter(row["world_id"] for row in panel_gap_rows)
    if len(panel_counts) != expected_worlds or set(panel_counts.values()) != {8}:
        raise PositionalActivationAnalysisError("fixed intervention panel must contain eight pairs per world")
    gap_worlds = _mean_by_world(panel_gap_rows, "gap")
    full_gap_worlds = _mean_by_world(gap_rows, "gap")
    indices = _bootstrap_indices(expected_worlds)
    gap = effect_summary(gap_worlds, gap_worlds, bootstrap_indices=indices)
    full_gap = effect_summary(full_gap_worlds, full_gap_worlds, bootstrap_indices=indices)
    prerequisite_set = set(prerequisite_record_ids or ())
    if prerequisite_record_ids is None:
        flagged = [row for row in component if row.get("intervention_prerequisite") is True]
        prerequisite_counts = Counter(row["world_id"] for row in flagged)
        if len(prerequisite_counts) != expected_worlds or set(prerequisite_counts.values()) != {16}:
            raise PositionalActivationAnalysisError(
                "fixed panel must have sixteen registered direct prerequisites per world"
            )
        prerequisite_rows = flagged
    else:
        registry = {row["record_id"]: row for row in component}
        if not prerequisite_set or not prerequisite_set.issubset(registry):
            raise PositionalActivationAnalysisError("direct-prerequisite registry does not resolve")
        prerequisite_rows = [registry[record_id] for record_id in sorted(prerequisite_set)]
    gates = {
        "retrieval_lookup_overall": all(
            value >= BEHAVIOR_THRESHOLDS["retrieval_lookup_overall_accuracy"]
            for value in family_accuracy.values()
        ),
        "retrieval_lookup_each_world": min(world_accuracy.values())
        >= BEHAVIOR_THRESHOLDS["retrieval_lookup_world_accuracy"],
        "retrieval_lookup_each_stratum": min(stratum_accuracy.values())
        >= BEHAVIOR_THRESHOLDS["retrieval_lookup_stratum_accuracy"],
        "target_first_composition": first_accuracy
        >= BEHAVIOR_THRESHOLDS["target_first_composition_accuracy"],
        "target_second_error": (1.0 - second_accuracy)
        >= BEHAVIOR_THRESHOLDS["target_second_error_rate"],
        "composition_accuracy_gap": (first_accuracy - second_accuracy)
        >= BEHAVIOR_THRESHOLDS["composition_accuracy_gap"],
        "G_bootstrap_lower_positive": gap["bootstrap_95"]["lower_95"] > 0.0,
        "fixed_panel_direct_prerequisites": all(row["correct"] for row in prerequisite_rows),
    }
    if role == "localization":
        gates["positive_G_worlds"] = gap["positive_world_count"] >= 6
    elif role == "holdout":
        gates["positive_G_worlds"] = gap["positive_world_count"] >= 12
    gates["pass"] = all(gates.values())
    return {
        "n_records": len(normalized),
        "n_worlds": expected_worlds,
        "family_accuracy": family_accuracy,
        "component_world_accuracy": world_accuracy,
        "component_stratum_accuracy": stratum_accuracy,
        "target_first_composition_accuracy": first_accuracy,
        "target_second_composition_accuracy": second_accuracy,
        "target_second_error_rate": 1.0 - second_accuracy,
        "composition_accuracy_gap": first_accuracy - second_accuracy,
        "G": gap,
        "G_panel": gap,
        "G_full": full_gap,
        "gates": gates,
    }


def _patch_margin(row: Mapping[str, Any]) -> float:
    margin = _diagnostic_margin(row)
    diagnostics = row.get("diagnostics", {})
    stored = diagnostics.get("expected_minus_distractor_margin") if isinstance(diagnostics, Mapping) else None
    if stored is not None and abs(margin - _finite(stored, "patch margin")) > 1e-7:
        raise PositionalActivationAnalysisError("patch diagnostic margin does not reconstruct")
    return margin


def _trace_engineering_pass(row: Mapping[str, Any]) -> bool:
    """Check the phase-independent hard properties of a persisted hook trace."""

    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        return False
    exact = {
        "model_calls": 1,
        "generation_used": False,
        "hook_calls": 1,
        "hook_removed": True,
        "finite_logits": True,
        "finite_activations": True,
        "post_activation_matches_expected": True,
        "non_target_tokens_unchanged": True,
    }
    for key, expected in exact.items():
        if key in trace and trace[key] != expected:
            return False
    # The runner must expose at least the core evidence; absence cannot count as
    # success merely because an older trace used different optional diagnostics.
    required = {"model_calls", "generation_used", "hook_calls", "hook_removed"}
    if not required.issubset(trace):
        return False
    for hash_key in ("pre_activation_sha256", "post_activation_sha256", "expected_activation_sha256"):
        if hash_key in trace:
            try:
                _validate_hash(trace[hash_key], hash_key)
            except PositionalActivationAnalysisError:
                return False
    return True


def _cell_effect(row: Mapping[str, Any]) -> float:
    condition = row.get("condition")
    baseline = _finite(row.get("baseline_margin"), "baseline_margin")
    patched = _patch_margin(row)
    if condition in {"positional_rescue", "answer_rescue_sham", "null_rescue_sham"}:
        return patched - baseline
    if condition in {"positional_damage", "answer_damage_sham", "null_damage_sham"}:
        return baseline - patched
    if condition == "identity":
        return patched - baseline
    raise PositionalActivationAnalysisError("patch condition changed")


def _specificity_summary(
    primary_world_values: Mapping[str, float],
    sham_world_values: Mapping[str, float],
    world_gaps: Mapping[str, float],
    *,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    worlds = sorted(primary_world_values)
    if worlds != sorted(sham_world_values) or worlds != sorted(world_gaps):
        raise PositionalActivationAnalysisError("specificity inputs do not share worlds")
    primary = np.asarray([primary_world_values[world] for world in worlds], dtype=np.float64)
    sham = np.asarray([sham_world_values[world] for world in worlds], dtype=np.float64)
    gaps = np.asarray([world_gaps[world] for world in worlds], dtype=np.float64)
    if not np.isfinite(primary).all() or not np.isfinite(sham).all() or not np.isfinite(gaps).all():
        raise PositionalActivationAnalysisError("specificity inputs are nonfinite")
    primary_samples = primary[bootstrap_indices].mean(axis=1)
    sham_samples = sham[bootstrap_indices].mean(axis=1)
    gap_samples = gaps[bootstrap_indices].mean(axis=1)
    samples = primary_samples - sham_samples
    point = float(primary.mean() - sham.mean())
    gap_mean = float(gaps.mean())
    ratio_defined = bool(gap_mean > 0.0)
    bootstrap_ratio_valid = bool(ratio_defined and np.all(gap_samples > 0.0))
    return {
        "world_primary_values": dict(zip(worlds, primary.tolist(), strict=True)),
        "world_sham_values": dict(zip(worlds, sham.tolist(), strict=True)),
        "n_worlds": len(worlds),
        "primary_mean": float(primary.mean()),
        "sham_magnitude_mean": float(sham.mean()),
        "mean": point,
        "gap_mean": gap_mean,
        "mean_over_G": float(point / gap_mean) if ratio_defined else None,
        "bootstrap_95": _interval(samples),
        "bootstrap_ratio_95": _interval(samples / gap_samples) if bootstrap_ratio_valid else None,
        "ratio_defined": ratio_defined,
        "bootstrap_ratio_valid": bootstrap_ratio_valid,
    }


def causal_metrics(
    patch_rows: Sequence[Mapping[str, Any]],
    behavioral: Mapping[str, Any],
    *,
    layer: int,
    expected_worlds: int,
    coordinate_separation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize bidirectional positional patches and pair-matched sham axes."""

    if layer not in LAYER_GRID:
        raise PositionalActivationAnalysisError("causal layer is outside the preregistered grid")
    if expected_worlds not in {LOCALIZATION_WORLDS, HOLDOUT_WORLDS}:
        raise PositionalActivationAnalysisError("causal phase world count changed")
    rows = [dict(row) for row in patch_rows if row.get("layer") == layer]
    worlds = sorted({row.get("world_id") for row in rows})
    if len(worlds) != expected_worlds or any(not isinstance(world, str) for world in worlds):
        raise PositionalActivationAnalysisError("causal world coverage changed")
    if any(row.get("condition") not in PATCH_CONDITIONS for row in rows):
        raise PositionalActivationAnalysisError("causal condition registry changed")
    identity = [row for row in rows if row["condition"] == "identity"]
    causal = [row for row in rows if row["condition"] != "identity"]
    if len(identity) != expected_worlds or Counter(row["world_id"] for row in identity) != Counter(worlds):
        raise PositionalActivationAnalysisError("identity sentinel coverage changed")
    expected_causal_conditions = set(PATCH_CONDITIONS) - {"identity"}
    pair_registry: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in causal:
        pair = row.get("pair_id", row.get("semantic_pair_id"))
        if not isinstance(pair, str) or not pair:
            raise PositionalActivationAnalysisError("causal semantic pair ID is invalid")
        key = (row["world_id"], pair)
        if row["condition"] in pair_registry[key]:
            raise PositionalActivationAnalysisError("causal pair/condition is duplicated")
        pair_registry[key].add(row["condition"])
        row["effect"] = _cell_effect(row)
    if (
        len(pair_registry) != expected_worlds * 8
        or any(conditions != expected_causal_conditions for conditions in pair_registry.values())
    ):
        raise PositionalActivationAnalysisError("causal panel is not eight complete pairs per world")

    gap_payload = behavioral.get("G")
    if not isinstance(gap_payload, Mapping) or not isinstance(gap_payload.get("world_values"), Mapping):
        raise PositionalActivationAnalysisError("behavioral natural-gap payload is missing")
    world_gaps = {str(key): _finite(value, "world G") for key, value in gap_payload["world_values"].items()}
    if sorted(world_gaps) != worlds:
        raise PositionalActivationAnalysisError("causal and behavioral worlds differ")
    indices = _bootstrap_indices(expected_worlds)

    by_condition = {condition: [row for row in causal if row["condition"] == condition] for condition in expected_causal_conditions}
    effects: dict[str, dict[str, Any]] = {}
    for condition, members in sorted(by_condition.items()):
        world_values = _mean_by_world(members, "effect")
        effects[condition] = effect_summary(world_values, world_gaps, bootstrap_indices=indices)

    def combined_rows(rescue_name: str, damage_name: str) -> list[dict[str, Any]]:
        registry: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for condition in (rescue_name, damage_name):
            for row in by_condition[condition]:
                pair = row.get("pair_id", row.get("semantic_pair_id"))
                key = (row["world_id"], pair)
                registry[key][condition] = row["effect"]
        if any(set(value) != {rescue_name, damage_name} for value in registry.values()):
            raise PositionalActivationAnalysisError("bidirectional causal pairs are incomplete")
        return [
            {
                "world_id": world,
                "pair_id": pair,
                "effect": (values[rescue_name] + values[damage_name]) / 2.0,
            }
            for (world, pair), values in sorted(registry.items())
        ]

    def magnitude_rows(rescue_name: str, damage_name: str) -> list[dict[str, Any]]:
        registry: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for condition in (rescue_name, damage_name):
            for row in by_condition[condition]:
                pair = row.get("pair_id", row.get("semantic_pair_id"))
                registry[(row["world_id"], pair)][condition] = row["effect"]
        if any(set(value) != {rescue_name, damage_name} for value in registry.values()):
            raise PositionalActivationAnalysisError("sham magnitude pairs are incomplete")
        return [
            {
                "world_id": world,
                "pair_id": pair,
                "effect": (abs(values[rescue_name]) + abs(values[damage_name])) / 2.0,
            }
            for (world, pair), values in sorted(registry.items())
        ]

    combined_members = {
        "positional_combined": combined_rows("positional_rescue", "positional_damage"),
        "answer_combined_sham": combined_rows("answer_rescue_sham", "answer_damage_sham"),
        "null_combined_sham": combined_rows("null_rescue_sham", "null_damage_sham"),
        "answer_sham_magnitude": magnitude_rows("answer_rescue_sham", "answer_damage_sham"),
        "null_sham_magnitude": magnitude_rows("null_rescue_sham", "null_damage_sham"),
    }
    combined_worlds: dict[str, dict[str, float]] = {}
    for name, members in combined_members.items():
        combined_worlds[name] = _mean_by_world(members, "effect")
        effects[name] = effect_summary(combined_worlds[name], world_gaps, bootstrap_indices=indices)
    specificity = {
        "combined_minus_absolute_answer": _specificity_summary(
            combined_worlds["positional_combined"],
            combined_worlds["answer_sham_magnitude"],
            world_gaps,
            bootstrap_indices=indices,
        ),
        "combined_minus_absolute_null": _specificity_summary(
            combined_worlds["positional_combined"],
            combined_worlds["null_sham_magnitude"],
            world_gaps,
            bootstrap_indices=indices,
        ),
    }
    identity_errors = {
        row["world_id"]: abs(_cell_effect(row))
        for row in identity
    }
    engineering_rows_pass = all(_trace_engineering_pass(row) for row in rows)
    identity_logit_checks = []
    for row in identity:
        trace = row.get("trace", {})
        expected_error = row.get(
            "identity_expected_logit_abs_error",
            trace.get("identity_expected_logit_abs_error") if isinstance(trace, Mapping) else None,
        )
        distractor_error = row.get(
            "identity_distractor_logit_abs_error",
            trace.get("identity_distractor_logit_abs_error") if isinstance(trace, Mapping) else None,
        )
        argmax_preserved = row.get(
            "identity_global_argmax_preserved",
            trace.get("identity_global_argmax_preserved") if isinstance(trace, Mapping) else None,
        )
        identity_logit_checks.append(
            expected_error is not None
            and distractor_error is not None
            and _finite(expected_error, "identity expected-logit error")
            <= CAUSAL_THRESHOLDS["identity_margin_tolerance"]
            and _finite(distractor_error, "identity distractor-logit error")
            <= CAUSAL_THRESHOLDS["identity_margin_tolerance"]
            and argmax_preserved is True
        )
    positive_required = (
        CAUSAL_THRESHOLDS["localization_positive_worlds"]
        if expected_worlds == LOCALIZATION_WORLDS
        else CAUSAL_THRESHOLDS["holdout_positive_worlds"]
    )

    def primary_gate(summary: Mapping[str, Any]) -> bool:
        return bool(
            summary.get("ratio_defined")
            and summary.get("mean_over_G") is not None
            and summary["mean_over_G"] >= CAUSAL_THRESHOLDS["minimum_effect_over_G"]
            and summary["positive_world_count"] >= positive_required
            and summary["bootstrap_95"]["lower_95"] > 0.0
        )

    def specificity_gate(summary: Mapping[str, Any]) -> bool:
        return bool(
            summary.get("ratio_defined")
            and summary.get("mean_over_G") is not None
            and summary["mean_over_G"] >= CAUSAL_THRESHOLDS["minimum_specificity_over_G"]
            and summary["bootstrap_95"]["lower_95"] > 0.0
        )

    gates = {
        "crossfit_natural_coordinate_separation": bool(
            isinstance(coordinate_separation, Mapping)
            and isinstance(coordinate_separation.get("gates"), Mapping)
            and coordinate_separation["gates"].get("pass") is True
        ),
        "engineering": engineering_rows_pass
        and all(identity_logit_checks)
        and max(identity_errors.values()) <= CAUSAL_THRESHOLDS["identity_margin_tolerance"],
        "positional_rescue": primary_gate(effects["positional_rescue"]),
        "positional_damage": primary_gate(effects["positional_damage"]),
        "positional_combined_bootstrap": effects["positional_combined"]["bootstrap_95"]["lower_95"] > 0.0,
        "combined_minus_absolute_answer": specificity_gate(
            specificity["combined_minus_absolute_answer"]
        ),
        "combined_minus_absolute_null": specificity_gate(
            specificity["combined_minus_absolute_null"]
        ),
    }
    gates["causal_gap_closure"] = gates["positional_rescue"]
    gates["natural_use"] = gates["positional_damage"]
    gates["specificity"] = bool(
        gates["combined_minus_absolute_answer"] and gates["combined_minus_absolute_null"]
    )
    gates["pass"] = bool(
        gates["crossfit_natural_coordinate_separation"]
        and gates["engineering"]
        and gates["causal_gap_closure"]
        and gates["natural_use"]
        and gates["positional_combined_bootstrap"]
        and gates["specificity"]
    )
    return {
        "layer": layer,
        "n_worlds": expected_worlds,
        "n_pairs": len(pair_registry),
        "n_patch_rows": len(rows),
        "G": dict(gap_payload),
        "crossfit_coordinate_separation": (
            None if coordinate_separation is None else dict(coordinate_separation)
        ),
        "effects": effects,
        "specificity": specificity,
        "identity_absolute_margin_errors": identity_errors,
        "maximum_identity_absolute_margin_error": max(identity_errors.values()),
        "gates": gates,
    }


def select_localization_layer(layer_metrics: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Select the shallowest preregistered layer that passes every causal gate."""

    if set(layer_metrics) != set(LAYER_GRID):
        raise PositionalActivationAnalysisError("localization layer registry changed")
    if any(
        not isinstance(layer_metrics[layer].get("gates"), Mapping)
        or layer_metrics[layer]["gates"].get("engineering") is not True
        for layer in LAYER_GRID
    ):
        return {
            "selected_layer": None,
            "eligible_layers": [],
            "selection_score": None,
            "status": "V5_LOCALIZATION_ENGINEERING_INVALID",
        }
    eligible: list[tuple[float, int]] = []
    for layer in LAYER_GRID:
        metrics = layer_metrics[layer]
        gates = metrics.get("gates")
        if not isinstance(gates, Mapping):
            raise PositionalActivationAnalysisError("localization gates are missing")
        if gates.get("pass") is True:
            rescue = metrics["effects"]["positional_rescue"]["mean_over_G"]
            damage = metrics["effects"]["positional_damage"]["mean_over_G"]
            eligible.append((min(_finite(rescue, "rescue ratio"), _finite(damage, "damage ratio")), layer))
    if not eligible:
        return {
            "selected_layer": None,
            "eligible_layers": [],
            "selection_score": None,
            "status": "V5_NO_PREREGISTERED_CAUSAL_LAYER",
        }
    selected = min(layer for _score, layer in eligible)
    selected_score = next(score for score, layer in eligible if layer == selected)
    return {
        "selected_layer": selected,
        "eligible_layers": [layer for _score, layer in sorted(eligible, key=lambda item: item[1])],
        "selection_score": selected_score,
        "selection_rule": "shallowest_layer_passing_all_preregistered_gates",
        "status": "LOCALIZATION_LAYER_SELECTED",
    }


def final_status(metrics: Mapping[str, Any]) -> str:
    gates = metrics.get("gates")
    if not isinstance(gates, Mapping):
        raise PositionalActivationAnalysisError("final gates are missing")
    if gates.get("engineering") is not True:
        return "V5_FINAL_ENGINEERING_INVALID"
    if gates.get("crossfit_natural_coordinate_separation") is not True:
        return "V5_HOLDOUT_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED"
    if gates.get("causal_gap_closure") is not True:
        return "V5_NO_REPLICATED_CAUSAL_GAP_CLOSURE"
    if gates.get("positional_combined_bootstrap") is not True:
        return "V5_NO_REPLICATED_CAUSAL_GAP_CLOSURE"
    if gates.get("specificity") is not True:
        return "V5_NONSPECIFIC_CAUSAL_GAP_CLOSURE"
    if gates.get("natural_use") is not True:
        return "V5_CAUSAL_GAP_CLOSURE_NATURAL_USE_NOT_ESTABLISHED"
    return "V5_CAUSAL_TARGET_ORDER_CONTEXT_ACTIVATION_GAP_SUPPORTED"


def current_environment_identity() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner module is unavailable")
    packages: dict[str, str] = {}
    for name in (
        "huggingface-hub",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PositionalActivationAnalysisError(
                f"locked numerical package is unavailable: {name}"
            ) from error
    try:
        import torch
    except ImportError as error:
        raise PositionalActivationAnalysisError(
            "torch is required to replay the frozen numerical runtime"
        ) from error
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "runtime": {
            "device": runner.DEVICE,
            "dtype": runner.DTYPE,
            "attention_implementation": runner.ATTENTION_IMPLEMENTATION,
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def validate_environment_lock(dependency: Mapping[str, Any]) -> None:
    current = current_environment_identity()
    observed = {key: dependency.get(key) for key in current}
    if observed != current:
        raise PositionalActivationAnalysisError("frozen numerical environment changed")


def load_frozen_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently verify the zero-forward plan and its implementation locks."""

    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner module is unavailable")
    plan_manifest = _load_json(runner.DEFAULT_PLAN_MANIFEST)
    design = _load_json(runner.DEFAULT_DESIGN)
    plan = plan_manifest.get("plan")
    if not isinstance(plan, dict):
        raise PositionalActivationAnalysisError("plan manifest has no plan")
    plan_core = {key: value for key, value in plan.items() if key != "call_plan_sha256"}
    if canonical_sha256(plan_core) != plan.get("call_plan_sha256"):
        raise PositionalActivationAnalysisError("call-plan canonical hash changed")
    expected_plan_manifest = {
        "schema_version": runner.PLAN_MANIFEST_SCHEMA,
        "status": "PLAN_AND_DESIGN_FROZEN_NO_FORWARD",
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_path": str(runner.DEFAULT_DESIGN),
        "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
        "tokenization_receipt_file_sha256": file_sha256(runner.DEFAULT_TOKENIZATION_RECEIPT),
        "dependency_lock_file_sha256": file_sha256(runner.DEFAULT_DEPENDENCY_LOCK),
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
        "plan": plan,
    }
    if plan_manifest != expected_plan_manifest:
        raise PositionalActivationAnalysisError("zero-forward plan manifest changed")
    if (
        design.get("schema_version") != runner.DESIGN_SCHEMA
        or design.get("call_plan_sha256") != plan["call_plan_sha256"]
        or design.get("expected_counts") != runner.EXPECTED_COUNTS
        or design.get("model_calls") != 0
        or design.get("generation_used") is not False
        or design.get("biological_model_calls") != 0
    ):
        raise PositionalActivationAnalysisError("frozen design changed")
    locks = design.get("locks")
    if not isinstance(locks, Mapping):
        raise PositionalActivationAnalysisError("design implementation locks are missing")
    implementation_names = (
        "runner",
        "analyzer",
        "model_hooks",
        "fixture_builder",
        "fixture",
        "fixture_manifest",
        "preregistration",
        "fixture_tests",
        "runner_tests",
        "analyzer_tests",
    )
    dependency_binding = locks.get("dependency_lock")
    tokenization_binding = locks.get("tokenization_receipt")
    for label, binding in (
        ("dependency lock", dependency_binding),
        ("tokenization receipt", tokenization_binding),
    ):
        if not isinstance(binding, Mapping):
            raise PositionalActivationAnalysisError(f"{label} binding is missing")
        path = Path(str(binding.get("path", "")))
        value = _load_json(path)
        core = {key: item for key, item in value.items() if key != "canonical_sha256"}
        if (
            file_sha256(path) != binding.get("file_sha256")
            or value.get("canonical_sha256") != binding.get("canonical_sha256")
            or canonical_sha256(core) != value.get("canonical_sha256")
        ):
            raise PositionalActivationAnalysisError(f"{label} changed")
    dependency = _load_json(Path(str(dependency_binding["path"])))
    if dependency.get("schema_version") != runner.DEPENDENCY_LOCK_SCHEMA:
        raise PositionalActivationAnalysisError("dependency lock schema changed")
    validate_environment_lock(dependency)
    implementation_registry = dependency.get("implementation_files")
    if not isinstance(implementation_registry, Mapping) or set(implementation_registry) != set(
        implementation_names
    ):
        raise PositionalActivationAnalysisError("dependency implementation registry changed")
    for name in implementation_names:
        binding = locks.get(name)
        if not isinstance(binding, Mapping):
            raise PositionalActivationAnalysisError(f"design implementation lock is missing: {name}")
        if dict(binding) != implementation_registry[name]:
            raise PositionalActivationAnalysisError(
                f"design and dependency implementation locks differ: {name}"
            )
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise PositionalActivationAnalysisError(f"frozen implementation changed: {name}")
    return plan, design


def _logical_array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype != np.dtype("<f4") or not value.flags.c_contiguous:
        raise PositionalActivationAnalysisError("array is not contiguous little-endian float32")
    digest = hashlib.sha256()
    rows = 1 if value.ndim == 0 else len(value)
    step = max(1, min(rows, 16))
    if value.ndim == 0:
        digest.update(memoryview(value.reshape(1).view(np.uint8)))
    else:
        for start in range(0, rows, step):
            chunk = np.ascontiguousarray(value[start : start + step])
            if not np.isfinite(chunk).all():
                raise PositionalActivationAnalysisError("array sidecar contains nonfinite values")
            digest.update(memoryview(chunk.view(np.uint8)))
    return digest.hexdigest()


def _load_receipt_array(receipt: Mapping[str, Any], *, mmap: bool) -> np.ndarray:
    path = Path(str(receipt.get("path", "")))
    if (
        not path.is_file()
        or file_sha256(path) != receipt.get("file_sha256")
        or path.stat().st_size != receipt.get("size_bytes")
    ):
        raise PositionalActivationAnalysisError("array file receipt changed")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    except (OSError, ValueError) as error:
        raise PositionalActivationAnalysisError(f"cannot load array sidecar: {path}") from error
    if (
        list(array.shape) != receipt.get("shape")
        or receipt.get("dtype") != "<f4"
        or array.dtype != np.dtype("<f4")
        or _logical_array_sha256(array) != receipt.get("logical_sha256")
    ):
        raise PositionalActivationAnalysisError("array logical receipt changed")
    return array


def load_execution_phase(
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify an execution manifest without invoking runner validation code."""

    if runner is None or phase not in runner.EXPECTED_COUNTS:
        raise PositionalActivationAnalysisError("execution phase is invalid")
    paths = runner.PHASE_PATHS[phase]
    manifest_path = paths["manifest"]
    manifest = _load_json(manifest_path)
    expected_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "design_file_sha256",
        "plan_manifest_file_sha256",
        "runner_sha256",
        "analyzer_sha256",
        "preregistration_sha256",
        "attempt",
        "records",
        "raw_logits_shards",
        "activations",
        "patched_activations",
        "prerequisite_bindings",
        "selected_layer",
        "phase_model_calls",
        "cumulative_model_calls",
        "generation_used",
        "biological_model_calls",
        "partial_resume_allowed",
    }
    expected_count = runner.EXPECTED_COUNTS[phase]
    if set(manifest) != expected_keys:
        raise PositionalActivationAnalysisError("execution manifest field registry changed")
    if (
        manifest["schema_version"] != runner.EXECUTION_MANIFEST_SCHEMA
        or manifest["status"] != "EXECUTION_COMPLETE_NOT_ANALYZED"
        or manifest["phase"] != phase
        or manifest["call_plan_sha256"] != plan["call_plan_sha256"]
        or manifest["design_file_sha256"] != file_sha256(runner.DEFAULT_DESIGN)
        or manifest["plan_manifest_file_sha256"] != file_sha256(runner.DEFAULT_PLAN_MANIFEST)
        or manifest["runner_sha256"] != file_sha256(Path(runner.__file__))
        or manifest["analyzer_sha256"] != design["locks"]["analyzer"]["sha256"]
        or manifest["preregistration_sha256"] != file_sha256(runner.FROZEN_PREREGISTRATION)
        or manifest["phase_model_calls"] != expected_count
        or manifest["cumulative_model_calls"] != runner.EXPECTED_CUMULATIVE_CALLS[phase]
        or manifest["generation_used"] is not False
        or manifest["biological_model_calls"] != 0
        or manifest["partial_resume_allowed"] is not False
    ):
        raise PositionalActivationAnalysisError("execution manifest identity changed")
    if phase == "fit-baseline":
        expected_prerequisites: dict[str, str] = {}
        expected_selected_layer = None
        authority_path = None
        expected_authority = None
    elif phase == "localization-baseline":
        authority_bindings = {
            "fit_baseline_execution_manifest_file_sha256": file_sha256(
                runner.PHASE_PATHS["fit-baseline"]["manifest"]
            ),
            "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        }
        authority_path = runner.DEFAULT_LOCALIZATION_BASELINE_ENTRY
        expected_authority = _authority(
            schema=LOCALIZATION_BASELINE_ENTRY_SCHEMA,
            status="LOCALIZATION_BASELINE_AUTHORIZED",
            plan=plan,
            bindings=authority_bindings,
        )
        expected_prerequisites = {
            **authority_bindings,
            "localization_baseline_entry_file_sha256": file_sha256(
                runner.DEFAULT_LOCALIZATION_BASELINE_ENTRY
            ),
        }
        expected_selected_layer = None
    elif phase == "localization-patch":
        authority_bindings = {
            "localization_baseline_execution_manifest_file_sha256": file_sha256(
                runner.PHASE_PATHS["localization-baseline"]["manifest"]
            ),
            "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        }
        authority_path = runner.DEFAULT_LOCALIZATION_PATCH_ENTRY
        expected_authority = _authority(
            schema=LOCALIZATION_PATCH_ENTRY_SCHEMA,
            status="LOCALIZATION_PATCH_AUTHORIZED",
            plan=plan,
            bindings=authority_bindings,
        )
        expected_prerequisites = {
            **authority_bindings,
            "localization_patch_entry_file_sha256": file_sha256(
                runner.DEFAULT_LOCALIZATION_PATCH_ENTRY
            ),
        }
        expected_selected_layer = None
    else:
        layer_lock = _load_json(runner.DEFAULT_LAYER_LOCK)
        selected = layer_lock.get("selected_layer")
        if selected not in LAYER_GRID:
            raise PositionalActivationAnalysisError("selected-layer lock changed")
        if phase == "holdout-baseline":
            authority_bindings = {
                "localization_patch_execution_manifest_file_sha256": file_sha256(
                    runner.PHASE_PATHS["localization-patch"]["manifest"]
                ),
                "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
                "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
            }
            authority_path = runner.DEFAULT_HOLDOUT_BASELINE_ENTRY
            expected_authority = _authority(
                schema=HOLDOUT_BASELINE_ENTRY_SCHEMA,
                status="HOLDOUT_BASELINE_AUTHORIZED",
                plan=plan,
                bindings=authority_bindings,
            )
            expected_prerequisites = {
                **authority_bindings,
                "holdout_baseline_entry_file_sha256": file_sha256(
                    runner.DEFAULT_HOLDOUT_BASELINE_ENTRY
                ),
            }
        else:
            authority_bindings = {
                "holdout_baseline_execution_manifest_file_sha256": file_sha256(
                    runner.PHASE_PATHS["holdout-baseline"]["manifest"]
                ),
                "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
                "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
            }
            authority_path = runner.DEFAULT_HOLDOUT_PATCH_ENTRY
            expected_authority = _authority(
                schema=HOLDOUT_PATCH_ENTRY_SCHEMA,
                status="HOLDOUT_PATCH_AUTHORIZED",
                plan=plan,
                bindings=authority_bindings,
            )
            expected_prerequisites = {
                **authority_bindings,
                "holdout_patch_entry_file_sha256": file_sha256(
                    runner.DEFAULT_HOLDOUT_PATCH_ENTRY
                ),
            }
        expected_selected_layer = selected
    if authority_path is not None:
        authority = _load_json(authority_path)
        validate_authority_content(authority, expected_authority)
    if phase in {"holdout-baseline", "holdout-patch"}:
        expected_layer_lock = {
            "schema_version": LAYER_LOCK_SCHEMA,
            "status": "LOCALIZATION_LAYER_SELECTED",
            "call_plan_sha256": plan["call_plan_sha256"],
            "selected_layer": expected_selected_layer,
            "selection_rule": "shallowest_layer_passing_all_preregistered_gates",
            "localization_patch_execution_manifest_file_sha256": file_sha256(
                runner.PHASE_PATHS["localization-patch"]["manifest"]
            ),
            "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
            "localization_analysis_file_sha256": file_sha256(
                LOCALIZATION_ANALYSIS_PATH
            ),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        if _load_json(runner.DEFAULT_LAYER_LOCK) != expected_layer_lock:
            raise PositionalActivationAnalysisError("selected-layer lock content changed")
    if (
        manifest["prerequisite_bindings"] != expected_prerequisites
        or manifest["selected_layer"] != expected_selected_layer
    ):
        raise PositionalActivationAnalysisError("execution phase authority graph changed")
    attempt_binding = manifest["attempt"]
    if not isinstance(attempt_binding, Mapping) or set(attempt_binding) != {"path", "file_sha256"}:
        raise PositionalActivationAnalysisError("attempt receipt changed")
    if (
        Path(str(attempt_binding["path"])).resolve() != paths["attempt"].resolve()
        or file_sha256(paths["attempt"]) != attempt_binding["file_sha256"]
    ):
        raise PositionalActivationAnalysisError("attempt file binding changed")
    attempt = _load_json(paths["attempt"])
    prior_calls = 0
    phase_order = tuple(runner.EXPECTED_COUNTS)
    phase_index = phase_order.index(phase)
    if phase_index:
        prior_calls = runner.EXPECTED_CUMULATIVE_CALLS[phase_order[phase_index - 1]]
    expected_attempt = {
        "schema_version": runner.ATTEMPT_SCHEMA,
        "status": "EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(runner.DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(runner.DEFAULT_PLAN_MANIFEST),
        "runner_sha256": file_sha256(Path(runner.__file__)),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "preregistration_sha256": file_sha256(runner.FROZEN_PREREGISTRATION),
        "prerequisite_bindings": dict(manifest["prerequisite_bindings"]),
        "predeclared_phase_calls": expected_count,
        "model_calls_before_attempt": prior_calls,
        "generation_used": False,
        "biological_model_calls": 0,
    }
    if attempt != expected_attempt:
        raise PositionalActivationAnalysisError("execution attempt does not reconstruct")
    record_receipt = manifest["records"]
    if not isinstance(record_receipt, Mapping):
        raise PositionalActivationAnalysisError("record receipt is invalid")
    records_path = Path(str(record_receipt.get("path", "")))
    if (
        records_path.resolve() != paths["records"].resolve()
        or not records_path.is_file()
        or file_sha256(records_path) != record_receipt.get("file_sha256")
        or records_path.stat().st_size != record_receipt.get("size_bytes")
    ):
        raise PositionalActivationAnalysisError("record file receipt changed")
    records = _load_jsonl(records_path)
    if (
        len(records) != expected_count
        or record_receipt.get("count") != expected_count
        or canonical_sha256(records) != record_receipt.get("canonical_sha256")
        or [row.get("call_index") for row in records] != list(range(expected_count))
    ):
        raise PositionalActivationAnalysisError("record registry changed")
    expected_specs = plan["raw_logits_shards"][phase]
    raw_receipts = manifest["raw_logits_shards"]
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(expected_specs):
        raise PositionalActivationAnalysisError("raw-logit shard registry changed")
    raw_shards: list[np.ndarray] = []
    for spec, receipt in zip(expected_specs, raw_receipts, strict=True):
        if not isinstance(receipt, Mapping) or any(receipt.get(key) != value for key, value in spec.items()):
            raise PositionalActivationAnalysisError("raw-logit shard plan changed")
        raw_shards.append(_load_receipt_array(receipt, mmap=True))
    sidecar_key = "activations" if phase.endswith("baseline") else "patched_activations"
    other_key = "patched_activations" if phase.endswith("baseline") else "activations"
    if manifest[other_key] is not None or not isinstance(manifest[sidecar_key], Mapping):
        raise PositionalActivationAnalysisError("execution sidecar registry changed")
    sidecar = _load_receipt_array(manifest[sidecar_key], mmap=False)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "records": records,
        "raw_shards": raw_shards,
        "sidecar": sidecar,
        "sidecar_receipt": manifest[sidecar_key],
    }


def _validated_baseline_phase(
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    role: str,
    layers: Sequence[int],
) -> dict[str, Any]:
    bundle = load_execution_phase(phase, plan, design)
    records = validate_baseline_records(
        plan,
        bundle["records"],
        bundle["sidecar"],
        bundle["raw_shards"],
        phase=phase,
        role=role,
        layers=layers,
    )
    expected_map = {
        record["cell_id"]: {
            "activation_row": record["activation_row"],
            "activation_sha256": record["activation_sha256"],
            "activation_layer_sha256": record["activation_layer_sha256"],
        }
        for record in records
        if record["activation_row"] is not None
    }
    if bundle["sidecar_receipt"].get("logical_id_map") != expected_map:
        raise PositionalActivationAnalysisError("baseline activation logical map changed")
    bundle["records"] = records
    return bundle


def _basis_from_fit(
    records: Sequence[Mapping[str, Any]], activations: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    composition = [row for row in records if row["family"] == "composition"]
    if len(composition) != FIT_WORLDS * 32:
        raise PositionalActivationAnalysisError("fit composition registry changed")
    basis_rows: list[np.ndarray] = []
    public_layers: dict[str, Any] = {}
    for layer_position, layer in enumerate(LAYER_GRID):
        world_activations: dict[str, np.ndarray] = {}
        world_factors: dict[str, list[Mapping[str, Any]]] = {}
        for world in sorted({row["world_id"] for row in composition}):
            members = sorted(
                (row for row in composition if row["world_id"] == world),
                key=lambda row: int(row["activation_row"]),
            )
            world_activations[world] = np.stack(
                [activations[int(row["activation_row"]), layer_position] for row in members]
            )
            world_factors[world] = [row["factors"] for row in members]
        result = construct_layer_basis(world_activations, world_factors)
        basis_rows.append(
            np.stack(
                [
                    result["center"],
                    result["directions"]["positional"],
                    result["directions"]["answer"],
                    result["directions"]["null"],
                ]
            )
        )
        public_layers[str(layer)] = {
            key: value
            for key, value in result.items()
            if key not in {"center", "directions"}
        }
        public_layers[str(layer)]["center_sha256"] = f32_sha256(result["center"])
    basis = np.ascontiguousarray(np.stack(basis_rows), dtype="<f4")
    if basis.shape[0:2] != (len(LAYER_GRID), 4) or not np.isfinite(basis).all():
        raise PositionalActivationAnalysisError("fit basis sidecar dimensions changed")
    gates = {
        "each_preregistered_layer": all(
            public_layers[str(layer)]["gates"]["pass"] for layer in LAYER_GRID
        )
    }
    gates["pass"] = all(gates.values())
    return basis, {"layers": public_layers, "gates": gates}


def _basis_sidecar_receipt(path: Path, basis: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "logical_sha256": f32_sha256(basis),
        "shape": list(basis.shape),
        "dtype": "<f4",
        "size_bytes": path.stat().st_size,
    }


def _load_locked_basis(plan: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    lock = _load_json(runner.DEFAULT_BASIS_LOCK)
    required = {
        "status": "FIT_BASIS_ADMITTED",
        "admitted": True,
        "call_plan_sha256": plan["call_plan_sha256"],
        "layers": list(LAYER_GRID),
        "axis_registry": ["center", "positional", "answer", "null"],
    }
    if any(lock.get(key) != value for key, value in required.items()):
        raise PositionalActivationAnalysisError("fit basis lock changed")
    receipt = lock.get("sidecar")
    if not isinstance(receipt, Mapping):
        raise PositionalActivationAnalysisError("fit basis sidecar receipt is missing")
    if Path(str(receipt.get("path", ""))).resolve() != runner.DEFAULT_BASIS_SIDECAR.resolve():
        raise PositionalActivationAnalysisError("fit basis sidecar path changed")
    basis = _load_receipt_array(receipt, mmap=False)
    expected_shape = (len(LAYER_GRID), 4, int(plan["model"]["hidden_width"]))
    if basis.shape != expected_shape:
        raise PositionalActivationAnalysisError("fit basis shape changed")
    for layer_position in range(len(LAYER_GRID)):
        for axis_position in range(1, 4):
            if abs(float(np.linalg.norm(basis[layer_position, axis_position])) - 1.0) > 1e-5:
                raise PositionalActivationAnalysisError("locked direction is not unit norm")
    expected_lock = {
        "schema_version": BASIS_LOCK_SCHEMA,
        "status": "FIT_BASIS_ADMITTED",
        "admitted": True,
        "call_plan_sha256": plan["call_plan_sha256"],
        "layers": list(LAYER_GRID),
        "axis_registry": ["center", "positional", "answer", "null"],
        "sidecar": _basis_sidecar_receipt(runner.DEFAULT_BASIS_SIDECAR, basis),
        "fit_baseline_execution_manifest_file_sha256": file_sha256(
            runner.PHASE_PATHS["fit-baseline"]["manifest"]
        ),
        "fit_analysis_file_sha256": file_sha256(FIT_ANALYSIS_PATH),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    if lock != expected_lock:
        raise PositionalActivationAnalysisError("fit basis lock content changed")
    return basis, lock


def _validated_patch_phase(
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    role: str,
    baseline_bundle: Mapping[str, Any],
    basis: np.ndarray,
    selected_layer: int | None,
) -> dict[str, Any]:
    bundle = load_execution_phase(phase, plan, design)
    records = validate_patch_records(
        plan,
        bundle["records"],
        bundle["sidecar"],
        bundle["raw_shards"],
        baseline_bundle["records"],
        baseline_bundle["sidecar"],
        basis,
        phase=phase,
        role=role,
        selected_layer=selected_layer,
    )
    expected_map = {
        record["record_id"]: {
            "patched_activation_row": record["patched_activation_row"],
            "patched_activation_sha256": record["patched_activation_sha256"],
        }
        for record in records
    }
    if bundle["sidecar_receipt"].get("logical_id_map") != expected_map:
        raise PositionalActivationAnalysisError("patched activation logical map changed")
    bundle["records"] = records
    return bundle


def _behavior_stage_status(behavior: Mapping[str, Any], *, role: str) -> str | None:
    gates = behavior["gates"]
    component_names = (
        "retrieval_lookup_overall",
        "retrieval_lookup_each_world",
        "retrieval_lookup_each_stratum",
        "fixed_panel_direct_prerequisites",
    )
    if not all(gates[name] for name in component_names):
        return {
            "fit": "V5_FIT_COMPONENT_ADMISSION_FAIL",
            "localization": "V5_LOCALIZATION_COMPONENT_ADMISSION_FAIL",
            "holdout": "V5_HOLDOUT_COMPONENT_ADMISSION_FAIL",
        }[role]
    gap_names = (
        "target_first_composition",
        "target_second_error",
        "composition_accuracy_gap",
        "G_bootstrap_lower_positive",
    ) + (("positive_G_worlds",) if role != "fit" else ())
    if not all(gates[name] for name in gap_names):
        if role == "fit":
            return "V5_FIT_COMPONENT_ADMISSION_FAIL"
        return {
            "localization": "V5_LOCALIZATION_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED",
            "holdout": "V5_HOLDOUT_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED",
        }[role]
    return None


def _authority(
    *, schema: str, status: str, plan: Mapping[str, Any], bindings: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "status": status,
        "call_plan_sha256": plan["call_plan_sha256"],
        **dict(bindings),
        "model_calls_issued_by_analyzer": 0,
        "generation_used": False,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }


def validate_authority_content(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject any status, prior-hash, scope, or zero-call authority mutation."""

    if dict(observed) != dict(expected):
        raise PositionalActivationAnalysisError("analyzer authorization content changed")


def analyze_fit() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    plan, design = load_frozen_plan()
    try:
        bundle = _validated_baseline_phase(
            "fit-baseline", plan, design, role="fit", layers=LAYER_GRID
        )
        behavior = behavioral_admission(
            bundle["records"], role="fit", expected_worlds=FIT_WORLDS
        )
    except PositionalActivationAnalysisError as error:
        analysis = {
            "schema_version": FIT_ANALYSIS_SCHEMA,
            "status": "V5_ENGINEERING_INVALID",
            "error": str(error),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(FIT_ANALYSIS_PATH, analysis)
        return analysis
    behavior_status = _behavior_stage_status(behavior, role="fit")
    if behavior_status is not None:
        analysis = {
            "schema_version": FIT_ANALYSIS_SCHEMA,
            "status": behavior_status,
            "behavior": behavior,
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(FIT_ANALYSIS_PATH, analysis)
        return analysis
    try:
        basis, basis_analysis = _basis_from_fit(bundle["records"], bundle["sidecar"])
    except PositionalActivationAnalysisError as error:
        analysis = {
            "schema_version": FIT_ANALYSIS_SCHEMA,
            "status": "V5_FIT_BASIS_INVALID",
            "behavior": behavior,
            "basis_error": str(error),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(FIT_ANALYSIS_PATH, analysis)
        return analysis
    if not basis_analysis["gates"]["pass"]:
        analysis = {
            "schema_version": FIT_ANALYSIS_SCHEMA,
            "status": "V5_FIT_BASIS_INVALID",
            "behavior": behavior,
            "basis": basis_analysis,
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(FIT_ANALYSIS_PATH, analysis)
        return analysis
    _write_array(runner.DEFAULT_BASIS_SIDECAR, basis)
    analysis = {
        "schema_version": FIT_ANALYSIS_SCHEMA,
        "status": "FIT_BASIS_ADMITTED",
        "behavior": behavior,
        "basis": basis_analysis,
        "fit_baseline_execution_manifest_file_sha256": file_sha256(bundle["manifest_path"]),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(FIT_ANALYSIS_PATH, analysis)
    basis_lock = {
        "schema_version": BASIS_LOCK_SCHEMA,
        "status": "FIT_BASIS_ADMITTED",
        "admitted": True,
        "call_plan_sha256": plan["call_plan_sha256"],
        "layers": list(LAYER_GRID),
        "axis_registry": ["center", "positional", "answer", "null"],
        "sidecar": _basis_sidecar_receipt(runner.DEFAULT_BASIS_SIDECAR, basis),
        "fit_baseline_execution_manifest_file_sha256": file_sha256(bundle["manifest_path"]),
        "fit_analysis_file_sha256": file_sha256(FIT_ANALYSIS_PATH),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(runner.DEFAULT_BASIS_LOCK, basis_lock)
    entry = _authority(
        schema=LOCALIZATION_BASELINE_ENTRY_SCHEMA,
        status="LOCALIZATION_BASELINE_AUTHORIZED",
        plan=plan,
        bindings={
            "fit_baseline_execution_manifest_file_sha256": file_sha256(bundle["manifest_path"]),
            "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        },
    )
    _write_json(runner.DEFAULT_LOCALIZATION_BASELINE_ENTRY, entry)
    return analysis


def analyze_localization_baseline() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    plan, design = load_frozen_plan()
    basis, _basis_lock = _load_locked_basis(plan)
    try:
        bundle = _validated_baseline_phase(
            "localization-baseline", plan, design, role="localization", layers=LAYER_GRID
        )
        behavior = behavioral_admission(
            bundle["records"], role="localization", expected_worlds=LOCALIZATION_WORLDS
        )
        coordinate_separation = {
            str(layer): crossfit_coordinate_separation(
                bundle["records"],
                bundle["sidecar"],
                basis,
                role="localization",
                layer=layer,
                expected_worlds=LOCALIZATION_WORLDS,
            )
            for layer in LAYER_GRID
        }
    except PositionalActivationAnalysisError as error:
        analysis = {
            "schema_version": LOCALIZATION_BASELINE_SCHEMA,
            "status": "V5_ENGINEERING_INVALID",
            "error": str(error),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(LOCALIZATION_BASELINE_ANALYSIS_PATH, analysis)
        return analysis
    status = _behavior_stage_status(behavior, role="localization")
    if status is None and not any(
        coordinate_separation[str(layer)]["gates"]["pass"] for layer in LAYER_GRID
    ):
        status = "V5_LOCALIZATION_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED"
    analysis = {
        "schema_version": LOCALIZATION_BASELINE_SCHEMA,
        "status": status or "LOCALIZATION_BASELINE_ADMITTED",
        "behavior": behavior,
        "crossfit_coordinate_separation": coordinate_separation,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(LOCALIZATION_BASELINE_ANALYSIS_PATH, analysis)
    if status is None:
        entry = _authority(
            schema=LOCALIZATION_PATCH_ENTRY_SCHEMA,
            status="LOCALIZATION_PATCH_AUTHORIZED",
            plan=plan,
            bindings={
                "localization_baseline_execution_manifest_file_sha256": file_sha256(
                    bundle["manifest_path"]
                ),
                "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
            },
        )
        _write_json(runner.DEFAULT_LOCALIZATION_PATCH_ENTRY, entry)
    return analysis


def analyze_localization_patch() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    plan, design = load_frozen_plan()
    basis, _basis_lock = _load_locked_basis(plan)
    baseline = _validated_baseline_phase(
        "localization-baseline", plan, design, role="localization", layers=LAYER_GRID
    )
    behavior = behavioral_admission(
        baseline["records"], role="localization", expected_worlds=LOCALIZATION_WORLDS
    )
    if _behavior_stage_status(behavior, role="localization") is not None:
        raise PositionalActivationAnalysisError("localization patch ran without admitted baseline")
    try:
        coordinate_separation = {
            layer: crossfit_coordinate_separation(
                baseline["records"],
                baseline["sidecar"],
                basis,
                role="localization",
                layer=layer,
                expected_worlds=LOCALIZATION_WORLDS,
            )
            for layer in LAYER_GRID
        }
        patches = _validated_patch_phase(
            "localization-patch",
            plan,
            design,
            role="localization",
            baseline_bundle=baseline,
            basis=basis,
            selected_layer=None,
        )
        layer_metrics = {
            layer: causal_metrics(
                patches["records"],
                behavior,
                layer=layer,
                expected_worlds=LOCALIZATION_WORLDS,
                coordinate_separation=coordinate_separation[layer],
            )
            for layer in LAYER_GRID
        }
    except PositionalActivationAnalysisError as error:
        analysis = {
            "schema_version": LOCALIZATION_ANALYSIS_SCHEMA,
            "status": "V5_LOCALIZATION_ENGINEERING_INVALID",
            "error": str(error),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(LOCALIZATION_ANALYSIS_PATH, analysis)
        return analysis
    selection = select_localization_layer(layer_metrics)
    status = selection["status"]
    analysis = {
        "schema_version": LOCALIZATION_ANALYSIS_SCHEMA,
        "status": status,
        "behavior": behavior,
        "layer_metrics": {str(layer): layer_metrics[layer] for layer in LAYER_GRID},
        "selection": selection,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(LOCALIZATION_ANALYSIS_PATH, analysis)
    selected = selection["selected_layer"]
    if selected is None:
        return analysis
    layer_lock = {
        "schema_version": LAYER_LOCK_SCHEMA,
        "status": "LOCALIZATION_LAYER_SELECTED",
        "call_plan_sha256": plan["call_plan_sha256"],
        "selected_layer": selected,
        "selection_rule": selection["selection_rule"],
        "localization_patch_execution_manifest_file_sha256": file_sha256(
            patches["manifest_path"]
        ),
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "localization_analysis_file_sha256": file_sha256(LOCALIZATION_ANALYSIS_PATH),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(runner.DEFAULT_LAYER_LOCK, layer_lock)
    entry = _authority(
        schema=HOLDOUT_BASELINE_ENTRY_SCHEMA,
        status="HOLDOUT_BASELINE_AUTHORIZED",
        plan=plan,
        bindings={
            "localization_patch_execution_manifest_file_sha256": file_sha256(
                patches["manifest_path"]
            ),
            "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
            "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        },
    )
    _write_json(runner.DEFAULT_HOLDOUT_BASELINE_ENTRY, entry)
    return analysis


def analyze_holdout_baseline() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    plan, design = load_frozen_plan()
    basis, _basis_lock = _load_locked_basis(plan)
    layer_lock = _load_json(runner.DEFAULT_LAYER_LOCK)
    selected = layer_lock.get("selected_layer")
    if selected not in LAYER_GRID:
        raise PositionalActivationAnalysisError("holdout selected layer is invalid")
    try:
        bundle = _validated_baseline_phase(
            "holdout-baseline", plan, design, role="holdout", layers=(selected,)
        )
        behavior = behavioral_admission(
            bundle["records"], role="holdout", expected_worlds=HOLDOUT_WORLDS
        )
        coordinate_separation = crossfit_coordinate_separation(
            bundle["records"],
            bundle["sidecar"],
            basis,
            role="holdout",
            layer=selected,
            expected_worlds=HOLDOUT_WORLDS,
        )
    except PositionalActivationAnalysisError as error:
        analysis = {
            "schema_version": HOLDOUT_BASELINE_SCHEMA,
            "status": "V5_ENGINEERING_INVALID",
            "error": str(error),
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_json(HOLDOUT_BASELINE_ANALYSIS_PATH, analysis)
        return analysis
    status = _behavior_stage_status(behavior, role="holdout")
    if status is None and coordinate_separation["gates"]["pass"] is not True:
        status = "V5_HOLDOUT_TARGET_ORDER_CONTEXT_GAP_NOT_REPLICATED"
    analysis = {
        "schema_version": HOLDOUT_BASELINE_SCHEMA,
        "status": status or "HOLDOUT_BASELINE_ADMITTED",
        "selected_layer": selected,
        "behavior": behavior,
        "crossfit_coordinate_separation": coordinate_separation,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(HOLDOUT_BASELINE_ANALYSIS_PATH, analysis)
    if status is None:
        entry = _authority(
            schema=HOLDOUT_PATCH_ENTRY_SCHEMA,
            status="HOLDOUT_PATCH_AUTHORIZED",
            plan=plan,
            bindings={
                "holdout_baseline_execution_manifest_file_sha256": file_sha256(
                    bundle["manifest_path"]
                ),
                "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
                "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
            },
        )
        _write_json(runner.DEFAULT_HOLDOUT_PATCH_ENTRY, entry)
    return analysis


def _render_final_markdown(analysis: Mapping[str, Any]) -> str:
    status = analysis["status"]
    metrics = analysis.get("metrics", {})
    layer = analysis.get("selected_layer")
    lines = [
        "# V5 causal TARGET-order/context activation-gap result",
        "",
        f"- Status: `{status}`",
        f"- Selected layer: `{layer}`",
        "- Scope: synthetic model/prompt/TARGET-token/layer-specific causal TARGET-order/context-state mediation only.",
        "- The design does not separate absolute token position from interference caused by the preceding OTHER fact.",
        "- Excluded: biology, latent biological knowledge, physical law, and model-family generalization.",
    ]
    if isinstance(metrics, Mapping) and "effects" in metrics:
        lines.extend(
            [
                "",
                "## Confirmatory holdout",
                "",
                f"- Positional rescue / G: `{metrics['effects']['positional_rescue']['mean_over_G']}`",
                f"- Positional damage / G: `{metrics['effects']['positional_damage']['mean_over_G']}`",
                f"- Passing worlds (rescue, damage): `{metrics['effects']['positional_rescue']['positive_world_count']}`, "
                f"`{metrics['effects']['positional_damage']['positive_world_count']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 한국어 해석",
            "",
            f"- 최종 상태: `{status}`",
            f"- 선택 층: `{layer}`",
            "- 허용되는 결론은 고정된 합성 과제의 특정 모델·프롬프트·TARGET 토큰·층에서 TARGET 순서/선행 문맥 상태가 인과적으로 매개한다는 범위뿐입니다.",
            "- 이 설계는 절대 토큰 위치와 앞선 OTHER fact의 간섭을 분리하지 않습니다.",
            "- 생물학, 잠재 생물학 지식, 물리 법칙 또는 모델 계열 일반화의 증거가 아닙니다.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_final() -> dict[str, Any]:
    if runner is None:
        raise PositionalActivationAnalysisError("V5 runner is unavailable")
    plan, design = load_frozen_plan()
    basis, _basis_lock = _load_locked_basis(plan)
    layer_lock = _load_json(runner.DEFAULT_LAYER_LOCK)
    selected = layer_lock.get("selected_layer")
    if selected not in LAYER_GRID:
        raise PositionalActivationAnalysisError("final selected layer is invalid")
    baseline = _validated_baseline_phase(
        "holdout-baseline", plan, design, role="holdout", layers=(selected,)
    )
    behavior = behavioral_admission(
        baseline["records"], role="holdout", expected_worlds=HOLDOUT_WORLDS
    )
    baseline_status = _behavior_stage_status(behavior, role="holdout")
    if baseline_status is not None:
        raise PositionalActivationAnalysisError("holdout patch ran without admitted baseline")
    try:
        coordinate_separation = crossfit_coordinate_separation(
            baseline["records"],
            baseline["sidecar"],
            basis,
            role="holdout",
            layer=selected,
            expected_worlds=HOLDOUT_WORLDS,
        )
        patches = _validated_patch_phase(
            "holdout-patch",
            plan,
            design,
            role="holdout",
            baseline_bundle=baseline,
            basis=basis,
            selected_layer=selected,
        )
        metrics = causal_metrics(
            patches["records"],
            behavior,
            layer=selected,
            expected_worlds=HOLDOUT_WORLDS,
            coordinate_separation=coordinate_separation,
        )
        status = final_status(metrics)
    except PositionalActivationAnalysisError as error:
        metrics = {"engineering_error": str(error)}
        status = "V5_FINAL_ENGINEERING_INVALID"
        patches = None
    analysis = {
        "schema_version": FINAL_ANALYSIS_SCHEMA,
        "status": status,
        "selected_layer": selected,
        "behavior": behavior,
        "metrics": metrics,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(FINAL_ANALYSIS_PATH, analysis)
    markdown = _render_final_markdown(analysis)
    _write_frozen_bytes(FINAL_MARKDOWN_PATH, markdown.encode("utf-8"))
    manifest = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "status": status,
        "call_plan_sha256": plan["call_plan_sha256"],
        "analysis": {
            "path": str(FINAL_ANALYSIS_PATH),
            "file_sha256": file_sha256(FINAL_ANALYSIS_PATH),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "markdown": {
            "path": str(FINAL_MARKDOWN_PATH),
            "file_sha256": file_sha256(FINAL_MARKDOWN_PATH),
        },
        "holdout_patch_execution_manifest_file_sha256": (
            None if patches is None else file_sha256(patches["manifest_path"])
        ),
        "basis_lock_file_sha256": file_sha256(runner.DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(runner.DEFAULT_LAYER_LOCK),
        "model_calls_issued_by_analyzer": 0,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_json(FINAL_MANIFEST_PATH, manifest)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "analyze-fit",
            "analyze-localization-baseline",
            "analyze-localization-patch",
            "analyze-holdout-baseline",
            "analyze-final",
        ),
    )
    args = parser.parse_args()
    actions = {
        "analyze-fit": analyze_fit,
        "analyze-localization-baseline": analyze_localization_baseline,
        "analyze-localization-patch": analyze_localization_patch,
        "analyze-holdout-baseline": analyze_holdout_baseline,
        "analyze-final": analyze_final,
    }
    result = actions[args.stage]()
    print(canonical_json({"stage": args.stage, "status": result["status"]}))


if __name__ == "__main__":
    main()

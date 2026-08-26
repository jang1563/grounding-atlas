#!/usr/bin/env python3
"""Staged analysis for the frozen V6A-R2 natural-token topology study.

The discovery gate can consume only discovery-component results.  The final
stage validates both sealed executions and gates confirmation components before
calling any topology routine.  This module performs no model forward or
generation call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER_MODULE = "eval.run_coherent_readout_v6a_r2_topology"
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_r2_topology"
    / "qwen2.5-7b-instruct"
)
PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
DESIGN = RESULT_ROOT / "design.json"
DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"

DISCOVERY_ATTEMPT = RESULT_ROOT / "discovery_components_attempt.json"
DISCOVERY_RECORDS = RESULT_ROOT / "discovery_components_records.jsonl"
DISCOVERY_EXECUTION = RESULT_ROOT / "discovery_components_execution_manifest.json"
DISCOVERY_RAW_ROOT = RESULT_ROOT / "raw_logits" / "discovery-components"
DISCOVERY_ANALYSIS = RESULT_ROOT / "discovery_components_analysis.json"

REMAINING_ATTEMPT = RESULT_ROOT / "remaining_main_attempt.json"
REMAINING_DISCOVERY_TOPOLOGY_RECORDS = (
    RESULT_ROOT / "remaining_main_discovery_topology_records.jsonl"
)
REMAINING_CONFIRMATION_COMPONENTS_RECORDS = (
    RESULT_ROOT / "remaining_main_confirmation_components_records.jsonl"
)
REMAINING_CONFIRMATION_TOPOLOGY_RECORDS = (
    RESULT_ROOT / "remaining_main_confirmation_topology_records.jsonl"
)
REMAINING_EXECUTION = RESULT_ROOT / "remaining_main_execution_manifest.json"
REMAINING_RAW_ROOT = RESULT_ROOT / "raw_logits" / "remaining-main"
FINAL_ANALYSIS = RESULT_ROOT / "topology_analysis.json"

ANALYSIS_SCHEMA = "coherent-readout-v6a-r2-topology-analysis-v1"
DISCOVERY_AUTHORIZATION_SCHEMA = (
    "coherent-readout-v6a-r2-remaining-main-authorization-v1"
)
EXECUTION_REVISION = "coherent-readout-v6a-r2-natural-token-topology-exec-v1"
ENGINEERING_INVALID = "V6A_R2_ENGINEERING_INVALID"
DISCOVERY_COMPONENT_FAIL = "V6A_R2_DISCOVERY_COMPONENT_FAIL"
DISCOVERY_COMPONENT_PASS = "V6A_R2_DISCOVERY_COMPONENT_PASS_REMAINING_AUTHORIZED"
CONFIRMATION_COMPONENT_FAIL = "V6A_R2_CONFIRMATION_COMPONENT_FAIL"
NO_REPLICATED_TOPOLOGY = "V6A_R2_NO_REPLICATED_TOPOLOGY_EFFECT"
ORDER_WITHOUT_REVERSAL = "V6A_R2_ORDER_EFFECT_WITHOUT_SIGN_REVERSAL"
REVERSAL_SUPPORTED = "V6A_R2_SCAFFOLD_SENSITIVE_ORDER_REVERSAL_SUPPORTED"

DISCOVERY_STAGE = "discovery-components"
REMAINING_STAGE = "remaining-main"
DISCOVERY_CALLS = 640
REMAINING_CALLS = 1664
TOTAL_CALLS = 2304
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260806
RAW_SHARD_ROWS = 64

RECORD_IDENTITY_FIELDS = (
    "block_call_index",
    "cell_id",
    "execution_block",
    "execution_stage",
    "global_call_index",
    "prompt_id",
    "stage_call_index",
)
CELL_SHARED_IDENTITY_FIELDS = (
    "block_call_index",
    "cell_id",
    "execution_block",
    "execution_stage",
    "global_call_index",
    "stage_call_index",
)
FILE_BINDING_FIELDS = frozenset({"path", "file_sha256", "size_bytes"})
SOURCE_LOCK_FIELDS = frozenset(
    {
        "plan_manifest_file_sha256",
        "design_file_sha256",
        "dependency_lock_file_sha256",
        "tokenization_receipt_file_sha256",
        "fixture_file_sha256",
        "fixture_manifest_file_sha256",
        "runner_file_sha256",
        "analyzer_file_sha256",
        "design_document_file_sha256",
        "sealed_v2_runner_file_sha256",
    }
)
PREFLIGHT_FIELDS = frozenset(
    {
        "execution_stage",
        "planned_calls",
        "global_start_row",
        "global_stop_row",
        "stage_raw_logits_expected_bytes",
        "total_raw_logits_expected_bytes",
        "model_safetensor_bytes",
        "mps_recommended_max_memory_bytes",
        "mps_allocated_before_bytes",
        "required_mps_headroom_bytes",
        "mps_bfloat16_kernel_pass",
        "disk_free_bytes",
        "required_disk_free_bytes",
        "stored_logits_dtype",
        "no_model_loaded",
        "no_model_forward_performed",
        "model_calls",
        "generation_calls",
        "pass",
    }
)
FORWARD_CONTRACT_FIELDS = frozenset(
    {
        "torch_inference_mode",
        "use_cache",
        "logits_to_keep",
        "return_dict",
        "teacher_forced_prompt_forward",
        "retained_logits_expression",
        "retained_logits_dtype",
        "retained_logits_shape",
    }
)
ATTEMPT_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "registration_state",
        "execution_revision",
        "execution_stage",
        "call_plan_sha256",
        "stage_plan_sha256",
        "scientific_registry_sha256",
        "expected_calls",
        "global_start_row",
        "global_stop_row",
        "stage_start_row",
        "stage_stop_row",
        "block_order",
        "block_counts",
        "partial_resume_allowed",
        "generation_used",
        "model",
        "forward_contract",
        "preflight",
        "source_locks",
    }
)
PRIOR_DISCOVERY_AUTHORIZATION_FIELDS = frozenset(
    {
        "execution_manifest_file_sha256",
        "execution_manifest_canonical_sha256",
        "authorization_file_sha256",
        "authorization_canonical_sha256",
        "authorization_status",
    }
)
EXECUTION_MANIFEST_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "registration_state",
        "execution_revision",
        "execution_stage",
        "call_plan_sha256",
        "stage_plan_sha256",
        "scientific_registry_sha256",
        "phase_model_calls",
        "cumulative_model_calls",
        "global_start_row",
        "global_stop_row",
        "block_order",
        "block_counts",
        "generation_used",
        "partial_resume_allowed",
        "raw_logits_shards",
        "attempt",
        "source_locks",
    }
)
DIAGNOSTIC_FIELDS = frozenset(
    {
        "expected_logit",
        "distractor_logit",
        "expected_minus_distractor_margin",
        "pairwise_correct",
        "pairwise_tie",
        "strict_unique_global_correct",
        "maximum_token_ids",
        "maximum_tie_count",
        "full_vocab_logits_sha256",
    }
)
RECORD_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "registration_state",
        "execution_revision",
        "record_identity_id",
        "record_id",
        "execution_stage",
        "execution_block",
        "global_call_index",
        "block_call_index",
        "stage_call_index",
        "call_plan_sha256",
        "stage_plan_sha256",
        "scientific_registry_sha256",
        "cell_id",
        "prompt_id",
        "world_id",
        "role",
        "family",
        "factors",
        "factor_levels",
        "expected_token_id",
        "distractor_token_id",
        "raw_logits_shard_index",
        "raw_logits_row_in_shard",
        "raw_logits_stage_row",
        "raw_logits_global_row",
        "raw_logits_row_sha256",
        "teacher_forced_prompt_forward",
        "generation_used",
        "model_calls",
    }
)

REMAINING_BLOCK_REGISTRY = {
    "discovery-topology": {
        "path_key": "remaining_discovery_topology_records",
        "stage_start_row": 0,
        "stage_stop_row": 512,
        "global_start_row": 640,
        "global_stop_row": 1152,
        "shard_start": 0,
        "shard_stop": 8,
    },
    "confirmation-components": {
        "path_key": "remaining_confirmation_components_records",
        "stage_start_row": 512,
        "stage_stop_row": 1152,
        "global_start_row": 1152,
        "global_stop_row": 1792,
        "shard_start": 8,
        "shard_stop": 18,
    },
    "confirmation-topology": {
        "path_key": "remaining_confirmation_topology_records",
        "stage_start_row": 1152,
        "stage_stop_row": 1664,
        "global_start_row": 1792,
        "global_stop_row": 2304,
        "shard_start": 18,
        "shard_stop": 26,
    },
}

PROPERTY_RETRIEVAL = "property_retrieval"
CODEBOOK_LOOKUP = "codebook_lookup"
SINGLE_TARGET = "single_target_composition"
TWO_FACT = "two_fact_composition"
COMPONENT_FAMILIES = (PROPERTY_RETRIEVAL, CODEBOOK_LOOKUP, SINGLE_TARGET)
ALL_FAMILIES = (*COMPONENT_FAMILIES, TWO_FACT)
COMPONENT_FAMILY_COUNTS = {
    PROPERTY_RETRIEVAL: 256,
    CODEBOOK_LOOKUP: 128,
    SINGLE_TARGET: 256,
}
PER_WORLD_COUNTS = {
    PROPERTY_RETRIEVAL: 32,
    CODEBOOK_LOOKUP: 16,
    SINGLE_TARGET: 32,
}
OVERALL_MINIMUMS = {
    PROPERTY_RETRIEVAL: 251,
    CODEBOOK_LOOKUP: 126,
    SINGLE_TARGET: 251,
}
PER_WORLD_MINIMUMS = {
    PROPERTY_RETRIEVAL: 29,
    CODEBOOK_LOOKUP: 15,
    SINGLE_TARGET: 29,
}
PAIRWISE_LABEL_MINIMUMS = {
    PROPERTY_RETRIEVAL: (15, 16),
    CODEBOOK_LOOKUP: (7, 8),
    SINGLE_TARGET: (15, 16),
}
FACTOR_MINIMUMS = {
    PROPERTY_RETRIEVAL: {
        "p": (116, 128),
        "v": (116, 128),
        "o": (116, 128),
        "q": (116, 128),
        "a": (116, 128),
    },
    CODEBOOK_LOOKUP: {
        "p": (58, 64),
        "m": (58, 64),
        "r": (58, 64),
        "v": (58, 64),
    },
    SINGLE_TARGET: {
        "p": (116, 128),
        "m": (116, 128),
        "r": (116, 128),
        "v": (116, 128),
        "u": (116, 128),
        "w": (116, 128),
        "q": (116, 128),
        "a": (116, 128),
    },
}


class R2TopologyAnalysisError(ValueError):
    """Raised when a frozen R2 engineering or analysis contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise R2TopologyAnalysisError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def f32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unknown = sorted(observed - set(expected))
        raise R2TopologyAnalysisError(
            f"{label} schema changed (missing={missing}, unknown={unknown})"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R2TopologyAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise R2TopologyAnalysisError(f"cannot bind regular artifact: {path}")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _stage_contract(stage: str) -> dict[str, Any]:
    if stage == DISCOVERY_STAGE:
        return {
            "attempt_status": (
                "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_ATTEMPT_STARTED_IMMUTABLE"
            ),
            "complete_status": (
                "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_COMPLETE_NOT_ANALYZED"
            ),
            "expected_calls": DISCOVERY_CALLS,
            "global_start_row": 0,
            "global_stop_row": DISCOVERY_CALLS,
            "stage_start_row": 0,
            "stage_stop_row": DISCOVERY_CALLS,
            "block_order": ["discovery-components"],
            "block_counts": {"discovery-components": DISCOVERY_CALLS},
            "cumulative_model_calls": DISCOVERY_CALLS,
        }
    if stage == REMAINING_STAGE:
        return {
            "attempt_status": "V6A_R2_REMAINING_MAIN_EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
            "complete_status": "V6A_R2_REMAINING_MAIN_EXECUTION_COMPLETE_NOT_ANALYZED",
            "expected_calls": REMAINING_CALLS,
            "global_start_row": DISCOVERY_CALLS,
            "global_stop_row": TOTAL_CALLS,
            "stage_start_row": 0,
            "stage_stop_row": REMAINING_CALLS,
            "block_order": list(REMAINING_BLOCK_REGISTRY),
            "block_counts": {
                block: registry["stage_stop_row"] - registry["stage_start_row"]
                for block, registry in REMAINING_BLOCK_REGISTRY.items()
            },
            "cumulative_model_calls": TOTAL_CALLS,
        }
    raise R2TopologyAnalysisError("unknown R2 execution stage")


def _expected_forward_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise R2TopologyAnalysisError("R2 plan model registry is missing")
    vocab_size = model.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise R2TopologyAnalysisError("R2 plan vocabulary size is invalid")
    return {
        "torch_inference_mode": True,
        "use_cache": False,
        "logits_to_keep": 1,
        "return_dict": True,
        "teacher_forced_prompt_forward": True,
        "retained_logits_expression": "output.logits[0,-1,:]",
        "retained_logits_dtype": "float32_little_endian",
        "retained_logits_shape": [vocab_size],
    }


def _expected_source_locks(bundle: Mapping[str, Any]) -> dict[str, str]:
    runner = bundle["runner"]
    paths = bundle["paths"]
    source_paths = {
        "plan_manifest_file_sha256": paths["plan_manifest"],
        "design_file_sha256": paths["design"],
        "dependency_lock_file_sha256": paths["dependency_lock"],
        "tokenization_receipt_file_sha256": paths["tokenization_receipt"],
        "fixture_file_sha256": Path(runner.FIXTURE),
        "fixture_manifest_file_sha256": Path(runner.FIXTURE_MANIFEST),
        "runner_file_sha256": Path(runner.__file__).resolve(),
        "analyzer_file_sha256": Path(__file__).resolve(),
        "design_document_file_sha256": Path(runner.DESIGN_DOCUMENT),
        "sealed_v2_runner_file_sha256": Path(runner.SEALED_V2_RUNNER),
    }
    if set(source_paths) != SOURCE_LOCK_FIELDS:
        raise R2TopologyAnalysisError("internal source-lock registry changed")
    return {key: file_sha256(path) for key, path in source_paths.items()}


def _validate_source_locks(
    observed: Any,
    bundle: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    if not isinstance(observed, Mapping):
        raise R2TopologyAnalysisError(f"{label} source locks are missing")
    _require_exact_keys(observed, SOURCE_LOCK_FIELDS, f"{label} source locks")
    expected = _expected_source_locks(bundle)
    if dict(observed) != expected:
        raise R2TopologyAnalysisError(f"{label} source locks changed")
    return expected


def _validate_preflight(
    observed: Any,
    plan: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise R2TopologyAnalysisError("stage attempt preflight is missing")
    _require_exact_keys(observed, PREFLIGHT_FIELDS, "stage attempt preflight")
    contract = _stage_contract(stage)
    specs = plan.get("raw_logits_shards", {}).get(stage)
    if not isinstance(specs, list):
        raise R2TopologyAnalysisError("planned raw-logit shards are missing")
    expected_stage_bytes = 0
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise R2TopologyAnalysisError("planned raw-logit shard is malformed")
        shape = spec.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
            or spec.get("dtype") != "<f4"
        ):
            raise R2TopologyAnalysisError("planned raw-logit shape is malformed")
        expected_stage_bytes += math.prod(shape) * 4
    disk = plan.get("disk_requirements")
    if not isinstance(disk, Mapping):
        raise R2TopologyAnalysisError("planned disk requirements are missing")
    fixed = {
        "execution_stage": stage,
        "planned_calls": contract["expected_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "stage_raw_logits_expected_bytes": expected_stage_bytes,
        "total_raw_logits_expected_bytes": disk.get("raw_logits_expected_bytes"),
        "stored_logits_dtype": "float32_little_endian",
        "no_model_loaded": True,
        "no_model_forward_performed": True,
        "model_calls": 0,
        "generation_calls": 0,
        "mps_bfloat16_kernel_pass": True,
        "pass": True,
    }
    for key, expected in fixed.items():
        if observed.get(key) != expected:
            raise R2TopologyAnalysisError(f"stage attempt preflight changed: {key}")
    integer_fields = (
        "stage_raw_logits_expected_bytes",
        "total_raw_logits_expected_bytes",
        "model_safetensor_bytes",
        "mps_recommended_max_memory_bytes",
        "mps_allocated_before_bytes",
        "required_mps_headroom_bytes",
        "disk_free_bytes",
        "required_disk_free_bytes",
        "model_calls",
        "generation_calls",
    )
    if any(
        isinstance(observed[key], bool)
        or not isinstance(observed[key], int)
        or observed[key] < 0
        for key in integer_fields
    ):
        raise R2TopologyAnalysisError("stage attempt preflight numeric field is invalid")
    if (
        observed["model_safetensor_bytes"] <= 0
        or observed["disk_free_bytes"] < observed["required_disk_free_bytes"]
        or observed["mps_recommended_max_memory_bytes"]
        - observed["mps_allocated_before_bytes"]
        < observed["required_mps_headroom_bytes"]
    ):
        raise R2TopologyAnalysisError("stage attempt preflight headroom check changed")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2TopologyAnalysisError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R2TopologyAnalysisError(f"{label} is not finite")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R2TopologyAnalysisError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise R2TopologyAnalysisError(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise R2TopologyAnalysisError(f"cannot read JSONL artifact: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise R2TopologyAnalysisError(f"record artifact is empty or malformed: {path}")
    return rows


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_frozen_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    result_root: Path = RESULT_ROOT,
) -> None:
    if not _is_within(path, result_root):
        raise R2TopologyAnalysisError("R2 analysis output is outside the R2 result root")
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise R2TopologyAnalysisError(f"refusing to overwrite frozen R2 artifact: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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


def _require_registered_analysis_write(
    result_root: Path,
    *,
    stage: str,
) -> None:
    """Reject premature or off-registry production analysis before any write."""

    runner = _runner_module()
    if (
        runner.REGISTRATION_STATE
        != runner.EXECUTION_ENABLED_REGISTRATION_STATE
    ):
        raise R2TopologyAnalysisError(
            "R2 analysis writes require exact frozen registration state"
        )
    if (
        result_root.resolve() != RESULT_ROOT.resolve()
        or result_root.resolve() != runner.RESULT_ROOT.resolve()
    ):
        raise R2TopologyAnalysisError(
            "R2 analysis writes require the exact registered result root"
        )
    paths = runner.artifact_paths(result_root)
    plan_manifest_path = paths["plan_manifest"]
    if not plan_manifest_path.is_file() or plan_manifest_path.is_symlink():
        raise R2TopologyAnalysisError(
            "R2 analysis write lacks a registered-main plan manifest"
        )
    plan_manifest = _load_json(plan_manifest_path)
    if (
        plan_manifest.get("artifact_scope") != "registered_main"
        or plan_manifest.get("status")
        != "R2_PLAN_FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
        or plan_manifest.get("registration_state")
        != runner.EXECUTION_ENABLED_REGISTRATION_STATE
    ):
        raise R2TopologyAnalysisError(
            "R2 analysis write lacks an exact registered-main plan"
        )
    execution_key = {
        "discovery-gate": "discovery_execution_manifest",
        "final": "remaining_execution_manifest",
    }.get(stage)
    if execution_key is None:
        raise R2TopologyAnalysisError("R2 analysis write stage is not registered")
    execution_path = paths[execution_key]
    if not execution_path.is_file() or execution_path.is_symlink():
        raise R2TopologyAnalysisError(
            "R2 analysis write requires its completed execution manifest"
        )


def diagnostics_from_full_vocab(
    row: np.ndarray,
    expected_token_id: int,
    distractor_token_id: int,
) -> dict[str, Any]:
    """Reconstruct natural-token diagnostics without generation or a model call."""

    values = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise R2TopologyAnalysisError("full-vocabulary logit row is invalid")
    if (
        isinstance(expected_token_id, bool)
        or not isinstance(expected_token_id, int)
        or isinstance(distractor_token_id, bool)
        or not isinstance(distractor_token_id, int)
        or expected_token_id == distractor_token_id
        or not 0 <= expected_token_id < len(values)
        or not 0 <= distractor_token_id < len(values)
    ):
        raise R2TopologyAnalysisError("natural answer-token IDs are invalid")
    expected = float(values[expected_token_id])
    distractor = float(values[distractor_token_id])
    maximum = float(values.max())
    maximum_ids = [int(index) for index in np.flatnonzero(values == maximum)]
    return {
        "expected_logit": expected,
        "distractor_logit": distractor,
        "expected_minus_distractor_margin": expected - distractor,
        "pairwise_correct": expected > distractor,
        "pairwise_tie": expected == distractor,
        "strict_unique_global_correct": maximum_ids == [expected_token_id],
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": f32_sha256(values),
    }


def _diagnostics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise R2TopologyAnalysisError("record diagnostics are missing")
    _require_exact_keys(diagnostics, DIAGNOSTIC_FIELDS, "record diagnostics")
    expected = _finite(diagnostics["expected_logit"], "expected_logit")
    distractor = _finite(diagnostics["distractor_logit"], "distractor_logit")
    margin = _finite(
        diagnostics["expected_minus_distractor_margin"],
        "expected_minus_distractor_margin",
    )
    if not math.isclose(margin, expected - distractor, rel_tol=0.0, abs_tol=0.0):
        raise R2TopologyAnalysisError("record margin does not reconstruct")
    if diagnostics["pairwise_correct"] is not (expected > distractor):
        raise R2TopologyAnalysisError("record pairwise correctness changed")
    if diagnostics["pairwise_tie"] is not (expected == distractor):
        raise R2TopologyAnalysisError("record pairwise tie changed")
    maximum_ids = diagnostics["maximum_token_ids"]
    expected_id = record.get("expected_token_id")
    if (
        not isinstance(maximum_ids, list)
        or not maximum_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in maximum_ids)
        or isinstance(expected_id, bool)
        or not isinstance(expected_id, int)
    ):
        raise R2TopologyAnalysisError("record maximum-token registry is invalid")
    strict = maximum_ids == [expected_id]
    if diagnostics["strict_unique_global_correct"] is not strict:
        raise R2TopologyAnalysisError("record strict-global correctness changed")
    if diagnostics["maximum_tie_count"] != len(maximum_ids):
        raise R2TopologyAnalysisError("record maximum-tie count changed")
    _require_sha256(
        diagnostics["full_vocab_logits_sha256"],
        "record full-vocabulary logit hash",
    )
    return diagnostics


def _split(record: Mapping[str, Any]) -> str:
    value = record.get("split", record.get("role"))
    if value not in {"discovery", "confirmation"}:
        raise R2TopologyAnalysisError("record split is invalid")
    return str(value)


def _factor(record: Mapping[str, Any], name: str) -> int:
    factors = record.get("factors")
    if not isinstance(factors, Mapping) or factors.get(name) not in {-1, 1}:
        raise R2TopologyAnalysisError(f"record factor {name} is invalid")
    return int(factors[name])


def _rate(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    if not records:
        raise R2TopologyAnalysisError("cannot summarize an empty record group")
    correct = sum(bool(_diagnostics(record)[metric]) for record in records)
    return {"correct": correct, "n": len(records), "accuracy": correct / len(records)}


def _group_gate(
    records: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    metric: str,
    minimum_correct: int,
    expected_n: int,
) -> tuple[dict[str, Any], bool]:
    summary: dict[str, Any] = {}
    passed = True
    for key in sorted(groups):
        members = list(groups[key])
        if len(members) != expected_n:
            raise R2TopologyAnalysisError(f"component group {key} has changed size")
        rate = _rate(members, metric)
        rate["minimum_correct"] = minimum_correct
        rate["pass"] = rate["correct"] >= minimum_correct
        summary[key] = rate
        passed = passed and rate["pass"]
    if sum(len(value) for value in groups.values()) < len(records):
        raise R2TopologyAnalysisError("component group registry is incomplete")
    return summary, passed


def component_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Apply every fixed pairwise and strict-global component gate to one split."""

    if split not in {"discovery", "confirmation"}:
        raise R2TopologyAnalysisError("component split is invalid")
    rows = list(records)
    if len(rows) != DISCOVERY_CALLS or any(_split(row) != split for row in rows):
        raise R2TopologyAnalysisError("component panel split or row count changed")
    if any(row.get("family") not in COMPONENT_FAMILIES for row in rows):
        raise R2TopologyAnalysisError("component panel contains a topology row")
    if len({row.get("record_id") for row in rows}) != len(rows):
        raise R2TopologyAnalysisError("component record IDs are duplicated")
    family_counts = Counter(str(row.get("family")) for row in rows)
    if family_counts != Counter(COMPONENT_FAMILY_COUNTS):
        raise R2TopologyAnalysisError("component family counts changed")
    worlds = sorted({str(row.get("world_id")) for row in rows})
    if len(worlds) != 8 or any(not world for world in worlds):
        raise R2TopologyAnalysisError("component world registry changed")

    gates: dict[str, bool] = {}
    families: dict[str, Any] = {}
    for family in COMPONENT_FAMILIES:
        members = [row for row in rows if row["family"] == family]
        pairwise = _rate(members, "pairwise_correct")
        strict_global = _rate(members, "strict_unique_global_correct")
        minimum = OVERALL_MINIMUMS[family]
        pairwise["minimum_correct"] = minimum
        strict_global["minimum_correct"] = minimum
        pairwise["pass"] = pairwise["correct"] >= minimum
        strict_global["pass"] = strict_global["correct"] >= minimum

        world_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in members:
            world_groups[str(row["world_id"])].append(row)
        expected_world_n = PER_WORLD_COUNTS[family]
        world_pairwise, world_pairwise_pass = _group_gate(
            members,
            world_groups,
            metric="pairwise_correct",
            minimum_correct=PER_WORLD_MINIMUMS[family],
            expected_n=expected_world_n,
        )
        world_global, world_global_pass = _group_gate(
            members,
            world_groups,
            metric="strict_unique_global_correct",
            minimum_correct=PER_WORLD_MINIMUMS[family],
            expected_n=expected_world_n,
        )

        label_minimum, label_n = PAIRWISE_LABEL_MINIMUMS[family]
        label_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in members:
            label = row.get("expected_answer")
            if not isinstance(label, str) or len(label) != 1:
                raise R2TopologyAnalysisError("component answer label is invalid")
            label_groups[label].append(row)
        labels, labels_pass = _group_gate(
            members,
            label_groups,
            metric="pairwise_correct",
            minimum_correct=label_minimum,
            expected_n=label_n,
        )

        factor_summaries: dict[str, Any] = {}
        factors_pass = True
        for factor, (factor_minimum, factor_n) in FACTOR_MINIMUMS[family].items():
            factor_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in members:
                level = _factor(row, factor)
                factor_groups[f"{factor}={level:+d}"].append(row)
            factor_summary, factor_pass = _group_gate(
                members,
                factor_groups,
                metric="pairwise_correct",
                minimum_correct=factor_minimum,
                expected_n=factor_n,
            )
            factor_summaries[factor] = factor_summary
            factors_pass = factors_pass and factor_pass

        joint_summaries: dict[str, Any] = {}
        joints_pass = True
        if family == PROPERTY_RETRIEVAL:
            for names, joint_minimum, joint_n in (
                (("q", "a"), 58, 64),
                (("o", "q", "a"), 29, 32),
            ):
                grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in members:
                    levels = tuple(_factor(row, name) for name in names)
                    key = "__".join(
                        f"{name}={level:+d}" for name, level in zip(names, levels, strict=True)
                    )
                    grouped[key].append(row)
                item, item_pass = _group_gate(
                    members,
                    grouped,
                    metric="pairwise_correct",
                    minimum_correct=joint_minimum,
                    expected_n=joint_n,
                )
                joint_summaries["_by_".join(names)] = item
                joints_pass = joints_pass and item_pass
        elif family == SINGLE_TARGET:
            grouped = defaultdict(list)
            for row in members:
                q, a = _factor(row, "q"), _factor(row, "a")
                grouped[f"q={q:+d}__a={a:+d}"].append(row)
            item, item_pass = _group_gate(
                members,
                grouped,
                metric="pairwise_correct",
                minimum_correct=58,
                expected_n=64,
            )
            joint_summaries["q_by_a"] = item
            joints_pass = item_pass

        family_pass = all(
            (
                pairwise["pass"],
                strict_global["pass"],
                world_pairwise_pass,
                world_global_pass,
                labels_pass,
                factors_pass,
                joints_pass,
            )
        )
        gates[f"{family}_pass"] = family_pass
        families[family] = {
            "pairwise_overall": pairwise,
            "strict_global_overall": strict_global,
            "pairwise_by_world": world_pairwise,
            "strict_global_by_world": world_global,
            "pairwise_by_answer_label": labels,
            "pairwise_by_factor": factor_summaries,
            "pairwise_joint_scaffolds": joint_summaries,
            "pass": family_pass,
        }
    gates["pass"] = all(gates.values())
    return {
        "split": split,
        "row_count": len(rows),
        "world_ids": worlds,
        "families": families,
        "gates": gates,
        "ties_are_ordinary_incorrect_rows": True,
        "single_row_veto": False,
    }


def _bootstrap_indices(n_worlds: int) -> np.ndarray:
    if n_worlds != 8:
        raise R2TopologyAnalysisError("registered bootstrap requires exactly eight worlds")
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        n_worlds,
        size=(BOOTSTRAP_DRAWS, n_worlds),
    )


def _effect_summary(
    world_values: Mapping[str, float],
    indices: np.ndarray,
) -> dict[str, Any]:
    worlds = sorted(world_values)
    if len(worlds) != 8 or indices.shape != (BOOTSTRAP_DRAWS, 8):
        raise R2TopologyAnalysisError("world effect or bootstrap shape changed")
    values = np.asarray(
        [_finite(world_values[world], f"effect[{world}]") for world in worlds],
        dtype=np.float64,
    )
    samples = values[indices].mean(axis=1)
    return {
        "world_values": dict(zip(worlds, values.tolist(), strict=True)),
        "n_worlds": 8,
        "mean": float(values.mean()),
        "positive_world_count": int(np.sum(values > 0.0)),
        "negative_world_count": int(np.sum(values < 0.0)),
        "zero_world_count": int(np.sum(values == 0.0)),
        "bootstrap_95": {
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "world_id",
            "interpretation": "fixed_panel_stability_not_population_confidence",
        },
    }


def _content_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (_factor(record, "p"), _factor(record, "m"), _factor(record, "u"))


def _margin(record: Mapping[str, Any]) -> float:
    return _finite(
        _diagnostics(record)["expected_minus_distractor_margin"],
        "expected_minus_distractor_margin",
    )


def _accuracy(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    return _rate(records, metric)["accuracy"]


def topology_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute registered matched topology estimands after both component gates."""

    rows = list(records)
    two_fact = [row for row in rows if row.get("family") == TWO_FACT]
    single = [row for row in rows if row.get("family") == SINGLE_TARGET]
    if len(two_fact) != 1024 or len(single) != 512:
        raise R2TopologyAnalysisError("topology or matched single-TARGET row count changed")
    if any(row.get("family") not in ALL_FAMILIES for row in rows):
        raise R2TopologyAnalysisError("unknown family entered topology analysis")

    split_results: dict[str, Any] = {}
    split_maximum_gates: dict[str, bool] = {}
    for split in ("discovery", "confirmation"):
        split_two = [row for row in two_fact if _split(row) == split]
        split_single = [row for row in single if _split(row) == split]
        if len(split_two) != 512 or len(split_single) != 256:
            raise R2TopologyAnalysisError(f"{split} topology split counts changed")
        worlds = sorted({str(row.get("world_id")) for row in split_two})
        if len(worlds) != 8:
            raise R2TopologyAnalysisError(f"{split} topology world registry changed")

        single_map: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in split_single:
            key = (row["world_id"], *_content_key(row), _factor(row, "q"), _factor(row, "a"))
            if key in single_map:
                raise R2TopologyAnalysisError("single-TARGET matched key is duplicated")
            single_map[key] = row
        if len(single_map) != 256:
            raise R2TopologyAnalysisError("single-TARGET matched registry changed")

        d_worlds: dict[tuple[int, int], dict[str, float]] = {
            (q, a): {} for q in (-1, 1) for a in (-1, 1)
        }
        matched_worlds: dict[tuple[int, int, int], dict[str, float]] = {
            (o, q, a): {}
            for o in (-1, 1)
            for q in (-1, 1)
            for a in (-1, 1)
        }
        for world in worlds:
            members = [row for row in split_two if row["world_id"] == world]
            if len(members) != 64:
                raise R2TopologyAnalysisError("two-fact world does not contain 64 rows")
            for q in (-1, 1):
                for a in (-1, 1):
                    differences: list[float] = []
                    for content in {
                        _content_key(row)
                        for row in members
                        if _factor(row, "q") == q and _factor(row, "a") == a
                    }:
                        pair = [
                            row
                            for row in members
                            if _content_key(row) == content
                            and _factor(row, "q") == q
                            and _factor(row, "a") == a
                        ]
                        by_o = {_factor(row, "o"): row for row in pair}
                        if len(pair) != 2 or set(by_o) != {-1, 1}:
                            raise R2TopologyAnalysisError("two-fact matched order pair changed")
                        differences.append(_margin(by_o[1]) - _margin(by_o[-1]))
                    if len(differences) != 8:
                        raise R2TopologyAnalysisError("topology cell lacks eight content profiles")
                    d_worlds[(q, a)][world] = float(np.mean(differences))

            for o in (-1, 1):
                for q in (-1, 1):
                    for a in (-1, 1):
                        contrasts: list[float] = []
                        cell = [
                            row
                            for row in members
                            if _factor(row, "o") == o
                            and _factor(row, "q") == q
                            and _factor(row, "a") == a
                        ]
                        if len(cell) != 8:
                            raise R2TopologyAnalysisError("matched two-fact cell size changed")
                        for row in cell:
                            key = (
                                world,
                                *_content_key(row),
                                q,
                                a,
                            )
                            if key not in single_map:
                                raise R2TopologyAnalysisError(
                                    "two-fact row lacks a matched single-TARGET row"
                                )
                            contrasts.append(_margin(row) - _margin(single_map[key]))
                        matched_worlds[(o, q, a)][world] = float(np.mean(contrasts))

        r_worlds = {
            world: d_worlds[(-1, -1)][world] - d_worlds[(1, 1)][world]
            for world in worlds
        }
        q_worlds = {
            world: 0.5
            * (d_worlds[(-1, -1)][world] + d_worlds[(-1, 1)][world])
            - 0.5 * (d_worlds[(1, -1)][world] + d_worlds[(1, 1)][world])
            for world in worlds
        }
        a_worlds = {
            world: 0.5
            * (d_worlds[(-1, -1)][world] + d_worlds[(1, -1)][world])
            - 0.5 * (d_worlds[(-1, 1)][world] + d_worlds[(1, 1)][world])
            for world in worlds
        }
        interaction_worlds = {
            world: 0.25
            * (
                d_worlds[(-1, -1)][world]
                - d_worlds[(-1, 1)][world]
                - d_worlds[(1, -1)][world]
                + d_worlds[(1, 1)][world]
            )
            for world in worlds
        }
        indices = _bootstrap_indices(8)
        d_summary = {
            f"q={q:+d}__a={a:+d}": _effect_summary(d_worlds[(q, a)], indices)
            for q in (-1, 1)
            for a in (-1, 1)
        }
        r_summary = _effect_summary(r_worlds, indices)
        q_summary = _effect_summary(q_worlds, indices)
        a_summary = _effect_summary(a_worlds, indices)
        interaction_summary = _effect_summary(interaction_worlds, indices)
        matched_summary = {
            f"o={o:+d}__q={q:+d}__a={a:+d}": _effect_summary(
                matched_worlds[(o, q, a)], indices
            )
            for o in (-1, 1)
            for q in (-1, 1)
            for a in (-1, 1)
        }

        v5_first = [
            row
            for row in split_two
            if _factor(row, "q") == -1
            and _factor(row, "a") == -1
            and _factor(row, "o") == -1
        ]
        v5_second = [
            row
            for row in split_two
            if _factor(row, "q") == -1
            and _factor(row, "a") == -1
            and _factor(row, "o") == 1
        ]
        v4_first = [
            row
            for row in split_two
            if _factor(row, "q") == 1
            and _factor(row, "a") == 1
            and _factor(row, "o") == -1
        ]
        v4_second = [
            row
            for row in split_two
            if _factor(row, "q") == 1
            and _factor(row, "a") == 1
            and _factor(row, "o") == 1
        ]
        if any(len(group) != 64 for group in (v5_first, v5_second, v4_first, v4_second)):
            raise R2TopologyAnalysisError("registered topology accuracy cell changed")
        accuracy_endpoints = {
            "metric": "pairwise_natural_token_accuracy",
            "v5_like_second_minus_first": _accuracy(v5_second, "pairwise_correct")
            - _accuracy(v5_first, "pairwise_correct"),
            "v4_like_first_minus_second": _accuracy(v4_first, "pairwise_correct")
            - _accuracy(v4_second, "pairwise_correct"),
        }
        reversal_gates = {
            "D_minus_minus_bootstrap_lower_positive": (
                d_summary["q=-1__a=-1"]["bootstrap_95"]["lower_95"] > 0.0
            ),
            "D_minus_minus_positive_at_least_6_worlds": (
                d_summary["q=-1__a=-1"]["positive_world_count"] >= 6
            ),
            "D_plus_plus_bootstrap_upper_negative": (
                d_summary["q=+1__a=+1"]["bootstrap_95"]["upper_95"] < 0.0
            ),
            "D_plus_plus_negative_at_least_6_worlds": (
                d_summary["q=+1__a=+1"]["negative_world_count"] >= 6
            ),
            "R_bootstrap_lower_positive": r_summary["bootstrap_95"]["lower_95"] > 0.0,
            "R_positive_at_least_6_worlds": r_summary["positive_world_count"] >= 6,
            "v5_like_accuracy_difference_at_least_0_10": (
                accuracy_endpoints["v5_like_second_minus_first"] >= 0.10
            ),
            "v4_like_accuracy_difference_at_least_0_10": (
                accuracy_endpoints["v4_like_first_minus_second"] >= 0.10
            ),
        }
        reversal_gates["pass"] = all(reversal_gates.values())
        split_maximum_gates[split] = reversal_gates["pass"]
        split_results[split] = {
            "D": d_summary,
            "R": r_summary,
            "Q": q_summary,
            "A": a_summary,
            "residual_o_by_q_by_a_interaction": interaction_summary,
            "residual_interaction_formula": (
                "0.25*(D(-1,-1)-D(-1,+1)-D(+1,-1)+D(+1,+1))"
            ),
            "matched_two_fact_minus_single_target": matched_summary,
            "accuracy_endpoints": accuracy_endpoints,
            "maximum_reversal_gates": reversal_gates,
        }

    replicated_r = all(
        split_results[split]["R"]["bootstrap_95"]["lower_95"] > 0.0
        and split_results[split]["R"]["positive_world_count"] >= 6
        for split in ("discovery", "confirmation")
    )
    if all(split_maximum_gates.values()):
        status = REVERSAL_SUPPORTED
    elif replicated_r:
        status = ORDER_WITHOUT_REVERSAL
    else:
        status = NO_REPLICATED_TOPOLOGY

    def replicated_nonzero(metric: str) -> dict[str, Any]:
        intervals = [
            split_results[split][metric]["bootstrap_95"]
            for split in ("discovery", "confirmation")
        ]
        positive = all(interval["lower_95"] > 0.0 for interval in intervals)
        negative = all(interval["upper_95"] < 0.0 for interval in intervals)
        return {
            "replicated_nonzero": positive or negative,
            "direction": "positive" if positive else "negative" if negative else None,
        }

    return {
        "status": status,
        "splits": split_results,
        "replicated_R_order_effect": replicated_r,
        "placement_descriptors": {
            "Q_task_placement": replicated_nonzero("Q"),
            "A_answer_set_placement": replicated_nonzero("A"),
            "residual_joint_interaction": replicated_nonzero(
                "residual_o_by_q_by_a_interaction"
            ),
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "world_id",
            "interpretation": "fixed_panel_stability_not_population_confidence",
        },
    }


def discovery_gate_from_validated_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate only the 640 discovery components; topology rows are forbidden."""

    rows = list(records)
    if any(row.get("execution_block") != "discovery-components" for row in rows):
        raise R2TopologyAnalysisError(
            "discovery gate attempted to access a non-component or topology row"
        )
    components = component_summary(rows, split="discovery")
    passed = components["gates"]["pass"]
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": "discovery-gate",
        "status": DISCOVERY_COMPONENT_PASS if passed else DISCOVERY_COMPONENT_FAIL,
        "engineering_valid": True,
        "components": components,
        "remaining_main_authorized": passed,
        "topology_rows_read": 0,
        "topology_inference_performed": False,
        "model_calls_issued_by_analyzer": 0,
        "generation_calls_issued_by_analyzer": 0,
    }


def final_from_validated_records(
    discovery_records: Sequence[Mapping[str, Any]],
    remaining_records: Sequence[Mapping[str, Any]],
    *,
    topology_function: Any = topology_summary,
) -> dict[str, Any]:
    """Gate confirmation components before invoking the topology function."""

    discovery = list(discovery_records)
    remaining = list(remaining_records)
    if len(discovery) != DISCOVERY_CALLS or len(remaining) != REMAINING_CALLS:
        raise R2TopologyAnalysisError("final staged record counts changed")
    all_rows = [*discovery, *remaining]
    call_indices = [row.get("call_index") for row in all_rows]
    if call_indices != list(range(TOTAL_CALLS)):
        raise R2TopologyAnalysisError("final global call order changed")
    if len({row.get("record_id") for row in all_rows}) != TOTAL_CALLS:
        raise R2TopologyAnalysisError("final record IDs are not globally unique")
    if any(row.get("execution_block") != "discovery-components" for row in discovery):
        raise R2TopologyAnalysisError("discovery execution contains topology access")
    expected_remaining_blocks = [
        *("discovery-topology" for _ in range(512)),
        *("confirmation-components" for _ in range(640)),
        *("confirmation-topology" for _ in range(512)),
    ]
    if [row.get("execution_block") for row in remaining] != expected_remaining_blocks:
        raise R2TopologyAnalysisError("remaining-main block order changed")

    discovery_components = component_summary(discovery, split="discovery")
    if discovery_components["gates"]["pass"] is not True:
        raise R2TopologyAnalysisError(
            "completed remaining-main stage lacks a passing discovery authorization"
        )
    confirmation_rows = remaining[512:1152]
    confirmation_components = component_summary(
        confirmation_rows,
        split="confirmation",
    )
    if confirmation_components["gates"]["pass"] is not True:
        return {
            "schema_version": ANALYSIS_SCHEMA,
            "stage": "final",
            "status": CONFIRMATION_COMPONENT_FAIL,
            "engineering_valid": True,
            "discovery_components": discovery_components,
            "confirmation_components": confirmation_components,
            "topology_suppressed": True,
            "topology_rows_validated_but_not_interpreted": 1024,
            "topology_inference_performed": False,
            "activation_design_authorized": False,
            "model_calls_issued_by_analyzer": 0,
            "generation_calls_issued_by_analyzer": 0,
        }

    topology = topology_function(all_rows)
    if not isinstance(topology, Mapping) or topology.get("status") not in {
        NO_REPLICATED_TOPOLOGY,
        ORDER_WITHOUT_REVERSAL,
        REVERSAL_SUPPORTED,
    }:
        raise R2TopologyAnalysisError("topology function returned an invalid status")
    status = str(topology["status"])
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": "final",
        "status": status,
        "engineering_valid": True,
        "discovery_components": discovery_components,
        "confirmation_components": confirmation_components,
        "topology_suppressed": False,
        "topology_inference_performed": True,
        "topology": dict(topology),
        "activation_design_authorized": status == REVERSAL_SUPPORTED,
        "activation_execution_authorized": False,
        "model_calls_issued_by_analyzer": 0,
        "generation_calls_issued_by_analyzer": 0,
    }


def _runner_module() -> Any:
    try:
        return importlib.import_module(RUNNER_MODULE)
    except (ImportError, AttributeError) as error:
        raise R2TopologyAnalysisError("R2 runner module is unavailable") from error


def _validate_planned_record_identities(
    prompts: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> None:
    """Validate seven-field prompt identities and six shared cell fields."""

    if not prompts or len(prompts) != len(cells):
        raise R2TopologyAnalysisError("R2 planned record identity rows changed")
    runner = _runner_module()
    for prompt, cell in zip(prompts, cells, strict=True):
        try:
            expected_identity = runner.record_identity_id(prompt)
        except Exception as error:
            raise R2TopologyAnalysisError(
                "R2 planned record identity cannot be reconstructed"
            ) from error
        if (
            prompt.get("record_identity_id") != expected_identity
            or cell.get("record_identity_id") != expected_identity
            or any(
                prompt.get(key) != cell.get(key)
                for key in CELL_SHARED_IDENTITY_FIELDS
            )
        ):
            raise R2TopologyAnalysisError("R2 planned record identity changed")
    if len({prompt["record_identity_id"] for prompt in prompts}) != len(prompts):
        raise R2TopologyAnalysisError("R2 planned record identities are duplicated")


def replay_frozen_plan(
    result_root: Path = RESULT_ROOT,
    *,
    dependency_replay_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently rebuild and compare the complete zero-forward plan."""

    runner = _runner_module()
    try:
        if dependency_replay_override is None:
            plan, design, dependency, receipt = runner.validate_frozen_plan(result_root)
            rebuilt_dependency = runner.dependency_lock(Path(__file__).resolve())
        else:
            plan, design, dependency, receipt = runner.validate_frozen_plan(
                result_root,
                dependency_replay_override=dependency_replay_override,
            )
            rebuilt_dependency = dict(dependency_replay_override)
        fixture, fixture_manifest = runner.load_and_rebuild_fixture()
        tokenizer = runner.load_tokenizer_from_sealed_v2(dependency)
        rebuilt_plan, rebuilt_receipt = runner.build_plan(
            tokenizer,
            fixture,
            dependency,
            result_root=result_root,
        )
    except Exception as error:
        raise R2TopologyAnalysisError("R2 frozen plan replay failed") from error
    if rebuilt_dependency != dependency:
        raise R2TopologyAnalysisError("R2 dependency registry does not independently rebuild")
    if rebuilt_plan != plan or rebuilt_receipt != receipt:
        raise R2TopologyAnalysisError("R2 plan does not independently rebuild exactly")
    if getattr(runner, "EXECUTION_REVISION", None) != EXECUTION_REVISION:
        raise R2TopologyAnalysisError("R2 runner execution revision changed")
    paths = runner.artifact_paths(result_root)
    manifest = _load_json(paths["plan_manifest"])
    if manifest.get("plan") != plan:
        raise R2TopologyAnalysisError("R2 plan manifest changed after runner validation")
    if (
        plan.get("expected_calls") != TOTAL_CALLS
        or plan.get("execution_revision") != EXECUTION_REVISION
        or plan.get("execution_stage_counts")
        != {DISCOVERY_STAGE: DISCOVERY_CALLS, REMAINING_STAGE: REMAINING_CALLS}
        or plan.get("block_counts")
        != {
            "discovery-components": 640,
            "discovery-topology": 512,
            "confirmation-components": 640,
            "confirmation-topology": 512,
        }
        or len(plan.get("prompts", [])) != TOTAL_CALLS
        or len(plan.get("cells", [])) != TOTAL_CALLS
    ):
        raise R2TopologyAnalysisError("R2 rebuilt plan scope changed")
    _validate_planned_record_identities(plan["prompts"], plan["cells"])
    return {
        "runner": runner,
        "paths": paths,
        "plan": plan,
        "design": design,
        "dependency": dependency,
        "receipt": receipt,
        "fixture": fixture,
        "fixture_manifest": fixture_manifest,
        "plan_manifest": manifest,
    }


def _logical_array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype != np.dtype("<f4") or value.ndim != 2:
        raise R2TopologyAnalysisError("raw-logit shard is not a float32 matrix")
    digest = hashlib.sha256()
    for start in range(0, len(value), 8):
        chunk = np.ascontiguousarray(value[start : start + 8], dtype="<f4")
        if not np.isfinite(chunk).all():
            raise R2TopologyAnalysisError("raw-logit shard contains nonfinite values")
        digest.update(memoryview(chunk.view(np.uint8)))
    return digest.hexdigest()


def _binding_path(
    binding: Mapping[str, Any],
    expected: Path,
    label: str,
    *,
    fields: frozenset[str] | set[str] | None = None,
) -> Path:
    _require_exact_keys(
        binding,
        FILE_BINDING_FIELDS if fields is None else fields,
        f"{label} binding",
    )
    path = Path(str(binding.get("path", "")))
    if path.resolve() != expected.resolve() or not path.is_file() or path.is_symlink():
        raise R2TopologyAnalysisError(f"{label} path changed")
    expected_size = binding.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size != path.stat().st_size
        or _require_sha256(binding.get("file_sha256"), f"{label} file hash")
        != file_sha256(path)
    ):
        raise R2TopologyAnalysisError(f"{label} file binding changed")
    return path


def _validate_attempt(
    attempt: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    stage: str,
    prior_discovery_authorization: Mapping[str, Any] | None = None,
) -> None:
    runner = bundle["runner"]
    plan = bundle["plan"]
    contract = _stage_contract(stage)
    expected_fields = set(ATTEMPT_COMMON_FIELDS)
    if stage == REMAINING_STAGE:
        expected_fields.add("prior_discovery_authorization")
    _require_exact_keys(attempt, expected_fields, "stage attempt")
    required = {
        "schema_version": runner.ATTEMPT_SCHEMA,
        "status": contract["attempt_status"],
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "execution_stage": stage,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "expected_calls": contract["expected_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "stage_start_row": contract["stage_start_row"],
        "stage_stop_row": contract["stage_stop_row"],
        "block_order": contract["block_order"],
        "block_counts": contract["block_counts"],
        "partial_resume_allowed": False,
        "generation_used": False,
        "model": plan["model"],
        "forward_contract": _expected_forward_contract(plan),
    }
    for key, expected in required.items():
        if attempt.get(key) != expected:
            raise R2TopologyAnalysisError(f"stage attempt lock changed: {key}")
    _validate_preflight(attempt.get("preflight"), plan, stage=stage)
    _validate_source_locks(attempt.get("source_locks"), bundle, "stage attempt")
    if stage == REMAINING_STAGE:
        observed_prior = attempt.get("prior_discovery_authorization")
        if not isinstance(observed_prior, Mapping):
            raise R2TopologyAnalysisError(
                "remaining attempt prior-discovery authorization is missing"
            )
        _require_exact_keys(
            observed_prior,
            PRIOR_DISCOVERY_AUTHORIZATION_FIELDS,
            "remaining attempt prior-discovery authorization",
        )
        for key in PRIOR_DISCOVERY_AUTHORIZATION_FIELDS - {"authorization_status"}:
            _require_sha256(observed_prior.get(key), f"remaining attempt {key}")
        if observed_prior.get("authorization_status") != DISCOVERY_COMPONENT_PASS:
            raise R2TopologyAnalysisError(
                "remaining attempt discovery authorization status changed"
            )
        if prior_discovery_authorization is not None and dict(observed_prior) != dict(
            prior_discovery_authorization
        ):
            raise R2TopologyAnalysisError(
                "remaining attempt prior-discovery authorization changed"
            )


def _validate_execution_manifest_header(
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    stage: str,
    prior_discovery_authorization: Mapping[str, Any] | None = None,
) -> None:
    runner = bundle["runner"]
    plan = bundle["plan"]
    contract = _stage_contract(stage)
    expected_fields = set(EXECUTION_MANIFEST_COMMON_FIELDS)
    if stage == DISCOVERY_STAGE:
        expected_fields.add("records")
    else:
        expected_fields.update({"record_blocks", "prior_discovery_authorization"})
    _require_exact_keys(manifest, expected_fields, "stage execution manifest")
    required = {
        "schema_version": runner.EXECUTION_SCHEMA,
        "status": contract["complete_status"],
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "execution_stage": stage,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "phase_model_calls": contract["expected_calls"],
        "cumulative_model_calls": contract["cumulative_model_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "block_order": contract["block_order"],
        "block_counts": contract["block_counts"],
        "generation_used": False,
        "partial_resume_allowed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise R2TopologyAnalysisError(f"stage execution manifest changed: {key}")
    _validate_source_locks(manifest.get("source_locks"), bundle, "execution manifest")
    if stage == REMAINING_STAGE:
        observed_prior = manifest.get("prior_discovery_authorization")
        if not isinstance(observed_prior, Mapping):
            raise R2TopologyAnalysisError(
                "remaining manifest prior-discovery authorization is missing"
            )
        _require_exact_keys(
            observed_prior,
            PRIOR_DISCOVERY_AUTHORIZATION_FIELDS,
            "remaining manifest prior-discovery authorization",
        )
        for key in PRIOR_DISCOVERY_AUTHORIZATION_FIELDS - {"authorization_status"}:
            _require_sha256(observed_prior.get(key), f"remaining manifest {key}")
        if observed_prior.get("authorization_status") != DISCOVERY_COMPONENT_PASS:
            raise R2TopologyAnalysisError(
                "remaining manifest discovery authorization status changed"
            )
        if prior_discovery_authorization is not None and dict(observed_prior) != dict(
            prior_discovery_authorization
        ):
            raise R2TopologyAnalysisError(
                "remaining manifest prior-discovery authorization changed"
            )


def _validate_stage_shard_registry(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    stage: str,
    raw_root: Path,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], Path]]:
    """Validate every shard opaquely without deserializing a logit matrix."""

    expected_specs = plan.get("raw_logits_shards", {}).get(stage)
    bindings = manifest.get("raw_logits_shards")
    if not isinstance(expected_specs, list) or not isinstance(bindings, list):
        raise R2TopologyAnalysisError("stage raw-logit registry is missing")
    expected_count = 10 if stage == DISCOVERY_STAGE else 26
    if len(expected_specs) != expected_count or len(bindings) != expected_count:
        raise R2TopologyAnalysisError("stage raw-logit shard count changed")
    expected_paths = {Path(str(spec["path"])).resolve() for spec in expected_specs}
    try:
        observed_paths = (
            {path.resolve() for path in raw_root.iterdir()} if raw_root.is_dir() else set()
        )
    except OSError as error:
        raise R2TopologyAnalysisError("cannot enumerate stage raw-logit root") from error
    if observed_paths != expected_paths:
        raise R2TopologyAnalysisError("stage raw-logit directory contains unbound entries")

    registry: list[tuple[Mapping[str, Any], Mapping[str, Any], Path]] = []
    for expected, binding in zip(expected_specs, bindings, strict=True):
        if not isinstance(expected, Mapping) or not isinstance(binding, Mapping):
            raise R2TopologyAnalysisError("stage raw-logit binding is malformed")
        raw_binding_fields = set(expected) | {
            "file_sha256",
            "size_bytes",
            "logical_sha256",
        }
        _require_exact_keys(binding, raw_binding_fields, "raw-logit shard binding")
        for key, value in expected.items():
            if binding.get(key) != value:
                raise R2TopologyAnalysisError("stage raw-logit plan binding changed")
        _require_sha256(binding.get("logical_sha256"), "raw-logit logical hash")
        path = _binding_path(
            binding,
            Path(str(expected["path"])),
            "raw-logit shard",
            fields=raw_binding_fields,
        )
        registry.append((expected, binding, path))
    return registry


def _load_stage_shard_subset(
    registry: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Path]],
    shard_indices: Sequence[int],
) -> dict[int, np.ndarray]:
    """Deserialize only explicitly admitted shard indices."""

    requested = list(shard_indices)
    if len(set(requested)) != len(requested) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in requested
    ):
        raise R2TopologyAnalysisError("raw-logit shard subset is invalid")
    arrays: dict[int, np.ndarray] = {}
    for index in requested:
        if not 0 <= index < len(registry):
            raise R2TopologyAnalysisError("raw-logit shard subset is out of range")
        expected, binding, path = registry[index]
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise R2TopologyAnalysisError("cannot load a stage raw-logit shard") from error
        if (
            list(array.shape) != expected["shape"]
            or array.dtype != np.dtype("<f4")
            or binding.get("logical_sha256") != _logical_array_sha256(array)
        ):
            raise R2TopologyAnalysisError("stage raw-logit shard content changed")
        arrays[index] = array
    return arrays


def _load_stage_shards(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    stage: str,
    raw_root: Path,
) -> list[np.ndarray]:
    registry = _validate_stage_shard_registry(
        manifest,
        plan,
        stage=stage,
        raw_root=raw_root,
    )
    arrays = _load_stage_shard_subset(registry, range(len(registry)))
    return [arrays[index] for index in range(len(registry))]


def _validate_remaining_envelope(
    bundle: Mapping[str, Any],
    *,
    prior_discovery_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash the sealed remaining stage without opening any behavioral payload."""

    runner = bundle["runner"]
    paths = bundle["paths"]
    plan = bundle["plan"]
    manifest = _load_json(paths["remaining_execution_manifest"])
    _validate_execution_manifest_header(
        manifest,
        bundle,
        stage=REMAINING_STAGE,
        prior_discovery_authorization=prior_discovery_authorization,
    )

    attempt_binding = manifest.get("attempt")
    if not isinstance(attempt_binding, Mapping):
        raise R2TopologyAnalysisError("remaining attempt binding is missing")
    attempt_path = _binding_path(
        attempt_binding,
        paths["remaining_attempt"],
        "remaining attempt",
        fields=FILE_BINDING_FIELDS,
    )
    attempt = _load_json(attempt_path)
    _validate_attempt(
        attempt,
        bundle,
        stage=REMAINING_STAGE,
        prior_discovery_authorization=prior_discovery_authorization,
    )
    if manifest["source_locks"] != attempt["source_locks"]:
        raise R2TopologyAnalysisError("remaining manifest/attempt source locks differ")
    if (
        manifest["prior_discovery_authorization"]
        != attempt["prior_discovery_authorization"]
    ):
        raise R2TopologyAnalysisError(
            "remaining manifest/attempt discovery authorization differs"
        )

    record_blocks = manifest.get("record_blocks")
    if not isinstance(record_blocks, Mapping) or set(record_blocks) != set(
        REMAINING_BLOCK_REGISTRY
    ):
        raise R2TopologyAnalysisError("remaining record-block registry changed")
    validated_blocks: dict[str, dict[str, Any]] = {}
    for block, registry in REMAINING_BLOCK_REGISTRY.items():
        binding = record_blocks.get(block)
        if not isinstance(binding, Mapping):
            raise R2TopologyAnalysisError(f"remaining {block} binding is missing")
        expected_fields = {
            "execution_block": block,
            "row_count": registry["stage_stop_row"] - registry["stage_start_row"],
            "global_start_row": registry["global_start_row"],
            "global_stop_row": registry["global_stop_row"],
            "stage_start_row": registry["stage_start_row"],
            "stage_stop_row": registry["stage_stop_row"],
            "raw_shard_indices": list(
                range(registry["shard_start"], registry["shard_stop"])
            ),
        }
        binding_fields = FILE_BINDING_FIELDS | frozenset(expected_fields) | {
            "canonical_sha256"
        }
        _require_exact_keys(
            binding,
            binding_fields,
            f"remaining {block} record-block binding",
        )
        if any(binding.get(key) != value for key, value in expected_fields.items()):
            raise R2TopologyAnalysisError(f"remaining {block} range binding changed")
        _require_sha256(
            binding.get("canonical_sha256"),
            f"remaining {block} canonical record hash",
        )
        path = _binding_path(
            binding,
            paths[str(registry["path_key"])],
            f"remaining {block} records",
            fields=binding_fields,
        )
        validated_blocks[block] = {"binding": binding, "path": path}

    shard_registry = _validate_stage_shard_registry(
        manifest,
        plan,
        stage=REMAINING_STAGE,
        raw_root=paths["remaining_raw_root"],
    )
    for block, registry in REMAINING_BLOCK_REGISTRY.items():
        for index in range(registry["shard_start"], registry["shard_stop"]):
            expected, binding, _path = shard_registry[index]
            if (
                expected.get("execution_block") != block
                or binding.get("execution_block") != block
            ):
                raise R2TopologyAnalysisError(
                    f"remaining {block} raw-shard allocation changed"
                )

    prompts = [
        prompt
        for prompt in plan["prompts"]
        if prompt["execution_stage"] == REMAINING_STAGE
    ]
    if len(prompts) != REMAINING_CALLS:
        raise R2TopologyAnalysisError("remaining prompt registry changed")
    for block, registry in REMAINING_BLOCK_REGISTRY.items():
        start = registry["stage_start_row"]
        stop = registry["stage_stop_row"]
        if any(prompt.get("execution_block") != block for prompt in prompts[start:stop]):
            raise R2TopologyAnalysisError(f"remaining {block} prompt slice changed")

    return {
        "manifest": manifest,
        "attempt": attempt,
        "record_blocks": validated_blocks,
        "shard_registry": shard_registry,
        "prompts": prompts,
        "loaded_blocks": [],
        "artifact_hashes": {
            "attempt_file_sha256": file_sha256(attempt_path),
            "execution_manifest_file_sha256": file_sha256(
                paths["remaining_execution_manifest"]
            ),
            "record_block_file_sha256": {
                block: value["binding"]["file_sha256"]
                for block, value in validated_blocks.items()
            },
            "raw_shard_file_sha256": [
                binding["file_sha256"] for _expected, binding, _path in shard_registry
            ],
        },
    }


def _compare_diagnostics(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    _require_exact_keys(observed, DIAGNOSTIC_FIELDS, "stored component diagnostics")
    if set(expected) != DIAGNOSTIC_FIELDS:
        raise R2TopologyAnalysisError("reconstructed diagnostic schema changed")
    for key, value in expected.items():
        candidate = observed.get(key)
        if isinstance(value, float):
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, (int, float))
                or not math.isclose(float(candidate), value, rel_tol=0.0, abs_tol=0.0)
            ):
                raise R2TopologyAnalysisError(f"stored diagnostic changed: {key}")
        elif candidate != value:
            raise R2TopologyAnalysisError(f"stored diagnostic changed: {key}")


def _validate_stage_record_structure(
    record: Mapping[str, Any],
    prompt: Mapping[str, Any],
    row: np.ndarray,
    *,
    stage: str,
    stage_call_index: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    expected_record_fields = set(RECORD_COMMON_FIELDS)
    if prompt.get("family") != TWO_FACT:
        expected_record_fields.add("diagnostics")
    _require_exact_keys(record, expected_record_fields, "stage record")
    global_index = prompt["global_call_index"]
    runner = _runner_module()
    try:
        expected_record_identity = runner.record_identity_id(prompt)
    except Exception as error:
        raise R2TopologyAnalysisError(
            "planned record identity cannot be reconstructed"
        ) from error
    if prompt.get("record_identity_id") != expected_record_identity:
        raise R2TopologyAnalysisError("planned record identity changed")
    required_identity = {
        "schema_version": runner.RECORD_SCHEMA,
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "record_identity_id": expected_record_identity,
        "execution_stage": stage,
        "execution_block": prompt["execution_block"],
        "global_call_index": global_index,
        "block_call_index": prompt["block_call_index"],
        "stage_call_index": stage_call_index,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "cell_id": prompt["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "world_id": prompt["world_id"],
        "role": prompt["role"],
        "family": prompt["family"],
        "expected_token_id": prompt["expected_token_id"],
        "distractor_token_id": prompt["distractor_token_id"],
    }
    for key, value in required_identity.items():
        if record.get(key) != value:
            raise R2TopologyAnalysisError(f"stage record identity changed: {key}")
    for key in ("factors", "factor_levels"):
        if record.get(key) != prompt[key]:
            raise R2TopologyAnalysisError(f"stage record {key} changed")
    if record.get("model_calls") != 1 or record.get("generation_used") is not False:
        raise R2TopologyAnalysisError("stage record call semantics changed")
    if record.get("teacher_forced_prompt_forward") is not True:
        raise R2TopologyAnalysisError("stage record is not a teacher-forced forward")
    if record.get("raw_logits_row_sha256") != f32_sha256(row):
        raise R2TopologyAnalysisError("stage record raw-logit row hash changed")
    core = {key: value for key, value in record.items() if key != "record_id"}
    if record.get("record_id") != canonical_sha256(core):
        raise R2TopologyAnalysisError("stage record canonical ID changed")
    return {
        **dict(record),
        "call_index": int(global_index),
        "split": str(prompt["role"]),
        "expected_answer": str(prompt["semantic_answer"]),
        "distractor_answer": str(prompt["semantic_distractor"]),
    }


def _normalize_stage_record(
    record: Mapping[str, Any],
    prompt: Mapping[str, Any],
    row: np.ndarray,
    *,
    stage: str,
    stage_call_index: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _validate_stage_record_structure(
        record,
        prompt,
        row,
        stage=stage,
        stage_call_index=stage_call_index,
        plan=plan,
    )
    diagnostics = diagnostics_from_full_vocab(
        row,
        int(prompt["expected_token_id"]),
        int(prompt["distractor_token_id"]),
    )
    observed_diagnostics = record.get("diagnostics")
    if not isinstance(observed_diagnostics, Mapping):
        raise R2TopologyAnalysisError("stage record diagnostics are missing")
    _compare_diagnostics(observed_diagnostics, diagnostics)
    normalized["diagnostics"] = diagnostics
    normalized["topology_diagnostics_deferred"] = False
    return normalized


def _normalize_topology_record(
    record: Mapping[str, Any],
    prompt: Mapping[str, Any],
    row: np.ndarray,
    *,
    stage_call_index: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive topology outcomes only after admission; none may be pre-stored."""

    if "diagnostics" in record:
        raise R2TopologyAnalysisError(
            "topology record contains prohibited pre-derived diagnostics"
        )
    normalized = _validate_stage_record_structure(
        record,
        prompt,
        row,
        stage=REMAINING_STAGE,
        stage_call_index=stage_call_index,
        plan=plan,
    )
    normalized["diagnostics"] = diagnostics_from_full_vocab(
        row,
        int(prompt["expected_token_id"]),
        int(prompt["distractor_token_id"]),
    )
    normalized["topology_diagnostics_derived_after_confirmation_gate"] = True
    return normalized


def _load_remaining_block(
    bundle: Mapping[str, Any],
    envelope: Mapping[str, Any],
    block: str,
) -> dict[str, Any]:
    """Open exactly one physically separated remaining-stage block."""

    if block not in REMAINING_BLOCK_REGISTRY:
        raise R2TopologyAnalysisError("unknown remaining record block")
    registry = REMAINING_BLOCK_REGISTRY[block]
    block_artifact = envelope["record_blocks"][block]
    records = _load_jsonl(block_artifact["path"])
    expected_count = registry["stage_stop_row"] - registry["stage_start_row"]
    if (
        len(records) != expected_count
        or block_artifact["binding"].get("canonical_sha256")
        != canonical_sha256(records)
    ):
        raise R2TopologyAnalysisError(f"remaining {block} record count changed")

    shard_indices = range(registry["shard_start"], registry["shard_stop"])
    shards = _load_stage_shard_subset(envelope["shard_registry"], shard_indices)
    prompts = envelope["prompts"]
    plan = bundle["plan"]
    normalized: list[dict[str, Any]] = []
    for offset, record in enumerate(records):
        stage_index = registry["stage_start_row"] + offset
        prompt = prompts[stage_index]
        shard_index, row_index = divmod(stage_index, RAW_SHARD_ROWS)
        if (
            record.get("raw_logits_shard_index") != shard_index
            or record.get("raw_logits_row_in_shard") != row_index
            or record.get("raw_logits_stage_row") != stage_index
            or record.get("raw_logits_global_row") != prompt["global_call_index"]
        ):
            raise R2TopologyAnalysisError("remaining raw-logit row location changed")
        row = np.ascontiguousarray(shards[shard_index][row_index], dtype="<f4")
        if block == "confirmation-components":
            value = _normalize_stage_record(
                record,
                prompt,
                row,
                stage=REMAINING_STAGE,
                stage_call_index=stage_index,
                plan=plan,
            )
        else:
            value = _normalize_topology_record(
                record,
                prompt,
                row,
                stage_call_index=stage_index,
                plan=plan,
            )
        normalized.append(value)
    if (
        len({record["record_id"] for record in normalized}) != expected_count
        or any(record["execution_block"] != block for record in normalized)
    ):
        raise R2TopologyAnalysisError(f"remaining {block} identity registry changed")
    return {
        "block": block,
        "records": normalized,
        "loaded_shard_indices": list(shard_indices),
        "record_file_sha256": block_artifact["binding"]["file_sha256"],
    }


def _validate_stage_execution(
    bundle: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    if stage == REMAINING_STAGE:
        raise R2TopologyAnalysisError(
            "generic stage validation is discovery-only; remaining-main must use "
            "the explicit confirmation-first final-analysis path"
        )

    runner = bundle["runner"]
    paths = bundle["paths"]
    plan = bundle["plan"]
    if stage == DISCOVERY_STAGE:
        attempt_path = paths["discovery_attempt"]
        records_path = paths["discovery_records"]
        execution_path = paths["discovery_execution_manifest"]
        raw_root = paths["discovery_raw_root"]
        expected_calls = DISCOVERY_CALLS
        expected_status = "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_COMPLETE_NOT_ANALYZED"
    else:
        raise R2TopologyAnalysisError("unknown R2 execution stage")

    manifest = _load_json(execution_path)
    _validate_execution_manifest_header(manifest, bundle, stage=stage)
    if manifest.get("status") != expected_status:
        raise R2TopologyAnalysisError("stage execution status changed")
    cumulative = DISCOVERY_CALLS if stage == DISCOVERY_STAGE else TOTAL_CALLS
    if manifest.get("cumulative_model_calls") != cumulative:
        raise R2TopologyAnalysisError("stage cumulative model-call count changed")

    attempt_binding = manifest.get("attempt")
    records_binding = manifest.get("records")
    if not isinstance(attempt_binding, Mapping) or not isinstance(records_binding, Mapping):
        raise R2TopologyAnalysisError("stage artifact bindings are missing")
    _binding_path(
        attempt_binding,
        attempt_path,
        "stage attempt",
        fields=FILE_BINDING_FIELDS,
    )
    record_binding_fields = FILE_BINDING_FIELDS | {"count", "canonical_sha256"}
    _binding_path(
        records_binding,
        records_path,
        "stage records",
        fields=record_binding_fields,
    )
    _require_sha256(
        records_binding.get("canonical_sha256"),
        "stage records canonical hash",
    )
    attempt = _load_json(attempt_path)
    _validate_attempt(attempt, bundle, stage=stage)
    if manifest["source_locks"] != attempt["source_locks"]:
        raise R2TopologyAnalysisError("discovery manifest/attempt source locks differ")

    records = _load_jsonl(records_path)
    if (
        len(records) != expected_calls
        or records_binding.get("count") != expected_calls
        or records_binding.get("canonical_sha256") != canonical_sha256(records)
    ):
        raise R2TopologyAnalysisError("stage record commitment changed")
    shards = _load_stage_shards(
        manifest,
        plan,
        stage=stage,
        raw_root=raw_root,
    )
    prompts = [
        prompt for prompt in plan["prompts"] if prompt["execution_stage"] == stage
    ]
    if len(prompts) != expected_calls:
        raise R2TopologyAnalysisError("stage prompt registry changed")
    normalized: list[dict[str, Any]] = []
    for stage_index, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
        shard_index, row_index = divmod(stage_index, RAW_SHARD_ROWS)
        if (
            record.get("raw_logits_shard_index") != shard_index
            or record.get("raw_logits_row_in_shard") != row_index
            or record.get("raw_logits_stage_row") != stage_index
            or record.get("raw_logits_global_row") != prompt["global_call_index"]
        ):
            raise R2TopologyAnalysisError("stage raw-logit row location changed")
        row = np.ascontiguousarray(shards[shard_index][row_index], dtype="<f4")
        value = _normalize_stage_record(
            record,
            prompt,
            row,
            stage=stage,
            stage_call_index=stage_index,
            plan=plan,
        )
        normalized.append(value)
    if len({row["record_id"] for row in normalized}) != expected_calls:
        raise R2TopologyAnalysisError("stage record IDs are duplicated")
    if stage == DISCOVERY_STAGE and any(
        row["family"] == TWO_FACT or row["execution_block"] != "discovery-components"
        for row in normalized
    ):
        raise R2TopologyAnalysisError("discovery execution exposed a topology row")
    return {
        "manifest": manifest,
        "attempt": attempt,
        "records": normalized,
        "raw_shards": shards,
        "artifact_hashes": {
            "attempt_file_sha256": file_sha256(attempt_path),
            "records_file_sha256": file_sha256(records_path),
            "execution_manifest_file_sha256": file_sha256(execution_path),
            "raw_shard_file_sha256": [
                binding["file_sha256"] for binding in manifest["raw_logits_shards"]
            ],
        },
    }


def _engineering_invalid(stage: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "stage": stage,
        "status": ENGINEERING_INVALID,
        "engineering_valid": False,
        "component_qualified": False,
        "authorization_issued": False,
        "error": str(error),
        "model_calls_issued_by_analyzer": 0,
        "generation_calls_issued_by_analyzer": 0,
        "activation_design_authorized": False,
        "activation_execution_authorized": False,
    }


def _discovery_analysis_payload(
    bundle: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = discovery_gate_from_validated_records(execution["records"])
    passed = analysis["remaining_main_authorized"] is True
    analysis.update(
        {
            "component_qualified": passed,
            "authorization_issued": passed,
            "call_plan_sha256": bundle["plan"]["call_plan_sha256"],
            "stage_plan_sha256": bundle["plan"]["stage_plan_sha256"][
                DISCOVERY_STAGE
            ],
            "artifact_validation": execution["artifact_hashes"],
            "claim_scope": "discovery_components_only_no_topology_result",
        }
    )
    return analysis


def analyze_discovery_gate(
    result_root: Path = RESULT_ROOT,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Replay and gate only discovery components, never topology artifacts."""

    if write:
        _require_registered_analysis_write(result_root, stage="discovery-gate")
    try:
        bundle = replay_frozen_plan(result_root)
        execution = _validate_stage_execution(bundle, stage=DISCOVERY_STAGE)
        analysis = _discovery_analysis_payload(bundle, execution)
    except Exception as error:
        analysis = _engineering_invalid("discovery-gate", error)
    if write:
        paths = _runner_module().artifact_paths(result_root)
        _write_frozen_json(
            paths["discovery_analysis"],
            analysis,
            result_root=result_root,
        )
    return analysis


def _replay_discovery_authorization(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Transitively replay discovery and exact-compare its frozen PASS."""

    path = bundle["paths"]["discovery_analysis"]
    authorization = _load_json(path)
    execution = _validate_stage_execution(bundle, stage=DISCOVERY_STAGE)
    replayed = _discovery_analysis_payload(bundle, execution)
    plan = bundle["plan"]
    if (
        authorization.get("status") != DISCOVERY_COMPONENT_PASS
        or authorization.get("engineering_valid") is not True
        or authorization.get("component_qualified") is not True
        or authorization.get("authorization_issued") is not True
        or authorization.get("call_plan_sha256") != plan["call_plan_sha256"]
        or authorization.get("stage_plan_sha256")
        != plan["stage_plan_sha256"][DISCOVERY_STAGE]
        or authorization.get("topology_rows_read") != 0
        or authorization.get("topology_inference_performed") is not False
        or authorization.get("remaining_main_authorized") is not True
        or authorization != replayed
        or canonical_sha256(authorization) != canonical_sha256(replayed)
    ):
        raise R2TopologyAnalysisError(
            "remaining-main discovery authorization does not replay exactly"
        )
    execution_manifest_path = bundle["paths"]["discovery_execution_manifest"]
    prior = {
        "execution_manifest_file_sha256": file_sha256(execution_manifest_path),
        "execution_manifest_canonical_sha256": canonical_sha256(
            execution["manifest"]
        ),
        "authorization_file_sha256": file_sha256(path),
        "authorization_canonical_sha256": canonical_sha256(authorization),
        "authorization_status": authorization["status"],
    }
    _require_exact_keys(
        prior,
        PRIOR_DISCOVERY_AUTHORIZATION_FIELDS,
        "replayed prior-discovery authorization",
    )
    return authorization, execution, prior


def validate_remaining_authorization_for_runner(
    result_root: Path = RESULT_ROOT,
    *,
    dependency_replay_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """No-write, no-model remaining-stage authorization for the runner."""

    bundle = replay_frozen_plan(
        result_root,
        dependency_replay_override=dependency_replay_override,
    )
    authorization, _execution, prior = _replay_discovery_authorization(bundle)
    return {
        "authorization": authorization,
        "authorization_file_sha256": prior["authorization_file_sha256"],
        "authorization_canonical_sha256": prior[
            "authorization_canonical_sha256"
        ],
        "discovery_execution_manifest_file_sha256": prior[
            "execution_manifest_file_sha256"
        ],
        "discovery_execution_manifest_canonical_sha256": prior[
            "execution_manifest_canonical_sha256"
        ],
        "authorization_status": prior["authorization_status"],
        "call_plan_sha256": bundle["plan"]["call_plan_sha256"],
        "stage_plan_sha256": bundle["plan"]["stage_plan_sha256"][DISCOVERY_STAGE],
        "replay_equal": True,
        "model_calls_issued_by_validator": 0,
        "generation_calls_issued_by_validator": 0,
    }


def analyze_final(
    result_root: Path = RESULT_ROOT,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Validate all 2,304 rows and gate confirmation before topology inference."""

    if write:
        _require_registered_analysis_write(result_root, stage="final")
    try:
        bundle = replay_frozen_plan(result_root)
        authorization, discovery, prior_authorization = (
            _replay_discovery_authorization(bundle)
        )
        envelope = _validate_remaining_envelope(
            bundle,
            prior_discovery_authorization=prior_authorization,
        )
        confirmation = _load_remaining_block(
            bundle,
            envelope,
            "confirmation-components",
        )
        discovery_components = component_summary(
            discovery["records"],
            split="discovery",
        )
        if discovery_components["gates"]["pass"] is not True:
            raise R2TopologyAnalysisError(
                "completed remaining-main stage lacks a passing discovery authorization"
            )
        confirmation_components = component_summary(
            confirmation["records"],
            split="confirmation",
        )
        if confirmation_components["gates"]["pass"] is not True:
            analysis = {
                "schema_version": ANALYSIS_SCHEMA,
                "stage": "final",
                "status": CONFIRMATION_COMPONENT_FAIL,
                "engineering_valid": True,
                "discovery_components": discovery_components,
                "confirmation_components": confirmation_components,
                "topology_suppressed": True,
                "topology_record_streams_opened": 0,
                "topology_raw_shards_loaded": 0,
                "topology_rows_interpreted": 0,
                "topology_artifacts_opaque_hash_validated": 1024,
                "topology_inference_performed": False,
                "activation_design_authorized": False,
                "activation_execution_authorized": False,
                "model_calls_issued_by_analyzer": 0,
                "generation_calls_issued_by_analyzer": 0,
            }
            semantic_rows_validated = 1280
        else:
            discovery_topology = _load_remaining_block(
                bundle,
                envelope,
                "discovery-topology",
            )
            confirmation_topology = _load_remaining_block(
                bundle,
                envelope,
                "confirmation-topology",
            )
            remaining_records = [
                *discovery_topology["records"],
                *confirmation["records"],
                *confirmation_topology["records"],
            ]
            if len(discovery["records"]) + len(remaining_records) != TOTAL_CALLS:
                raise R2TopologyAnalysisError(
                    "final analysis did not validate all 2,304 rows"
                )
            analysis = final_from_validated_records(
                discovery["records"],
                remaining_records,
            )
            semantic_rows_validated = TOTAL_CALLS
        analysis.update(
            {
                "component_qualified": analysis["status"]
                not in {CONFIRMATION_COMPONENT_FAIL, ENGINEERING_INVALID},
                "authorization_issued": analysis["status"] == REVERSAL_SUPPORTED,
                "call_plan_sha256": bundle["plan"]["call_plan_sha256"],
                "stage_plan_sha256": bundle["plan"]["stage_plan_sha256"],
                "discovery_authorization_file_sha256": file_sha256(
                    bundle["paths"]["discovery_analysis"]
                ),
                "artifact_validation": {
                    "opaque_file_bound_row_count": TOTAL_CALLS,
                    "semantically_deserialized_row_count": semantic_rows_validated,
                    "topology_rows_not_opened_before_confirmation_gate": 1024,
                    "discovery": discovery["artifact_hashes"],
                    "remaining_main": envelope["artifact_hashes"],
                    "discovery_authorization_sha256": file_sha256(
                        bundle["paths"]["discovery_analysis"]
                    ),
                    "discovery_authorization_status": authorization["status"],
                },
                "claim_boundary": (
                    "Behavior-only inference in one pinned model and synthetic prompt family; "
                    "no activation-gap, latent-knowledge, biological, physical-law, universal, "
                    "or model-family claim."
                ),
            }
        )
    except Exception as error:
        analysis = _engineering_invalid("final", error)
    if write:
        paths = _runner_module().artifact_paths(result_root)
        _write_frozen_json(
            paths["final_analysis"],
            analysis,
            result_root=result_root,
        )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("discovery-gate", "final"),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "discovery-gate":
        analysis = analyze_discovery_gate()
    else:
        analysis = analyze_final()
    print(canonical_json({"stage": args.stage, "status": analysis["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

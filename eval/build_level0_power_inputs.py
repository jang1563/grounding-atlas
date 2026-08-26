"""Build authenticated Phase-0 inputs from a passed development Level-0 result.

This bridge extracts already donor-aggregated O/R/I nuisance contrasts for every
manifest readout-by-input-family group.  It emits only a development donor
matrix and a ``level0_only`` power configuration.  It does not run the power
simulation and cannot select or claim a final confirmatory donor count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from eval import coherent_binary_readout as coherent
    from eval import phase0_power as phase0
except ImportError:  # pragma: no cover - direct script execution
    import coherent_binary_readout as coherent
    import phase0_power as phase0


FROZEN_SEED = 20_260_802
FROZEN_REPLICATES = 10_000
FROZEN_MARGIN = 0.06
FROZEN_SUPPORT = (-2.0, 2.0)
FROZEN_MAX_OUT_OF_SUPPORT_FRACTION = 0.001
LEVEL0_ONLY_SCENARIO = "level0_only_all_nuisance_contrasts"

RESULT_KEYS = {
    "artifact_type",
    "schema_version",
    "status",
    "mode",
    "margin_lock_status",
    "level0_pass",
    "design",
    "design_sha256",
    "raw_records_sha256",
    "full_vocab_sidecar_sha256",
    "analysis_code_sha256",
    "n_records",
    "n_items",
    "n_donors",
    "donor_ids",
    "validation",
    "global_format_adherence",
    "groups",
    "claim_boundary",
}
GROUP_KEYS = {
    "readout_id",
    "input_family",
    "n_donors",
    "n_items",
    "n_records",
    "extraction_coherence",
    "format_adherence",
    "nuisance_equivalence",
    "item_guardrail",
    "donor_effects",
    "pass",
}
ESTIMANDS = ("O", "R", "I")


class Level0PowerInputError(ValueError):
    """Raised when a Level-0 result cannot enter Phase 0."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise Level0PowerInputError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Level0PowerInputError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Level0PowerInputError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Level0PowerInputError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise Level0PowerInputError(f"{label} must be finite")
    return output


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Level0PowerInputError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise Level0PowerInputError(f"{label} must be a lowercase SHA-256 digest") from error
    if value != value.lower():
        raise Level0PowerInputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_id(readout: str, family: str, estimand: str) -> str:
    return f"level0::{readout}::{family}::{estimand}"


def validate_passed_development_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the artifact is a complete passed development result."""

    checked = _mapping(result, "Level-0 result")
    _exact_keys(checked, RESULT_KEYS, "Level-0 result")
    if checked["artifact_type"] != "groundbench.level0_coherent_binary_readout":
        raise Level0PowerInputError("unexpected Level-0 artifact_type")
    if checked["schema_version"] != coherent.ANALYSIS_SCHEMA:
        raise Level0PowerInputError("unsupported Level-0 analysis schema")
    if checked["mode"] != "development":
        raise Level0PowerInputError("only development-mode Level-0 results may enter Phase 0")
    if checked["status"] not in {
        "DEVELOPMENT_LEVEL0_CANDIDATE_PASS_MARGIN_NOT_QUALIFIED",
        "DEVELOPMENT_LEVEL0_PASS_NOT_CONFIRMATORY",
    }:
        raise Level0PowerInputError("development Level-0 result does not have the passing status")
    if checked["level0_pass"] is not True:
        raise Level0PowerInputError("development Level-0 result did not pass")

    design = coherent.validate_design(_mapping(checked["design"], "Level-0 design"))
    if design["mode"] != "development":
        raise Level0PowerInputError("embedded Level-0 design is not development mode")
    if checked["margin_lock_status"] != design["margin_lock_status"]:
        raise Level0PowerInputError("result margin-lock status does not match design")
    if design["expected_confirmatory_donors"] is not None:
        raise Level0PowerInputError("development design declares confirmatory donors")
    if not math.isclose(design["equivalence_margin"], FROZEN_MARGIN, abs_tol=1e-12):
        raise Level0PowerInputError("Level-0 equivalence margin is not frozen at 0.06")
    if not math.isclose(design["alpha"], 0.05, abs_tol=1e-12):
        raise Level0PowerInputError("Level-0 alpha is not frozen at 0.05")
    if checked["design_sha256"] != coherent.canonical_sha256(design):
        raise Level0PowerInputError("Level-0 design SHA-256 does not match embedded design")
    _sha256(checked["raw_records_sha256"], "raw_records_sha256")
    _sha256(checked["full_vocab_sidecar_sha256"], "full_vocab_sidecar_sha256")
    _sha256(checked["analysis_code_sha256"], "analysis_code_sha256")

    donor_ids = checked["donor_ids"]
    if (
        not isinstance(donor_ids, list)
        or not donor_ids
        or any(not isinstance(donor, str) or not donor for donor in donor_ids)
        or donor_ids != sorted(set(donor_ids))
    ):
        raise Level0PowerInputError("donor_ids must be a nonempty sorted unique list")
    n_donors = _positive_int(checked["n_donors"], "n_donors")
    if n_donors != len(donor_ids) or donor_ids != design["expected_donor_ids"]:
        raise Level0PowerInputError("result donors do not match the frozen development design")
    n_records = _positive_int(checked["n_records"], "n_records")
    n_items = _positive_int(checked["n_items"], "n_items")
    if n_records != len(design["expected_record_ids"]):
        raise Level0PowerInputError("result record count does not match the frozen call plan")

    validation = _mapping(checked["validation"], "validation")
    if (
        validation.get("expected_records") != n_records
        or validation.get("observed_records") != n_records
        or validation.get("missing_records") != []
        or validation.get("unexpected_records") != []
        or validation.get("finite_logits") is not True
    ):
        raise Level0PowerInputError("Level-0 call-plan validation is not complete")
    global_format = _mapping(checked["global_format_adherence"], "global_format_adherence")
    if global_format.get("pass") is not True:
        raise Level0PowerInputError("global Level-0 format adherence did not pass")

    expected_groups = {
        f"{readout}::{family}": (readout, family)
        for readout in design["required_readouts"]
        for family in design["required_input_families"]
    }
    groups = _mapping(checked["groups"], "groups")
    if set(groups) != set(expected_groups):
        raise Level0PowerInputError("Level-0 groups do not cover the manifest cross-product")

    normalized_groups: dict[str, Any] = {}
    total_items = 0
    total_records = 0
    for group_id, (readout, family) in sorted(expected_groups.items()):
        group = _mapping(groups[group_id], f"group {group_id}")
        _exact_keys(group, GROUP_KEYS, f"group {group_id}")
        if group["readout_id"] != readout or group["input_family"] != family:
            raise Level0PowerInputError(f"group {group_id} identity mismatch")
        if group["pass"] is not True:
            raise Level0PowerInputError(f"group {group_id} did not pass")
        if group["n_donors"] != n_donors:
            raise Level0PowerInputError(f"group {group_id} donor count mismatch")
        group_items = _positive_int(group["n_items"], f"group {group_id} n_items")
        group_records = _positive_int(group["n_records"], f"group {group_id} n_records")
        total_items += group_items
        total_records += group_records
        for gate_name in (
            "extraction_coherence",
            "format_adherence",
            "item_guardrail",
        ):
            gate = _mapping(group[gate_name], f"group {group_id} {gate_name}")
            if gate.get("pass") is not True:
                raise Level0PowerInputError(f"group {group_id} {gate_name} did not pass")

        donor_effects = _mapping(group["donor_effects"], f"group {group_id} effects")
        if set(donor_effects) != set(donor_ids):
            raise Level0PowerInputError(f"group {group_id} donor coverage mismatch")
        normalized_effects: dict[str, dict[str, float]] = {}
        for donor in donor_ids:
            effects = _mapping(donor_effects[donor], f"group {group_id} donor {donor} effects")
            if set(effects) != set(ESTIMANDS):
                raise Level0PowerInputError(f"group {group_id} donor {donor} must contain O/R/I")
            normalized_effects[donor] = {
                estimand: _finite(
                    effects[estimand],
                    f"group {group_id} donor {donor} {estimand}",
                )
                for estimand in ESTIMANDS
            }

        stored_equivalence = _mapping(group["nuisance_equivalence"], f"group {group_id} equivalence")
        if set(stored_equivalence) != set(ESTIMANDS):
            raise Level0PowerInputError(f"group {group_id} equivalence coverage mismatch")
        for estimand in ESTIMANDS:
            recomputed = coherent.equivalence_summary(
                [normalized_effects[donor][estimand] for donor in donor_ids],
                margin=FROZEN_MARGIN,
                alpha=0.05,
            )
            stored = _mapping(
                stored_equivalence[estimand],
                f"group {group_id} {estimand} equivalence",
            )
            if coherent.canonical_json(stored) != coherent.canonical_json(recomputed):
                raise Level0PowerInputError(f"group {group_id} {estimand} equivalence does not match donor effects")
            if recomputed["pass"] is not True:
                raise Level0PowerInputError(f"group {group_id} {estimand} equivalence did not pass")
        normalized_groups[group_id] = {**dict(group), "donor_effects": normalized_effects}

    if total_items != n_items or total_records != n_records:
        raise Level0PowerInputError("group totals do not match the Level-0 result totals")
    return {**dict(checked), "design": design, "groups": normalized_groups}


def build_level0_power_inputs(
    result: Mapping[str, Any], *, source_result_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive and validate the donor matrix and level0-only power config."""

    source_digest = _sha256(source_result_sha256, "source_result_sha256")
    checked = validate_passed_development_result(result)
    design = checked["design"]
    donor_ids = checked["donor_ids"]

    component_ids = sorted(
        _component_id(readout, family, estimand)
        for readout in design["required_readouts"]
        for family in design["required_input_families"]
        for estimand in ESTIMANDS
    )
    component_source: dict[str, tuple[str, str, str]] = {}
    for readout in design["required_readouts"]:
        for family in design["required_input_families"]:
            for estimand in ESTIMANDS:
                component_source[_component_id(readout, family, estimand)] = (
                    readout,
                    family,
                    estimand,
                )

    matrix = {
        "schema_version": phase0.MATRIX_SCHEMA,
        "mode": "development_only",
        "source_artifact_sha256": source_digest,
        "component_ids": component_ids,
        "donors": [
            {
                "donor_id": donor,
                "values": {
                    component_id: checked["groups"][f"{readout}::{family}"]["donor_effects"][donor][estimand]
                    for component_id, (readout, family, estimand) in sorted(component_source.items())
                },
            }
            for donor in donor_ids
        ],
    }
    matrix = phase0.validate_development_matrix(matrix)

    components = [
        {
            "component_id": component_id,
            "test": "equivalence",
            "margin": FROZEN_MARGIN,
            "boundary": None,
            "alternative_mean": 0.0,
            "support_lower": FROZEN_SUPPORT[0],
            "support_upper": FROZEN_SUPPORT[1],
            "require_recurrence": False,
        }
        for component_id in component_ids
    ]
    config = {
        "schema_version": phase0.CONFIG_SCHEMA,
        "mode": "development_only",
        "power_scope": "level0_only",
        "development_matrix_sha256": phase0.canonical_sha256(matrix),
        "seed": FROZEN_SEED,
        "simulation_replicates": FROZEN_REPLICATES,
        "candidate_n": list(phase0.CANDIDATE_N),
        "test_alpha": 0.05,
        "mc_family_alpha": 0.05,
        "power_target": 0.80,
        "minimum_development_donors": phase0.MINIMUM_DEVELOPMENT_DONORS,
        "covariance_method": "ledoit_wolf",
        "max_out_of_support_fraction": FROZEN_MAX_OUT_OF_SUPPORT_FRACTION,
        "components": components,
        "scenarios": [
            {
                "scenario_id": LEVEL0_ONLY_SCENARIO,
                "component_ids": component_ids,
                "required": True,
            }
        ],
    }
    config = phase0.validate_power_config(config, matrix)
    return matrix, config


def build_from_result_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Level0PowerInputError("Level-0 result file is not valid UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise Level0PowerInputError("Level-0 result file must contain one JSON object")
    return build_level0_power_inputs(result, source_result_sha256=hashlib.sha256(raw).hexdigest())


def artifact_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the single deterministic on-disk encoding used by the bridge."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_power_inputs(
    *,
    source_result: Path,
    output_matrix: Path,
    output_config: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = {
        source_result.resolve(),
        output_matrix.resolve(),
        output_config.resolve(),
    }
    if len(resolved) != 3:
        raise Level0PowerInputError("source, matrix, and config paths must be distinct")
    matrix, config = build_from_result_file(source_result)
    output_matrix.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_matrix.write_bytes(artifact_bytes(matrix))
    output_config.write_bytes(artifact_bytes(config))
    return matrix, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level0-result", type=Path, required=True)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    matrix, config = write_power_inputs(
        source_result=args.level0_result,
        output_matrix=args.output_matrix,
        output_config=args.output_config,
    )
    print(
        coherent.canonical_json(
            {
                "config_sha256": phase0.canonical_sha256(config),
                "matrix_sha256": phase0.canonical_sha256(matrix),
                "power_scope": "level0_only",
                "source_result_sha256": matrix["source_artifact_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()

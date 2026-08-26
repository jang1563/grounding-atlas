"""Development-only donor-level power simulation for the coherent readout design.

The simulator consumes an authenticated matrix of development-donor estimands.
It estimates a shrinkage covariance, generates one joint 20-donor simulation
cube, and reuses prefixes of that cube for every candidate confirmatory donor
count from 12 through 20.  Selection is based on the complete conjunction of
the registered exact donor gates, never on average component power.

This module cannot authorize confirmatory execution.  It only locks a proposed
donor count after the cohort, assay, and full preregistration gates are met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.stats import beta
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf

try:
    from eval.coherent_binary_readout import canonical_json, canonical_sha256
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from coherent_binary_readout import canonical_json, canonical_sha256


MATRIX_SCHEMA = "phase0-development-donor-matrix-v1"
CONFIG_SCHEMA = "phase0-donor-power-config-v1"
RESULT_SCHEMA = "phase0-donor-power-result-v1"

CANDIDATE_N = tuple(range(12, 21))
MINIMUM_DEVELOPMENT_DONORS = 8
MINIMUM_SIMULATION_REPLICATES = 10_000

MATRIX_KEYS = {
    "schema_version",
    "mode",
    "source_artifact_sha256",
    "component_ids",
    "donors",
}
DONOR_KEYS = {"donor_id", "values"}
CONFIG_KEYS = {
    "schema_version",
    "mode",
    "power_scope",
    "development_matrix_sha256",
    "seed",
    "simulation_replicates",
    "candidate_n",
    "test_alpha",
    "mc_family_alpha",
    "power_target",
    "minimum_development_donors",
    "covariance_method",
    "max_out_of_support_fraction",
    "components",
    "scenarios",
}
COMPONENT_KEYS = {
    "component_id",
    "test",
    "margin",
    "boundary",
    "alternative_mean",
    "support_lower",
    "support_upper",
    "require_recurrence",
}
SCENARIO_KEYS = {"scenario_id", "component_ids", "required"}


class Phase0PowerError(ValueError):
    """Raised when a Phase-0 input violates its frozen contract."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise Phase0PowerError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase0PowerError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Phase0PowerError(f"{label} must be an integer")
    return int(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise Phase0PowerError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise Phase0PowerError(f"{label} must be finite")
    return output


def _digest(value: Any, label: str) -> str:
    output = _string(value, label)
    if len(output) != 64:
        raise Phase0PowerError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(output, 16)
    except ValueError as error:
        raise Phase0PowerError(f"{label} is not hexadecimal") from error
    if output != output.lower():
        raise Phase0PowerError(f"{label} must use lowercase hexadecimal")
    return output


def _name_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise Phase0PowerError(f"{label} must be a nonempty list")
    output = [_string(item, f"{label} item") for item in value]
    if output != sorted(set(output)):
        raise Phase0PowerError(f"{label} must be unique and sorted")
    return output


def validate_development_matrix(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonically order the development-donor matrix."""

    _exact_keys(matrix, MATRIX_KEYS, "development matrix")
    if matrix["schema_version"] != MATRIX_SCHEMA:
        raise Phase0PowerError("unsupported development-matrix schema")
    if matrix["mode"] != "development_only":
        raise Phase0PowerError("development matrix mode must be development_only")
    component_ids = _name_list(matrix["component_ids"], "component_ids")
    source_digest = _digest(matrix["source_artifact_sha256"], "source_artifact_sha256")
    donors_value = matrix["donors"]
    if not isinstance(donors_value, list) or len(donors_value) < 2:
        raise Phase0PowerError("development matrix requires at least two donors")

    donors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, donor in enumerate(donors_value):
        if not isinstance(donor, Mapping):
            raise Phase0PowerError(f"donor {index} must be an object")
        _exact_keys(donor, DONOR_KEYS, f"donor {index}")
        donor_id = _string(donor["donor_id"], f"donor {index} donor_id")
        if donor_id in seen:
            raise Phase0PowerError(f"duplicate development donor: {donor_id}")
        seen.add(donor_id)
        values = donor["values"]
        if not isinstance(values, Mapping) or set(values) != set(component_ids):
            raise Phase0PowerError(f"donor {donor_id} component coverage does not match component_ids")
        donors.append(
            {
                "donor_id": donor_id,
                "values": {
                    component_id: _number(
                        values[component_id],
                        f"donor {donor_id} component {component_id}",
                    )
                    for component_id in component_ids
                },
            }
        )
    donors.sort(key=lambda item: item["donor_id"])
    return {
        "schema_version": MATRIX_SCHEMA,
        "mode": "development_only",
        "source_artifact_sha256": source_digest,
        "component_ids": component_ids,
        "donors": donors,
    }


def _validate_component(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, COMPONENT_KEYS, "power component")
    component_id = _string(value["component_id"], "component_id")
    test = value["test"]
    if test not in {"equivalence", "greater", "less"}:
        raise Phase0PowerError(f"component {component_id} has an unknown test")
    alternative = _number(value["alternative_mean"], "alternative_mean")
    support_lower = _number(value["support_lower"], "support_lower")
    support_upper = _number(value["support_upper"], "support_upper")
    if not support_lower < support_upper:
        raise Phase0PowerError(f"component {component_id} has invalid support")
    if not support_lower <= alternative <= support_upper:
        raise Phase0PowerError(f"component {component_id} alternative lies outside support")
    if not isinstance(value["require_recurrence"], bool):
        raise Phase0PowerError("require_recurrence must be boolean")

    if test == "equivalence":
        if value["boundary"] is not None:
            raise Phase0PowerError("equivalence component boundary must be null")
        margin = _number(value["margin"], "equivalence margin")
        if margin <= 0.0 or not support_lower < -margin < margin < support_upper:
            raise Phase0PowerError(f"component {component_id} has an invalid equivalence margin")
        if alternative != 0.0:
            raise Phase0PowerError("nuisance-equivalence alternative_mean must be exactly zero")
        if value["require_recurrence"]:
            raise Phase0PowerError("equivalence components cannot require directional recurrence")
        boundary: float | None = None
    else:
        if value["margin"] is not None:
            raise Phase0PowerError("directional component margin must be null")
        boundary = _number(value["boundary"], "directional boundary")
        registered = {
            "greater": {(0.20, 0.30), (0.10, 0.15), (0.05, 0.075)},
            "less": {(-0.20, -0.30), (-0.10, -0.15), (-0.05, -0.075)},
        }
        if not any(
            math.isclose(boundary, expected_boundary, abs_tol=1e-12)
            and math.isclose(alternative, expected_alternative, abs_tol=1e-12)
            for expected_boundary, expected_alternative in registered[test]
        ):
            raise Phase0PowerError(f"component {component_id} does not use a frozen design alternative")
        if not support_lower <= boundary <= support_upper:
            raise Phase0PowerError(f"component {component_id} boundary lies outside support")
        margin = None

    return {
        "component_id": component_id,
        "test": test,
        "margin": margin,
        "boundary": boundary,
        "alternative_mean": alternative,
        "support_lower": support_lower,
        "support_upper": support_upper,
        "require_recurrence": value["require_recurrence"],
    }


def validate_power_config(config: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a power configuration and its semantic matrix authentication."""

    locked_matrix = validate_development_matrix(matrix)
    _exact_keys(config, CONFIG_KEYS, "power config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise Phase0PowerError("unsupported power-config schema")
    if config["mode"] != "development_only":
        raise Phase0PowerError("power config mode must be development_only")
    power_scope = config["power_scope"]
    if power_scope != "level0_only":
        raise Phase0PowerError(
            "Phase-0 schema v1 supports only level0_only; a final n_conf requires "
            "a future config covering the complete primary claim hierarchy"
        )
    expected_matrix_digest = canonical_sha256(locked_matrix)
    if _digest(config["development_matrix_sha256"], "development_matrix_sha256") != expected_matrix_digest:
        raise Phase0PowerError("power config does not authenticate development matrix")

    seed = _integer(config["seed"], "seed")
    if not 0 <= seed < 2**64:
        raise Phase0PowerError("seed must lie in the unsigned 64-bit range")
    replicates = _integer(config["simulation_replicates"], "simulation_replicates")
    if not MINIMUM_SIMULATION_REPLICATES <= replicates <= 1_000_000:
        raise Phase0PowerError(f"simulation_replicates must be {MINIMUM_SIMULATION_REPLICATES}-1000000")
    if config["candidate_n"] != list(CANDIDATE_N):
        raise Phase0PowerError("candidate_n must be the complete ordered range 12-20")
    test_alpha = _number(config["test_alpha"], "test_alpha")
    mc_alpha = _number(config["mc_family_alpha"], "mc_family_alpha")
    power_target = _number(config["power_target"], "power_target")
    if not math.isclose(test_alpha, 0.05, abs_tol=1e-12):
        raise Phase0PowerError("test_alpha must remain frozen at 0.05")
    if not math.isclose(mc_alpha, 0.05, abs_tol=1e-12):
        raise Phase0PowerError("mc_family_alpha must remain frozen at 0.05")
    if not math.isclose(power_target, 0.80, abs_tol=1e-12):
        raise Phase0PowerError("power_target must remain frozen at 0.80")
    if config["minimum_development_donors"] != MINIMUM_DEVELOPMENT_DONORS:
        raise Phase0PowerError(f"minimum_development_donors must equal {MINIMUM_DEVELOPMENT_DONORS}")
    if config["covariance_method"] != "ledoit_wolf":
        raise Phase0PowerError("covariance_method must be ledoit_wolf")
    max_out = _number(config["max_out_of_support_fraction"], "max_out_of_support_fraction")
    if not 0.0 <= max_out <= 0.01:
        raise Phase0PowerError("max_out_of_support_fraction must lie in [0,0.01]")

    components_value = config["components"]
    if not isinstance(components_value, list) or not components_value:
        raise Phase0PowerError("components must be a nonempty list")
    components = [_validate_component(value) for value in components_value]
    component_ids = [value["component_id"] for value in components]
    if component_ids != sorted(set(component_ids)):
        raise Phase0PowerError("components must be unique and sorted by component_id")
    if component_ids != locked_matrix["component_ids"]:
        raise Phase0PowerError("power components must exactly match development-matrix components")
    component_by_id = {component["component_id"]: component for component in components}
    for donor in locked_matrix["donors"]:
        for component_id, observed in donor["values"].items():
            component = component_by_id[component_id]
            if not component["support_lower"] <= observed <= component["support_upper"]:
                raise Phase0PowerError(
                    f"development donor {donor['donor_id']} component {component_id} "
                    "lies outside the registered support"
                )

    scenarios_value = config["scenarios"]
    if not isinstance(scenarios_value, list) or not scenarios_value:
        raise Phase0PowerError("scenarios must be a nonempty list")
    scenarios: list[dict[str, Any]] = []
    for value in scenarios_value:
        if not isinstance(value, Mapping):
            raise Phase0PowerError("scenario must be an object")
        _exact_keys(value, SCENARIO_KEYS, "scenario")
        scenario_id = _string(value["scenario_id"], "scenario_id")
        scenario_components = _name_list(value["component_ids"], f"scenario {scenario_id} component_ids")
        if not set(scenario_components) <= set(component_ids):
            raise Phase0PowerError(f"scenario {scenario_id} contains an unknown component")
        if not isinstance(value["required"], bool):
            raise Phase0PowerError("scenario required must be boolean")
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "component_ids": scenario_components,
                "required": value["required"],
            }
        )
    scenario_ids = [value["scenario_id"] for value in scenarios]
    if scenario_ids != sorted(set(scenario_ids)):
        raise Phase0PowerError("scenarios must be unique and sorted by scenario_id")
    if not any(value["required"] for value in scenarios):
        raise Phase0PowerError("at least one power scenario must be required")
    covered = {component_id for scenario in scenarios for component_id in scenario["component_ids"]}
    if covered != set(component_ids):
        raise Phase0PowerError("every component must occur in at least one scenario")

    return {
        "schema_version": CONFIG_SCHEMA,
        "mode": "development_only",
        "power_scope": power_scope,
        "development_matrix_sha256": expected_matrix_digest,
        "seed": seed,
        "simulation_replicates": replicates,
        "candidate_n": list(CANDIDATE_N),
        "test_alpha": test_alpha,
        "mc_family_alpha": mc_alpha,
        "power_target": power_target,
        "minimum_development_donors": MINIMUM_DEVELOPMENT_DONORS,
        "covariance_method": "ledoit_wolf",
        "max_out_of_support_fraction": max_out,
        "components": components,
        "scenarios": scenarios,
    }


def _t_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def _signed_sums(values: np.ndarray) -> np.ndarray:
    output = np.zeros(1, dtype=float)
    for value in values:
        output = np.concatenate((output - value, output + value))
    return output


def exact_sign_flip_p_mitm(values: Sequence[float], *, direction: str, null: float = 0.0) -> float:
    """Exact Rademacher p-value using a meet-in-the-middle pair count.

    This has the same exhaustive assignment semantics as
    ``coherent_binary_readout.exact_sign_flip_p`` but avoids materializing all
    ``2**20`` complete sign vectors during a power simulation.
    """

    vector = np.asarray(values, dtype=float) - null
    if vector.ndim != 1 or not 2 <= len(vector) <= 20:
        raise Phase0PowerError("exact sign-flip requires 2-20 donor values")
    if not np.isfinite(vector).all():
        raise Phase0PowerError("exact sign-flip donor values must be finite")
    if direction not in {"greater", "less"}:
        raise Phase0PowerError(f"unknown exact sign-flip direction: {direction}")
    split = len(vector) // 2
    left = _signed_sums(vector[:split])
    right = np.sort(_signed_sums(vector[split:]))
    observed_sum = float(vector.sum())
    scale = max(1.0, float(np.sum(np.abs(vector))))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if direction == "greater":
        thresholds = observed_sum - tolerance - left
        locations = np.searchsorted(right, thresholds, side="left")
        extreme = int(np.sum(len(right) - locations))
    else:
        thresholds = observed_sum + tolerance - left
        locations = np.searchsorted(right, thresholds, side="right")
        extreme = int(np.sum(locations))
    return extreme / float(1 << len(vector))


def _equivalence_gate(values: np.ndarray, component: Mapping[str, Any], alpha: float) -> bool:
    margin = float(component["margin"])
    ci_lower, ci_upper = _t_interval(values, 0.90)
    if not (ci_lower > -margin and ci_upper < margin):
        return False
    lodo = (float(values.sum()) - values) / (len(values) - 1)
    if not bool(np.all(lodo > -margin) and np.all(lodo < margin)):
        return False
    lower_p = exact_sign_flip_p_mitm(values, direction="greater", null=-margin)
    if not lower_p < alpha:
        return False
    upper_p = exact_sign_flip_p_mitm(values, direction="less", null=margin)
    return bool(upper_p < alpha)


def _directional_gate(values: np.ndarray, component: Mapping[str, Any], alpha: float) -> bool:
    direction = str(component["test"])
    boundary = float(component["boundary"])
    ci_lower, ci_upper = _t_interval(values, 0.95)
    if direction == "greater":
        if not ci_lower > boundary:
            return False
        directional = values > boundary
    else:
        if not ci_upper < boundary:
            return False
        directional = values < boundary
    lodo = (float(values.sum()) - values) / (len(values) - 1)
    if direction == "greater" and not bool(np.all(lodo > boundary)):
        return False
    if direction == "less" and not bool(np.all(lodo < boundary)):
        return False
    if component["require_recurrence"]:
        recurrence = math.ceil(0.80 * len(values))
        if int(np.sum(directional)) < recurrence:
            return False
    exact_p = exact_sign_flip_p_mitm(values, direction=direction, null=boundary)
    return bool(exact_p < alpha)


def donor_gate_pass(values: Sequence[float], component: Mapping[str, Any], *, alpha: float = 0.05) -> bool:
    """Apply one complete registered donor gate to a simulated vector."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not 12 <= len(vector) <= 20:
        raise Phase0PowerError("power donor gate requires 12-20 donor values")
    if not np.isfinite(vector).all():
        raise Phase0PowerError("power donor gate values must be finite")
    if component["test"] == "equivalence":
        return _equivalence_gate(vector, component, alpha)
    return _directional_gate(vector, component, alpha)


def _pilot_matrix_array(matrix: Mapping[str, Any], component_ids: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[donor["values"][component_id] for component_id in component_ids] for donor in matrix["donors"]],
        dtype=float,
    )


def _pilot_diagnostics(values: np.ndarray, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, component in enumerate(components):
        vector = values[:, index]
        ci90 = _t_interval(vector, 0.90)
        margin_qualified: bool | None = None
        if component["test"] == "equivalence":
            margin = float(component["margin"])
            margin_qualified = bool(ci90[0] > -margin and ci90[1] < margin)
        output[str(component["component_id"])] = {
            "mean": float(vector.mean()),
            "standard_deviation": float(vector.std(ddof=1)),
            "ci90_lower": ci90[0],
            "ci90_upper": ci90[1],
            "equivalence_margin_qualified": margin_qualified,
        }
    return output


def _estimate_covariance(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(values)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    if covariance.shape != (values.shape[1], values.shape[1]):
        raise Phase0PowerError("covariance estimator returned the wrong shape")
    if not np.isfinite(covariance).all():
        raise Phase0PowerError("covariance estimator returned nonfinite values")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise Phase0PowerError("estimated covariance is materially non-PSD")
    effective_rank = int(np.sum(eigenvalues > 1e-12 * scale))
    return covariance, {
        "method": "ledoit_wolf",
        "shrinkage": float(estimator.shrinkage_),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "effective_rank": effective_rank,
        "dimension": values.shape[1],
    }


def _covariance_square_root(covariance: np.ndarray) -> np.ndarray:
    checked = np.asarray(covariance, dtype=float)
    if checked.ndim != 2 or checked.shape[0] != checked.shape[1]:
        raise Phase0PowerError("covariance must be square")
    if not np.isfinite(checked).all():
        raise Phase0PowerError("covariance must be finite")
    checked = 0.5 * (checked + checked.T)
    eigenvalues, eigenvectors = np.linalg.eigh(checked)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1e-10 * scale:
        raise Phase0PowerError("covariance is materially non-PSD")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvectors * np.sqrt(eigenvalues)


def _simulate_joint_cube(
    *,
    means: np.ndarray,
    covariance: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Generate B x 20 x p draws; candidate n always uses a shared prefix."""

    root = _covariance_square_root(covariance)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    standard = rng.standard_normal((replicates, max(CANDIDATE_N), len(means)))
    return means[None, None, :] + standard @ root.T


def _out_of_support(cube: np.ndarray, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    component_results: dict[str, Any] = {}
    total_out = 0
    total_values = cube.shape[0] * cube.shape[1] * cube.shape[2]
    for index, component in enumerate(components):
        values = cube[:, :, index]
        outside = (values < component["support_lower"]) | (values > component["support_upper"])
        count = int(np.sum(outside))
        total_out += count
        component_results[str(component["component_id"])] = {
            "outside_count": count,
            "total_count": int(values.size),
            "fraction": count / values.size,
        }
    return {
        "outside_count": total_out,
        "total_count": total_values,
        "fraction": total_out / total_values,
        "components": component_results,
    }


def _support_gate(support: Mapping[str, Any], maximum: float) -> bool:
    """Require both pooled and every component-specific tail rate to pass."""

    return bool(
        support["fraction"] <= maximum
        and all(component["fraction"] <= maximum for component in support["components"].values())
    )


def _clopper_pearson(passes: int, replicates: int, *, alpha: float) -> tuple[float, float]:
    if not 0 <= passes <= replicates or replicates <= 0:
        raise Phase0PowerError("invalid binomial count for Monte Carlo interval")
    if not 0.0 < alpha < 1.0:
        raise Phase0PowerError("Monte Carlo interval alpha must lie in (0,1)")
    lower = 0.0 if passes == 0 else float(beta.ppf(alpha, passes, replicates - passes + 1))
    upper = 1.0 if passes == replicates else float(beta.ppf(1.0 - alpha, passes + 1, replicates - passes))
    return lower, upper


def _power_summary(mask: np.ndarray, *, mc_alpha: float) -> dict[str, Any]:
    passes = int(np.sum(mask))
    replicates = int(len(mask))
    lower, upper = _clopper_pearson(passes, replicates, alpha=mc_alpha)
    return {
        "passes": passes,
        "replicates": replicates,
        "power": passes / replicates,
        "clopper_pearson_lower": lower,
        "clopper_pearson_upper": upper,
        "mc_one_sided_alpha": mc_alpha,
    }


def _component_passes(
    cube: np.ndarray,
    *,
    component_index: int,
    component: Mapping[str, Any],
    n_conf: int,
    alpha: float,
) -> np.ndarray:
    vectors = cube[:, :n_conf, component_index]
    unique, inverse = np.unique(vectors, axis=0, return_inverse=True)
    unique_passes = np.asarray(
        [donor_gate_pass(vector, component, alpha=alpha) for vector in unique],
        dtype=bool,
    )
    return unique_passes[inverse]


def _candidate_results(
    cube: np.ndarray,
    config: Mapping[str, Any],
    *,
    mc_alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    components = config["components"]
    component_index = {component["component_id"]: index for index, component in enumerate(components)}
    for n_conf in CANDIDATE_N:
        masks: dict[str, np.ndarray] = {}
        component_results: dict[str, Any] = {}
        for index, component in enumerate(components):
            component_id = str(component["component_id"])
            masks[component_id] = _component_passes(
                cube,
                component_index=index,
                component=component,
                n_conf=n_conf,
                alpha=float(config["test_alpha"]),
            )
            component_results[component_id] = _power_summary(masks[component_id], mc_alpha=mc_alpha)
        scenario_results: dict[str, Any] = {}
        for scenario in config["scenarios"]:
            scenario_mask = np.ones(cube.shape[0], dtype=bool)
            for component_id in scenario["component_ids"]:
                scenario_mask &= masks[component_id]
            summary = _power_summary(scenario_mask, mc_alpha=mc_alpha)
            summary["required"] = bool(scenario["required"])
            summary["component_ids"] = list(scenario["component_ids"])
            scenario_results[str(scenario["scenario_id"])] = summary
        output[str(n_conf)] = {
            "n_conf": n_conf,
            "r_conf": math.ceil(0.80 * n_conf),
            "components": component_results,
            "scenarios": scenario_results,
        }
        if set(component_index) != set(component_results):  # defensive invariant
            raise Phase0PowerError("component simulation coverage mismatch")
    return output


def _select_n_conf(candidates: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    target = float(config["power_target"])
    required = {str(scenario["scenario_id"]) for scenario in config["scenarios"] if scenario["required"]}
    for n_conf in CANDIDATE_N:
        scenarios = candidates[str(n_conf)]["scenarios"]
        if all(scenarios[scenario_id]["clopper_pearson_lower"] >= target for scenario_id in required):
            return n_conf
    return None


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _base_result(
    *,
    status: str,
    matrix: Mapping[str, Any],
    config: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "development_only": True,
        "power_scope": config["power_scope"],
        "candidate_n_selected": None,
        "selected_n_conf": None,
        "development_matrix_sha256": canonical_sha256(matrix),
        "power_config_sha256": canonical_sha256(config),
        "source_artifact_sha256": matrix["source_artifact_sha256"],
        "analysis_code_sha256": _code_sha256(),
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "rng": "numpy.PCG64DXSM",
        },
        "pilot": dict(pilot),
        "covariance": None,
        "out_of_support": None,
        "mc_alpha_per_bound": None,
        "candidates": {},
        "claim_boundary": (
            "Development-only simulation. A selected donor count does not authorize "
            "confirmatory model execution or establish biology, knowledge, readout, "
            "integration, gain, activation, or a physical law."
        ),
    }


def simulate_phase0_power(matrix: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the authenticated development-only Phase-0 power simulation."""

    locked_matrix = validate_development_matrix(matrix)
    locked_config = validate_power_config(config, locked_matrix)
    component_ids = locked_matrix["component_ids"]
    components = locked_config["components"]
    pilot_values = _pilot_matrix_array(locked_matrix, component_ids)
    pilot_components = _pilot_diagnostics(pilot_values, components)
    pilot = {
        "n_development_donors": len(locked_matrix["donors"]),
        "minimum_required_for_covariance": max(
            MINIMUM_DEVELOPMENT_DONORS, len(components) + 2
        ),
        "donor_ids": [donor["donor_id"] for donor in locked_matrix["donors"]],
        "components": pilot_components,
    }
    result = _base_result(status="PHASE0_PENDING", matrix=locked_matrix, config=locked_config, pilot=pilot)
    if len(locked_matrix["donors"]) < pilot["minimum_required_for_covariance"]:
        result["status"] = "PILOT_COVARIANCE_UNSTABLE"
        return result
    if any(diagnostic["equivalence_margin_qualified"] is False for diagnostic in pilot_components.values()):
        result["status"] = "MARGIN_NOT_QUALIFIED"
        return result

    covariance, covariance_diagnostics = _estimate_covariance(pilot_values)
    result["covariance"] = covariance_diagnostics
    means = np.asarray([component["alternative_mean"] for component in components], dtype=float)
    cube = _simulate_joint_cube(
        means=means,
        covariance=covariance,
        replicates=int(locked_config["simulation_replicates"]),
        seed=int(locked_config["seed"]),
    )
    support = _out_of_support(cube, components)
    support["maximum_allowed_fraction"] = locked_config["max_out_of_support_fraction"]
    support["pass"] = _support_gate(support, float(locked_config["max_out_of_support_fraction"]))
    result["out_of_support"] = support
    if not support["pass"]:
        result["status"] = "SIMULATION_MODEL_OUT_OF_SUPPORT"
        return result

    required_scenarios = sum(bool(scenario["required"]) for scenario in locked_config["scenarios"])
    mc_alpha = float(locked_config["mc_family_alpha"]) / (len(CANDIDATE_N) * required_scenarios)
    result["mc_alpha_per_bound"] = mc_alpha
    candidates = _candidate_results(cube, locked_config, mc_alpha=mc_alpha)
    result["candidates"] = candidates
    selected = _select_n_conf(candidates, locked_config)
    result["candidate_n_selected"] = selected
    if selected is None:
        result["status"] = "NO_GO_POWER_GT20"
    else:
        result["status"] = "LEVEL0_ONLY_POWER_SELECTED_DEVELOPMENT_ONLY"
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-0 donor-level power simulation",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Power scope: `{result['power_scope']}`.",
        "",
        f"Candidate donor count selected in this scope: `{result['candidate_n_selected']}`.",
        "",
        f"Final confirmatory donor count locked: `{result['selected_n_conf']}`.",
        "",
    ]
    if result["candidates"]:
        lines.extend(
            [
                "| n | recurrence | required-scenario minimum CP lower |",
                "|---:|---:|---:|",
            ]
        )
        for key, candidate in result["candidates"].items():
            required_lowers = [
                scenario["clopper_pearson_lower"]
                for scenario in candidate["scenarios"].values()
                if scenario["required"]
            ]
            lines.append(f"| {key} | {candidate['r_conf']} | {min(required_lowers):.4f} |")
        lines.append("")
    lines.extend(
        [
            f"Development matrix SHA-256: `{result['development_matrix_sha256']}`.",
            "",
            f"Power config SHA-256: `{result['power_config_sha256']}`.",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase0PowerError(f"{label} must contain one JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()

    matrix = _load_object(args.development_matrix, "development matrix")
    config = _load_object(args.config, "power config")
    result = simulate_phase0_power(matrix, config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(canonical_json({"output": str(args.output_json), "status": result["status"]}))


if __name__ == "__main__":
    main()

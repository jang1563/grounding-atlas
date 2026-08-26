from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from eval import coherent_binary_readout as coherent
from eval import phase0_power as power

SOURCE_SHA256 = "a" * 64


def _equivalence_component(component_id: str = "level0:lineage:full:O") -> dict:
    return {
        "component_id": component_id,
        "test": "equivalence",
        "margin": 0.06,
        "boundary": None,
        "alternative_mean": 0.0,
        "support_lower": -2.0,
        "support_upper": 2.0,
        "require_recurrence": False,
    }


def _directional_component(
    component_id: str = "biology:lineage",
    *,
    support_lower: float = -2.0,
    support_upper: float = 2.0,
) -> dict:
    return {
        "component_id": component_id,
        "test": "greater",
        "margin": None,
        "boundary": 0.20,
        "alternative_mean": 0.30,
        "support_lower": support_lower,
        "support_upper": support_upper,
        "require_recurrence": True,
    }


def _matrix(
    components: list[dict],
    *,
    n_donors: int = 8,
    values: dict[str, list[float]] | None = None,
) -> dict:
    component_ids = [component["component_id"] for component in components]
    if component_ids != sorted(component_ids):
        raise AssertionError("test components must be sorted")
    values = values or {component_id: [0.0] * n_donors for component_id in component_ids}
    return {
        "schema_version": power.MATRIX_SCHEMA,
        "mode": "development_only",
        "source_artifact_sha256": SOURCE_SHA256,
        "component_ids": component_ids,
        "donors": [
            {
                "donor_id": f"d{index:02d}",
                "values": {component_id: values[component_id][index] for component_id in component_ids},
            }
            for index in range(n_donors)
        ],
    }


def _config(
    matrix: dict,
    components: list[dict],
    *,
    scenarios: list[dict] | None = None,
    seed: int = 90210,
    max_out_of_support_fraction: float = 0.001,
) -> dict:
    component_ids = [component["component_id"] for component in components]
    scenarios = scenarios or [
        {
            "scenario_id": "primary",
            "component_ids": component_ids,
            "required": True,
        }
    ]
    return {
        "schema_version": power.CONFIG_SCHEMA,
        "mode": "development_only",
        "power_scope": "level0_only",
        "development_matrix_sha256": power.canonical_sha256(power.validate_development_matrix(matrix)),
        "seed": seed,
        "simulation_replicates": power.MINIMUM_SIMULATION_REPLICATES,
        "candidate_n": list(power.CANDIDATE_N),
        "test_alpha": 0.05,
        "mc_family_alpha": 0.05,
        "power_target": 0.80,
        "minimum_development_donors": power.MINIMUM_DEVELOPMENT_DONORS,
        "covariance_method": "ledoit_wolf",
        "max_out_of_support_fraction": max_out_of_support_fraction,
        "components": components,
        "scenarios": scenarios,
    }


def test_development_matrix_is_authenticated_and_canonically_ordered() -> None:
    components = [_equivalence_component()]
    matrix = _matrix(components)
    matrix["donors"] = list(reversed(matrix["donors"]))
    locked = power.validate_development_matrix(matrix)
    assert [donor["donor_id"] for donor in locked["donors"]] == sorted(donor["donor_id"] for donor in matrix["donors"])
    config = _config(matrix, components)
    assert power.validate_power_config(config, matrix)["development_matrix_sha256"] == power.canonical_sha256(locked)

    config["development_matrix_sha256"] = "b" * 64
    with pytest.raises(power.Phase0PowerError, match="does not authenticate"):
        power.validate_power_config(config, matrix)


def test_input_schemas_fail_closed_on_duplicates_extras_and_bad_components() -> None:
    components = [_equivalence_component()]
    matrix = _matrix(components)
    matrix["unexpected"] = True
    with pytest.raises(power.Phase0PowerError, match="schema mismatch"):
        power.validate_development_matrix(matrix)

    matrix = _matrix(components)
    matrix["donors"][1]["donor_id"] = matrix["donors"][0]["donor_id"]
    with pytest.raises(power.Phase0PowerError, match="duplicate development donor"):
        power.validate_development_matrix(matrix)

    matrix = _matrix(components)
    config = _config(matrix, components)
    config["candidate_n"] = list(range(12, 20))
    with pytest.raises(power.Phase0PowerError, match="complete ordered range"):
        power.validate_power_config(config, matrix)

    bad_component = _directional_component()
    bad_component["alternative_mean"] = 0.20
    bad_matrix = _matrix([bad_component])
    bad_config = _config(bad_matrix, [bad_component])
    with pytest.raises(power.Phase0PowerError, match="frozen design alternative"):
        power.validate_power_config(bad_config, bad_matrix)

    bounded_component = _directional_component(support_lower=0.20, support_upper=0.40)
    bounded_matrix = _matrix(
        [bounded_component],
        values={bounded_component["component_id"]: [0.30] * 7 + [0.41]},
    )
    bounded_config = _config(bounded_matrix, [bounded_component])
    with pytest.raises(power.Phase0PowerError, match="registered support"):
        power.validate_power_config(bounded_config, bounded_matrix)


@pytest.mark.parametrize("n_donors", [8, 12])
@pytest.mark.parametrize("direction", ["greater", "less"])
def test_meet_in_middle_exact_p_matches_shared_exhaustive_helper(n_donors: int, direction: str) -> None:
    rng = np.random.default_rng(20260802 + n_donors)
    vector = rng.normal(0.01, 0.08, n_donors)
    observed = power.exact_sign_flip_p_mitm(vector, direction=direction, null=0.06)
    expected = coherent.exact_sign_flip_p(vector, direction=direction, null=0.06)
    assert observed == pytest.approx(expected, abs=0.0)


def test_exact_equivalence_and_directional_boundaries_are_strict() -> None:
    equivalence = _equivalence_component()
    assert power.donor_gate_pass(np.zeros(12), equivalence)
    assert power.exact_sign_flip_p_mitm(np.zeros(12), direction="greater", null=-0.06) == pytest.approx(1 / 4096)
    assert not power.donor_gate_pass(np.full(12, 0.06), equivalence)

    directional = _directional_component()
    assert power.donor_gate_pass(np.full(12, 0.30), directional)
    assert not power.donor_gate_pass(np.full(12, 0.20), directional)

    recurrence_failure = np.asarray([0.30] * 9 + [0.20] * 3)
    assert not power.donor_gate_pass(recurrence_failure, directional)


def test_recurrence_count_is_frozen_for_every_candidate_n() -> None:
    assert {n_conf: int(np.ceil(0.80 * n_conf)) for n_conf in power.CANDIDATE_N} == {
        12: 10,
        13: 11,
        14: 12,
        15: 12,
        16: 13,
        17: 14,
        18: 15,
        19: 16,
        20: 16,
    }


def test_joint_cube_is_deterministic_and_candidate_counts_share_prefixes() -> None:
    means = np.asarray([0.0, 0.3])
    covariance = np.asarray([[0.01, 0.004], [0.004, 0.02]])
    first = power._simulate_joint_cube(means=means, covariance=covariance, replicates=7, seed=42)
    second = power._simulate_joint_cube(means=means, covariance=covariance, replicates=7, seed=42)
    assert first.shape == (7, 20, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, :12, :], second[:, :20, :][:, :12, :])


def test_covariance_square_root_accepts_singular_psd_and_rejects_non_psd() -> None:
    singular = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    root = power._covariance_square_root(singular)
    assert root @ root.T == pytest.approx(singular)
    with pytest.raises(power.Phase0PowerError, match="non-PSD"):
        power._covariance_square_root(np.asarray([[1.0, 2.0], [2.0, 1.0]]))


def test_clopper_pearson_is_one_sided_and_bonferroni_selection_is_conservative() -> None:
    lower, upper = power._clopper_pearson(8000, 10_000, alpha=0.05 / 9)
    assert 0.78 < lower < 0.80
    assert 0.80 < upper < 0.82

    config = {
        "power_target": 0.80,
        "scenarios": [
            {"scenario_id": "a", "required": True},
            {"scenario_id": "b", "required": True},
        ],
    }
    candidates = {}
    for n_conf in power.CANDIDATE_N:
        lower_a = 0.81 if n_conf >= 14 else 0.79
        lower_b = 0.81 if n_conf >= 15 else 0.79
        candidates[str(n_conf)] = {
            "scenarios": {
                "a": {"clopper_pearson_lower": lower_a},
                "b": {"clopper_pearson_lower": lower_b},
            }
        }
    assert power._select_n_conf(candidates, config) == 15


def test_candidate_power_uses_full_conjunction_not_average_component_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = [
        _equivalence_component("a"),
        _equivalence_component("b"),
    ]
    config = {
        "components": components,
        "scenarios": [
            {
                "scenario_id": "joint",
                "component_ids": ["a", "b"],
                "required": True,
            }
        ],
        "test_alpha": 0.05,
    }
    cube = np.zeros((4, 20, 2))

    def fake_component_passes(_cube: np.ndarray, *, component_index: int, **_kwargs: object) -> np.ndarray:
        return (
            np.asarray([True, True, False, False]) if component_index == 0 else np.asarray([False, False, True, True])
        )

    monkeypatch.setattr(power, "_component_passes", fake_component_passes)
    result = power._candidate_results(cube, config, mc_alpha=0.05 / 9)
    candidate = result["12"]
    assert candidate["components"]["a"]["power"] == 0.5
    assert candidate["components"]["b"]["power"] == 0.5
    assert candidate["scenarios"]["joint"]["power"] == 0.0


def test_too_few_development_donors_stops_before_covariance_simulation() -> None:
    components = [_equivalence_component()]
    matrix = _matrix(components, n_donors=7)
    config = _config(matrix, components)
    result = power.simulate_phase0_power(matrix, config)
    assert result["status"] == "PILOT_COVARIANCE_UNSTABLE"
    assert result["selected_n_conf"] is None
    assert result["covariance"] is None
    assert result["candidates"] == {}


def test_covariance_dimension_requires_two_more_pilots_than_components() -> None:
    components = [
        _equivalence_component(f"component-{index:02d}") for index in range(7)
    ]
    matrix = _matrix(components, n_donors=8)
    config = _config(matrix, components)
    result = power.simulate_phase0_power(matrix, config)
    assert result["status"] == "PILOT_COVARIANCE_UNSTABLE"
    assert result["pilot"]["minimum_required_for_covariance"] == 9


def test_unqualified_development_margin_stops_fail_closed() -> None:
    components = [_equivalence_component()]
    component_id = components[0]["component_id"]
    matrix = _matrix(
        components,
        values={component_id: [0.06] * 8},
    )
    config = _config(matrix, components)
    result = power.simulate_phase0_power(matrix, config)
    assert result["status"] == "MARGIN_NOT_QUALIFIED"
    assert result["pilot"]["components"][component_id]["equivalence_margin_qualified"] is False
    assert result["selected_n_conf"] is None


def test_zero_covariance_full_simulation_selects_twelve_and_is_byte_stable() -> None:
    components = [_equivalence_component()]
    matrix = _matrix(components)
    config = _config(matrix, components)
    first = power.simulate_phase0_power(matrix, config)
    second = power.simulate_phase0_power(matrix, config)
    assert first == second
    assert power.canonical_json(first) == power.canonical_json(second)
    assert first["status"] == "LEVEL0_ONLY_POWER_SELECTED_DEVELOPMENT_ONLY"
    assert first["candidate_n_selected"] == 12
    assert first["selected_n_conf"] is None
    assert first["candidates"]["12"]["r_conf"] == 10
    assert first["candidates"]["12"]["scenarios"]["primary"]["clopper_pearson_lower"] >= 0.80
    assert first["mc_alpha_per_bound"] == pytest.approx(0.05 / 9)
    assert first["out_of_support"]["fraction"] == 0.0

    invalid_full_config = _config(matrix, components)
    invalid_full_config["power_scope"] = "full_claim_hierarchy"
    with pytest.raises(power.Phase0PowerError, match="complete primary claim hierarchy"):
        power.validate_power_config(invalid_full_config, matrix)


def test_material_out_of_support_rate_invalidates_simulation_before_power() -> None:
    components = [_directional_component(support_lower=0.20, support_upper=0.40)]
    component_id = components[0]["component_id"]
    matrix = _matrix(
        components,
        values={component_id: [0.20, 0.40] * 4},
    )
    config = _config(matrix, components, max_out_of_support_fraction=0.001)
    result = power.simulate_phase0_power(matrix, config)
    assert result["status"] == "SIMULATION_MODEL_OUT_OF_SUPPORT"
    assert not result["out_of_support"]["pass"]
    assert result["out_of_support"]["fraction"] > 0.001
    assert result["candidates"] == {}
    assert result["selected_n_conf"] is None

    diluted = {
        "fraction": 0.0005,
        "components": {
            "bad": {"fraction": 0.002},
            "good": {"fraction": 0.0},
        },
    }
    assert not power._support_gate(diluted, 0.001)


def test_low_joint_power_returns_no_go_without_extrapolating_past_twenty() -> None:
    components = [_directional_component(support_lower=-10.0, support_upper=10.0)]
    component_id = components[0]["component_id"]
    matrix = _matrix(
        components,
        values={component_id: [-0.20, 0.80] * 4},
    )
    config = _config(matrix, components)
    result = power.simulate_phase0_power(matrix, config)
    assert result["status"] == "NO_GO_POWER_GT20"
    assert result["selected_n_conf"] is None
    assert set(result["candidates"]) == {str(value) for value in range(12, 21)}
    assert result["candidates"]["20"]["scenarios"]["primary"]["clopper_pearson_lower"] < 0.80


def test_cli_writes_deterministic_json_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    components = [_equivalence_component()]
    matrix = _matrix(components)
    config = _config(matrix, components)
    matrix_path = tmp_path / "matrix.json"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase0_power.py",
            "--development-matrix",
            str(matrix_path),
            "--config",
            str(config_path),
            "--output-json",
            str(output_path),
            "--output-markdown",
            str(markdown_path),
        ],
    )
    power.main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "LEVEL0_ONLY_POWER_SELECTED_DEVELOPMENT_ONLY"
    assert result["candidate_n_selected"] == 12
    assert result["selected_n_conf"] is None
    assert "Development-only simulation" in markdown_path.read_text(encoding="utf-8")

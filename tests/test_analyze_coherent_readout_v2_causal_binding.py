from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from eval import (
    analyze_coherent_readout_v2_causal_binding as analyzer,
)
from eval import run_coherent_readout_v2_causal_binding as runner

TRACE = {
    "hook_calls": 1,
    "non_target_tokens_unchanged": True,
    "patched_token_matches_source": True,
}


def _diagnostics(
    margin: float,
    *,
    mass: float = 0.99,
    offset: float = 0.0,
    greedy_token_id: int | None = None,
) -> dict[str, Any]:
    x_logit = float(offset + margin)
    y_logit = float(offset)
    logsumexp = float(np.logaddexp(x_logit, y_logit) - math.log(mass))
    if greedy_token_id is None:
        greedy_token_id = (
            runner.X_TOKEN_ID if margin > 0.0 else runner.Y_TOKEN_ID
        )
    greedy_logit = x_logit if greedy_token_id == runner.X_TOKEN_ID else y_logit
    return {
        "x_logit": x_logit,
        "y_logit": y_logit,
        "x_minus_y_margin": float(margin),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": float(mass),
        "greedy_token_id": greedy_token_id,
        "greedy_logit": greedy_logit,
        "maximum_token_ids": [greedy_token_id],
        "maximum_tie_count": 1,
        "full_vocab_logits_sha256": "ab" * 32,
    }


def _prompt(
    item_id: str,
    pair_id: str,
    role: str,
    *,
    bank_role: str = "discovery",
) -> dict[str, Any]:
    return {
        "bank_role": bank_role,
        "prompt_id": f"prompt-{item_id}-{role}",
        "prompt_role": role,
        "item_id": item_id,
        "pair_id": pair_id,
        "truth_polarity": "positive" if item_id.endswith("a") else "negative",
        "order": "positive_first" if role == "D" else "negative_first",
        "mapping": "positive_is_x",
        "execution_input_sha256": "cd" * 32,
        "input_token_count": 12,
    }


def _baseline_record(
    prompt: Mapping[str, Any],
    *,
    phase: str,
    activation_row: int,
    activations: np.ndarray,
    diagnostics: Mapping[str, Any],
    duplicate: bool,
) -> dict[str, Any]:
    layer_hashes = [
        runner.f32_sha256(activations[activation_row, layer])
        for layer in range(runner.MODEL_LAYERS)
    ]
    return runner._baseline_record(
        prompt,
        phase=phase,
        activation_row=activation_row,
        measurement={
            "diagnostics": dict(diagnostics),
            "duplicate_diagnostics": dict(diagnostics) if duplicate else None,
            "activation_layer_sha256": layer_hashes,
        },
    )


def _metric_rows(
    *, n_pairs: int = 4, layer: int = 4
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baselines: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    condition_margins = {
        "forward_paired": 0.0,
        "forward_same_pair_x": -1.0,
        "forward_unrelated_x": -1.0,
        "reverse_paired": 0.0,
        "reverse_same_pair": 1.0,
        "reverse_unrelated": 1.0,
        "identity_r": -2.0,
        "identity_d": 2.0,
    }
    for pair_index in range(n_pairs):
        pair_id = f"pair-{pair_index:02d}"
        for polarity in ("a", "b"):
            item_id = f"item-{pair_index:02d}-{polarity}"
            baselines.extend(
                [
                    {
                        "item_id": item_id,
                        "pair_id": pair_id,
                        "prompt_role": "D",
                        "diagnostics": _diagnostics(2.0),
                    },
                    {
                        "item_id": item_id,
                        "pair_id": pair_id,
                        "prompt_role": "R",
                        "diagnostics": _diagnostics(-2.0),
                    },
                ]
            )
            for condition, margin in condition_margins.items():
                dependency_cluster_id = (
                    f"dyad-{pair_index // 2:02d}"
                    if condition in {"forward_unrelated_x", "reverse_unrelated"}
                    else pair_id
                )
                patches.append(
                    {
                        "item_id": item_id,
                        "pair_id": pair_id,
                        "condition": condition,
                        "dependency_cluster_id": dependency_cluster_id,
                        "source_target_token_count_difference": 0,
                        "layer": layer,
                        "diagnostics": _diagnostics(
                            margin if margin != 0.0 else 1e-9
                        ),
                        "hook_trace": dict(TRACE),
                    }
                )
                patches[-1]["diagnostics"]["x_minus_y_margin"] = margin
                patches[-1]["diagnostics"]["x_logit"] = margin
    return baselines, patches


def _holdout_baselines(
    *,
    d_correct_count: int = 38,
    r_incorrect_count: int = 30,
    mass: float = 0.95,
    gap: float = 1.0,
) -> list[dict[str, Any]]:
    if d_correct_count < 0 or r_incorrect_count < 0:
        raise ValueError("counts must be nonnegative")
    item_count = 40
    d_wrong = set(range(d_correct_count, item_count))
    r_wrong = set(range(r_incorrect_count - len(d_wrong))) | d_wrong
    assert len(r_wrong) == r_incorrect_count
    rows: list[dict[str, Any]] = []
    for index in range(item_count):
        pair_id = f"pair-{index // 2:02d}"
        item_id = f"item-{index:02d}"
        d_is_correct = index not in d_wrong
        r_is_incorrect = index in r_wrong
        if d_is_correct and r_is_incorrect:
            d_margin, r_margin = gap / 2.0, -gap / 2.0
        elif d_is_correct and not r_is_incorrect:
            d_margin, r_margin = 1.5 * gap, 0.5 * gap
        elif not d_is_correct and r_is_incorrect:
            d_margin, r_margin = -0.5 * gap, -1.5 * gap
        else:
            raise AssertionError("positive aggregate gap forbids this count overlap")
        rows.extend(
            [
                {
                    "item_id": item_id,
                    "pair_id": pair_id,
                    "prompt_role": "D",
                    "diagnostics": _diagnostics(d_margin, mass=mass),
                },
                {
                    "item_id": item_id,
                    "pair_id": pair_id,
                    "prompt_role": "R",
                    "diagnostics": _diagnostics(r_margin, mass=mass),
                },
            ]
        )
    return rows


def _decision_metric(
    *,
    ratio: float,
    positive_pairs: int,
    lodo_ratio: float,
    lodo_mean: float,
) -> dict[str, Any]:
    return {
        "summary_defined": True,
        "n_inference_clusters": 8,
        "gap_denominator_positive": True,
        "all_lodo_gap_denominators_positive": True,
        "recovery_fraction_defined": True,
        "mean_over_gap": ratio,
        "positive_pair_count": positive_pairs,
        "lodo_over_gap": [lodo_ratio] * 8,
        "leave_one_pair_out_means": [lodo_mean] * 8,
    }


def test_validate_diagnostics_accepts_a_self_consistent_exact_schema() -> None:
    diagnostics = _diagnostics(1.0, mass=0.75)

    assert analyzer._validate_diagnostics(diagnostics) == diagnostics


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.update(extra=1),
        lambda row: row.update(x_minus_y_margin=2.0),
        lambda row: row.update(label_probability_mass=1.01),
        lambda row: row.update(maximum_tie_count=2),
        lambda row: row.update(maximum_token_ids=[]),
        lambda row: row.update(full_vocab_logsumexp=float("nan")),
        lambda row: row.update(greedy_logit=float("inf")),
        lambda row: row.update(label_probability_mass=0.5),
        lambda row: row.update(full_vocab_logits_sha256="G" * 64),
    ],
)
def test_validate_diagnostics_rejects_schema_and_numeric_tampering(mutator: Any) -> None:
    diagnostics = _diagnostics(1.0, mass=0.75)
    mutator(diagnostics)

    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_diagnostics(diagnostics)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_tie_count", 1.0),
        ("maximum_tie_count", True),
        ("greedy_token_id", True),
    ],
)
def test_validate_diagnostics_rejects_non_integer_global_maximum_fields(
    field: str, value: Any
) -> None:
    diagnostics = _diagnostics(1.0)
    diagnostics[field] = value

    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_diagnostics(diagnostics)


def test_validate_diagnostics_binds_native_greedy_logit_to_label_logit() -> None:
    diagnostics = _diagnostics(1.0, mass=0.5)
    diagnostics["greedy_logit"] = diagnostics["x_logit"] + 0.25

    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_diagnostics(diagnostics)


def test_pair_means_aggregate_two_truth_items_and_reject_incomplete_pairs() -> None:
    mapping = {"a+": "pair-a", "a-": "pair-a", "b+": "pair-b", "b-": "pair-b"}
    values = {"a+": 1.0, "a-": 3.0, "b+": -2.0, "b-": 2.0}

    assert analyzer._pair_means(values, mapping) == {"pair-a": 2.0, "pair-b": 0.0}
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._pair_means({"a+": 1.0}, mapping)


def test_cluster_bootstrap_is_deterministic_for_fixed_seed() -> None:
    values = {"p1": -1.0, "p2": 0.5, "p3": 2.0, "p4": 3.0}

    first = analyzer._bootstrap_summary(values, draws=257, seed=260802)
    second = analyzer._bootstrap_summary(values, draws=257, seed=260802)

    assert first == second
    assert first["draws"] == 257
    assert first["seed"] == 260802
    assert first["lower_95"] <= first["upper_95"]


def test_secondary_flip_bootstrap_allows_one_or_two_items_per_pair() -> None:
    result = analyzer._flip_rate_summary(
        {"a": 1.0, "b": 0.0, "c": 1.0},
        {"a": "pair-1", "b": "pair-1", "c": "pair-2"},
        seed=7,
    )

    assert result["pair_values"] == {"pair-1": 0.5, "pair-2": 1.0}
    assert result["point"] == pytest.approx(0.75)


def test_discovery_decision_uses_transfer_boundaries_and_exact_length_specificity() -> None:
    transfer = _decision_metric(
        ratio=0.30, positive_pairs=7, lodo_ratio=0.20, lodo_mean=0.01
    )
    specificity = _decision_metric(
        ratio=0.20, positive_pairs=7, lodo_ratio=0.0, lodo_mean=1e-9
    )
    metrics = {
        "metrics": {
            "F": transfer,
            "N": copy.deepcopy(transfer),
            "S_F_same_exact_length": specificity,
            "S_F_unrelated_exact_length": copy.deepcopy(specificity),
            "S_N_same_exact_length": copy.deepcopy(specificity),
            "S_N_unrelated_exact_length": copy.deepcopy(specificity),
        }
    }

    assert analyzer._discovery_decision(metrics) == {
        "transfer_pass": True,
        "specificity_diagnostic_pass": True,
        "specificity_used_for_layer_selection": False,
    }
    metrics["metrics"]["S_N_unrelated_exact_length"][
        "leave_one_pair_out_means"
    ][0] = 0.0
    assert analyzer._discovery_decision(metrics) == {
        "transfer_pass": True,
        "specificity_diagnostic_pass": False,
        "specificity_used_for_layer_selection": False,
    }
    metrics["metrics"]["F"]["mean_over_gap"] = 0.30 - 1e-9
    assert analyzer._discovery_decision(metrics) == {
        "transfer_pass": False,
        "specificity_diagnostic_pass": False,
        "specificity_used_for_layer_selection": False,
    }


def test_nonpositive_exact_subset_gap_makes_specificity_ratio_undefined_without_crash() -> None:
    transfer_pairs = {f"pair-{index}": 0.4 for index in range(8)}
    transfer_gaps = {f"pair-{index}": 1.0 for index in range(8)}
    transfer = analyzer._effect_summary(
        transfer_pairs, gap_pair_values=transfer_gaps
    )
    positive_specificity = analyzer._effect_summary(
        {f"unit-{index}": 0.3 for index in range(4)},
        gap_pair_values={f"unit-{index}": 1.0 for index in range(4)},
    )
    undefined_specificity = analyzer._effect_summary(
        {"unit-0": 0.3, "unit-1": 0.3},
        gap_pair_values={"unit-0": -1.0, "unit-1": -1.0},
    )
    metrics = {
        "metrics": {
            "F": transfer,
            "N": copy.deepcopy(transfer),
            "S_F_same_exact_length": undefined_specificity,
            "S_F_unrelated_exact_length": positive_specificity,
            "S_N_same_exact_length": copy.deepcopy(positive_specificity),
            "S_N_unrelated_exact_length": copy.deepcopy(positive_specificity),
        }
    }

    assert undefined_specificity["summary_defined"] is True
    assert undefined_specificity["gap_denominator_positive"] is False
    assert undefined_specificity["recovery_fraction_defined"] is False
    assert undefined_specificity["mean_over_gap"] is None
    assert analyzer._discovery_decision(metrics) == {
        "transfer_pass": True,
        "specificity_diagnostic_pass": False,
        "specificity_used_for_layer_selection": False,
    }


@pytest.mark.parametrize(
    ("decisions", "expected_layer", "expected_status"),
    [
        (
            {
                0: {"transfer_pass": True, "specificity_diagnostic_pass": False},
                4: {"transfer_pass": False, "specificity_diagnostic_pass": False},
                8: {"transfer_pass": True, "specificity_diagnostic_pass": True},
                27: {"transfer_pass": True, "specificity_diagnostic_pass": True},
            },
            0,
            "LOCALIZATION_COMPLETE_HOLDOUT_BASELINE_AUTHORIZED",
        ),
        (
            {
                0: {"transfer_pass": False, "specificity_diagnostic_pass": True},
                4: {"transfer_pass": True, "specificity_diagnostic_pass": False},
                8: {"transfer_pass": True, "specificity_diagnostic_pass": True},
                27: {"transfer_pass": True, "specificity_diagnostic_pass": True},
            },
            4,
            "LOCALIZATION_COMPLETE_HOLDOUT_BASELINE_AUTHORIZED",
        ),
        (
            {
                0: {"transfer_pass": False, "specificity_diagnostic_pass": False},
                4: {"transfer_pass": False, "specificity_diagnostic_pass": False},
                8: {"transfer_pass": False, "specificity_diagnostic_pass": False},
                27: {"transfer_pass": True, "specificity_diagnostic_pass": True},
            },
            None,
            "LOCALIZATION_STOP_ONLY_FINAL_COPY_CONTROL",
        ),
    ],
)
def test_discovery_layer_selection_uses_earliest_nonfinal_transfer_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decisions: Mapping[int, Mapping[str, bool]],
    expected_layer: int | None,
    expected_status: str,
) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    for name in (
        "discovery_baselines.jsonl",
        "discovery_activations.npy",
        "discovery_patches.jsonl",
    ):
        (result_root / name).write_bytes(name.encode())
    design_path = tmp_path / "design.json"
    design_path.write_text("{}\n", encoding="utf-8")
    layer_lock_path = tmp_path / "layer_lock.json"
    layer_grid = tuple(decisions)
    monkeypatch.setattr(runner, "RESULT_ROOT", result_root)
    monkeypatch.setattr(runner, "DEFAULT_DESIGN", design_path)
    monkeypatch.setattr(runner, "DEFAULT_LAYER_LOCK", layer_lock_path)
    monkeypatch.setattr(runner, "LAYER_GRID", layer_grid)
    monkeypatch.setattr(
        analyzer,
        "_load_plan",
        lambda: (
            {
                "call_plan_sha256": "plan-sha",
                "locks": {"analyzer_sha256": "analyzer-sha"},
            },
            {
                "selection_rule": {
                    "selection_basis": "transfer_only",
                    "tie_break": "earliest_nonfinal_grid_layer",
                }
            },
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_execution_manifest",
        lambda *args, **kwargs: ({}, np.zeros((1, 1, 1), dtype="<f4")),
    )
    monkeypatch.setattr(runner, "_phase_prompts", lambda *args: [])
    monkeypatch.setattr(runner, "_phase_templates", lambda *args: [])
    monkeypatch.setattr(runner, "load_jsonl", lambda *args: [])
    monkeypatch.setattr(analyzer, "_validate_baselines", lambda *args, **kwargs: [])
    monkeypatch.setattr(analyzer, "_validate_patches", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        analyzer,
        "_engineering_gates",
        lambda _baselines, _patches, _prompts: {
            "duplicate_pass": True,
            "prior_xy_pass": True,
            "prior_full_vocab_row_sha256_pass": True,
            "identity_pass": True,
            "final_block_source_copy_pass": True,
            "hook_trace_pass": True,
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_metrics_for_layer",
        lambda *args, layer, **kwargs: {"layer": layer, "decision": decisions[layer]},
    )
    monkeypatch.setattr(analyzer, "_discovery_decision", lambda row: row["decision"])

    analysis, layer_lock = analyzer.analyze_discovery()

    assert analysis["selected_layer"] == expected_layer
    assert analysis["status"] == expected_status
    assert analysis["selection_class"] == (
        "transfer_only_earliest_nonfinal_grid_layer"
        if expected_layer is not None
        else None
    )
    assert layer_lock["selected_layer"] == expected_layer
    assert layer_lock["holdout_baseline_authorized"] is (
        expected_layer is not None
    )


def test_holdout_gate_passes_exact_inclusive_rate_mass_and_gap_boundaries() -> None:
    result, cohort = analyzer._holdout_baseline_metrics(
        _holdout_baselines(), selected_layer=4
    )

    assert result["metrics"]["d_correct_rate"] == pytest.approx(0.95)
    assert result["metrics"]["r_incorrect_rate"] == pytest.approx(0.75)
    assert result["metrics"]["gap_mean"] == pytest.approx(1.0)
    assert result["gate"]["all_pass"] is True
    assert len(cohort["item_ids"]) == 28


@pytest.mark.parametrize(
    ("kwargs", "failed_gate"),
    [
        ({"d_correct_count": 37}, "d_correct_rate_pass"),
        ({"r_incorrect_count": 29}, "r_incorrect_rate_pass"),
        ({"mass": 0.95 - 1e-8}, "d_label_mass_pass"),
        ({"gap": 1.0 - 1e-8}, "gap_point_pass"),
    ],
)
def test_holdout_gate_rejects_values_immediately_below_frozen_floors(
    kwargs: Mapping[str, Any], failed_gate: str
) -> None:
    result, _ = analyzer._holdout_baseline_metrics(
        _holdout_baselines(**kwargs), selected_layer=4
    )

    assert result["gate"][failed_gate] is False
    assert result["gate"]["all_pass"] is False


def test_holdout_bootstrap_lower_bound_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "_bootstrap_summary",
        lambda *args, **kwargs: {
            "draws": 10000,
            "seed": 260802,
            "lower_95": 0.0,
            "upper_95": 2.0,
        },
    )

    result, _ = analyzer._holdout_baseline_metrics(
        _holdout_baselines(), selected_layer=4
    )

    assert result["gate"]["gap_point_pass"] is True
    assert result["gate"]["gap_bootstrap_pass"] is False
    assert result["gate"]["all_pass"] is False


def test_forward_reverse_and_specificity_estimands_have_frozen_orientation() -> None:
    baselines, patches = _metric_rows()

    result = analyzer._metrics_for_layer(
        baselines, patches, layer=4, bootstrap=False
    )

    metrics = result["metrics"]
    assert result["gap_mean"] == pytest.approx(4.0)
    assert metrics["F"]["mean"] == pytest.approx(2.0)
    assert metrics["N"]["mean"] == pytest.approx(2.0)
    assert metrics["F"]["gap_denominator_positive"] is True
    assert metrics["F"]["all_lodo_gap_denominators_positive"] is True
    assert metrics["F"]["recovery_fraction_defined"] is True
    assert metrics["F"]["mean_over_gap"] == pytest.approx(0.5)
    assert metrics["N"]["mean_over_gap"] == pytest.approx(0.5)
    for name in ("S_F_same", "S_F_unrelated", "S_N_same", "S_N_unrelated"):
        assert metrics[name]["mean"] == pytest.approx(1.0)
        assert metrics[name]["mean_over_gap"] == pytest.approx(0.25)
    assert metrics["identity_R"]["mean"] == pytest.approx(0.0)
    assert metrics["identity_D"]["mean"] == pytest.approx(0.0)
    exact_metric_names = {
        name for name in metrics if name.endswith("_exact_length")
    }
    assert exact_metric_names == {
        "S_F_same_exact_length",
        "S_F_unrelated_exact_length",
        "S_N_same_exact_length",
        "S_N_unrelated_exact_length",
    }
    assert metrics["S_F_same_exact_length"]["inference_unit"] == (
        "exact_length_lexical_pair"
    )
    assert metrics["S_F_unrelated_exact_length"]["inference_unit"] == (
        "exact_length_reciprocal_control_dyad"
    )
    assert result["length_matched_specificity_units"] == {
        "same_pair_lexical_pairs": [f"pair-{index:02d}" for index in range(4)],
        "unrelated_control_dyads": ["dyad-00", "dyad-01"],
    }


def test_identity_gate_passes_within_tolerance_and_rejects_excess() -> None:
    baselines, patches = _metric_rows(n_pairs=2)
    identity_rows = [
        row for row in patches if row["condition"] in {"identity_r", "identity_d"}
    ]
    for row in identity_rows:
        role_margin = -2.0 if row["condition"] == "identity_r" else 2.0
        row["diagnostics"] = _diagnostics(role_margin, offset=5e-5)

    boundary = analyzer._holdout_engineering(baselines, identity_rows)

    assert boundary["maximum_identity_xy_difference"] == pytest.approx(5e-5)
    assert boundary["identity_pass"] is True
    identity_rows[0]["diagnostics"] = _diagnostics(-2.0, offset=1.0001e-4)
    exceeded = analyzer._holdout_engineering(baselines, identity_rows)
    assert exceeded["identity_pass"] is False


def test_discovery_engineering_gates_bind_prior_records_with_current_signature() -> None:
    d_diagnostics = _diagnostics(2.0)
    r_diagnostics = _diagnostics(-2.0)
    prompts = [
        {
            "prompt_id": "prompt-d",
            "prior_syntax_record_id": "record-d",
            "prior_x_logit": d_diagnostics["x_logit"],
            "prior_y_logit": d_diagnostics["y_logit"],
            "prior_full_vocab_logits_sha256": d_diagnostics[
                "full_vocab_logits_sha256"
            ],
        },
        {
            "prompt_id": "prompt-r",
            "prior_syntax_record_id": "record-r",
            "prior_x_logit": r_diagnostics["x_logit"],
            "prior_y_logit": r_diagnostics["y_logit"],
            "prior_full_vocab_logits_sha256": r_diagnostics[
                "full_vocab_logits_sha256"
            ],
        },
    ]
    baselines = [
        {
            "prompt_id": "prompt-d",
            "item_id": "item-a",
            "prompt_role": "D",
            "diagnostics": d_diagnostics,
            "duplicate_diagnostics": copy.deepcopy(d_diagnostics),
        },
        {
            "prompt_id": "prompt-r",
            "item_id": "item-a",
            "prompt_role": "R",
            "diagnostics": r_diagnostics,
            "duplicate_diagnostics": copy.deepcopy(r_diagnostics),
        },
    ]
    patches = [
        {
            "item_id": "item-a",
            "condition": "identity_d",
            "layer": 4,
            "diagnostics": copy.deepcopy(d_diagnostics),
            "hook_trace": dict(TRACE),
        },
        {
            "item_id": "item-a",
            "condition": "identity_r",
            "layer": 4,
            "diagnostics": copy.deepcopy(r_diagnostics),
            "hook_trace": dict(TRACE),
        },
        {
            "item_id": "item-a",
            "condition": "forward_paired",
            "layer": 27,
            "diagnostics": copy.deepcopy(d_diagnostics),
            "hook_trace": dict(TRACE),
        },
        {
            "item_id": "item-a",
            "condition": "reverse_paired",
            "layer": 27,
            "diagnostics": copy.deepcopy(r_diagnostics),
            "hook_trace": dict(TRACE),
        },
    ]

    gates = analyzer._engineering_gates(baselines, patches, prompts)

    assert gates["prior_xy_pass"] is True
    assert gates["prior_full_vocab_row_sha256_pass"] is True
    assert gates["duplicate_pass"] is True
    assert gates["identity_pass"] is True
    assert gates["final_block_source_copy_pass"] is True
    tampered_prompts = copy.deepcopy(prompts)
    tampered_prompts[0]["prior_x_logit"] += 1e-3
    assert analyzer._engineering_gates(
        baselines, patches, tampered_prompts
    )["prior_xy_pass"] is False


def test_plan_bound_baseline_validation_reconstructs_and_rejects_tampering() -> None:
    activations = np.arange(
        2 * runner.MODEL_LAYERS * 3, dtype="<f4"
    ).reshape(2, runner.MODEL_LAYERS, 3)
    prompts = [_prompt("item-a", "pair-1", "D"), _prompt("item-a", "pair-1", "R")]
    records = [
        _baseline_record(
            prompt,
            phase="discovery",
            activation_row=index,
            activations=activations,
            diagnostics=_diagnostics(1.0 if prompt["prompt_role"] == "D" else -1.0),
            duplicate=True,
        )
        for index, prompt in enumerate(prompts)
    ]

    validated = analyzer._validate_baselines(
        records,
        prompts,
        activations,
        phase="discovery",
        require_duplicate=True,
    )
    assert validated == records

    identity_tamper = copy.deepcopy(records)
    identity_tamper[0]["pair_id"] = "substituted-pair"
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_baselines(
            identity_tamper,
            prompts,
            activations,
            phase="discovery",
            require_duplicate=True,
        )

    activation_tamper = activations.copy()
    activation_tamper[0, 0, 0] += 1.0
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_baselines(
            records,
            prompts,
            activation_tamper,
            phase="discovery",
            require_duplicate=True,
        )


def test_plan_bound_patch_validation_reconstructs_and_rejects_layer_and_trace_tampering() -> None:
    activations = np.zeros((2, runner.MODEL_LAYERS, 2), dtype="<f4")
    prompts = [_prompt("item-a", "pair-1", "D"), _prompt("item-a", "pair-1", "R")]
    baselines = [
        _baseline_record(
            prompt,
            phase="holdout_baseline",
            activation_row=index,
            activations=activations,
            diagnostics=_diagnostics(1.0 if prompt["prompt_role"] == "D" else -1.0),
            duplicate=False,
        )
        for index, prompt in enumerate(prompts)
    ]
    item = {
        "item_id": "item-a",
        "pair_id": "pair-1",
        "truth_polarity": "positive",
        "unrelated_control_cluster_id": "dyad-1",
    }
    template = runner._patch_template(
        bank_role="holdout",
        item=item,
        condition="forward_paired",
        source_prompt=prompts[0],
        target_prompt=prompts[1],
        layer=None,
    )
    record = runner._patch_record(
        template,
        phase="holdout_patch",
        layer=4,
        source_baseline=baselines[0],
        target_baseline=baselines[1],
        diagnostics=_diagnostics(0.5),
        trace=TRACE,
    )

    assert analyzer._validate_patches(
        [record],
        [template],
        baselines,
        phase="holdout_patch",
        selected_layer=4,
    ) == [record]
    assert record["dependency_cluster_id"] == "pair-1"
    assert record["source_target_token_count_difference"] == 0

    layer_tamper = copy.deepcopy(record)
    layer_tamper["layer"] = 8
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_patches(
            [layer_tamper],
            [template],
            baselines,
            phase="holdout_patch",
            selected_layer=4,
        )

    trace_tamper = copy.deepcopy(record)
    trace_tamper["hook_trace"]["hook_calls"] = 2
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_patches(
            [trace_tamper],
            [template],
            baselines,
            phase="holdout_patch",
            selected_layer=4,
        )

    dependency_tamper = copy.deepcopy(record)
    dependency_tamper["dependency_cluster_id"] = "substituted-dyad"
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_patches(
            [dependency_tamper],
            [template],
            baselines,
            phase="holdout_patch",
            selected_layer=4,
        )

    length_tamper = copy.deepcopy(record)
    length_tamper["source_target_token_count_difference"] = 1
    with pytest.raises(analyzer.CausalBindingAnalysisError):
        analyzer._validate_patches(
            [length_tamper],
            [template],
            baselines,
            phase="holdout_patch",
            selected_layer=4,
        )

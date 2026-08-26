from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval import analyze_coherent_readout_v3_cross_codebook as analyzer


def _condition_margins(
    *,
    recipient: float = -1.0,
    primary_effect: float = 1.4,
) -> dict[str, float]:
    margins = {
        analyzer.PRIMARY_CONDITION: recipient + primary_effect,
        "content_same": recipient + 0.1,
        "answer_anticopy": recipient + 0.3,
        "codebook_anticopy": recipient + 0.3,
        "distractor_anticopy": recipient + 0.3,
        "query_anticopy": recipient + 0.3,
        "order_anticopy": recipient + 0.3,
        "full_anticopy": recipient + 0.3,
        "null_0_anticopy": recipient + 0.2,
        "null_1_anticopy": recipient + 0.2,
        "null_2_anticopy": recipient + 0.2,
        "null_3_anticopy": recipient + 0.2,
        "content_erase": recipient + 0.6,
        "content_rescue_same": recipient + 0.1,
        "content_rescue_opposite": recipient + primary_effect,
        "null_0_erase": recipient + 0.1,
        "null_1_erase": recipient + 0.1,
        "null_2_erase": recipient + 0.1,
        "null_3_erase": recipient + 0.1,
        "identity": recipient,
        "full_text_counterfactual": 1.0,
    }
    return margins


def _engineering_receipt(*, trace_pass: bool = True) -> dict[str, Any]:
    return analyzer._engineering_receipt(
        {
            "identity_absolute_x_logit_error": 0.0,
            "identity_absolute_y_logit_error": 0.0,
            "identity_global_max_preserved": True,
            "rescue_same_absolute_x_logit_error": 0.0,
            "rescue_same_absolute_y_logit_error": 0.0,
            "rescue_opposite_absolute_x_logit_error": 0.0,
            "rescue_opposite_absolute_y_logit_error": 0.0,
            "trace_projector_and_sidecar_pass": trace_pass,
            "source_reference_pass": True,
            "patched_activation_hash_pass": True,
            "fit_only_center_pass": True,
        }
    )


def _final_rows(
    *,
    worlds: int = 32,
    primary_effect: float = 1.4,
    counterfactual_argmax: bool = True,
    unique_xy: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    strata = (
        ("P", "identity"),
        ("P", "swapped"),
        ("Q", "identity"),
        ("Q", "swapped"),
    )
    for world_index in range(worlds):
        for replicate in range(2):
            for queried_property, codebook in strata:
                recipient = -1.0
                rows.append(
                    {
                        "cell_id": (f"cell-{world_index:03d}-{replicate}-{queried_property}-{codebook}"),
                        "world_id": f"world-{world_index:03d}",
                        "queried_property": queried_property,
                        "codebook_id": codebook,
                        "recipient_margin": recipient,
                        "counterfactual_margin": 1.0,
                        "recipient_label_probability_mass": 0.99,
                        "primary_label_probability_mass": 0.98,
                        "primary_global_argmax_is_counterfactual": (counterfactual_argmax),
                        "primary_unique_global_argmax_in_xy": unique_xy,
                        "engineering_receipt": _engineering_receipt(),
                        "condition_margins": _condition_margins(
                            recipient=recipient,
                            primary_effect=primary_effect,
                        ),
                    }
                )
    return rows


def _localization_rows() -> list[dict[str, Any]]:
    required = {
        analyzer.PRIMARY_CONDITION,
        *analyzer.SPECIFICITY_CONTROLS,
        *analyzer.SIMULTANEOUS_NULLS,
        *analyzer.ENGINEERING_CONDITIONS,
    }
    return [
        {
            "cell_id": row["cell_id"],
            "world_id": row["world_id"],
            "queried_property": row["queried_property"],
            "codebook_id": row["codebook_id"],
            "recipient_margin": row["recipient_margin"],
            "counterfactual_margin": row["counterfactual_margin"],
            "condition_margins": {key: value for key, value in row["condition_margins"].items() if key in required},
        }
        for row in _final_rows(worlds=8)
    ]


def _effect_gate_summary(*, ratio: float, lower: float = 1e-9) -> dict[str, Any]:
    return {
        "ratio_defined": True,
        "mean_over_G": ratio,
        "bootstrap_95": {"lower_95": lower, "upper_95": 1.0},
    }


def _diagnostics(native_answer: str, *, mass: float = 0.99) -> dict[str, Any]:
    x_logit, y_logit = (1.0, -1.0) if native_answer == "X" else (-1.0, 1.0)
    token_id = analyzer.X_TOKEN_ID if native_answer == "X" else analyzer.Y_TOKEN_ID
    return {
        "x_logit": x_logit,
        "y_logit": y_logit,
        "x_minus_y_margin": x_logit - y_logit,
        "full_vocab_logsumexp": float(np.logaddexp(x_logit, y_logit) - math.log(mass)),
        "label_probability_mass": mass,
        "greedy_token_id": token_id,
        "greedy_logit": max(x_logit, y_logit),
        "maximum_token_ids": [token_id],
        "maximum_tie_count": 1,
        "full_vocab_logits_sha256": "ab" * 32,
    }


def _projected_trace(
    recipient: np.ndarray,
    source: np.ndarray,
    direction: np.ndarray,
    *,
    layer: int = 16,
) -> tuple[dict[str, Any], np.ndarray]:
    expected = analyzer.runner.expected_intervention_activation(
        recipient,
        operation="projected_patch",
        source=source,
        direction=direction,
    )
    patched = expected.copy()
    recipient64 = recipient.astype(np.float64)
    source64 = source.astype(np.float64)
    patched64 = patched.astype(np.float64)
    direction64 = direction.astype(np.float64)
    unit = direction64 / np.linalg.norm(direction64)
    delta = patched64 - recipient64
    orthogonal = delta - unit * float(unit @ delta)
    full_displacement = float(np.linalg.norm(source64 - recipient64))
    projector = analyzer.runner._projector_diagnostics(direction)
    expected_hash = analyzer.f32_sha256(expected)
    return (
        {
            "hook_calls": 1,
            "hook_removed": True,
            "non_target_tokens_unchanged": True,
            "pre_activation_matches_registered_recipient": True,
            "post_activation_matches_expected": True,
            "pre_activation_sha256": analyzer.f32_sha256(recipient),
            "post_activation_sha256": expected_hash,
            "expected_activation_sha256": expected_hash,
            "post_expected_l2_error": 0.0,
            "post_expected_l2_tolerance": 1e-6 * max(1.0, float(np.linalg.norm(expected.astype(np.float64)))),
            "displacement_l2": float(np.linalg.norm(delta)),
            **projector,
            "orthogonal_displacement_l2": float(np.linalg.norm(orthogonal)),
            "orthogonal_displacement_tolerance": 1e-6 * max(1.0, float(np.linalg.norm(recipient64))),
            "corresponding_full_displacement_l2": full_displacement,
            "selective_displacement_tolerance": 1e-6 * max(1.0, full_displacement),
            "orthogonal_displacement_pass": True,
            "selective_not_larger_than_full_pass": True,
            "pre_axis_coefficient": float(unit @ recipient64),
            "source_axis_coefficient": float(unit @ source64),
            "post_axis_coefficient": float(unit @ patched64),
            "expected_axis_coefficient": float(unit @ expected.astype(np.float64)),
            "post_expected_axis_coefficient_error": 0.0,
            "finite_activations": True,
            "operation": "projected_patch",
            "intervention_kind": "projected_patch",
            "layer": layer,
            "token_index": -1,
            "strength": 1.0,
            "model_calls": 1,
            "generation_used": False,
            "patched_activation_hash_pass": True,
            "finite_logits": True,
            "direction_name": "content",
        },
        patched,
    )


def _behavior_rows(
    worlds: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    by_factors = {}
    for world_index in range(worlds):
        world_id = f"world-{world_index:03d}"
        for entity, prop, distractor, codebook, order in itertools.product(
            ("a", "b"),
            ("P", "Q"),
            ("P", "Q"),
            ("identity", "swapped"),
            ("first", "second"),
        ):
            native = "X" if (prop == "P") == (codebook == "identity") else "Y"
            cell_id = f"{world_id}-{entity}-{prop}-{distractor}-{codebook}-{order}"
            row = {
                "cell_id": cell_id,
                "world_id": world_id,
                "queried_property": prop,
                "codebook_id": codebook,
                "native_answer": native,
                "diagnostics": _diagnostics(native),
            }
            rows.append(row)
            by_factors[(world_id, entity, prop, distractor, codebook, order)] = row
    pairs = []
    for world_index in range(worlds):
        world_id = f"world-{world_index:03d}"
        for entity, prop, codebook in itertools.product(("a", "b"), ("P", "Q"), ("identity", "swapped")):
            distractor = "P" if entity == "a" else "Q"
            order = "first" if entity == "a" else "second"
            recipient = by_factors[(world_id, entity, prop, distractor, codebook, order)]
            counterfactual = by_factors[
                (
                    world_id,
                    entity,
                    "Q" if prop == "P" else "P",
                    distractor,
                    codebook,
                    order,
                )
            ]
            pairs.append(
                {
                    "recipient_cell_id": recipient["cell_id"],
                    "counterfactual_cell_id": counterfactual["cell_id"],
                    "world_id": world_id,
                    "queried_property": prop,
                    "codebook_id": codebook,
                }
            )
    return rows, pairs


def test_target_margin_is_z_counterfactual_minus_z_native() -> None:
    assert analyzer.target_oriented_margin(native_answer="X", x_logit=4.0, y_logit=1.5) == pytest.approx(-2.5)
    assert analyzer.target_oriented_margin(native_answer="Y", x_logit=4.0, y_logit=1.5) == pytest.approx(2.5)
    with pytest.raises(analyzer.CrossCodebookAnalysisError):
        analyzer.target_oriented_margin(native_answer="Z", x_logit=4.0, y_logit=1.5)


def test_behavioral_admission_reconstructs_full_factorial_and_G() -> None:
    rows, pairs = _behavior_rows()

    result = analyzer._behavioral_admission(rows, pairs, expected_worlds=2)

    assert result["n_cells"] == 64
    assert result["n_worlds"] == 2
    assert result["native_accuracy"] == 1.0
    assert set(result["counterfactual_accuracy_by_transition_and_codebook"]) == {
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    }
    assert result["G"]["mean"] == pytest.approx(4.0)
    assert result["gates"]["pass"] is True


def test_behavioral_admission_rejects_missing_rows_and_diagnostic_tampering() -> None:
    rows, pairs = _behavior_rows()
    with pytest.raises(analyzer.CrossCodebookAnalysisError):
        analyzer._behavioral_admission(rows[:-1], pairs, expected_worlds=2)

    tampered = copy.deepcopy(rows)
    tampered[0]["diagnostics"]["x_minus_y_margin"] += 1.0
    with pytest.raises(analyzer.CrossCodebookAnalysisError):
        analyzer._behavioral_admission(tampered, pairs, expected_worlds=2)

    impossible_maximum = copy.deepcopy(rows)
    diagnostics = impossible_maximum[0]["diagnostics"]
    diagnostics["y_logit"] = diagnostics["x_logit"]
    diagnostics["x_minus_y_margin"] = 0.0
    diagnostics["full_vocab_logsumexp"] = float(
        np.logaddexp(diagnostics["x_logit"], diagnostics["y_logit"]) - math.log(diagnostics["label_probability_mass"])
    )
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="Y membership"):
        analyzer._behavioral_admission(impossible_maximum, pairs, expected_worlds=2)


def test_strict_diagnostic_engineering_catches_common_mode_identity_shift() -> None:
    baseline = _diagnostics("X")
    shifted = copy.deepcopy(baseline)
    shifted["x_logit"] += 0.01
    shifted["y_logit"] += 0.01
    shifted["greedy_logit"] += 0.01
    shifted["full_vocab_logsumexp"] += 0.01
    assert shifted["x_minus_y_margin"] == baseline["x_minus_y_margin"]
    diagnostics = {
        "recipient_baseline": baseline,
        "identity": shifted,
        analyzer.PRIMARY_CONDITION: baseline,
        "content_same": baseline,
        "content_rescue_same": baseline,
        "content_rescue_opposite": baseline,
    }

    result = analyzer._strict_final_diagnostic_engineering(diagnostics, trace_projector_and_sidecar_pass=True)

    assert result["identity"]["absolute_x_logit_difference"] == pytest.approx(0.01)
    assert result["identity"]["absolute_y_logit_difference"] == pytest.approx(0.01)
    assert result["identity"]["pass"] is False
    assert result["pass"] is False


def test_strict_diagnostic_engineering_requires_trace_and_duplicate_equivalence() -> None:
    baseline = _diagnostics("X")
    diagnostics = {
        "recipient_baseline": baseline,
        "identity": baseline,
        analyzer.PRIMARY_CONDITION: baseline,
        "content_same": baseline,
        "content_rescue_same": baseline,
        "content_rescue_opposite": baseline,
    }

    assert (
        analyzer._strict_final_diagnostic_engineering(diagnostics, trace_projector_and_sidecar_pass=True)["pass"]
        is True
    )
    assert (
        analyzer._strict_final_diagnostic_engineering(diagnostics, trace_projector_and_sidecar_pass=False)["pass"]
        is False
    )

    altered = copy.deepcopy(diagnostics)
    altered["content_rescue_same"] = _diagnostics("Y")
    assert (
        analyzer._strict_final_diagnostic_engineering(altered, trace_projector_and_sidecar_pass=True)["pass"] is False
    )


def test_world_aggregation_respects_cluster_dependence_not_cell_multiplicity() -> None:
    values = {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 10.0}
    mapping = {"a1": "world-a", "a2": "world-a", "a3": "world-a", "b1": "world-b"}

    world_means = analyzer._world_means(values, mapping)

    assert world_means == {"world-a": 0.0, "world-b": 10.0}
    assert np.mean(list(world_means.values())) == pytest.approx(5.0)
    assert np.mean(list(values.values())) == pytest.approx(2.5)


def test_common_world_bootstrap_is_deterministic_and_has_locked_size() -> None:
    first = analyzer._common_bootstrap_indices(32)
    second = analyzer._common_bootstrap_indices(32)

    assert np.array_equal(first, second)
    assert first.shape == (10_000, 32)
    assert first.min() >= 0
    assert first.max() < 32


@pytest.mark.parametrize(
    "gaps",
    [
        {"w0": -1.0, "w1": -1.0, "w2": -1.0},
        {"w0": 5.0, "w1": -1.0, "w2": -1.0},
    ],
)
def test_ratio_is_undefined_for_nonpositive_aggregate_or_any_loo_G(
    gaps: dict[str, float],
) -> None:
    values = {world: 1.0 for world in gaps}
    indices = analyzer._common_bootstrap_indices(3, draws=101, seed=7)

    summary = analyzer._effect_summary(values, gaps, bootstrap_indices=indices)

    assert summary["ratio_defined"] is False
    assert summary["mean_over_G"] is None
    assert summary["bootstrap_ratio_95"] is None


def test_zero_bootstrap_denominator_invalidates_ratio_interval_without_dropping_draw() -> None:
    values = {"w0": 1.0, "w1": 1.0, "w2": 1.0}
    gaps = {"w0": 0.0, "w1": 1.0, "w2": 1.0}
    indices = np.asarray([[0, 0, 0], [0, 1, 2], [1, 1, 2]], dtype=int)

    summary = analyzer._effect_summary(values, gaps, bootstrap_indices=indices)

    assert summary["ratio_defined"] is True
    assert summary["bootstrap_ratio_valid"] is False
    assert summary["bootstrap_ratio_95"] is None


def test_final_metrics_use_world_clusters_common_draws_and_all_strata() -> None:
    metrics = analyzer._final_metrics(_final_rows())

    assert metrics["n_cells"] == 256
    assert metrics["n_worlds"] == 32
    assert metrics["bootstrap"] == {
        "dependency_unit": "symbolic_world",
        "draws": 10_000,
        "seed": 260804,
        "common_draws_for_all_comparisons": True,
        "index_matrix_sha256": metrics["bootstrap"]["index_matrix_sha256"],
    }
    assert metrics["G"]["mean"] == pytest.approx(2.0)
    assert metrics["primary"]["mean"] == pytest.approx(1.4)
    assert metrics["primary"]["mean_over_G"] == pytest.approx(0.7)
    assert set(metrics["primary_strata"]) == {
        "P_to_Q__identity",
        "P_to_Q__swapped",
        "Q_to_P__identity",
        "Q_to_P__swapped",
    }
    assert all(summary["positive_world_fraction"] == 1.0 for summary in metrics["primary_strata"].values())


def test_counterfactual_flip_gate_uses_unique_global_argmax_not_margin_sign() -> None:
    rows = _final_rows(counterfactual_argmax=False, unique_xy=False)
    assert all(row["condition_margins"][analyzer.PRIMARY_CONDITION] > 0.0 for row in rows)

    metrics = analyzer._final_metrics(rows)

    assert metrics["primary_channel"]["counterfactual_global_argmax_rate"]["point"] == 0.0
    assert analyzer._final_gates(metrics)["primary"]["pass"] is False


def test_simultaneous_null_uses_itemwise_max_before_world_aggregation() -> None:
    rows = _final_rows()
    for index, row in enumerate(rows):
        primary = row["condition_margins"][analyzer.PRIMARY_CONDITION]
        row["condition_margins"]["null_0_anticopy"] = primary if index % 2 == 0 else primary - 2.0
        row["condition_margins"]["null_1_anticopy"] = primary - 2.0 if index % 2 == 0 else primary
        row["condition_margins"]["null_2_anticopy"] = primary - 3.0
        row["condition_margins"]["null_3_anticopy"] = primary - 4.0

    metrics = analyzer._final_metrics(rows)

    assert metrics["simultaneous_max_null"]["mean"] == pytest.approx(0.0)
    assert metrics["simultaneous_max_null"]["mean_over_G"] == pytest.approx(0.0)


def test_gate_boundaries_are_inclusive_for_points_and_strict_for_bootstrap() -> None:
    at_boundary = _effect_gate_summary(ratio=0.20, lower=1e-12)
    below_ratio = _effect_gate_summary(ratio=0.20 - 1e-12, lower=1e-12)
    zero_lower = _effect_gate_summary(ratio=0.20, lower=0.0)

    assert analyzer._effect_gate(at_boundary, minimum_ratio=0.20) is True
    assert analyzer._effect_gate(below_ratio, minimum_ratio=0.20) is False
    assert analyzer._effect_gate(zero_lower, minimum_ratio=0.20) is False


def test_passing_final_rows_satisfy_primary_specificity_and_natural_use() -> None:
    metrics = analyzer._final_metrics(_final_rows())
    gates = analyzer._final_gates(metrics)

    assert gates["engineering"]["pass"] is True
    assert gates["primary"]["pass"] is True
    assert gates["specificity"]["pass"] is True
    assert gates["natural_use"]["pass"] is True
    assert (
        analyzer._final_status(
            engineering_pass=gates["engineering"]["pass"],
            primary_pass=gates["primary"]["pass"],
            specificity_pass=gates["specificity"]["pass"],
            natural_use_pass=gates["natural_use"]["pass"],
        )
        == "CONTENT_RECOMPOSITION_AND_PARTIAL_NATURAL_USE_SUPPORTED"
    )


def test_strict_raw_engineering_failure_cannot_receive_scientific_status() -> None:
    rows = _final_rows()
    rows[0]["engineering_receipt"] = _engineering_receipt(trace_pass=False)

    gates = analyzer._final_gates(analyzer._final_metrics(rows))

    assert gates["engineering"]["strict_artifact_pass"] is False
    assert gates["engineering"]["pass"] is False
    assert (
        analyzer._final_status(
            engineering_pass=gates["engineering"]["pass"],
            primary_pass=gates["primary"]["pass"],
            specificity_pass=gates["specificity"]["pass"],
            natural_use_pass=gates["natural_use"]["pass"],
        )
        == "FINAL_STOP_ENGINEERING_INVALID"
    )


@pytest.mark.parametrize(
    ("engineering", "primary", "specificity", "natural", "expected"),
    [
        (False, True, True, True, "FINAL_STOP_ENGINEERING_INVALID"),
        (
            True,
            False,
            False,
            False,
            "NO_REPLICATED_PROJECTED_CONTENT_RECOMPOSITION",
        ),
        (
            True,
            True,
            False,
            False,
            "NONSPECIFIC_PROJECTED_TRANSFER_REPLICATED",
        ),
        (
            True,
            True,
            True,
            False,
            "CONTENT_RECOMPOSITION_SUPPORTED_NATURAL_USE_NOT_ESTABLISHED",
        ),
        (
            True,
            True,
            True,
            True,
            "CONTENT_RECOMPOSITION_AND_PARTIAL_NATURAL_USE_SUPPORTED",
        ),
    ],
)
def test_final_status_ladder_is_strict(
    engineering: bool,
    primary: bool,
    specificity: bool,
    natural: bool,
    expected: str,
) -> None:
    assert (
        analyzer._final_status(
            engineering_pass=engineering,
            primary_pass=primary,
            specificity_pass=specificity,
            natural_use_pass=natural,
        )
        == expected
    )


def test_final_status_requires_explicit_engineering_argument() -> None:
    with pytest.raises(TypeError):
        analyzer._final_status(  # type: ignore[call-arg]
            primary_pass=True,
            specificity_pass=True,
            natural_use_pass=True,
        )


def test_final_gates_reject_missing_control_registries() -> None:
    metrics = analyzer._final_metrics(_final_rows())
    metrics["specificity"].pop("content_same")
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="specificity registry"):
        analyzer._final_gates(metrics)


def test_localization_selects_earliest_preregistered_passing_layer() -> None:
    decisions = {
        layer: {
            "causal_anticopy_pass": layer in {16, 20, 24},
            "control_components": {condition: layer in {12, 16, 20, 24} for condition in analyzer.SPECIFICITY_CONTROLS},
            "max_four_null_means_pass": layer in {12, 16, 20, 24},
            "control_separation_pass": layer in {12, 16, 20, 24},
            "identity_pass": True,
            "pass": layer in {16, 20, 24},
        }
        for layer in analyzer.LAYER_GRID
    }

    assert analyzer._select_localization_layer(decisions, engineering_pass=True) == (
        16,
        "LOCALIZATION_LAYER_LOCKED_HOLDOUT_BASELINE_AUTHORIZED",
    )
    decisions[16]["identity_pass"] = False
    decisions[16]["pass"] = False
    assert analyzer._select_localization_layer(decisions, engineering_pass=True) == (
        20,
        "LOCALIZATION_LAYER_LOCKED_HOLDOUT_BASELINE_AUTHORIZED",
    )
    assert analyzer._select_localization_layer(decisions, engineering_pass=False) == (
        None,
        "LOCALIZATION_STOP_ENGINEERING_INVALID",
    )
    for decision in decisions.values():
        decision["causal_anticopy_pass"] = False
        decision["pass"] = False
    assert analyzer._select_localization_layer(decisions, engineering_pass=True) == (
        None,
        "LOCALIZATION_STOP_NO_PREREGISTERED_LAYER",
    )

    decisions[8]["pass"] = True
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="does not recompute"):
        analyzer._select_localization_layer(decisions, engineering_pass=True)


def test_localization_metrics_enforce_all_seven_controls_and_four_nulls() -> None:
    metrics = analyzer._localization_metrics(_localization_rows(), layer=16)
    decision = analyzer._localization_decision(metrics)

    assert metrics["n_worlds"] == 8
    assert metrics["n_recipients"] == 64
    assert set(metrics["specificity"]) == set(analyzer.SPECIFICITY_CONTROLS)
    assert set(metrics["nulls"]) == set(analyzer.SIMULTANEOUS_NULLS)
    assert decision["causal_anticopy_pass"] is True
    assert all(decision["control_components"].values())
    assert decision["max_four_null_means_pass"] is True
    assert decision["identity_pass"] is True
    assert decision["pass"] is True


def test_localization_identity_or_one_control_failure_blocks_layer() -> None:
    rows = _localization_rows()
    rows[0]["condition_margins"]["identity"] += 1e-3
    decision = analyzer._localization_decision(analyzer._localization_metrics(rows, layer=16))
    assert decision["identity_pass"] is False
    assert decision["pass"] is False

    rows = _localization_rows()
    for row in rows:
        row["condition_margins"]["content_same"] = row["condition_margins"][analyzer.PRIMARY_CONDITION]
    decision = analyzer._localization_decision(analyzer._localization_metrics(rows, layer=16))
    assert decision["control_components"]["content_same"] is False
    assert decision["control_separation_pass"] is False
    assert decision["pass"] is False


def test_localization_max_null_uses_null_means_and_every_loo() -> None:
    rows = _localization_rows()
    for row in rows:
        recipient = row["recipient_margin"]
        if row["world_id"] == "world-000":
            null_effect = 0.0
        else:
            null_effect = 1.5
        row["condition_margins"]["null_0_anticopy"] = recipient + null_effect
    decision = analyzer._localization_decision(analyzer._localization_metrics(rows, layer=16))
    assert decision["max_four_null_means_pass"] is False

    rows = _localization_rows()
    for index, row in enumerate(rows):
        recipient = row["recipient_margin"]
        high, low = recipient + 2.0, recipient - 1.0
        row["condition_margins"]["null_0_anticopy"] = high if index % 2 else low
        row["condition_margins"]["null_1_anticopy"] = low if index % 2 else high
    decision = analyzer._localization_decision(analyzer._localization_metrics(rows, layer=16))
    assert decision["max_four_null_means_pass"] is True


def test_localization_rejects_globally_balanced_but_world_imbalanced_strata() -> None:
    rows = _localization_rows()
    first = next(
        row
        for row in rows
        if row["world_id"] == "world-000" and row["queried_property"] == "P" and row["codebook_id"] == "identity"
    )
    second = next(
        row
        for row in rows
        if row["world_id"] == "world-001" and row["queried_property"] == "P" and row["codebook_id"] == "swapped"
    )
    first["codebook_id"], second["codebook_id"] = (
        second["codebook_id"],
        first["codebook_id"],
    )

    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="two recipients per world"):
        analyzer._localization_metrics(rows, layer=16)


def test_localization_decision_rejects_missing_null_registry() -> None:
    metrics = analyzer._localization_metrics(_localization_rows(), layer=16)
    metrics["nulls"].pop("null_3_anticopy")
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="null registry"):
        analyzer._localization_decision(metrics)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.update(extra=True),
        lambda row: row.update(recipient_margin=float("nan")),
        lambda row: row.update(queried_property="R"),
        lambda row: row["condition_margins"].pop("content_same"),
        lambda row: row.update(primary_global_argmax_is_counterfactual=1),
        lambda row: row.update(
            primary_global_argmax_is_counterfactual=True,
            primary_unique_global_argmax_in_xy=False,
        ),
        lambda row: row["engineering_receipt"].update(payload_canonical_sha256="00" * 32),
        lambda row: (
            row["condition_margins"].update(content_anticopy=0.0),
            row.update(
                primary_global_argmax_is_counterfactual=False,
                primary_unique_global_argmax_in_xy=True,
            ),
        ),
    ],
)
def test_final_row_tampering_is_rejected(mutator: Any) -> None:
    rows = _final_rows(worlds=1)
    mutator(rows[0])

    with pytest.raises(analyzer.CrossCodebookAnalysisError):
        analyzer._normalized_final_rows(rows)


def test_bilingual_bundle_binds_json_markdown_and_hashes(tmp_path: Path) -> None:
    analysis = {
        "schema_version": analyzer.FINAL_ANALYSIS_SCHEMA,
        "status": "NO_REPLICATED_PROJECTED_CONTENT_RECOMPOSITION",
        "selected_layer": 16,
        "claim_boundaries": analyzer.CLAIM_BOUNDARIES,
    }
    json_path = tmp_path / "analysis.json"
    markdown_path = tmp_path / "analysis.md"
    manifest_path = tmp_path / "analysis.manifest.json"

    manifest = analyzer._write_analysis_bundle(
        analysis,
        json_path=json_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## English" in markdown
    assert "## 한국어" in markdown
    assert manifest["analysis_json"]["file_sha256"] == analyzer.file_sha256(json_path)
    assert manifest["analysis_markdown"]["file_sha256"] == analyzer.file_sha256(markdown_path)
    assert manifest["claim_boundaries"] == analyzer.CLAIM_BOUNDARIES

    same = analyzer._write_analysis_bundle(
        analysis,
        json_path=json_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
    )
    assert same == manifest

    changed = copy.deepcopy(analysis)
    changed["selected_layer"] = 20
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="refusing to overwrite"):
        analyzer._write_analysis_bundle(
            changed,
            json_path=json_path,
            markdown_path=markdown_path,
            manifest_path=manifest_path,
        )


def test_vector_hash_uses_exact_little_endian_float32_bytes() -> None:
    vector = np.asarray([1.0, -2.5, 3.25], dtype="<f4")
    same_values_f64 = vector.astype(np.float64)

    assert analyzer.f32_sha256(vector) == analyzer.f32_sha256(same_values_f64)
    tampered = vector.copy()
    tampered[1] += np.float32(1e-3)
    assert analyzer.f32_sha256(tampered) != analyzer.f32_sha256(vector)


def test_erasure_center_is_exact_fit_only_float64_mean_cast_to_f32() -> None:
    activations = np.zeros((512, 5, 1536), dtype="<f4")
    activations[0, 0, 0] = np.float32(1e8)
    activations[1:, 0, 0] = np.float32(0.125)

    observed = analyzer._fit_only_intercept(activations)
    expected = np.mean(activations.astype(np.float64), axis=0).astype("<f4")

    assert observed.dtype == np.dtype("<f4")
    assert np.array_equal(observed, expected)
    with pytest.raises(analyzer.CrossCodebookAnalysisError):
        analyzer._fit_only_intercept(np.zeros((1024, 1536), dtype="<f4"))


def test_copying_and_mutating_metrics_does_not_change_status_inputs() -> None:
    metrics = analyzer._final_metrics(_final_rows())
    copied = copy.deepcopy(metrics)
    copied["primary"]["mean_over_G"] = 0.0

    assert analyzer._final_gates(metrics)["primary"]["pass"] is True
    assert analyzer._final_gates(copied)["primary"]["pass"] is False


def test_patch_trace_reconstructs_actual_patched_sidecar_and_rejects_tamper() -> None:
    recipient = np.asarray([1.0, -2.0, 0.5, 3.0], dtype="<f4")
    source = np.asarray([-1.0, 4.0, 0.25, 2.0], dtype="<f4")
    direction = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4")
    trace, patched = _projected_trace(recipient, source, direction)

    validated, reconstructed = analyzer._validate_patch_trace(
        trace,
        layer=16,
        operation="projected_patch",
        direction_name="content",
        recipient=recipient,
        source=source,
        direction=direction,
        center=None,
        patched_activation=patched,
    )
    assert validated["patched_activation_hash_pass"] is True
    assert np.array_equal(reconstructed, patched)

    tampered = patched.copy()
    tampered[0] += np.float32(0.01)
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="hook trace identity"):
        analyzer._validate_patch_trace(
            trace,
            layer=16,
            operation="projected_patch",
            direction_name="content",
            recipient=recipient,
            source=source,
            direction=direction,
            center=None,
            patched_activation=tampered,
        )


def test_baseline_record_reconstructs_sidecar_and_rejects_vector_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = {
        "role": "holdout",
        "world_id": "world-001",
        "cell_id": "cell-001",
        "queried_property": "P",
        "codebook_id": "identity",
        "distractor_property": "Q",
        "queried_entity_slot": "a",
        "fact_line_order": "query_first",
        "native_answer": "X",
    }
    prompt = {
        "cell_id": cell["cell_id"],
        "prompt_id": "prompt-001",
        "input_token_count": 3,
        "execution_input_ids": [1, 2, 3],
        "execution_attention_mask": [1, 1, 1],
    }
    plan = {
        "call_plan_sha256": "cd" * 32,
        "cell_registry": [cell],
        "prompts": [prompt],
    }
    layers = [16]
    block = np.asarray([[1.0, -2.0, 3.0, 0.5]], dtype="<f4")
    trace = {
        "use_cache": False,
        "return_dict": True,
        "generation_used": False,
        "teacher_forced_prompt_forward": True,
        "capture_layers": layers,
        "capture_counts": [1],
        "captures_removed": True,
        "final_attended_token_index": -1,
        "model_calls": 1,
    }
    row = analyzer.runner._baseline_record(
        plan,
        prompt,
        cell,
        phase="holdout-baseline",
        activation_row=0,
        layers=layers,
        activations=block,
        diagnostics=_diagnostics("X"),
        trace=trace,
    )
    template = analyzer.runner._baseline_template(cell, prompt)
    monkeypatch.setattr(
        analyzer.runner,
        "_phase_baseline_templates",
        lambda _plan, _role: [template],
    )
    manifest = {
        "activations": {
            "logical_id_map": {
                cell["cell_id"]: {
                    "activation_row": 0,
                    "activation_sha256": row["activation_sha256"],
                    "activation_layer_sha256": row["activation_layer_sha256"],
                }
            }
        }
    }
    sidecar = block.copy()
    observed = analyzer._validate_baseline_records(
        plan,
        manifest,
        [row],
        sidecar,
        phase="holdout-baseline",
        role="holdout",
        layers=layers,
    )
    assert observed == [row]

    tampered = sidecar.copy()
    tampered[0, 0] += np.float32(0.01)
    with pytest.raises(analyzer.CrossCodebookAnalysisError, match="does not reconstruct"):
        analyzer._validate_baseline_records(
            plan,
            manifest,
            [row],
            tampered,
            phase="holdout-baseline",
            role="holdout",
            layers=layers,
        )


@pytest.mark.parametrize(
    ("admitted", "expected_status"),
    [
        (True, "LOCALIZATION_BASELINE_ADMITTED_PATCH_AUTHORIZED"),
        (False, "LOCALIZATION_BASELINE_STOP_NOT_ADMITTED"),
    ],
)
def test_localization_baseline_stage_is_a_no_forward_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
    admitted: bool,
    expected_status: str,
) -> None:
    plan = {"call_plan_sha256": "12" * 32}
    design: dict[str, Any] = {}
    basis_lock = {"localization_baseline_authorized": True}
    manifest = {
        "records": {"file_sha256": "34" * 32},
        "activations": {"file_sha256": "56" * 32},
    }
    writes: dict[Path, dict[str, Any]] = {}

    monkeypatch.setattr(analyzer, "_load_plan", lambda: (plan, design))
    monkeypatch.setattr(analyzer, "analyze_fit_basis", lambda: ({}, basis_lock))
    monkeypatch.setattr(analyzer, "_load_json", lambda _path: basis_lock)
    monkeypatch.setattr(analyzer.runner, "_load_basis_artifacts", lambda _plan: (None, None, None))
    monkeypatch.setattr(
        analyzer,
        "_validate_phase_manifest",
        lambda *_args, **_kwargs: (
            manifest,
            [],
            np.zeros((1, 1), dtype="<f4"),
            None,
        ),
    )
    monkeypatch.setattr(analyzer, "_validate_baseline_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(analyzer, "_behavioral_inputs", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        analyzer,
        "_behavioral_admission",
        lambda *_args, **_kwargs: {"gates": {"pass": admitted}},
    )
    monkeypatch.setattr(analyzer, "file_sha256", lambda _path: "ab" * 32)
    monkeypatch.setattr(
        analyzer,
        "_write_json",
        lambda path, value: writes.__setitem__(Path(path), copy.deepcopy(dict(value))),
    )
    monkeypatch.setattr(
        analyzer.runner,
        "_load_model",
        lambda: pytest.fail("analyzer must not execute a model forward"),
    )

    analysis, entry = analyzer.analyze_localization_baseline()

    assert analysis["status"] == expected_status
    assert entry["status"] == expected_status
    assert entry["localization_patch_authorized"] is admitted
    assert writes[analyzer.LOCALIZATION_ENTRY_PATH] == entry


def test_fit_basis_degeneracy_writes_global_stop_without_reduced_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"call_plan_sha256": "12" * 32}
    design = {
        "locks": {
            "analyzer": {"sha256": "34" * 32},
            "runner": {"sha256": "56" * 32},
        }
    }
    manifest = {
        "records": {"file_sha256": "78" * 32},
        "activations": {"file_sha256": "9a" * 32},
    }
    writes: dict[Path, dict[str, Any]] = {}
    monkeypatch.setattr(analyzer, "_load_plan", lambda: (plan, design))
    monkeypatch.setattr(
        analyzer,
        "_validate_phase_manifest",
        lambda *_args, **_kwargs: (
            manifest,
            [],
            np.zeros((1, 1), dtype="<f4"),
            None,
        ),
    )
    monkeypatch.setattr(analyzer, "_validate_baseline_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(analyzer, "_behavioral_inputs", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        analyzer,
        "_behavioral_admission",
        lambda *_args, **_kwargs: {"gates": {"pass": True}},
    )

    def fail_basis(*_args: Any, **_kwargs: Any) -> None:
        raise analyzer.runner.CrossCodebookRunnerError("ineligible residual for content at layer 8")

    monkeypatch.setattr(analyzer.runner, "_fit_basis_with_calculations", fail_basis)
    monkeypatch.setattr(analyzer, "file_sha256", lambda _path: "ab" * 32)
    monkeypatch.setattr(
        analyzer,
        "_write_json",
        lambda path, value: writes.__setitem__(Path(path), copy.deepcopy(dict(value))),
    )
    monkeypatch.setattr(
        analyzer,
        "_write_array",
        lambda *_args, **_kwargs: pytest.fail("invalid basis must not be persisted"),
    )
    monkeypatch.setattr(
        analyzer,
        "_write_f64_array",
        lambda *_args, **_kwargs: pytest.fail("invalid calculations must not be persisted"),
    )

    analysis, lock = analyzer.analyze_fit_basis()

    assert analysis["status"] == "FIT_STOP_BASIS_LOCK_INVALID"
    assert analysis["basis_engineering"]["all_five_grid_layers_required"] is True
    assert lock["engineering_pass"] is False
    assert lock["localization_baseline_authorized"] is False
    assert lock["basis_sidecar"] is None
    assert writes[analyzer.runner.DEFAULT_BASIS_LOCK] == lock


def test_stop_status_registry_includes_localization_baseline_admission_stop() -> None:
    assert "LOCALIZATION_BASELINE_STOP_NOT_ADMITTED" in analyzer.STOP_STATUSES


def test_persisted_target_oriented_margin_is_recomputed() -> None:
    diagnostics = _diagnostics("X")
    row = {
        "native_answer": "X",
        "diagnostics": diagnostics,
        "target_oriented_margin": -2.0,
    }
    assert analyzer._baseline_margin(row) == pytest.approx(-2.0)
    assert analyzer._patch_margin(row) == pytest.approx(-2.0)

    row["target_oriented_margin"] = -1.5
    with pytest.raises(
        analyzer.CrossCodebookAnalysisError, match="target-oriented margin"
    ):
        analyzer._baseline_margin(row)

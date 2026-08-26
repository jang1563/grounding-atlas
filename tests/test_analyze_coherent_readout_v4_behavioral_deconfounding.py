from __future__ import annotations

import copy
import importlib.util
import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from eval import analyze_coherent_readout_v4_behavioral_deconfounding as analyzer

LABEL_TOKEN_IDS = {"P": 10, "Q": 11, "X": 55, "Y": 56}


def _mapping_answer(target_property: str, mapping_id: str) -> str:
    if mapping_id == "identity":
        return "X" if target_property == "P" else "Y"
    return "Y" if target_property == "P" else "X"


def _diagnostics(
    answer_labels: list[str],
    correct_answer: str,
    prediction: str,
    *,
    label_mass: float = 0.99,
) -> dict[str, Any]:
    if prediction == "TIE":
        by_text = {answer_labels[0]: 5.0, answer_labels[1]: 5.0}
        maximum_ids = sorted(LABEL_TOKEN_IDS[label] for label in answer_labels)
        greedy_token_id = maximum_ids[0]
        greedy_logit = 5.0
    elif prediction == "NONLABEL":
        by_text = {
            answer_labels[0]: 5.0 if correct_answer == answer_labels[0] else 0.0,
            answer_labels[1]: 5.0 if correct_answer == answer_labels[1] else 0.0,
        }
        maximum_ids = [999]
        greedy_token_id = 999
        greedy_logit = 6.0
        label_mass = 0.25
    else:
        by_text = {
            answer_labels[0]: 5.0 if prediction == answer_labels[0] else 0.0,
            answer_labels[1]: 5.0 if prediction == answer_labels[1] else 0.0,
        }
        maximum_ids = [LABEL_TOKEN_IDS[prediction]]
        greedy_token_id = maximum_ids[0]
        greedy_logit = 5.0
    logits = [by_text[label] for label in answer_labels]
    label_logsumexp = float(math.log(math.exp(logits[0]) + math.exp(logits[1])))
    full_logsumexp = label_logsumexp - math.log(label_mass)
    if full_logsumexp < greedy_logit:
        raise AssertionError("synthetic diagnostic has an impossible log-sum-exp")
    incorrect = answer_labels[1] if correct_answer == answer_labels[0] else answer_labels[0]
    return {
        "label_logits": logits,
        "label_logit_by_text": by_text,
        "first_minus_second_margin": logits[0] - logits[1],
        "correct_minus_incorrect_margin": by_text[correct_answer] - by_text[incorrect],
        "full_vocab_logsumexp": full_logsumexp,
        "label_probability_mass": label_mass,
        "greedy_token_id": greedy_token_id,
        "greedy_logit": greedy_logit,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": "a" * 64,
    }


def _row(
    *,
    family: str,
    world_id: str,
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
    rule_order: str | None,
    option_order: str,
    semantic_bundle_id: str,
    permutation_index: int,
) -> dict[str, Any]:
    if family == "property_retrieval":
        answer_labels = ["P", "Q"]
        displayed_options = ["P", "Q"] if option_order == "p_then_q" else ["Q", "P"]
        correct_answer = target_property
        first_rule = None
        v3 = None
    else:
        answer_labels = ["X", "Y"]
        displayed_options = ["X", "Y"] if option_order == "x_then_y" else ["Y", "X"]
        assert mapping_id is not None and rule_order is not None
        correct_answer = _mapping_answer(target_property, mapping_id)
        first_property = "P" if rule_order == "p_rule_first" else "Q"
        first_rule = _mapping_answer(first_property, mapping_id)
        v3 = (
            "X"
            if family == "composition"
            and target_property == "P"
            and mapping_id == "identity"
            and target_fact_order == "target_first"
            else "Y"
            if family == "composition"
            else None
        )
    cell_id = (
        f"{family}__{world_id}__p-{target_property}__m-{mapping_id}__f-{target_fact_order}"
        f"__r-{rule_order}__o-{option_order}"
    )
    _, stratum_id = analyzer._expected_registry_ids(
        family=family,
        world_id=world_id,
        target_property=target_property,
        mapping_id=mapping_id,
        target_fact_order=target_fact_order,
        rule_order=rule_order,
        option_order=option_order,
    )
    return {
        "cell_id": cell_id,
        "world_id": world_id,
        "family_id": family,
        "stratum_id": stratum_id,
        "target_property": target_property,
        "mapping_id": mapping_id,
        "target_fact_order": target_fact_order,
        "rule_order": rule_order,
        "option_order": option_order,
        "answer_labels": answer_labels,
        "displayed_options": displayed_options,
        "correct_answer": correct_answer,
        "correct_option_position": "first" if displayed_options[0] == correct_answer else "last",
        "v3_heuristic_answer": v3,
        "last_option_heuristic_answer": displayed_options[-1],
        "first_rule_output_heuristic_answer": first_rule,
        "semantic_bundle_id": semantic_bundle_id,
        "permutation_index": permutation_index,
        "label_token_ids": [LABEL_TOKEN_IDS[label] for label in answer_labels],
        "diagnostics": _diagnostics(answer_labels, correct_answer, correct_answer),
    }


def _perfect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for world_id in analyzer.WORLD_IDS:
        for target_property in ("P", "Q"):
            retrieval_bundle = f"retrieval:{world_id}:p-{target_property.lower()}"
            for permutation_index, (fact_order, option_order) in enumerate(
                itertools.product(
                    ("target_first", "target_second"),
                    ("p_then_q", "q_then_p"),
                )
            ):
                rows.append(
                    _row(
                        family="property_retrieval",
                        world_id=world_id,
                        target_property=target_property,
                        mapping_id=None,
                        target_fact_order=fact_order,
                        rule_order=None,
                        option_order=option_order,
                        semantic_bundle_id=retrieval_bundle,
                        permutation_index=permutation_index,
                    )
                )
            for mapping_id in ("identity", "swapped"):
                lookup_bundle = f"lookup:{world_id}:p-{target_property.lower()}:m-{mapping_id}"
                for permutation_index, (rule_order, option_order) in enumerate(
                    itertools.product(
                        ("p_rule_first", "q_rule_first"),
                        ("x_then_y", "y_then_x"),
                    )
                ):
                    rows.append(
                        _row(
                            family="codebook_lookup",
                            world_id=world_id,
                            target_property=target_property,
                            mapping_id=mapping_id,
                            target_fact_order=None,
                            rule_order=rule_order,
                            option_order=option_order,
                            semantic_bundle_id=lookup_bundle,
                            permutation_index=permutation_index,
                        )
                    )
                composition_bundle = f"composition:{world_id}:p-{target_property.lower()}:m-{mapping_id}"
                for permutation_index, (fact_order, rule_order, option_order) in enumerate(
                    itertools.product(
                        ("target_first", "target_second"),
                        ("p_rule_first", "q_rule_first"),
                        ("x_then_y", "y_then_x"),
                    )
                ):
                    rows.append(
                        _row(
                            family="composition",
                            world_id=world_id,
                            target_property=target_property,
                            mapping_id=mapping_id,
                            target_fact_order=fact_order,
                            rule_order=rule_order,
                            option_order=option_order,
                            semantic_bundle_id=composition_bundle,
                            permutation_index=permutation_index,
                        )
                    )
    assert len(rows) == 448
    return rows


def _set_prediction(row: dict[str, Any], prediction: str, *, label_mass: float = 0.99) -> None:
    row["diagnostics"] = _diagnostics(
        row["answer_labels"],
        row["correct_answer"],
        prediction,
        label_mass=label_mass,
    )


def _first(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    return next(row for row in rows if row["family_id"] == family)


def _fail_one(rows: list[dict[str, Any]], family: str) -> None:
    row = _first(rows, family)
    wrong = row["answer_labels"][1] if row["correct_answer"] == row["answer_labels"][0] else row["answer_labels"][0]
    _set_prediction(row, wrong)


def _set_composition_policy(
    rows: list[dict[str, Any]],
    policy: Callable[[dict[str, Any]], str],
) -> None:
    for row in rows:
        if row["family_id"] == "composition":
            _set_prediction(row, policy(row))


def test_perfect_448_row_registry_qualifies() -> None:
    analysis = analyzer.analyze_records(_perfect_rows())
    assert analysis["status"] == analyzer.STATUS_QUALIFIED
    assert analysis["coverage"]["observed_total_rows"] == 448
    assert analysis["coverage"]["family_counts"] == analyzer.FAMILY_TOTALS
    assert all(analysis["component_gates"].values())
    for family in analyzer.FAMILY_ORDER:
        assert analysis["accuracy"][family]["overall_confirmatory_full_vocab_accuracy"]["rate"] == 1.0
        assert (
            analysis["accuracy"][family]["order_invariance"]["semantic_bundle_all_permutations_correct"]["rate"] == 1.0
        )
        assert analysis["accuracy"][family]["world_cluster_bootstrap"]["lower_95"] == 1.0
    assert analysis["composition_policy_comparison"]["dominant_failure_classification"] == (
        "NO_REGISTERED_HEURISTIC_DOMINANT"
    )
    assert analysis["model_forwards_executed_by_analyzer"] == 0
    assert analysis["terminal_no_downstream_authorization"] is True


@pytest.mark.parametrize(
    ("failed_families", "expected_status"),
    [
        (("property_retrieval",), analyzer.STATUS_RETRIEVAL_FAIL),
        (("codebook_lookup",), analyzer.STATUS_LOOKUP_FAIL),
        (
            ("property_retrieval", "codebook_lookup"),
            analyzer.STATUS_BOTH_COMPONENTS_FAIL,
        ),
        (("composition",), analyzer.STATUS_COMPOSITION_FAIL),
    ],
)
def test_component_status_precedence(
    failed_families: tuple[str, ...],
    expected_status: str,
) -> None:
    rows = _perfect_rows()
    for family in failed_families:
        _fail_one(rows, family)
    analysis = analyzer.analyze_records(rows)
    assert analysis["status"] == expected_status


def test_engineering_failure_has_first_precedence() -> None:
    rows = _perfect_rows()
    _fail_one(rows, "property_retrieval")
    tied = _first(rows, "composition")
    _set_prediction(tied, "TIE")
    analysis = analyzer.analyze_records(rows)
    assert analysis["status"] == analyzer.STATUS_ENGINEERING_INVALID
    assert analysis["component_gates"]["engineering"] is False
    assert analysis["channel_checks"]["by_family"]["composition"]["gates"]["no_global_argmax_ties"] is False


def test_full_vocab_accuracy_is_confirmatory_and_two_label_preference_is_descriptive() -> None:
    rows = _perfect_rows()
    row = _first(rows, "property_retrieval")
    _set_prediction(row, "NONLABEL")
    analysis = analyzer.analyze_records(rows)
    retrieval = analysis["accuracy"]["property_retrieval"]
    assert retrieval["overall_confirmatory_full_vocab_accuracy"]["count"] == 63
    assert retrieval["descriptive_two_label_preference_accuracy"]["count"] == 64
    assert analysis["channel_checks"]["by_family"]["property_retrieval"]["gates"]["pass"] is True
    assert retrieval["order_invariance"]["semantic_bundle_family_label_prediction_invariant"]["count"] == 15
    assert analysis["status"] == analyzer.STATUS_RETRIEVAL_FAIL


@pytest.mark.parametrize(
    ("policy_name", "policy", "expected"),
    [
        ("constant_y", lambda row: "Y", "CONSTANT_Y_HEURISTIC_DOMINANT"),
        ("constant_x", lambda row: "X", "CONSTANT_X_HEURISTIC_DOMINANT"),
        (
            "v3",
            lambda row: row["v3_heuristic_answer"],
            "V3_HEURISTIC_DOMINANT",
        ),
        (
            "last",
            lambda row: row["last_option_heuristic_answer"],
            "LAST_DISPLAYED_OPTION_HEURISTIC_DOMINANT",
        ),
        (
            "first_rule",
            lambda row: row["first_rule_output_heuristic_answer"],
            "FIRST_DISPLAYED_RULE_OUTPUT_HEURISTIC_DOMINANT",
        ),
    ],
)
def test_registered_failure_policy_dominance(
    policy_name: str,
    policy: Callable[[dict[str, Any]], str],
    expected: str,
) -> None:
    del policy_name
    rows = _perfect_rows()
    _set_composition_policy(rows, policy)
    analysis = analyzer.analyze_records(rows)
    assert analysis["component_gates"]["composition"] is False
    comparison = analysis["composition_policy_comparison"]
    assert comparison["dominant_failure_classification"] == expected
    assert comparison["policies"][comparison["selected_dominant_policy"]]["model_match"]["rate"] == 1.0


def test_exact_top_match_tie_returns_multiple_registered_heuristics() -> None:
    rows = _perfect_rows()
    v3_x = [row for row in rows if row["family_id"] == "composition" and row["v3_heuristic_answer"] == "X"]
    assert len(v3_x) == 32
    for row in rows:
        if row["family_id"] == "composition":
            _set_prediction(row, "Y")
    for row in v3_x[:16]:
        _set_prediction(row, "X")
    analysis = analyzer.analyze_records(rows)
    comparison = analysis["composition_policy_comparison"]
    assert comparison["policies"][analyzer.POLICY_V3]["model_match"]["rate"] == 0.9375
    assert comparison["policies"][analyzer.POLICY_CONSTANT_Y]["model_match"]["rate"] == 0.9375
    assert comparison["dominant_failure_classification"] == "MULTIPLE_REGISTERED_HEURISTICS"
    assert comparison["selected_dominant_policy"] is None


def test_heuristic_classification_is_suppressed_when_composition_passes() -> None:
    analysis = analyzer.analyze_records(_perfect_rows())
    comparison = analysis["composition_policy_comparison"]
    assert comparison["qualifying_failure_heuristics"] == []
    assert comparison["selected_dominant_policy"] is None


def test_world_bootstrap_is_deterministic_and_world_clustered() -> None:
    first = analyzer.analyze_records(_perfect_rows())
    second = analyzer.analyze_records(_perfect_rows())
    first_bootstrap = first["accuracy"]["composition"]["world_cluster_bootstrap"]
    second_bootstrap = second["accuracy"]["composition"]["world_cluster_bootstrap"]
    assert first_bootstrap == second_bootstrap
    assert first_bootstrap["draws"] == 10_000
    assert first_bootstrap["seed"] == 260805
    assert first_bootstrap["unit"] == "world_id"


def test_exact_coverage_rejects_missing_duplicate_and_wrong_factorial_rows() -> None:
    rows = _perfect_rows()
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="448"):
        analyzer.analyze_records(rows[:-1])
    duplicated = copy.deepcopy(rows)
    duplicated[-1]["cell_id"] = duplicated[0]["cell_id"]
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="duplicated"):
        analyzer.analyze_records(duplicated)
    wrong = copy.deepcopy(rows)
    target = _first(wrong, "composition")
    target["target_fact_order"] = "unexpected"
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="factor levels"):
        analyzer.analyze_records(wrong)


def test_stored_ledger_and_heuristics_are_independently_reconstructed() -> None:
    cases = (
        ("correct_answer", "Y", "intended ledger"),
        ("last_option_heuristic_answer", "X", "last-option"),
        ("first_rule_output_heuristic_answer", "Y", "first-rule"),
        ("v3_heuristic_answer", "Y", "V3 heuristic"),
    )
    for field, value, message in cases:
        rows = _perfect_rows()
        row = next(
            item
            for item in rows
            if item["family_id"] == "composition"
            and item["target_property"] == "P"
            and item["mapping_id"] == "identity"
            and item["target_fact_order"] == "target_first"
            and item["rule_order"] == "p_rule_first"
            and item["option_order"] == "x_then_y"
        )
        row[field] = value
        with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match=message):
            analyzer.analyze_records(rows)


def test_diagnostic_algebra_tamper_is_rejected() -> None:
    rows = _perfect_rows()
    row = _first(rows, "composition")
    row["diagnostics"]["correct_minus_incorrect_margin"] += 0.5
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="does not reconstruct"):
        analyzer.analyze_records(rows)


def test_diagnostic_adapter_accepts_the_runner_full_vocab_contract() -> None:
    runner = analyzer.runner
    assert runner is not None
    logits = np.full(runner.MODEL_VOCAB_SIZE, -12.0, dtype=np.float32)
    logits[55] = 5.0
    logits[56] = 0.0
    raw = runner.full_vocab_diagnostics(
        logits,
        answer_labels=["X", "Y"],
        label_token_ids=[55, 56],
        correct_answer="X",
    )
    recomputed = analyzer._diagnostics_from_full_vocab(
        logits,
        answer_labels=["X", "Y"],
        label_token_ids=[55, 56],
        correct_answer="X",
    )
    assert recomputed == raw
    normalized = analyzer._normalize_diagnostics(
        raw,
        answer_labels=("X", "Y"),
        label_token_ids={"X": 55, "Y": 56},
        correct_answer="X",
    )
    assert normalized["maximum_token_ids"] == [55]
    assert normalized["correct_minus_incorrect_margin"] == 5.0


def test_artifact_validation_failure_forces_engineering_status() -> None:
    analysis = analyzer.analyze_records(
        _perfect_rows(),
        artifact_validation={"pass": False, "reason": "synthetic failure"},
    )
    assert analysis["status"] == analyzer.STATUS_ENGINEERING_INVALID


def test_claim_boundaries_are_explicit() -> None:
    analysis = analyzer.analyze_records(_perfect_rows())
    boundaries = analysis["claim_boundaries"]
    assert boundaries["causal_mechanism_inference"] == "forbidden"
    assert boundaries["activation_gap_inference"] == "forbidden"
    assert boundaries["latent_knowledge_inference"] == "forbidden"
    assert boundaries["biological_inference"] == "forbidden"
    assert boundaries["physical_law_inference"] == "forbidden"
    assert boundaries["model_family_generalization"] == "forbidden"


def test_frozen_output_writer_is_idempotent_and_refuses_changes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    analyzer._write_frozen_bytes(path, b"first")
    analyzer._write_frozen_bytes(path, b"first")
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="refusing"):
        analyzer._write_frozen_bytes(path, b"second")


def test_frozen_bundle_preflights_all_outputs_before_any_write(tmp_path: Path) -> None:
    absent = tmp_path / "absent.json"
    conflict = tmp_path / "conflict.json"
    conflict.write_bytes(b"old")
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="refusing"):
        analyzer._write_frozen_bundle({absent: b"new", conflict: b"different"})
    assert not absent.exists()
    assert conflict.read_bytes() == b"old"


def test_runner_bundle_records_are_enriched_from_the_validated_cell_registry() -> None:
    cells = _perfect_rows()
    cell_only_fields = {
        "semantic_bundle_id",
        "permutation_index",
        "v3_heuristic_answer",
        "last_option_heuristic_answer",
        "first_rule_output_heuristic_answer",
    }
    records = [{key: value for key, value in row.items() if key not in cell_only_fields} for row in cells]
    enriched = analyzer._enrich_records_from_cells(records, {"cell_registry": cells})
    assert len(enriched) == 448
    assert all(field in enriched[0] for field in cell_only_fields)
    analyzer.analyze_records(enriched)


def test_runner_bundle_enrichment_rejects_record_cell_disagreement() -> None:
    cells = _perfect_rows()
    records = copy.deepcopy(cells)
    records[0]["target_property"] = "Q" if cells[0]["target_property"] == "P" else "P"
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="differs"):
        analyzer._enrich_records_from_cells(records, {"cell_registry": cells})


def test_bundle_adapter_independently_recomputes_every_raw_vocab_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "MODEL_VOCAB_SIZE", 64)
    cells = _perfect_rows()
    frozen_cells = [
        {key: value for key, value in cell.items() if key not in {"diagnostics", "label_token_ids"}} for cell in cells
    ]
    matrix = np.full((448, 64), -12.0, dtype="<f4")
    records = []
    for index, cell in enumerate(cells):
        correct_token = LABEL_TOKEN_IDS[cell["correct_answer"]]
        matrix[index, correct_token] = 5.0
        diagnostics = analyzer._diagnostics_from_full_vocab(
            matrix[index],
            answer_labels=cell["answer_labels"],
            label_token_ids=cell["label_token_ids"],
            correct_answer=cell["correct_answer"],
        )
        records.append(
            {
                **cell,
                "full_vocab_logits_row": index,
                "diagnostics": diagnostics,
            }
        )
    bundle = {
        "records": records,
        "plan_manifest": {"cell_registry": frozen_cells},
        "full_vocab_logits": matrix,
    }
    replayed = analyzer._bundle_records(bundle)
    assert len(replayed) == 448
    assert analyzer.analyze_records(replayed)["status"] == analyzer.STATUS_QUALIFIED

    tampered = copy.deepcopy(records)
    tampered[0]["diagnostics"]["greedy_logit"] -= 1.0
    with pytest.raises(analyzer.BehavioralDeconfoundingAnalysisError, match="independently reconstruct"):
        analyzer._bundle_records({**bundle, "records": tampered})


def test_actual_fixture_builder_cells_replay_through_analyzer_without_writes() -> None:
    builder_path = (
        Path(__file__).resolve().parents[1]
        / "signal"
        / "syntax"
        / "build_coherent_readout_v4_behavioral_deconfounding_bank.py"
    )
    specification = importlib.util.spec_from_file_location("v4_fixture_builder_test", builder_path)
    assert specification is not None and specification.loader is not None
    builder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(builder)
    fixture = builder.build_fixture()
    records = []
    for cell in fixture["cells"]:
        answer_labels = cell["answer_labels"]
        correct_answer = cell["correct_answer"]
        records.append(
            {
                **cell,
                "label_token_ids": [LABEL_TOKEN_IDS[label] for label in answer_labels],
                "diagnostics": _diagnostics(answer_labels, correct_answer, correct_answer),
            }
        )
    analysis = analyzer.analyze_records(records)
    assert analysis["status"] == analyzer.STATUS_QUALIFIED
    assert analysis["normalized_row_registry_canonical_sha256"]

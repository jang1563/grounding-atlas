from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from eval import analyze_coherent_readout_v2_syntax as analyzer
from eval import coherent_binary_readout as coherent
from eval import run_coherent_readout_v2_syntax as runner

VOCAB_SIZE = 64
RowBuilder = Callable[[dict[str, Any], float], np.ndarray]
FROZEN_FIXTURE = runner.validate_fixture(
    json.loads(analyzer.FROZEN_FIXTURE_PATH.read_text(encoding="utf-8"))
)
FIRST_PAIR_ID = FROZEN_FIXTURE["pair_registry"][0]["pair_id"]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair_registry() -> list[dict[str, str]]:
    return [dict(pair) for pair in FROZEN_FIXTURE["pair_registry"]]


def _items() -> list[dict[str, Any]]:
    return [dict(item) for item in FROZEN_FIXTURE["items"]]


def _candidate_registry() -> list[dict[str, Any]]:
    output = []
    for index, definition in enumerate(runner.candidate_definitions()):
        output.append(
            {
                **definition,
                "candidate_definition_sha256": runner.canonical_sha256(definition),
                "x_token_id": 10 + 2 * index,
                "y_token_id": 11 + 2 * index,
            }
        )
    return output


def _environment() -> dict[str, Any]:
    return runner.environment_lock()


def _row_for_score(identity: dict[str, Any], score: float) -> np.ndarray:
    if not -1.0 < score < 1.0:
        raise ValueError("synthetic score must lie strictly inside (-1,1)")
    energy = 2.0 * math.atanh(score)
    row = np.full(VOCAB_SIZE, -20.0, dtype="<f4")
    row[identity["correct_token_id"]] = energy / 2.0
    row[identity["wrong_token_id"]] = -energy / 2.0
    return row


def _build_artifacts(
    *,
    role: str = "syntax_selection",
    row_builder: RowBuilder | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    if row_builder is None:
        row_builder = _row_for_score
    role_lock = runner.FROZEN_ROLE_LOCKS[role]
    candidate_registry = _candidate_registry()
    candidate_by_id = {row["candidate_id"]: row for row in candidate_registry}
    environment = _environment()
    analyzer_sha256 = analyzer.file_sha256(Path(analyzer.__file__))
    runner_sha256 = analyzer.file_sha256(Path(runner.__file__))
    chat_template_sha256 = _sha("synthetic-chat-template")
    records: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for candidate_id in runner.CANDIDATE_ORDER:
        candidate = candidate_by_id[candidate_id]
        for item in _items():
            for order, mapping in runner.FORM_KEYS:
                x_token_id = candidate["x_token_id"]
                y_token_id = candidate["y_token_id"]
                positive_token_id, negative_token_id = (
                    (x_token_id, y_token_id)
                    if mapping == "positive_is_x"
                    else (y_token_id, x_token_id)
                )
                correct_token_id, wrong_token_id = (
                    (positive_token_id, negative_token_id)
                    if item["truth_polarity"] == "positive"
                    else (negative_token_id, positive_token_id)
                )
                uniqueness = f"{candidate_id}|{item['item_id']}|{order}|{mapping}"
                messages = runner.render_candidate_messages(
                    candidate_id, item, order, mapping
                )
                user_content = next(
                    message["content"] for message in messages if message["role"] == "user"
                )
                identity = {
                    "schema_version": runner.RECORD_SCHEMA,
                    "candidate_id": candidate_id,
                    "candidate_rank": candidate["candidate_rank"],
                    "candidate_definition_sha256": candidate[
                        "candidate_definition_sha256"
                    ],
                    "item_id": item["item_id"],
                    "pair_id": item["pair_id"],
                    "cluster_id": item["cluster_id"],
                    "fixture_item_sha256": item["fixture_item_sha256"],
                    "declaration_sha256": item["declaration_sha256"],
                    "positive_class": item["positive_class"],
                    "negative_class": item["negative_class"],
                    "declared_class": item["declared_class"],
                    "truth_polarity": item["truth_polarity"],
                    "order": order,
                    "mapping": mapping,
                    "messages_sha256": runner.canonical_sha256(messages),
                    "user_content_sha256": runner.text_sha256(user_content),
                    "prompt_sha256": _sha(f"prompt|{uniqueness}"),
                    "render_mode": candidate["render_mode"],
                    "add_generation_prompt": candidate["add_generation_prompt"],
                    "continue_final_message": candidate["continue_final_message"],
                    "enable_thinking": candidate["enable_thinking"],
                    "x_answer_text": candidate["x_answer_text"],
                    "y_answer_text": candidate["y_answer_text"],
                    "x_token_id": x_token_id,
                    "y_token_id": y_token_id,
                    "positive_token_id": positive_token_id,
                    "negative_token_id": negative_token_id,
                    "correct_token_id": correct_token_id,
                    "wrong_token_id": wrong_token_id,
                    "execution_input_sha256": _sha(f"input|{uniqueness}"),
                    "input_token_count": 20,
                    "model_role": role,
                    "model_id": role_lock["model_id"],
                    "model_revision": role_lock["model_revision"],
                    "model_weights_sha256": role_lock["model_weights_sha256"],
                    "tokenizer_id": role_lock["model_id"],
                    "tokenizer_revision": role_lock["model_revision"],
                    "chat_template_sha256": chat_template_sha256,
                    "dtype": "float32",
                    "device": "mps",
                    "logits_source": runner.LOGITS_SOURCE,
                    "vocab_size": VOCAB_SIZE,
                    "runner_code_sha256": runner_sha256,
                    "analyzer_code_sha256": analyzer_sha256,
                    "environment_sha256": environment["environment_sha256"],
                    "full_vocab_logits_row": len(records),
                }
                score = 0.80
                row = np.ascontiguousarray(row_builder(identity, score), dtype="<f4")
                diagnostics = coherent.full_vocab_diagnostics(
                    row,
                    x_token_id=x_token_id,
                    y_token_id=y_token_id,
                )
                maximum = float(np.max(row))
                maximum_token_ids = [int(value) for value in np.flatnonzero(row == maximum)]
                record = {
                    "record_id": runner.record_id(identity),
                    **identity,
                    **diagnostics,
                    "maximum_token_ids": maximum_token_ids,
                    "maximum_tie_count": len(maximum_token_ids),
                }
                record["forward_trace_sha256"] = runner.forward_trace_sha256(record)
                assert set(record) == runner.RECORD_KEYS
                records.append(record)
                rows.append(row)
    matrix = np.ascontiguousarray(np.stack(rows), dtype="<f4")
    design = {
        "schema_version": runner.DESIGN_SCHEMA,
        "mode": "development",
        "purpose": "syntax_selection_only",
        "analysis_id": FROZEN_FIXTURE["analysis_id"],
        "model_role": role,
        "selection_eligible_model": role == "syntax_selection",
        "source_fixture_sha256": runner.FROZEN_FIXTURE_SHA256,
        "source_manifest_sha256": runner.FROZEN_FIXTURE_MANIFEST_SHA256,
        "source_fixture_canonical_sha256": FROZEN_FIXTURE["fixture_canonical_sha256"],
        "preregistration_sha256": runner.FROZEN_PREREG_SHA256,
        "expected_runner_code_sha256": runner_sha256,
        "expected_analyzer_code_sha256": analyzer_sha256,
        "expected_environment": environment,
        "candidate_bank_sha256": runner.candidate_bank_sha256(),
        "candidate_registry": candidate_registry,
        "pair_registry": _pair_registry(),
        "expected_item_ids": [item["item_id"] for item in _items()],
        "expected_model_id": role_lock["model_id"],
        "expected_model_revision": role_lock["model_revision"],
        "expected_model_weights_sha256": role_lock["model_weights_sha256"],
        "expected_tokenizer_id": role_lock["model_id"],
        "expected_tokenizer_revision": role_lock["model_revision"],
        "expected_tokenizer_vocab_size": VOCAB_SIZE,
        "expected_chat_template_sha256": chat_template_sha256,
        "expected_dtype": "float32",
        "expected_device": "mps",
        "expected_vocab_size": VOCAB_SIZE,
        "gate_config": runner.default_gate_config(),
        "expected_record_count": 256,
        "expected_record_ids": sorted(record["record_id"] for record in records),
        "call_plan_sha256": runner.call_plan_sha256(records),
        "confirmatory_execution_allowed": False,
        "claim_scope": "syntax_selection_only_no_biology_knowledge_or_activation_claim",
    }
    return design, records, matrix


def test_complete_pass_selects_frozen_priority_and_emits_bindable_projection() -> None:
    design, records, matrix = _build_artifacts()
    result = analyzer.analyze_syntax_selection(records, design, matrix)

    assert result["status"] == "SYNTAX_SELECTION_PASS"
    assert result["selected_candidate_id"] == runner.CANDIDATE_ORDER[0]
    assert result["eligible_ranking_order"] == list(runner.CANDIDATE_ORDER)
    projection = result["selected_candidate_projection"]
    assert projection["candidate_id"] == runner.CANDIDATE_ORDER[0]
    assert projection["source_design_sha256"] == result["design_sha256"]
    assert (
        projection["source_records_canonical_sha256"]
        == result["raw_records_canonical_sha256"]
    )
    assert (
        projection["source_full_vocab_matrix_sha256"]
        == result["full_vocab_matrix_sha256"]
    )
    assert result["selected_candidate_projection_sha256"] == analyzer.canonical_sha256(
        projection
    )
    assert all(candidate["eligible"] for candidate in result["candidates"].values())
    assert analyzer.render_markdown(result) == analyzer.render_markdown(result)


def test_always_x_cannot_false_pass_semantic_correctness() -> None:
    def always_x(identity: dict[str, Any], _score: float) -> np.ndarray:
        row = np.full(VOCAB_SIZE, -20.0, dtype="<f4")
        row[identity["x_token_id"]] = 5.0
        row[identity["y_token_id"]] = -5.0
        return row

    design, records, matrix = _build_artifacts(row_builder=always_x)
    result = analyzer.analyze_syntax_selection(records, design, matrix)

    assert result["status"] == "SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT"
    assert result["selected_candidate_id"] is None
    for candidate in result["candidates"].values():
        assert candidate["native_count"] == 64
        assert candidate["native_correct_count"] == 32
        assert not candidate["eligible"]
        assert "overall_native_correct_below_floor" in candidate["failure_reasons"]


def test_pair_polarity_cancellation_fails_mandatory_d_estimand() -> None:
    def cancelling(identity: dict[str, Any], _score: float) -> np.ndarray:
        positive_first = identity["order"] == "positive_first"
        positive_truth = identity["truth_polarity"] == "positive"
        high = positive_first == positive_truth
        return _row_for_score(identity, 0.74 if high else 0.66)

    design, records, matrix = _build_artifacts(row_builder=cancelling)
    result = analyzer.analyze_syntax_selection(records, design, matrix)

    assert result["status"] == "SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT"
    candidate = result["candidates"][runner.CANDIDATE_ORDER[0]]
    assert candidate["pair_estimands"]["M_O"]["pass"]
    assert not candidate["pair_estimands"]["D_O"]["pass"]
    assert candidate["pair_estimands"]["D_O"]["mean"] > 0.06
    assert "pair_equivalence_failed::D_O" in candidate["failure_reasons"]
    assert candidate["item_range_pass_count"] == 16


def test_tied_full_vocab_argmax_is_valid_but_non_native() -> None:
    target = (
        runner.CANDIDATE_ORDER[0],
        FIRST_PAIR_ID,
        "negative",
        "positive_first",
        "positive_is_x",
    )

    def one_tie(identity: dict[str, Any], score: float) -> np.ndarray:
        row = _row_for_score(identity, score)
        key = (
            identity["candidate_id"],
            identity["pair_id"],
            identity["truth_polarity"],
            identity["order"],
            identity["mapping"],
        )
        if key == target:
            row[0] = row[identity["correct_token_id"]]
        return row

    design, records, matrix = _build_artifacts(row_builder=one_tie)
    result = analyzer.analyze_syntax_selection(records, design, matrix)
    candidate = result["candidates"][runner.CANDIDATE_ORDER[0]]

    assert candidate["tied_maximum_count"] == 1
    assert candidate["native_count"] == 63
    assert candidate["native_correct_count"] == 63
    assert candidate["eligible"]
    assert result["selected_candidate_id"] == runner.CANDIDATE_ORDER[1]


def test_sidecar_tampering_is_rejected_before_scoring() -> None:
    design, records, matrix = _build_artifacts()
    tampered = matrix.copy()
    tampered[0, 0] += np.float32(0.25)

    with pytest.raises(analyzer.SyntaxAnalysisError, match="sidecar"):
        analyzer.analyze_syntax_selection(records, design, tampered)


def test_retained_scalar_must_exactly_match_sidecar_reconstruction() -> None:
    design, records, matrix = _build_artifacts()
    tampered = copy.deepcopy(records)
    tampered[0]["full_vocab_logsumexp"] += 5e-8

    with pytest.raises(analyzer.SyntaxAnalysisError, match="full_vocab_logsumexp"):
        analyzer.analyze_syntax_selection(tampered, design, matrix)


def test_relaxed_threshold_design_is_rejected() -> None:
    design, records, matrix = _build_artifacts()
    relaxed = copy.deepcopy(design)
    relaxed["gate_config"]["overall_native_correct_min_count"] = 0

    with pytest.raises(analyzer.SyntaxAnalysisError, match="gate configuration"):
        analyzer.analyze_syntax_selection(records, relaxed, matrix)


def test_environment_provenance_schema_cannot_be_removed_and_rehashed() -> None:
    design, records, matrix = _build_artifacts()
    tampered = copy.deepcopy(design)
    environment = tampered["expected_environment"]
    environment.pop("execution_source_sha256")
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    environment["environment_sha256"] = runner.canonical_sha256(environment_body)

    with pytest.raises(analyzer.SyntaxAnalysisError, match="expected_environment schema"):
        analyzer.analyze_syntax_selection(records, tampered, matrix)


def test_changed_execution_environment_cannot_be_rehashed_and_replayed() -> None:
    design, records, matrix = _build_artifacts()
    tampered = copy.deepcopy(design)
    environment = tampered["expected_environment"]
    environment["dependencies"]["numpy"] = "replayed-version"
    environment_body = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    environment["environment_sha256"] = runner.canonical_sha256(environment_body)

    with pytest.raises(analyzer.SyntaxAnalysisError, match="analysis environment"):
        analyzer.analyze_syntax_selection(records, tampered, matrix)


def test_frozen_pair_registry_cannot_be_replaced() -> None:
    design, records, matrix = _build_artifacts()
    tampered = copy.deepcopy(design)
    tampered["pair_registry"][0]["positive_class"] = "replacement-class"

    with pytest.raises(analyzer.SyntaxAnalysisError, match="frozen syntax fixture"):
        analyzer.analyze_syntax_selection(records, tampered, matrix)


def test_candidate_mixing_is_rejected_and_ranking_is_global() -> None:
    def rank_second(identity: dict[str, Any], score: float) -> np.ndarray:
        row = _row_for_score(identity, score)
        degraded_forms = {
            ("positive_first", "positive_is_x"),
            ("positive_first", "positive_is_y"),
            ("negative_first", "positive_is_x"),
        }
        if (
            identity["candidate_id"] == runner.CANDIDATE_ORDER[0]
            and identity["pair_id"] == FIRST_PAIR_ID
            and identity["truth_polarity"] == "positive"
            and (identity["order"], identity["mapping"]) in degraded_forms
        ):
            row[0] = 10.0
        return row

    design, records, matrix = _build_artifacts(row_builder=rank_second)
    result = analyzer.analyze_syntax_selection(records, design, matrix)
    assert result["candidates"][runner.CANDIDATE_ORDER[0]]["eligible"]
    assert result["selected_candidate_id"] == runner.CANDIDATE_ORDER[1]
    assert result["selected_candidate_projection"]["candidate_id"] == runner.CANDIDATE_ORDER[1]

    mixed = copy.deepcopy(records)
    mixed[0] = copy.deepcopy(mixed[64])
    with pytest.raises(analyzer.SyntaxAnalysisError, match="record registry"):
        analyzer.analyze_syntax_selection(mixed, design, matrix)


def test_record_order_does_not_change_deterministic_analysis() -> None:
    design, records, matrix = _build_artifacts()
    forward = analyzer.analyze_syntax_selection(records, design, matrix)
    reverse = analyzer.analyze_syntax_selection(reversed(records), design, matrix)

    assert analyzer.canonical_json(forward) == analyzer.canonical_json(reverse)


def test_smoke_role_reports_metrics_without_winner_authority() -> None:
    design, records, matrix = _build_artifacts(role="smoke_only")
    result = analyzer.analyze_syntax_selection(records, design, matrix)

    assert result["status"] == "SYNTAX_SMOKE_ANALYSIS_COMPLETE_NO_SELECTION_AUTHORITY"
    assert result["selection_authorized"] is False
    assert result["selected_candidate_id"] is None
    assert result["selected_candidate_projection"] is None
    assert "eligible_ranking_order" not in result
    for candidate in result["candidates"].values():
        assert candidate["selection_evaluation_performed"] is False
        assert "diagnostic_summary" in candidate
        assert not {"eligible", "eligible_rank", "selected", "ranking_metrics"} & set(
            candidate
        )

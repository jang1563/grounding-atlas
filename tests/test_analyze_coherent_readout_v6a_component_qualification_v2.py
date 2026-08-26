from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from eval import analyze_coherent_readout_v6a_component_qualification_v2 as analyzer
from eval import run_coherent_readout_v6a_component_qualification_v2 as runner


class _CharacterTokenizer:
    is_fast = True
    chat_template = "character-tokenizer-analysis-test-template"

    def apply_chat_template(self, messages, *, tokenize=False, **_flags):
        assert tokenize is False
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user>"
            f"<assistant>{messages[2]['content']}"
        )

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _fixture() -> dict:
    return json.loads(analyzer.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _mock_v1_scientific_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()
    cells = sorted(
        (dict(cell) for cell in fixture["cells"]), key=lambda cell: cell["cell_id"]
    )
    prompts = [
        runner.render_prompt(_CharacterTokenizer(), fixture, cell) for cell in cells
    ]
    v1_plan = {
        "cells": cells,
        "prompts": prompts,
        "call_plan_sha256": "v1-call-plan",
    }
    monkeypatch.setattr(runner, "validate_v1_engineering_stop", lambda: v1_plan)
    monkeypatch.setattr(analyzer, "_validate_v1_engineering_stop", lambda: v1_plan)


def _component_records() -> list[dict]:
    records = []
    for index, cell in enumerate(_fixture()["cells"]):
        records.append(
            {
                "record_id": f"record-{index:03d}",
                "world_id": cell["world_id"],
                "family": cell["family"],
                "factors": cell["factors"],
                "expected_answer": cell["correct_answer"],
                "distractor_answer": cell["distractor_answer"],
                "expected_token_id": 1,
                "distractor_token_id": 2,
                "diagnostics": {
                    "answer_correct": True,
                    "answer_tie": False,
                    "maximum_token_ids": [1],
                },
            }
        )
    return records


def _mark_incorrect(record: dict) -> None:
    record["diagnostics"]["answer_correct"] = False
    record["diagnostics"]["maximum_token_ids"] = [record["distractor_token_id"]]


def test_runner_and_analyzer_independently_rebuild_the_same_plan() -> None:
    fixture = _fixture()
    dependency = {
        "model": {
            "snapshot_revision": "mock-revision",
            "config": {
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "vocab_size": 200_000,
            },
        }
    }

    runner_plan, runner_receipt = runner.build_plan(
        _CharacterTokenizer(), fixture, dependency
    )
    analyzer_plan, analyzer_receipt = analyzer._rebuild_plan_and_receipt(
        _CharacterTokenizer(), fixture, dependency
    )

    assert analyzer_plan == runner_plan
    assert analyzer_receipt == runner_receipt
    assert analyzer_plan["composition_calls"] == 0
    assert len(analyzer_plan["symbol_token_contracts"]) == 32


def test_independent_plan_replay_is_zero_write_and_pre_execution_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "ATTEMPT": tmp_path / "attempt.json",
        "RECORDS": tmp_path / "records.jsonl",
        "EXECUTION_MANIFEST": tmp_path / "execution.json",
        "ANALYSIS": tmp_path / "analysis.json",
        "RAW_LOGIT_ROOT": tmp_path / "raw",
    }
    for name, path in paths.items():
        monkeypatch.setattr(analyzer, name, path)
    plan = {
        "call_plan_sha256": "a" * 64,
        "scientific_registry_sha256": "b" * 64,
    }
    monkeypatch.setattr(analyzer, "_load_and_validate_plan", lambda: {"plan": plan})

    result = analyzer.replay_frozen_plan()

    assert result == {
        "stage": "replay-plan",
        "status": "V2_INDEPENDENT_PLAN_REPLAY_PASS_ZERO_FORWARD",
        "call_plan_sha256": "a" * 64,
        "scientific_registry_sha256": "b" * 64,
        "expected_calls": analyzer.EXPECTED_CALLS,
        "composition_calls": 0,
        "model_calls": 0,
        "generation_used": False,
        "artifacts_written": 0,
    }
    assert list(tmp_path.iterdir()) == []

    paths["ATTEMPT"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        analyzer.V6AQualificationAnalysisError, match="must precede"
    ):
        analyzer.replay_frozen_plan()


def test_analyzer_rejects_any_v1_scientific_registry_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    dependency = {
        "model": {
            "snapshot_revision": "mock-revision",
            "config": {
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "vocab_size": 200_000,
            },
        }
    }
    v1_plan = analyzer._validate_v1_engineering_stop()
    mismatched = copy.deepcopy(v1_plan)
    mismatched["prompts"][0]["execution_attention_mask"][0] = 0
    monkeypatch.setattr(analyzer, "_validate_v1_engineering_stop", lambda: mismatched)

    with pytest.raises(
        analyzer.V6AQualificationAnalysisError,
        match="scientific prompt registry differs",
    ):
        analyzer._rebuild_plan_and_receipt(
            _CharacterTokenizer(), fixture, dependency
        )


def test_analyzer_loader_smoke_receipt_rejects_dependency_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke_receipt = tmp_path / "loader_smoke_receipt.json"
    monkeypatch.setattr(analyzer, "LOADER_SMOKE_RECEIPT", smoke_receipt)
    frozen_dependency_hash = "f" * 64
    receipt = {
        "status": "V2_LOADER_SMOKE_PASS_ZERO_FORWARD",
        "execution_revision": "v2_mps_allocator_warmup_bypass",
        "model_id": analyzer.MODEL_ID,
        "model_revision": analyzer.MODEL_REVISION,
        "mps_allocator_warmup_policy": analyzer.MPS_ALLOCATOR_WARMUP_POLICY,
        "warmup_bypass_calls": 1,
        "model_calls": 0,
        "generation_used": False,
        "fixture_prompt_records_consumed": 0,
        "composition_calls": 0,
        "dependency_canonical_sha256": frozen_dependency_hash,
    }
    smoke_receipt.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert analyzer._validate_loader_smoke_receipt(
        {"canonical_sha256": frozen_dependency_hash}
    ) == receipt
    with pytest.raises(
        analyzer.V6AQualificationAnalysisError, match="receipt changed"
    ):
        analyzer._validate_loader_smoke_receipt({"canonical_sha256": "0" * 64})


def test_analyzer_independently_locks_v1_model_and_shared_scientific_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = {"model_id": analyzer.MODEL_ID, "assets": {"weights": "locked"}}
    crosswalk = {
        "builder": "builder",
        "fixture": "fixture",
        "fixture_manifest": "fixture_manifest",
        "v1_preregistration": "preregistration",
        "v6a_topology_design": "v6a_topology_design",
        "builder_test": "builder_test",
    }
    v1_implementations = {
        v1_name: {"sha256": f"hash-{v1_name}"}
        for v1_name in crosswalk.values()
    }
    v2_implementations = {
        v2_name: {"sha256": v1_implementations[v1_name]["sha256"]}
        for v2_name, v1_name in crosswalk.items()
    }
    monkeypatch.setattr(
        analyzer, "_validate_v1_engineering_stop_hashes_only", lambda: None
    )
    monkeypatch.setattr(
        analyzer,
        "_load_json",
        lambda path: {
            "model": copy.deepcopy(model),
            "implementation_files": copy.deepcopy(v1_implementations),
        }
        if path == analyzer.V1_DEPENDENCY_LOCK
        else (_ for _ in ()).throw(AssertionError(path)),
    )

    analyzer._validate_v1_dependency_identity(model, v2_implementations)
    with pytest.raises(analyzer.V6AQualificationAnalysisError, match="model assets"):
        analyzer._validate_v1_dependency_identity(
            {**model, "assets": {"weights": "drifted"}}, v2_implementations
        )
    drifted_implementations = copy.deepcopy(v2_implementations)
    drifted_implementations["v6a_topology_design"]["sha256"] = "drifted"
    with pytest.raises(
        analyzer.V6AQualificationAnalysisError,
        match="shared scientific dependency",
    ):
        analyzer._validate_v1_dependency_identity(model, drifted_implementations)


def test_all_correct_component_panel_passes_every_registered_gate() -> None:
    summary = analyzer.component_summary(_component_records())

    assert summary["gates"]["pass"] is True
    assert summary["family_accuracy"]["property_retrieval"] == {
        "correct": 256,
        "n": 256,
        "accuracy": 1.0,
    }
    assert summary["family_accuracy"]["codebook_lookup"]["correct"] == 128
    assert summary["unique_label_argmax_count"] == 384
    assert summary["exact_integer_requirements"]["effective_codebook_lookup_overall"] == "128_of_128"


def test_lookup_label_gate_forces_effective_128_of_128() -> None:
    records = _component_records()
    lookup = next(record for record in records if record["family"] == "codebook_lookup")
    _mark_incorrect(lookup)

    summary = analyzer.component_summary(records)

    assert summary["family_accuracy"]["codebook_lookup"]["accuracy"] == 127 / 128
    assert summary["gates"]["codebook_lookup_overall_accuracy"] is True
    assert summary["gates"]["every_world_accuracy"] is True
    assert summary["gates"]["every_answer_label_accuracy"] is False
    assert summary["gates"]["pass"] is False


def test_order_by_scaffold_gate_catches_concentrated_retrieval_errors() -> None:
    records = _component_records()
    candidates = [
        record
        for record in records
        if record["family"] == "property_retrieval"
        and record["factors"]["o"] == -1
        and record["factors"]["q"] == -1
        and record["factors"]["a"] == -1
    ]
    selected = []
    used_worlds = set()
    used_answers = set()
    for record in candidates:
        if (
            record["world_id"] not in used_worlds
            and record["expected_answer"] not in used_answers
        ):
            selected.append(record)
            used_worlds.add(record["world_id"])
            used_answers.add(record["expected_answer"])
        if len(selected) == 4:
            break
    assert len(selected) == 4
    for record in selected:
        _mark_incorrect(record)

    summary = analyzer.component_summary(records)

    assert summary["family_accuracy"]["property_retrieval"]["accuracy"] == 252 / 256
    assert summary["gates"]["property_retrieval_overall_accuracy"] is True
    assert summary["gates"]["every_world_accuracy"] is True
    assert summary["gates"]["every_answer_label_accuracy"] is True
    assert summary["gates"]["every_retrieval_q_by_a_accuracy"] is True
    assert summary["gates"]["every_retrieval_o_by_q_by_a_accuracy"] is False
    assert summary["gates"]["pass"] is False


def test_unique_global_argmax_and_answer_tie_are_independent_all_row_gates() -> None:
    records = _component_records()
    outside_pair = copy.deepcopy(records)
    outside_pair[0]["diagnostics"]["maximum_token_ids"] = [99]
    outside_summary = analyzer.component_summary(outside_pair)
    assert outside_summary["gates"]["every_row_unique_global_argmax_in_registered_pair"] is False
    assert outside_summary["gates"]["every_row_no_answer_tie"] is True

    tied = copy.deepcopy(records)
    tied[0]["diagnostics"]["answer_tie"] = True
    tied_summary = analyzer.component_summary(tied)
    assert tied_summary["gates"]["every_row_unique_global_argmax_in_registered_pair"] is True
    assert tied_summary["gates"]["every_row_no_answer_tie"] is False


def test_full_vocab_diagnostics_reconstruct_pairwise_and_global_outputs() -> None:
    row = np.asarray([-1.0, 3.0, 1.0, 2.0], dtype=np.float32)
    record = {
        "expected_token_id": 1,
        "distractor_token_id": 2,
        "expected_answer": "α",
        "distractor_answer": "β",
    }

    diagnostics = analyzer.diagnostics_from_full_vocab(row, record)

    assert diagnostics["answer_correct"] is True
    assert diagnostics["expected_minus_distractor_margin"] == 2.0
    assert diagnostics["greedy_token_id"] == 1
    assert diagnostics["maximum_token_ids"] == [1]
    assert diagnostics["full_vocab_logits_sha256"] == analyzer.f32_sha256(row)

    row[0] = np.nan
    with pytest.raises(analyzer.V6AQualificationAnalysisError, match="invalid"):
        analyzer.diagnostics_from_full_vocab(row, record)


def test_unexpected_malformed_artifact_error_becomes_engineering_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer, "ANALYSIS", tmp_path / "analysis.json")

    def malformed_plan():
        raise ValueError("non-finite JSON constant")

    monkeypatch.setattr(analyzer, "_load_and_validate_plan", malformed_plan)

    result = analyzer.analyze_qualification()

    assert result["status"] == analyzer.ENGINEERING_INVALID
    assert result["engineering_valid"] is False
    assert result["component_qualified"] is False
    assert result["model_calls_issued_by_analyzer"] == 0


def test_raw_shard_registry_rejects_extra_unbound_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setattr(analyzer, "RAW_LOGIT_ROOT", raw_root)
    monkeypatch.setattr(analyzer, "EXPECTED_CALLS", 1)
    monkeypatch.setattr(analyzer, "RAW_SHARD_ROWS", 1)
    array = np.asarray([[0.0, 1.0, 2.0]], dtype="<f4")
    path = raw_root / "shard_000.npy"
    np.save(path, array, allow_pickle=False)
    spec = {
        "index": 0,
        "start_row": 0,
        "stop_row": 1,
        "rows": 1,
        "shape": [1, 3],
        "dtype": "<f4",
        "path": str(path),
    }
    binding = {
        **spec,
        "file_sha256": analyzer.file_sha256(path),
        "logical_sha256": analyzer._logical_array_sha256(array),
        "size_bytes": path.stat().st_size,
    }
    manifest = {"raw_logits_shards": [binding]}
    plan = {"raw_logits_shards": [spec]}

    assert len(analyzer._load_raw_shards(manifest, plan)) == 1
    (raw_root / "unbound.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(analyzer.V6AQualificationAnalysisError, match="unbound"):
        analyzer._load_raw_shards(manifest, plan)


def test_frozen_analysis_write_is_idempotent_but_not_overwritable(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    analyzer._write_frozen_json(path, {"status": "one"})
    analyzer._write_frozen_json(path, {"status": "one"})
    with pytest.raises(analyzer.V6AQualificationAnalysisError, match="overwrite"):
        analyzer._write_frozen_json(path, {"status": "two"})

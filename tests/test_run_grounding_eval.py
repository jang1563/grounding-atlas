import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_grounding_eval as rge  # noqa: E402


@pytest.mark.parametrize(
    ("text", "probability", "valid"),
    [
        ("0.42", 0.42, True),
        (".75", 0.75, True),
        ("Probability: 75", 0.5, False),
        ("75%", 0.5, False),
        ("0.8 because of 2 reasons", 0.5, False),
        ("I cannot answer; confidence 75", 0.5, False),
        ("1.2", 0.5, False),
        ("-0.20", 0.5, False),
        ("I cannot determine this.", 0.5, False),
        ("", 0.5, False),
    ],
)
def test_probability_parser_reports_invalid_responses(text, probability, valid):
    parsed, ok = rge.parse_prob_with_status(text)
    assert parsed == pytest.approx(probability)
    assert ok is valid


def test_score_task_reports_proper_scores_and_paired_delta():
    y = np.asarray([0, 0, 1, 1])
    matched = np.asarray([0.1, 0.2, 0.8, 0.9])
    scrambled = np.asarray([0.8, 0.7, 0.3, 0.2])
    parsed = np.asarray([True, False, True, True])
    record = rge.score_task(
        matched,
        y,
        scrambled,
        y,
        reference=0.95,
        rng=np.random.default_rng(0),
        parsed=parsed,
        scr_parsed=np.ones(4, dtype=bool),
        reference_comparability="same_entities_same_split",
    )

    assert record["output_auroc"] == 1.0
    assert record["brier"] < 0.05
    assert record["log_loss"] < 0.3
    assert record["valid_response_rate"] == 0.75
    assert record["n_invalid"] == 1
    assert record["reference_score"] == 0.95
    assert record["reference_gap"] == pytest.approx(-0.05)
    assert record["memo_n"] == 4
    assert record["memo_delta"] == 1.0
    assert record["memo_delta_ci"][0] == pytest.approx(1.0)


def test_score_task_rejects_unpaired_scrambled_predictions():
    y = np.asarray([0, 0, 1, 1])
    with pytest.raises(ValueError, match="explicitly paired"):
        rge.score_task(
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            y,
            np.asarray([0.2, 0.8]),
            np.asarray([0, 1]),
            reference=None,
            rng=np.random.default_rng(0),
        )


def test_selective_metrics_are_invariant_to_confidence_tie_order():
    y = np.asarray([0] * 96 + [1] * 4)
    probability = np.full(len(y), 0.5)
    permutation = np.random.default_rng(4).permutation(len(y))

    assert rge.aurc(probability, y) == pytest.approx(
        rge.aurc(probability[permutation], y[permutation])
    )
    assert rge.sel_acc(probability, y) == pytest.approx(
        rge.sel_acc(probability[permutation], y[permutation])
    )
    assert rge.sel_acc(probability, y) == pytest.approx(0.96)


def test_dry_evaluation_writes_v5_contract_without_touching_legacy_results(monkeypatch, tmp_path):
    monkeypatch.setattr(rge, "OUT", str(tmp_path))
    scorecard = rge.evaluate(
        model="contract-test",
        tasks=["admet/herg", "herg/graph"],
        n=40,
        seed=0,
        dry=True,
        merge=False,
    )
    run_dir = tmp_path / f"contract-test__{rge.PROMPT_VERSION}"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    raw = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines()]

    assert set(scorecard) == {"admet/herg", "herg/graph"}
    assert manifest["prompt_version"] == "v5"
    assert manifest["score_schema"] == rge.SCORE_SCHEMA
    assert manifest["score_schema"] == 5
    assert manifest["metadata_contract_version"] == "phase0-v2"
    assert manifest["truth_taxonomy_version"] == "groundbench-t0-t5-v1"
    assert manifest["task_truth_metadata"] == {
        task_id: {
            "truth_level_code": "T2",
            "target_source_kind": "operational_assay_aggregation",
        }
        for task_id in scorecard
    }
    assert manifest["sampling_contract"].startswith("shared entity intersection")
    assert all(
        {
            "entity_id",
            "source_label",
            "target_label",
            "truth_level_code",
            "target_source_kind",
            "truth_level",
            "biological_question_id",
            "task_family_id",
            "split_group_id",
            "split_group_scope",
            "intervention_pair_id",
            "factor_levels",
        } <= row.keys()
        for row in raw
    )
    assert all(row["truth_level_code"] == "T2" for row in raw)
    assert all(row["target_source_kind"] == "operational_assay_aggregation" for row in raw)
    assert all(row["truth_level"] == row["truth_level_code"] for row in raw)
    assert all(
        row["factor_levels"]["causal_status"] == "descriptive_interface_only"
        for row in raw
    )
    assert all(row["intervention_pair_id"] is None for row in raw)
    matched = {
        task: [row["entity_id"] for row in raw if row["task"] == task and row["condition"] == "matched"]
        for task in scorecard
    }
    assert matched["admet/herg"] == matched["herg/graph"]
    scrambled = [row for row in raw if row["condition"] == "scrambled"]
    assert scrambled
    assert {row["entity_id"] for row in scrambled}.issubset(set(matched["admet/herg"]))


def test_task_scores_and_rows_do_not_depend_on_execution_order(monkeypatch, tmp_path):
    first_out = tmp_path / "first"
    monkeypatch.setattr(rge, "OUT", str(first_out))
    first = rge.evaluate(
        model="order-test",
        tasks=["methyl/age"],
        n=40,
        seed=11,
        dry=True,
        merge=False,
    )
    first_raw = (first_out / f"order-test__{rge.PROMPT_VERSION}" / "raw.jsonl").read_text()

    second_out = tmp_path / "second"
    monkeypatch.setattr(rge, "OUT", str(second_out))
    second = rge.evaluate(
        model="order-test",
        tasks=["admet/cyp3a4", "methyl/age"],
        n=40,
        seed=11,
        dry=True,
        merge=False,
    )
    second_rows = [
        line
        for line in (
            second_out / f"order-test__{rge.PROMPT_VERSION}" / "raw.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["task"] == "methyl/age"
    ]

    assert first["methyl/age"] == second["methyl/age"]
    assert first_raw.splitlines() == second_rows


def test_incremental_merge_rejects_a_changed_run_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(rge, "OUT", str(tmp_path))
    rge.evaluate(
        model="merge-test",
        tasks=["methyl/age"],
        n=20,
        seed=0,
        dry=True,
        merge=True,
    )

    with pytest.raises(ValueError, match="different contracts"):
        rge.evaluate(
            model="merge-test",
            tasks=["msa/conservation"],
            n=40,
            seed=0,
            dry=True,
            merge=True,
        )

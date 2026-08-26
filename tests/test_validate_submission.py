import json
from pathlib import Path

from eval import run_grounding_eval as rge
from eval import validate_submission


def _dry_admet_submission(monkeypatch, tmp_path):
    reference_source = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "benchmark"
        / "ceilings.json"
    )
    (tmp_path / "ceilings.json").write_text(reference_source.read_text())
    monkeypatch.setattr(rge, "OUT", str(tmp_path))
    rge.evaluate(
        model="validator-test",
        tasks=["admet/herg"],
        n=40,
        seed=3,
        dry=True,
        merge=False,
        provider="test-provider",
        model_revision="test-provider/revision-1",
    )
    run_dir = Path(tmp_path) / f"validator-test__{rge.PROMPT_VERSION}"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Exercise structural validation of an otherwise release-shaped fixture.
    manifest["dry_run"] = False
    manifest["working_tree_clean"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))
    checkout_state = {
        "data_commit": manifest["data_commit"],
        "working_tree_clean": True,
        "code_data_fingerprint": manifest["code_data_fingerprint"],
    }
    monkeypatch.setattr(
        validate_submission,
        "_repository_state",
        lambda: checkout_state,
    )
    return run_dir


def test_validator_accepts_paired_renotation_and_scrambling(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)

    errors, warnings = validate_submission.validate(run_dir, allow_partial=True)

    assert errors == []
    assert any("missing" in warning and "CORE" in warning for warning in warnings)


def test_validator_recomputes_renotation_delta(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    scorecard_path = run_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text())
    scorecard["admet/herg"]["matched_minus_re_notation_auroc"] = 0.999
    scorecard_path.write_text(json.dumps(scorecard, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("matched_minus_re_notation_auroc disagrees" in error for error in errors)


def test_validator_rejects_control_label_tampering(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    raw_path = run_dir / "raw.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    control = next(row for row in rows if row["condition"] == "re_notation")
    control["source_label"] = 1 - control["source_label"]
    control["target_label"] = 1 - control["target_label"]
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("labels do not match paired matched rows" in error for error in errors)


def test_validator_rejects_omitted_registered_control_row(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    raw_path = run_dir / "raw.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    omitted = next(
        index for index, row in enumerate(rows)
        if row["condition"] == "re_notation"
    )
    rows.pop(omitted)
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("entity IDs do not match the registered sampler" in error for error in errors)


def test_validator_rejects_task_metadata_drift(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    scorecard_path = run_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text())
    scorecard["admet/herg"]["task_status"] = "active"
    scorecard["admet/herg"]["intervention_pair_field"] = "fabricated_pair"
    scorecard_path.write_text(json.dumps(scorecard, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("task_status disagrees with the task registry" in error for error in errors)
    assert any(
        "intervention_pair_field disagrees with the task registry" in error
        for error in errors
    )


def test_validator_rejects_descriptive_string_as_truth_code(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    scorecard_path = run_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text())
    scorecard["admet/herg"]["truth_level_code"] = (
        scorecard["admet/herg"]["target_source_kind"]
    )
    scorecard["admet/herg"]["truth_level"] = (
        scorecard["admet/herg"]["target_source_kind"]
    )
    scorecard_path.write_text(json.dumps(scorecard, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("truth_level_code" in error and "outside T0-T5" in error for error in errors)
    assert any("truth_level_code disagrees with the task registry" in error for error in errors)


def test_validator_rejects_raw_truth_metadata_and_manifest_drift(
    monkeypatch,
    tmp_path,
):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    raw_path = run_dir / "raw.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["target_source_kind"] = "database_assertion"
    rows[0]["truth_level"] = "T4"
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task_truth_metadata"]["admet/herg"]["truth_level_code"] = "T5"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("task_truth_metadata does not exactly match" in error for error in errors)
    assert any("raw deprecated truth_level alias" in error for error in errors)
    assert any("raw target_source_kind disagrees" in error for error in errors)


def test_validator_rejects_causalized_factor_levels_and_fake_split_group(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    raw_path = run_dir / "raw.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["factor_levels"]["causal_status"] = "causal_assignment"
    rows[0]["split_group_id"] = "fabricated:scaffold"
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("descriptive interface contract" in error for error in errors)
    assert any("proxy split_group_id must equal entity_id" in error for error in errors)


def test_validator_rejects_score_schema_and_decode_drift(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["score_schema"] = rge.SCORE_SCHEMA - 1
    manifest["decode"]["max_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("score_schema" in error for error in errors)
    assert any("fixed decode" in error for error in errors)


def test_validator_recomputes_provenance_against_checkout(monkeypatch, tmp_path):
    run_dir = _dry_admet_submission(monkeypatch, tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "0" * 64
    manifest["code_data_fingerprint"] = "1" * 64
    task_id = next(iter(manifest["task_data_sha256"]))
    manifest["task_data_sha256"][task_id] = "2" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("dataset_sha256 does not match" in error for error in errors)
    assert any("code_data_fingerprint does not match" in error for error in errors)
    assert any("task_data_sha256 values do not match" in error for error in errors)


def test_validator_recomputes_pair_group_artifact(monkeypatch, tmp_path):
    reference_source = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "benchmark"
        / "ceilings.json"
    )
    (tmp_path / "ceilings.json").write_text(reference_source.read_text())
    monkeypatch.setattr(rge, "OUT", str(tmp_path))
    rge.evaluate(
        model="validator-pair-test",
        tasks=["single_cell/cd8t_nk:name", "single_cell/cd8t_nk:anon"],
        n=40,
        seed=4,
        dry=True,
        merge=False,
        provider="test-provider",
        model_revision="test-provider/revision-1",
    )
    run_dir = Path(tmp_path) / f"validator-pair-test__{rge.PROMPT_VERSION}"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dry_run"] = False
    manifest["working_tree_clean"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))
    checkout_state = {
        "data_commit": manifest["data_commit"],
        "working_tree_clean": True,
        "code_data_fingerprint": manifest["code_data_fingerprint"],
    }
    monkeypatch.setattr(
        validate_submission,
        "_repository_state",
        lambda: checkout_state,
    )
    pair_path = run_dir / "pair_group_comparisons.json"
    pair_artifact = json.loads(pair_path.read_text())
    pair_artifact["comparisons"][0]["task_a_minus_task_b_auroc"] = 0.999
    pair_path.write_text(json.dumps(pair_artifact, indent=2))

    errors, _ = validate_submission.validate(run_dir, allow_partial=True)

    assert any("pair_group_comparisons.json disagrees" in error for error in errors)

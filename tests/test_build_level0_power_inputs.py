from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from eval import build_level0_power_inputs as bridge
from eval import coherent_binary_readout as coherent
from eval import phase0_power as phase0

DONORS = [f"d{index:02d}" for index in range(8)]
READOUTS = ["lineage", "state"]
FAMILIES = ["target_depleted", "unmodified"]
N_GROUPS = len(READOUTS) * len(FAMILIES)
N_ITEMS = len(DONORS) * N_GROUPS
N_RECORDS = N_ITEMS * 4


def _passed_result() -> dict:
    expected_record_ids = sorted(coherent.text_sha256(f"record-{index:04d}") for index in range(N_RECORDS))
    design = coherent.default_design(
        mode="development",
        required_readouts=READOUTS,
        readout_classes={
            "lineage": {"positive_class": "NK", "negative_class": "CD8 T"},
            "state": {
                "positive_class": "cytotoxic-high",
                "negative_class": "cytotoxic-low",
            },
        },
        required_input_families=FAMILIES,
        source_fixture_sha256=coherent.text_sha256("source fixture"),
        source_manifest_sha256=coherent.text_sha256("source manifest"),
        preregistration_sha256=coherent.text_sha256("preregistration"),
        runner_code_sha256=coherent.text_sha256("runner code"),
        call_plan_sha256=coherent.canonical_sha256(expected_record_ids),
        margin_lock_sha256=coherent.text_sha256("margin lock"),
        margin_lock_status="phase0_qualified",
        model_id="open-model/test-model",
        model_revision="open-model-revision",
        tokenizer_id="open-model/test-tokenizer",
        tokenizer_revision="tokenizer-revision",
        chat_template_sha256="a" * 64,
        dtype="float32",
        x_token_id=101,
        y_token_id=102,
        vocab_size=1000,
        expected_donor_ids=DONORS,
        expected_source_items=[{"source_item_id": f"source-{donor}", "donor_id": donor} for donor in DONORS],
        expected_record_ids=expected_record_ids,
    )
    groups: dict[str, dict] = {}
    for group_index, (readout, family) in enumerate((readout, family) for readout in READOUTS for family in FAMILIES):
        donor_effects: dict[str, dict[str, float]] = {}
        for donor_index, donor in enumerate(DONORS):
            centered = donor_index - (len(DONORS) - 1) / 2
            donor_effects[donor] = {
                estimand: centered * 0.0001 * (group_index * 3 + effect_index + 1)
                for effect_index, estimand in enumerate(("O", "R", "I"))
            }
        equivalence = {
            estimand: coherent.equivalence_summary(
                [donor_effects[donor][estimand] for donor in DONORS],
                margin=0.06,
                alpha=0.05,
            )
            for estimand in ("O", "R", "I")
        }
        groups[f"{readout}::{family}"] = {
            "readout_id": readout,
            "input_family": family,
            "n_donors": len(DONORS),
            "n_items": len(DONORS),
            "n_records": len(DONORS) * 4,
            "extraction_coherence": {"max_residual": 0.0, "pass": True},
            "format_adherence": {"fraction": 1.0, "pass": True},
            "nuisance_equivalence": equivalence,
            "item_guardrail": {"range_pass_fraction": 1.0, "pass": True},
            "donor_effects": donor_effects,
            "pass": True,
        }
    return {
        "artifact_type": "groundbench.level0_coherent_binary_readout",
        "schema_version": coherent.ANALYSIS_SCHEMA,
        "status": "DEVELOPMENT_LEVEL0_PASS_NOT_CONFIRMATORY",
        "mode": "development",
        "margin_lock_status": "phase0_qualified",
        "level0_pass": True,
        "design": design,
        "design_sha256": coherent.canonical_sha256(design),
        "raw_records_sha256": "b" * 64,
        "full_vocab_sidecar_sha256": "f" * 64,
        "analysis_code_sha256": "c" * 64,
        "n_records": N_RECORDS,
        "n_items": N_ITEMS,
        "n_donors": len(DONORS),
        "donor_ids": DONORS,
        "validation": {
            "expected_records": N_RECORDS,
            "observed_records": N_RECORDS,
            "missing_records": [],
            "unexpected_records": [],
            "finite_logits": True,
            "max_complement_error": 0.0,
        },
        "global_format_adherence": {
            "adherent_records": N_RECORDS,
            "record_count": N_RECORDS,
            "fraction": 1.0,
            "required": 0.95,
            "pass": True,
        },
        "groups": groups,
        "claim_boundary": "Development Level 0 only.",
    }


def _write_result(path: Path, result: dict, *, indent: int = 2) -> None:
    path.write_text(json.dumps(result, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def test_bridge_derives_every_donor_first_manifest_component_and_authenticates_source(
    tmp_path: Path,
) -> None:
    result = _passed_result()
    source = tmp_path / "level0.json"
    _write_result(source, result)
    matrix, config = bridge.build_from_result_file(source)

    expected_components = sorted(
        f"level0::{readout}::{family}::{estimand}"
        for readout in READOUTS
        for family in FAMILIES
        for estimand in ("O", "R", "I")
    )
    assert matrix["component_ids"] == expected_components
    assert len(expected_components) == N_GROUPS * 3
    assert matrix["source_artifact_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert [donor["donor_id"] for donor in matrix["donors"]] == DONORS

    donor = "d07"
    component = "level0::state::unmodified::R"
    expected = result["groups"]["state::unmodified"]["donor_effects"][donor]["R"]
    observed = next(row for row in matrix["donors"] if row["donor_id"] == donor)["values"][component]
    assert observed == expected
    assert phase0.validate_development_matrix(matrix) == matrix
    assert phase0.validate_power_config(config, matrix) == config


def test_level0_only_config_is_sorted_frozen_and_never_claims_final_n_conf() -> None:
    matrix, config = bridge.build_level0_power_inputs(_passed_result(), source_result_sha256="d" * 64)
    assert config["power_scope"] == "level0_only"
    assert config["seed"] == bridge.FROZEN_SEED == 20_260_802
    assert config["simulation_replicates"] == 10_000
    assert config["candidate_n"] == list(range(12, 21))
    assert config["development_matrix_sha256"] == phase0.canonical_sha256(matrix)
    assert [component["component_id"] for component in config["components"]] == sorted(matrix["component_ids"])
    assert all(
        component["test"] == "equivalence"
        and component["margin"] == 0.06
        and component["alternative_mean"] == 0.0
        and component["support_lower"] == -2.0
        and component["support_upper"] == 2.0
        and component["boundary"] is None
        and component["require_recurrence"] is False
        for component in config["components"]
    )
    assert config["scenarios"] == [
        {
            "scenario_id": bridge.LEVEL0_ONLY_SCENARIO,
            "component_ids": matrix["component_ids"],
            "required": True,
        }
    ]
    rendered = phase0.canonical_json({"matrix": matrix, "config": config})
    assert "selected_n_conf" not in rendered
    assert "candidate_n_selected" not in rendered

    candidate = _passed_result()
    candidate["status"] = "DEVELOPMENT_LEVEL0_CANDIDATE_PASS_MARGIN_NOT_QUALIFIED"
    candidate["margin_lock_status"] = "candidate_unqualified"
    candidate["design"]["margin_lock_status"] = "candidate_unqualified"
    candidate["design_sha256"] = coherent.canonical_sha256(candidate["design"])
    candidate_matrix, candidate_config = bridge.build_level0_power_inputs(
        candidate, source_result_sha256="e" * 64
    )
    assert candidate_config["power_scope"] == "level0_only"
    assert candidate_matrix["component_ids"] == matrix["component_ids"]


def test_output_files_are_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "level0.json"
    _write_result(source, _passed_result())
    first_matrix = tmp_path / "first" / "matrix.json"
    first_config = tmp_path / "first" / "config.json"
    second_matrix = tmp_path / "second" / "matrix.json"
    second_config = tmp_path / "second" / "config.json"
    bridge.write_power_inputs(
        source_result=source,
        output_matrix=first_matrix,
        output_config=first_config,
    )
    bridge.write_power_inputs(
        source_result=source,
        output_matrix=second_matrix,
        output_config=second_config,
    )
    assert first_matrix.read_bytes() == second_matrix.read_bytes()
    assert first_config.read_bytes() == second_config.read_bytes()
    assert first_matrix.read_bytes().endswith(b"\n")
    assert first_config.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result.update(level0_pass=False), "did not pass"),
        (
            lambda result: result.update(status="DEVELOPMENT_LEVEL0_FAIL"),
            "passing status",
        ),
        (lambda result: result.update(mode="confirmatory"), "development-mode"),
    ],
)
def test_failed_or_confirmatory_artifacts_are_refused(mutation, message: str) -> None:
    result = _passed_result()
    mutation(result)
    with pytest.raises(bridge.Level0PowerInputError, match=message):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)


def test_tampered_design_group_and_donor_effects_are_refused() -> None:
    result = _passed_result()
    result["design_sha256"] = "e" * 64
    with pytest.raises(bridge.Level0PowerInputError, match="design SHA-256"):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)

    result = _passed_result()
    result["groups"]["lineage::unmodified"]["pass"] = False
    with pytest.raises(bridge.Level0PowerInputError, match="did not pass"):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)

    result = _passed_result()
    result["groups"]["lineage::unmodified"]["donor_effects"]["d00"]["O"] += 0.01
    with pytest.raises(bridge.Level0PowerInputError, match="does not match donor effects"):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)


def test_nonfrozen_margin_is_refused_even_with_an_updated_design_hash() -> None:
    result = _passed_result()
    result["design"]["equivalence_margin"] = 0.05
    result["design_sha256"] = coherent.canonical_sha256(result["design"])
    with pytest.raises(bridge.Level0PowerInputError, match="not frozen at 0.06"):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)


def test_exact_source_bytes_are_bound_into_both_outputs(tmp_path: Path) -> None:
    result = _passed_result()
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    _write_result(compact, result, indent=1)
    _write_result(pretty, result, indent=4)
    compact_matrix, compact_config = bridge.build_from_result_file(compact)
    pretty_matrix, pretty_config = bridge.build_from_result_file(pretty)
    assert compact_matrix["source_artifact_sha256"] != pretty_matrix["source_artifact_sha256"]
    assert compact_config["development_matrix_sha256"] != pretty_config["development_matrix_sha256"]


def test_cli_emits_only_input_hashes_and_rejects_path_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "level0.json"
    matrix = tmp_path / "matrix.json"
    config = tmp_path / "config.json"
    _write_result(source, _passed_result())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_level0_power_inputs.py",
            "--level0-result",
            str(source),
            "--output-matrix",
            str(matrix),
            "--output-config",
            str(config),
        ],
    )
    bridge.main()
    output = json.loads(capsys.readouterr().out)
    assert output["power_scope"] == "level0_only"
    assert "selected_n_conf" not in output
    assert "candidate_n_selected" not in output
    assert json.loads(matrix.read_text(encoding="utf-8"))["source_artifact_sha256"] == output["source_result_sha256"]

    with pytest.raises(bridge.Level0PowerInputError, match="paths must be distinct"):
        bridge.write_power_inputs(
            source_result=source,
            output_matrix=source,
            output_config=config,
        )


def test_invalid_json_and_schema_extras_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(bridge.Level0PowerInputError, match="not valid UTF-8 JSON"):
        bridge.build_from_result_file(invalid)

    result = copy.deepcopy(_passed_result())
    result["unexpected"] = True
    with pytest.raises(bridge.Level0PowerInputError, match="schema mismatch"):
        bridge.build_level0_power_inputs(result, source_result_sha256="d" * 64)

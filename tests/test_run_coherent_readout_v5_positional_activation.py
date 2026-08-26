from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import run_coherent_readout_v5_positional_activation as runner


def _fixture() -> dict:
    return json.loads(runner.FIXTURE.read_text(encoding="utf-8"))


class _CharacterTokenizer:
    """Small offset-preserving tokenizer; it never touches model assets."""

    def apply_chat_template(self, messages, *, tokenize=False, **_flags):
        assert tokenize is False
        return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

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


def test_fixture_rebuild_and_split_contract_are_exact() -> None:
    fixture = runner.load_and_rebuild_fixture()
    assert len(fixture["worlds"]) == 32
    assert len(fixture["cells"]) == 1792
    assert {
        role: sum(world["role"] == role for world in fixture["worlds"]) for role in runner.ROLE_ORDER
    } == runner.ROLE_WORLD_COUNTS
    for world in fixture["worlds"]:
        rows = [cell for cell in fixture["cells"] if cell["world_id"] == world["world_id"]]
        assert sum(cell["intervention_prerequisite"] for cell in rows) == 16
        assert len(world["intervention_pairs"]) == 8


def test_system_message_is_shared_by_fixture_and_runner() -> None:
    assert _fixture()["system_message"] == runner.SYSTEM_EXACT


def test_mock_tokenizer_resolves_exact_target_property_site() -> None:
    cell = next(cell for cell in _fixture()["cells"] if cell["family"] == "composition")
    prompt = runner.render_prompt(_CharacterTokenizer(), cell)
    index = prompt["target_property_token_index"]
    span = prompt["target_property_rendered_span"]
    assert index is not None and span is not None
    assert prompt["rendered_text"][slice(*span)] == cell["target_property"]
    assert prompt["execution_input_ids"][index] == ord(cell["target_property"])
    assert prompt["expected_token_id"] == ord(cell["correct_answer"])


def test_target_span_rejects_split_or_shared_token() -> None:
    with pytest.raises(runner.PositionalActivationRunnerError, match="exactly one token"):
        runner.locate_exact_token([(0, 1), (1, 2)], (0, 2), "AB")
    with pytest.raises(runner.PositionalActivationRunnerError, match="shares a lexical token"):
        runner.locate_exact_token([(0, 2)], (1, 2), "XA")


def test_patch_condition_direction_and_source_contract_is_exact() -> None:
    expected = {
        "positional_rescue": ("second", "first", "positional"),
        "positional_damage": ("first", "second", "positional"),
        "answer_rescue_sham": ("second", "first", "answer"),
        "answer_damage_sham": ("first", "second", "answer"),
        "null_rescue_sham": ("second", "first", "null"),
        "null_damage_sham": ("first", "second", "null"),
    }
    assert {name: runner._condition_spec(name) for name in runner.PATCH_CONDITIONS} == expected
    template = runner._patch_template(
        role="localization",
        world_id="w",
        pair_id="pair",
        first_cell_id="first",
        second_cell_id="second",
        condition="positional_rescue",
        layer=12,
    )
    assert template["recipient_cell_id"] == "second"
    assert template["source_cell_id"] == "first"
    assert template["direction_name"] == "positional"
    assert template["operation"] == "positional_coordinate_patch"
    sham = runner._patch_template(
        role="localization",
        world_id="w",
        pair_id="pair",
        first_cell_id="first",
        second_cell_id="second",
        condition="answer_rescue_sham",
        layer=12,
    )
    assert sham["operation"] == "dose_matched_additive_sham"


def test_exact_call_plan_and_shard_counts() -> None:
    assert runner.EXPECTED_COUNTS == {
        "fit-baseline": 448,
        "localization-baseline": 448,
        "localization-patch": 1568,
        "holdout-baseline": 896,
        "holdout-patch": 784,
    }
    assert sum(runner.EXPECTED_COUNTS.values()) == 4144
    assert runner.EXPECTED_CUMULATIVE_CALLS["holdout-patch"] == 4144
    assert sum(spec["rows"] for spec in runner.raw_shard_specs("localization-patch")) == 1568
    assert all(spec["rows"] <= runner.RAW_SHARD_ROWS for spec in runner.raw_shard_specs("holdout-patch"))


def test_all_256_intervention_pairs_require_matched_token_shapes() -> None:
    worlds = []
    prompts = {}
    for world_index in range(32):
        pairs = []
        for pair_index in range(8):
            first = f"w{world_index}:p{pair_index}:first"
            second = f"w{world_index}:p{pair_index}:second"
            pair_id = f"w{world_index}:p{pair_index}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "target_first_cell_id": first,
                    "target_second_cell_id": second,
                }
            )
            for cell_id, token_index in ((first, 2), (second, 4)):
                prompts[cell_id] = {
                    "input_token_count": 7,
                    "execution_attention_mask": [1] * 7,
                    "target_property_token_index": token_index,
                }
        worlds.append({"world_id": f"w{world_index}", "intervention_pairs": pairs})
    receipts = runner.matched_pair_shape_receipts(worlds, prompts)
    assert len(receipts) == 256
    prompts["w0:p0:second"] = {
        **prompts["w0:p0:second"],
        "input_token_count": 8,
        "execution_attention_mask": [1] * 8,
    }
    with pytest.raises(runner.PositionalActivationRunnerError, match="token shapes differ"):
        runner.matched_pair_shape_receipts(worlds, prompts)


def test_patch_template_panel_counts_from_fixture() -> None:
    fixture = _fixture()
    localization = [world for world in fixture["worlds"] if world["role"] == "localization"]
    holdout = [world for world in fixture["worlds"] if world["role"] == "holdout"]
    assert (
        sum(
            len(world["intervention_pairs"]) * len(runner.PATCH_CONDITIONS) + 1
            for world in localization
            for _layer in runner.LAYER_GRID
        )
        == 1568
    )
    assert sum(len(world["intervention_pairs"]) * len(runner.PATCH_CONDITIONS) + 1 for world in holdout) == 784


def test_projected_patch_copies_only_the_registered_coordinate() -> None:
    recipient = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    source = np.asarray([5.0, 7.0, 11.0], dtype=np.float32)
    direction = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    result = runner.expected_projected_patch(recipient, source, direction)
    assert result == pytest.approx([5.0, 2.0, 3.0])
    identity = runner.expected_projected_patch(recipient, recipient, direction)
    assert np.array_equal(identity, recipient)
    answer_axis = np.asarray([0.0, 3.0, 0.0], dtype=np.float32)
    dose_matched = runner.expected_directional_displacement(recipient, answer_axis, 4.0)
    assert dose_matched == pytest.approx([1.0, 6.0, 3.0])
    assert np.linalg.norm(dose_matched - recipient) == pytest.approx(4.0)


def test_full_vocab_diagnostics_preserve_raw_row_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "MODEL_VOCAB_SIZE", 8)
    row = np.asarray([-3.0, 1.5, 0.25, -2.0, 4.0, 0.0, -1.0, -4.0], dtype=np.float32)
    value = runner.full_vocab_diagnostics(
        row,
        expected_answer="A",
        expected_token_id=1,
        distractor_answer="B",
        distractor_token_id=2,
    )
    assert value["answer_correct"] is True
    assert value["expected_minus_distractor_margin"] == pytest.approx(1.25)
    assert value["greedy_token_id"] == 4
    assert value["full_vocab_logits_sha256"] == runner.f32_sha256(row)


def test_raw_shard_writer_is_predeclared_and_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "RESULT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "MODEL_VOCAB_SIZE", 4)
    monkeypatch.setattr(runner, "RAW_SHARD_ROWS", 2)
    counts = dict(runner.EXPECTED_COUNTS)
    counts["fit-baseline"] = 3
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", counts)
    specs = runner.raw_shard_specs("fit-baseline")
    writer = runner.RawLogitShardWriter("fit-baseline", specs)
    bindings = [writer.append(np.full(4, value, dtype=np.float32)) for value in (1, 2, 3)]
    receipts = writer.finalize()
    assert [binding["raw_logits_global_row"] for binding in bindings] == [0, 1, 2]
    assert [receipt["shape"] for receipt in receipts] == [[2, 4], [1, 4]]
    with pytest.raises(runner.PositionalActivationRunnerError, match="already exists"):
        runner.RawLogitShardWriter("fit-baseline", specs)


def test_atomic_frozen_write_rejects_changed_replay(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    runner.write_json(path, {"a": 1})
    runner.write_json(path, {"a": 1})
    with pytest.raises(runner.PositionalActivationRunnerError, match="refusing to overwrite"):
        runner.write_json(path, {"a": 2})


@pytest.mark.parametrize(
    "drift",
    (
        {"packages": {"numpy": "drifted"}},
        {"runtime": {"device": "cpu", "dtype": "float32"}},
    ),
)
def test_current_dependency_replay_rejects_package_and_runtime_drift(
    drift: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = {
        "schema_version": runner.DEPENDENCY_LOCK_SCHEMA,
        "python": "3.x",
        "platform": "platform",
        "machine": "machine",
        "packages": {"numpy": "locked"},
        "implementation_files": {"runner": {"path": "runner", "sha256": "a" * 64}},
        "runtime": {"device": "mps", "dtype": "float32"},
        "canonical_sha256": "b" * 64,
    }
    current = {**stored, **drift}
    monkeypatch.setattr(runner, "_dependency_lock", lambda _path: current)
    with pytest.raises(runner.PositionalActivationRunnerError, match="dependency lock drifted"):
        runner._require_current_dependency_lock(stored)


def test_current_dependency_replay_accepts_exact_object(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = {"schema_version": runner.DEPENDENCY_LOCK_SCHEMA, "canonical_sha256": "a" * 64}
    monkeypatch.setattr(runner, "_dependency_lock", lambda _path: dict(stored))
    runner._require_current_dependency_lock(stored)


def test_plan_validator_enforces_zero_forward_and_4144_calls() -> None:
    core = {
        "schema_version": runner.PLAN_SCHEMA,
        "expected_counts": dict(runner.EXPECTED_COUNTS),
        "total_model_calls": 4144,
        "baseline_templates": [{}] * 1792,
        "patch_templates": [{}] * 2352,
        "model_calls_before_plan_freeze": 0,
        "generation_used": False,
    }
    plan = {**core, "call_plan_sha256": runner.canonical_sha256(core)}
    design = {
        "schema_version": runner.DESIGN_SCHEMA,
        "call_plan_sha256": plan["call_plan_sha256"],
        "model_calls": 0,
        "generation_used": False,
    }
    runner.validate_plan(plan, design)
    edited = dict(plan)
    edited["total_model_calls"] = 4143
    with pytest.raises(runner.PositionalActivationRunnerError):
        runner.validate_plan(edited, design)


def test_authority_rejects_counterfeit_and_extra_fields(tmp_path: Path) -> None:
    plan = {"call_plan_sha256": "plan"}
    expected = {
        "schema_version": runner.LOCALIZATION_PATCH_ENTRY_SCHEMA,
        "status": "LOCALIZATION_PATCH_AUTHORIZED",
        "call_plan_sha256": "plan",
        "basis_lock_file_sha256": "basis",
        "model_calls_issued_by_analyzer": 0,
        "generation_used": False,
        "claim_boundaries": runner.CLAIM_BOUNDARIES,
    }
    path = tmp_path / "valid.json"
    runner.write_json(path, expected)
    runner._require_authority(
        path,
        plan=plan,
        schema=runner.LOCALIZATION_PATCH_ENTRY_SCHEMA,
        status="LOCALIZATION_PATCH_AUTHORIZED",
        bindings={"basis_lock_file_sha256": "basis"},
    )
    counterfeits = (
        {**expected, "model_calls_issued_by_analyzer": 1},
        {**expected, "generation_used": True},
        {**expected, "claim_boundaries": {**runner.CLAIM_BOUNDARIES, "biology_inference": "allowed"}},
        {**expected, "unexpected_field": "smuggled"},
    )
    for index, counterfeit in enumerate(counterfeits):
        counterfeit_path = tmp_path / f"counterfeit_{index}.json"
        runner.write_json(counterfeit_path, counterfeit)
        with pytest.raises(runner.PositionalActivationRunnerError, match="authority content"):
            runner._require_authority(
                counterfeit_path,
                plan=plan,
                schema=runner.LOCALIZATION_PATCH_ENTRY_SCHEMA,
                status="LOCALIZATION_PATCH_AUTHORIZED",
                bindings={"basis_lock_file_sha256": "basis"},
            )


def test_selected_layer_lock_requires_exact_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    basis_lock = tmp_path / "basis_lock.json"
    patch_manifest = tmp_path / "localization_patch_manifest.json"
    localization_analysis = result_root / "localization_analysis.json"
    for path, value in (
        (basis_lock, {"basis": True}),
        (patch_manifest, {"patch": True}),
        (localization_analysis, {"analysis": True}),
    ):
        runner.write_json(path, value)
    monkeypatch.setattr(runner, "RESULT_ROOT", result_root)
    monkeypatch.setattr(runner, "DEFAULT_BASIS_LOCK", basis_lock)
    monkeypatch.setitem(runner.PHASE_PATHS["localization-patch"], "manifest", patch_manifest)
    plan = {"call_plan_sha256": "plan"}
    exact = {
        "schema_version": runner.LAYER_LOCK_SCHEMA,
        "status": "LOCALIZATION_LAYER_SELECTED",
        "call_plan_sha256": "plan",
        "selected_layer": 20,
        "selection_rule": "shallowest_layer_passing_all_preregistered_gates",
        "localization_patch_execution_manifest_file_sha256": runner.file_sha256(patch_manifest),
        "basis_lock_file_sha256": runner.file_sha256(basis_lock),
        "localization_analysis_file_sha256": runner.file_sha256(localization_analysis),
        "claim_boundaries": runner.CLAIM_BOUNDARIES,
    }
    valid = tmp_path / "valid_layer_lock.json"
    runner.write_json(valid, exact)
    monkeypatch.setattr(runner, "DEFAULT_LAYER_LOCK", valid)
    assert runner._selected_layer(plan) == 20

    counterfeit = tmp_path / "counterfeit_layer_lock.json"
    runner.write_json(counterfeit, {**exact, "unexpected_field": "smuggled"})
    monkeypatch.setattr(runner, "DEFAULT_LAYER_LOCK", counterfeit)
    with pytest.raises(runner.PositionalActivationRunnerError, match="layer lock content"):
        runner._selected_layer(plan)


def test_family_neutral_prerequisite_is_preserved_in_template() -> None:
    cell = next(
        cell
        for cell in _fixture()["cells"]
        if cell["family"] == "codebook_lookup" and not cell["intervention_prerequisite"]
    )
    prompt = {"prompt_id": "prompt"}
    template = runner._baseline_template(cell, prompt)
    assert template["intervention_prerequisite"] is False


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch unavailable")
def test_mock_model_captures_and_identity_patches_target_token() -> None:
    import torch

    class Block(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value

        def forward(self, hidden):
            return hidden + self.value

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(0.01) for _ in range(runner.MODEL_LAYERS)])

    class FakeLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model = Backbone()

        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            hidden = (
                torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    runner.MODEL_WIDTH,
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                + self.anchor
            )
            for block in self.model.layers:
                hidden = block(hidden)
            logits = (
                torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    runner.MODEL_VOCAB_SIZE,
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                + hidden[..., :1]
            )
            return SimpleNamespace(logits=logits)

    model = FakeLM().eval()
    prompt = {
        "execution_input_ids": [1, 2, 3, 4],
        "execution_attention_mask": [1, 1, 1, 1],
        "input_token_count": 4,
        "target_property_token_index": 2,
    }
    logits, captured, trace = runner._baseline_forward(model, prompt, layers=(12,))
    assert logits.shape == (runner.MODEL_VOCAB_SIZE,)
    assert captured is not None and captured.shape == (1, runner.MODEL_WIDTH)
    assert trace["token_index"] == 2
    direction = np.zeros(runner.MODEL_WIDTH, dtype=np.float32)
    direction[0] = 1.0
    patched_logits, patched, patch_trace = runner._patch_forward(
        model,
        prompt,
        layer=12,
        recipient_activation=captured[0],
        source_activation=captured[0],
        direction=direction,
        signed_scalar=0.0,
        positional_scalar_d=0.0,
    )
    assert patched_logits.shape == logits.shape
    assert np.array_equal(patched, captured[0])
    assert patch_trace["hook_removed"] is True
    assert patch_trace["non_target_tokens_unchanged"] is True
    assert patch_trace["registered_positional_dose"] == 0.0
    assert patch_trace["applied_displacement_l2"] == 0.0

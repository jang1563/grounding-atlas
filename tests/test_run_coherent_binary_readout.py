from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import causal_intervention as causal  # noqa: E402
import coherent_binary_readout as coherent  # noqa: E402
import run_coherent_binary_readout as runner  # noqa: E402

MODEL_REVISION = "mock-model-snapshot"
TOKENIZER_REVISION = "mock-tokenizer-snapshot"
TOKENIZER_VOCAB_SIZE = 128
MODEL_VOCAB_SIZE = 132
X_TOKEN_ID = ord("X")
Y_TOKEN_ID = ord("Y")


class MockTokenizer:
    chat_template = "mock-template-v1"

    def __len__(self) -> int:
        return TOKENIZER_VOCAB_SIZE

    def get_chat_template(self) -> str:
        return self.chat_template

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        values = [ord(character) for character in text]
        assert all(value < TOKENIZER_VOCAB_SIZE for value in values)
        return values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool = False,
        return_attention_mask: bool = False,
    ) -> Any:
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        rendered = f"<U>{messages[0]['content']}<A>"
        if not tokenize:
            return rendered
        input_ids = self.encode(rendered, add_special_tokens=False)
        if not return_dict:
            return input_ids
        encoded: dict[str, list[int]] = {"input_ids": input_ids}
        if return_attention_mask:
            encoded["attention_mask"] = [1] * len(input_ids)
        return encoded


class NonSingletonTokenizer(MockTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith("<A>X"):
            values.append(1)
        return values


class MockModel:
    def __init__(self, vocab_size: int = MODEL_VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.calls: list[dict[str, Any]] = []
        self.eval_calls = 0

    def eval(self) -> MockModel:
        self.eval_calls += 1
        return self

    def __call__(
        self,
        *,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        use_cache: bool,
        return_dict: bool,
    ) -> dict[str, np.ndarray]:
        assert use_cache is False
        assert return_dict is True
        assert len(input_ids) == len(attention_mask) == 1
        assert len(input_ids[0]) == len(attention_mask[0])
        assert set(attention_mask[0]) == {1}
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_ids": copy.deepcopy(input_ids),
                "attention_mask": copy.deepcopy(attention_mask),
            }
        )
        row = np.full(self.vocab_size, -4.0, dtype=np.float32)
        row[X_TOKEN_ID] = np.float32(0.25 + call_index / 100.0)
        row[Y_TOKEN_ID] = np.float32(-0.5 + call_index / 200.0)
        row[self.vocab_size - 1] = np.float32(1.5)
        logits = np.zeros(
            (1, len(input_ids[0]), self.vocab_size),
            dtype=np.float32,
        )
        logits[0, -1, :] = row
        return {"logits": logits}


def _fixture(donors: tuple[str, ...] = ("d01", "d02")) -> dict[str, Any]:
    readouts = {
        "cytotoxic_state": {
            "negative_class": "cytotoxic-low",
            "positive_class": "cytotoxic-high",
        },
        "lineage": {
            "negative_class": "CD8 T cell",
            "positive_class": "NK cell",
        },
    }
    items = []
    for donor in donors:
        source_entity_id = f"source:{donor}:cell"
        item_id = f"development:{donor}:cell:unmodified"
        gene_sentence = f"GNLY NKG7 CCL5 B2M {donor.upper()}"
        for readout_id, classes in readouts.items():
            items.append(
                {
                    "schema_version": "coherent-readout-development-item-v1",
                    "item_id": item_id,
                    "donor_id": donor,
                    "source_entity_id": source_entity_id,
                    "fixture_record_id": coherent.text_sha256(
                        f"fixture|{donor}|{readout_id}"
                    ),
                    "input_family": "unmodified",
                    "readout_id": readout_id,
                    **classes,
                    "gene_sentence": gene_sentence,
                    "gene_sentence_sha256": coherent.text_sha256(gene_sentence),
                    "confirmatory_eligibility": "prohibited",
                }
            )
    return {
        "schema_version": runner.INPUT_SCHEMA,
        "analysis_id": "mock-development-fixture",
        "mode": "development",
        "firewall": {"confirmatory_eligibility": "prohibited"},
        "donor_ids": list(donors),
        "input_families": ["unmodified"],
        "readouts": readouts,
        "items": items,
    }


def _plan(
    fixture: dict[str, Any], tokenizer: MockTokenizer | None = None
) -> dict[str, Any]:
    return runner.build_call_plan(
        fixture,
        tokenizer=tokenizer or MockTokenizer(),
        model_vocab_size=MODEL_VOCAB_SIZE,
        model_id="mock/model",
        model_revision=MODEL_REVISION,
        tokenizer_id="mock/tokenizer",
        tokenizer_revision=TOKENIZER_REVISION,
        dtype="float32",
        source_fixture_sha256=coherent.text_sha256("source fixture"),
        source_manifest_sha256=coherent.text_sha256("source manifest"),
        preregistration_sha256=coherent.text_sha256("preregistration"),
        margin_lock_sha256=runner.candidate_margin_lock_sha256(),
        margin_lock_status="candidate_unqualified",
    )


def test_plan_freezes_exact_four_forms_and_sidecar_rows_before_forward() -> None:
    plan = _plan(_fixture())

    assert plan["record_count"] == 2 * 2 * 4
    assert plan["vocab_size"] == MODEL_VOCAB_SIZE
    assert plan["tokenizer_vocab_size"] == TOKENIZER_VOCAB_SIZE
    assert plan["x_token_id"] == X_TOKEN_ID
    assert plan["y_token_id"] == Y_TOKEN_ID
    assert [record["planned_index"] for record in plan["records"]] == list(
        range(plan["record_count"])
    )
    assert [record["full_vocab_logits_row"] for record in plan["records"]] == list(
        range(plan["record_count"])
    )
    grouped = Counter(
        (
            record["source_item_id"],
            record["item_id"],
            record["readout_id"],
            record["input_family"],
        )
        for record in plan["records"]
    )
    assert set(grouped.values()) == {4}
    for key in grouped:
        forms = {
            (record["order"], record["mapping"])
            for record in plan["records"]
            if (
                record["source_item_id"],
                record["item_id"],
                record["readout_id"],
                record["input_family"],
            )
            == key
        }
        assert forms == set(coherent.FORM_KEYS)

    for record in plan["records"]:
        assert record["record_id"] == coherent.record_id(record)
        assert record["prompt_sha256"] == coherent.text_sha256(
            record["rendered_chat"]
        )
        assert record["user_prompt_sha256"] == coherent.text_sha256(
            record["user_prompt"]
        )
        assert record["execution_input_sha256"] == (
            causal.execution_input_sha256(
                record["execution_input_ids"],
                record["execution_attention_mask"],
            )
        )
    assert plan["call_plan_sha256"] == coherent.call_plan_sha256(plan["records"])


def test_plan_only_writes_exact_valid_design_without_forward(
    tmp_path: Path,
) -> None:
    plan = _plan(_fixture())
    design_path = tmp_path / "design.json"
    manifest_path = tmp_path / "plan-manifest.json"

    manifest = runner.write_plan_manifest(
        plan=plan,
        output_design_path=design_path,
        output_manifest_path=manifest_path,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
    )

    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design == runner.design_from_plan(plan)
    assert coherent.validate_design(design) == design
    assert set(design) == coherent.DESIGN_KEYS
    assert design["expected_vocab_size"] == MODEL_VOCAB_SIZE
    assert manifest["forward_passes_executed"] == 0
    assert manifest["design_artifact"]["validated"] is True
    assert manifest["design"]["call_plan_sha256"] == plan["call_plan_sha256"]


def test_run_writes_raw_records_and_full_vocab_sidecar_one_forward_per_form(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    tokenizer = MockTokenizer()
    plan = _plan(fixture, tokenizer)
    design = runner.design_from_plan(plan)
    model = MockModel()
    records_path = tmp_path / "raw.jsonl"
    logits_path = tmp_path / "raw.full_logits.npy"
    manifest_path = tmp_path / "execution.json"

    records, manifest = runner.run_development(
        input_manifest=fixture,
        design=design,
        tokenizer=tokenizer,
        model=model,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        model_vocab_size=MODEL_VOCAB_SIZE,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        dtype="float32",
        source_fixture_sha256=plan["source_fixture_sha256"],
        source_manifest_sha256=plan["source_manifest_sha256"],
        preregistration_sha256=plan["preregistration_sha256"],
        margin_lock_sha256=plan["margin_lock_sha256"],
        margin_lock_status=plan["margin_lock_status"],
        output_records_path=records_path,
        output_logits_path=logits_path,
        output_manifest_path=manifest_path,
    )

    assert model.eval_calls == 1
    assert len(model.calls) == len(records) == plan["record_count"]
    disk_records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert disk_records == records
    assert all(set(record) == coherent.RECORD_KEYS for record in records)
    assert all("s" not in record and "q_positive" not in record for record in records)
    matrix = np.load(logits_path, allow_pickle=False)
    assert matrix.dtype == np.dtype("<f4")
    assert matrix.shape == (len(records), MODEL_VOCAB_SIZE)
    matrix_sha256 = coherent.verify_full_vocab_sidecar(records, design, matrix)
    assert matrix_sha256 == coherent.full_vocab_matrix_sha256(matrix)
    for record in records:
        diagnostics = coherent.full_vocab_diagnostics(
            matrix[record["full_vocab_logits_row"]],
            x_token_id=record["x_token_id"],
            y_token_id=record["y_token_id"],
        )
        assert diagnostics["full_vocab_logits_sha256"] == record[
            "full_vocab_logits_sha256"
        ]
        assert diagnostics["greedy_token_id"] == record["greedy_token_id"]
        assert coherent.validate_record(record) == record
    assert manifest["forward_passes_executed"] == len(records)
    assert manifest["biological_analysis_run"] is False
    assert manifest["confirmatory_execution"] is False
    assert manifest["output"]["full_vocab_logits_sidecar"][
        "matrix_sha256"
    ] == matrix_sha256


def test_non_singleton_labels_fail_during_planning() -> None:
    with pytest.raises(coherent.CoherentReadoutError, match="context-stable token"):
        _plan(_fixture(), NonSingletonTokenizer())


def test_confirmatory_design_is_rejected_before_any_model_forward(
    tmp_path: Path,
) -> None:
    donors = tuple(f"d{index:02d}" for index in range(12))
    fixture = _fixture(donors)
    tokenizer = MockTokenizer()
    plan = _plan(fixture, tokenizer)
    design = runner.design_from_plan(plan)
    confirmatory = copy.deepcopy(design)
    confirmatory["mode"] = "confirmatory"
    confirmatory["expected_confirmatory_donors"] = len(donors)
    confirmatory["margin_lock_status"] = "phase0_qualified"
    assert coherent.validate_design(confirmatory)["mode"] == "confirmatory"
    model = MockModel()

    with pytest.raises(runner.CoherentReadoutRunnerError, match="refuses confirmatory"):
        runner.run_development(
            input_manifest=fixture,
            design=confirmatory,
            tokenizer=tokenizer,
            model=model,
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
            model_vocab_size=MODEL_VOCAB_SIZE,
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            dtype="float32",
            source_fixture_sha256=plan["source_fixture_sha256"],
            source_manifest_sha256=plan["source_manifest_sha256"],
            preregistration_sha256=plan["preregistration_sha256"],
            margin_lock_sha256=plan["margin_lock_sha256"],
            margin_lock_status=plan["margin_lock_status"],
            output_records_path=tmp_path / "never.jsonl",
            output_logits_path=tmp_path / "never.npy",
            output_manifest_path=tmp_path / "never-manifest.json",
        )
    assert model.eval_calls == 0
    assert model.calls == []
    assert list(tmp_path.iterdir()) == []


def test_hf_loader_moves_model_explicitly_without_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.float16 = object()
    fake_torch.bfloat16 = object()
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True)
    )
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda value: f"device:{value}"

    class FakeModel:
        def __init__(self) -> None:
            self.to_calls: list[str] = []
            self.eval_calls = 0

        def to(self, device: str) -> FakeModel:
            self.to_calls.append(device)
            return self

        def eval(self) -> FakeModel:
            self.eval_calls += 1
            return self

    loaded = FakeModel()

    class FakeAutoModel:
        call: tuple[str, dict[str, Any]] | None = None

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            cls.call = (model_id, kwargs)
            return loaded

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = runner._load_hf_model(
        "mock/model",
        MODEL_REVISION,
        "float32",
        local_files_only=True,
        device="mps",
    )

    assert result is loaded
    assert FakeAutoModel.call is not None
    model_id, kwargs = FakeAutoModel.call
    assert model_id == "mock/model"
    assert "device_map" not in kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["dtype"] is fake_torch.float32
    assert loaded.to_calls == ["device:mps"]
    assert loaded.eval_calls == 1

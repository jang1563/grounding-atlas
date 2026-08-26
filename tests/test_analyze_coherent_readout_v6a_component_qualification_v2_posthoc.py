from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval import analyze_coherent_readout_v6a_component_qualification_v2_posthoc as posthoc


class _NaturalTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        registry = {
            "ANSWER:": [10, 11],
            "ANSWER: α": [10, 11, 101],
            "ANSWER: β": [10, 11, 102],
        }
        return registry[text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return {101: " α", 102: " β"}[token_ids[0]]


def test_natural_surface_must_be_one_prompt_style_contextual_token() -> None:
    tokenizer = _NaturalTokenizer()

    assert (
        posthoc.natural_contextual_token_id(
            tokenizer,
            "ANSWER:",
            "α",
            101,
            201,
        )
        == 101
    )
    with pytest.raises(posthoc.V2ContextualPosthocError, match="prompt-style token"):
        posthoc.natural_contextual_token_id(
            tokenizer,
            "ANSWER:",
            "α",
            999,
            201,
        )


def test_natural_token_must_decode_exactly_and_differ_from_bare_token() -> None:
    tokenizer = _NaturalTokenizer()
    with pytest.raises(posthoc.V2ContextualPosthocError, match="does not differ"):
        posthoc.natural_contextual_token_id(
            tokenizer,
            "ANSWER:",
            "α",
            101,
            101,
        )

    class _WrongDecodeTokenizer(_NaturalTokenizer):
        def decode(self, token_ids, **kwargs):
            super().decode(token_ids, **kwargs)
            return "α"

    with pytest.raises(posthoc.V2ContextualPosthocError, match="decode exactly"):
        posthoc.natural_contextual_token_id(
            _WrongDecodeTokenizer(),
            "ANSWER:",
            "α",
            101,
            201,
        )


def test_contextual_row_diagnostics_distinguish_unique_answer_and_pair_tie() -> None:
    record = {"expected_answer": "α", "distractor_answer": "β"}
    unique = np.asarray([0.0, 1.0, 6.0, 2.0], dtype="<f4")
    tied = np.asarray([0.0, 1.0, 6.0, 6.0], dtype="<f4")

    unique_result = posthoc.contextual_row_diagnostics(unique, record, 2, 3)
    tied_result = posthoc.contextual_row_diagnostics(tied, record, 2, 3)

    assert unique_result["answer_correct"] is True
    assert unique_result["strict_unique_expected_global_max"] is True
    assert unique_result["maximum_set_contained_in_natural_pair"] is True
    assert tied_result["answer_tie"] is True
    assert tied_result["maximum_token_ids"] == [2, 3]
    assert tied_result["strict_unique_expected_global_max"] is False
    assert tied_result["maximum_set_contained_in_natural_pair"] is True


def test_exact_observed_checks_are_not_relaxed() -> None:
    summary = {
        "families": {
            "property_retrieval": {
                "n": 256,
                "strict_pairwise_correct": 256,
                "strict_pairwise_accuracy": 1.0,
                "exact_answer_ties": 0,
                "strict_unique_expected_global_max": 256,
                "maximum_set_contained_in_natural_pair": 256,
            },
            "codebook_lookup": {
                "n": 128,
                "strict_pairwise_correct": 127,
                "strict_pairwise_accuracy": 127 / 128,
                "exact_answer_ties": 1,
                "strict_unique_expected_global_max": 127,
                "maximum_set_contained_in_natural_pair": 128,
            },
        },
        "all_rows_maximum_set_contained_in_natural_pair": 384,
        "bare_failure_call_indices": [62, 207, 250],
        "bare_failures_corrected_by_natural_surface": [207, 250],
        "bare_failures_unresolved_by_natural_surface": [62],
    }

    posthoc._assert_exact_observed_checks(summary)
    changed = json.loads(json.dumps(summary))
    changed["families"]["codebook_lookup"]["strict_pairwise_correct"] = 128
    with pytest.raises(posthoc.V2ContextualPosthocError, match="lookup observation"):
        posthoc._assert_exact_observed_checks(changed)


def test_posthoc_writer_is_separate_idempotent_and_refuses_differing_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = tmp_path / "sealed"
    output_root = tmp_path / "posthoc"
    output = output_root / "analysis.json"
    monkeypatch.setattr(posthoc, "SEALED_RESULT_ROOT", sealed)
    monkeypatch.setattr(posthoc, "POSTHOC_RESULT_ROOT", output_root)

    posthoc._write_frozen_posthoc(output, {"value": 1})
    posthoc._write_frozen_posthoc(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(posthoc.V2ContextualPosthocError, match="differing"):
        posthoc._write_frozen_posthoc(output, {"value": 2})
    with pytest.raises(posthoc.V2ContextualPosthocError, match="sealed V2"):
        posthoc._write_frozen_posthoc(sealed / "forbidden.json", {"value": 1})
    with pytest.raises(posthoc.V2ContextualPosthocError, match="dedicated"):
        posthoc._write_frozen_posthoc(tmp_path / "elsewhere.json", {"value": 1})


def test_sealed_hash_registry_covers_every_scientific_source_artifact() -> None:
    assert set(posthoc.SEALED_ARTIFACT_SHA256) == {
        "qualification_analysis.json",
        "plan_manifest.json",
        "design.json",
        "dependency_lock.json",
        "tokenization_receipt.json",
        "loader_smoke_receipt.json",
        "qualification_baseline_attempt.json",
        "qualification_baseline_execution_manifest.json",
        "qualification_baseline_records.jsonl",
    }
    assert set(posthoc.SEALED_RAW_SHARD_SHA256) == {
        f"shard_{index:03d}.npy" for index in range(6)
    }

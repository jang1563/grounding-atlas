#!/usr/bin/env python3
"""Descriptive natural-context token replay of the sealed V6A V2 qualification.

This post-hoc issues no model calls, does not alter the terminal V2 component
failure, and writes only to a separate result root.  It first delegates the
complete scientific-artifact replay to the sealed V2 analyzer, then evaluates
the already-stored full-vocabulary logits at the natural one-token surface
``" " + glyph``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from eval import analyze_coherent_readout_v6a_component_qualification_v2 as v2
else:
    import analyze_coherent_readout_v6a_component_qualification_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
SEALED_RESULT_ROOT = v2.RESULT_ROOT
POSTHOC_RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_component_qualification_v2_posthoc"
    / "qwen2.5-7b-instruct"
)
OUTPUT = POSTHOC_RESULT_ROOT / "contextual_token_posthoc_analysis.json"

POSTHOC_SCHEMA = "coherent-readout-v6a-qualification-v2-contextual-token-posthoc-v2"
POSTHOC_STATUS = "V2_CONTEXTUAL_TOKEN_POSTHOC_DESCRIPTIVE_COMPLETE"
NATURAL_SURFACE_PREFIX = " "

# These hashes bind the completed, terminal V2 artifact set.  A later file that
# merely remains internally self-consistent is not accepted as the same source.
SEALED_ARTIFACT_SHA256 = {
    "qualification_analysis.json": "595cff448a3f72011e119f37556c95e11ea0fe4c4daef7d79541f507f4987cb8",
    "plan_manifest.json": "27b33868fb6269e2dd73ea50c8f41c21d6ab9fbb67a82c795b476993fa185758",
    "design.json": "b299caf0d1549f90f9ac12cae23ba944203bfd0f7e55227e7df0fb56b2b93d42",
    "dependency_lock.json": "8b52bb32ce972a36d316ae5b6f63810956d0b9cba955f11e8fd2b15b01a8a73e",
    "tokenization_receipt.json": "b8eda62d4bca9bc7cdebda721f617a6f3f103d663333ec83ec8dbf4a22eb48e5",
    "loader_smoke_receipt.json": "1489e46fab9ad3a082bafdefd1fd785b4e01f62bdb4759873e4eb17bb60e7f2d",
    "qualification_baseline_attempt.json": "8950a68ac52c9b83aa51f2c9eeb38f389caad030c487421ebfadba0c509131f1",
    "qualification_baseline_execution_manifest.json": (
        "2c27f0491d2fbbb741dc96d9bde17b2c17b383fe0e05ac31a1a298e07209f354"
    ),
    "qualification_baseline_records.jsonl": (
        "8ba92c45a72173680f080d9cd9ceffe9006a833fbd3f023d9cf3833499c3b162"
    ),
}
SEALED_RAW_SHARD_SHA256 = {
    "shard_000.npy": "5374be775a9e6d6682f6a03b22aafeea6992b9233ee697a922752ce90b663eba",
    "shard_001.npy": "36404e1b2cd822152fcd42fd8638d68e87f4f2c0f13476d7f14d749633653789",
    "shard_002.npy": "47f417ea955d238fb17c90d384a50da47735b4051d8e47a8ff4d65cfe4cb7dc2",
    "shard_003.npy": "ca2646612d1d95ccba5d9c5f7bd909f9e764de2632f9ee8fadbc76176ffd6df7",
    "shard_004.npy": "933012f78833544b8caae3094fed888e176d474c7fd3acb6a5e6540faf35d7a5",
    "shard_005.npy": "f8494885842590cf6ff201cc5c5b7218d6297402b234323402a7230b9bc2ffc1",
}

EXPECTED_BARE_FAILURE_CALLS = [62, 207, 250]
EXPECTED_NATURAL_CORRECTED_CALLS = [207, 250]
EXPECTED_NATURAL_UNRESOLVED_CALLS = [62]


class V2ContextualPosthocError(ValueError):
    """Raised when the sealed source or descriptive replay contract changes."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContextualPosthocError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise V2ContextualPosthocError(f"JSON artifact is not an object: {path}")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_frozen_posthoc(path: Path, value: Mapping[str, Any]) -> None:
    if _is_within(path, SEALED_RESULT_ROOT):
        raise V2ContextualPosthocError("post-hoc output cannot enter the sealed V2 result root")
    if not _is_within(path, POSTHOC_RESULT_ROOT):
        raise V2ContextualPosthocError("post-hoc output is outside its dedicated result root")
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise V2ContextualPosthocError(
                f"refusing to overwrite differing frozen post-hoc output: {path}"
            )
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def verify_sealed_v2_artifacts() -> dict[str, Any]:
    """Bind the exact terminal V2 files and its immutable component-fail status."""

    observed: dict[str, str] = {}
    for relative, expected in SEALED_ARTIFACT_SHA256.items():
        path = SEALED_RESULT_ROOT / relative
        actual = v2.file_sha256(path)
        if actual != expected:
            raise V2ContextualPosthocError(f"sealed V2 artifact hash changed: {relative}")
        observed[relative] = actual
    for filename, expected in SEALED_RAW_SHARD_SHA256.items():
        path = v2.RAW_LOGIT_ROOT / filename
        actual = v2.file_sha256(path)
        if actual != expected:
            raise V2ContextualPosthocError(f"sealed V2 raw shard hash changed: {filename}")
        observed[f"raw_logits/qualification-baseline/{filename}"] = actual

    source = _load_json(v2.ANALYSIS)
    if (
        source.get("schema_version") != v2.ANALYSIS_SCHEMA
        or source.get("status") != v2.COMPONENT_FAIL
        or source.get("engineering_valid") is not True
        or source.get("component_qualified") is not False
        or source.get("model_calls_issued_by_analyzer") != 0
        or source.get("composition_calls_analyzed") != 0
    ):
        raise V2ContextualPosthocError("sealed V2 terminal component-fail identity changed")
    components = source.get("components")
    if not isinstance(components, Mapping) or components.get("gates", {}).get("pass") is not False:
        raise V2ContextualPosthocError("sealed V2 component gate result changed")
    return {"analysis": source, "hashes": observed}


def _rendered_context(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    cell: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> str:
    messages = [
        {"role": "system", "content": fixture["system_message"]},
        {"role": "user", "content": cell["prompt_text"]},
        {"role": "assistant", "content": fixture["assistant_prefill"]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **v2.CHAT_FLAGS)
    except Exception as error:
        raise V2ContextualPosthocError("cannot replay a sealed rendered context") from error
    if not isinstance(rendered, str) or not rendered.endswith(str(fixture["assistant_prefill"])):
        raise V2ContextualPosthocError("sealed rendered context no longer ends at ANSWER:")
    ids = v2._as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False),
        "post-hoc rendered context",
    )
    if ids != prompt["execution_input_ids"]:
        raise V2ContextualPosthocError("post-hoc rendered context differs from the sealed prompt")
    return rendered


def natural_contextual_token_id(
    tokenizer: Any,
    rendered: str,
    glyph: str,
    prompt_style_token_id: int,
    bare_continuation_token_id: int,
) -> int:
    """Prove that one natural space-plus-glyph is the prompt-style token."""

    if not isinstance(glyph, str) or len(glyph) != 1:
        raise V2ContextualPosthocError("natural answer glyph is not one character")
    prefix = v2._as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False),
        "post-hoc prefix",
    )
    surface = NATURAL_SURFACE_PREFIX + glyph
    combined = v2._as_int_vector(
        tokenizer.encode(rendered + surface, add_special_tokens=False),
        "post-hoc prefix plus natural answer",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise V2ContextualPosthocError(
            f"natural answer surface {surface!r} is not one contextual token"
        )
    token_id = combined[-1]
    if token_id != prompt_style_token_id:
        raise V2ContextualPosthocError(
            f"natural answer surface {surface!r} does not use its prompt-style token"
        )
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:
        raise V2ContextualPosthocError("cannot decode the natural answer token") from error
    if decoded != surface:
        raise V2ContextualPosthocError(
            f"prompt-style token does not decode exactly to {surface!r}"
        )
    if token_id == bare_continuation_token_id:
        raise V2ContextualPosthocError(
            "prompt-style token does not differ from the sealed bare continuation token"
        )
    return token_id


def _row_from_execution(
    execution: Mapping[str, Any], record: Mapping[str, Any]
) -> np.ndarray:
    shard_index = record["raw_logits_shard_index"]
    row_index = record["raw_logits_row_in_shard"]
    shards = execution["raw_shards"]
    try:
        row = shards[shard_index][row_index]
    except (IndexError, KeyError, TypeError) as error:
        raise V2ContextualPosthocError("sealed raw-logit row binding changed") from error
    return np.ascontiguousarray(row, dtype="<f4")


def contextual_row_diagnostics(
    row: np.ndarray,
    record: Mapping[str, Any],
    expected_token_id: int,
    distractor_token_id: int,
) -> dict[str, Any]:
    natural_record = {
        **record,
        "expected_token_id": expected_token_id,
        "distractor_token_id": distractor_token_id,
    }
    diagnostics = v2.diagnostics_from_full_vocab(row, natural_record)
    maximum_ids = diagnostics["maximum_token_ids"]
    natural_pair = {expected_token_id, distractor_token_id}
    return {
        **diagnostics,
        "maximum_set_contained_in_natural_pair": set(maximum_ids) <= natural_pair,
        "strict_unique_expected_global_max": maximum_ids == [expected_token_id],
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family, expected_n in v2.EXPECTED_FAMILY_COUNTS.items():
        group = [row for row in rows if row["family"] == family]
        if len(group) != expected_n:
            raise V2ContextualPosthocError(f"post-hoc family count changed: {family}")
        strict = sum(row["natural_diagnostics"]["answer_correct"] for row in group)
        ties = sum(row["natural_diagnostics"]["answer_tie"] for row in group)
        strict_unique = sum(
            row["natural_diagnostics"]["strict_unique_expected_global_max"]
            for row in group
        )
        contained = sum(
            row["natural_diagnostics"]["maximum_set_contained_in_natural_pair"]
            for row in group
        )
        families[family] = {
            "n": len(group),
            "strict_pairwise_correct": strict,
            "strict_pairwise_accuracy": strict / len(group),
            "exact_answer_ties": ties,
            "strict_unique_expected_global_max": strict_unique,
            "maximum_set_contained_in_natural_pair": contained,
        }

    bare_failures = [
        row["call_index"] for row in rows if not row["bare_diagnostics"]["answer_correct"]
    ]
    corrected = [
        row["call_index"]
        for row in rows
        if row["call_index"] in bare_failures
        and row["natural_diagnostics"]["answer_correct"]
    ]
    unresolved = [
        row["call_index"]
        for row in rows
        if row["call_index"] in bare_failures
        and not row["natural_diagnostics"]["answer_correct"]
    ]
    return {
        "families": families,
        "all_rows_maximum_set_contained_in_natural_pair": sum(
            row["natural_diagnostics"]["maximum_set_contained_in_natural_pair"]
            for row in rows
        ),
        "bare_failure_call_indices": bare_failures,
        "bare_failures_corrected_by_natural_surface": corrected,
        "bare_failures_unresolved_by_natural_surface": unresolved,
    }


def _assert_exact_observed_checks(summary: Mapping[str, Any]) -> None:
    retrieval = summary["families"]["property_retrieval"]
    lookup = summary["families"]["codebook_lookup"]
    if retrieval != {
        "n": 256,
        "strict_pairwise_correct": 256,
        "strict_pairwise_accuracy": 1.0,
        "exact_answer_ties": 0,
        "strict_unique_expected_global_max": 256,
        "maximum_set_contained_in_natural_pair": 256,
    }:
        raise V2ContextualPosthocError("natural retrieval observation changed")
    if lookup != {
        "n": 128,
        "strict_pairwise_correct": 127,
        "strict_pairwise_accuracy": 127 / 128,
        "exact_answer_ties": 1,
        "strict_unique_expected_global_max": 127,
        "maximum_set_contained_in_natural_pair": 128,
    }:
        raise V2ContextualPosthocError("natural lookup observation changed")
    if (
        summary["all_rows_maximum_set_contained_in_natural_pair"] != 384
        or summary["bare_failure_call_indices"] != EXPECTED_BARE_FAILURE_CALLS
        or summary["bare_failures_corrected_by_natural_surface"]
        != EXPECTED_NATURAL_CORRECTED_CALLS
        or summary["bare_failures_unresolved_by_natural_surface"]
        != EXPECTED_NATURAL_UNRESOLVED_CALLS
    ):
        raise V2ContextualPosthocError("natural contextual-token call-level observation changed")


def build_posthoc_analysis() -> dict[str, Any]:
    """Replay sealed artifacts and compute descriptive natural-token diagnostics."""

    sealed = verify_sealed_v2_artifacts()
    # These are the sealed analyzer's complete immutable plan/execution replay
    # paths.  Neither function performs generation or a model forward.
    plan_bundle = v2._load_and_validate_plan()
    execution = v2._validate_execution(plan_bundle)
    plan = plan_bundle["plan"]
    fixture = plan_bundle["fixture"]
    tokenizer = v2._load_tokenizer(plan_bundle["dependency"])

    contracts = {
        item["symbol"]: item for item in plan.get("symbol_token_contracts", [])
    }
    if len(contracts) != 32:
        raise V2ContextualPosthocError("sealed symbol-token contract count changed")
    cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    prompts = plan["prompts"]
    records = execution["records"]
    if len(prompts) != v2.EXPECTED_CALLS or len(records) != v2.EXPECTED_CALLS:
        raise V2ContextualPosthocError("sealed post-hoc row count changed")

    natural_contract_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for record, prompt in zip(records, prompts, strict=True):
        if record["call_index"] != len(rows) or record["cell_id"] != prompt["cell_id"]:
            raise V2ContextualPosthocError("sealed post-hoc row order changed")
        cell = cells.get(prompt["cell_id"])
        if cell is None:
            raise V2ContextualPosthocError("sealed prompt does not resolve to a cell")
        rendered = _rendered_context(tokenizer, fixture, cell, prompt)
        natural_ids: dict[str, int] = {}
        for role, glyph in (
            ("expected", record["expected_answer"]),
            ("distractor", record["distractor_answer"]),
        ):
            contract = contracts.get(glyph)
            if contract is None:
                raise V2ContextualPosthocError("answer glyph has no sealed token contract")
            token_id = natural_contextual_token_id(
                tokenizer,
                rendered,
                glyph,
                contract["prompt_token_id"],
                contract["continuation_token_id"],
            )
            natural_ids[role] = token_id
            natural_contract_counts[glyph] += 1
        if natural_ids["expected"] == natural_ids["distractor"]:
            raise V2ContextualPosthocError("natural answer-token pair collapsed")

        row = _row_from_execution(execution, record)
        diagnostics = contextual_row_diagnostics(
            row,
            record,
            natural_ids["expected"],
            natural_ids["distractor"],
        )
        rows.append(
            {
                "call_index": record["call_index"],
                "cell_id": record["cell_id"],
                "prompt_id": record["prompt_id"],
                "world_id": record["world_id"],
                "family": record["family"],
                "expected_answer": record["expected_answer"],
                "distractor_answer": record["distractor_answer"],
                "bare_expected_token_id": record["expected_token_id"],
                "bare_distractor_token_id": record["distractor_token_id"],
                "natural_expected_surface": NATURAL_SURFACE_PREFIX
                + record["expected_answer"],
                "natural_distractor_surface": NATURAL_SURFACE_PREFIX
                + record["distractor_answer"],
                "natural_expected_token_id": natural_ids["expected"],
                "natural_distractor_token_id": natural_ids["distractor"],
                "bare_diagnostics": record["diagnostics"],
                "natural_diagnostics": diagnostics,
            }
        )

    expected_contract_counts = Counter()
    for row in rows:
        expected_contract_counts[row["expected_answer"]] += 1
        expected_contract_counts[row["distractor_answer"]] += 1
    if natural_contract_counts != expected_contract_counts:
        raise V2ContextualPosthocError("natural token proof did not cover every answer context")

    summary = _summary(rows)
    _assert_exact_observed_checks(summary)
    natural_contracts = [
        {
            "glyph": glyph,
            "natural_surface": NATURAL_SURFACE_PREFIX + glyph,
            "natural_token_id": contracts[glyph]["prompt_token_id"],
            "bare_token_id": contracts[glyph]["continuation_token_id"],
            "contexts_proved": natural_contract_counts[glyph],
            "uses_prompt_style_token_in_every_context": True,
            "decodes_exactly_to_natural_surface": True,
            "differs_from_bare_continuation_token": True,
        }
        for glyph in sorted(contracts)
    ]
    return {
        "schema_version": POSTHOC_SCHEMA,
        "status": POSTHOC_STATUS,
        "source_terminal_status": v2.COMPONENT_FAIL,
        "source_component_qualification_reopened": False,
        "authorization_issued": False,
        "descriptive_only": True,
        "natural_answer_surface_rule": "one ASCII space followed by one registered glyph",
        "source_artifact_hashes": sealed["hashes"],
        "source_call_plan_sha256": plan["call_plan_sha256"],
        "source_scientific_registry_sha256": plan["scientific_registry_sha256"],
        "token_contract": {
            "rendered_contexts": len(rows),
            "answer_continuations_proved": sum(natural_contract_counts.values()),
            "all_prompt_style_tokens_decode_exactly": True,
            "all_prompt_style_tokens_differ_from_bare_continuation_tokens": True,
            "glyph_contracts": natural_contracts,
        },
        "summary": summary,
        "rows": rows,
        "source_model_calls_observed": v2.EXPECTED_CALLS,
        "model_calls_issued_by_posthoc": 0,
        "generation_calls_issued_by_posthoc": 0,
        "composition_calls_analyzed": 0,
        "claim_boundary": (
            "Post-hoc description of alternative contextual tokenization on sealed logits only; "
            "the V2 component failure remains terminal and no topology, activation-gap, latent-"
            "knowledge, biological, or physical-law claim is authorized."
        ),
    }


def analyze_and_write() -> dict[str, Any]:
    analysis = build_posthoc_analysis()
    _write_frozen_posthoc(OUTPUT, analysis)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    analysis = analyze_and_write()
    print(
        v2.canonical_json(
            {
                "status": analysis["status"],
                "source_terminal_status": analysis["source_terminal_status"],
                "model_calls_issued_by_posthoc": 0,
                "output": str(OUTPUT),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze the disposable V6A direct-component model qualification.

This analyzer performs no language-model forward.  It independently rebuilds
the fixture, rendering plan, tokenizer receipt, raw-logit diagnostics, and all
immutable hash bindings before computing the registered direct-component
gates.  Qualification contains no composition item, so this file cannot read
or report a composition result, TARGET-order gap, or topology interaction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_component_qualification"
    / "qwen2.5-7b-instruct"
)
BUILDER = ROOT / "signal" / "syntax" / "build_coherent_readout_v6a_component_qualification_bank.py"
FIXTURE = BUILDER.with_name("coherent_readout_v6a_component_qualification_bank.json")
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
RUNNER = ROOT / "eval" / "run_coherent_readout_v6a_component_qualification.py"
PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V6A_COMPONENT_QUALIFICATION_PREREG.md"
V6A_DESIGN = ROOT / "docs" / "COHERENT_READOUT_V6A_TOPOLOGY_IDENTIFICATION_DESIGN.md"
BUILDER_TEST = ROOT / "tests" / "test_build_coherent_readout_v6a_component_qualification_bank.py"
RUNNER_TEST = ROOT / "tests" / "test_run_coherent_readout_v6a_component_qualification.py"
ANALYZER_TEST = ROOT / "tests" / "test_analyze_coherent_readout_v6a_component_qualification.py"

DESIGN = RESULT_ROOT / "design.json"
PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"
DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
ATTEMPT = RESULT_ROOT / "qualification_baseline_attempt.json"
RECORDS = RESULT_ROOT / "qualification_baseline_records.jsonl"
EXECUTION_MANIFEST = RESULT_ROOT / "qualification_baseline_execution_manifest.json"
RAW_LOGIT_ROOT = RESULT_ROOT / "raw_logits" / "qualification-baseline"
ANALYSIS = RESULT_ROOT / "qualification_analysis.json"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEVICE = "mps"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MODEL_LOADING_DEVICE_MAP = DEVICE
EXPECTED_CALLS = 384
RAW_SHARD_ROWS = 64
EXPECTED_FAMILY_COUNTS = {"property_retrieval": 256, "codebook_lookup": 128}
ALLOWED_FAMILIES = frozenset(EXPECTED_FAMILY_COUNTS)
FACTOR_NAMES = ("p", "m", "r", "v", "o", "q", "a")
FAMILY_FACTORS = {
    "property_retrieval": ("p", "v", "o", "q", "a"),
    "codebook_lookup": ("p", "m", "r", "v"),
}
CHAT_FLAGS = {
    "add_generation_prompt": False,
    "continue_final_message": True,
    "enable_thinking": False,
}
PACKAGE_NAMES = (
    "accelerate",
    "huggingface-hub",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)

PLAN_SCHEMA = "coherent-readout-v6a-qualification-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v6a-qualification-design-v1"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v6a-qualification-plan-manifest-v1"
TOKENIZATION_SCHEMA = "coherent-readout-v6a-qualification-tokenization-v1"
DEPENDENCY_SCHEMA = "coherent-readout-v6a-qualification-dependencies-v1"
PROMPT_SCHEMA = "coherent-readout-v6a-qualification-prompt-v1"
ATTEMPT_SCHEMA = "coherent-readout-v6a-qualification-attempt-v1"
RECORD_SCHEMA = "coherent-readout-v6a-qualification-record-v1"
EXECUTION_SCHEMA = "coherent-readout-v6a-qualification-execution-v1"
ANALYSIS_SCHEMA = "coherent-readout-v6a-qualification-analysis-v1"

ENGINEERING_INVALID = "V6A_QUALIFICATION_ENGINEERING_INVALID"
COMPONENT_FAIL = "V6A_QUALIFICATION_COMPONENT_FAIL"
COMPONENT_PASS = "V6A_QUALIFICATION_COMPONENT_PASS"

OVERALL_ACCURACY_THRESHOLD = 0.98
MARGINAL_ACCURACY_THRESHOLD = 0.90

CLAIM_BOUNDARIES = {
    "supported_scope": "disposable_direct_component_model_qualification_only",
    "composition_inference": "forbidden",
    "target_order_gap_inference": "forbidden",
    "topology_interaction_inference": "forbidden",
    "activation_gap_inference": "forbidden",
    "biology_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
}


class V6AQualificationAnalysisError(ValueError):
    """Raised when immutable V6A qualification replay fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V6AQualificationAnalysisError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def f32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V6AQualificationAnalysisError(f"{label} is not a lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V6AQualificationAnalysisError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise V6AQualificationAnalysisError(f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise V6AQualificationAnalysisError(f"cannot read JSONL: {path}") from error
    if not values or any(not isinstance(value, dict) for value in values):
        raise V6AQualificationAnalysisError("qualification records are empty or malformed")
    return values


def _write_frozen_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise V6AQualificationAnalysisError(f"refusing to overwrite frozen qualification analysis: {path}")
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


def _builder_module() -> Any:
    if not BUILDER.is_file():
        raise V6AQualificationAnalysisError("qualification fixture builder is missing")
    spec = importlib.util.spec_from_file_location("v6a_qualification_builder_replay", BUILDER)
    if spec is None or spec.loader is None:
        raise V6AQualificationAnalysisError("qualification fixture builder cannot be imported")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise V6AQualificationAnalysisError("qualification fixture builder import failed") from error
    return module


def _load_and_rebuild_fixture() -> tuple[dict[str, Any], dict[str, Any], Any]:
    fixture = _load_json(FIXTURE)
    manifest = _load_json(FIXTURE_MANIFEST)
    builder = _builder_module()
    try:
        rebuilt = builder.build_fixture()
    except Exception as error:
        raise V6AQualificationAnalysisError("qualification fixture rebuild failed") from error
    if fixture != rebuilt:
        raise V6AQualificationAnalysisError("qualification fixture does not rebuild exactly")
    expected_manifest = {
        "schema_version": builder.MANIFEST_SCHEMA,
        "analysis_id": builder.ANALYSIS_ID,
        "design_date": builder.DESIGN_DATE,
        "fixture_path": builder.relative_path(FIXTURE),
        "fixture_file_sha256": file_sha256(FIXTURE),
        "fixture_canonical_sha256": canonical_sha256(fixture),
        "builder_path": builder.relative_path(BUILDER),
        "builder_file_sha256": file_sha256(BUILDER),
        "world_count": builder.WORLD_COUNT,
        "family_counts": builder.FAMILY_COUNTS,
        "cell_count": builder.EXPECTED_CALL_COUNT,
        "composition_cell_count": 0,
        "model_calls": 0,
        "tokenizer_calls": 0,
        "biological_model_calls": 0,
    }
    if manifest != expected_manifest:
        raise V6AQualificationAnalysisError("qualification fixture manifest changed")
    firewall = fixture.get("calibration_firewall", {})
    if (
        fixture.get("expected_call_count") != EXPECTED_CALLS
        or fixture.get("family_counts") != EXPECTED_FAMILY_COUNTS
        or firewall.get("composition_calls") != 0
        or firewall.get("composition_gap_or_topology_effect_available_to_selection") is not False
        or firewall.get("direct_retrieval_topology_accuracy_available") is not True
    ):
        raise V6AQualificationAnalysisError("qualification fixture firewall changed")
    cells = fixture.get("cells")
    worlds = fixture.get("worlds")
    if not isinstance(cells, list) or not isinstance(worlds, list):
        raise V6AQualificationAnalysisError("qualification fixture registry is malformed")
    if any(cell.get("family") not in ALLOWED_FAMILIES for cell in cells):
        raise V6AQualificationAnalysisError("qualification fixture contains a forbidden family")
    symbols = [symbol for world in worlds for symbol in world.get("symbols", [])]
    if (
        len(symbols) != 32
        or len(set(symbols)) != len(symbols)
        or set(symbols) & set(builder.PRIOR_ASCII_SYMBOLS)
        or set(symbols) & set(builder.V6A_RESERVED_SYMBOLS)
        or fixture.get("v6a_reserved_symbols_sha256") != canonical_sha256(sorted(builder.V6A_RESERVED_SYMBOLS))
    ):
        raise V6AQualificationAnalysisError("qualification symbol firewall changed")
    return fixture, manifest, builder


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    try:
        raw = list(value)
    except TypeError as error:
        raise V6AQualificationAnalysisError(f"{label} is not a vector") from error
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise V6AQualificationAnalysisError(f"{label} must contain one row")
        raw = raw[0]
    if not raw or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise V6AQualificationAnalysisError(f"{label} is not an integer vector")
    return [int(item) for item in raw]


def _contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(tokenizer.encode(rendered, add_special_tokens=False), "prompt")
    combined = _as_int_vector(
        tokenizer.encode(rendered + answer, add_special_tokens=False),
        "prompt plus answer",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise V6AQualificationAnalysisError(f"registered answer {answer!r} is not one continuation token")
    return combined[-1]


def _symbol_occurrence_receipts(
    rendered: str,
    offsets: Sequence[Sequence[int]],
    input_ids: Sequence[int],
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for symbol in symbols:
        starts = [index for index in range(len(rendered)) if rendered.startswith(symbol, index)]
        if not starts:
            raise V6AQualificationAnalysisError(f"registered symbol {symbol!r} is absent")
        for start in starts:
            end = start + len(symbol)
            matches = [index for index, pair in enumerate(offsets) if int(pair[0]) < end and int(pair[1]) > start]
            if len(matches) != 1:
                raise V6AQualificationAnalysisError("registered symbol occurrence is not exactly one prompt token")
            token_index = matches[0]
            token_start, token_end = map(int, offsets[token_index])
            if rendered[token_start:token_end].strip() != symbol:
                raise V6AQualificationAnalysisError("registered symbol shares a lexical prompt token")
            receipts.append(
                {
                    "symbol": symbol,
                    "character_start": start,
                    "character_end": end,
                    "token_index": token_index,
                    "token_id": int(input_ids[token_index]),
                }
            )
    return receipts


def _load_tokenizer(dependency: Mapping[str, Any]) -> Any:
    snapshot = Path(str(dependency.get("model", {}).get("snapshot_path", "")))
    if not snapshot.is_dir():
        raise V6AQualificationAnalysisError("locked model snapshot is missing")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise V6AQualificationAnalysisError("transformers is required for token replay") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as error:
        raise V6AQualificationAnalysisError("locked tokenizer cannot be loaded") from error
    if not getattr(tokenizer, "is_fast", False):
        raise V6AQualificationAnalysisError("locked tokenizer is not fast")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise V6AQualificationAnalysisError("locked tokenizer chat template is missing")
    return tokenizer


def _render_prompt(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_text = cell.get("prompt_text")
    if not isinstance(prompt_text, str) or cell.get("prompt_sha256") != text_sha256(prompt_text):
        raise V6AQualificationAnalysisError("qualification cell prompt hash changed")
    messages = [
        {"role": "system", "content": fixture["system_message"]},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": fixture["assistant_prefill"]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    except Exception as error:
        raise V6AQualificationAnalysisError("qualification chat rendering failed") from error
    if not isinstance(rendered, str) or not rendered.endswith(str(fixture["assistant_prefill"])):
        raise V6AQualificationAnalysisError("rendered chat does not end at assistant prefill")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = _as_int_vector(encoded["input_ids"], "rendered input IDs")
    offsets = [tuple(int(item) for item in pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets) or not ids:
        raise V6AQualificationAnalysisError("rendered token IDs and offsets differ")
    final_start, final_end = offsets[-1]
    if final_end != len(rendered) or rendered[final_start:final_end] != ":":
        raise V6AQualificationAnalysisError("final response site is not the ANSWER colon")
    expected_answer = str(cell["correct_answer"])
    distractor_answer = str(cell["distractor_answer"])
    expected_token_id = _contextual_token_id(tokenizer, rendered, expected_answer)
    distractor_token_id = _contextual_token_id(tokenizer, rendered, distractor_answer)
    if expected_token_id == distractor_token_id:
        raise V6AQualificationAnalysisError("registered answer tokens collapsed")
    matching_worlds = [world for world in fixture["worlds"] if world["world_id"] == cell["world_id"]]
    if len(matching_worlds) != 1:
        raise V6AQualificationAnalysisError("cell does not resolve to exactly one world")
    world_symbols = list(matching_worlds[0]["symbols"])
    present_symbols = sorted(symbol for symbol in world_symbols if symbol in prompt_text)
    if not present_symbols:
        raise V6AQualificationAnalysisError("prompt contains no registered world symbol")
    if not set(cell["answer_options"]) <= set(world_symbols):
        raise V6AQualificationAnalysisError("answer option is outside its world registry")
    occurrences = _symbol_occurrence_receipts(rendered, offsets, ids, present_symbols)
    core = {
        "schema_version": PROMPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "world_id": cell["world_id"],
        "family": cell["family"],
        "factors": cell["factors"],
        "factor_levels": cell["factor_levels"],
        "expected_answer": expected_answer,
        "distractor_answer": distractor_answer,
        "expected_token_id": expected_token_id,
        "distractor_token_id": distractor_token_id,
        "rendered_text_sha256": text_sha256(rendered),
        "execution_input_ids": ids,
        "execution_attention_mask": [1] * len(ids),
        "input_token_count": len(ids),
        "response_site_index": len(ids) - 1,
        "response_site_token_id": ids[-1],
        "response_site_text": ":",
        "registered_symbols_present": present_symbols,
        "symbol_occurrences": occurrences,
    }
    return {**core, "prompt_id": canonical_sha256(core)}


def _topology_shape_receipts(prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        if prompt["family"] != "property_retrieval":
            continue
        factors = prompt["factors"]
        groups[(prompt["world_id"], factors["p"], factors["v"])].append(prompt)
    if len(groups) != 32 or set(map(len, groups.values())) != {8}:
        raise V6AQualificationAnalysisError("retrieval topology-octet registry changed")
    receipts: list[dict[str, Any]] = []
    expected_vertices = {(o, q, a) for o in (-1, 1) for q in (-1, 1) for a in (-1, 1)}
    for key, members in sorted(groups.items()):
        observed_vertices = {
            (
                member["factors"]["o"],
                member["factors"]["q"],
                member["factors"]["a"],
            )
            for member in members
        }
        if observed_vertices != expected_vertices:
            raise V6AQualificationAnalysisError("topology octet vertices changed")
        shapes = {
            (
                member["input_token_count"],
                member["response_site_index"],
                member["response_site_token_id"],
            )
            for member in members
        }
        masks = {tuple(member["execution_attention_mask"]) for member in members}
        if len(shapes) != 1 or len(masks) != 1:
            raise V6AQualificationAnalysisError("topology octet response shape changed")
        receipts.append(
            {
                "world_id": key[0],
                "p": key[1],
                "v": key[2],
                "member_prompt_ids": sorted(member["prompt_id"] for member in members),
                "shape": list(next(iter(shapes))),
            }
        )
    return receipts


def _validate_symbol_token_contracts(
    prompts: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]
) -> list[dict[str, Any]]:
    registered = set(fixture["qualification_symbols"])
    prompt_ids: dict[str, set[int]] = defaultdict(set)
    continuation_ids: dict[str, set[int]] = defaultdict(set)
    prompt_occurrences: Counter[str] = Counter()
    continuation_contexts: Counter[str] = Counter()
    for prompt in prompts:
        for occurrence in prompt["symbol_occurrences"]:
            symbol = occurrence["symbol"]
            if symbol not in registered:
                raise V6AQualificationAnalysisError("unregistered symbol entered token receipt")
            prompt_ids[symbol].add(int(occurrence["token_id"]))
            prompt_occurrences[symbol] += 1
        for symbol, token_id in (
            (prompt["expected_answer"], prompt["expected_token_id"]),
            (prompt["distractor_answer"], prompt["distractor_token_id"]),
        ):
            if symbol not in registered:
                raise V6AQualificationAnalysisError("unregistered continuation answer")
            continuation_ids[symbol].add(int(token_id))
            continuation_contexts[symbol] += 1
    if set(prompt_ids) != registered or set(continuation_ids) != registered:
        raise V6AQualificationAnalysisError("registered symbol token coverage changed")
    if any(len(token_ids) != 1 for token_ids in prompt_ids.values()):
        raise V6AQualificationAnalysisError("prompt symbol token ID is context-unstable")
    if any(len(token_ids) != 1 for token_ids in continuation_ids.values()):
        raise V6AQualificationAnalysisError("continuation symbol token ID is context-unstable")
    return [
        {
            "symbol": symbol,
            "prompt_token_id": next(iter(prompt_ids[symbol])),
            "continuation_token_id": next(iter(continuation_ids[symbol])),
            "prompt_occurrence_count": prompt_occurrences[symbol],
            "continuation_context_count": continuation_contexts[symbol],
        }
        for symbol in fixture["qualification_symbols"]
    ]


def _raw_shard_specs(vocab_size: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, EXPECTED_CALLS, RAW_SHARD_ROWS)):
        stop = min(start + RAW_SHARD_ROWS, EXPECTED_CALLS)
        specs.append(
            {
                "index": index,
                "start_row": start,
                "stop_row": stop,
                "rows": stop - start,
                "shape": [stop - start, vocab_size],
                "dtype": "<f4",
                "path": str(RAW_LOGIT_ROOT / f"shard_{index:03d}.npy"),
            }
        )
    return specs


def _rebuild_plan_and_receipt(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_cells = fixture.get("cells")
    if not isinstance(raw_cells, list):
        raise V6AQualificationAnalysisError("qualification fixture cells are missing")
    cells = sorted((dict(cell) for cell in raw_cells), key=lambda cell: cell["cell_id"])
    prompts = [_render_prompt(tokenizer, fixture, cell) for cell in cells]
    if (
        len(prompts) != EXPECTED_CALLS
        or len({prompt["prompt_id"] for prompt in prompts}) != EXPECTED_CALLS
        or any(prompt["family"] not in ALLOWED_FAMILIES for prompt in prompts)
    ):
        raise V6AQualificationAnalysisError("qualification prompt allocation changed")
    shape_receipts = _topology_shape_receipts(prompts)
    symbol_token_contracts = _validate_symbol_token_contracts(prompts, fixture)
    config = dependency.get("model", {}).get("config", {})
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise V6AQualificationAnalysisError("locked vocabulary size is invalid")
    plan_core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": fixture["analysis_id"],
        "mode": fixture["mode"],
        "model": {
            "model_id": MODEL_ID,
            "snapshot_revision": dependency["model"]["snapshot_revision"],
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "vocab_size": vocab_size,
            "device": DEVICE,
            "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
            "model_dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "chat_flags": CHAT_FLAGS,
        "prompts": prompts,
        "cells": cells,
        "topology_shape_receipts": shape_receipts,
        "symbol_token_contracts": symbol_token_contracts,
        "raw_logits_shards": _raw_shard_specs(vocab_size),
        "expected_calls": EXPECTED_CALLS,
        "composition_calls": 0,
        "model_calls_before_plan_freeze": 0,
        "generation_used": False,
        "logits_to_keep": 1,
        "biological_model_calls": 0,
    }
    plan = {**plan_core, "call_plan_sha256": canonical_sha256(plan_core)}
    receipt_core = {
        "schema_version": TOKENIZATION_SCHEMA,
        "prompt_count": len(prompts),
        "topology_octet_count": len(shape_receipts),
        "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "chat_flags": CHAT_FLAGS,
        "response_site_token_ids": sorted({prompt["response_site_token_id"] for prompt in prompts}),
        "response_site_indices": sorted({prompt["response_site_index"] for prompt in prompts}),
        "answer_token_ids": sorted(
            {
                token_id
                for prompt in prompts
                for token_id in (prompt["expected_token_id"], prompt["distractor_token_id"])
            }
        ),
        "symbol_token_contracts": symbol_token_contracts,
        "input_token_count_min": min(prompt["input_token_count"] for prompt in prompts),
        "input_token_count_max": max(prompt["input_token_count"] for prompt in prompts),
        "prompt_receipts": [
            {
                "cell_id": prompt["cell_id"],
                "prompt_id": prompt["prompt_id"],
                "rendered_text_sha256": prompt["rendered_text_sha256"],
                "input_token_count": prompt["input_token_count"],
                "response_site_index": prompt["response_site_index"],
                "response_site_token_id": prompt["response_site_token_id"],
                "expected_token_id": prompt["expected_token_id"],
                "distractor_token_id": prompt["distractor_token_id"],
            }
            for prompt in prompts
        ],
        "model_calls": 0,
    }
    receipt = {**receipt_core, "canonical_sha256": canonical_sha256(receipt_core)}
    return plan, receipt


def _validate_dependency_lock(dependency: Mapping[str, Any]) -> None:
    expected_dependency_keys = {
        "schema_version",
        "python",
        "platform",
        "machine",
        "packages",
        "model",
        "implementation_files",
        "runtime",
        "canonical_sha256",
    }
    if set(dependency) != expected_dependency_keys:
        raise V6AQualificationAnalysisError("qualification dependency fields changed")
    if dependency.get("schema_version") != DEPENDENCY_SCHEMA:
        raise V6AQualificationAnalysisError("qualification dependency schema changed")
    core = {key: value for key, value in dependency.items() if key != "canonical_sha256"}
    if canonical_sha256(core) != dependency.get("canonical_sha256"):
        raise V6AQualificationAnalysisError("qualification dependency canonical hash changed")
    if (
        dependency.get("python") != platform.python_version()
        or dependency.get("platform") != platform.platform()
        or dependency.get("machine") != platform.machine()
    ):
        raise V6AQualificationAnalysisError("qualification Python/platform lock changed")
    packages = dependency.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(PACKAGE_NAMES):
        raise V6AQualificationAnalysisError("qualification package registry changed")
    for name in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise V6AQualificationAnalysisError(f"locked package is missing: {name}") from error
        if packages[name] != version:
            raise V6AQualificationAnalysisError(f"locked package version changed: {name}")
    try:
        import torch
        import transformers
    except ImportError as error:
        raise V6AQualificationAnalysisError("torch and transformers are required") from error
    expected_runtime = {
        "device": DEVICE,
        "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
        "model_dtype": MODEL_DTYPE,
        "stored_logits_dtype": "float32",
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "mps_is_built": bool(torch.backends.mps.is_built()),
        "mps_is_available": bool(torch.backends.mps.is_available()),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "transformers_version": transformers.__version__,
    }
    if dependency.get("runtime") != expected_runtime:
        raise V6AQualificationAnalysisError("qualification numerical runtime changed")
    model = dependency.get("model")
    if (
        not isinstance(model, Mapping)
        or set(model) != {"model_id", "snapshot_revision", "snapshot_path", "config", "assets"}
        or model.get("model_id") != MODEL_ID
        or model.get("snapshot_revision") != MODEL_REVISION
    ):
        raise V6AQualificationAnalysisError("qualification model lock changed")
    snapshot = Path(str(model.get("snapshot_path", "")))
    assets = model.get("assets")
    allowed_suffixes = {".json", ".jinja", ".model", ".safetensors", ".txt", ".tiktoken"}
    observed_asset_paths = (
        {
            str(path.relative_to(snapshot))
            for path in snapshot.rglob("*")
            if path.is_file() and path.suffix in allowed_suffixes
        }
        if snapshot.is_dir()
        else set()
    )
    if (
        not snapshot.is_dir()
        or not snapshot.is_absolute()
        or snapshot.resolve() != snapshot
        or snapshot.name != model.get("snapshot_revision")
        or not isinstance(assets, Mapping)
        or set(assets) != observed_asset_paths
        or not any(Path(relative).suffix == ".safetensors" for relative in observed_asset_paths)
    ):
        raise V6AQualificationAnalysisError("qualification model assets are missing")
    for relative, binding in assets.items():
        if (
            not isinstance(relative, str)
            or not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "size_bytes"}
            or isinstance(binding.get("size_bytes"), bool)
            or not isinstance(binding.get("size_bytes"), int)
            or binding["size_bytes"] < 0
        ):
            raise V6AQualificationAnalysisError("qualification model asset binding is malformed")
        path = snapshot / relative
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or file_sha256(path) != _validate_hash(binding.get("sha256"), str(path))
        ):
            raise V6AQualificationAnalysisError(f"qualification model asset changed: {path}")
    config = _load_json(snapshot / "config.json")
    expected_config_lock = {
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype"),
    }
    if model.get("config") != expected_config_lock:
        raise V6AQualificationAnalysisError("qualification model configuration changed")
    implementations = dependency.get("implementation_files")
    required_names = {
        "runner",
        "analyzer",
        "builder",
        "fixture",
        "fixture_manifest",
        "preregistration",
        "v6a_topology_design",
        "builder_test",
        "runner_test",
        "analyzer_test",
    }
    if not isinstance(implementations, Mapping) or set(implementations) != required_names:
        raise V6AQualificationAnalysisError("qualification implementation registry changed")
    expected_paths = {
        "runner": RUNNER,
        "analyzer": Path(__file__),
        "builder": BUILDER,
        "fixture": FIXTURE,
        "fixture_manifest": FIXTURE_MANIFEST,
        "preregistration": PREREGISTRATION,
        "v6a_topology_design": V6A_DESIGN,
        "builder_test": BUILDER_TEST,
        "runner_test": RUNNER_TEST,
        "analyzer_test": ANALYZER_TEST,
    }
    bound_paths: set[Path] = set()
    for name, binding in implementations.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise V6AQualificationAnalysisError(f"implementation binding is malformed: {name}")
        path = Path(str(binding.get("path", "")))
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise V6AQualificationAnalysisError(f"qualification implementation is missing: {name}") from error
        bound_paths.add(resolved)
        if path != expected_paths[name].resolve() or resolved != expected_paths[name].resolve():
            raise V6AQualificationAnalysisError(f"qualification implementation path changed: {name}")
        if not path.is_file() or file_sha256(path) != _validate_hash(binding.get("sha256"), str(path)):
            raise V6AQualificationAnalysisError(f"qualification implementation changed: {name}")
    if V6A_DESIGN.resolve() not in bound_paths:
        raise V6AQualificationAnalysisError("downstream V6A topology design is absent from the dependency lock")


def _load_and_validate_plan() -> dict[str, Any]:
    fixture, fixture_manifest, _ = _load_and_rebuild_fixture()
    dependency = _load_json(DEPENDENCY_LOCK)
    _validate_dependency_lock(dependency)
    tokenizer = _load_tokenizer(dependency)
    rebuilt_plan, rebuilt_receipt = _rebuild_plan_and_receipt(tokenizer, fixture, dependency)

    receipt = _load_json(TOKENIZATION_RECEIPT)
    if receipt != rebuilt_receipt:
        raise V6AQualificationAnalysisError("qualification tokenization receipt changed")
    design = _load_json(DESIGN)
    expected_design = {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": rebuilt_plan["analysis_id"],
        "mode": rebuilt_plan["mode"],
        "model": rebuilt_plan["model"],
        "call_plan_sha256": rebuilt_plan["call_plan_sha256"],
        "expected_calls": EXPECTED_CALLS,
        "composition_calls": 0,
        "claim_scope": "direct_component_model_qualification_only",
        "prohibited_selection_inputs": [
            "composition_accuracy",
            "composition_margin",
            "target_order_gap",
            "topology_interaction",
        ],
        "locks": {
            "fixture_file_sha256": file_sha256(FIXTURE),
            "fixture_manifest_file_sha256": file_sha256(FIXTURE_MANIFEST),
            "fixture_canonical_sha256": fixture_manifest["fixture_canonical_sha256"],
            "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
            "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
        },
        "model_calls": 0,
    }
    if design != expected_design:
        raise V6AQualificationAnalysisError("qualification design changed")
    manifest = _load_json(PLAN_MANIFEST)
    expected_manifest = {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": "QUALIFICATION_PLAN_FROZEN_NO_FORWARD",
        "call_plan_sha256": rebuilt_plan["call_plan_sha256"],
        "plan": rebuilt_plan,
        "design_file_sha256": file_sha256(DESIGN),
        "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
        "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
        "model_calls": 0,
        "composition_calls": 0,
    }
    if manifest != expected_manifest:
        raise V6AQualificationAnalysisError("qualification frozen plan changed")
    return {
        "fixture": fixture,
        "fixture_manifest": fixture_manifest,
        "dependency": dependency,
        "receipt": receipt,
        "design": design,
        "plan_manifest": manifest,
        "plan": rebuilt_plan,
    }


def _logical_array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype != np.dtype("<f4") or value.ndim != 2:
        raise V6AQualificationAnalysisError("raw-logit shard is not a float32 matrix")
    digest = hashlib.sha256()
    for start in range(0, len(value), 8):
        chunk = np.ascontiguousarray(value[start : start + 8], dtype="<f4")
        if not np.isfinite(chunk).all():
            raise V6AQualificationAnalysisError("raw-logit shard contains nonfinite values")
        digest.update(memoryview(chunk.view(np.uint8)))
    return digest.hexdigest()


def _load_raw_shards(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[np.ndarray]:
    bindings = manifest.get("raw_logits_shards")
    specs = plan.get("raw_logits_shards")
    if not isinstance(bindings, list) or not isinstance(specs, list):
        raise V6AQualificationAnalysisError("raw-logit shard registry is missing")
    if len(bindings) != len(specs) or len(bindings) != math.ceil(EXPECTED_CALLS / RAW_SHARD_ROWS):
        raise V6AQualificationAnalysisError("raw-logit shard count changed")
    expected_entries = {Path(str(spec["path"])) for spec in specs}
    try:
        observed_entries = set(RAW_LOGIT_ROOT.iterdir()) if RAW_LOGIT_ROOT.is_dir() else set()
    except OSError as error:
        raise V6AQualificationAnalysisError("cannot enumerate the raw-logit directory") from error
    if observed_entries != expected_entries:
        raise V6AQualificationAnalysisError("raw-logit directory contains unbound entries")
    arrays: list[np.ndarray] = []
    for observed, expected in zip(bindings, specs, strict=True):
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise V6AQualificationAnalysisError("raw-logit shard binding is malformed")
        expected_keys = set(expected) | {"file_sha256", "logical_sha256", "size_bytes"}
        if set(observed) != expected_keys or any(observed[key] != value for key, value in expected.items()):
            raise V6AQualificationAnalysisError("raw-logit shard plan binding changed")
        path = Path(str(observed["path"]))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != observed["size_bytes"]
            or file_sha256(path) != _validate_hash(observed["file_sha256"], str(path))
        ):
            raise V6AQualificationAnalysisError(f"raw-logit shard file changed: {path}")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise V6AQualificationAnalysisError(f"cannot load raw-logit shard: {path}") from error
        if (
            list(array.shape) != observed["shape"]
            or array.dtype != np.dtype("<f4")
            or _logical_array_sha256(array) != observed["logical_sha256"]
        ):
            raise V6AQualificationAnalysisError(f"raw-logit shard content changed: {path}")
        arrays.append(array)
    return arrays


def diagnostics_from_full_vocab(row: np.ndarray, record: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconstruct every stored full-vocabulary diagnostic."""

    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise V6AQualificationAnalysisError("raw full-vocabulary logit row is invalid")
    expected_id = record.get("expected_token_id")
    distractor_id = record.get("distractor_token_id")
    if (
        isinstance(expected_id, bool)
        or not isinstance(expected_id, int)
        or isinstance(distractor_id, bool)
        or not isinstance(distractor_id, int)
        or expected_id == distractor_id
        or not 0 <= expected_id < len(value)
        or not 0 <= distractor_id < len(value)
    ):
        raise V6AQualificationAnalysisError("registered answer-token IDs are invalid")
    expected_answer = record.get("expected_answer")
    distractor_answer = record.get("distractor_answer")
    if not all(isinstance(answer, str) and answer for answer in (expected_answer, distractor_answer)):
        raise V6AQualificationAnalysisError("registered answer labels are invalid")
    expected_logit = float(value[expected_id])
    distractor_logit = float(value[distractor_id])
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    peak = float(value.astype(np.float64).max())
    logsumexp = peak + math.log(float(np.exp(value.astype(np.float64) - peak).sum()))
    label_logsumexp = float(np.logaddexp(expected_logit, distractor_logit))
    if expected_logit > distractor_logit:
        predicted_answer: str | None = expected_answer
        predicted_id: int | None = expected_id
    elif distractor_logit > expected_logit:
        predicted_answer = distractor_answer
        predicted_id = distractor_id
    else:
        predicted_answer = None
        predicted_id = None
    return {
        "expected_logit": expected_logit,
        "distractor_logit": distractor_logit,
        "expected_minus_distractor_margin": expected_logit - distractor_logit,
        "predicted_answer": predicted_answer,
        "predicted_token_id": predicted_id,
        "answer_correct": expected_logit > distractor_logit,
        "answer_tie": expected_logit == distractor_logit,
        "greedy_token_id": maximum_ids[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "full_vocab_logits_sha256": f32_sha256(value),
    }


def _validate_diagnostics(record: Mapping[str, Any], row: np.ndarray) -> dict[str, Any]:
    expected = diagnostics_from_full_vocab(row, record)
    observed = record.get("diagnostics")
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise V6AQualificationAnalysisError("stored full-vocabulary diagnostic schema changed")
    for key, value in expected.items():
        candidate = observed[key]
        if isinstance(value, float):
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                raise V6AQualificationAnalysisError(f"stored diagnostic is not numeric: {key}")
            if not math.isclose(float(candidate), value, rel_tol=1e-12, abs_tol=1e-12):
                raise V6AQualificationAnalysisError(f"stored diagnostic does not reconstruct from raw logits: {key}")
        elif candidate != value:
            raise V6AQualificationAnalysisError(f"stored diagnostic does not reconstruct from raw logits: {key}")
    return expected


def _record_core(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "record_id"}


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    shards: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != EXPECTED_CALLS:
        raise V6AQualificationAnalysisError("qualification plan prompt registry changed")
    if len(records) != EXPECTED_CALLS:
        raise V6AQualificationAnalysisError("qualification record count changed")
    if len({record.get("record_id") for record in records}) != EXPECTED_CALLS:
        raise V6AQualificationAnalysisError("qualification record IDs are duplicated")
    validated: list[dict[str, Any]] = []
    for call_index, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
        if not isinstance(record, Mapping) or not isinstance(prompt, Mapping):
            raise V6AQualificationAnalysisError("qualification record or prompt is malformed")
        shard_index = call_index // RAW_SHARD_ROWS
        row_index = call_index % RAW_SHARD_ROWS
        if shard_index >= len(shards) or row_index >= len(shards[shard_index]):
            raise V6AQualificationAnalysisError("raw-logit row registry is out of range")
        row = np.ascontiguousarray(shards[shard_index][row_index], dtype="<f4")
        if record.get("raw_logits_row_sha256") != f32_sha256(row):
            raise V6AQualificationAnalysisError("raw-logit row hash changed")
        diagnostics = _validate_diagnostics(record, row)
        expected_core = {
            "schema_version": RECORD_SCHEMA,
            "phase": "qualification-baseline",
            "call_index": call_index,
            "call_plan_sha256": plan["call_plan_sha256"],
            "cell_id": prompt["cell_id"],
            "prompt_id": prompt["prompt_id"],
            "world_id": prompt["world_id"],
            "family": prompt["family"],
            "factors": prompt["factors"],
            "factor_levels": prompt["factor_levels"],
            "expected_answer": prompt["expected_answer"],
            "distractor_answer": prompt["distractor_answer"],
            "expected_token_id": prompt["expected_token_id"],
            "distractor_token_id": prompt["distractor_token_id"],
            "response_site_index": prompt["response_site_index"],
            "response_site_token_id": prompt["response_site_token_id"],
            "raw_logits_global_row": call_index,
            "raw_logits_shard_index": shard_index,
            "raw_logits_row_in_shard": row_index,
            "raw_logits_row_sha256": f32_sha256(row),
            "diagnostics": diagnostics,
            "teacher_forced_prompt_forward": True,
            "generation_used": False,
            "model_calls": 1,
            "composition_calls": 0,
            "biological_model_calls": 0,
        }
        if dict(_record_core(record)) != expected_core:
            raise V6AQualificationAnalysisError(
                "qualification record does not reconstruct from its plan and raw logits"
            )
        if record.get("record_id") != canonical_sha256(expected_core):
            raise V6AQualificationAnalysisError("qualification record canonical ID changed")
        if record["family"] not in ALLOWED_FAMILIES:
            raise V6AQualificationAnalysisError("qualification record contains composition")
        validated.append(dict(record))
    if Counter(record["family"] for record in validated) != Counter(EXPECTED_FAMILY_COUNTS):
        raise V6AQualificationAnalysisError("qualification record family counts changed")
    return validated


def _validate_attempt(
    attempt: Mapping[str, Any],
    plan: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "expected_calls",
        "partial_resume_allowed",
        "composition_calls",
        "model_id",
        "model_revision",
        "model_loading_device_map",
        "preflight",
        "locks",
    }
    if set(attempt) != expected_keys:
        raise V6AQualificationAnalysisError("qualification attempt fields changed")
    if (
        attempt["schema_version"] != ATTEMPT_SCHEMA
        or attempt["status"] != "QUALIFICATION_EXECUTION_ATTEMPT_STARTED_IMMUTABLE"
        or attempt["phase"] != "qualification-baseline"
        or attempt["call_plan_sha256"] != plan["call_plan_sha256"]
        or attempt["expected_calls"] != EXPECTED_CALLS
        or attempt["partial_resume_allowed"] is not False
        or attempt["composition_calls"] != 0
        or attempt["model_id"] != MODEL_ID
        or attempt["model_revision"] != MODEL_REVISION
        or attempt["model_loading_device_map"] != MODEL_LOADING_DEVICE_MAP
    ):
        raise V6AQualificationAnalysisError("qualification attempt identity changed")

    preflight = attempt["preflight"]
    preflight_keys = {
        "mps_bfloat16_kernel_pass",
        "model_safetensor_bytes",
        "mps_recommended_max_memory_bytes",
        "mps_allocated_before_bytes",
        "required_mps_headroom_bytes",
        "raw_logits_expected_bytes",
        "disk_free_bytes",
        "required_disk_free_bytes",
    }
    if not isinstance(preflight, Mapping) or set(preflight) != preflight_keys:
        raise V6AQualificationAnalysisError("qualification attempt preflight fields changed")
    integer_fields = preflight_keys - {"mps_bfloat16_kernel_pass"}
    if any(isinstance(preflight[field], bool) or not isinstance(preflight[field], int) for field in integer_fields):
        raise V6AQualificationAnalysisError("qualification attempt preflight values are not integers")
    if preflight["mps_bfloat16_kernel_pass"] is not True:
        raise V6AQualificationAnalysisError("qualification MPS bfloat16 preflight did not pass")

    model_safetensor_bytes = sum(
        int(asset["size_bytes"])
        for name, asset in dependency["model"]["assets"].items()
        if name.endswith(".safetensors")
    )
    required_mps_headroom_bytes = math.ceil(model_safetensor_bytes * 1.10)
    raw_logits_expected_bytes = sum(
        math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize for spec in plan["raw_logits_shards"]
    )
    required_disk_free_bytes = raw_logits_expected_bytes + 1024**3
    if (
        model_safetensor_bytes <= 0
        or preflight["model_safetensor_bytes"] != model_safetensor_bytes
        or preflight["required_mps_headroom_bytes"] != required_mps_headroom_bytes
        or preflight["raw_logits_expected_bytes"] != raw_logits_expected_bytes
        or preflight["required_disk_free_bytes"] != required_disk_free_bytes
        or preflight["mps_recommended_max_memory_bytes"] <= 0
        or preflight["mps_allocated_before_bytes"] < 0
        or preflight["disk_free_bytes"] < 0
        or preflight["mps_recommended_max_memory_bytes"] - preflight["mps_allocated_before_bytes"]
        < required_mps_headroom_bytes
        or preflight["disk_free_bytes"] < required_disk_free_bytes
    ):
        raise V6AQualificationAnalysisError("qualification attempt preflight values changed")

    expected_locks = {
        "plan_manifest_file_sha256": file_sha256(PLAN_MANIFEST),
        "design_file_sha256": file_sha256(DESIGN),
        "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
        "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
        "fixture_file_sha256": file_sha256(FIXTURE),
        "runner_file_sha256": dependency["implementation_files"]["runner"]["sha256"],
        "analyzer_file_sha256": dependency["implementation_files"]["analyzer"]["sha256"],
    }
    if attempt["locks"] != expected_locks:
        raise V6AQualificationAnalysisError("qualification attempt locks changed")


def _validate_execution(plan_bundle: Mapping[str, Any]) -> dict[str, Any]:
    plan = plan_bundle["plan"]
    dependency = plan_bundle["dependency"]
    manifest = _load_json(EXECUTION_MANIFEST)
    expected_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "phase_model_calls",
        "cumulative_model_calls",
        "composition_calls",
        "generation_used",
        "partial_resume_allowed",
        "records",
        "raw_logits_shards",
        "attempt",
        "plan_manifest_file_sha256",
        "design_file_sha256",
        "dependency_lock_file_sha256",
        "tokenization_receipt_file_sha256",
        "runner_file_sha256",
        "analyzer_file_sha256",
        "biological_model_calls",
    }
    if set(manifest) != expected_keys:
        raise V6AQualificationAnalysisError("qualification execution manifest fields changed")
    analyzer_binding = dependency["implementation_files"]["analyzer"]
    runner_binding = dependency["implementation_files"]["runner"]
    if (
        manifest["schema_version"] != EXECUTION_SCHEMA
        or manifest["status"] != "QUALIFICATION_EXECUTION_COMPLETE_NOT_ANALYZED"
        or manifest["phase"] != "qualification-baseline"
        or manifest["call_plan_sha256"] != plan["call_plan_sha256"]
        or manifest["phase_model_calls"] != EXPECTED_CALLS
        or manifest["cumulative_model_calls"] != EXPECTED_CALLS
        or manifest["composition_calls"] != 0
        or manifest["generation_used"] is not False
        or manifest["partial_resume_allowed"] is not False
        or manifest["biological_model_calls"] != 0
        or manifest["plan_manifest_file_sha256"] != file_sha256(PLAN_MANIFEST)
        or manifest["design_file_sha256"] != file_sha256(DESIGN)
        or manifest["dependency_lock_file_sha256"] != file_sha256(DEPENDENCY_LOCK)
        or manifest["tokenization_receipt_file_sha256"] != file_sha256(TOKENIZATION_RECEIPT)
        or manifest["runner_file_sha256"] != runner_binding["sha256"]
        or manifest["analyzer_file_sha256"] != analyzer_binding["sha256"]
    ):
        raise V6AQualificationAnalysisError("qualification execution identity changed")
    attempt_binding = manifest["attempt"]
    if not isinstance(attempt_binding, Mapping):
        raise V6AQualificationAnalysisError("qualification attempt binding is malformed")
    if set(attempt_binding) != {"path", "file_sha256"}:
        raise V6AQualificationAnalysisError("qualification attempt binding fields changed")
    attempt_path = Path(str(attempt_binding["path"]))
    if (
        attempt_path != ATTEMPT
        or not attempt_path.is_file()
        or file_sha256(attempt_path) != attempt_binding["file_sha256"]
    ):
        raise V6AQualificationAnalysisError("qualification attempt artifact changed")
    _validate_attempt(_load_json(attempt_path), plan, dependency)

    records_binding = manifest["records"]
    if not isinstance(records_binding, Mapping):
        raise V6AQualificationAnalysisError("qualification record binding is malformed")
    expected_record_keys = {"path", "count", "file_sha256", "canonical_sha256", "size_bytes"}
    if set(records_binding) != expected_record_keys:
        raise V6AQualificationAnalysisError("qualification record binding fields changed")
    records_path = Path(str(records_binding["path"]))
    if (
        records_path != RECORDS
        or not records_path.is_file()
        or records_path.stat().st_size != records_binding["size_bytes"]
        or file_sha256(records_path) != records_binding["file_sha256"]
    ):
        raise V6AQualificationAnalysisError("qualification records artifact changed")
    records = _load_jsonl(records_path)
    if records_binding["count"] != EXPECTED_CALLS or canonical_sha256(records) != records_binding["canonical_sha256"]:
        raise V6AQualificationAnalysisError("qualification records commitment changed")
    shards = _load_raw_shards(manifest, plan)
    validated_records = _validate_records(records, plan, shards)
    return {
        "manifest": manifest,
        "records": validated_records,
        "raw_shards": shards,
    }


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise V6AQualificationAnalysisError("cannot calculate accuracy for an empty group")
    correct = sum(bool(row["pairwise_answer_correct"]) for row in rows)
    return {"correct": correct, "n": len(rows), "accuracy": correct / len(rows)}


def component_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct registered component accuracy and all marginal gates."""

    rows: list[dict[str, Any]] = []
    for record in records:
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise V6AQualificationAnalysisError("qualification diagnostics are missing")
        maxima = diagnostics.get("maximum_token_ids")
        if not isinstance(maxima, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in maxima
        ):
            raise V6AQualificationAnalysisError("qualification maximum-token registry is invalid")
        expected_id = record["expected_token_id"]
        distractor_id = record["distractor_token_id"]
        unique_label_argmax = len(maxima) == 1 and maxima[0] in {expected_id, distractor_id}
        answer_correct = diagnostics.get("answer_correct")
        if not isinstance(answer_correct, bool):
            raise V6AQualificationAnalysisError("qualification answer_correct is not Boolean")
        row = dict(record)
        row["unique_label_argmax"] = unique_label_argmax
        row["pairwise_answer_correct"] = answer_correct
        rows.append(row)

    family_accuracy: dict[str, Any] = {}
    world_accuracy: dict[str, Any] = {}
    answer_accuracy: dict[str, Any] = {}
    factor_accuracy: dict[str, Any] = {}
    retrieval_scaffold_accuracy: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for family in sorted(ALLOWED_FAMILIES):
        members = [row for row in rows if row["family"] == family]
        family_accuracy[family] = _rate(members)
        worlds = sorted({str(row["world_id"]) for row in members})
        for world in worlds:
            key = f"{family}__world={world}"
            world_accuracy[key] = _rate([row for row in members if row["world_id"] == world])
        answers = sorted({str(row["expected_answer"]) for row in members})
        for answer in answers:
            key = f"{family}__answer={answer}"
            answer_accuracy[key] = _rate([row for row in members if row["expected_answer"] == answer])
        for factor in FAMILY_FACTORS[family]:
            for level in (-1, 1):
                key = f"{family}__{factor}={level:+d}"
                factor_accuracy[key] = _rate([row for row in members if row["factors"][factor] == level])
        gates[f"{family}_overall_accuracy"] = family_accuracy[family]["accuracy"] >= OVERALL_ACCURACY_THRESHOLD

    retrieval = [row for row in rows if row["family"] == "property_retrieval"]
    for q in (-1, 1):
        for a in (-1, 1):
            key = f"property_retrieval__q={q:+d}__a={a:+d}"
            retrieval_scaffold_accuracy[key] = _rate(
                [row for row in retrieval if row["factors"]["q"] == q and row["factors"]["a"] == a]
            )
            for o in (-1, 1):
                key = f"property_retrieval__o={o:+d}__q={q:+d}__a={a:+d}"
                retrieval_scaffold_accuracy[key] = _rate(
                    [
                        row
                        for row in retrieval
                        if row["factors"]["o"] == o and row["factors"]["q"] == q and row["factors"]["a"] == a
                    ]
                )

    gates["every_world_accuracy"] = all(
        value["accuracy"] >= MARGINAL_ACCURACY_THRESHOLD for value in world_accuracy.values()
    )
    gates["every_answer_label_accuracy"] = all(
        value["accuracy"] >= MARGINAL_ACCURACY_THRESHOLD for value in answer_accuracy.values()
    )
    gates["every_factor_level_accuracy"] = all(
        value["accuracy"] >= MARGINAL_ACCURACY_THRESHOLD for value in factor_accuracy.values()
    )
    gates["every_retrieval_q_by_a_accuracy"] = all(
        value["accuracy"] >= MARGINAL_ACCURACY_THRESHOLD
        for key, value in retrieval_scaffold_accuracy.items()
        if "__o=" not in key
    )
    gates["every_retrieval_o_by_q_by_a_accuracy"] = all(
        value["accuracy"] >= MARGINAL_ACCURACY_THRESHOLD
        for key, value in retrieval_scaffold_accuracy.items()
        if "__o=" in key
    )
    gates["every_row_unique_global_argmax_in_registered_pair"] = all(row["unique_label_argmax"] for row in rows)
    gates["every_row_no_answer_tie"] = all(row["diagnostics"]["answer_tie"] is False for row in rows)
    gates["pass"] = all(gates.values())
    return {
        "thresholds": {
            "family_overall_accuracy": OVERALL_ACCURACY_THRESHOLD,
            "world_answer_factor_marginal_accuracy": MARGINAL_ACCURACY_THRESHOLD,
            "retrieval_joint_scaffold_accuracy": MARGINAL_ACCURACY_THRESHOLD,
        },
        "exact_integer_requirements": {
            "property_retrieval_overall": "at_least_251_of_256",
            "codebook_lookup_overall": "at_least_126_of_128_before_stricter_label_gates",
            "property_retrieval_per_world": "at_least_29_of_32",
            "property_retrieval_per_answer_symbol": "at_least_15_of_16",
            "property_retrieval_per_factor_level": "at_least_116_of_128",
            "property_retrieval_per_q_by_a": "at_least_58_of_64",
            "property_retrieval_per_o_by_q_by_a": "at_least_29_of_32",
            "codebook_lookup_per_world": "at_least_15_of_16",
            "codebook_lookup_per_answer_symbol": "8_of_8",
            "codebook_lookup_per_factor_level": "at_least_58_of_64",
            "effective_codebook_lookup_overall": "128_of_128",
        },
        "family_accuracy": family_accuracy,
        "world_accuracy": world_accuracy,
        "answer_label_accuracy": answer_accuracy,
        "factor_level_accuracy": factor_accuracy,
        "retrieval_joint_scaffold_accuracy": retrieval_scaffold_accuracy,
        "unique_label_argmax_count": sum(row["unique_label_argmax"] for row in rows),
        "row_count": len(rows),
        "gates": gates,
    }


def analyze_qualification() -> dict[str, Any]:
    """Issue one terminal engineering-invalid, component-fail, or pass result."""

    try:
        plan_bundle = _load_and_validate_plan()
        execution = _validate_execution(plan_bundle)
        components = component_summary(execution["records"])
    except Exception as error:
        analysis = {
            "schema_version": ANALYSIS_SCHEMA,
            "status": ENGINEERING_INVALID,
            "engineering_valid": False,
            "component_qualified": False,
            "error": str(error),
            "model_calls_issued_by_analyzer": 0,
            "composition_calls_analyzed": 0,
            "claim_boundaries": CLAIM_BOUNDARIES,
        }
        _write_frozen_json(ANALYSIS, analysis)
        return analysis

    status = COMPONENT_PASS if components["gates"]["pass"] else COMPONENT_FAIL
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "engineering_valid": True,
        "component_qualified": status == COMPONENT_PASS,
        "components": components,
        "artifact_validation": {
            "call_plan_sha256": plan_bundle["plan"]["call_plan_sha256"],
            "plan_manifest_file_sha256": file_sha256(PLAN_MANIFEST),
            "design_file_sha256": file_sha256(DESIGN),
            "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
            "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
            "fixture_file_sha256": file_sha256(FIXTURE),
            "fixture_manifest_file_sha256": file_sha256(FIXTURE_MANIFEST),
            "execution_manifest_file_sha256": file_sha256(EXECUTION_MANIFEST),
            "records_file_sha256": file_sha256(RECORDS),
            "raw_float32_full_vocab_shard_count": len(execution["raw_shards"]),
            "tokenizer_contract_replayed": True,
            "response_site_contract_replayed": True,
            "topology_octet_shapes_replayed": True,
            "fixture_rebuilt_exactly": True,
            "composition_firewall_replayed": True,
        },
        "model_calls_observed": EXPECTED_CALLS,
        "model_calls_issued_by_analyzer": 0,
        "composition_calls_analyzed": 0,
        "generation_used": False,
        "biological_model_calls": 0,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    _write_frozen_json(ANALYSIS, analysis)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("analyze-qualification",),
        default="analyze-qualification",
    )
    args = parser.parse_args()
    result = analyze_qualification()
    print(canonical_json({"stage": args.stage, "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

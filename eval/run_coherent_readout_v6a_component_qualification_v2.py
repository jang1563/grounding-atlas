"""Plan and run the V2 engineering revision of V6A model qualification.

Planning uses the tokenizer but performs zero model forwards. Execution is a
single immutable 384-forward phase that stores full-vocabulary float32 logits.
The bank contains no composition prompts and cannot expose a topology gap.
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
import shutil
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
    / "coherent_readout_v6a_component_qualification_v2"
    / "qwen2.5-7b-instruct"
)
BUILDER = (
    ROOT
    / "signal"
    / "syntax"
    / "build_coherent_readout_v6a_component_qualification_bank.py"
)
FIXTURE = BUILDER.with_name("coherent_readout_v6a_component_qualification_bank.json")
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V6A_COMPONENT_QUALIFICATION_V2_PREREG.md"
V1_PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V6A_COMPONENT_QUALIFICATION_PREREG.md"
V6A_TOPOLOGY_DESIGN = ROOT / "docs" / "COHERENT_READOUT_V6A_TOPOLOGY_IDENTIFICATION_DESIGN.md"
DEFAULT_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v6a_component_qualification_v2.py"
BUILDER_TEST = ROOT / "tests" / "test_build_coherent_readout_v6a_component_qualification_bank.py"
RUNNER_TEST = ROOT / "tests" / "test_run_coherent_readout_v6a_component_qualification_v2.py"
ANALYZER_TEST = ROOT / "tests" / "test_analyze_coherent_readout_v6a_component_qualification_v2.py"

V1_RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_component_qualification"
    / "qwen2.5-7b-instruct"
)
V1_PLAN_MANIFEST = V1_RESULT_ROOT / "plan_manifest.json"
V1_DESIGN = V1_RESULT_ROOT / "design.json"
V1_DEPENDENCY_LOCK = V1_RESULT_ROOT / "dependency_lock.json"
V1_TOKENIZATION_RECEIPT = V1_RESULT_ROOT / "tokenization_receipt.json"
V1_ATTEMPT = V1_RESULT_ROOT / "qualification_baseline_attempt.json"
V1_ANALYSIS = V1_RESULT_ROOT / "qualification_analysis.json"
V1_RECORDS = V1_RESULT_ROOT / "qualification_baseline_records.jsonl"
V1_EXECUTION_MANIFEST = V1_RESULT_ROOT / "qualification_baseline_execution_manifest.json"
V1_RAW_LOGIT_ROOT = V1_RESULT_ROOT / "raw_logits" / "qualification-baseline"
V1_PLAN_MANIFEST_SHA256 = "0be35c0d6635924658985b5ea47b6f02b09c4245e10ba00bd28735e00eedaaf9"
V1_DESIGN_SHA256 = "7a252615bd7d8927fffd0b6b20a98a61d4e4430442da235e9027bf1b71648e7c"
V1_DEPENDENCY_LOCK_SHA256 = "e7efebc16a4b14522e2a305d82150425c17e74f99b68fb7186264b466096b391"
V1_TOKENIZATION_RECEIPT_SHA256 = (
    "d3e95323377e67ddcfab9ffa713c78ba4e8db28de2c2c16c909ab35df0fc3162"
)
V1_ATTEMPT_SHA256 = "fb13237f77995932b3fb9fced04ddc1395d123a7e56f3592b1bd0657e911af3b"
V1_ANALYSIS_SHA256 = "34f1a622862b54c66e99882b014706ba117390153b358361953ab6293c60bec8"

DESIGN = RESULT_ROOT / "design.json"
PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"
DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
ATTEMPT = RESULT_ROOT / "qualification_baseline_attempt.json"
RECORDS = RESULT_ROOT / "qualification_baseline_records.jsonl"
EXECUTION_MANIFEST = RESULT_ROOT / "qualification_baseline_execution_manifest.json"
RAW_LOGIT_ROOT = RESULT_ROOT / "raw_logits" / "qualification-baseline"
ANALYSIS = RESULT_ROOT / "qualification_analysis.json"
LOADER_SMOKE_RECEIPT = RESULT_ROOT / "loader_smoke_receipt.json"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEVICE = "mps"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MODEL_LOADING_DEVICE_MAP = DEVICE
MPS_ALLOCATOR_WARMUP_POLICY = "skip_transformers_warmup_only_for_all_mps_map_v2"
RAW_SHARD_ROWS = 64
EXPECTED_CALLS = 384
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

PLAN_SCHEMA = "coherent-readout-v6a-qualification-plan-v2"
DESIGN_SCHEMA = "coherent-readout-v6a-qualification-design-v2"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v6a-qualification-plan-manifest-v2"
TOKENIZATION_SCHEMA = "coherent-readout-v6a-qualification-tokenization-v2"
DEPENDENCY_SCHEMA = "coherent-readout-v6a-qualification-dependencies-v2"
PROMPT_SCHEMA = "coherent-readout-v6a-qualification-prompt-v1"
ATTEMPT_SCHEMA = "coherent-readout-v6a-qualification-attempt-v2"
RECORD_SCHEMA = "coherent-readout-v6a-qualification-record-v2"
EXECUTION_SCHEMA = "coherent-readout-v6a-qualification-execution-v2"


class V6AQualificationRunnerError(RuntimeError):
    """Raised when a qualification planning or execution lock fails."""


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    if path.exists():
        raise V6AQualificationRunnerError(f"refusing to overwrite differing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AQualificationRunnerError(f"stale temporary artifact exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V6AQualificationRunnerError(f"refusing to overwrite array artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AQualificationRunnerError(f"stale array temporary exists: {temporary}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise V6AQualificationRunnerError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V6AQualificationRunnerError(f"artifact is not an object: {path}")
    return value


def _builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("v6a_qualification_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise V6AQualificationRunnerError("qualification builder cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_and_rebuild_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    observed = _load_json(FIXTURE)
    manifest = _load_json(FIXTURE_MANIFEST)
    builder = _builder_module()
    rebuilt = builder.build_fixture()
    if observed != rebuilt:
        raise V6AQualificationRunnerError("qualification fixture does not rebuild exactly")
    if manifest.get("fixture_file_sha256") != file_sha256(FIXTURE):
        raise V6AQualificationRunnerError("qualification fixture file hash changed")
    if manifest.get("fixture_canonical_sha256") != canonical_sha256(observed):
        raise V6AQualificationRunnerError("qualification fixture canonical hash changed")
    if manifest.get("builder_file_sha256") != file_sha256(BUILDER):
        raise V6AQualificationRunnerError("qualification builder hash changed")
    if observed.get("expected_call_count") != EXPECTED_CALLS:
        raise V6AQualificationRunnerError("qualification call count changed")
    if any(cell.get("family") == "composition" for cell in observed.get("cells", [])):
        raise V6AQualificationRunnerError("qualification fixture contains composition")
    return observed, manifest


def _snapshot_path() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise V6AQualificationRunnerError("huggingface-hub is required") from error
    path = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.model",
                "*.safetensors",
                "*.txt",
                "*.tiktoken",
            ],
        )
    ).resolve()
    if not path.is_dir():
        raise V6AQualificationRunnerError("cached model snapshot is missing")
    if path.name != MODEL_REVISION:
        raise V6AQualificationRunnerError("cached model revision changed")
    return path


def _model_assets(snapshot: Path) -> dict[str, Any]:
    allowed_suffixes = {".json", ".jinja", ".model", ".safetensors", ".txt", ".tiktoken"}
    files = sorted(
        path for path in snapshot.rglob("*") if path.is_file() and path.suffix in allowed_suffixes
    )
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise V6AQualificationRunnerError("cached model assets are incomplete")
    return {
        str(path.relative_to(snapshot)): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    }


def _implementation_paths(analyzer_path: Path) -> dict[str, Path]:
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError as error:
        raise V6AQualificationRunnerError("transformers modeling utilities are required") from error
    paths = {
        "runner": Path(__file__).resolve(),
        "analyzer": analyzer_path.resolve(),
        "builder": BUILDER.resolve(),
        "fixture": FIXTURE.resolve(),
        "fixture_manifest": FIXTURE_MANIFEST.resolve(),
        "preregistration": PREREGISTRATION.resolve(),
        "v1_preregistration": V1_PREREGISTRATION.resolve(),
        "v1_plan_manifest": V1_PLAN_MANIFEST.resolve(),
        "v1_design": V1_DESIGN.resolve(),
        "v1_dependency_lock": V1_DEPENDENCY_LOCK.resolve(),
        "v1_tokenization_receipt": V1_TOKENIZATION_RECEIPT.resolve(),
        "v1_attempt": V1_ATTEMPT.resolve(),
        "v1_analysis": V1_ANALYSIS.resolve(),
        "v6a_topology_design": V6A_TOPOLOGY_DESIGN.resolve(),
        "builder_test": BUILDER_TEST.resolve(),
        "runner_test": RUNNER_TEST.resolve(),
        "analyzer_test": ANALYZER_TEST.resolve(),
        "transformers_modeling_utils": Path(modeling_utils.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise V6AQualificationRunnerError(f"implementation files are missing: {missing}")
    return paths


def _validate_v1_dependency_identity(
    model: Mapping[str, Any], implementation_files: Mapping[str, Any]
) -> None:
    validate_v1_engineering_stop_hashes_only()
    v1_dependency = _load_json(V1_DEPENDENCY_LOCK)
    if model != v1_dependency.get("model"):
        raise V6AQualificationRunnerError("V2 model assets or configuration differ from V1")
    v1_implementations = v1_dependency.get("implementation_files")
    if not isinstance(v1_implementations, Mapping):
        raise V6AQualificationRunnerError("V1 implementation registry is malformed")
    shared_crosswalk = {
        "builder": "builder",
        "fixture": "fixture",
        "fixture_manifest": "fixture_manifest",
        "v1_preregistration": "preregistration",
        "v6a_topology_design": "v6a_topology_design",
        "builder_test": "builder_test",
    }
    for v2_name, v1_name in shared_crosswalk.items():
        v2_binding = implementation_files.get(v2_name)
        v1_binding = v1_implementations.get(v1_name)
        if (
            not isinstance(v2_binding, Mapping)
            or not isinstance(v1_binding, Mapping)
            or v2_binding.get("sha256") != v1_binding.get("sha256")
        ):
            raise V6AQualificationRunnerError(
                f"V2 shared scientific dependency differs from V1: {v2_name}"
            )


def dependency_lock(analyzer_path: Path = DEFAULT_ANALYZER) -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise V6AQualificationRunnerError("torch and transformers are required") from error
    snapshot = _snapshot_path()
    config = _load_json(snapshot / "config.json")
    paths = _implementation_paths(analyzer_path)
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise V6AQualificationRunnerError(f"required package is missing: {name}") from error
    core = {
        "schema_version": DEPENDENCY_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "model": {
            "model_id": MODEL_ID,
            "snapshot_revision": snapshot.name,
            "snapshot_path": str(snapshot),
            "config": {
                "architectures": config.get("architectures"),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "vocab_size": config.get("vocab_size"),
                "torch_dtype": config.get("torch_dtype"),
            },
            "assets": _model_assets(snapshot),
        },
        "implementation_files": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
        },
        "runtime": {
            "device": DEVICE,
            "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
            "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
            "model_dtype": MODEL_DTYPE,
            "stored_logits_dtype": "float32",
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "transformers_version": transformers.__version__,
        },
    }
    if not core["runtime"]["mps_is_built"] or not core["runtime"]["mps_is_available"]:
        raise V6AQualificationRunnerError("MPS is not available")
    _validate_v1_dependency_identity(core["model"], core["implementation_files"])
    return {**core, "canonical_sha256": canonical_sha256(core)}


def _load_tokenizer(snapshot: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise V6AQualificationRunnerError("transformers is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise V6AQualificationRunnerError("fast tokenizer is required")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise V6AQualificationRunnerError("chat template is missing")
    return tokenizer


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise V6AQualificationRunnerError(f"{label} must contain one row")
        raw = raw[0]
    if not raw or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise V6AQualificationRunnerError(f"{label} is not an integer vector")
    return [int(item) for item in raw]


def contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(tokenizer.encode(rendered, add_special_tokens=False), "prompt")
    combined = _as_int_vector(
        tokenizer.encode(rendered + answer, add_special_tokens=False), "prompt plus answer"
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise V6AQualificationRunnerError(f"answer {answer!r} is not one continuation token")
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
            raise V6AQualificationRunnerError(f"registered symbol {symbol!r} is absent")
        for start in starts:
            end = start + len(symbol)
            matches = [
                index
                for index, pair in enumerate(offsets)
                if int(pair[0]) < end and int(pair[1]) > start
            ]
            if len(matches) != 1:
                raise V6AQualificationRunnerError("symbol occurrence is not exactly one prompt token")
            token_index = matches[0]
            token_start, token_end = map(int, offsets[token_index])
            if rendered[token_start:token_end].strip() != symbol:
                raise V6AQualificationRunnerError("symbol shares a lexical prompt token")
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


def render_prompt(tokenizer: Any, fixture: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    prompt_text = cell.get("prompt_text")
    if not isinstance(prompt_text, str) or cell.get("prompt_sha256") != text_sha256(prompt_text):
        raise V6AQualificationRunnerError("cell prompt hash changed")
    messages = [
        {"role": "system", "content": fixture["system_message"]},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": fixture["assistant_prefill"]},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    if not isinstance(rendered, str) or not rendered.endswith(str(fixture["assistant_prefill"])):
        raise V6AQualificationRunnerError("rendered chat does not end at assistant prefill")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = _as_int_vector(encoded["input_ids"], "rendered input IDs")
    offsets = [tuple(int(item) for item in pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets) or not ids:
        raise V6AQualificationRunnerError("rendered token IDs and offsets differ")
    final_start, final_end = offsets[-1]
    if final_end != len(rendered) or rendered[final_start:final_end] != ":":
        raise V6AQualificationRunnerError("final response site is not the ANSWER colon")
    expected_answer = str(cell["correct_answer"])
    distractor_answer = str(cell["distractor_answer"])
    expected_token_id = contextual_token_id(tokenizer, rendered, expected_answer)
    distractor_token_id = contextual_token_id(tokenizer, rendered, distractor_answer)
    if expected_token_id == distractor_token_id:
        raise V6AQualificationRunnerError("answer tokens collapsed")
    matching_worlds = [
        world for world in fixture["worlds"] if world["world_id"] == cell["world_id"]
    ]
    if len(matching_worlds) != 1:
        raise V6AQualificationRunnerError("cell does not resolve to exactly one world")
    world_symbols = list(matching_worlds[0]["symbols"])
    present_symbols = sorted(symbol for symbol in world_symbols if symbol in prompt_text)
    if not present_symbols:
        raise V6AQualificationRunnerError("prompt contains no registered world symbol")
    if not set(cell["answer_options"]) <= set(world_symbols):
        raise V6AQualificationRunnerError("answer option is outside its world registry")
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


def _validate_topology_shapes(prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        if prompt["family"] != "property_retrieval":
            continue
        factors = prompt["factors"]
        key = (prompt["world_id"], factors["p"], factors["v"])
        groups[key].append(prompt)
    if len(groups) != 8 * 4 or set(map(len, groups.values())) != {8}:
        raise V6AQualificationRunnerError("retrieval topology-octet registry changed")
    receipts = []
    expected_vertices = {
        (o, q, a) for o in (-1, 1) for q in (-1, 1) for a in (-1, 1)
    }
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
            raise V6AQualificationRunnerError("topology octet vertices changed")
        shapes = {
            (
                member["input_token_count"],
                member["response_site_index"],
                member["response_site_token_id"],
            )
            for member in members
        }
        if len(shapes) != 1:
            raise V6AQualificationRunnerError("topology octet response shape changed")
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
                raise V6AQualificationRunnerError("unregistered symbol entered token receipt")
            prompt_ids[symbol].add(int(occurrence["token_id"]))
            prompt_occurrences[symbol] += 1
        for symbol, token_id in (
            (prompt["expected_answer"], prompt["expected_token_id"]),
            (prompt["distractor_answer"], prompt["distractor_token_id"]),
        ):
            if symbol not in registered:
                raise V6AQualificationRunnerError("unregistered continuation answer")
            continuation_ids[symbol].add(int(token_id))
            continuation_contexts[symbol] += 1
    if set(prompt_ids) != registered or set(continuation_ids) != registered:
        raise V6AQualificationRunnerError("registered symbol token coverage changed")
    if any(len(token_ids) != 1 for token_ids in prompt_ids.values()):
        raise V6AQualificationRunnerError("prompt symbol token ID is context-unstable")
    if any(len(token_ids) != 1 for token_ids in continuation_ids.values()):
        raise V6AQualificationRunnerError("continuation symbol token ID is context-unstable")
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


def raw_shard_specs(vocab_size: int) -> list[dict[str, Any]]:
    specs = []
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


def validate_v1_engineering_stop_hashes_only() -> None:
    """Validate the sealed V1 file set without parsing any scientific prompt record."""
    expected_files = {
        V1_PLAN_MANIFEST: V1_PLAN_MANIFEST_SHA256,
        V1_DESIGN: V1_DESIGN_SHA256,
        V1_DEPENDENCY_LOCK: V1_DEPENDENCY_LOCK_SHA256,
        V1_TOKENIZATION_RECEIPT: V1_TOKENIZATION_RECEIPT_SHA256,
        V1_ATTEMPT: V1_ATTEMPT_SHA256,
        V1_ANALYSIS: V1_ANALYSIS_SHA256,
    }
    for path, expected_hash in expected_files.items():
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise V6AQualificationRunnerError(f"V1 engineering-stop artifact changed: {path}")
    if any(
        path.exists()
        for path in (V1_RECORDS, V1_EXECUTION_MANIFEST, V1_RAW_LOGIT_ROOT)
    ):
        raise V6AQualificationRunnerError("V1 unexpectedly contains scientific execution output")


def validate_v1_engineering_stop() -> dict[str, Any]:
    validate_v1_engineering_stop_hashes_only()
    manifest = _load_json(V1_PLAN_MANIFEST)
    design = _load_json(V1_DESIGN)
    dependency = _load_json(V1_DEPENDENCY_LOCK)
    receipt = _load_json(V1_TOKENIZATION_RECEIPT)
    attempt = _load_json(V1_ATTEMPT)
    analysis = _load_json(V1_ANALYSIS)
    plan = manifest.get("plan")
    dependency_core = {
        key: value for key, value in dependency.items() if key != "canonical_sha256"
    }
    receipt_core = {key: value for key, value in receipt.items() if key != "canonical_sha256"}
    if (
        manifest.get("status") != "QUALIFICATION_PLAN_FROZEN_NO_FORWARD"
        or not isinstance(plan, dict)
        or manifest.get("design_file_sha256") != V1_DESIGN_SHA256
        or manifest.get("dependency_lock_file_sha256") != V1_DEPENDENCY_LOCK_SHA256
        or manifest.get("tokenization_receipt_file_sha256")
        != V1_TOKENIZATION_RECEIPT_SHA256
        or plan.get("expected_calls") != EXPECTED_CALLS
        or plan.get("composition_calls") != 0
        or plan.get("model_calls_before_plan_freeze") != 0
        or design.get("call_plan_sha256") != plan.get("call_plan_sha256")
        or design.get("locks", {}).get("dependency_lock_file_sha256")
        != V1_DEPENDENCY_LOCK_SHA256
        or design.get("locks", {}).get("tokenization_receipt_file_sha256")
        != V1_TOKENIZATION_RECEIPT_SHA256
        or design.get("locks", {}).get("fixture_file_sha256") != file_sha256(FIXTURE)
        or design.get("locks", {}).get("fixture_manifest_file_sha256")
        != file_sha256(FIXTURE_MANIFEST)
        or canonical_sha256(dependency_core) != dependency.get("canonical_sha256")
        or canonical_sha256(receipt_core) != receipt.get("canonical_sha256")
        or attempt.get("call_plan_sha256") != plan.get("call_plan_sha256")
        or attempt.get("composition_calls") != 0
        or attempt.get("locks", {}).get("design_file_sha256") != V1_DESIGN_SHA256
        or attempt.get("locks", {}).get("dependency_lock_file_sha256")
        != V1_DEPENDENCY_LOCK_SHA256
        or attempt.get("locks", {}).get("tokenization_receipt_file_sha256")
        != V1_TOKENIZATION_RECEIPT_SHA256
        or analysis.get("status") != "V6A_QUALIFICATION_ENGINEERING_INVALID"
        or analysis.get("engineering_valid") is not False
        or analysis.get("model_calls_issued_by_analyzer") != 0
        or analysis.get("composition_calls_analyzed") != 0
    ):
        raise V6AQualificationRunnerError("V1 engineering-stop content changed")
    return plan


def build_plan(
    tokenizer: Any, fixture: Mapping[str, Any], dependency: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = sorted((dict(cell) for cell in fixture["cells"]), key=lambda cell: cell["cell_id"])
    family_counts = Counter(cell.get("family") for cell in cells)
    expected_family_counts = Counter(fixture["family_counts"])
    if family_counts != expected_family_counts or set(family_counts) != {
        "property_retrieval",
        "codebook_lookup",
    }:
        raise V6AQualificationRunnerError("qualification family registry changed")
    if any(cell.get("family") == "composition" for cell in cells):
        raise V6AQualificationRunnerError("qualification call plan contains composition")
    prompts = [render_prompt(tokenizer, fixture, cell) for cell in cells]
    if len(prompts) != EXPECTED_CALLS or len({prompt["prompt_id"] for prompt in prompts}) != len(
        prompts
    ):
        raise V6AQualificationRunnerError("rendered prompt allocation changed")
    shape_receipts = _validate_topology_shapes(prompts)
    symbol_token_contracts = _validate_symbol_token_contracts(prompts, fixture)
    v1_plan = validate_v1_engineering_stop()
    scientific_registry = {"cells": cells, "prompts": prompts}
    v1_scientific_registry = {
        "cells": v1_plan.get("cells"),
        "prompts": v1_plan.get("prompts"),
    }
    if scientific_registry != v1_scientific_registry:
        raise V6AQualificationRunnerError("V2 scientific prompt registry differs from V1")
    scientific_registry_sha256 = canonical_sha256(scientific_registry)
    config = dependency["model"]["config"]
    vocab_size = config["vocab_size"]
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise V6AQualificationRunnerError("model vocabulary size is invalid")
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
            "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
            "model_dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "chat_flags": CHAT_FLAGS,
        "prompts": prompts,
        "cells": cells,
        "topology_shape_receipts": shape_receipts,
        "symbol_token_contracts": symbol_token_contracts,
        "raw_logits_shards": raw_shard_specs(vocab_size),
        "execution_revision": "v2_mps_allocator_warmup_bypass",
        "v1_call_plan_sha256": v1_plan["call_plan_sha256"],
        "scientific_registry_sha256": scientific_registry_sha256,
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


def _downstream_paths() -> list[Path]:
    return [
        ATTEMPT,
        RECORDS,
        EXECUTION_MANIFEST,
        ANALYSIS,
        RAW_LOGIT_ROOT,
        *[Path(spec["path"]) for spec in raw_shard_specs(1)],
    ]


def validate_loader_smoke_receipt(dependency: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(LOADER_SMOKE_RECEIPT)
    expected_keys = {
        "status",
        "execution_revision",
        "model_id",
        "model_revision",
        "mps_allocator_warmup_policy",
        "warmup_bypass_calls",
        "model_calls",
        "generation_used",
        "fixture_prompt_records_consumed",
        "composition_calls",
        "dependency_canonical_sha256",
    }
    if set(receipt) != expected_keys or (
        receipt["status"] != "V2_LOADER_SMOKE_PASS_ZERO_FORWARD"
        or receipt["execution_revision"] != "v2_mps_allocator_warmup_bypass"
        or receipt["model_id"] != MODEL_ID
        or receipt["model_revision"] != MODEL_REVISION
        or receipt["mps_allocator_warmup_policy"] != MPS_ALLOCATOR_WARMUP_POLICY
        or receipt["warmup_bypass_calls"] != 1
        or receipt["model_calls"] != 0
        or receipt["generation_used"] is not False
        or receipt["fixture_prompt_records_consumed"] != 0
        or receipt["composition_calls"] != 0
        or receipt["dependency_canonical_sha256"] != dependency["canonical_sha256"]
    ):
        raise V6AQualificationRunnerError("V2 loader-smoke receipt changed")
    return receipt


def _design_payload(
    plan: Mapping[str, Any], fixture_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "mode": plan["mode"],
        "model": plan["model"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "expected_calls": EXPECTED_CALLS,
        "composition_calls": 0,
        "execution_revision": plan["execution_revision"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
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
            "v1_plan_manifest_file_sha256": file_sha256(V1_PLAN_MANIFEST),
            "v1_attempt_file_sha256": file_sha256(V1_ATTEMPT),
            "v1_analysis_file_sha256": file_sha256(V1_ANALYSIS),
            "loader_smoke_receipt_file_sha256": file_sha256(LOADER_SMOKE_RECEIPT),
        },
        "model_calls": 0,
    }


def _plan_manifest_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": "QUALIFICATION_PLAN_FROZEN_NO_FORWARD",
        "call_plan_sha256": plan["call_plan_sha256"],
        "plan": dict(plan),
        "design_file_sha256": file_sha256(DESIGN),
        "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
        "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
        "loader_smoke_receipt_file_sha256": file_sha256(LOADER_SMOKE_RECEIPT),
        "model_calls": 0,
        "composition_calls": 0,
        "execution_revision": plan["execution_revision"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
    }


def run_plan(analyzer_path: Path = DEFAULT_ANALYZER) -> dict[str, Any]:
    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise V6AQualificationRunnerError("only the preregistered default analyzer is allowed")
    if any(path.exists() for path in _downstream_paths()):
        raise V6AQualificationRunnerError("execution artifacts exist before qualification plan")
    fixture, fixture_manifest = load_and_rebuild_fixture()
    dependency = dependency_lock(analyzer_path)
    validate_loader_smoke_receipt(dependency)
    tokenizer = _load_tokenizer(Path(dependency["model"]["snapshot_path"]))
    plan, receipt = build_plan(tokenizer, fixture, dependency)
    _atomic_write(DEPENDENCY_LOCK, _artifact_bytes(dependency))
    _atomic_write(TOKENIZATION_RECEIPT, _artifact_bytes(receipt))
    design = _design_payload(plan, fixture_manifest)
    _atomic_write(DESIGN, _artifact_bytes(design))
    manifest = _plan_manifest_payload(plan)
    _atomic_write(PLAN_MANIFEST, _artifact_bytes(manifest))
    return manifest


def validate_frozen_plan() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design = _load_json(DESIGN)
    manifest = _load_json(PLAN_MANIFEST)
    receipt = _load_json(TOKENIZATION_RECEIPT)
    dependency = _load_json(DEPENDENCY_LOCK)
    if manifest.get("status") != "QUALIFICATION_PLAN_FROZEN_NO_FORWARD":
        raise V6AQualificationRunnerError("qualification plan status changed")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise V6AQualificationRunnerError("qualification call plan is missing")
    core = {key: value for key, value in plan.items() if key != "call_plan_sha256"}
    if canonical_sha256(core) != plan.get("call_plan_sha256"):
        raise V6AQualificationRunnerError("qualification call-plan hash changed")
    if design.get("call_plan_sha256") != plan.get("call_plan_sha256"):
        raise V6AQualificationRunnerError("qualification design binding changed")
    if plan.get("expected_calls") != EXPECTED_CALLS or plan.get("composition_calls") != 0:
        raise V6AQualificationRunnerError("qualification scope or call count changed")
    if len(plan.get("prompts", [])) != EXPECTED_CALLS:
        raise V6AQualificationRunnerError("qualification prompt count changed")
    prompts = plan["prompts"]
    cells = plan.get("cells", [])
    expected_families = Counter(
        {"property_retrieval": 8 * 32, "codebook_lookup": 8 * 16}
    )
    if Counter(prompt.get("family") for prompt in prompts) != expected_families:
        raise V6AQualificationRunnerError("qualification prompt families changed")
    if Counter(cell.get("family") for cell in cells) != expected_families:
        raise V6AQualificationRunnerError("qualification cell families changed")
    if any(item.get("family") == "composition" for item in [*prompts, *cells]):
        raise V6AQualificationRunnerError("qualification frozen plan contains composition")
    _validate_topology_shapes(prompts)
    if manifest.get("design_file_sha256") != file_sha256(DESIGN):
        raise V6AQualificationRunnerError("qualification design file changed")
    if manifest.get("dependency_lock_file_sha256") != file_sha256(DEPENDENCY_LOCK):
        raise V6AQualificationRunnerError("qualification dependency lock changed")
    if manifest.get("tokenization_receipt_file_sha256") != file_sha256(TOKENIZATION_RECEIPT):
        raise V6AQualificationRunnerError("qualification token receipt changed")
    receipt_core = {key: value for key, value in receipt.items() if key != "canonical_sha256"}
    if canonical_sha256(receipt_core) != receipt.get("canonical_sha256"):
        raise V6AQualificationRunnerError("qualification token receipt canonical hash changed")
    analyzer_path = Path(dependency["implementation_files"]["analyzer"]["path"])
    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise V6AQualificationRunnerError("frozen analyzer is not the preregistered default")
    current_dependency = dependency_lock(analyzer_path)
    if current_dependency != dependency:
        raise V6AQualificationRunnerError("qualification dependency or implementation drifted")
    fixture, fixture_manifest = load_and_rebuild_fixture()
    tokenizer = _load_tokenizer(Path(dependency["model"]["snapshot_path"]))
    rebuilt_plan, rebuilt_receipt = build_plan(tokenizer, fixture, dependency)
    if rebuilt_plan != plan:
        raise V6AQualificationRunnerError("qualification call plan does not replay exactly")
    if rebuilt_receipt != receipt:
        raise V6AQualificationRunnerError("qualification token receipt does not replay exactly")
    if design != _design_payload(plan, fixture_manifest):
        raise V6AQualificationRunnerError("qualification design does not replay exactly")
    if manifest != _plan_manifest_payload(plan):
        raise V6AQualificationRunnerError("qualification plan manifest does not replay exactly")
    return plan, design, dependency


def _load_model(snapshot: Path, plan: Mapping[str, Any]) -> Any:
    try:
        import torch
        import transformers.modeling_utils as modeling_utils
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise V6AQualificationRunnerError("torch and transformers are required") from error
    if plan["model"].get("mps_allocator_warmup_policy") != MPS_ALLOCATOR_WARMUP_POLICY:
        raise V6AQualificationRunnerError("V2 MPS allocator-warmup policy changed")
    original_warmup = modeling_utils.caching_allocator_warmup
    bypass_calls = 0

    def skip_all_mps_warmup(_model: Any, expanded_device_map: Mapping[str, Any], _quantizer: Any) -> None:
        nonlocal bypass_calls
        device_types = {torch.device(device).type for device in expanded_device_map.values()}
        if device_types != {DEVICE}:
            raise V6AQualificationRunnerError(
                "V2 allocator-warmup bypass encountered a non-MPS device map"
            )
        bypass_calls += 1

    modeling_utils.caching_allocator_warmup = skip_all_mps_warmup
    try:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=MODEL_LOADING_DEVICE_MAP,
            attn_implementation=ATTENTION_IMPLEMENTATION,
            low_cpu_mem_usage=True,
        ).eval()
    finally:
        modeling_utils.caching_allocator_warmup = original_warmup
    if modeling_utils.caching_allocator_warmup is not original_warmup or bypass_calls != 1:
        raise V6AQualificationRunnerError("V2 allocator-warmup bypass did not execute exactly once")
    model._v6a_mps_allocator_warmup_bypass_calls = bypass_calls
    config = model.config
    if (
        getattr(config, "hidden_size", None) != plan["model"]["hidden_size"]
        or getattr(config, "num_hidden_layers", None) != plan["model"]["num_hidden_layers"]
        or getattr(config, "vocab_size", None) != plan["model"]["vocab_size"]
    ):
        raise V6AQualificationRunnerError("loaded model dimensions changed")
    if getattr(config, "_attn_implementation", None) != ATTENTION_IMPLEMENTATION:
        raise V6AQualificationRunnerError("loaded attention implementation changed")
    parameters = list(model.parameters())
    if not parameters or {parameter.device.type for parameter in parameters} != {DEVICE}:
        raise V6AQualificationRunnerError("loaded model parameters are not all on MPS")
    if {parameter.dtype for parameter in parameters if parameter.is_floating_point()} != {
        torch.bfloat16
    }:
        raise V6AQualificationRunnerError("loaded model parameters are not all bfloat16")
    buffers = list(model.buffers())
    if any(buffer.device.type != DEVICE for buffer in buffers):
        raise V6AQualificationRunnerError("loaded model buffers are not all on MPS")
    allowed_buffer_dtypes = {torch.bfloat16, torch.float32}
    if any(
        buffer.is_floating_point() and buffer.dtype not in allowed_buffer_dtypes
        for buffer in buffers
    ):
        raise V6AQualificationRunnerError("loaded model has an unexpected floating buffer dtype")
    return model


def full_vocab_diagnostics(row: np.ndarray, prompt: Mapping[str, Any]) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    expected_id = prompt["expected_token_id"]
    distractor_id = prompt["distractor_token_id"]
    expected_logit = float(value[expected_id])
    distractor_logit = float(value[distractor_id])
    maximum = float(value.max())
    maxima = [int(index) for index in np.flatnonzero(value == maximum)]
    peak = float(value.astype(np.float64).max())
    logsumexp = peak + math.log(float(np.exp(value.astype(np.float64) - peak).sum()))
    label_logsumexp = float(np.logaddexp(expected_logit, distractor_logit))
    if expected_logit > distractor_logit:
        predicted_answer = prompt["expected_answer"]
        predicted_id = expected_id
    elif distractor_logit > expected_logit:
        predicted_answer = prompt["distractor_answer"]
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
        "greedy_token_id": maxima[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maxima,
        "maximum_tie_count": len(maxima),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "full_vocab_logits_sha256": f32_sha256(value),
    }


class RawShardWriter:
    def __init__(self, specs: Sequence[Mapping[str, Any]]) -> None:
        self.specs = [dict(spec) for spec in specs]
        self.buffer: list[np.ndarray] = []
        self.written: list[dict[str, Any]] = []
        self.next_index = 0

    def append(self, row: np.ndarray) -> tuple[int, int]:
        shard_index = self.next_index
        row_in_shard = len(self.buffer)
        self.buffer.append(np.ascontiguousarray(row, dtype="<f4"))
        spec = self.specs[shard_index]
        if len(self.buffer) == spec["rows"]:
            self._flush()
        return shard_index, row_in_shard

    def _flush(self) -> None:
        spec = self.specs[self.next_index]
        matrix = np.ascontiguousarray(np.stack(self.buffer), dtype="<f4")
        if list(matrix.shape) != spec["shape"]:
            raise V6AQualificationRunnerError("raw-logit shard shape changed")
        path = Path(spec["path"])
        _atomic_save_npy(path, matrix)
        self.written.append(
            {
                **spec,
                "file_sha256": file_sha256(path),
                "logical_sha256": f32_sha256(matrix),
                "size_bytes": path.stat().st_size,
            }
        )
        self.buffer = []
        self.next_index += 1

    def finish(self) -> list[dict[str, Any]]:
        if self.buffer:
            self._flush()
        if self.next_index != len(self.specs):
            raise V6AQualificationRunnerError("raw-logit shards are incomplete")
        return self.written


def _record_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in records
    )


def execution_preflight(
    plan: Mapping[str, Any], dependency: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise V6AQualificationRunnerError("torch is required") from error
    if plan.get("logits_to_keep") != 1:
        raise V6AQualificationRunnerError("qualification must retain only final-site logits")
    weight_bytes = sum(
        int(asset["size_bytes"])
        for name, asset in dependency["model"]["assets"].items()
        if name.endswith(".safetensors")
    )
    if weight_bytes <= 0:
        raise V6AQualificationRunnerError("model weight-byte registry is empty")
    recommended_bytes = int(torch.mps.recommended_max_memory())
    allocated_before = int(torch.mps.current_allocated_memory())
    required_mps_headroom = math.ceil(weight_bytes * 1.10)
    if recommended_bytes - allocated_before < required_mps_headroom:
        raise V6AQualificationRunnerError("insufficient MPS headroom for locked bfloat16 model")
    tensor = torch.ones((2, 2), dtype=torch.bfloat16, device=DEVICE)
    kernel_result = (tensor @ tensor).float().cpu().tolist()
    del tensor
    torch.mps.synchronize()
    torch.mps.empty_cache()
    if kernel_result != [[2.0, 2.0], [2.0, 2.0]]:
        raise V6AQualificationRunnerError("MPS bfloat16 kernel preflight failed")
    raw_bytes = sum(
        math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
        for spec in plan["raw_logits_shards"]
    )
    disk_free_bytes = shutil.disk_usage(RESULT_ROOT).free
    required_disk_free_bytes = raw_bytes + 1024**3
    if disk_free_bytes < required_disk_free_bytes:
        raise V6AQualificationRunnerError("insufficient disk space for raw qualification logits")
    return {
        "mps_bfloat16_kernel_pass": True,
        "model_safetensor_bytes": weight_bytes,
        "mps_recommended_max_memory_bytes": recommended_bytes,
        "mps_allocated_before_bytes": allocated_before,
        "required_mps_headroom_bytes": required_mps_headroom,
        "raw_logits_expected_bytes": raw_bytes,
        "disk_free_bytes": disk_free_bytes,
        "required_disk_free_bytes": required_disk_free_bytes,
    }


def _attempt_payload(
    plan: Mapping[str, Any], dependency: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "QUALIFICATION_EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "phase": "qualification-baseline",
        "call_plan_sha256": plan["call_plan_sha256"],
        "expected_calls": EXPECTED_CALLS,
        "partial_resume_allowed": False,
        "composition_calls": 0,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
        "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
        "execution_revision": plan["execution_revision"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "preflight": dict(preflight),
        "locks": {
            "plan_manifest_file_sha256": file_sha256(PLAN_MANIFEST),
            "design_file_sha256": file_sha256(DESIGN),
            "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
            "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
            "fixture_file_sha256": file_sha256(FIXTURE),
            "runner_file_sha256": dependency["implementation_files"]["runner"]["sha256"],
            "analyzer_file_sha256": dependency["implementation_files"]["analyzer"]["sha256"],
            "v1_plan_manifest_file_sha256": file_sha256(V1_PLAN_MANIFEST),
            "v1_attempt_file_sha256": file_sha256(V1_ATTEMPT),
            "v1_analysis_file_sha256": file_sha256(V1_ANALYSIS),
        },
    }


def run_qualification() -> dict[str, Any]:
    plan, _, dependency = validate_frozen_plan()
    if any(path.exists() for path in [ATTEMPT, RECORDS, EXECUTION_MANIFEST, ANALYSIS]) or RAW_LOGIT_ROOT.exists():
        raise V6AQualificationRunnerError("qualification execution is not one-shot clean")
    preflight = execution_preflight(plan, dependency)
    _atomic_write(ATTEMPT, _artifact_bytes(_attempt_payload(plan, dependency, preflight)))
    model = _load_model(Path(dependency["model"]["snapshot_path"]), plan)
    try:
        import torch
    except ImportError as error:
        raise V6AQualificationRunnerError("torch is required") from error
    writer = RawShardWriter(plan["raw_logits_shards"])
    records: list[dict[str, Any]] = []
    for call_index, prompt in enumerate(plan["prompts"]):
        ids = torch.tensor([prompt["execution_input_ids"]], dtype=torch.long, device=DEVICE)
        mask = torch.tensor([prompt["execution_attention_mask"]], dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=plan["logits_to_keep"],
                return_dict=True,
            )
        logits = np.ascontiguousarray(
            output.logits[0, -1, :].detach().float().cpu().numpy(), dtype="<f4"
        )
        if logits.shape != (plan["model"]["vocab_size"],) or not np.isfinite(logits).all():
            raise V6AQualificationRunnerError("model returned invalid logits")
        shard_index, row_in_shard = writer.append(logits)
        diagnostics = full_vocab_diagnostics(logits, prompt)
        record_core = {
            "schema_version": RECORD_SCHEMA,
            "phase": "qualification-baseline",
            "execution_revision": plan["execution_revision"],
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
            "raw_logits_row_in_shard": row_in_shard,
            "raw_logits_row_sha256": f32_sha256(logits),
            "diagnostics": diagnostics,
            "teacher_forced_prompt_forward": True,
            "generation_used": False,
            "model_calls": 1,
            "composition_calls": 0,
            "biological_model_calls": 0,
        }
        records.append({**record_core, "record_id": canonical_sha256(record_core)})
        if (call_index + 1) % 32 == 0 or call_index + 1 == EXPECTED_CALLS:
            print(
                json.dumps(
                    {
                        "phase": "qualification-baseline",
                        "completed": call_index + 1,
                        "total": EXPECTED_CALLS,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    shards = writer.finish()
    payload = _record_bytes(records)
    _atomic_write(RECORDS, payload)
    manifest = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "QUALIFICATION_EXECUTION_COMPLETE_NOT_ANALYZED",
        "phase": "qualification-baseline",
        "execution_revision": plan["execution_revision"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
        "call_plan_sha256": plan["call_plan_sha256"],
        "phase_model_calls": EXPECTED_CALLS,
        "cumulative_model_calls": EXPECTED_CALLS,
        "composition_calls": 0,
        "generation_used": False,
        "partial_resume_allowed": False,
        "records": {
            "path": str(RECORDS),
            "count": len(records),
            "file_sha256": file_sha256(RECORDS),
            "canonical_sha256": canonical_sha256(records),
            "size_bytes": RECORDS.stat().st_size,
        },
        "raw_logits_shards": shards,
        "attempt": {"path": str(ATTEMPT), "file_sha256": file_sha256(ATTEMPT)},
        "plan_manifest_file_sha256": file_sha256(PLAN_MANIFEST),
        "design_file_sha256": file_sha256(DESIGN),
        "dependency_lock_file_sha256": file_sha256(DEPENDENCY_LOCK),
        "tokenization_receipt_file_sha256": file_sha256(TOKENIZATION_RECEIPT),
        "runner_file_sha256": file_sha256(Path(__file__)),
        "analyzer_file_sha256": dependency["implementation_files"]["analyzer"]["sha256"],
        "biological_model_calls": 0,
    }
    _atomic_write(EXECUTION_MANIFEST, _artifact_bytes(manifest))
    return manifest


def run_loader_smoke(analyzer_path: Path = DEFAULT_ANALYZER) -> dict[str, Any]:
    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise V6AQualificationRunnerError("only the preregistered default analyzer is allowed")
    planning_paths = [
        DESIGN,
        PLAN_MANIFEST,
        TOKENIZATION_RECEIPT,
        DEPENDENCY_LOCK,
        LOADER_SMOKE_RECEIPT,
    ]
    if any(path.exists() for path in [*planning_paths, *_downstream_paths()]):
        raise V6AQualificationRunnerError("V2 artifacts exist before load-only smoke")
    validate_v1_engineering_stop_hashes_only()
    dependency = dependency_lock(analyzer_path)
    config = dependency["model"]["config"]
    loader_plan = {
        "model": {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "vocab_size": config["vocab_size"],
            "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
        }
    }
    model = _load_model(Path(dependency["model"]["snapshot_path"]), loader_plan)
    bypass_calls = getattr(model, "_v6a_mps_allocator_warmup_bypass_calls", None)
    if bypass_calls != 1:
        raise V6AQualificationRunnerError("V2 loader smoke did not register one bypass")
    try:
        import torch
    except ImportError as error:
        raise V6AQualificationRunnerError("torch is required") from error
    del model
    torch.mps.synchronize()
    torch.mps.empty_cache()
    result = {
        "status": "V2_LOADER_SMOKE_PASS_ZERO_FORWARD",
        "execution_revision": "v2_mps_allocator_warmup_bypass",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
        "warmup_bypass_calls": bypass_calls,
        "model_calls": 0,
        "generation_used": False,
        "fixture_prompt_records_consumed": 0,
        "composition_calls": 0,
        "dependency_canonical_sha256": dependency["canonical_sha256"],
    }
    _atomic_write(LOADER_SMOKE_RECEIPT, _artifact_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("loader-smoke", "plan", "qualification-baseline"), required=True
    )
    args = parser.parse_args()
    if args.phase == "loader-smoke":
        result = run_loader_smoke()
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    elif args.phase == "plan":
        result = run_plan()
        print(
            json.dumps(
                {
                    "phase": "plan",
                    "status": result["status"],
                    "model_calls": 0,
                    "composition_calls": 0,
                    "planned_calls": EXPECTED_CALLS,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        result = run_qualification()
        print(
            json.dumps(
                {
                    "phase": result["phase"],
                    "status": result["status"],
                    "model_calls": result["phase_model_calls"],
                    "composition_calls": 0,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan and execute the frozen V6A-R2 natural-token topology study.

The exact 2,304-row design was frozen before any R2 model forward.  Production
planning is one-shot, accepts no overrides, and can target only the registered
result root.  Execution is split into a 640-call discovery-component stage and
an indivisible 1,664-call remainder authorized only by a replayed discovery
PASS.  Model loading delegates to the exact hash-bound sealed V2 MPS helper.
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
import unicodedata
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
    / "coherent_readout_v6a_r2_topology"
    / "qwen2.5-7b-instruct"
)

BUILDER = ROOT / "signal" / "syntax" / "build_coherent_readout_v6a_r2_topology_bank.py"
FIXTURE = BUILDER.with_name("coherent_readout_v6a_r2_topology_bank.json")
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
DESIGN_DOCUMENT = ROOT / "docs" / "COHERENT_READOUT_V6A_R2_NATURAL_TOKEN_TOPOLOGY_DESIGN.md"
PRE_EXECUTION_REPAIR1_DOCUMENT = (
    ROOT / "docs" / "COHERENT_READOUT_V6A_R2_PRE_EXECUTION_REPAIR1.md"
)
PRE_EXECUTION_REPAIR1_RECEIPT = (
    ROOT
    / "signal"
    / "syntax"
    / "coherent_readout_v6a_r2_pre_execution_repair1.json"
)
RETIRED_ZERO_FORWARD_RESULT_ROOT = RESULT_ROOT.with_name(
    "qwen2.5-7b-instruct.quarantine-zero-forward-replay-fail-20260803-3642c02d"
)
V6A_PREDECESSOR_DESIGN = ROOT / "docs" / "COHERENT_READOUT_V6A_TOPOLOGY_IDENTIFICATION_DESIGN.md"
V2_PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V6A_COMPONENT_QUALIFICATION_V2_PREREG.md"
DEFAULT_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v6a_r2_topology.py"
BUILDER_TEST = ROOT / "tests" / "test_build_coherent_readout_v6a_r2_topology_bank.py"
RUNNER_TEST = ROOT / "tests" / "test_run_coherent_readout_v6a_r2_topology.py"
ANALYZER_TEST = ROOT / "tests" / "test_analyze_coherent_readout_v6a_r2_topology.py"

SEALED_V2_RUNNER = ROOT / "eval" / "run_coherent_readout_v6a_component_qualification_v2.py"
SEALED_V2_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v6a_component_qualification_v2.py"
SEALED_V2_POSTHOC = (
    ROOT / "eval" / "analyze_coherent_readout_v6a_component_qualification_v2_posthoc.py"
)
SEALED_V2_POSTHOC_TEST = (
    ROOT / "tests" / "test_analyze_coherent_readout_v6a_component_qualification_v2_posthoc.py"
)
V2_RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_component_qualification_v2"
    / "qwen2.5-7b-instruct"
)
V2_DEPENDENCY_LOCK = V2_RESULT_ROOT / "dependency_lock.json"
V2_TERMINAL_ANALYSIS = V2_RESULT_ROOT / "qualification_analysis.json"
V2_POSTHOC_OUTPUT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v6a_component_qualification_v2_posthoc"
    / "qwen2.5-7b-instruct"
    / "contextual_token_posthoc_analysis.json"
)

HISTORICAL_FIXTURE_SOURCES = {
    "v2_bio_fixture": ROOT / "signal" / "single_cell" / "coherent_readout_v2_bio_fixture.json",
    "v2_syntax_bank": ROOT / "signal" / "syntax" / "coherent_readout_v2_syntax_bank.json",
    "v2_causal_binding_bank": (
        ROOT / "signal" / "syntax" / "coherent_readout_v2_causal_binding_bank.json"
    ),
    "v3_content_routing_bank": (
        ROOT / "signal" / "syntax" / "coherent_readout_v3_content_routing_bank.json"
    ),
    "v4_behavioral_deconfounding_bank": (
        ROOT
        / "signal"
        / "syntax"
        / "coherent_readout_v4_behavioral_deconfounding_bank.json"
    ),
    "v5_positional_activation_bank": (
        ROOT / "signal" / "syntax" / "coherent_readout_v5_positional_activation_bank.json"
    ),
    "v6a_component_qualification_bank": (
        ROOT
        / "signal"
        / "syntax"
        / "coherent_readout_v6a_component_qualification_bank.json"
    ),
}
HISTORICAL_ARTIFACT_SOURCES = {
    "v3_tokenization_receipt": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v3_cross_codebook"
        / "qwen2.5-1.5b-instruct"
        / "tokenization_receipt.json"
    ),
    "v4_tokenization_receipt": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v4_behavioral_deconfounding"
        / "qwen2.5-1.5b-instruct"
        / "tokenization_receipt.json"
    ),
    "v5_tokenization_receipt": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v5_positional_activation"
        / "qwen2.5-1.5b-instruct"
        / "tokenization_receipt.json"
    ),
    "v2_causal_binding_plan": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v2_causal_binding"
        / "qwen2.5-1.5b-instruct"
        / "plan_manifest.json"
    ),
    "v2_syntax_0_5b_plan": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v2_syntax"
        / "qwen2.5-0.5b-instruct"
        / "plan_manifest.json"
    ),
    "v2_syntax_1_5b_plan": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v2_syntax"
        / "qwen2.5-1.5b-instruct"
        / "plan_manifest.json"
    ),
    "v6a_v1_tokenization_receipt": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v6a_component_qualification"
        / "qwen2.5-7b-instruct"
        / "tokenization_receipt.json"
    ),
    "v6a_v2_tokenization_receipt": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v6a_component_qualification_v2"
        / "qwen2.5-7b-instruct"
        / "tokenization_receipt.json"
    ),
    "v6a_v1_plan": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v6a_component_qualification"
        / "qwen2.5-7b-instruct"
        / "plan_manifest.json"
    ),
    "v6a_v2_plan": (
        ROOT
        / "results"
        / "benchmark"
        / "single_cell"
        / "coherent_readout_v6a_component_qualification_v2"
        / "qwen2.5-7b-instruct"
        / "plan_manifest.json"
    ),
}
HISTORICAL_SOURCE_SHA256 = {
    "v2_bio_fixture": "d9c8256cc249f5f3b1b5ea07d99bdb80927b1c2b6b50bcb17540ae3ea0dd601a",
    "v2_syntax_bank": "d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8",
    "v2_causal_binding_bank": "2c40ba0c796202059056aec4535fd7656eab2b446d8895816bbae2034ebcbcdb",
    "v3_content_routing_bank": "a63dced290410ef6d463a0f2c04431dcea871ea564f2a6d0b2e0a05b4bb0d78f",
    "v4_behavioral_deconfounding_bank": "8d988b36d99677e798628b932fea86efd68317cd4a8653bbcd2f12a3294021c2",
    "v5_positional_activation_bank": "defa5ed2c0ab1f0f6c7ac7cf5eaa4abe453daac682676b9d783770ffae6da903",
    "v6a_component_qualification_bank": "084ba96dc785f26ca99387ea758bb542ce974e8bd86fa97ff4b6abc6b7a13cd6",
    "v3_tokenization_receipt": "089d320370cd1f1e933ce1d2aacd232f5b1eb02a4d012cd1c7b56febc9d9e554",
    "v4_tokenization_receipt": "a7c600a7af28366f2dc369b60adcf2277b86bf0121e862306bfb527af8f84543",
    "v5_tokenization_receipt": "1e9401e92a5969d4f735b1094e0dda77167184a1a970ee01aa74752b4cb42c19",
    "v2_causal_binding_plan": "f22b5659ca527e39b451157bf3f9ce7994d9a0c29e8401cd76c9a47fec5b282a",
    "v2_syntax_0_5b_plan": "d787e55469021f22ea7fc141845b9844b6e1f5708cbfcf2f1f4d24a07b711cff",
    "v2_syntax_1_5b_plan": "94608ea69b98f9bffab9c7ac5b0591017d237cdcddb7cbabd74f21af7f5738e6",
    "v6a_v1_tokenization_receipt": "d3e95323377e67ddcfab9ffa713c78ba4e8db28de2c2c16c909ab35df0fc3162",
    "v6a_v2_tokenization_receipt": "b8eda62d4bca9bc7cdebda721f617a6f3f103d663333ec83ec8dbf4a22eb48e5",
    "v6a_v1_plan": "0be35c0d6635924658985b5ea47b6f02b09c4245e10ba00bd28735e00eedaaf9",
    "v6a_v2_plan": "27b33868fb6269e2dd73ea50c8f41c21d6ab9fbb67a82c795b476993fa185758",
}

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEVICE = "mps"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MODEL_LOADING_DEVICE_MAP = DEVICE
MPS_ALLOCATOR_WARMUP_POLICY = "skip_transformers_warmup_only_for_all_mps_map_v2"
SEALED_V2_RUNNER_SHA256 = "cbff3d552ca8517c00e2dfbe6be1c63bbb0b7278e493de9f18edfc14cd511867"
SEALED_V2_DEPENDENCY_LOCK_SHA256 = (
    "8b52bb32ce972a36d316ae5b6f63810956d0b9cba955f11e8fd2b15b01a8a73e"
)
V2_TERMINAL_ANALYSIS_SHA256 = (
    "595cff448a3f72011e119f37556c95e11ea0fe4c4daef7d79541f507f4987cb8"
)
V2_POSTHOC_OUTPUT_SHA256 = (
    "8b3ed9a43241286fb72e10edea0d27f2a2ead113c482422d5340ff51f6438ed8"
)

REGISTRATION_STATE = "FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
EXECUTION_ENABLED_REGISTRATION_STATE = "FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
EXECUTION_REVISION = "coherent-readout-v6a-r2-natural-token-topology-exec-v1"
EXPECTED_CALLS = 2_304
DISCOVERY_COMPONENT_CALLS = 640
REMAINING_MAIN_CALLS = 1_664
RAW_SHARD_ROWS = 64
RESPONSE_SITE_TOKEN_ID = 25
NATURAL_SURFACE_PREFIX = " "
CHAT_FLAGS = {
    "add_generation_prompt": False,
    "continue_final_message": True,
    "enable_thinking": False,
}

DISCOVERY_COMPONENTS_STAGE = "discovery-components"
REMAINING_MAIN_STAGE = "remaining-main"
EXECUTION_STAGES = (DISCOVERY_COMPONENTS_STAGE, REMAINING_MAIN_STAGE)
BLOCK_ORDER = (
    "discovery-components",
    "discovery-topology",
    "confirmation-components",
    "confirmation-topology",
)
BLOCK_COUNTS = {
    "discovery-components": 640,
    "discovery-topology": 512,
    "confirmation-components": 640,
    "confirmation-topology": 512,
}
REMAINING_BLOCK_LAYOUT = {
    "discovery-topology": {
        "record_artifact_key": "remaining_discovery_topology_records",
        "call_count": 512,
        "stage_start_row": 0,
        "stage_stop_row": 512,
        "global_start_row": 640,
        "global_stop_row": 1_152,
        "raw_shard_indices": list(range(0, 8)),
    },
    "confirmation-components": {
        "record_artifact_key": "remaining_confirmation_components_records",
        "call_count": 640,
        "stage_start_row": 512,
        "stage_stop_row": 1_152,
        "global_start_row": 1_152,
        "global_stop_row": 1_792,
        "raw_shard_indices": list(range(8, 18)),
    },
    "confirmation-topology": {
        "record_artifact_key": "remaining_confirmation_topology_records",
        "call_count": 512,
        "stage_start_row": 1_152,
        "stage_stop_row": 1_664,
        "global_start_row": 1_792,
        "global_stop_row": 2_304,
        "raw_shard_indices": list(range(18, 26)),
    },
}
SPLIT_FAMILY_COUNTS = {
    "discovery": {
        "property_retrieval": 256,
        "codebook_lookup": 128,
        "single_target_composition": 256,
        "two_fact_composition": 512,
    },
    "confirmation": {
        "property_retrieval": 256,
        "codebook_lookup": 128,
        "single_target_composition": 256,
        "two_fact_composition": 512,
    },
}
FAMILY_COUNTS = {
    "property_retrieval": 512,
    "codebook_lookup": 256,
    "single_target_composition": 512,
    "two_fact_composition": 1_024,
}

COMPONENT_GATES = {
    "applied_separately_to_splits": ["discovery", "confirmation"],
    "overall": {
        "pairwise_natural_token": {
            "property_retrieval": "at_least_251_of_256",
            "single_target_composition": "at_least_251_of_256",
            "codebook_lookup": "at_least_126_of_128",
        },
        "strict_unique_global_correct": {
            "property_retrieval": "at_least_251_of_256",
            "single_target_composition": "at_least_251_of_256",
            "codebook_lookup": "at_least_126_of_128",
        },
    },
    "per_world": {
        "pairwise_natural_token": {
            "property_retrieval": "at_least_29_of_32",
            "single_target_composition": "at_least_29_of_32",
            "codebook_lookup": "at_least_15_of_16",
        },
        "strict_unique_global_correct": {
            "property_retrieval": "at_least_29_of_32",
            "single_target_composition": "at_least_29_of_32",
            "codebook_lookup": "at_least_15_of_16",
        },
    },
    "pairwise_per_answer_label": {
        "property_retrieval": "at_least_15_of_16",
        "single_target_composition": "at_least_15_of_16",
        "codebook_lookup": "at_least_7_of_8",
    },
    "pairwise_per_factor_level": {
        "n_128": "at_least_116_of_128",
        "n_64": "at_least_58_of_64",
    },
    "pairwise_joint_scaffolds": {
        "property_retrieval_q_by_a": "at_least_58_of_64",
        "property_retrieval_o_by_q_by_a": "at_least_29_of_32",
        "single_target_composition_q_by_a": "at_least_58_of_64",
    },
    "tie_rule": "ordinary_incorrect_row_no_universal_veto",
    "global_outside_pair_rule": "strict_incorrect_row_no_universal_veto",
    "inference_scope": "registered_fixed_panel_error_budgets_not_iid_intervals",
}

PLAN_SCHEMA = "coherent-readout-v6a-r2-topology-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v6a-r2-topology-design-v1"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v6a-r2-topology-plan-manifest-v1"
TOKENIZATION_SCHEMA = "coherent-readout-v6a-r2-topology-tokenization-v1"
DEPENDENCY_SCHEMA = "coherent-readout-v6a-r2-topology-dependencies-v1"
HISTORICAL_FIREWALL_SCHEMA = "coherent-readout-v6a-r2-historical-firewall-v1"
CANDIDATE_UNIVERSE_SCHEMA = "coherent-readout-v6a-r2-candidate-universe-v1"
PROMPT_SCHEMA = "coherent-readout-v6a-r2-topology-prompt-v1"
ATTEMPT_SCHEMA = "coherent-readout-v6a-r2-topology-attempt-v1"
RECORD_SCHEMA = "coherent-readout-v6a-r2-topology-record-v1"
EXECUTION_SCHEMA = "coherent-readout-v6a-r2-topology-execution-v1"

DISCOVERY_ATTEMPT_STATUS = (
    "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_ATTEMPT_STARTED_IMMUTABLE"
)
REMAINING_ATTEMPT_STATUS = "V6A_R2_REMAINING_MAIN_EXECUTION_ATTEMPT_STARTED_IMMUTABLE"
DISCOVERY_COMPLETE_STATUS = (
    "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_COMPLETE_NOT_ANALYZED"
)
REMAINING_COMPLETE_STATUS = "V6A_R2_REMAINING_MAIN_EXECUTION_COMPLETE_NOT_ANALYZED"
DISCOVERY_AUTHORIZED_STATUS = "V6A_R2_DISCOVERY_COMPONENT_PASS_REMAINING_AUTHORIZED"

RECORD_IDENTITY_KEYS = frozenset(
    {
        "block_call_index",
        "cell_id",
        "execution_block",
        "execution_stage",
        "global_call_index",
        "prompt_id",
        "stage_call_index",
    }
)
CELL_SHARED_IDENTITY_FIELDS = (
    "block_call_index",
    "cell_id",
    "execution_block",
    "execution_stage",
    "global_call_index",
    "stage_call_index",
)
DISCOVERY_AUTHORIZATION_REPLAY_KEYS = frozenset(
    {
        "authorization",
        "authorization_file_sha256",
        "authorization_canonical_sha256",
        "discovery_execution_manifest_file_sha256",
        "discovery_execution_manifest_canonical_sha256",
        "authorization_status",
        "call_plan_sha256",
        "stage_plan_sha256",
        "replay_equal",
        "model_calls_issued_by_validator",
        "generation_calls_issued_by_validator",
    }
)

PACKAGE_NAMES = (
    "accelerate",
    "huggingface-hub",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)

HISTORICAL_SCALAR_TOKEN_ID_KEYS = frozenset(
    {
        "expected_token_id",
        "distractor_token_id",
        "x_next_token_id",
        "y_next_token_id",
        "x_token_id",
        "y_token_id",
        "continuation_token_id",
        "prompt_token_id",
    }
)
HISTORICAL_LIST_TOKEN_ID_KEYS = frozenset(
    {"answer_token_ids", "expected_label_token_ids", "label_token_ids"}
)
HISTORICAL_EXPECTED_POOL_COUNTS = {
    "single_codepoint_scalars": 65,
    "answer_token_ids": 91,
    "entity_strings": 2_010,
    "instance_keys": 40,
    "user_prompt_hashes": 4_752,
    "rendered_text_hashes": 4_304,
}

SELECTION_UNICODE_VERSION = "15.1.0"
SELECTION_CODEPOINT_RANGES = ((0x00C0, 0x02AF), (0x0370, 0x052F))
SELECTION_SALT = "V6A-R2-main-replacement-v1|"
ORIGINAL_V6A_SYMBOLS = tuple(
    "àáâäåæçèéêíîóôöøúüýþăąćčđęıłńőœśşšżžơưǎǐǒǔǝǥǧǫǯǵǹǻǽ"
    "туфхцчшэяµÀÁÂ"
)
NATURAL_TOKEN_INVALID_SYMBOLS = tuple("ăąćęıńőơưǎǐǒǔǝǥǧǫǯǵǹǻǽ")
EXPECTED_REPLACEMENT_SYMBOLS = tuple("ŻЧАŚЯКŁŞÜУОÅНЕШЮіЦÎÈĐГ")
EXPECTED_ELIGIBLE_CANDIDATES = tuple(
    "ŻЧАŚЯКŁŞÜУОÅНЕШЮіЦÎÈĐГПÃÄÖØИЗЛÇДФÔÉСЖБХЭМТРİВ"
)


class V6AR2RunnerError(RuntimeError):
    """Raised when an R2 planning, hash, tokenizer, or stage lock fails."""


class V6AR2BehavioralExecutionDisabled(V6AR2RunnerError):
    """Raised when exact frozen behavioral-execution conditions are absent."""


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


def record_identity_core(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact seven-field, pre-forward record identity object."""

    core = {
        "block_call_index": value.get("block_call_index"),
        "cell_id": value.get("cell_id"),
        "execution_block": value.get("execution_block"),
        "execution_stage": value.get("execution_stage"),
        "global_call_index": value.get("global_call_index"),
        "prompt_id": value.get("prompt_id"),
        "stage_call_index": value.get("stage_call_index"),
    }
    if set(core) != RECORD_IDENTITY_KEYS:
        raise V6AR2RunnerError("R2 record-identity key registry changed")
    if (
        any(
            isinstance(core[key], bool) or not isinstance(core[key], int)
            for key in ("block_call_index", "global_call_index", "stage_call_index")
        )
        or any(
            not isinstance(core[key], str) or not core[key]
            for key in (
                "cell_id",
                "execution_block",
                "execution_stage",
                "prompt_id",
            )
        )
    ):
        raise V6AR2RunnerError("R2 record identity is malformed")
    return core


def record_identity_id(value: Mapping[str, Any]) -> str:
    return canonical_sha256(record_identity_core(value))


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
        raise V6AR2RunnerError(f"refusing to overwrite differing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AR2RunnerError(f"stale temporary artifact exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    """Create one immutable NumPy artifact without an overwrite path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V6AR2RunnerError(f"refusing to overwrite raw-logit shard: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AR2RunnerError(f"stale temporary raw-logit shard exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise V6AR2RunnerError(f"required artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V6AR2RunnerError(f"cannot load JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise V6AR2RunnerError(f"JSON artifact is not an object: {path}")
    return value


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _historical_firewall_registry() -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Rebuild the exact V2--V6A historical identity firewall from sealed JSON."""

    source_groups = (
        ("fixture", HISTORICAL_FIXTURE_SOURCES),
        ("artifact", HISTORICAL_ARTIFACT_SOURCES),
    )
    pools: dict[str, set[Any]] = {
        name: set() for name in HISTORICAL_EXPECTED_POOL_COUNTS
    }
    source_receipts: list[dict[str, Any]] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list):
            if key in HISTORICAL_LIST_TOKEN_ID_KEYS:
                pools["answer_token_ids"].update(
                    item
                    for item in value
                    if isinstance(item, int) and not isinstance(item, bool)
                )
            for child in value:
                visit(child, key)
            return
        if isinstance(value, str):
            if len(value) == 1 and key != "role":
                pools["single_codepoint_scalars"].add(value)
            if key is not None and "entity" in key:
                pools["entity_strings"].add(value)
            if key == "instance_key":
                pools["instance_keys"].add(value)
            if key == "prompt_sha256":
                pools["user_prompt_hashes"].add(value)
            if key == "rendered_text_sha256":
                pools["rendered_text_hashes"].add(value)
        if (
            key in HISTORICAL_SCALAR_TOKEN_ID_KEYS
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            pools["answer_token_ids"].add(value)

    for source_kind, sources in source_groups:
        for name, path in sources.items():
            expected_sha256 = HISTORICAL_SOURCE_SHA256.get(name)
            if not path.is_file() or expected_sha256 is None:
                raise V6AR2RunnerError(f"historical firewall source is missing: {name}")
            observed_sha256 = file_sha256(path)
            if observed_sha256 != expected_sha256:
                raise V6AR2RunnerError(f"historical firewall source hash changed: {name}")
            payload = _load_json(path)
            visit(payload)
            source_receipts.append(
                {
                    "name": name,
                    "kind": source_kind,
                    "path": _relative_path(path),
                    "sha256": observed_sha256,
                    "size_bytes": path.stat().st_size,
                    "schema_version": payload.get("schema_version"),
                }
            )

    sorted_pools = {name: sorted(values) for name, values in pools.items()}
    observed_counts = {name: len(values) for name, values in sorted_pools.items()}
    if observed_counts != HISTORICAL_EXPECTED_POOL_COUNTS:
        raise V6AR2RunnerError(
            "historical firewall pool counts changed: "
            f"observed={observed_counts}, expected={HISTORICAL_EXPECTED_POOL_COUNTS}"
        )
    if len(source_receipts) != 17:
        raise V6AR2RunnerError("historical firewall source count changed")
    receipt = {
        "schema_version": HISTORICAL_FIREWALL_SCHEMA,
        "fixture_source_count": len(HISTORICAL_FIXTURE_SOURCES),
        "artifact_source_count": len(HISTORICAL_ARTIFACT_SOURCES),
        "source_count": len(source_receipts),
        "sources": source_receipts,
        "source_registry_sha256": canonical_sha256(source_receipts),
        "extraction_contract": {
            "recursive_dict_and_list_walk": True,
            "single_codepoint_scalar_rule": "len(value)==1 and key!='role'",
            "scalar_token_id_keys": sorted(HISTORICAL_SCALAR_TOKEN_ID_KEYS),
            "list_token_id_keys": sorted(HISTORICAL_LIST_TOKEN_ID_KEYS),
            "entity_rule": "string value whose key contains 'entity'",
            "instance_key_rule": "string value at key 'instance_key'",
            "user_prompt_hash_rule": "string value at key 'prompt_sha256'",
            "rendered_hash_rule": "string value at key 'rendered_text_sha256'",
        },
        "pool_counts": observed_counts,
        "pool_sha256": {
            name: canonical_sha256(values) for name, values in sorted_pools.items()
        },
    }
    return sorted_pools, receipt


def historical_firewall_dependency_receipt() -> dict[str, Any]:
    """Return the source-bound public receipt used in dependency locks and tests."""

    _, receipt = _historical_firewall_registry()
    return receipt


def _tokenizer_special_id_receipt(tokenizer: Any) -> tuple[set[int], dict[str, Any]]:
    raw_special_ids = getattr(tokenizer, "all_special_ids", None)
    if not isinstance(raw_special_ids, (list, tuple, set)):
        raise V6AR2RunnerError("tokenizer does not expose all_special_ids")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_special_ids):
        raise V6AR2RunnerError("tokenizer all_special_ids registry is malformed")
    special_ids = {int(value) for value in raw_special_ids}

    added_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added_decoder, Mapping):
        added_ids = {
            int(value)
            for value in added_decoder
            if isinstance(value, int) and not isinstance(value, bool)
        }
        mechanism = "added_tokens_decoder"
    else:
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        if not callable(get_added_vocab):
            raise V6AR2RunnerError(
                "tokenizer does not expose added_tokens_decoder or get_added_vocab"
            )
        added_vocab = get_added_vocab()
        if not isinstance(added_vocab, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in added_vocab.values()
        ):
            raise V6AR2RunnerError("tokenizer added-token registry is malformed")
        added_ids = {int(value) for value in added_vocab.values()}
        mechanism = "get_added_vocab"
    prohibited_ids = special_ids | added_ids
    receipt = {
        "all_special_ids": sorted(special_ids),
        "all_special_ids_sha256": canonical_sha256(sorted(special_ids)),
        "added_token_id_mechanism": mechanism,
        "added_token_ids": sorted(added_ids),
        "added_token_ids_sha256": canonical_sha256(sorted(added_ids)),
        "prohibited_token_ids_sha256": canonical_sha256(sorted(prohibited_ids)),
    }
    return prohibited_ids, receipt


def _reproduce_candidate_universe(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    historical_pools: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Independently enumerate and tokenize the exact Unicode 15.1 candidates."""

    if unicodedata.unidata_version != SELECTION_UNICODE_VERSION:
        raise V6AR2RunnerError("Unicode database version is not the registered 15.1.0")
    provenance = fixture.get("symbol_replacement_provenance")
    if not isinstance(provenance, Mapping):
        raise V6AR2RunnerError("R2 replacement provenance is missing")
    expected_provenance = {
        "original_v6a_symbols": list(ORIGINAL_V6A_SYMBOLS),
        "natural_token_invalid_symbols_in_slot_order": list(
            NATURAL_TOKEN_INVALID_SYMBOLS
        ),
        "replacement_symbols_in_slot_order": list(EXPECTED_REPLACEMENT_SYMBOLS),
        "selection_salt": SELECTION_SALT,
        "unicode_database_version": SELECTION_UNICODE_VERSION,
        "candidate_codepoint_ranges_inclusive": [
            [lower, upper] for lower, upper in SELECTION_CODEPOINT_RANGES
        ],
        "eligible_candidates_in_hash_order": list(EXPECTED_ELIGIBLE_CANDIDATES),
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise V6AR2RunnerError(f"R2 replacement provenance changed: {key}")
    if EXPECTED_ELIGIBLE_CANDIDATES[:22] != EXPECTED_REPLACEMENT_SYMBOLS:
        raise V6AR2RunnerError("registered first-22 replacement list is internally inconsistent")

    prohibited_token_ids, special_receipt = _tokenizer_special_id_receipt(tokenizer)
    historical_scalars = set(historical_pools["single_codepoint_scalars"])
    excluded_scalars = historical_scalars | set(ORIGINAL_V6A_SYMBOLS) | set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    eligible: list[tuple[bytes, int, str, int]] = []
    enumerated_count = 0
    nfc_letter_count = 0
    for lower, upper in SELECTION_CODEPOINT_RANGES:
        for codepoint in range(lower, upper + 1):
            enumerated_count += 1
            glyph = chr(codepoint)
            if (
                unicodedata.normalize("NFC", glyph) != glyph
                or not unicodedata.category(glyph).startswith("L")
            ):
                continue
            nfc_letter_count += 1
            if glyph in excluded_scalars:
                continue
            surface = NATURAL_SURFACE_PREFIX + glyph
            token_ids = _as_int_vector(
                tokenizer.encode(surface, add_special_tokens=False),
                f"candidate U+{codepoint:04X}",
            )
            if len(token_ids) != 1:
                continue
            token_id = token_ids[0]
            if token_id in prohibited_token_ids or _decode_one(tokenizer, token_id) != surface:
                continue
            digest = hashlib.sha256((SELECTION_SALT + glyph).encode("utf-8")).digest()
            eligible.append((digest, codepoint, glyph, token_id))
    eligible.sort(key=lambda item: (item[0], item[1]))
    observed_glyphs = tuple(item[2] for item in eligible)
    if observed_glyphs != EXPECTED_ELIGIBLE_CANDIDATES or len(observed_glyphs) != 45:
        raise V6AR2RunnerError("tokenizer-reproduced R2 eligible candidate universe changed")
    if observed_glyphs[:22] != EXPECTED_REPLACEMENT_SYMBOLS:
        raise V6AR2RunnerError("tokenizer-reproduced first-22 replacements changed")
    token_ids = [item[3] for item in eligible]
    if len(set(token_ids)) != 45:
        raise V6AR2RunnerError("eligible candidate token IDs are not globally unique")

    return {
        "schema_version": CANDIDATE_UNIVERSE_SCHEMA,
        "unicode_database_version": unicodedata.unidata_version,
        "candidate_codepoint_ranges_inclusive": [
            [lower, upper] for lower, upper in SELECTION_CODEPOINT_RANGES
        ],
        "enumerated_codepoint_count": enumerated_count,
        "nfc_letter_scalar_count_before_exclusions": nfc_letter_count,
        "exclusion_pool_counts": {
            "historical_single_codepoint_scalars": len(historical_scalars),
            "original_v6a_symbols": len(ORIGINAL_V6A_SYMBOLS),
            "ascii_uppercase": 26,
            "union": len(excluded_scalars),
        },
        "exclusion_pool_sha256": canonical_sha256(sorted(excluded_scalars)),
        "selection_salt": SELECTION_SALT,
        "selection_order": "sha256(salt+utf8_glyph)_bytes_then_codepoint",
        "eligible_candidate_count": len(eligible),
        "eligible_candidates_in_hash_order": [
            {
                "glyph": glyph,
                "codepoint": f"U+{codepoint:04X}",
                "selection_sha256": digest.hex(),
                "natural_surface": NATURAL_SURFACE_PREFIX + glyph,
                "natural_token_id": token_id,
            }
            for digest, codepoint, glyph, token_id in eligible
        ],
        "eligible_glyphs_sha256": canonical_sha256(list(observed_glyphs)),
        "first_22_replacements": list(observed_glyphs[:22]),
        "first_22_replacements_sha256": canonical_sha256(
            list(observed_glyphs[:22])
        ),
        "special_and_added_token_proof": special_receipt,
        "model_calls": 0,
    }


def _validate_candidate_receipt_static(receipt: Mapping[str, Any]) -> None:
    rows = receipt.get("eligible_candidates_in_hash_order")
    if (
        receipt.get("schema_version") != CANDIDATE_UNIVERSE_SCHEMA
        or receipt.get("unicode_database_version") != SELECTION_UNICODE_VERSION
        or receipt.get("candidate_codepoint_ranges_inclusive")
        != [[lower, upper] for lower, upper in SELECTION_CODEPOINT_RANGES]
        or receipt.get("selection_salt") != SELECTION_SALT
        or not isinstance(rows, list)
        or len(rows) != 45
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise V6AR2RunnerError("R2 candidate-universe receipt structure changed")
    glyphs = [str(row["glyph"]) for row in rows]
    if glyphs != list(EXPECTED_ELIGIBLE_CANDIDATES):
        raise V6AR2RunnerError("R2 candidate-universe glyph order changed")
    for row, glyph in zip(rows, EXPECTED_ELIGIBLE_CANDIDATES, strict=True):
        codepoint = ord(glyph)
        expected_digest = hashlib.sha256(
            (SELECTION_SALT + glyph).encode("utf-8")
        ).hexdigest()
        if (
            row.get("codepoint") != f"U+{codepoint:04X}"
            or row.get("selection_sha256") != expected_digest
            or row.get("natural_surface") != NATURAL_SURFACE_PREFIX + glyph
            or isinstance(row.get("natural_token_id"), bool)
            or not isinstance(row.get("natural_token_id"), int)
        ):
            raise V6AR2RunnerError("R2 candidate-universe row changed")
    token_ids = [int(row["natural_token_id"]) for row in rows]
    if len(set(token_ids)) != 45:
        raise V6AR2RunnerError("R2 candidate-universe token IDs changed")
    special = receipt.get("special_and_added_token_proof")
    if not isinstance(special, Mapping):
        raise V6AR2RunnerError("R2 special-token proof changed")
    all_special = special.get("all_special_ids")
    added = special.get("added_token_ids")
    if (
        not isinstance(all_special, list)
        or not isinstance(added, list)
        or special.get("all_special_ids_sha256") != canonical_sha256(all_special)
        or special.get("added_token_ids_sha256") != canonical_sha256(added)
        or special.get("prohibited_token_ids_sha256")
        != canonical_sha256(sorted(set(all_special) | set(added)))
        or set(token_ids) & (set(all_special) | set(added))
    ):
        raise V6AR2RunnerError("R2 candidate special/added-token proof changed")
    if (
        receipt.get("eligible_candidate_count") != 45
        or receipt.get("eligible_glyphs_sha256") != canonical_sha256(glyphs)
        or receipt.get("first_22_replacements")
        != list(EXPECTED_REPLACEMENT_SYMBOLS)
        or receipt.get("first_22_replacements_sha256")
        != canonical_sha256(list(EXPECTED_REPLACEMENT_SYMBOLS))
        or receipt.get("model_calls") != 0
    ):
        raise V6AR2RunnerError("R2 candidate-universe summary changed")


def _import_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise V6AR2RunnerError(f"required Python source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V6AR2RunnerError(f"cannot import Python source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder_module() -> Any:
    return _import_module(BUILDER, "v6a_r2_topology_builder")


def _sealed_v2_runner_module() -> Any:
    if file_sha256(SEALED_V2_RUNNER) != SEALED_V2_RUNNER_SHA256:
        raise V6AR2RunnerError("sealed V2 MPS loader source hash changed")
    return _import_module(SEALED_V2_RUNNER, "sealed_v6a_v2_runner")


def _sealed_v2_posthoc_module() -> Any:
    return _import_module(SEALED_V2_POSTHOC, "sealed_v6a_v2_contextual_posthoc")


def _analyzer_module() -> Any:
    return _import_module(DEFAULT_ANALYZER, "v6a_r2_topology_analyzer_for_runner")


def artifact_paths(result_root: Path = RESULT_ROOT) -> dict[str, Path]:
    root = result_root.resolve()
    return {
        "result_root": root,
        "design": root / "design.json",
        "plan_manifest": root / "plan_manifest.json",
        "tokenization_receipt": root / "tokenization_receipt.json",
        "dependency_lock": root / "dependency_lock.json",
        "discovery_attempt": root / "discovery_components_attempt.json",
        "discovery_records": root / "discovery_components_records.jsonl",
        "discovery_execution_manifest": root / "discovery_components_execution_manifest.json",
        "discovery_analysis": root / "discovery_components_analysis.json",
        "discovery_raw_root": root / "raw_logits" / DISCOVERY_COMPONENTS_STAGE,
        "remaining_attempt": root / "remaining_main_attempt.json",
        "remaining_discovery_topology_records": (
            root / "remaining_main_discovery_topology_records.jsonl"
        ),
        "remaining_confirmation_components_records": (
            root / "remaining_main_confirmation_components_records.jsonl"
        ),
        "remaining_confirmation_topology_records": (
            root / "remaining_main_confirmation_topology_records.jsonl"
        ),
        "remaining_execution_manifest": root / "remaining_main_execution_manifest.json",
        "remaining_raw_root": root / "raw_logits" / REMAINING_MAIN_STAGE,
        "final_analysis": root / "topology_analysis.json",
    }


def _planning_paths(paths: Mapping[str, Path]) -> list[Path]:
    return [
        paths["design"],
        paths["plan_manifest"],
        paths["tokenization_receipt"],
        paths["dependency_lock"],
    ]


def _discovery_paths(paths: Mapping[str, Path]) -> list[Path]:
    return [
        paths["discovery_attempt"],
        paths["discovery_records"],
        paths["discovery_execution_manifest"],
        paths["discovery_analysis"],
        paths["discovery_raw_root"],
    ]


def _remaining_paths(paths: Mapping[str, Path]) -> list[Path]:
    return [
        paths["remaining_attempt"],
        paths["remaining_discovery_topology_records"],
        paths["remaining_confirmation_components_records"],
        paths["remaining_confirmation_topology_records"],
        paths["remaining_execution_manifest"],
        paths["remaining_raw_root"],
        paths["final_analysis"],
    ]


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise V6AR2RunnerError(f"{label} must contain exactly one row")
        raw = raw[0]
    if not raw or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise V6AR2RunnerError(f"{label} is not a nonempty integer vector")
    return [int(item) for item in raw]


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:
        raise V6AR2RunnerError("tokenizer cannot decode a registered answer token") from error
    if not isinstance(value, str):
        raise V6AR2RunnerError("tokenizer decode did not return text")
    return value


def _prefix_extension_token_id(
    tokenizer: Any,
    rendered: str,
    surface: str,
    *,
    label: str,
) -> int:
    prefix = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False),
        f"{label} prefix",
    )
    combined = _as_int_vector(
        tokenizer.encode(rendered + surface, add_special_tokens=False),
        f"{label} prefix plus surface",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise V6AR2RunnerError(f"{label} is not a prefix-preserving one-token continuation")
    return combined[-1]


def _registered_symbols(fixture: Mapping[str, Any]) -> list[str]:
    symbols = fixture.get("r2_symbols")
    if (
        not isinstance(symbols, list)
        or len(symbols) != 64
        or any(not isinstance(symbol, str) or len(symbol) != 1 for symbol in symbols)
        or len(set(symbols)) != 64
    ):
        raise V6AR2RunnerError("R2 fixture does not expose the exact 64-glyph registry")
    return list(symbols)


def _cell_split(cell: Mapping[str, Any]) -> str:
    value = cell.get("role")
    if value not in ("discovery", "confirmation"):
        raise V6AR2RunnerError("cell role is not discovery or confirmation")
    return str(value)


def _execution_block(cell: Mapping[str, Any]) -> str:
    role = _cell_split(cell)
    stage = cell.get("stage")
    family = cell.get("family")
    if stage == "components" and family != "two_fact_composition":
        expected = f"{role}-components"
    elif stage == "topology" and family == "two_fact_composition":
        expected = f"{role}-topology"
    else:
        raise V6AR2RunnerError("cell stage/family firewall changed")
    if cell.get("execution_block") != expected:
        raise V6AR2RunnerError("cell execution-block ledger changed")
    return expected


def _execution_stage(block: str) -> str:
    if block == "discovery-components":
        return DISCOVERY_COMPONENTS_STAGE
    if block in BLOCK_ORDER[1:]:
        return REMAINING_MAIN_STAGE
    raise V6AR2RunnerError("unknown execution block")


def _ordered_cells(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = fixture.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 fixture does not contain 2,304 cells")
    copied = [dict(cell) for cell in cells]
    if any(not isinstance(cell.get("cell_id"), str) for cell in copied):
        raise V6AR2RunnerError("R2 cell ID registry is malformed")
    if len({cell["cell_id"] for cell in copied}) != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 cell IDs are not globally unique")
    block_rank = {block: index for index, block in enumerate(BLOCK_ORDER)}
    copied.sort(key=lambda cell: (block_rank[_execution_block(cell)], cell["cell_id"]))
    if Counter(_execution_block(cell) for cell in copied) != Counter(BLOCK_COUNTS):
        raise V6AR2RunnerError("R2 execution block counts changed")
    if Counter(cell.get("family") for cell in copied) != Counter(FAMILY_COUNTS):
        raise V6AR2RunnerError("R2 family counts changed")
    observed_split_family = {
        split: Counter(cell.get("family") for cell in copied if _cell_split(cell) == split)
        for split in ("discovery", "confirmation")
    }
    if observed_split_family != {
        split: Counter(counts) for split, counts in SPLIT_FAMILY_COUNTS.items()
    }:
        raise V6AR2RunnerError("R2 split-by-family counts changed")
    return copied


def load_and_rebuild_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    observed = _load_json(FIXTURE)
    manifest = _load_json(FIXTURE_MANIFEST)
    builder = _builder_module()
    required_constants = {
        "EXPECTED_CALL_COUNT": EXPECTED_CALLS,
        "PER_WORLD_FAMILY_COUNTS": {
            "property_retrieval": 32,
            "codebook_lookup": 16,
            "single_target_composition": 32,
            "two_fact_composition": 64,
        },
        "FAMILY_COUNTS": FAMILY_COUNTS,
        "DISCOVERY_COMPONENT_CALL_COUNT": DISCOVERY_COMPONENT_CALLS,
        "REMAINING_MAIN_CALL_COUNT": REMAINING_MAIN_CALLS,
        "EXECUTION_BLOCK_COUNTS": BLOCK_COUNTS,
        "EXECUTION_BLOCK_ORDER": BLOCK_ORDER,
    }
    for name, expected in required_constants.items():
        if getattr(builder, name, None) != expected:
            raise V6AR2RunnerError(f"R2 builder constant changed: {name}")
    rebuilt = builder.build_fixture()
    if observed != rebuilt:
        raise V6AR2RunnerError("R2 fixture does not rebuild exactly")
    builder.validate_fixture(observed)
    if manifest.get("fixture_file_sha256") != file_sha256(FIXTURE):
        raise V6AR2RunnerError("R2 fixture file hash changed")
    if manifest.get("fixture_canonical_sha256") != canonical_sha256(observed):
        raise V6AR2RunnerError("R2 fixture canonical hash changed")
    if manifest.get("builder_file_sha256") != file_sha256(BUILDER):
        raise V6AR2RunnerError("R2 builder file hash changed")
    if observed.get("expected_call_count") != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 fixture expected-call count changed")
    _registered_symbols(observed)
    _ordered_cells(observed)
    return observed, manifest


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
            raise V6AR2RunnerError(f"registered prompt symbol {symbol!r} is absent")
        for start in starts:
            end = start + len(symbol)
            matches = [
                index
                for index, pair in enumerate(offsets)
                if int(pair[0]) < end and int(pair[1]) > start
            ]
            if len(matches) != 1:
                raise V6AR2RunnerError("glyph occurrence is not exactly one prompt token")
            token_index = matches[0]
            token_start, token_end = map(int, offsets[token_index])
            if rendered[token_start:token_end] != NATURAL_SURFACE_PREFIX + symbol:
                raise V6AR2RunnerError("prompt glyph is not its exact natural space-plus-glyph token")
            receipts.append(
                {
                    "symbol": symbol,
                    "character_start": start,
                    "character_end": end,
                    "token_index": token_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "token_id": int(input_ids[token_index]),
                }
            )
    return receipts


def render_prompt(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_text = cell.get("prompt_text")
    if not isinstance(prompt_text, str) or cell.get("prompt_sha256") != text_sha256(prompt_text):
        raise V6AR2RunnerError("cell prompt text or hash changed")
    if fixture.get("assistant_prefill") != "ANSWER:" or cell.get("assistant_prefill") != "ANSWER:":
        raise V6AR2RunnerError("assistant prefill changed")

    semantic_answer = cell.get("correct_answer")
    semantic_distractor = cell.get("distractor_answer")
    if (
        not isinstance(semantic_answer, str)
        or len(semantic_answer) != 1
        or not isinstance(semantic_distractor, str)
        or len(semantic_distractor) != 1
        or semantic_answer == semantic_distractor
    ):
        raise V6AR2RunnerError("cell semantic answer ledger changed")
    expected_surface = NATURAL_SURFACE_PREFIX + semantic_answer
    distractor_surface = NATURAL_SURFACE_PREFIX + semantic_distractor
    if (
        cell.get("correct_answer_surface") != expected_surface
        or cell.get("distractor_answer_surface") != distractor_surface
    ):
        raise V6AR2RunnerError("cell natural answer surface changed")

    messages = [
        {"role": "system", "content": fixture["system_message"]},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": fixture["assistant_prefill"]},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    except Exception as error:
        raise V6AR2RunnerError("chat template failed while rendering an R2 prompt") from error
    if not isinstance(rendered, str) or not rendered.endswith("ANSWER:"):
        raise V6AR2RunnerError("rendered R2 chat does not end at ANSWER:")

    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise V6AR2RunnerError("tokenizer cannot return R2 prompt offsets") from error
    input_ids = _as_int_vector(encoded["input_ids"], "R2 input IDs")
    direct_ids = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False),
        "R2 directly encoded input IDs",
    )
    if input_ids != direct_ids:
        raise V6AR2RunnerError("offset and direct R2 tokenization disagree")
    offsets = [list(map(int, pair)) for pair in encoded["offset_mapping"]]
    if len(offsets) != len(input_ids):
        raise V6AR2RunnerError("R2 offset and input-ID lengths disagree")
    if rendered[-1] != ":" or input_ids[-1] != RESPONSE_SITE_TOKEN_ID:
        raise V6AR2RunnerError("R2 colon response site is not token ID 25")

    expected_token_id = _prefix_extension_token_id(
        tokenizer,
        rendered,
        expected_surface,
        label="natural correct answer",
    )
    distractor_token_id = _prefix_extension_token_id(
        tokenizer,
        rendered,
        distractor_surface,
        label="natural distractor answer",
    )
    if _decode_one(tokenizer, expected_token_id) != expected_surface:
        raise V6AR2RunnerError("natural correct token does not decode to space plus glyph")
    if _decode_one(tokenizer, distractor_token_id) != distractor_surface:
        raise V6AR2RunnerError("natural distractor token does not decode to space plus glyph")
    if expected_token_id == distractor_token_id:
        raise V6AR2RunnerError("natural correct and distractor token IDs collide")

    bare_expected_token_id = _prefix_extension_token_id(
        tokenizer,
        rendered,
        semantic_answer,
        label="prohibited bare correct answer",
    )
    bare_distractor_token_id = _prefix_extension_token_id(
        tokenizer,
        rendered,
        semantic_distractor,
        label="prohibited bare distractor answer",
    )
    if _decode_one(tokenizer, bare_expected_token_id) != semantic_answer:
        raise V6AR2RunnerError("bare correct token does not decode to its glyph")
    if _decode_one(tokenizer, bare_distractor_token_id) != semantic_distractor:
        raise V6AR2RunnerError("bare distractor token does not decode to its glyph")
    if (
        bare_expected_token_id == expected_token_id
        or bare_distractor_token_id == distractor_token_id
    ):
        raise V6AR2RunnerError("natural and prohibited bare answer token IDs are not distinct")

    registry = _registered_symbols(fixture)
    present_symbols = [symbol for symbol in registry if symbol in rendered]
    occurrences = _symbol_occurrence_receipts(
        rendered,
        offsets,
        input_ids,
        present_symbols,
    )
    if semantic_answer not in present_symbols or semantic_distractor not in present_symbols:
        raise V6AR2RunnerError("answer pair is not present in its R2 prompt")

    block = _execution_block(cell)
    stage = _execution_stage(block)
    message_sha256 = canonical_sha256(messages)
    prompt_core = {
        "schema_version": PROMPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "world_id": cell["world_id"],
        "global_index": cell["global_index"],
        "role_index": cell["role_index"],
        "role": _cell_split(cell),
        "family": cell["family"],
        "cell_stage": cell["stage"],
        "execution_stage": stage,
        "execution_block": block,
        "factors": dict(cell["factors"]),
        "factor_levels": dict(cell["factor_levels"]),
        "foldover_g": cell["foldover_g"],
        "semantic_answer": semantic_answer,
        "semantic_distractor": semantic_distractor,
        "expected_answer_surface": expected_surface,
        "distractor_answer_surface": distractor_surface,
        "expected_token_id": expected_token_id,
        "distractor_token_id": distractor_token_id,
        "bare_expected_surface": semantic_answer,
        "bare_distractor_surface": semantic_distractor,
        "bare_expected_token_id": bare_expected_token_id,
        "bare_distractor_token_id": bare_distractor_token_id,
        "bare_token_variants_registered": False,
        "bare_token_pooling_allowed": False,
        "registered_symbols_present": present_symbols,
        "symbol_occurrences": occurrences,
        "messages_sha256": message_sha256,
        "rendered_text_sha256": text_sha256(rendered),
        "execution_input_ids": input_ids,
        "execution_attention_mask": [1] * len(input_ids),
        "input_token_count": len(input_ids),
        "response_site_index": len(input_ids) - 1,
        "response_site_token_id": input_ids[-1],
        "response_site_text": rendered[-1],
        "teacher_forced_prompt_forward_planned": True,
        "generation_used": False,
        "model_calls_before_plan_freeze": 0,
    }
    return {**prompt_core, "prompt_id": canonical_sha256(prompt_core)}


def _validate_symbol_token_contracts(
    prompts: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry = _registered_symbols(fixture)
    registered = set(registry)
    prompt_ids: dict[str, set[int]] = defaultdict(set)
    natural_ids: dict[str, set[int]] = defaultdict(set)
    bare_ids: dict[str, set[int]] = defaultdict(set)
    prompt_occurrences: Counter[str] = Counter()
    answer_contexts: Counter[str] = Counter()

    for prompt in prompts:
        for occurrence in prompt["symbol_occurrences"]:
            symbol = occurrence["symbol"]
            if symbol not in registered:
                raise V6AR2RunnerError("unregistered glyph entered an R2 occurrence receipt")
            prompt_ids[symbol].add(int(occurrence["token_id"]))
            prompt_occurrences[symbol] += 1
        for semantic_key, natural_key, bare_key in (
            ("semantic_answer", "expected_token_id", "bare_expected_token_id"),
            ("semantic_distractor", "distractor_token_id", "bare_distractor_token_id"),
        ):
            symbol = prompt[semantic_key]
            if symbol not in registered:
                raise V6AR2RunnerError("unregistered glyph entered an R2 answer ledger")
            natural_ids[symbol].add(int(prompt[natural_key]))
            bare_ids[symbol].add(int(prompt[bare_key]))
            answer_contexts[symbol] += 1

    if set(prompt_ids) != registered or set(natural_ids) != registered or set(bare_ids) != registered:
        raise V6AR2RunnerError("R2 symbol token coverage changed")
    if any(len(values) != 1 for values in prompt_ids.values()):
        raise V6AR2RunnerError("R2 prompt-occurrence token ID is context-unstable")
    if any(len(values) != 1 for values in natural_ids.values()):
        raise V6AR2RunnerError("R2 natural continuation token ID is context-unstable")
    if any(len(values) != 1 for values in bare_ids.values()):
        raise V6AR2RunnerError("R2 bare continuation token ID is context-unstable")

    contracts: list[dict[str, Any]] = []
    observed_natural_ids: set[int] = set()
    observed_bare_ids: set[int] = set()
    for symbol in registry:
        prompt_id = next(iter(prompt_ids[symbol]))
        natural_id = next(iter(natural_ids[symbol]))
        bare_id = next(iter(bare_ids[symbol]))
        if prompt_id != natural_id:
            raise V6AR2RunnerError(
                "R2 prompt-occurrence token ID differs from its natural continuation token ID"
            )
        if natural_id == bare_id:
            raise V6AR2RunnerError("R2 natural token ID equals its prohibited bare token ID")
        if natural_id in observed_natural_ids:
            raise V6AR2RunnerError("R2 natural token IDs are not globally unique")
        if bare_id in observed_bare_ids:
            raise V6AR2RunnerError("R2 bare token IDs are not globally unique")
        observed_natural_ids.add(natural_id)
        observed_bare_ids.add(bare_id)
        contracts.append(
            {
                "symbol": symbol,
                "natural_surface": NATURAL_SURFACE_PREFIX + symbol,
                "natural_token_id": natural_id,
                "prompt_occurrence_token_id": prompt_id,
                "bare_surface": symbol,
                "bare_token_id": bare_id,
                "bare_variant_registered": False,
                "bare_variant_pooling_allowed": False,
                "prompt_occurrence_count": prompt_occurrences[symbol],
                "answer_context_count": answer_contexts[symbol],
            }
        )
    if observed_natural_ids & observed_bare_ids:
        raise V6AR2RunnerError("R2 natural and bare token registries overlap")
    return contracts


def _validate_topology_shapes(
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        factors = prompt["factors"]
        family = prompt["family"]
        if family == "property_retrieval":
            key = (family, prompt["world_id"], factors["p"], factors["v"])
        elif family in ("two_fact_composition", "single_target_composition"):
            key = (
                family,
                prompt["world_id"],
                factors["p"],
                factors["m"],
                factors["u"],
            )
        elif family == "codebook_lookup":
            continue
        else:
            raise V6AR2RunnerError("unknown family entered R2 topology receipts")
        groups[key].append(prompt)

    expected_group_counts = {
        "property_retrieval": 16 * 4,
        "two_fact_composition": 16 * 8,
        "single_target_composition": 16 * 8,
    }
    if Counter(key[0] for key in groups) != Counter(expected_group_counts):
        raise V6AR2RunnerError("R2 topology-mate group registry changed")

    receipts: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        family = str(key[0])
        if family in ("property_retrieval", "two_fact_composition"):
            observed_vertices = {
                (member["factors"]["o"], member["factors"]["q"], member["factors"]["a"])
                for member in members
            }
            expected_vertices = {
                (o, q, a) for o in (-1, 1) for q in (-1, 1) for a in (-1, 1)
            }
            expected_size = 8
        else:
            observed_vertices = {
                (member["factors"]["q"], member["factors"]["a"])
                for member in members
            }
            expected_vertices = {(q, a) for q in (-1, 1) for a in (-1, 1)}
            expected_size = 4
        if len(members) != expected_size or observed_vertices != expected_vertices:
            raise V6AR2RunnerError("R2 topology-mate vertices changed")
        shapes = {
            (
                int(member["input_token_count"]),
                int(member["response_site_index"]),
                int(member["response_site_token_id"]),
                len(member["execution_attention_mask"]),
            )
            for member in members
        }
        if len(shapes) != 1:
            raise V6AR2RunnerError("R2 topology mates do not have equal response shape")
        receipts.append(
            {
                "family": family,
                "group_key": list(key[1:]),
                "member_prompt_ids": sorted(str(member["prompt_id"]) for member in members),
                "vertices": sorted([list(vertex) for vertex in observed_vertices]),
                "shape": list(next(iter(shapes))),
            }
        )
    return receipts


def _validate_plan_firewalls(
    cells: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    symbol_contracts: Sequence[Mapping[str, Any]],
    historical_pools: Mapping[str, Sequence[Any]],
    historical_source_receipt: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if len(cells) != EXPECTED_CALLS or len(prompts) != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 plan row count changed")
    identity_fields = {
        "cell_id": [str(cell["cell_id"]) for cell in cells],
        "prompt_id": [str(prompt["prompt_id"]) for prompt in prompts],
        "user_prompt_sha256": [str(cell["prompt_sha256"]) for cell in cells],
        "rendered_text_sha256": [str(prompt["rendered_text_sha256"]) for prompt in prompts],
    }
    for label, values in identity_fields.items():
        if len(set(values)) != EXPECTED_CALLS:
            raise V6AR2RunnerError(f"R2 {label} values are not globally unique")

    if [cell["cell_id"] for cell in cells] != [prompt["cell_id"] for prompt in prompts]:
        raise V6AR2RunnerError("R2 cell and rendered-prompt order differs")
    if any(prompt["response_site_token_id"] != RESPONSE_SITE_TOKEN_ID for prompt in prompts):
        raise V6AR2RunnerError("R2 plan contains a non-colon response site")
    if any(prompt["generation_used"] is not False for prompt in prompts):
        raise V6AR2RunnerError("R2 plan contains generation")

    registry = set(_registered_symbols(fixture))
    historical_scalars = set(historical_pools["single_codepoint_scalars"])
    natural_ids = {int(contract["natural_token_id"]) for contract in symbol_contracts}
    bare_ids = {int(contract["bare_token_id"]) for contract in symbol_contracts}
    historical_token_ids = {
        int(value) for value in historical_pools["answer_token_ids"]
    }
    candidate_rows = candidate_receipt.get("eligible_candidates_in_hash_order")
    if not isinstance(candidate_rows, list) or [
        row.get("glyph") for row in candidate_rows if isinstance(row, Mapping)
    ] != list(EXPECTED_ELIGIBLE_CANDIDATES):
        raise V6AR2RunnerError("R2 candidate-universe receipt changed")
    special_proof = candidate_receipt.get("special_and_added_token_proof")
    if not isinstance(special_proof, Mapping):
        raise V6AR2RunnerError("R2 special-token proof is missing")
    prohibited_token_ids = {
        int(value)
        for key in ("all_special_ids", "added_token_ids")
        for value in special_proof.get(key, [])
    }
    if (natural_ids | bare_ids) & prohibited_token_ids:
        raise V6AR2RunnerError("an R2 answer token is special or added/control")
    if registry & historical_scalars:
        raise V6AR2RunnerError("R2 glyph registry overlaps the historical scalar pool")
    if natural_ids & historical_token_ids:
        raise V6AR2RunnerError("R2 natural answer IDs overlap historical answer IDs")
    if bare_ids & historical_token_ids:
        raise V6AR2RunnerError("R2 bare answer IDs overlap historical answer IDs")

    worlds = fixture.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 16:
        raise V6AR2RunnerError("R2 world registry changed")
    world_ids = [world.get("world_id") for world in worlds]
    instance_keys = [world.get("instance_key") for world in worlds]
    target_entities = [world.get("target_entity") for world in worlds]
    other_entities = [world.get("other_entity") for world in worlds]
    for label, values in (
        ("world IDs", world_ids),
        ("instance keys", instance_keys),
        ("target entities", target_entities),
        ("other entities", other_entities),
    ):
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != 16:
            raise V6AR2RunnerError(f"R2 {label} are not unique")
    if set(target_entities) & set(other_entities):
        raise V6AR2RunnerError("R2 target and other entity registries overlap")
    r2_entities = set(target_entities) | set(other_entities)
    r2_instance_keys = set(instance_keys)
    r2_user_hashes = set(identity_fields["user_prompt_sha256"])
    r2_rendered_hashes = set(identity_fields["rendered_text_sha256"])
    historical_overlap_registry = {
        "single_codepoint_scalars": sorted(registry & historical_scalars),
        "natural_answer_token_ids": sorted(natural_ids & historical_token_ids),
        "bare_answer_token_ids": sorted(bare_ids & historical_token_ids),
        "entity_strings": sorted(
            r2_entities & set(historical_pools["entity_strings"])
        ),
        "instance_keys": sorted(
            r2_instance_keys & set(historical_pools["instance_keys"])
        ),
        "user_prompt_hashes": sorted(
            r2_user_hashes & set(historical_pools["user_prompt_hashes"])
        ),
        "rendered_text_hashes": sorted(
            r2_rendered_hashes & set(historical_pools["rendered_text_hashes"])
        ),
    }
    if any(historical_overlap_registry.values()):
        raise V6AR2RunnerError(
            f"R2 historical identity firewall failed: {historical_overlap_registry}"
        )

    split_symbols: dict[str, set[str]] = {"discovery": set(), "confirmation": set()}
    for world in worlds:
        role = world.get("role")
        symbols = world.get("symbols")
        if role not in split_symbols or not isinstance(symbols, list) or len(symbols) != 4:
            raise V6AR2RunnerError("R2 world role or symbol allocation changed")
        split_symbols[str(role)].update(str(symbol) for symbol in symbols)
    if (
        len(split_symbols["discovery"]) != 32
        or len(split_symbols["confirmation"]) != 32
        or split_symbols["discovery"] & split_symbols["confirmation"]
        or split_symbols["discovery"] | split_symbols["confirmation"] != registry
    ):
        raise V6AR2RunnerError("R2 discovery/confirmation symbol firewall changed")
    contract_by_symbol = {
        str(contract["symbol"]): contract for contract in symbol_contracts
    }
    split_token_ids = {
        split: {
            int(contract_by_symbol[symbol][token_key])
            for symbol in symbols
            for token_key in ("natural_token_id", "bare_token_id")
        }
        for split, symbols in split_symbols.items()
    }
    if split_token_ids["discovery"] & split_token_ids["confirmation"]:
        raise V6AR2RunnerError("R2 discovery/confirmation answer-token firewall changed")
    split_worlds = {
        split: [world for world in worlds if world.get("role") == split]
        for split in ("discovery", "confirmation")
    }
    split_entities = {
        split: {
            str(world[key])
            for world in members
            for key in ("target_entity", "other_entity")
        }
        for split, members in split_worlds.items()
    }
    split_keys = {
        split: {str(world["instance_key"]) for world in members}
        for split, members in split_worlds.items()
    }
    split_user_hashes = {
        split: {
            str(cell["prompt_sha256"])
            for cell in cells
            if cell.get("role") == split
        }
        for split in ("discovery", "confirmation")
    }
    split_rendered_hashes = {
        split: {
            str(prompt["rendered_text_sha256"])
            for prompt in prompts
            if prompt.get("role") == split
        }
        for split in ("discovery", "confirmation")
    }
    split_firewalls = {
        "answer_token_ids": split_token_ids,
        "entity_strings": split_entities,
        "instance_keys": split_keys,
        "user_prompt_hashes": split_user_hashes,
        "rendered_text_hashes": split_rendered_hashes,
    }
    if any(
        values["discovery"] & values["confirmation"]
        for values in split_firewalls.values()
    ):
        raise V6AR2RunnerError("R2 discovery/confirmation identity firewall changed")

    observed_blocks = Counter(prompt["execution_block"] for prompt in prompts)
    observed_stages = Counter(prompt["execution_stage"] for prompt in prompts)
    if observed_blocks != Counter(BLOCK_COUNTS):
        raise V6AR2RunnerError("R2 rendered execution-block counts changed")
    if observed_stages != Counter(
        {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        }
    ):
        raise V6AR2RunnerError("R2 rendered execution-stage counts changed")
    if any(prompt["execution_block"] != "discovery-components" for prompt in prompts[:640]):
        raise V6AR2RunnerError("R2 first 640 rows are not discovery components")
    expected_remaining_blocks = [
        *("discovery-topology" for _ in range(512)),
        *("confirmation-components" for _ in range(640)),
        *("confirmation-topology" for _ in range(512)),
    ]
    if [prompt["execution_block"] for prompt in prompts[640:]] != expected_remaining_blocks:
        raise V6AR2RunnerError("R2 sealed remaining-main block order changed")

    return {
        "cell_ids_unique": True,
        "prompt_ids_unique": True,
        "user_prompt_hashes_unique": True,
        "rendered_chat_hashes_unique": True,
        "historical_source_receipt": dict(historical_source_receipt),
        "historical_pool_counts": dict(historical_source_receipt["pool_counts"]),
        "historical_pool_sha256": dict(historical_source_receipt["pool_sha256"]),
        "historical_overlap_registry": historical_overlap_registry,
        "historical_glyph_token_entity_key_prompt_rendered_firewall": True,
        "candidate_universe_sha256": canonical_sha256(candidate_receipt),
        "candidate_universe_reproduced": True,
        "discovery_confirmation_symbol_firewall": True,
        "discovery_confirmation_token_entity_key_prompt_rendered_firewall": True,
        "world_entity_key_firewall": True,
        "stage_order": list(BLOCK_ORDER),
        "stage_counts": {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        },
    }


def raw_shard_specs(
    vocab_size: int,
    result_root: Path = RESULT_ROOT,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise V6AR2RunnerError("model vocabulary size is invalid")
    paths = artifact_paths(result_root)
    phase_specs: dict[str, list[dict[str, Any]]] = {}
    phase_registry = (
        (DISCOVERY_COMPONENTS_STAGE, 0, DISCOVERY_COMPONENT_CALLS, paths["discovery_raw_root"]),
        (REMAINING_MAIN_STAGE, DISCOVERY_COMPONENT_CALLS, REMAINING_MAIN_CALLS, paths["remaining_raw_root"]),
    )
    for phase, global_start, count, raw_root in phase_registry:
        specs: list[dict[str, Any]] = []
        for index, phase_start in enumerate(range(0, count, RAW_SHARD_ROWS)):
            phase_stop = min(phase_start + RAW_SHARD_ROWS, count)
            rows = phase_stop - phase_start
            if phase == DISCOVERY_COMPONENTS_STAGE:
                execution_block = "discovery-components"
                block_shard_index = index
                block_start_row = phase_start
                block_stop_row = phase_stop
            else:
                matches = [
                    (block, layout)
                    for block, layout in REMAINING_BLOCK_LAYOUT.items()
                    if int(layout["stage_start_row"]) <= phase_start
                    and phase_stop <= int(layout["stage_stop_row"])
                ]
                if len(matches) != 1:
                    raise V6AR2RunnerError(
                        "remaining raw shard crosses a registered block boundary"
                    )
                execution_block, layout = matches[0]
                block_shard_index = index - int(layout["raw_shard_indices"][0])
                block_start_row = phase_start - int(layout["stage_start_row"])
                block_stop_row = phase_stop - int(layout["stage_start_row"])
            specs.append(
                {
                    "execution_stage": phase,
                    "execution_block": execution_block,
                    "index": index,
                    "block_shard_index": block_shard_index,
                    "phase_start_row": phase_start,
                    "phase_stop_row": phase_stop,
                    "block_start_row": block_start_row,
                    "block_stop_row": block_stop_row,
                    "global_start_row": global_start + phase_start,
                    "global_stop_row": global_start + phase_stop,
                    "rows": rows,
                    "shape": [rows, vocab_size],
                    "dtype": "<f4",
                    "path": str(raw_root / f"shard_{index:03d}.npy"),
                }
            )
        phase_specs[phase] = specs
    if (
        len(phase_specs[DISCOVERY_COMPONENTS_STAGE]) != 10
        or len(phase_specs[REMAINING_MAIN_STAGE]) != 26
        or sum(spec["rows"] for specs in phase_specs.values() for spec in specs) != EXPECTED_CALLS
        or any(spec["rows"] != RAW_SHARD_ROWS for specs in phase_specs.values() for spec in specs)
    ):
        raise V6AR2RunnerError("R2 raw-logit shard allocation changed")
    observed_remaining_groups = {
        block: [
            spec["index"]
            for spec in phase_specs[REMAINING_MAIN_STAGE]
            if spec["execution_block"] == block
        ]
        for block in REMAINING_BLOCK_LAYOUT
    }
    if observed_remaining_groups != {
        block: layout["raw_shard_indices"]
        for block, layout in REMAINING_BLOCK_LAYOUT.items()
    }:
        raise V6AR2RunnerError("R2 remaining raw-shard block groups changed")
    return phase_specs


def disk_preflight(
    shard_specs: Mapping[str, Sequence[Mapping[str, Any]]],
    result_root: Path = RESULT_ROOT,
    *,
    reserve_bytes: int = 1024**3,
) -> dict[str, Any]:
    if reserve_bytes < 0:
        raise V6AR2RunnerError("disk reserve cannot be negative")
    specs = [spec for phase in EXECUTION_STAGES for spec in shard_specs[phase]]
    expected_bytes = sum(
        math.prod(spec["shape"]) * 4
        for spec in specs
        if spec.get("dtype") == "<f4"
    )
    if len(specs) != 36 or expected_bytes <= 0:
        raise V6AR2RunnerError("R2 raw-logit disk plan is malformed")
    probe = result_root.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = int(shutil.disk_usage(probe).free)
    required_bytes = expected_bytes + reserve_bytes
    if free_bytes < required_bytes:
        raise V6AR2RunnerError("insufficient disk space for both R2 full-vocabulary stages")
    return {
        "stored_logits_dtype": "float32_little_endian",
        "raw_logits_expected_bytes": expected_bytes,
        "reserve_bytes": reserve_bytes,
        "required_disk_free_bytes": required_bytes,
        "disk_free_bytes": free_bytes,
        "probe_path": str(probe),
        "pass": True,
    }


def _implementation_paths(analyzer_path: Path = DEFAULT_ANALYZER) -> dict[str, Path]:
    paths = {
        "runner": Path(__file__).resolve(),
        "analyzer": analyzer_path.resolve(),
        "builder": BUILDER.resolve(),
        "fixture": FIXTURE.resolve(),
        "fixture_manifest": FIXTURE_MANIFEST.resolve(),
        "design_document": DESIGN_DOCUMENT.resolve(),
        "pre_execution_repair1_document": PRE_EXECUTION_REPAIR1_DOCUMENT.resolve(),
        "pre_execution_repair1_receipt": PRE_EXECUTION_REPAIR1_RECEIPT.resolve(),
        "predecessor_v6a_design": V6A_PREDECESSOR_DESIGN.resolve(),
        "v2_preregistration": V2_PREREGISTRATION.resolve(),
        "builder_test": BUILDER_TEST.resolve(),
        "runner_test": RUNNER_TEST.resolve(),
        "analyzer_test": ANALYZER_TEST.resolve(),
        "sealed_v2_runner": SEALED_V2_RUNNER.resolve(),
        "sealed_v2_analyzer": SEALED_V2_ANALYZER.resolve(),
        "sealed_v2_posthoc_source": SEALED_V2_POSTHOC.resolve(),
        "sealed_v2_posthoc_test": SEALED_V2_POSTHOC_TEST.resolve(),
        "sealed_v2_dependency_lock": V2_DEPENDENCY_LOCK.resolve(),
        "sealed_v2_terminal_analysis": V2_TERMINAL_ANALYSIS.resolve(),
        "sealed_v2_posthoc_output": V2_POSTHOC_OUTPUT.resolve(),
    }
    sealed_names = (
        "plan_manifest.json",
        "design.json",
        "tokenization_receipt.json",
        "loader_smoke_receipt.json",
        "qualification_baseline_attempt.json",
        "qualification_baseline_execution_manifest.json",
        "qualification_baseline_records.jsonl",
    )
    for filename in sealed_names:
        paths[f"sealed_v2_{filename.replace('.', '_')}"] = (V2_RESULT_ROOT / filename).resolve()
    for index in range(6):
        paths[f"sealed_v2_raw_logits_shard_{index:03d}"] = (
            V2_RESULT_ROOT
            / "raw_logits"
            / "qualification-baseline"
            / f"shard_{index:03d}.npy"
        ).resolve()
    for source_kind, sources in (
        ("fixture", HISTORICAL_FIXTURE_SOURCES),
        ("artifact", HISTORICAL_ARTIFACT_SOURCES),
    ):
        for name, path in sources.items():
            paths[f"historical_{source_kind}_{name}"] = path.resolve()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise V6AR2RunnerError(f"R2 implementation/dependency files are missing: {missing}")
    return paths


def _validate_v2_terminal_and_posthoc() -> dict[str, Any]:
    exact_hashes = {
        V2_DEPENDENCY_LOCK: SEALED_V2_DEPENDENCY_LOCK_SHA256,
        V2_TERMINAL_ANALYSIS: V2_TERMINAL_ANALYSIS_SHA256,
        V2_POSTHOC_OUTPUT: V2_POSTHOC_OUTPUT_SHA256,
        SEALED_V2_RUNNER: SEALED_V2_RUNNER_SHA256,
    }
    for path, expected in exact_hashes.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise V6AR2RunnerError(f"sealed V2/R2 provenance hash changed: {path}")
    terminal = _load_json(V2_TERMINAL_ANALYSIS)
    if (
        terminal.get("status") != "V6A_QUALIFICATION_COMPONENT_FAIL"
        or terminal.get("engineering_valid") is not True
        or terminal.get("component_qualified") is not False
        or terminal.get("composition_calls_analyzed") != 0
        or terminal.get("model_calls_issued_by_analyzer") != 0
    ):
        raise V6AR2RunnerError("sealed V2 terminal component-fail identity changed")
    posthoc = _load_json(V2_POSTHOC_OUTPUT)
    if (
        posthoc.get("schema_version")
        != "coherent-readout-v6a-qualification-v2-contextual-token-posthoc-v2"
        or posthoc.get("status") != "V2_CONTEXTUAL_TOKEN_POSTHOC_DESCRIPTIVE_COMPLETE"
        or posthoc.get("source_terminal_status") != "V6A_QUALIFICATION_COMPONENT_FAIL"
        or posthoc.get("descriptive_only") is not True
        or posthoc.get("authorization_issued") is not False
        or posthoc.get("source_component_qualification_reopened") is not False
        or posthoc.get("model_calls_issued_by_posthoc") != 0
        or posthoc.get("generation_calls_issued_by_posthoc") != 0
        or posthoc.get("composition_calls_analyzed") != 0
        or posthoc.get("natural_answer_surface_rule")
        != "one ASCII space followed by one registered glyph"
    ):
        raise V6AR2RunnerError("sealed V2 contextual-token posthoc identity changed")
    return {
        "terminal_status": terminal["status"],
        "terminal_analysis_sha256": V2_TERMINAL_ANALYSIS_SHA256,
        "posthoc_status": posthoc["status"],
        "posthoc_output_sha256": V2_POSTHOC_OUTPUT_SHA256,
        "posthoc_descriptive_only": True,
        "posthoc_authorization_issued": False,
    }


def _validate_pre_execution_repair1() -> dict[str, Any]:
    receipt = _load_json(PRE_EXECUTION_REPAIR1_RECEIPT)
    _validate_canonical_envelope(receipt, "R2 pre-execution Repair-1 receipt")
    if (
        receipt.get("schema_version")
        != "coherent-readout-v6a-r2-pre-execution-repair1-v1"
        or receipt.get("status")
        != "ZERO_FORWARD_PLAN_REPLAY_ENGINEERING_INVALID_RETIRED"
        or receipt.get("model_calls") != 0
        or receipt.get("generation_calls") != 0
        or receipt.get("behavioral_calls") != 0
        or receipt.get("failure")
        != {
            "analyzer_error": "R2 planned record identity changed",
            "mandatory_replay_passed": False,
            "status": "ZERO_FORWARD_PLAN_REPLAY_ENGINEERING_INVALID",
        }
    ):
        raise V6AR2RunnerError("R2 pre-execution Repair-1 status changed")
    retired_relative = receipt.get("retired_result_root")
    if (
        not isinstance(retired_relative, str)
        or (ROOT / retired_relative).resolve()
        != RETIRED_ZERO_FORWARD_RESULT_ROOT.resolve()
        or not RETIRED_ZERO_FORWARD_RESULT_ROOT.is_dir()
        or RETIRED_ZERO_FORWARD_RESULT_ROOT.is_symlink()
    ):
        raise V6AR2RunnerError("R2 retired zero-forward plan root changed")
    inventory = receipt.get("retired_inventory")
    expected_names = {
        "dependency_lock.json",
        "design.json",
        "plan_manifest.json",
        "tokenization_receipt.json",
    }
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != expected_names
        or {path.name for path in RETIRED_ZERO_FORWARD_RESULT_ROOT.iterdir()}
        != expected_names
    ):
        raise V6AR2RunnerError("R2 retired zero-forward inventory changed")
    for name, binding in inventory.items():
        path = RETIRED_ZERO_FORWARD_RESULT_ROOT / str(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"file_sha256", "size_bytes"}
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != binding.get("size_bytes")
            or file_sha256(path) != binding.get("file_sha256")
        ):
            raise V6AR2RunnerError(f"R2 retired zero-forward file changed: {name}")

    retired_manifest = _load_json(
        RETIRED_ZERO_FORWARD_RESULT_ROOT / "plan_manifest.json"
    )
    retired_plan = retired_manifest.get("plan")
    if not isinstance(retired_plan, Mapping):
        raise V6AR2RunnerError("R2 retired zero-forward plan is missing")
    prompts = retired_plan.get("prompts")
    cells = retired_plan.get("cells")
    if (
        not isinstance(prompts, list)
        or not isinstance(cells, list)
        or len(prompts) != EXPECTED_CALLS
        or len(cells) != EXPECTED_CALLS
    ):
        raise V6AR2RunnerError("R2 retired zero-forward row registry changed")
    reconstructed = [record_identity_id(prompt) for prompt in prompts]
    identity_audit = {
        "cell_prompt_shared_six_match": sum(
            all(prompt.get(key) == cell.get(key) for key in CELL_SHARED_IDENTITY_FIELDS)
            for prompt, cell in zip(prompts, cells, strict=True)
        ),
        "cell_prompt_stored_identity_match": sum(
            prompt.get("record_identity_id") == cell.get("record_identity_id")
            for prompt, cell in zip(prompts, cells, strict=True)
        ),
        "cells_missing_prompt_id_by_design": sum(
            "prompt_id" not in cell for cell in cells
        ),
        "prompt_identity_reconstructed": sum(
            prompt.get("record_identity_id") == identity
            for prompt, identity in zip(prompts, reconstructed, strict=True)
        ),
        "unique_record_identity_ids": len(set(reconstructed)),
    }
    if identity_audit != receipt.get("identity_audit") or any(
        value != EXPECTED_CALLS for value in identity_audit.values()
    ):
        raise V6AR2RunnerError("R2 retired zero-forward identity audit changed")

    retired_registry = {
        "call_plan_sha256": retired_plan.get("call_plan_sha256"),
        "dependency_canonical_sha256": retired_plan.get(
            "dependency_canonical_sha256"
        ),
        "execution_revision": retired_plan.get("execution_revision"),
        "ordered_cell_ids_sha256": canonical_sha256(
            [cell["cell_id"] for cell in cells]
        ),
        "ordered_prompt_ids_sha256": canonical_sha256(
            [prompt["prompt_id"] for prompt in prompts]
        ),
        "ordered_record_identity_ids_sha256": canonical_sha256(reconstructed),
        "plan_manifest_canonical_sha256": canonical_sha256(retired_manifest),
        "plan_manifest_file_sha256": file_sha256(
            RETIRED_ZERO_FORWARD_RESULT_ROOT / "plan_manifest.json"
        ),
        "scientific_registry_sha256": retired_plan.get(
            "scientific_registry_sha256"
        ),
        "stage_plan_sha256": retired_plan.get("stage_plan_sha256"),
    }
    if retired_registry != receipt.get("retired_plan"):
        raise V6AR2RunnerError("R2 retired zero-forward registry changed")
    if (
        retired_manifest.get("model_calls") != 0
        or retired_manifest.get("generation_used") is not False
        or retired_manifest.get("artifact_scope") != "registered_main"
    ):
        raise V6AR2RunnerError("R2 retired zero-forward plan was not planning-only")

    retired_dependency = _load_json(
        RETIRED_ZERO_FORWARD_RESULT_ROOT / "dependency_lock.json"
    )
    implementation = retired_dependency.get("implementation_files")
    if not isinstance(implementation, Mapping):
        raise V6AR2RunnerError("R2 retired dependency source registry changed")
    source_key_map = {
        "analyzer_file_sha256": "analyzer",
        "analyzer_test_file_sha256": "analyzer_test",
        "design_document_file_sha256": "design_document",
        "runner_file_sha256": "runner",
        "runner_test_file_sha256": "runner_test",
    }
    observed_sources = {
        receipt_key: implementation.get(implementation_key, {}).get("sha256")
        for receipt_key, implementation_key in source_key_map.items()
    }
    if observed_sources != receipt.get("source_bindings_before_repair"):
        raise V6AR2RunnerError("R2 pre-repair source binding changed")
    return {
        "status": receipt["status"],
        "repair_receipt_canonical_sha256": receipt["canonical_sha256"],
        "repair_receipt_file_sha256": file_sha256(
            PRE_EXECUTION_REPAIR1_RECEIPT
        ),
        "retired_plan_manifest_file_sha256": retired_registry[
            "plan_manifest_file_sha256"
        ],
        "retired_scientific_registry_sha256": retired_registry[
            "scientific_registry_sha256"
        ],
        "model_calls": 0,
        "behavioral_calls": 0,
    }


def _validate_repair1_plan_preservation(plan: Mapping[str, Any]) -> None:
    receipt = _load_json(PRE_EXECUTION_REPAIR1_RECEIPT)
    _validate_canonical_envelope(receipt, "R2 pre-execution Repair-1 receipt")
    retired = receipt.get("retired_plan")
    prompts = plan.get("prompts")
    cells = plan.get("cells")
    if (
        not isinstance(retired, Mapping)
        or not isinstance(prompts, list)
        or not isinstance(cells, list)
        or len(prompts) != EXPECTED_CALLS
        or len(cells) != EXPECTED_CALLS
    ):
        raise V6AR2RunnerError("R2 Repair-1 preservation registry is malformed")
    observed = {
        "scientific_registry_sha256": plan.get("scientific_registry_sha256"),
        "stage_plan_sha256": plan.get("stage_plan_sha256"),
        "ordered_prompt_ids_sha256": canonical_sha256(
            [prompt["prompt_id"] for prompt in prompts]
        ),
        "ordered_record_identity_ids_sha256": canonical_sha256(
            [prompt["record_identity_id"] for prompt in prompts]
        ),
        "ordered_cell_ids_sha256": canonical_sha256(
            [cell["cell_id"] for cell in cells]
        ),
        "execution_revision": plan.get("execution_revision"),
    }
    expected = {key: retired.get(key) for key in observed}
    if observed != expected:
        raise V6AR2RunnerError(
            "R2 Repair-1 changed a frozen scientific or identity registry"
        )


def dependency_lock(analyzer_path: Path = DEFAULT_ANALYZER) -> dict[str, Any]:
    """Build a zero-forward dependency registry from the sealed V2 runtime."""

    provenance = _validate_v2_terminal_and_posthoc()
    repair1_provenance = _validate_pre_execution_repair1()
    _, historical_receipt = _historical_firewall_registry()
    v2_dependency = _load_json(V2_DEPENDENCY_LOCK)
    model = v2_dependency.get("model")
    if not isinstance(model, Mapping):
        raise V6AR2RunnerError("sealed V2 model registry is malformed")
    if (
        model.get("snapshot_revision") != MODEL_REVISION
        or model.get("config", {}).get("vocab_size") != 152_064
    ):
        raise V6AR2RunnerError("sealed V2 model identity changed")
    snapshot = Path(str(model.get("snapshot_path", ""))).resolve()
    if not snapshot.is_dir() or snapshot.name != MODEL_REVISION:
        raise V6AR2RunnerError("sealed V2 model snapshot is unavailable")
    assets = model.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise V6AR2RunnerError("sealed V2 model-asset hash registry is empty")
    for relative, binding in assets.items():
        path = snapshot / str(relative)
        if (
            not isinstance(binding, Mapping)
            or not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or file_sha256(path) != binding.get("sha256")
        ):
            raise V6AR2RunnerError(f"sealed model asset changed: {relative}")

    paths = _implementation_paths(analyzer_path)
    if file_sha256(paths["sealed_v2_runner"]) != SEALED_V2_RUNNER_SHA256:
        raise V6AR2RunnerError("R2 is not bound to the exact sealed V2 MPS loader")
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise V6AR2RunnerError(f"required package is missing: {name}") from error

    for source in historical_receipt["sources"]:
        implementation_name = f"historical_{source['kind']}_{source['name']}"
        path = paths.get(implementation_name)
        if path is None or file_sha256(path) != source["sha256"]:
            raise V6AR2RunnerError(
                f"historical dependency implementation binding changed: {implementation_name}"
            )

    core = {
        "schema_version": DEPENDENCY_SCHEMA,
        "registration_state": REGISTRATION_STATE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "model": dict(model),
        "runtime": {
            "device": DEVICE,
            "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
            "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
            "model_dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "stored_logits_dtype": "float32_little_endian",
            "model_loader_source": "sealed_v2_runner._load_model",
            "model_loader_source_sha256": SEALED_V2_RUNNER_SHA256,
        },
        "implementation_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "historical_firewall_sources": historical_receipt,
        "v2_terminal_and_posthoc_provenance": provenance,
        "pre_execution_repair1_provenance": repair1_provenance,
        "model_calls": 0,
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def load_tokenizer_from_sealed_v2(dependency: Mapping[str, Any]) -> Any:
    binding = dependency.get("implementation_files", {}).get("sealed_v2_runner", {})
    if (
        binding.get("sha256") != SEALED_V2_RUNNER_SHA256
        or file_sha256(SEALED_V2_RUNNER) != SEALED_V2_RUNNER_SHA256
    ):
        raise V6AR2RunnerError("sealed V2 tokenizer helper binding changed")
    module = _sealed_v2_runner_module()
    tokenizer = module._load_tokenizer(Path(dependency["model"]["snapshot_path"]))
    if not getattr(tokenizer, "is_fast", False):
        raise V6AR2RunnerError("R2 requires the sealed fast tokenizer")
    return tokenizer


def _replay_dependency_lock(
    observed: Mapping[str, Any],
    *,
    integration_test_only: bool,
    dependency_replay_override: Mapping[str, Any] | None,
) -> None:
    if integration_test_only:
        if dependency_replay_override is None:
            raise V6AR2RunnerError(
                "integration-test dependency replay requires the original override"
            )
        expected = dict(dependency_replay_override)
    else:
        if dependency_replay_override is not None:
            raise V6AR2RunnerError("registered dependency replay cannot be overridden")
        expected = dependency_lock(DEFAULT_ANALYZER)
    _validate_canonical_envelope(expected, "replayed R2 dependency lock")
    if dict(observed) != expected:
        raise V6AR2RunnerError("R2 dependency lock does not replay exactly")


def load_model_via_sealed_v2(
    snapshot: Path,
    plan: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> Any:
    """Delegate exactly to the hash-bound sealed V2 MPS loader."""

    binding = dependency.get("implementation_files", {}).get("sealed_v2_runner", {})
    if (
        binding.get("sha256") != SEALED_V2_RUNNER_SHA256
        or file_sha256(SEALED_V2_RUNNER) != SEALED_V2_RUNNER_SHA256
        or plan.get("model", {}).get("mps_allocator_warmup_policy")
        != MPS_ALLOCATOR_WARMUP_POLICY
    ):
        raise V6AR2RunnerError("sealed V2 MPS loader binding changed")
    return _sealed_v2_runner_module()._load_model(snapshot, plan)


def build_plan(
    tokenizer: Any,
    fixture: Mapping[str, Any],
    dependency: Mapping[str, Any],
    *,
    result_root: Path = RESULT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the complete plan and tokenizer receipt with zero model calls."""

    cells = _ordered_cells(fixture)
    block_indices: Counter[str] = Counter()
    stage_indices: Counter[str] = Counter()
    planned_cells: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for global_call_index, source_cell in enumerate(cells):
        block = _execution_block(source_cell)
        stage = _execution_stage(block)
        cell = {
            **source_cell,
            "execution_block": block,
            "execution_stage": stage,
            "global_call_index": global_call_index,
            "block_call_index": block_indices[block],
            "stage_call_index": stage_indices[stage],
        }
        block_indices[block] += 1
        stage_indices[stage] += 1
        prompt = render_prompt(tokenizer, fixture, cell)
        prompt.update(
            {
                "global_call_index": global_call_index,
                "block_call_index": cell["block_call_index"],
                "stage_call_index": cell["stage_call_index"],
            }
        )
        identity_id = record_identity_id(prompt)
        cell["record_identity_id"] = identity_id
        prompt["record_identity_id"] = identity_id
        planned_cells.append(cell)
        prompts.append(prompt)

    if block_indices != Counter(BLOCK_COUNTS):
        raise V6AR2RunnerError("R2 planned block indices changed")
    if stage_indices != Counter(
        {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        }
    ):
        raise V6AR2RunnerError("R2 planned stage indices changed")
    if any(
        prompt.get("record_identity_id") != record_identity_id(prompt)
        or cell.get("record_identity_id") != prompt.get("record_identity_id")
        for cell, prompt in zip(planned_cells, prompts, strict=True)
    ) or len({prompt["record_identity_id"] for prompt in prompts}) != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 pre-forward record identities changed")

    topology_receipts = _validate_topology_shapes(prompts)
    symbol_contracts = _validate_symbol_token_contracts(prompts, fixture)
    historical_pools, historical_source_receipt = _historical_firewall_registry()
    if dependency.get("historical_firewall_sources") != historical_source_receipt:
        raise V6AR2RunnerError("dependency lock historical source receipt changed")
    candidate_receipt = _reproduce_candidate_universe(
        tokenizer,
        fixture,
        historical_pools,
    )
    _validate_candidate_receipt_static(candidate_receipt)
    firewalls = _validate_plan_firewalls(
        planned_cells,
        prompts,
        fixture,
        symbol_contracts,
        historical_pools,
        historical_source_receipt,
        candidate_receipt,
    )
    config = dependency.get("model", {}).get("config", {})
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= max(
        contract["natural_token_id"] for contract in symbol_contracts
    ):
        raise V6AR2RunnerError("R2 model vocabulary does not cover its natural answer IDs")
    if any(contract["bare_token_id"] >= vocab_size for contract in symbol_contracts):
        raise V6AR2RunnerError("R2 model vocabulary does not cover its bare-token receipts")
    if any(
        candidate["natural_token_id"] >= vocab_size
        for candidate in candidate_receipt["eligible_candidates_in_hash_order"]
    ):
        raise V6AR2RunnerError("R2 model vocabulary does not cover its candidate universe")
    shard_specs = raw_shard_specs(vocab_size, result_root)
    disk_observation = disk_preflight(shard_specs, result_root)
    disk_requirements = {
        "stored_logits_dtype": disk_observation["stored_logits_dtype"],
        "raw_logits_expected_bytes": disk_observation["raw_logits_expected_bytes"],
        "reserve_bytes": disk_observation["reserve_bytes"],
        "required_disk_free_bytes": disk_observation["required_disk_free_bytes"],
        "planning_preflight_pass": disk_observation["pass"],
    }

    stage_prompt_ids = {
        stage: [prompt["prompt_id"] for prompt in prompts if prompt["execution_stage"] == stage]
        for stage in EXECUTION_STAGES
    }
    stage_plan_sha256 = {
        stage: canonical_sha256(prompt_ids) for stage, prompt_ids in stage_prompt_ids.items()
    }
    scientific_registry = {
        "cells": planned_cells,
        "prompts": prompts,
        "symbol_token_contracts": symbol_contracts,
        "topology_shape_receipts": topology_receipts,
        "candidate_universe_receipt": candidate_receipt,
        "historical_firewall_receipt": firewalls,
    }
    scientific_registry_sha256 = canonical_sha256(scientific_registry)
    model = {
        "model_id": MODEL_ID,
        "snapshot_revision": dependency["model"]["snapshot_revision"],
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": vocab_size,
        "device": DEVICE,
        "model_loading_device_map": MODEL_LOADING_DEVICE_MAP,
        "mps_allocator_warmup_policy": MPS_ALLOCATOR_WARMUP_POLICY,
        "model_dtype": MODEL_DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "model_loader_source": "sealed_v2_runner._load_model",
        "model_loader_source_sha256": SEALED_V2_RUNNER_SHA256,
    }
    plan_core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": fixture["analysis_id"],
        "registration_state": REGISTRATION_STATE,
        "execution_revision": EXECUTION_REVISION,
        "mode": fixture["mode"],
        "model": model,
        "chat_flags": CHAT_FLAGS,
        "natural_answer_surface_rule": "one ASCII space followed by one registered glyph",
        "bare_token_variants_registered": False,
        "bare_token_pooling_allowed": False,
        "response_site": {
            "assistant_prefill": "ANSWER:",
            "final_attended_text": ":",
            "final_attended_token_id": RESPONSE_SITE_TOKEN_ID,
        },
        "cells": planned_cells,
        "prompts": prompts,
        "symbol_token_contracts": symbol_contracts,
        "topology_shape_receipts": topology_receipts,
        "firewall_receipt": firewalls,
        "candidate_universe_receipt": candidate_receipt,
        "raw_logits_shards": shard_specs,
        "disk_requirements": disk_requirements,
        "stage_plan_sha256": stage_plan_sha256,
        "stage_prompt_ids": stage_prompt_ids,
        "block_order": list(BLOCK_ORDER),
        "block_counts": dict(BLOCK_COUNTS),
        "execution_stage_order": list(EXECUTION_STAGES),
        "execution_stage_counts": {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        },
        "remaining_output_blocks": REMAINING_BLOCK_LAYOUT,
        "family_counts": dict(FAMILY_COUNTS),
        "split_family_counts": SPLIT_FAMILY_COUNTS,
        "component_gates": COMPONENT_GATES,
        "analysis_access_firewall": {
            DISCOVERY_COMPONENTS_STAGE: (
                "admission analyzer may read only the first 640 discovery-component rows"
            ),
            REMAINING_MAIN_STAGE: (
                "one sealed 1664-call phase with three physical record blocks and no "
                "derived topology diagnostics"
            ),
            "final": (
                "opaque-hash all remaining paths; deserialize confirmation components "
                "and shards 8-17 first; on FAIL never open topology records or arrays"
            ),
        },
        "expected_calls": EXPECTED_CALLS,
        "model_calls_before_plan_freeze": 0,
        "planned_behavioral_model_calls": EXPECTED_CALLS,
        "generation_used": False,
        "logits_to_keep": 1,
        "stored_logits_dtype": "float32_little_endian",
        "partial_resume_allowed": False,
        "scientific_registry_sha256": scientific_registry_sha256,
        "dependency_canonical_sha256": dependency["canonical_sha256"],
        "v2_terminal_status": "V6A_QUALIFICATION_COMPONENT_FAIL",
        "v2_contextual_posthoc_role": "exploratory_engineering_evidence_only",
        "biological_model_calls": 0,
    }
    plan = {**plan_core, "call_plan_sha256": canonical_sha256(plan_core)}
    if (
        REGISTRATION_STATE == EXECUTION_ENABLED_REGISTRATION_STATE
        and result_root.resolve() == RESULT_ROOT.resolve()
    ):
        _validate_repair1_plan_preservation(plan)

    receipt_core = {
        "schema_version": TOKENIZATION_SCHEMA,
        "registration_state": REGISTRATION_STATE,
        "prompt_count": len(prompts),
        "model_calls": 0,
        "chat_template_sha256": text_sha256(tokenizer.chat_template),
        "chat_flags": CHAT_FLAGS,
        "response_site_token_ids": sorted(
            {int(prompt["response_site_token_id"]) for prompt in prompts}
        ),
        "response_site_indices": sorted(
            {int(prompt["response_site_index"]) for prompt in prompts}
        ),
        "natural_answer_token_ids": sorted(
            {int(contract["natural_token_id"]) for contract in symbol_contracts}
        ),
        "bare_answer_token_ids": sorted(
            {int(contract["bare_token_id"]) for contract in symbol_contracts}
        ),
        "bare_token_variants_registered": False,
        "bare_token_pooling_allowed": False,
        "symbol_token_contracts": symbol_contracts,
        "topology_shape_receipts": topology_receipts,
        "candidate_universe_receipt": candidate_receipt,
        "historical_firewall_receipt": firewalls,
        "input_token_count_min": min(prompt["input_token_count"] for prompt in prompts),
        "input_token_count_max": max(prompt["input_token_count"] for prompt in prompts),
        "stage_plan_sha256": stage_plan_sha256,
        "prompt_receipts": [
            {
                "global_call_index": prompt["global_call_index"],
                "stage_call_index": prompt["stage_call_index"],
                "execution_stage": prompt["execution_stage"],
                "execution_block": prompt["execution_block"],
                "record_identity_id": prompt["record_identity_id"],
                "cell_id": prompt["cell_id"],
                "prompt_id": prompt["prompt_id"],
                "rendered_text_sha256": prompt["rendered_text_sha256"],
                "input_token_count": prompt["input_token_count"],
                "response_site_index": prompt["response_site_index"],
                "response_site_token_id": prompt["response_site_token_id"],
                "expected_token_id": prompt["expected_token_id"],
                "distractor_token_id": prompt["distractor_token_id"],
                "bare_expected_token_id": prompt["bare_expected_token_id"],
                "bare_distractor_token_id": prompt["bare_distractor_token_id"],
            }
            for prompt in prompts
        ],
    }
    receipt = {**receipt_core, "canonical_sha256": canonical_sha256(receipt_core)}
    return plan, receipt


def _validate_canonical_envelope(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "canonical_sha256"}
    if value.get("canonical_sha256") != canonical_sha256(core):
        raise V6AR2RunnerError(f"{label} canonical hash changed")


def _design_payload(
    plan: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    integration_test_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": DESIGN_SCHEMA,
        "registration_state": REGISTRATION_STATE,
        "artifact_scope": "integration_test_only" if integration_test_only else "registered_main",
        "analysis_id": plan["analysis_id"],
        "mode": plan["mode"],
        "execution_revision": EXECUTION_REVISION,
        "model": plan["model"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "expected_calls": EXPECTED_CALLS,
        "family_counts": dict(FAMILY_COUNTS),
        "split_family_counts": SPLIT_FAMILY_COUNTS,
        "block_order": list(BLOCK_ORDER),
        "block_counts": dict(BLOCK_COUNTS),
        "execution_stage_order": list(EXECUTION_STAGES),
        "execution_stage_counts": {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        },
        "remaining_output_blocks": plan["remaining_output_blocks"],
        "candidate_universe_sha256": canonical_sha256(
            plan["candidate_universe_receipt"]
        ),
        "historical_firewall_source_registry_sha256": plan["firewall_receipt"][
            "historical_source_receipt"
        ]["source_registry_sha256"],
        "historical_firewall_pool_counts": plan["firewall_receipt"][
            "historical_pool_counts"
        ],
        "component_gates": COMPONENT_GATES,
        "natural_answer_contract": {
            "surface": "one ASCII space followed by one registered glyph",
            "one_token": True,
            "prompt_occurrence_id_equals_continuation_id": True,
            "bare_variants_registered": False,
            "bare_variant_pooling_allowed": False,
            "colon_response_site_token_id": RESPONSE_SITE_TOKEN_ID,
        },
        "staging_firewall": {
            "discovery_components_first": True,
            "discovery_component_calls": DISCOVERY_COMPONENT_CALLS,
            "stop_on_discovery_component_fail": True,
            "remaining_main_is_one_sealed_phase": True,
            "remaining_main_calls": REMAINING_MAIN_CALLS,
            "remaining_record_streams_physically_separate": True,
            "confirmation_component_raw_shards": list(range(8, 18)),
            "topology_records_store_diagnostics": False,
            "topology_files_opened_before_confirmation_pass": False,
            "effect_based_optional_stopping": False,
            "partial_resume_allowed": False,
        },
        "v2_provenance": {
            "terminal_status": "V6A_QUALIFICATION_COMPONENT_FAIL",
            "contextual_posthoc": "exploratory_engineering_evidence_only",
            "qualification_reopened": False,
            "authorization_inherited": False,
        },
        "claim_scope": "behavior_only_synthetic_prompt_topology_if_replicated",
        "forbidden_claims": [
            "activation_gap",
            "latent_knowledge",
            "biology",
            "physical_law",
            "universal_recency_mechanism",
            "model_family_generality",
        ],
        "locks": {
            "fixture_file_sha256": file_sha256(FIXTURE),
            "fixture_manifest_file_sha256": file_sha256(FIXTURE_MANIFEST),
            "fixture_canonical_sha256": fixture_manifest["fixture_canonical_sha256"],
            "dependency_lock_file_sha256": file_sha256(paths["dependency_lock"]),
            "tokenization_receipt_file_sha256": file_sha256(paths["tokenization_receipt"]),
            "design_document_file_sha256": file_sha256(DESIGN_DOCUMENT),
            "sealed_v2_runner_file_sha256": SEALED_V2_RUNNER_SHA256,
            "sealed_v2_terminal_analysis_file_sha256": V2_TERMINAL_ANALYSIS_SHA256,
            "sealed_v2_posthoc_output_file_sha256": V2_POSTHOC_OUTPUT_SHA256,
        },
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }


def _plan_manifest_payload(
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    integration_test_only: bool,
) -> dict[str, Any]:
    if integration_test_only:
        status = "R2_PLAN_FROZEN_ZERO_FORWARD"
    else:
        status = "R2_PLAN_FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
    return {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": status,
        "registration_state": REGISTRATION_STATE,
        "artifact_scope": "integration_test_only" if integration_test_only else "registered_main",
        "call_plan_sha256": plan["call_plan_sha256"],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"],
        "plan": dict(plan),
        "design_file_sha256": file_sha256(paths["design"]),
        "dependency_lock_file_sha256": file_sha256(paths["dependency_lock"]),
        "tokenization_receipt_file_sha256": file_sha256(paths["tokenization_receipt"]),
        "fixture_file_sha256": file_sha256(FIXTURE),
        "fixture_manifest_file_sha256": file_sha256(FIXTURE_MANIFEST),
        "model_calls": 0,
        "generation_used": False,
        "expected_calls": EXPECTED_CALLS,
        "partial_resume_allowed": False,
    }


def run_plan(
    *,
    result_root: Path = RESULT_ROOT,
    integration_test_only: bool = False,
    fixture_bundle: tuple[dict[str, Any], dict[str, Any]] | None = None,
    dependency_override: dict[str, Any] | None = None,
    tokenizer_override: Any | None = None,
) -> dict[str, Any]:
    """Freeze the registered plan once, before any R2 model load or forward."""

    resolved_root = result_root.resolve()
    if integration_test_only:
        if REGISTRATION_STATE != "DRAFT_ZERO_FORWARD":
            raise V6AR2RunnerError(
                "integration-test planning is allowed only in DRAFT_ZERO_FORWARD"
            )
        if resolved_root == RESULT_ROOT.resolve():
            raise V6AR2RunnerError(
                "integration-test planning cannot target the real R2 result root"
            )
    else:
        if REGISTRATION_STATE != EXECUTION_ENABLED_REGISTRATION_STATE:
            raise V6AR2RunnerError(
                "real R2 planning requires exact registration state "
                "FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
            )
        if resolved_root != RESULT_ROOT.resolve():
            raise V6AR2RunnerError("registered R2 planning must target the exact result root")
        if any(
            value is not None
            for value in (fixture_bundle, dependency_override, tokenizer_override)
        ):
            raise V6AR2RunnerError("registered R2 planning accepts no test overrides")
    paths = artifact_paths(resolved_root)
    if any(
        path.exists()
        for path in [
            *_planning_paths(paths),
            *_discovery_paths(paths),
            *_remaining_paths(paths),
        ]
    ):
        raise V6AR2RunnerError("R2 plan root is not one-shot clean")

    fixture, fixture_manifest = fixture_bundle or load_and_rebuild_fixture()
    dependency = dependency_override or dependency_lock(DEFAULT_ANALYZER)
    _validate_canonical_envelope(dependency, "R2 dependency lock")
    tokenizer = tokenizer_override or load_tokenizer_from_sealed_v2(dependency)
    plan, receipt = build_plan(tokenizer, fixture, dependency, result_root=resolved_root)

    _atomic_write(paths["dependency_lock"], _artifact_bytes(dependency))
    _atomic_write(paths["tokenization_receipt"], _artifact_bytes(receipt))
    design = _design_payload(
        plan,
        fixture_manifest,
        paths,
        integration_test_only=integration_test_only,
    )
    _atomic_write(paths["design"], _artifact_bytes(design))
    manifest = _plan_manifest_payload(
        plan,
        paths,
        integration_test_only=integration_test_only,
    )
    _atomic_write(paths["plan_manifest"], _artifact_bytes(manifest))
    return manifest


def validate_frozen_plan(
    result_root: Path = RESULT_ROOT,
    *,
    dependency_replay_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = artifact_paths(result_root)
    design = _load_json(paths["design"])
    manifest = _load_json(paths["plan_manifest"])
    receipt = _load_json(paths["tokenization_receipt"])
    dependency = _load_json(paths["dependency_lock"])
    _validate_canonical_envelope(receipt, "R2 tokenization receipt")
    _validate_canonical_envelope(dependency, "R2 dependency lock")
    integration_test_only = manifest.get("artifact_scope") == "integration_test_only"
    expected_status = (
        "R2_PLAN_FROZEN_ZERO_FORWARD"
        if integration_test_only
        else "R2_PLAN_FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
    )
    _replay_dependency_lock(
        dependency,
        integration_test_only=integration_test_only,
        dependency_replay_override=dependency_replay_override,
    )
    if (
        manifest.get("status") != expected_status
        or manifest.get("registration_state") != REGISTRATION_STATE
        or design.get("registration_state") != REGISTRATION_STATE
        or manifest.get("model_calls") != 0
        or design.get("model_calls") != 0
    ):
        raise V6AR2RunnerError("R2 frozen zero-forward status changed")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise V6AR2RunnerError("R2 frozen plan is missing")
    plan_core = {key: value for key, value in plan.items() if key != "call_plan_sha256"}
    if canonical_sha256(plan_core) != plan.get("call_plan_sha256"):
        raise V6AR2RunnerError("R2 call-plan hash changed")
    if (
        plan.get("expected_calls") != EXPECTED_CALLS
        or plan.get("registration_state") != REGISTRATION_STATE
        or plan.get("execution_revision") != EXECUTION_REVISION
        or len(plan.get("prompts", [])) != EXPECTED_CALLS
        or len(plan.get("cells", [])) != EXPECTED_CALLS
        or plan.get("dependency_canonical_sha256") != dependency.get("canonical_sha256")
        or plan.get("execution_stage_counts")
        != {
            DISCOVERY_COMPONENTS_STAGE: DISCOVERY_COMPONENT_CALLS,
            REMAINING_MAIN_STAGE: REMAINING_MAIN_CALLS,
        }
    ):
        raise V6AR2RunnerError("R2 frozen plan scope changed")
    if any(
        prompt.get("record_identity_id") != record_identity_id(prompt)
        or cell.get("record_identity_id") != prompt.get("record_identity_id")
        for cell, prompt in zip(plan["cells"], plan["prompts"], strict=True)
    ) or len(
        {prompt["record_identity_id"] for prompt in plan["prompts"]}
    ) != EXPECTED_CALLS:
        raise V6AR2RunnerError("R2 frozen pre-forward record identities changed")
    file_bindings = {
        paths["design"]: manifest.get("design_file_sha256"),
        paths["dependency_lock"]: manifest.get("dependency_lock_file_sha256"),
        paths["tokenization_receipt"]: manifest.get("tokenization_receipt_file_sha256"),
        FIXTURE: manifest.get("fixture_file_sha256"),
        FIXTURE_MANIFEST: manifest.get("fixture_manifest_file_sha256"),
    }
    for path, expected in file_bindings.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise V6AR2RunnerError(f"R2 frozen artifact binding changed: {path}")
    if (
        design.get("call_plan_sha256") != plan["call_plan_sha256"]
        or design.get("scientific_registry_sha256") != plan["scientific_registry_sha256"]
        or manifest.get("stage_plan_sha256") != plan["stage_plan_sha256"]
        or receipt.get("stage_plan_sha256") != plan["stage_plan_sha256"]
    ):
        raise V6AR2RunnerError("R2 design/receipt/plan binding changed")
    fixture = _load_json(FIXTURE)
    historical_pools, historical_source_receipt = _historical_firewall_registry()
    if dependency.get("historical_firewall_sources") != historical_source_receipt:
        raise V6AR2RunnerError("R2 dependency historical source replay changed")
    candidate_receipt = plan.get("candidate_universe_receipt")
    if not isinstance(candidate_receipt, Mapping):
        raise V6AR2RunnerError("R2 frozen candidate-universe receipt is missing")
    _validate_candidate_receipt_static(candidate_receipt)
    symbol_contracts = _validate_symbol_token_contracts(plan["prompts"], fixture)
    replayed_firewall = _validate_plan_firewalls(
        plan["cells"],
        plan["prompts"],
        fixture,
        symbol_contracts,
        historical_pools,
        historical_source_receipt,
        candidate_receipt,
    )
    if (
        replayed_firewall != plan.get("firewall_receipt")
        or receipt.get("historical_firewall_receipt") != replayed_firewall
        or receipt.get("candidate_universe_receipt") != candidate_receipt
    ):
        raise V6AR2RunnerError("R2 historical/candidate firewall receipt changed")
    _validate_topology_shapes(plan["prompts"])
    if symbol_contracts != plan.get("symbol_token_contracts"):
        raise V6AR2RunnerError("R2 frozen natural-token contract registry changed")
    expected_shards = raw_shard_specs(plan["model"]["vocab_size"], result_root)
    if plan.get("raw_logits_shards") != expected_shards:
        raise V6AR2RunnerError("R2 frozen full-vocabulary shard plan changed")
    fixture_manifest = _load_json(FIXTURE_MANIFEST)
    if design != _design_payload(
        plan,
        fixture_manifest,
        paths,
        integration_test_only=integration_test_only,
    ):
        raise V6AR2RunnerError("R2 design artifact does not replay exactly")
    if manifest != _plan_manifest_payload(
        plan,
        paths,
        integration_test_only=integration_test_only,
    ):
        raise V6AR2RunnerError("R2 plan manifest does not replay exactly")
    return plan, design, dependency, receipt


def _stage_contract(stage: str) -> dict[str, Any]:
    if stage == DISCOVERY_COMPONENTS_STAGE:
        return {
            "attempt_status": DISCOVERY_ATTEMPT_STATUS,
            "complete_status": DISCOVERY_COMPLETE_STATUS,
            "expected_calls": DISCOVERY_COMPONENT_CALLS,
            "global_start_row": 0,
            "global_stop_row": DISCOVERY_COMPONENT_CALLS,
            "stage_start_row": 0,
            "stage_stop_row": DISCOVERY_COMPONENT_CALLS,
            "block_order": ["discovery-components"],
            "block_counts": {"discovery-components": DISCOVERY_COMPONENT_CALLS},
            "cumulative_model_calls": DISCOVERY_COMPONENT_CALLS,
        }
    if stage == REMAINING_MAIN_STAGE:
        return {
            "attempt_status": REMAINING_ATTEMPT_STATUS,
            "complete_status": REMAINING_COMPLETE_STATUS,
            "expected_calls": REMAINING_MAIN_CALLS,
            "global_start_row": DISCOVERY_COMPONENT_CALLS,
            "global_stop_row": EXPECTED_CALLS,
            "stage_start_row": 0,
            "stage_stop_row": REMAINING_MAIN_CALLS,
            "block_order": list(REMAINING_BLOCK_LAYOUT),
            "block_counts": {
                block: int(layout["call_count"])
                for block, layout in REMAINING_BLOCK_LAYOUT.items()
            },
            "cumulative_model_calls": EXPECTED_CALLS,
        }
    raise V6AR2RunnerError(f"unknown R2 execution stage: {stage}")


def forward_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    vocab_size = plan.get("model", {}).get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise V6AR2RunnerError("R2 forward contract has an invalid vocabulary size")
    return {
        "torch_inference_mode": True,
        "use_cache": False,
        "logits_to_keep": 1,
        "return_dict": True,
        "teacher_forced_prompt_forward": True,
        "retained_logits_expression": "output.logits[0,-1,:]",
        "retained_logits_dtype": "float32_little_endian",
        "retained_logits_shape": [vocab_size],
    }


def full_vocab_diagnostics(
    row: np.ndarray,
    expected_token_id: int,
    distractor_token_id: int,
) -> dict[str, Any]:
    """Compute the component-only diagnostics independently replayed by analysis."""

    values = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise V6AR2RunnerError("R2 full-vocabulary logit row is invalid")
    if (
        isinstance(expected_token_id, bool)
        or not isinstance(expected_token_id, int)
        or isinstance(distractor_token_id, bool)
        or not isinstance(distractor_token_id, int)
        or expected_token_id == distractor_token_id
        or not 0 <= expected_token_id < len(values)
        or not 0 <= distractor_token_id < len(values)
    ):
        raise V6AR2RunnerError("R2 answer-token IDs are invalid")
    expected = float(values[expected_token_id])
    distractor = float(values[distractor_token_id])
    maximum = float(values.max())
    maximum_ids = [int(index) for index in np.flatnonzero(values == maximum)]
    return {
        "expected_logit": expected,
        "distractor_logit": distractor,
        "expected_minus_distractor_margin": expected - distractor,
        "pairwise_correct": expected > distractor,
        "pairwise_tie": expected == distractor,
        "strict_unique_global_correct": maximum_ids == [expected_token_id],
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": f32_sha256(values),
    }


class RawShardWriter:
    """One-shot writer for an already frozen stage shard registry."""

    def __init__(self, specs: Sequence[Mapping[str, Any]]) -> None:
        self.specs = [dict(spec) for spec in specs]
        self.buffer: list[np.ndarray] = []
        self.written: list[dict[str, Any]] = []
        self.next_shard = 0
        self.rows_seen = 0
        if not self.specs:
            raise V6AR2RunnerError("R2 raw-logit shard registry is empty")

    def append(self, row: np.ndarray) -> tuple[int, int]:
        if self.next_shard >= len(self.specs):
            raise V6AR2RunnerError("R2 raw-logit writer received excess rows")
        spec = self.specs[self.next_shard]
        value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
        if list(value.shape) != [int(spec["shape"][1])] or not np.isfinite(value).all():
            raise V6AR2RunnerError("model returned invalid R2 full-vocabulary logits")
        shard_index = int(spec["index"])
        row_in_shard = len(self.buffer)
        self.buffer.append(value)
        self.rows_seen += 1
        if len(self.buffer) == int(spec["rows"]):
            self._flush()
        return shard_index, row_in_shard

    def _flush(self) -> None:
        spec = self.specs[self.next_shard]
        matrix = np.ascontiguousarray(np.stack(self.buffer), dtype="<f4")
        if list(matrix.shape) != list(spec["shape"]):
            raise V6AR2RunnerError("R2 raw-logit shard shape changed")
        path = Path(str(spec["path"]))
        _atomic_save_npy(path, matrix)
        self.written.append(
            {
                **spec,
                "file_sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "logical_sha256": f32_sha256(matrix),
            }
        )
        self.buffer = []
        self.next_shard += 1

    def finish(self) -> list[dict[str, Any]]:
        if self.buffer:
            self._flush()
        expected_rows = sum(int(spec["rows"]) for spec in self.specs)
        if (
            self.next_shard != len(self.specs)
            or self.rows_seen != expected_rows
            or len(self.written) != len(self.specs)
        ):
            raise V6AR2RunnerError("R2 raw-logit stage is incomplete")
        return self.written


def _forward_one(
    model: Any,
    prompt: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
) -> np.ndarray:
    """Issue exactly one sealed teacher-forced forward at the colon site."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise V6AR2RunnerError("torch is required for R2 execution") from error
    if plan.get("execution_revision") != EXECUTION_REVISION:
        raise V6AR2RunnerError("R2 execution revision changed before forward")
    if plan.get("logits_to_keep") != 1:
        raise V6AR2RunnerError("R2 must retain exactly one final-position logit row")
    ids = torch_module.tensor(
        [prompt["execution_input_ids"]],
        dtype=torch_module.long,
        device=DEVICE,
    )
    mask = torch_module.tensor(
        [prompt["execution_attention_mask"]],
        dtype=torch_module.long,
        device=DEVICE,
    )
    with torch_module.inference_mode():
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    try:
        logits = output.logits[0, -1, :]
        if hasattr(logits, "detach"):
            logits = logits.detach()
        if hasattr(logits, "float"):
            logits = logits.float()
        if hasattr(logits, "cpu"):
            logits = logits.cpu()
        if hasattr(logits, "numpy"):
            logits = logits.numpy()
        value = np.ascontiguousarray(np.asarray(logits, dtype="<f4"))
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise V6AR2RunnerError("R2 model output does not expose final-site logits") from error
    if (
        list(value.shape) != [int(plan["model"]["vocab_size"])]
        or not np.isfinite(value).all()
    ):
        raise V6AR2RunnerError("R2 model returned invalid final-site logits")
    return value


def execution_preflight(
    plan: Mapping[str, Any],
    dependency: Mapping[str, Any],
    *,
    stage: str,
    result_root: Path,
    reserve_bytes: int = 1024**3,
) -> dict[str, Any]:
    """Run environment checks without loading or forwarding the registered model."""

    try:
        import torch
    except ImportError as error:
        raise V6AR2RunnerError("torch is required for R2 execution preflight") from error
    contract = _stage_contract(stage)
    specs = plan.get("raw_logits_shards", {}).get(stage)
    if not isinstance(specs, list) or len(specs) != (10 if stage == DISCOVERY_COMPONENTS_STAGE else 26):
        raise V6AR2RunnerError("R2 stage raw-logit preflight registry changed")
    stage_bytes = sum(math.prod(spec["shape"]) * 4 for spec in specs)
    all_specs = [
        spec
        for registered_stage in EXECUTION_STAGES
        for spec in plan["raw_logits_shards"][registered_stage]
    ]
    total_bytes = sum(math.prod(spec["shape"]) * 4 for spec in all_specs)
    if (
        any(spec.get("dtype") != "<f4" for spec in all_specs)
        or total_bytes != plan.get("disk_requirements", {}).get("raw_logits_expected_bytes")
    ):
        raise V6AR2RunnerError("R2 full-vocabulary disk registry changed")
    assets = dependency.get("model", {}).get("assets", {})
    weight_bytes = sum(
        int(binding["size_bytes"])
        for name, binding in assets.items()
        if str(name).endswith(".safetensors")
    )
    if weight_bytes <= 0:
        raise V6AR2RunnerError("R2 model weight-byte registry is empty")
    if not torch.backends.mps.is_available():
        raise V6AR2RunnerError("registered MPS runtime is unavailable")
    recommended = int(torch.mps.recommended_max_memory())
    allocated = int(torch.mps.current_allocated_memory())
    required_headroom = math.ceil(weight_bytes * 1.10)
    if recommended - allocated < required_headroom:
        raise V6AR2RunnerError("insufficient MPS headroom for the sealed R2 model")
    tensor = torch.ones((2, 2), dtype=torch.bfloat16, device=DEVICE)
    kernel_result = (tensor @ tensor).float().cpu().tolist()
    del tensor
    torch.mps.synchronize()
    torch.mps.empty_cache()
    if kernel_result != [[2.0, 2.0], [2.0, 2.0]]:
        raise V6AR2RunnerError("R2 MPS bfloat16 kernel preflight failed")
    probe = result_root.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk_free = int(shutil.disk_usage(probe).free)
    required_disk = stage_bytes + reserve_bytes
    if disk_free < required_disk:
        raise V6AR2RunnerError("insufficient disk space for the registered R2 stage")
    return {
        "execution_stage": stage,
        "planned_calls": contract["expected_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "stage_raw_logits_expected_bytes": stage_bytes,
        "total_raw_logits_expected_bytes": total_bytes,
        "model_safetensor_bytes": weight_bytes,
        "mps_recommended_max_memory_bytes": recommended,
        "mps_allocated_before_bytes": allocated,
        "required_mps_headroom_bytes": required_headroom,
        "mps_bfloat16_kernel_pass": True,
        "disk_free_bytes": disk_free,
        "required_disk_free_bytes": required_disk,
        "stored_logits_dtype": "float32_little_endian",
        "no_model_loaded": True,
        "no_model_forward_performed": True,
        "model_calls": 0,
        "generation_calls": 0,
        "pass": True,
    }


def _source_locks(paths: Mapping[str, Path]) -> dict[str, str]:
    """Bind every source needed to interpret a behavioral stage."""

    return {
        "plan_manifest_file_sha256": file_sha256(paths["plan_manifest"]),
        "design_file_sha256": file_sha256(paths["design"]),
        "dependency_lock_file_sha256": file_sha256(paths["dependency_lock"]),
        "tokenization_receipt_file_sha256": file_sha256(
            paths["tokenization_receipt"]
        ),
        "fixture_file_sha256": file_sha256(FIXTURE),
        "fixture_manifest_file_sha256": file_sha256(FIXTURE_MANIFEST),
        "runner_file_sha256": file_sha256(Path(__file__).resolve()),
        "analyzer_file_sha256": file_sha256(DEFAULT_ANALYZER),
        "design_document_file_sha256": file_sha256(DESIGN_DOCUMENT),
        "sealed_v2_runner_file_sha256": file_sha256(SEALED_V2_RUNNER),
    }


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise V6AR2RunnerError(f"R2 artifact binding target is invalid: {path}")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _prior_discovery_authorization_binding(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    execution = _load_json(paths["discovery_execution_manifest"])
    authorization = _load_json(paths["discovery_analysis"])
    return {
        "execution_manifest_file_sha256": file_sha256(
            paths["discovery_execution_manifest"]
        ),
        "execution_manifest_canonical_sha256": canonical_sha256(execution),
        "authorization_file_sha256": file_sha256(paths["discovery_analysis"]),
        "authorization_canonical_sha256": canonical_sha256(authorization),
        "authorization_status": authorization.get("status"),
    }


def _prior_discovery_authorization_from_replay(
    replayed: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if (
        not isinstance(replayed, Mapping)
        or set(replayed) != DISCOVERY_AUTHORIZATION_REPLAY_KEYS
    ):
        raise V6AR2RunnerError(
            "remaining execution received a malformed authorization replay"
        )
    prior = {
        "execution_manifest_file_sha256": replayed.get(
            "discovery_execution_manifest_file_sha256"
        ),
        "execution_manifest_canonical_sha256": replayed.get(
            "discovery_execution_manifest_canonical_sha256"
        ),
        "authorization_file_sha256": replayed.get("authorization_file_sha256"),
        "authorization_canonical_sha256": replayed.get(
            "authorization_canonical_sha256"
        ),
        "authorization_status": replayed.get("authorization_status"),
    }
    if (
        prior != _prior_discovery_authorization_binding(paths)
        or prior.get("authorization_status") != DISCOVERY_AUTHORIZED_STATUS
        or replayed.get("replay_equal") is not True
        or replayed.get("model_calls_issued_by_validator") != 0
        or replayed.get("generation_calls_issued_by_validator") != 0
    ):
        raise V6AR2RunnerError(
            "remaining execution lost its passing authorization"
        )
    return prior


def _attempt_payload(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    stage: str,
    prior_discovery_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = _stage_contract(stage)
    payload: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": contract["attempt_status"],
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "execution_stage": stage,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "expected_calls": contract["expected_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "stage_start_row": contract["stage_start_row"],
        "stage_stop_row": contract["stage_stop_row"],
        "block_order": contract["block_order"],
        "block_counts": contract["block_counts"],
        "partial_resume_allowed": False,
        "generation_used": False,
        "model": plan["model"],
        "forward_contract": forward_contract(plan),
        "preflight": dict(preflight),
        "source_locks": _source_locks(paths),
    }
    if stage == REMAINING_MAIN_STAGE:
        if prior_discovery_authorization is None:
            raise V6AR2RunnerError(
                "remaining attempt lacks its replayed discovery authorization"
            )
        payload["prior_discovery_authorization"] = dict(
            prior_discovery_authorization
        )
    elif prior_discovery_authorization is not None:
        raise V6AR2RunnerError(
            "discovery attempt unexpectedly received a prior authorization"
        )
    return payload


def _execution_record(
    prompt: Mapping[str, Any],
    row: np.ndarray,
    plan: Mapping[str, Any],
    *,
    stage: str,
    shard_index: int,
    row_in_shard: int,
) -> dict[str, Any]:
    stage_index = int(prompt["stage_call_index"])
    expected_identity = record_identity_id(prompt)
    if prompt.get("record_identity_id") != expected_identity:
        raise V6AR2RunnerError("R2 planned record identity changed before execution")
    core: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "record_identity_id": expected_identity,
        "execution_stage": stage,
        "execution_block": prompt["execution_block"],
        "global_call_index": prompt["global_call_index"],
        "block_call_index": prompt["block_call_index"],
        "stage_call_index": stage_index,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "cell_id": prompt["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "world_id": prompt["world_id"],
        "role": prompt["role"],
        "family": prompt["family"],
        "factors": prompt["factors"],
        "factor_levels": prompt["factor_levels"],
        "expected_token_id": prompt["expected_token_id"],
        "distractor_token_id": prompt["distractor_token_id"],
        "raw_logits_shard_index": shard_index,
        "raw_logits_row_in_shard": row_in_shard,
        "raw_logits_stage_row": stage_index,
        "raw_logits_global_row": prompt["global_call_index"],
        "raw_logits_row_sha256": f32_sha256(row),
        "teacher_forced_prompt_forward": True,
        "generation_used": False,
        "model_calls": 1,
    }
    if prompt["family"] == "two_fact_composition":
        if prompt["execution_block"] not in {
            "discovery-topology",
            "confirmation-topology",
        }:
            raise V6AR2RunnerError("two-fact row escaped the topology blocks")
    else:
        if prompt["execution_block"] not in {
            "discovery-components",
            "confirmation-components",
        }:
            raise V6AR2RunnerError("component row escaped the component blocks")
        core["diagnostics"] = full_vocab_diagnostics(
            row,
            int(prompt["expected_token_id"]),
            int(prompt["distractor_token_id"]),
        )
    return {**core, "record_id": canonical_sha256(core)}


def _record_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in records
    )


def _record_binding(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **_file_binding(path),
        "count": len(records),
        "canonical_sha256": canonical_sha256(list(records)),
    }


def _remaining_record_block_binding(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    block: str,
) -> dict[str, Any]:
    layout = REMAINING_BLOCK_LAYOUT[block]
    return {
        **_file_binding(path),
        "execution_block": block,
        "row_count": len(records),
        "global_start_row": layout["global_start_row"],
        "global_stop_row": layout["global_stop_row"],
        "stage_start_row": layout["stage_start_row"],
        "stage_stop_row": layout["stage_stop_row"],
        "raw_shard_indices": list(layout["raw_shard_indices"]),
        "canonical_sha256": canonical_sha256(list(records)),
    }


def _execute_stage(
    plan: Mapping[str, Any],
    dependency: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    *,
    stage: str,
    result_root: Path,
    model_loader: Any,
    torch_module: Any | None = None,
    prior_discovery_authorization: Mapping[str, Any] | None = None,
    progress: Any = print,
) -> dict[str, Any]:
    """Execute one already-authorized stage; tests inject only private seams."""

    paths = artifact_paths(result_root)
    contract = _stage_contract(stage)
    rows = [dict(prompt) for prompt in prompts]
    if (
        len(rows) != contract["expected_calls"]
        or [row.get("stage_call_index") for row in rows]
        != list(range(contract["expected_calls"]))
        or any(row.get("execution_stage") != stage for row in rows)
        or preflight.get("execution_stage") != stage
        or preflight.get("pass") is not True
    ):
        raise V6AR2RunnerError("R2 authorized stage execution registry changed")
    guarded_paths = (
        _discovery_paths(paths)
        if stage == DISCOVERY_COMPONENTS_STAGE
        else _remaining_paths(paths)
    )
    if any(path.exists() for path in guarded_paths):
        raise V6AR2RunnerError("R2 stage execution is not one-shot clean")
    attempt_path = (
        paths["discovery_attempt"]
        if stage == DISCOVERY_COMPONENTS_STAGE
        else paths["remaining_attempt"]
    )
    execution_path = (
        paths["discovery_execution_manifest"]
        if stage == DISCOVERY_COMPONENTS_STAGE
        else paths["remaining_execution_manifest"]
    )
    attempt = _attempt_payload(
        plan,
        preflight,
        paths,
        stage=stage,
        prior_discovery_authorization=prior_discovery_authorization,
    )
    _atomic_write(attempt_path, _artifact_bytes(attempt))

    # The immutable attempt above must exist before this exact sealed loader runs.
    model = model_loader(
        Path(str(dependency["model"]["snapshot_path"])),
        plan,
        dependency,
    )
    writer = RawShardWriter(plan["raw_logits_shards"][stage])
    records_by_block: dict[str, list[dict[str, Any]]] = {
        block: [] for block in contract["block_order"]
    }
    for index, prompt in enumerate(rows):
        logits = _forward_one(
            model,
            prompt,
            plan,
            torch_module=torch_module,
        )
        shard_index, row_in_shard = writer.append(logits)
        record = _execution_record(
            prompt,
            logits,
            plan,
            stage=stage,
            shard_index=shard_index,
            row_in_shard=row_in_shard,
        )
        records_by_block[str(prompt["execution_block"])].append(record)
        if (index + 1) % 32 == 0 or index + 1 == len(rows):
            progress(
                json.dumps(
                    {
                        "execution_stage": stage,
                        "completed": index + 1,
                        "total": len(rows),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    shards = writer.finish()
    if {block: len(values) for block, values in records_by_block.items()} != contract[
        "block_counts"
    ]:
        raise V6AR2RunnerError("R2 executed block counts changed")

    source_locks = _source_locks(paths)
    if source_locks != attempt["source_locks"]:
        raise V6AR2RunnerError("R2 bound sources changed during execution")
    manifest: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": contract["complete_status"],
        "registration_state": plan["registration_state"],
        "execution_revision": EXECUTION_REVISION,
        "execution_stage": stage,
        "call_plan_sha256": plan["call_plan_sha256"],
        "stage_plan_sha256": plan["stage_plan_sha256"][stage],
        "scientific_registry_sha256": plan["scientific_registry_sha256"],
        "phase_model_calls": contract["expected_calls"],
        "cumulative_model_calls": contract["cumulative_model_calls"],
        "global_start_row": contract["global_start_row"],
        "global_stop_row": contract["global_stop_row"],
        "block_order": contract["block_order"],
        "block_counts": contract["block_counts"],
        "generation_used": False,
        "partial_resume_allowed": False,
        "raw_logits_shards": shards,
        "attempt": _file_binding(attempt_path),
        "source_locks": source_locks,
    }
    if stage == DISCOVERY_COMPONENTS_STAGE:
        records = records_by_block["discovery-components"]
        _atomic_write(paths["discovery_records"], _record_bytes(records))
        manifest["records"] = _record_binding(paths["discovery_records"], records)
    else:
        if prior_discovery_authorization is None:
            raise V6AR2RunnerError("remaining execution lost prior authorization")
        block_bindings: dict[str, Any] = {}
        for block, layout in REMAINING_BLOCK_LAYOUT.items():
            block_records = records_by_block[block]
            record_path = paths[str(layout["record_artifact_key"])]
            _atomic_write(record_path, _record_bytes(block_records))
            block_bindings[block] = _remaining_record_block_binding(
                record_path,
                block_records,
                block=block,
            )
        manifest["record_blocks"] = block_bindings
        manifest["prior_discovery_authorization"] = dict(
            prior_discovery_authorization
        )
    _atomic_write(execution_path, _artifact_bytes(manifest))
    return manifest


def _require_non_draft_behavior(*, allow_draft_test: bool) -> None:
    if allow_draft_test:
        if REGISTRATION_STATE == "DRAFT_ZERO_FORWARD":
            return
    elif REGISTRATION_STATE == EXECUTION_ENABLED_REGISTRATION_STATE:
        return
    raise V6AR2BehavioralExecutionDisabled(
        "R2 behavioral execution requires exact registration state "
        "FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
    )


def validate_discovery_stage_authorization(
    plan: Mapping[str, Any],
    *,
    result_root: Path = RESULT_ROOT,
    allow_draft_test: bool = False,
    dependency_replay_override: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _require_non_draft_behavior(allow_draft_test=allow_draft_test)
    paths = artifact_paths(result_root)
    frozen_plan, _, _, _ = validate_frozen_plan(
        result_root,
        dependency_replay_override=dependency_replay_override,
    )
    if dict(plan) != frozen_plan:
        raise V6AR2RunnerError("R2 discovery authorization plan differs from frozen plan")
    if any(not path.is_file() for path in _planning_paths(paths)):
        raise V6AR2RunnerError("R2 planning artifacts are incomplete")
    if any(path.exists() for path in [*_discovery_paths(paths), *_remaining_paths(paths)]):
        raise V6AR2RunnerError("R2 discovery stage is not one-shot clean")
    prompts = [
        dict(prompt)
        for prompt in plan.get("prompts", [])
        if prompt.get("execution_stage") == DISCOVERY_COMPONENTS_STAGE
    ]
    if (
        len(prompts) != DISCOVERY_COMPONENT_CALLS
        or [prompt.get("global_call_index") for prompt in prompts] != list(range(640))
        or canonical_sha256([prompt["prompt_id"] for prompt in prompts])
        != plan.get("stage_plan_sha256", {}).get(DISCOVERY_COMPONENTS_STAGE)
    ):
        raise V6AR2RunnerError("R2 discovery-component stage registry changed")
    return prompts


def validate_remaining_stage_authorization(
    plan: Mapping[str, Any],
    *,
    result_root: Path = RESULT_ROOT,
    allow_draft_test: bool = False,
    dependency_replay_override: Mapping[str, Any] | None = None,
    authorization_replay_override: Any | None = None,
) -> list[dict[str, Any]]:
    _require_non_draft_behavior(allow_draft_test=allow_draft_test)
    if authorization_replay_override is not None and not (
        allow_draft_test and REGISTRATION_STATE == "DRAFT_ZERO_FORWARD"
    ):
        raise V6AR2RunnerError(
            "registered remaining authorization replay cannot be overridden"
        )
    paths = artifact_paths(result_root)
    frozen_plan, _, _, _ = validate_frozen_plan(
        result_root,
        dependency_replay_override=dependency_replay_override,
    )
    if dict(plan) != frozen_plan:
        raise V6AR2RunnerError("R2 remaining authorization plan differs from frozen plan")
    required_discovery = [
        paths["discovery_attempt"],
        paths["discovery_records"],
        paths["discovery_execution_manifest"],
        paths["discovery_analysis"],
        paths["discovery_raw_root"],
    ]
    if any(not path.exists() for path in required_discovery):
        raise V6AR2RunnerError("R2 remaining-main stage lacks complete discovery artifacts")
    if any(path.exists() for path in _remaining_paths(paths)):
        raise V6AR2RunnerError("R2 remaining-main stage is not one-shot clean")
    execution = _load_json(paths["discovery_execution_manifest"])
    admission = _load_json(paths["discovery_analysis"])
    if (
        execution.get("status") != DISCOVERY_COMPLETE_STATUS
        or execution.get("phase_model_calls") != DISCOVERY_COMPONENT_CALLS
        or execution.get("call_plan_sha256") != plan.get("call_plan_sha256")
        or execution.get("stage_plan_sha256")
        != plan.get("stage_plan_sha256", {}).get(DISCOVERY_COMPONENTS_STAGE)
    ):
        raise V6AR2RunnerError("R2 discovery execution authorization binding changed")
    if (
        admission.get("status") != DISCOVERY_AUTHORIZED_STATUS
        or admission.get("engineering_valid") is not True
        or admission.get("component_qualified") is not True
        or admission.get("authorization_issued") is not True
        or admission.get("call_plan_sha256") != plan.get("call_plan_sha256")
        or admission.get("stage_plan_sha256")
        != plan.get("stage_plan_sha256", {}).get(DISCOVERY_COMPONENTS_STAGE)
    ):
        raise V6AR2RunnerError("R2 discovery component admission did not authorize remaining-main")
    prompts = [
        dict(prompt)
        for prompt in plan.get("prompts", [])
        if prompt.get("execution_stage") == REMAINING_MAIN_STAGE
    ]
    if (
        len(prompts) != REMAINING_MAIN_CALLS
        or [prompt.get("global_call_index") for prompt in prompts] != list(range(640, 2304))
        or canonical_sha256([prompt["prompt_id"] for prompt in prompts])
        != plan.get("stage_plan_sha256", {}).get(REMAINING_MAIN_STAGE)
    ):
        raise V6AR2RunnerError("R2 remaining-main stage registry changed")
    replay = (
        authorization_replay_override
        if authorization_replay_override is not None
        else _analyzer_module().validate_remaining_authorization_for_runner
    )
    try:
        replayed = replay(
            result_root,
            dependency_replay_override=dependency_replay_override,
        )
    except Exception as error:
        raise V6AR2RunnerError(
            "R2 discovery authorization did not independently replay"
        ) from error
    if (
        not isinstance(replayed, Mapping)
        or set(replayed) != DISCOVERY_AUTHORIZATION_REPLAY_KEYS
    ):
        raise V6AR2RunnerError("R2 discovery authorization replay receipt changed")
    if (
        replayed.get("authorization") != admission
        or replayed.get("authorization_file_sha256")
        != file_sha256(paths["discovery_analysis"])
        or replayed.get("authorization_canonical_sha256")
        != canonical_sha256(admission)
        or replayed.get("discovery_execution_manifest_file_sha256")
        != file_sha256(paths["discovery_execution_manifest"])
        or replayed.get("discovery_execution_manifest_canonical_sha256")
        != canonical_sha256(execution)
        or replayed.get("authorization_status") != DISCOVERY_AUTHORIZED_STATUS
        or replayed.get("call_plan_sha256") != plan["call_plan_sha256"]
        or replayed.get("stage_plan_sha256")
        != plan["stage_plan_sha256"][DISCOVERY_COMPONENTS_STAGE]
        or replayed.get("replay_equal") is not True
        or replayed.get("model_calls_issued_by_validator") != 0
        or replayed.get("generation_calls_issued_by_validator") != 0
    ):
        raise V6AR2RunnerError("R2 discovery authorization replay differs from frozen artifacts")
    return prompts


def _require_registered_main_artifacts(result_root: Path = RESULT_ROOT) -> None:
    if (
        REGISTRATION_STATE != EXECUTION_ENABLED_REGISTRATION_STATE
        or result_root.resolve() != RESULT_ROOT.resolve()
    ):
        raise V6AR2BehavioralExecutionDisabled(
            "R2 production execution is not at the exact frozen result root"
        )
    manifest = _load_json(artifact_paths(result_root)["plan_manifest"])
    if (
        manifest.get("artifact_scope") != "registered_main"
        or manifest.get("status")
        != "R2_PLAN_FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"
        or manifest.get("registration_state") != EXECUTION_ENABLED_REGISTRATION_STATE
    ):
        raise V6AR2RunnerError("R2 behavioral execution lacks a registered-main plan")


def run_discovery_components() -> dict[str, Any]:
    _require_non_draft_behavior(allow_draft_test=False)
    _require_registered_main_artifacts(RESULT_ROOT)
    plan, _design, dependency, _receipt = validate_frozen_plan(RESULT_ROOT)

    # Independently rebuild the complete tokenizer/fixture plan before any attempt.
    try:
        replayed = _analyzer_module().replay_frozen_plan(RESULT_ROOT)
    except Exception as error:
        raise V6AR2RunnerError(
            "R2 discovery plan did not independently replay before execution"
        ) from error
    if not isinstance(replayed, Mapping) or replayed.get("plan") != plan:
        raise V6AR2RunnerError("R2 discovery plan replay differs from the frozen plan")

    prompts = validate_discovery_stage_authorization(
        plan,
        result_root=RESULT_ROOT,
    )
    preflight = execution_preflight(
        plan,
        dependency,
        stage=DISCOVERY_COMPONENTS_STAGE,
        result_root=RESULT_ROOT,
    )
    return _execute_stage(
        plan,
        dependency,
        prompts,
        preflight,
        stage=DISCOVERY_COMPONENTS_STAGE,
        result_root=RESULT_ROOT,
        model_loader=load_model_via_sealed_v2,
    )


def run_remaining_main() -> dict[str, Any]:
    _require_non_draft_behavior(allow_draft_test=False)
    _require_registered_main_artifacts(RESULT_ROOT)
    plan, _design, dependency, _receipt = validate_frozen_plan(RESULT_ROOT)
    prompts = validate_remaining_stage_authorization(
        plan,
        result_root=RESULT_ROOT,
    )
    paths = artifact_paths(RESULT_ROOT)
    replayed = _analyzer_module().validate_remaining_authorization_for_runner(
        RESULT_ROOT
    )
    prior = _prior_discovery_authorization_from_replay(replayed, paths)
    preflight = execution_preflight(
        plan,
        dependency,
        stage=REMAINING_MAIN_STAGE,
        result_root=RESULT_ROOT,
    )
    # Preflight can be nontrivial.  Rebind the five immutable discovery fields
    # immediately afterward so concurrent tampering cannot reach attempt write.
    replayed_after_preflight = (
        _analyzer_module().validate_remaining_authorization_for_runner(
            RESULT_ROOT
        )
    )
    prior_after_preflight = _prior_discovery_authorization_from_replay(
        replayed_after_preflight,
        paths,
    )
    if prior_after_preflight != prior:
        raise V6AR2RunnerError(
            "remaining authorization changed during execution preflight"
        )
    return _execute_stage(
        plan,
        dependency,
        prompts,
        preflight,
        stage=REMAINING_MAIN_STAGE,
        result_root=RESULT_ROOT,
        model_loader=load_model_via_sealed_v2,
        prior_discovery_authorization=prior_after_preflight,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("plan", DISCOVERY_COMPONENTS_STAGE, REMAINING_MAIN_STAGE),
        required=True,
    )
    args = parser.parse_args()
    if args.phase == "plan":
        # The public command is deliberately unable to opt into test-only writes.
        run_plan()
    elif args.phase == DISCOVERY_COMPONENTS_STAGE:
        run_discovery_components()
    else:
        run_remaining_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

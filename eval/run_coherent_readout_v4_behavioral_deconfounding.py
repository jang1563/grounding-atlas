"""Freeze and run the V4 synthetic behavioral-deconfounding diagnostic.

``plan`` performs exhaustive tokenizer-only preflight and writes a complete,
immutable 448-call plan.  ``behavior`` consumes that frozen plan exactly once,
without generation or logit processors, and stores every raw next-token
vocabulary row.  This runner does not perform component attribution and makes
no biological, latent-knowledge, activation-gap, or physical-law claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import itertools
import json
import math
import os
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import model_hooks
    from . import run_coherent_binary_readout as v1_runner
    from . import run_coherent_readout_v2_causal_binding as v2_runner
    from . import run_coherent_readout_v2_syntax as syntax_runner
except ImportError:  # direct execution from eval/
    import model_hooks
    import run_coherent_binary_readout as v1_runner
    import run_coherent_readout_v2_causal_binding as v2_runner
    import run_coherent_readout_v2_syntax as syntax_runner


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v4_behavioral_deconfounding"
    / "qwen2.5-1.5b-instruct"
)

PLAN_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-plan-manifest-v1"
DESIGN_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-design-v1"
RECEIPT_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-tokenization-v1"
DEPENDENCY_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-dependency-lock-v1"
CALL_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-call-v1"
ATTEMPT_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-attempt-v1"
RECORD_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-record-v1"
EXECUTION_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-execution-v1"
EXECUTION_INPUT_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-input-v1"

FROZEN_PREREGISTRATION = (
    ROOT / "docs" / "COHERENT_READOUT_V4_BEHAVIORAL_DECONFOUNDING_PREREG.md"
)
FIXTURE_BUILDER = (
    ROOT
    / "signal"
    / "syntax"
    / "build_coherent_readout_v4_behavioral_deconfounding_bank.py"
)
FIXTURE = (
    ROOT
    / "signal"
    / "syntax"
    / "coherent_readout_v4_behavioral_deconfounding_bank.json"
)
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
DEFAULT_ANALYZER = (
    ROOT / "eval" / "analyze_coherent_readout_v4_behavioral_deconfounding.py"
)

# These are replaced with the final byte/canonical digests only after the
# preregistration, builder, fixture, and manifest have all been frozen.  A
# runtime with any PENDING value is rejected before tokenization or execution.
FROZEN_PREREGISTRATION_SHA256 = (
    "af63bb4fabcff96486adf9715c5ade276a44fe5f07dc07add02cb452836e0bdb"
)
FIXTURE_BUILDER_SHA256 = (
    "2af597f4031db819c94a5b4ec6d6845a513e0484eaa9a6da8ab61ce9c6e178da"
)
FIXTURE_SHA256 = (
    "8d988b36d99677e798628b932fea86efd68317cd4a8653bbcd2f12a3294021c2"
)
FIXTURE_CANONICAL_SHA256 = (
    "dd46347c62f1f3823a9813b5d1a302e0e26d86d90c1a4d5d0e28f523dfa73f39"
)
FIXTURE_MANIFEST_SHA256 = (
    "ad957af864564c9e79cb9b2330a2e66076385ceeac5eead833909a4c59b252cb"
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_WEIGHTS_SHA256 = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
MODEL_CONFIG_SHA256 = (
    "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
MODEL_LAYERS = 28
MODEL_WIDTH = 1536
MODEL_VOCAB_SIZE = 151_936
TOKENIZER_VOCAB_SIZE = 151_665
DEVICE = "mps"
DTYPE = "float32"
ATTENTION_IMPLEMENTATION = "sdpa"
LOGITS_SOURCE = "raw_model_output_before_processors"
FIXTURE_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-bank-v1"
FIXTURE_CELL_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-cell-v1"
FIXTURE_WORLD_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-world-v1"
FIXTURE_MANIFEST_SCHEMA = (
    "coherent-readout-v4-behavioral-deconfounding-bank-manifest-v1"
)
ANALYSIS_ID = "coherent-readout-v4-behavioral-deconfounding-v1"
FREEZE_DATE = "2026-08-02"
MODE = "prospective_development_synthetic_nonbiological"
PURPOSE = (
    "prospectively distinguish property retrieval, codebook lookup, intended "
    "composition, and registered label/order heuristics before any activation study"
)
NEUTRAL_SYSTEM_MESSAGE = (
    "Follow the user's labeled task. Reply with exactly the requested "
    "single-character label and nothing else."
)
REGISTERED_COMPOSITION_POLICIES = [
    "intended_compositional_rule",
    "frozen_v3_heuristic",
    "last_displayed_option",
    "first_displayed_codebook_rule_output",
    "constant_y",
    "constant_x",
]

EXPECTED_CALL_COUNT = 448
FAMILY_COUNTS = {
    "composition": 256,
    "property_retrieval": 64,
    "codebook_lookup": 128,
}
FAMILY_ORDER = tuple(FAMILY_COUNTS)
EXPECTED_LABEL_TOKEN_IDS = {"P": 47, "Q": 48, "X": 55, "Y": 56}
CHAT_FLAGS = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "enable_thinking": False,
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260_805

DEFAULT_PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
DEFAULT_DESIGN = RESULT_ROOT / "design.json"
DEFAULT_TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"
DEFAULT_DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
DEFAULT_BEHAVIOR_ATTEMPT = RESULT_ROOT / "behavior_attempt.json"
DEFAULT_BEHAVIOR_RECORDS = RESULT_ROOT / "behavior_records.jsonl"
DEFAULT_BEHAVIOR_LOGITS = RESULT_ROOT / "behavior_full_vocab_logits.npy"
DEFAULT_BEHAVIOR_EXECUTION_MANIFEST = RESULT_ROOT / "behavior_execution_manifest.json"

PLAN_FILENAMES = {
    "plan_manifest": "plan_manifest.json",
    "design": "design.json",
    "tokenization_receipt": "tokenization_receipt.json",
    "dependency_lock": "dependency_lock.json",
}
BEHAVIOR_FILENAMES = {
    "attempt": "behavior_attempt.json",
    "records": "behavior_records.jsonl",
    "full_vocab_logits": "behavior_full_vocab_logits.npy",
    "execution_manifest": "behavior_execution_manifest.json",
}
ANALYSIS_FILENAMES = (
    "behavior_analysis.json",
    "analysis.md",
    "analysis_manifest.json",
)

FACTOR_FIELDS = (
    "target_property",
    "mapping_id",
    "target_fact_order",
    "rule_order",
    "option_order",
)

FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "freeze_date",
    "mode",
    "purpose",
    "neutral_system_message",
    "world_registry",
    "family_counts",
    "expected_call_count",
    "registered_composition_policies",
    "cells",
    "model_calls_made_by_builder",
    "biological_model_calls",
}
WORLD_KEYS = {
    "schema_version",
    "world_id",
    "world_index",
    "target_entity",
    "other_entity",
}
CELL_KEYS = {
    "schema_version",
    "cell_id",
    "world_id",
    "world_index",
    "family_id",
    "stratum_id",
    "target_entity",
    "other_entity",
    "target_property",
    "other_property",
    "mapping_id",
    "target_fact_order",
    "rule_order",
    "option_order",
    "answer_labels",
    "displayed_options",
    "correct_answer",
    "correct_option_position",
    "rule_lines",
    "fact_lines",
    "prompt_lines",
    "prompt_text",
    "prompt_sha256",
    "semantic_bundle_id",
    "permutation_index",
    "v3_heuristic_answer",
    "last_option_heuristic_answer",
    "first_rule_output_heuristic_answer",
    "model_calls_made_by_builder",
    "biological_model_calls",
}
FIXTURE_MANIFEST_KEYS = {
    "schema_version",
    "analysis_id",
    "fixture_path",
    "fixture_file_sha256",
    "fixture_canonical_sha256",
    "builder_path",
    "builder_file_sha256",
    "world_count",
    "family_counts",
    "cell_count",
    "model_calls",
    "biological_model_calls",
}

CALL_KEYS = {
    "schema_version",
    "call_id",
    "planned_index",
    "full_vocab_logits_row",
    "cell_id",
    "world_id",
    "family_id",
    "stratum_id",
    "fixture_cell_sha256",
    *FACTOR_FIELDS,
    "answer_labels",
    "correct_answer",
    "correct_option_position",
    "displayed_options",
    "rule_lines",
    "prompt_lines",
    "prompt_text",
    "prompt_text_sha256",
    "messages",
    "messages_sha256",
    "neutral_system_message_sha256",
    "rendered_text",
    "rendered_text_sha256",
    "execution_input_ids",
    "execution_attention_mask",
    "execution_input_sha256",
    "input_token_count",
    "final_attended_token_index",
    "label_token_ids",
    "correct_token_id",
    "incorrect_token_id",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "chat_template_sha256",
    "device",
    "dtype",
    "vocab_size",
    "logits_source",
}

DIAGNOSTIC_KEYS = {
    "label_logits",
    "label_logit_by_text",
    "first_minus_second_margin",
    "correct_minus_incorrect_margin",
    "full_vocab_logsumexp",
    "label_probability_mass",
    "greedy_token_id",
    "greedy_logit",
    "maximum_token_ids",
    "maximum_tie_count",
    "correct_is_global_maximum",
    "unique_global_argmax_is_correct",
    "greedy_matches_correct",
    "full_vocab_logits_sha256",
}

RECORD_KEYS = {
    "schema_version",
    "record_id",
    "call_id",
    "planned_index",
    "full_vocab_logits_row",
    "cell_id",
    "world_id",
    "family_id",
    "stratum_id",
    "fixture_cell_sha256",
    *FACTOR_FIELDS,
    "answer_labels",
    "correct_answer",
    "correct_option_position",
    "displayed_options",
    "rule_lines",
    "prompt_lines",
    "prompt_text",
    "prompt_text_sha256",
    "rendered_text_sha256",
    "messages_sha256",
    "execution_input_sha256",
    "input_token_count",
    "label_token_ids",
    "correct_token_id",
    "incorrect_token_id",
    "diagnostics",
    "forward_trace",
    "call_plan_sha256",
    "runner_sha256",
    "preregistration_sha256",
    "biological_model_calls",
}


class BehavioralDeconfoundingRunnerError(ValueError):
    """Raised when V4 planning or execution violates the frozen contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise BehavioralDeconfoundingRunnerError("float32 value is not finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise BehavioralDeconfoundingRunnerError(
            f"{label} keys changed (missing={missing}, extra={extra})"
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BehavioralDeconfoundingRunnerError(
            f"cannot read JSON artifact: {path}"
        ) from error
    if not isinstance(value, dict):
        raise BehavioralDeconfoundingRunnerError(
            f"JSON artifact must be an object: {path}"
        )
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise BehavioralDeconfoundingRunnerError(
            f"cannot read JSONL artifact: {path}"
        ) from error
    if not lines or any(not isinstance(value, dict) for value in values):
        raise BehavioralDeconfoundingRunnerError("JSONL rows must be objects")
    return values


def _atomic_frozen_write(path: Path, payload: bytes) -> None:
    """Atomically write once; an exactly identical planning replay is a no-op."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise BehavioralDeconfoundingRunnerError(
                f"refusing to overwrite differing frozen artifact: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise BehavioralDeconfoundingRunnerError(
            f"stale atomic temporary exists: {temporary}"
        )
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_frozen_write(path, payload)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(value)) + "\n" for value in values).encode(
        "utf-8"
    )
    _atomic_frozen_write(path, payload)


def write_f32_sidecar(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if array.ndim != 2 or not np.isfinite(array).all():
        raise BehavioralDeconfoundingRunnerError(
            "full-vocabulary sidecar must be a finite f32 matrix"
        )
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_frozen_write(path, buffer.getvalue())


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise BehavioralDeconfoundingRunnerError(
            f"required distribution is unavailable: {distribution}"
        ) from error


def _require_final_hash_constants() -> None:
    values = (
        FROZEN_PREREGISTRATION_SHA256,
        FIXTURE_BUILDER_SHA256,
        FIXTURE_SHA256,
        FIXTURE_CANONICAL_SHA256,
        FIXTURE_MANIFEST_SHA256,
    )
    if any(value.startswith("PENDING_") for value in values):
        raise BehavioralDeconfoundingRunnerError(
            "V4 static hash ledger is not yet frozen"
        )


def _require_frozen_static_inputs() -> None:
    _require_final_hash_constants()
    expected = (
        (
            FROZEN_PREREGISTRATION,
            FROZEN_PREREGISTRATION_SHA256,
            "preregistration",
        ),
        (FIXTURE_BUILDER, FIXTURE_BUILDER_SHA256, "fixture builder"),
        (FIXTURE, FIXTURE_SHA256, "fixture"),
        (FIXTURE_MANIFEST, FIXTURE_MANIFEST_SHA256, "fixture manifest"),
    )
    for path, digest, label in expected:
        if not path.is_file() or file_sha256(path) != digest:
            raise BehavioralDeconfoundingRunnerError(
                f"{label} differs from its frozen hash"
            )


def _load_builder_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "v4_behavioral_deconfounding_builder", FIXTURE_BUILDER
    )
    if specification is None or specification.loader is None:
        raise BehavioralDeconfoundingRunnerError(
            "cannot import the frozen V4 fixture builder"
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _family_id(cell: Mapping[str, Any]) -> str:
    value = cell.get("family_id")
    if value not in FAMILY_COUNTS:
        raise BehavioralDeconfoundingRunnerError("fixture family identifier is invalid")
    return str(value)


def _answer_labels(cell: Mapping[str, Any]) -> list[str]:
    value = cell.get("answer_labels")
    if not isinstance(value, list) or tuple(value) not in (("X", "Y"), ("P", "Q")):
        raise BehavioralDeconfoundingRunnerError(
            "answer_labels must be exactly [X,Y] or [P,Q]"
        )
    return [str(item) for item in value]


def _stratum_id(cell: Mapping[str, Any]) -> str:
    existing = cell.get("stratum_id")
    if not isinstance(existing, str) or not existing:
        raise BehavioralDeconfoundingRunnerError("fixture stratum_id is invalid")
    return existing


_CODEBOOKS = {
    "identity": {"P": "X", "Q": "Y"},
    "swapped": {"P": "Y", "Q": "X"},
}


def _expected_mapping_answer(target_property: str, mapping_id: str) -> str:
    try:
        return _CODEBOOKS[mapping_id][target_property]
    except KeyError as error:
        raise BehavioralDeconfoundingRunnerError(
            "fixture mapping factors are invalid"
        ) from error


def _expected_rule_lines(mapping_id: str, rule_order: str) -> list[str]:
    if rule_order == "p_rule_first":
        properties = ("P", "Q")
    elif rule_order == "q_rule_first":
        properties = ("Q", "P")
    else:
        raise BehavioralDeconfoundingRunnerError("fixture rule_order is invalid")
    return [
        f"RULE: {property_name} maps to {_expected_mapping_answer(property_name, mapping_id)}."
        for property_name in properties
    ]


def _expected_displayed_options(family_id: str, option_order: str) -> list[str]:
    if family_id == "property_retrieval":
        if option_order == "p_then_q":
            return ["P", "Q"]
        if option_order == "q_then_p":
            return ["Q", "P"]
    else:
        if option_order == "x_then_y":
            return ["X", "Y"]
        if option_order == "y_then_x":
            return ["Y", "X"]
    raise BehavioralDeconfoundingRunnerError(
        "fixture option_order is invalid for its family"
    )


def _expected_fact_lines(
    target_entity: str,
    other_entity: str,
    target_property: str,
    target_fact_order: str,
) -> list[str]:
    other_property = "Q" if target_property == "P" else "P"
    target = f"TARGET FACT: {target_entity} has property {target_property}."
    other = f"OTHER FACT: {other_entity} has property {other_property}."
    if target_fact_order == "target_first":
        return [target, other]
    if target_fact_order == "target_second":
        return [other, target]
    raise BehavioralDeconfoundingRunnerError("fixture target_fact_order is invalid")


def _expected_prompt_parts(
    *,
    family_id: str,
    target_entity: str,
    other_entity: str,
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
    rule_order: str | None,
    displayed_options: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    option_text = ", ".join(displayed_options)
    if family_id == "composition":
        if mapping_id is None or target_fact_order is None or rule_order is None:
            raise BehavioralDeconfoundingRunnerError(
                "composition prompt factors are incomplete"
            )
        rules = _expected_rule_lines(mapping_id, rule_order)
        facts = _expected_fact_lines(
            target_entity, other_entity, target_property, target_fact_order
        )
        lines = [
            "CODEBOOK RULES",
            *rules,
            "LABELED FACTS",
            *facts,
            "TASK: Apply the CODEBOOK RULES to the property in TARGET FACT.",
            f"Valid output codes (display order carries no meaning): {option_text}.",
        ]
        return lines, rules, facts
    if family_id == "property_retrieval":
        if mapping_id is not None or rule_order is not None or target_fact_order is None:
            raise BehavioralDeconfoundingRunnerError(
                "retrieval prompt factors are invalid"
            )
        facts = _expected_fact_lines(
            target_entity, other_entity, target_property, target_fact_order
        )
        lines = [
            "LABELED FACTS",
            *facts,
            "TASK: Return the property stated in TARGET FACT.",
            f"Valid output properties (display order carries no meaning): {option_text}.",
        ]
        return lines, [], facts
    if family_id == "codebook_lookup":
        if mapping_id is None or rule_order is None or target_fact_order is not None:
            raise BehavioralDeconfoundingRunnerError(
                "lookup prompt factors are invalid"
            )
        rules = _expected_rule_lines(mapping_id, rule_order)
        lines = [
            "CODEBOOK RULES",
            *rules,
            f"GIVEN PROPERTY: {target_property}.",
            "TASK: Return the code assigned by the CODEBOOK RULES to GIVEN PROPERTY.",
            f"Valid output codes (display order carries no meaning): {option_text}.",
        ]
        return lines, rules, []
    raise BehavioralDeconfoundingRunnerError("fixture family is invalid")


def _expected_fixture_cell(
    cell: Mapping[str, Any], world: Mapping[str, Any]
) -> dict[str, Any]:
    family_id = _family_id(cell)
    target_property = cell.get("target_property")
    mapping_id = cell.get("mapping_id")
    target_fact_order = cell.get("target_fact_order")
    rule_order = cell.get("rule_order")
    option_order = cell.get("option_order")
    if target_property not in {"P", "Q"}:
        raise BehavioralDeconfoundingRunnerError("fixture target_property is invalid")
    if not isinstance(option_order, str):
        raise BehavioralDeconfoundingRunnerError("fixture option_order is invalid")
    displayed = _expected_displayed_options(family_id, option_order)
    labels = ["P", "Q"] if family_id == "property_retrieval" else ["X", "Y"]
    if family_id == "property_retrieval":
        correct_answer = target_property
        permutations = list(
            itertools.product(
                ("target_first", "target_second"), ("p_then_q", "q_then_p")
            )
        )
        permutation_tuple = (target_fact_order, option_order)
    elif family_id == "codebook_lookup":
        if not isinstance(mapping_id, str):
            raise BehavioralDeconfoundingRunnerError("lookup mapping_id is invalid")
        correct_answer = _expected_mapping_answer(target_property, mapping_id)
        permutations = list(
            itertools.product(
                ("p_rule_first", "q_rule_first"), ("x_then_y", "y_then_x")
            )
        )
        permutation_tuple = (rule_order, option_order)
    else:
        if not isinstance(mapping_id, str):
            raise BehavioralDeconfoundingRunnerError(
                "composition mapping_id is invalid"
            )
        correct_answer = _expected_mapping_answer(target_property, mapping_id)
        permutations = list(
            itertools.product(
                ("target_first", "target_second"),
                ("p_rule_first", "q_rule_first"),
                ("x_then_y", "y_then_x"),
            )
        )
        permutation_tuple = (target_fact_order, rule_order, option_order)
    try:
        permutation_index = permutations.index(permutation_tuple)
    except ValueError as error:
        raise BehavioralDeconfoundingRunnerError(
            "fixture permutation factors are invalid"
        ) from error
    target_entity = str(world["target_entity"])
    other_entity = str(world["other_entity"])
    prompt_lines, rule_lines, fact_lines = _expected_prompt_parts(
        family_id=family_id,
        target_entity=target_entity,
        other_entity=other_entity,
        target_property=target_property,
        mapping_id=mapping_id if isinstance(mapping_id, str) else None,
        target_fact_order=(
            target_fact_order if isinstance(target_fact_order, str) else None
        ),
        rule_order=rule_order if isinstance(rule_order, str) else None,
        displayed_options=displayed,
    )
    prompt_text = "\n".join(prompt_lines)
    factor_parts = [
        f"p-{target_property.lower()}",
        f"m-{mapping_id or 'none'}",
        f"f-{target_fact_order or 'none'}",
        f"r-{rule_order or 'none'}",
        f"o-{option_order}",
    ]
    cell_id = ":".join(
        ["behavior-v4", str(world["world_id"]), family_id, *factor_parts]
    )
    if family_id == "property_retrieval":
        semantic_bundle_id = (
            f"retrieval:{world['world_id']}:p-{target_property.lower()}"
        )
        stratum_id = (
            f"retrieval:p-{target_property.lower()}:"
            f"f-{target_fact_order}:o-{option_order}"
        )
    elif family_id == "codebook_lookup":
        semantic_bundle_id = (
            f"lookup:{world['world_id']}:p-{target_property.lower()}:m-{mapping_id}"
        )
        stratum_id = (
            f"lookup:p-{target_property.lower()}:m-{mapping_id}:"
            f"r-{rule_order}:o-{option_order}"
        )
    else:
        semantic_bundle_id = (
            f"composition:{world['world_id']}:p-{target_property.lower()}:m-{mapping_id}"
        )
        stratum_id = (
            f"composition:p-{target_property.lower()}:m-{mapping_id}:"
            f"f-{target_fact_order}:r-{rule_order}:o-{option_order}"
        )
    if mapping_id is None or rule_order is None:
        first_rule = None
    else:
        first_property = "P" if rule_order == "p_rule_first" else "Q"
        first_rule = _expected_mapping_answer(first_property, mapping_id)
    if family_id == "composition":
        v3_answer = (
            "X"
            if target_property == "P"
            and mapping_id == "identity"
            and target_fact_order == "target_first"
            else "Y"
        )
    else:
        v3_answer = None
    return {
        "schema_version": FIXTURE_CELL_SCHEMA,
        "cell_id": cell_id,
        "world_id": world["world_id"],
        "world_index": world["world_index"],
        "family_id": family_id,
        "stratum_id": stratum_id,
        "target_entity": target_entity,
        "other_entity": other_entity,
        "target_property": target_property,
        "other_property": "Q" if target_property == "P" else "P",
        "mapping_id": mapping_id,
        "target_fact_order": target_fact_order,
        "rule_order": rule_order,
        "option_order": option_order,
        "answer_labels": labels,
        "displayed_options": displayed,
        "correct_answer": correct_answer,
        "correct_option_position": (
            "first" if displayed[0] == correct_answer else "last"
        ),
        "rule_lines": rule_lines,
        "fact_lines": fact_lines,
        "prompt_lines": prompt_lines,
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "semantic_bundle_id": semantic_bundle_id,
        "permutation_index": permutation_index,
        "v3_heuristic_answer": v3_answer,
        "last_option_heuristic_answer": displayed[-1],
        "first_rule_output_heuristic_answer": first_rule,
        "model_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def _validate_fixture_semantics(fixture: Mapping[str, Any]) -> None:
    _exact_keys(fixture, FIXTURE_KEYS, "fixture")
    cells = fixture.get("cells")
    worlds = fixture.get("world_registry")
    system = fixture.get("neutral_system_message")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALL_COUNT:
        raise BehavioralDeconfoundingRunnerError("fixture must contain 448 cells")
    if not isinstance(worlds, list) or len(worlds) != 8:
        raise BehavioralDeconfoundingRunnerError("fixture must contain eight worlds")
    expected_header = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": MODE,
        "purpose": PURPOSE,
        "neutral_system_message": NEUTRAL_SYSTEM_MESSAGE,
        "family_counts": FAMILY_COUNTS,
        "expected_call_count": EXPECTED_CALL_COUNT,
        "registered_composition_policies": REGISTERED_COMPOSITION_POLICIES,
        "model_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    if any(fixture[field] != expected for field, expected in expected_header.items()):
        raise BehavioralDeconfoundingRunnerError("fixture header changed")
    if system != NEUTRAL_SYSTEM_MESSAGE:
        raise BehavioralDeconfoundingRunnerError("fixture system message changed")

    world_ids: list[str] = []
    world_by_id: dict[str, Mapping[str, Any]] = {}
    for index, world in enumerate(worlds, start=1):
        if not isinstance(world, dict):
            raise BehavioralDeconfoundingRunnerError("fixture world must be an object")
        _exact_keys(world, WORLD_KEYS, "fixture world")
        expected_world = {
            "schema_version": FIXTURE_WORLD_SCHEMA,
            "world_id": f"behavior_world_{index:03d}",
            "world_index": index,
            "target_entity": f"referent_a_{index:03d}",
            "other_entity": f"referent_b_{index:03d}",
        }
        if world != expected_world:
            raise BehavioralDeconfoundingRunnerError("fixture world identity changed")
        world_ids.append(world["world_id"])
        world_by_id[world["world_id"]] = world
    cell_ids: set[str] = set()
    family_counts = {family: 0 for family in FAMILY_ORDER}
    world_counts = {str(world_id): 0 for world_id in world_ids}
    combinations: dict[tuple[str, str], set[tuple[Any, ...]]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise BehavioralDeconfoundingRunnerError("fixture cell must be an object")
        _exact_keys(cell, CELL_KEYS, "fixture cell")
        cell_id = cell.get("cell_id")
        world_id = cell.get("world_id")
        family = _family_id(cell)
        if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
            raise BehavioralDeconfoundingRunnerError(
                "fixture cell IDs are missing or duplicated"
            )
        if world_id not in world_counts:
            raise BehavioralDeconfoundingRunnerError(
                "fixture cell references an unknown world"
            )
        expected_cell = _expected_fixture_cell(cell, world_by_id[str(world_id)])
        if cell != expected_cell:
            raise BehavioralDeconfoundingRunnerError(
                f"fixture cell does not independently reconstruct: {cell_id}"
            )
        cell_ids.add(cell_id)
        world_counts[str(world_id)] += 1
        family_counts[family] += 1
        labels = _answer_labels(cell)
        if cell.get("correct_answer") not in labels:
            raise BehavioralDeconfoundingRunnerError(
                "fixture correct answer is outside answer_labels"
            )
        if cell.get("target_property") not in {"P", "Q"}:
            raise BehavioralDeconfoundingRunnerError("target_property level changed")
        prompt_text = cell.get("prompt_text")
        if not isinstance(prompt_text, str) or text_sha256(prompt_text) != cell.get(
            "prompt_sha256"
        ):
            raise BehavioralDeconfoundingRunnerError("fixture prompt digest changed")
        for field in ("displayed_options", "rule_lines", "prompt_lines"):
            if not isinstance(cell.get(field), list):
                raise BehavioralDeconfoundingRunnerError(
                    f"fixture {field} must be a list"
                )

        mapping_id = cell.get("mapping_id")
        fact_order = cell.get("target_fact_order")
        rule_order = cell.get("rule_order")
        option_order = cell.get("option_order")
        if family == "composition":
            valid = (
                mapping_id in {"identity", "swapped"}
                and fact_order in {"target_first", "target_second"}
                and rule_order in {"p_rule_first", "q_rule_first"}
                and option_order in {"x_then_y", "y_then_x"}
                and labels == ["X", "Y"]
            )
            key = (
                cell["target_property"],
                mapping_id,
                fact_order,
                rule_order,
                option_order,
            )
        elif family == "property_retrieval":
            valid = (
                mapping_id is None
                and fact_order in {"target_first", "target_second"}
                and rule_order is None
                and option_order in {"p_then_q", "q_then_p"}
                and labels == ["P", "Q"]
            )
            key = (cell["target_property"], fact_order, option_order)
        else:
            valid = (
                mapping_id in {"identity", "swapped"}
                and fact_order is None
                and rule_order in {"p_rule_first", "q_rule_first"}
                and option_order in {"x_then_y", "y_then_x"}
                and labels == ["X", "Y"]
            )
            key = (cell["target_property"], mapping_id, rule_order, option_order)
        if not valid:
            raise BehavioralDeconfoundingRunnerError(
                f"fixture factor levels changed for {family}"
            )
        combinations.setdefault((str(world_id), family), set()).add(key)

    if family_counts != FAMILY_COUNTS or set(world_counts.values()) != {56}:
        raise BehavioralDeconfoundingRunnerError(
            "fixture family or per-world allocation changed"
        )
    expected_per_world = {
        "composition": 32,
        "property_retrieval": 8,
        "codebook_lookup": 16,
    }
    for world_id in world_counts:
        for family, expected in expected_per_world.items():
            if len(combinations.get((world_id, family), set())) != expected:
                raise BehavioralDeconfoundingRunnerError(
                    "fixture factorial coverage changed"
                )


def load_and_rebuild_fixture() -> dict[str, Any]:
    """Require byte identity and deterministic semantic rebuild identity."""

    _require_frozen_static_inputs()
    fixture = _load_json(FIXTURE)
    if canonical_sha256(fixture) != FIXTURE_CANONICAL_SHA256:
        raise BehavioralDeconfoundingRunnerError("fixture canonical hash changed")
    manifest = _load_json(FIXTURE_MANIFEST)
    _exact_keys(manifest, FIXTURE_MANIFEST_KEYS, "fixture manifest")
    expected_manifest = {
        "schema_version": FIXTURE_MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "fixture_path": str(FIXTURE),
        "fixture_file_sha256": FIXTURE_SHA256,
        "fixture_canonical_sha256": FIXTURE_CANONICAL_SHA256,
        "builder_path": str(FIXTURE_BUILDER),
        "builder_file_sha256": FIXTURE_BUILDER_SHA256,
        "world_count": 8,
        "family_counts": FAMILY_COUNTS,
        "cell_count": EXPECTED_CALL_COUNT,
        "model_calls": 0,
        "biological_model_calls": 0,
    }
    if manifest != expected_manifest:
        raise BehavioralDeconfoundingRunnerError(
            "fixture manifest does not reconstruct from frozen inputs"
        )
    builder = _load_builder_module()
    build = getattr(builder, "build_fixture", None)
    validate = getattr(builder, "validate_fixture", None)
    if not callable(build) or not callable(validate):
        raise BehavioralDeconfoundingRunnerError(
            "fixture builder API must expose build_fixture and validate_fixture"
        )
    rebuilt = build()
    validate(rebuilt)
    if rebuilt != fixture or canonical_sha256(rebuilt) != FIXTURE_CANONICAL_SHA256:
        raise BehavioralDeconfoundingRunnerError(
            "fixture deterministic rebuild differs from disk"
        )
    _validate_fixture_semantics(fixture)
    return fixture


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    try:
        raw = list(value)
    except TypeError as error:
        raise BehavioralDeconfoundingRunnerError(
            f"{label} must be an integer vector"
        ) from error
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise BehavioralDeconfoundingRunnerError(
                f"{label} must contain exactly one row"
            )
        raw = raw[0]
    if not raw or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise BehavioralDeconfoundingRunnerError(
            f"{label} must be a nonempty integer vector"
        )
    return [int(item) for item in raw]


def _contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "rendered prompt"
    )
    combined = _as_int_vector(
        tokenizer.encode(rendered + answer, add_special_tokens=False),
        f"rendered prompt plus {answer}",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise BehavioralDeconfoundingRunnerError(
            f"answer label {answer} is not exactly one contextual token"
        )
    return combined[-1]


def execution_input_sha256(input_ids: Sequence[int], attention_mask: Sequence[int]) -> str:
    return canonical_sha256(
        {
            "schema_version": EXECUTION_INPUT_SCHEMA,
            "input_ids": list(input_ids),
            "attention_mask": list(attention_mask),
        }
    )


def _cell_field(cell: Mapping[str, Any], field: str) -> Any:
    return cell.get(field)


def render_call(
    tokenizer: Any,
    cell: Mapping[str, Any],
    *,
    planned_index: int,
    neutral_system_message: str,
) -> dict[str, Any]:
    """Render one frozen fixture cell and bind all execution bytes."""

    if text_sha256(str(cell.get("prompt_text"))) != cell.get("prompt_sha256"):
        raise BehavioralDeconfoundingRunnerError("fixture prompt digest changed")
    messages = [
        {"role": "system", "content": neutral_system_message},
        {"role": "user", "content": cell["prompt_text"]},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    if not isinstance(rendered, str) or not rendered:
        raise BehavioralDeconfoundingRunnerError(
            "chat template did not produce nonempty rendered text"
        )
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **CHAT_FLAGS,
    )
    if isinstance(tokenized, Mapping):
        if "input_ids" not in tokenized:
            raise BehavioralDeconfoundingRunnerError(
                "tokenized chat template omitted input_ids"
            )
        input_ids = _as_int_vector(tokenized["input_ids"], "chat-template IDs")
        attention_mask = _as_int_vector(
            tokenized.get("attention_mask", [1] * len(input_ids)),
            "chat-template attention mask",
        )
    else:
        input_ids = _as_int_vector(tokenized, "chat-template IDs")
        attention_mask = [1] * len(input_ids)
    retokenized = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "retokenized IDs"
    )
    if retokenized != input_ids:
        raise BehavioralDeconfoundingRunnerError(
            "rendered chat does not retokenize identically"
        )
    if len(attention_mask) != len(input_ids) or attention_mask != [1] * len(input_ids):
        raise BehavioralDeconfoundingRunnerError(
            "attention mask must attend every planned token"
        )
    labels = _answer_labels(cell)
    label_token_ids = [
        _contextual_token_id(tokenizer, rendered, answer) for answer in labels
    ]
    if len(set(label_token_ids)) != 2:
        raise BehavioralDeconfoundingRunnerError("answer token IDs are not distinct")
    expected_ids = [EXPECTED_LABEL_TOKEN_IDS[label] for label in labels]
    if label_token_ids != expected_ids:
        raise BehavioralDeconfoundingRunnerError(
            "contextual answer token IDs changed"
        )
    correct_answer = str(cell["correct_answer"])
    correct_index = labels.index(correct_answer)
    incorrect_index = 1 - correct_index
    core = {
        "schema_version": CALL_SCHEMA,
        "planned_index": planned_index,
        "full_vocab_logits_row": planned_index,
        "cell_id": cell["cell_id"],
        "world_id": cell["world_id"],
        "family_id": _family_id(cell),
        "stratum_id": _stratum_id(cell),
        "fixture_cell_sha256": canonical_sha256(cell),
        **{field: _cell_field(cell, field) for field in FACTOR_FIELDS},
        "answer_labels": labels,
        "correct_answer": correct_answer,
        "correct_option_position": cell["correct_option_position"],
        "displayed_options": cell["displayed_options"],
        "rule_lines": cell["rule_lines"],
        "prompt_lines": cell["prompt_lines"],
        "prompt_text": cell["prompt_text"],
        "prompt_text_sha256": cell["prompt_sha256"],
        "messages": messages,
        "messages_sha256": canonical_sha256(messages),
        "neutral_system_message_sha256": text_sha256(neutral_system_message),
        "rendered_text": rendered,
        "rendered_text_sha256": text_sha256(rendered),
        "execution_input_ids": input_ids,
        "execution_attention_mask": attention_mask,
        "execution_input_sha256": execution_input_sha256(input_ids, attention_mask),
        "input_token_count": len(input_ids),
        "final_attended_token_index": len(input_ids) - 1,
        "label_token_ids": label_token_ids,
        "correct_token_id": label_token_ids[correct_index],
        "incorrect_token_id": label_token_ids[incorrect_index],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "device": DEVICE,
        "dtype": DTYPE,
        "vocab_size": MODEL_VOCAB_SIZE,
        "logits_source": LOGITS_SOURCE,
    }
    value = {**core, "call_id": canonical_sha256(core)}
    _exact_keys(value, CALL_KEYS, "planned call")
    return value


def _verify_cached_plan_assets() -> None:
    for filename, digest in (
        ("config.json", MODEL_CONFIG_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("tokenizer.json", TOKENIZER_JSON_SHA256),
    ):
        path = v2_runner._verify_cached_model_asset(filename, digest)
        if not path.is_file():
            raise BehavioralDeconfoundingRunnerError(
                f"cached model asset is missing: {filename}"
            )


def _verify_cached_model_assets() -> None:
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    _verify_cached_plan_assets()


def _config_and_tokenizer() -> tuple[Any, int]:
    _verify_cached_plan_assets()
    tokenizer = v1_runner._load_hf_tokenizer(
        MODEL_ID, MODEL_ID, MODEL_REVISION, local_files_only=True
    )
    vocab_size = v1_runner._load_hf_config_vocab_size(
        MODEL_ID, MODEL_REVISION, local_files_only=True
    )
    if vocab_size != MODEL_VOCAB_SIZE or len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise BehavioralDeconfoundingRunnerError(
            "model or tokenizer vocabulary lock changed"
        )
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise BehavioralDeconfoundingRunnerError("effective chat template changed")
    return tokenizer, vocab_size


def _dependency_lock(analyzer_path: Path) -> dict[str, Any]:
    if not analyzer_path.is_file():
        raise BehavioralDeconfoundingRunnerError(
            f"frozen analyzer is missing: {analyzer_path}"
        )
    try:
        import torch
    except ImportError as error:
        raise BehavioralDeconfoundingRunnerError(
            "torch is required for the V4 dependency lock"
        ) from error
    test_paths = sorted(ROOT.glob("tests/test_*v4*"))
    if not test_paths:
        raise BehavioralDeconfoundingRunnerError("V4 test-file registry is empty")
    implementation_files = {
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__)),
        },
        "analyzer": {
            "path": str(analyzer_path.resolve()),
            "sha256": file_sha256(analyzer_path),
        },
        "fixture_builder": {
            "path": str(FIXTURE_BUILDER),
            "sha256": FIXTURE_BUILDER_SHA256,
        },
        "model_hooks": {
            "path": str(ROOT / "eval" / "model_hooks.py"),
            "sha256": file_sha256(ROOT / "eval" / "model_hooks.py"),
        },
        "run_coherent_binary_readout": {
            "path": str(ROOT / "eval" / "run_coherent_binary_readout.py"),
            "sha256": file_sha256(
                ROOT / "eval" / "run_coherent_binary_readout.py"
            ),
        },
        "run_coherent_readout_v2_causal_binding": {
            "path": str(
                ROOT / "eval" / "run_coherent_readout_v2_causal_binding.py"
            ),
            "sha256": file_sha256(
                ROOT / "eval" / "run_coherent_readout_v2_causal_binding.py"
            ),
        },
        "run_coherent_readout_v2_syntax": {
            "path": str(ROOT / "eval" / "run_coherent_readout_v2_syntax.py"),
            "sha256": file_sha256(
                ROOT / "eval" / "run_coherent_readout_v2_syntax.py"
            ),
        },
        "tests": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in test_paths
        ],
    }
    core = {
        "schema_version": DEPENDENCY_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in (
                "huggingface-hub",
                "numpy",
                "safetensors",
                "tokenizers",
                "torch",
                "transformers",
            )
        },
        "runtime": {
            "device": DEVICE,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "implementation_files": implementation_files,
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def _validate_dependency_lock(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "python",
        "platform",
        "machine",
        "packages",
        "runtime",
        "implementation_files",
        "canonical_sha256",
    }
    _exact_keys(value, expected_keys, "dependency lock")
    if value["schema_version"] != DEPENDENCY_SCHEMA:
        raise BehavioralDeconfoundingRunnerError("dependency schema changed")
    core = {key: value[key] for key in expected_keys - {"canonical_sha256"}}
    if canonical_sha256(core) != value["canonical_sha256"]:
        raise BehavioralDeconfoundingRunnerError(
            "dependency-lock canonical hash changed"
        )
    implementations = value["implementation_files"]
    if not isinstance(implementations, dict):
        raise BehavioralDeconfoundingRunnerError(
            "dependency implementation registry is invalid"
        )
    for name, entry in implementations.items():
        entries = entry if name == "tests" else [entry]
        if not isinstance(entries, list):
            raise BehavioralDeconfoundingRunnerError(
                "dependency implementation entry is invalid"
            )
        for item in entries:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise BehavioralDeconfoundingRunnerError(
                    "dependency file lock is invalid"
                )
            path = Path(item["path"])
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise BehavioralDeconfoundingRunnerError(
                    f"dependency file changed after plan freeze: {path}"
                )


def _receipt_from_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "model_calls": 0,
        "generation_used": False,
        "logit_processors_used": False,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "chat_flags": CHAT_FLAGS,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "expected_label_token_ids": EXPECTED_LABEL_TOKEN_IDS,
        "prompt_count": len(calls),
        "token_count_min": min(call["input_token_count"] for call in calls),
        "token_count_max": max(call["input_token_count"] for call in calls),
        "prompt_receipts": [
            {
                "planned_index": call["planned_index"],
                "cell_id": call["cell_id"],
                "call_id": call["call_id"],
                "family_id": call["family_id"],
                "answer_labels": call["answer_labels"],
                "label_token_ids": call["label_token_ids"],
                "rendered_text_sha256": call["rendered_text_sha256"],
                "execution_input_sha256": call["execution_input_sha256"],
                "input_token_count": call["input_token_count"],
            }
            for call in calls
        ],
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def _validate_tokenization_receipt(
    receipt: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> None:
    expected = _receipt_from_calls(calls)
    if receipt != expected:
        raise BehavioralDeconfoundingRunnerError(
            "tokenization receipt differs from the complete call plan"
        )


def build_plan(
    tokenizer: Any,
    analyzer_path: Path = DEFAULT_ANALYZER,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the exhaustive tokenizer-only plan without loading model weights."""

    fixture = load_and_rebuild_fixture()
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise BehavioralDeconfoundingRunnerError("effective chat template changed")
    dependency = _dependency_lock(analyzer_path)
    system = fixture["neutral_system_message"]
    cells = fixture["cells"]
    calls = [
        render_call(
            tokenizer,
            cell,
            planned_index=index,
            neutral_system_message=system,
        )
        for index, cell in enumerate(cells)
    ]
    if len(calls) != EXPECTED_CALL_COUNT or len(
        {call["call_id"] for call in calls}
    ) != EXPECTED_CALL_COUNT:
        raise BehavioralDeconfoundingRunnerError(
            "call plan must contain exactly 448 unique calls"
        )
    if [call["full_vocab_logits_row"] for call in calls] != list(
        range(EXPECTED_CALL_COUNT)
    ):
        raise BehavioralDeconfoundingRunnerError("call-plan row order changed")
    observed_family_counts = {
        family: sum(call["family_id"] == family for call in calls)
        for family in FAMILY_ORDER
    }
    if observed_family_counts != FAMILY_COUNTS:
        raise BehavioralDeconfoundingRunnerError("call-plan family counts changed")
    receipt = _receipt_from_calls(calls)
    call_plan_sha256 = canonical_sha256(calls)
    model_lock = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "decoder_layers": MODEL_LAYERS,
        "hidden_width": MODEL_WIDTH,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "device": DEVICE,
        "dtype": DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
    }
    core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": fixture["analysis_id"],
        "freeze_date": fixture["freeze_date"],
        "mode": fixture["mode"],
        "purpose": fixture["purpose"],
        "status": "PLAN_FROZEN_ZERO_FORWARD",
        "model": model_lock,
        "locks": {
            "preregistration": {
                "path": str(FROZEN_PREREGISTRATION),
                "sha256": FROZEN_PREREGISTRATION_SHA256,
            },
            "fixture_builder": {
                "path": str(FIXTURE_BUILDER),
                "sha256": FIXTURE_BUILDER_SHA256,
            },
            "fixture": {
                "path": str(FIXTURE),
                "sha256": FIXTURE_SHA256,
                "canonical_sha256": FIXTURE_CANONICAL_SHA256,
            },
            "fixture_manifest": {
                "path": str(FIXTURE_MANIFEST),
                "sha256": FIXTURE_MANIFEST_SHA256,
            },
            "tokenization_receipt_canonical_sha256": receipt["canonical_sha256"],
            "dependency_lock_canonical_sha256": dependency["canonical_sha256"],
        },
        "neutral_system_message": system,
        "chat_flags": CHAT_FLAGS,
        "factor_contract": {
            "family_order": list(FAMILY_ORDER),
            "family_counts": FAMILY_COUNTS,
            "factor_fields": list(FACTOR_FIELDS),
            "target_property": ["P", "Q"],
            "mapping_id": ["identity", "swapped", None],
            "target_fact_order": ["target_first", "target_second", None],
            "rule_order": ["p_rule_first", "q_rule_first", None],
            "option_order": [
                "x_then_y",
                "y_then_x",
                "p_then_q",
                "q_then_p",
            ],
            "answer_label_pairs": [["X", "Y"], ["P", "Q"]],
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "symbolic_world",
        },
        "cell_registry": cells,
        "calls": calls,
        "record_count": EXPECTED_CALL_COUNT,
        "expected_call_ids": [call["call_id"] for call in calls],
        "call_plan_sha256": call_plan_sha256,
        "model_calls_before_plan_freeze": 0,
        "generation_used": False,
        "logit_processors_used": False,
        "biological_model_calls": 0,
    }
    plan = {**core, "canonical_sha256": canonical_sha256(core)}
    validate_plan_manifest(plan)
    return plan, receipt, dependency


def validate_plan_manifest(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "analysis_id",
        "freeze_date",
        "mode",
        "purpose",
        "status",
        "model",
        "locks",
        "neutral_system_message",
        "chat_flags",
        "factor_contract",
        "bootstrap",
        "cell_registry",
        "calls",
        "record_count",
        "expected_call_ids",
        "call_plan_sha256",
        "model_calls_before_plan_freeze",
        "generation_used",
        "logit_processors_used",
        "biological_model_calls",
        "canonical_sha256",
    }
    _exact_keys(plan, expected_keys, "plan manifest")
    if (
        plan["schema_version"] != PLAN_SCHEMA
        or plan["status"] != "PLAN_FROZEN_ZERO_FORWARD"
        or plan["record_count"] != EXPECTED_CALL_COUNT
        or plan["model_calls_before_plan_freeze"] != 0
        or plan["generation_used"] is not False
        or plan["logit_processors_used"] is not False
        or plan["biological_model_calls"] != 0
    ):
        raise BehavioralDeconfoundingRunnerError("plan manifest header changed")
    calls = plan["calls"]
    if not isinstance(calls, list) or len(calls) != EXPECTED_CALL_COUNT:
        raise BehavioralDeconfoundingRunnerError("plan call registry changed")
    if not isinstance(plan["cell_registry"], list) or len(
        plan["cell_registry"]
    ) != EXPECTED_CALL_COUNT:
        raise BehavioralDeconfoundingRunnerError("plan cell registry changed")
    cell_by_id = {
        cell.get("cell_id"): cell
        for cell in plan["cell_registry"]
        if isinstance(cell, dict)
    }
    if len(cell_by_id) != EXPECTED_CALL_COUNT or None in cell_by_id:
        raise BehavioralDeconfoundingRunnerError("plan cell IDs changed")
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise BehavioralDeconfoundingRunnerError("planned call is not an object")
        _exact_keys(call, CALL_KEYS, "planned call")
        core = {key: call[key] for key in CALL_KEYS - {"call_id"}}
        if canonical_sha256(core) != call["call_id"]:
            raise BehavioralDeconfoundingRunnerError("planned call ID changed")
        if call["planned_index"] != index or call["full_vocab_logits_row"] != index:
            raise BehavioralDeconfoundingRunnerError("planned call order changed")
        cell = cell_by_id.get(call["cell_id"])
        if cell is None or canonical_sha256(cell) != call["fixture_cell_sha256"]:
            raise BehavioralDeconfoundingRunnerError(
                "planned call no longer binds its fixture cell"
            )
        expected_cell_fields = {
            "world_id": cell["world_id"],
            "family_id": _family_id(cell),
            "stratum_id": _stratum_id(cell),
            **{field: cell.get(field) for field in FACTOR_FIELDS},
            "answer_labels": _answer_labels(cell),
            "correct_answer": cell["correct_answer"],
            "correct_option_position": cell["correct_option_position"],
            "displayed_options": cell["displayed_options"],
            "rule_lines": cell["rule_lines"],
            "prompt_lines": cell["prompt_lines"],
            "prompt_text": cell["prompt_text"],
            "prompt_text_sha256": cell["prompt_sha256"],
        }
        if any(call[field] != expected for field, expected in expected_cell_fields.items()):
            raise BehavioralDeconfoundingRunnerError(
                "planned call semantic fields differ from its fixture cell"
            )
        expected_messages = [
            {"role": "system", "content": plan["neutral_system_message"]},
            {"role": "user", "content": call["prompt_text"]},
        ]
        if (
            call["messages"] != expected_messages
            or call["messages_sha256"] != canonical_sha256(expected_messages)
            or call["neutral_system_message_sha256"]
            != text_sha256(plan["neutral_system_message"])
            or call["rendered_text_sha256"] != text_sha256(call["rendered_text"])
        ):
            raise BehavioralDeconfoundingRunnerError(
                "planned prompt or message digest changed"
            )
        if execution_input_sha256(
            call["execution_input_ids"], call["execution_attention_mask"]
        ) != call["execution_input_sha256"]:
            raise BehavioralDeconfoundingRunnerError(
                "planned execution input hash changed"
            )
        if call["input_token_count"] != len(call["execution_input_ids"]):
            raise BehavioralDeconfoundingRunnerError("planned token count changed")
        if call["final_attended_token_index"] != call["input_token_count"] - 1:
            raise BehavioralDeconfoundingRunnerError(
                "planned final attended token index changed"
            )
        if call["execution_attention_mask"] != [1] * call["input_token_count"]:
            raise BehavioralDeconfoundingRunnerError("planned attention mask changed")
        if call["label_token_ids"] != [
            EXPECTED_LABEL_TOKEN_IDS[label] for label in call["answer_labels"]
        ]:
            raise BehavioralDeconfoundingRunnerError("planned label-token IDs changed")
        correct_index = call["answer_labels"].index(call["correct_answer"])
        if call["correct_token_id"] != call["label_token_ids"][correct_index]:
            raise BehavioralDeconfoundingRunnerError("planned correct token changed")
        if call["incorrect_token_id"] != call["label_token_ids"][1 - correct_index]:
            raise BehavioralDeconfoundingRunnerError("planned incorrect token changed")
        expected_model_fields = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "device": DEVICE,
            "dtype": DTYPE,
            "vocab_size": MODEL_VOCAB_SIZE,
            "logits_source": LOGITS_SOURCE,
        }
        if any(call[field] != expected for field, expected in expected_model_fields.items()):
            raise BehavioralDeconfoundingRunnerError(
                "planned model/tokenizer execution lock changed"
            )
    observed_family_counts = {
        family: sum(call["family_id"] == family for call in calls)
        for family in FAMILY_ORDER
    }
    if observed_family_counts != FAMILY_COUNTS:
        raise BehavioralDeconfoundingRunnerError("planned family allocation changed")
    world_ids = {cell["world_id"] for cell in plan["cell_registry"]}
    if len(world_ids) != 8 or any(
        sum(call["world_id"] == world_id for call in calls) != 56
        for world_id in world_ids
    ):
        raise BehavioralDeconfoundingRunnerError("planned world allocation changed")
    if plan["expected_call_ids"] != [call["call_id"] for call in calls]:
        raise BehavioralDeconfoundingRunnerError("expected call-ID order changed")
    if plan["call_plan_sha256"] != canonical_sha256(calls):
        raise BehavioralDeconfoundingRunnerError("call-plan hash changed")
    core = {key: plan[key] for key in expected_keys - {"canonical_sha256"}}
    if plan["canonical_sha256"] != canonical_sha256(core):
        raise BehavioralDeconfoundingRunnerError("plan canonical hash changed")


def design_from_plan(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    validate_plan_manifest(plan)
    _validate_tokenization_receipt(receipt, plan["calls"])
    _validate_dependency_lock(dependency)
    return {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "mode": plan["mode"],
        "purpose": plan["purpose"],
        "status": "DESIGN_FROZEN_ZERO_FORWARD",
        "claim_scope": "single_model_synthetic_behavioral_failure_localization_only",
        "prohibited_claims": [
            "biology",
            "latent_knowledge",
            "activation_gap",
            "physical_law",
            "component_causality",
            "model_family_generality",
        ],
        "model": plan["model"],
        "factor_contract": plan["factor_contract"],
        "bootstrap": plan["bootstrap"],
        "expected_call_count": EXPECTED_CALL_COUNT,
        "expected_call_ids": plan["expected_call_ids"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "plan_manifest_canonical_sha256": plan["canonical_sha256"],
        "tokenization_receipt_canonical_sha256": receipt["canonical_sha256"],
        "dependency_lock_canonical_sha256": dependency["canonical_sha256"],
        "static_locks": plan["locks"],
        "implementation_locks": dependency["implementation_files"],
        "model_calls": 0,
        "generation_used": False,
        "logit_processors_used": False,
        "biological_model_calls": 0,
    }


def validate_plan_bundle(
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    receipt: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> None:
    validate_plan_manifest(plan)
    _validate_tokenization_receipt(receipt, plan["calls"])
    _validate_dependency_lock(dependency)
    expected_design = design_from_plan(plan, receipt, dependency)
    if design != expected_design:
        raise BehavioralDeconfoundingRunnerError(
            "design differs from the frozen plan bundle"
        )
    locks = plan["locks"]
    if (
        locks["tokenization_receipt_canonical_sha256"]
        != receipt["canonical_sha256"]
        or locks["dependency_lock_canonical_sha256"]
        != dependency["canonical_sha256"]
    ):
        raise BehavioralDeconfoundingRunnerError("plan cross-artifact locks changed")


def _paths(result_root: Path) -> dict[str, Path]:
    result_root = result_root.resolve()
    return {
        **{key: result_root / name for key, name in PLAN_FILENAMES.items()},
        **{key: result_root / name for key, name in BEHAVIOR_FILENAMES.items()},
    }


def _require_absent(paths: Sequence[Path], label: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise BehavioralDeconfoundingRunnerError(
            f"refusing {label}; frozen or downstream artifacts already exist: {existing}"
        )


def _with_atomic_temporaries(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        result.extend((path, path.with_name(f".{path.name}.tmp")))
    return result


def _analysis_paths(result_root: Path) -> list[Path]:
    return [result_root.resolve() / filename for filename in ANALYSIS_FILENAMES]


def run_plan(
    *,
    result_root: Path = RESULT_ROOT,
    analyzer_path: Path = DEFAULT_ANALYZER,
) -> dict[str, Any]:
    """Write the complete tokenizer-only plan; never instantiate the model."""

    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise BehavioralDeconfoundingRunnerError(
            "V4 planning requires the default frozen analyzer"
        )
    paths = _paths(result_root)
    _require_absent(
        _with_atomic_temporaries(
            [paths[key] for key in BEHAVIOR_FILENAMES]
            + _analysis_paths(result_root)
        ),
        "plan replay after execution",
    )
    tokenizer, vocab_size = _config_and_tokenizer()
    if vocab_size != MODEL_VOCAB_SIZE:
        raise BehavioralDeconfoundingRunnerError("model vocabulary lock changed")
    plan, receipt, dependency = build_plan(tokenizer, analyzer_path)
    design = design_from_plan(plan, receipt, dependency)
    validate_plan_bundle(plan, design, receipt, dependency)
    write_json(paths["tokenization_receipt"], receipt)
    write_json(paths["dependency_lock"], dependency)
    write_json(paths["plan_manifest"], plan)
    write_json(paths["design"], design)
    return {
        "status": "PLAN_FROZEN_ZERO_FORWARD",
        "result_root": str(result_root),
        "record_count": EXPECTED_CALL_COUNT,
        "call_plan_sha256": plan["call_plan_sha256"],
        "model_calls": 0,
    }


def load_and_validate_plan(
    result_root: Path = RESULT_ROOT,
) -> dict[str, dict[str, Any]]:
    _require_frozen_static_inputs()
    paths = _paths(result_root)
    plan = _load_json(paths["plan_manifest"])
    design = _load_json(paths["design"])
    receipt = _load_json(paths["tokenization_receipt"])
    dependency = _load_json(paths["dependency_lock"])
    validate_plan_bundle(plan, design, receipt, dependency)
    if dependency != _dependency_lock(DEFAULT_ANALYZER):
        raise BehavioralDeconfoundingRunnerError(
            "current dependency environment differs from the frozen lock"
        )
    fixture = load_and_rebuild_fixture()
    if fixture["cells"] != plan["cell_registry"]:
        raise BehavioralDeconfoundingRunnerError(
            "frozen plan cell registry differs from fixture"
        )
    tokenizer, vocab_size = _config_and_tokenizer()
    if vocab_size != MODEL_VOCAB_SIZE:
        raise BehavioralDeconfoundingRunnerError("model vocabulary lock changed")
    rebuilt_plan, rebuilt_receipt, rebuilt_dependency = build_plan(
        tokenizer, DEFAULT_ANALYZER
    )
    rebuilt_design = design_from_plan(
        rebuilt_plan, rebuilt_receipt, rebuilt_dependency
    )
    if (
        rebuilt_plan != plan
        or rebuilt_receipt != receipt
        or rebuilt_dependency != dependency
        or rebuilt_design != design
    ):
        raise BehavioralDeconfoundingRunnerError(
            "tokenizer-only plan reconstruction differs from frozen artifacts"
        )
    return {
        "plan_manifest": plan,
        "design": design,
        "tokenization_receipt": receipt,
        "dependency_lock": dependency,
    }


def validate_loaded_model(model: Any) -> None:
    try:
        import torch
    except ImportError as error:
        raise BehavioralDeconfoundingRunnerError(
            "torch is required for model validation"
        ) from error
    layers = model_hooks.resolve_decoder_layers(model)
    config = getattr(model, "config", None)
    if len(layers) != MODEL_LAYERS:
        raise BehavioralDeconfoundingRunnerError("decoder layer count changed")
    if getattr(config, "hidden_size", None) != MODEL_WIDTH:
        raise BehavioralDeconfoundingRunnerError("hidden width changed")
    if getattr(config, "vocab_size", None) != MODEL_VOCAB_SIZE:
        raise BehavioralDeconfoundingRunnerError("model vocabulary size changed")
    if getattr(config, "_attn_implementation", None) != ATTENTION_IMPLEMENTATION:
        raise BehavioralDeconfoundingRunnerError(
            "attention implementation changed"
        )
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or {tensor.device.type for tensor in tensors} != {DEVICE}:
        raise BehavioralDeconfoundingRunnerError(
            "loaded tensors are not all on the frozen device"
        )
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating or {tensor.dtype for tensor in floating} != {torch.float32}:
        raise BehavioralDeconfoundingRunnerError(
            "loaded floating tensors are not all float32"
        )


def _load_model() -> Any:
    _verify_cached_model_assets()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise BehavioralDeconfoundingRunnerError(
            "behavior execution requires torch and transformers"
        ) from error
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(torch.device(DEVICE)).eval()
    validate_loaded_model(model)
    return model


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise BehavioralDeconfoundingRunnerError(
            "loaded model exposes no parameter device"
        ) from error


def _expected_forward_trace(call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_calls": 1,
        "teacher_forced_prompt_forward": True,
        "generation_used": False,
        "logit_processors_used": False,
        "use_cache": False,
        "return_dict": True,
        "input_shape": [1, call["input_token_count"]],
        "attention_mask_shape": [1, call["input_token_count"]],
        "output_logits_shape": [
            1,
            call["input_token_count"],
            MODEL_VOCAB_SIZE,
        ],
        "selected_batch_index": 0,
        "selected_token_index": call["final_attended_token_index"],
        "full_vocab_row_shape": [MODEL_VOCAB_SIZE],
        "raw_logits_source": LOGITS_SOURCE,
    }


def _forward_next_logits(
    model: Any, call: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Obtain one unprocessed next-token vocabulary row; never generate."""

    input_ids = [list(call["execution_input_ids"])]
    attention_mask = [list(call["execution_attention_mask"])]
    try:
        import torch
    except ImportError:
        torch = None
    is_torch = torch is not None and isinstance(model, torch.nn.Module)
    context = torch.inference_mode() if is_torch else nullcontext()
    if is_torch:
        device = _model_device(model)
        input_value = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_value = torch.tensor(attention_mask, dtype=torch.long, device=device)
    else:
        input_value = input_ids
        mask_value = attention_mask
    with context:
        output = model(
            input_ids=input_value,
            attention_mask=mask_value,
            use_cache=False,
            return_dict=True,
        )
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    shape = getattr(logits, "shape", None)
    expected_shape = (1, call["input_token_count"], MODEL_VOCAB_SIZE)
    if shape is None or tuple(int(item) for item in shape) != expected_shape:
        raise BehavioralDeconfoundingRunnerError(
            f"model returned invalid raw logits shape: {shape}"
        )
    row = logits[0, -1, :]
    if hasattr(row, "detach"):
        row = row.detach().float().cpu().numpy()
    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise BehavioralDeconfoundingRunnerError(
            "model returned an invalid full-vocabulary row"
        )
    return value, _expected_forward_trace(call)


def full_vocab_diagnostics(
    row: np.ndarray,
    *,
    answer_labels: Sequence[str],
    label_token_ids: Sequence[int],
    correct_answer: str,
) -> dict[str, Any]:
    """Compute exact generic-label and global argmax diagnostics from raw f32."""

    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise BehavioralDeconfoundingRunnerError("vocabulary row is invalid")
    labels = list(answer_labels)
    token_ids = [int(item) for item in label_token_ids]
    if tuple(labels) not in (("X", "Y"), ("P", "Q")) or token_ids != [
        EXPECTED_LABEL_TOKEN_IDS[label] for label in labels
    ]:
        raise BehavioralDeconfoundingRunnerError("generic label registry is invalid")
    if correct_answer not in labels:
        raise BehavioralDeconfoundingRunnerError(
            "correct answer is outside the generic label registry"
        )
    label_logits = [float(value[token_id]) for token_id in token_ids]
    correct_index = labels.index(correct_answer)
    incorrect_index = 1 - correct_index
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    row64 = value.astype(np.float64)
    peak = float(row64.max())
    logsumexp = peak + math.log(float(np.exp(row64 - peak).sum()))
    label_logsumexp = float(np.logaddexp(label_logits[0], label_logits[1]))
    correct_token_id = token_ids[correct_index]
    greedy_token_id = maximum_ids[0]
    result = {
        "label_logits": label_logits,
        "label_logit_by_text": {
            label: label_logits[index] for index, label in enumerate(labels)
        },
        "first_minus_second_margin": label_logits[0] - label_logits[1],
        "correct_minus_incorrect_margin": (
            label_logits[correct_index] - label_logits[incorrect_index]
        ),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "greedy_token_id": greedy_token_id,
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "correct_is_global_maximum": correct_token_id in maximum_ids,
        "unique_global_argmax_is_correct": (
            len(maximum_ids) == 1 and maximum_ids[0] == correct_token_id
        ),
        "greedy_matches_correct": greedy_token_id == correct_token_id,
        "full_vocab_logits_sha256": hashlib.sha256(
            value.tobytes(order="C")
        ).hexdigest(),
    }
    _exact_keys(result, DIAGNOSTIC_KEYS, "full-vocabulary diagnostics")
    return result


def behavior_record(
    plan: Mapping[str, Any],
    call: Mapping[str, Any],
    row: np.ndarray,
    *,
    forward_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the exact raw record associated with one planned row."""

    trace = _expected_forward_trace(call)
    if forward_trace is not None and dict(forward_trace) != trace:
        raise BehavioralDeconfoundingRunnerError("forward trace changed")
    identity = {
        "schema_version": RECORD_SCHEMA,
        "call_id": call["call_id"],
        "call_plan_sha256": plan["call_plan_sha256"],
    }
    record = {
        "schema_version": RECORD_SCHEMA,
        "record_id": canonical_sha256(identity),
        "call_id": call["call_id"],
        "planned_index": call["planned_index"],
        "full_vocab_logits_row": call["full_vocab_logits_row"],
        "cell_id": call["cell_id"],
        "world_id": call["world_id"],
        "family_id": call["family_id"],
        "stratum_id": call["stratum_id"],
        "fixture_cell_sha256": call["fixture_cell_sha256"],
        **{field: call[field] for field in FACTOR_FIELDS},
        "answer_labels": call["answer_labels"],
        "correct_answer": call["correct_answer"],
        "correct_option_position": call["correct_option_position"],
        "displayed_options": call["displayed_options"],
        "rule_lines": call["rule_lines"],
        "prompt_lines": call["prompt_lines"],
        "prompt_text": call["prompt_text"],
        "prompt_text_sha256": call["prompt_text_sha256"],
        "rendered_text_sha256": call["rendered_text_sha256"],
        "messages_sha256": call["messages_sha256"],
        "execution_input_sha256": call["execution_input_sha256"],
        "input_token_count": call["input_token_count"],
        "label_token_ids": call["label_token_ids"],
        "correct_token_id": call["correct_token_id"],
        "incorrect_token_id": call["incorrect_token_id"],
        "diagnostics": full_vocab_diagnostics(
            row,
            answer_labels=call["answer_labels"],
            label_token_ids=call["label_token_ids"],
            correct_answer=call["correct_answer"],
        ),
        "forward_trace": trace,
        "call_plan_sha256": plan["call_plan_sha256"],
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "biological_model_calls": 0,
    }
    _exact_keys(record, RECORD_KEYS, "behavior record")
    return record


def execute_call_plan(
    model: Any, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Execute all 448 calls in their frozen order."""

    validate_plan_manifest(plan)
    if not callable(getattr(model, "eval", None)):
        raise BehavioralDeconfoundingRunnerError("model must expose eval()")
    model.eval()
    records: list[dict[str, Any]] = []
    matrix = np.empty((EXPECTED_CALL_COUNT, MODEL_VOCAB_SIZE), dtype="<f4")
    for index, call in enumerate(plan["calls"]):
        if call["planned_index"] != index:
            raise BehavioralDeconfoundingRunnerError("call-plan order changed")
        row, trace = _forward_next_logits(model, call)
        matrix[index] = row
        records.append(
            behavior_record(plan, call, matrix[index], forward_trace=trace)
        )
        if (index + 1) % 32 == 0 or index + 1 == EXPECTED_CALL_COUNT:
            print(f"behavior forward {index + 1}/{EXPECTED_CALL_COUNT}", flush=True)
    _validate_records_and_matrix(plan, records, matrix)
    return records, matrix


def _validate_records_and_matrix(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
) -> None:
    value = np.asarray(matrix)
    if value.dtype != np.dtype("<f4") or value.shape != (
        EXPECTED_CALL_COUNT,
        MODEL_VOCAB_SIZE,
    ):
        raise BehavioralDeconfoundingRunnerError(
            "full-vocabulary sidecar shape or dtype changed"
        )
    if not np.isfinite(value).all():
        raise BehavioralDeconfoundingRunnerError(
            "full-vocabulary sidecar contains non-finite values"
        )
    if len(records) != EXPECTED_CALL_COUNT:
        raise BehavioralDeconfoundingRunnerError("behavior record count changed")
    for index, (record, call) in enumerate(zip(records, plan["calls"], strict=True)):
        if not isinstance(record, Mapping):
            raise BehavioralDeconfoundingRunnerError("behavior record is not an object")
        _exact_keys(record, RECORD_KEYS, "behavior record")
        expected = behavior_record(plan, call, value[index])
        if dict(record) != expected:
            raise BehavioralDeconfoundingRunnerError(
                f"behavior record does not reconstruct from sidecar row {index}"
            )


def _attempt_payload(
    bundle: Mapping[str, Mapping[str, Any]], result_root: Path
) -> dict[str, Any]:
    paths = _paths(result_root)
    plan = bundle["plan_manifest"]
    core = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "BEHAVIOR_ATTEMPT_FROZEN_BEFORE_MODEL_LOAD",
        "phase": "behavior",
        "result_root": str(result_root.resolve()),
        "plan_manifest_file_sha256": file_sha256(paths["plan_manifest"]),
        "design_file_sha256": file_sha256(paths["design"]),
        "tokenization_receipt_file_sha256": file_sha256(
            paths["tokenization_receipt"]
        ),
        "dependency_lock_file_sha256": file_sha256(paths["dependency_lock"]),
        "plan_manifest_canonical_sha256": plan["canonical_sha256"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "dependency_lock_canonical_sha256": bundle["dependency_lock"][
            "canonical_sha256"
        ],
        "runner_sha256": file_sha256(Path(__file__)),
        "analyzer_sha256": bundle["dependency_lock"]["implementation_files"][
            "analyzer"
        ]["sha256"],
        "model": plan["model"],
        "expected_model_calls": EXPECTED_CALL_COUNT,
        "model_calls_at_attempt_write": 0,
        "generation_used": False,
        "logit_processors_used": False,
        "biological_model_calls": 0,
        "intended_outputs": {
            "records": str(paths["records"]),
            "full_vocab_logits": str(paths["full_vocab_logits"]),
            "execution_manifest": str(paths["execution_manifest"]),
        },
    }
    return {**core, "attempt_id": canonical_sha256(core)}


def _validate_attempt(
    attempt: Mapping[str, Any],
    bundle: Mapping[str, Mapping[str, Any]],
    result_root: Path,
) -> None:
    expected = _attempt_payload(bundle, result_root)
    if attempt != expected:
        raise BehavioralDeconfoundingRunnerError(
            "behavior attempt differs from its frozen pre-forward payload"
        )


def _execution_manifest_payload(
    bundle: Mapping[str, Mapping[str, Any]],
    attempt: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    result_root: Path,
) -> dict[str, Any]:
    paths = _paths(result_root)
    plan = bundle["plan_manifest"]
    return {
        "schema_version": EXECUTION_SCHEMA,
        "status": "BEHAVIOR_RAW_EXECUTION_COMPLETE_NOT_ANALYZED",
        "phase": "behavior",
        "mode": plan["mode"],
        "purpose": plan["purpose"],
        "claim_scope": bundle["design"]["claim_scope"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "plan_manifest_canonical_sha256": plan["canonical_sha256"],
        "plan_manifest_file_sha256": file_sha256(paths["plan_manifest"]),
        "design_file_sha256": file_sha256(paths["design"]),
        "tokenization_receipt_file_sha256": file_sha256(
            paths["tokenization_receipt"]
        ),
        "dependency_lock_file_sha256": file_sha256(paths["dependency_lock"]),
        "attempt_id": attempt["attempt_id"],
        "attempt_file_sha256": file_sha256(paths["attempt"]),
        "model": plan["model"],
        "execution": {
            "model_calls": EXPECTED_CALL_COUNT,
            "generation_used": False,
            "logit_processors_used": False,
            "teacher_forced_prompt_forwards": EXPECTED_CALL_COUNT,
            "use_cache": False,
            "return_dict": True,
            "logits_source": LOGITS_SOURCE,
            "biological_model_calls": 0,
        },
        "records": {
            "path": str(paths["records"]),
            "file_sha256": file_sha256(paths["records"]),
            "record_count": len(records),
            "record_ids": [record["record_id"] for record in records],
            "call_ids": [record["call_id"] for record in records],
            "canonical_sha256": canonical_sha256(list(records)),
        },
        "full_vocab_logits": {
            "path": str(paths["full_vocab_logits"]),
            "file_sha256": file_sha256(paths["full_vocab_logits"]),
            "logical_f32_sha256": f32_sha256(matrix),
            "dtype": "float32",
            "byte_order": "little",
            "shape": [EXPECTED_CALL_COUNT, MODEL_VOCAB_SIZE],
            "row_indices": list(range(EXPECTED_CALL_COUNT)),
            "row_call_ids": [record["call_id"] for record in records],
            "row_record_ids": [record["record_id"] for record in records],
            "row_sha256": [
                record["diagnostics"]["full_vocab_logits_sha256"]
                for record in records
            ],
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
    }


def _load_f32_sidecar(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BehavioralDeconfoundingRunnerError(
            f"cannot load full-vocabulary sidecar: {path}"
        ) from error
    if value.dtype != np.dtype("<f4"):
        raise BehavioralDeconfoundingRunnerError(
            "full-vocabulary sidecar on-disk dtype is not little-endian float32"
        )
    array = np.ascontiguousarray(value)
    if array.shape != (EXPECTED_CALL_COUNT, MODEL_VOCAB_SIZE) or not np.isfinite(
        array
    ).all():
        raise BehavioralDeconfoundingRunnerError(
            "full-vocabulary sidecar shape or finiteness changed"
        )
    return array


def validate_behavior_artifacts(
    result_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    """Load and independently reconstruct every frozen execution artifact."""

    bundle: dict[str, Any] = dict(load_and_validate_plan(result_root))
    paths = _paths(result_root)
    attempt = _load_json(paths["attempt"])
    _validate_attempt(attempt, bundle, result_root)
    records = load_jsonl(paths["records"])
    matrix = _load_f32_sidecar(paths["full_vocab_logits"])
    _validate_records_and_matrix(bundle["plan_manifest"], records, matrix)
    manifest = _load_json(paths["execution_manifest"])
    expected_manifest = _execution_manifest_payload(
        bundle, attempt, records, matrix, result_root
    )
    if manifest != expected_manifest:
        raise BehavioralDeconfoundingRunnerError(
            "behavior execution manifest does not reconstruct from artifacts"
        )
    bundle.update(
        {
            "attempt": attempt,
            "records": records,
            "execution_manifest": manifest,
            "full_vocab_logits": matrix,
        }
    )
    return bundle


def run_behavior(*, result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    """Execute the frozen behavior bank exactly once."""

    paths = _paths(result_root)
    _require_absent(
        _with_atomic_temporaries(
            [paths[key] for key in BEHAVIOR_FILENAMES]
            + _analysis_paths(result_root)
        ),
        "behavior re-entry",
    )
    bundle = load_and_validate_plan(result_root)
    attempt = _attempt_payload(bundle, result_root)
    write_json(paths["attempt"], attempt)
    model = _load_model()
    records, matrix = execute_call_plan(model, bundle["plan_manifest"])
    _validate_records_and_matrix(bundle["plan_manifest"], records, matrix)
    write_jsonl(paths["records"], records)
    write_f32_sidecar(paths["full_vocab_logits"], matrix)
    manifest = _execution_manifest_payload(
        bundle, attempt, records, matrix, result_root
    )
    write_json(paths["execution_manifest"], manifest)
    del model
    try:
        import torch

        torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass
    validated = validate_behavior_artifacts(result_root)
    if validated["execution_manifest"]["status"] != (
        "BEHAVIOR_RAW_EXECUTION_COMPLETE_NOT_ANALYZED"
    ):
        raise BehavioralDeconfoundingRunnerError(
            "completed behavior execution failed terminal validation"
        )
    return {
        "status": manifest["status"],
        "result_root": str(result_root),
        "record_count": len(records),
        "model_calls": EXPECTED_CALL_COUNT,
        "call_plan_sha256": bundle["plan_manifest"]["call_plan_sha256"],
        "biological_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "behavior"), required=True)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    args = parser.parse_args()
    try:
        if args.phase == "plan":
            result = run_plan(
                result_root=args.result_root,
                analyzer_path=args.analyzer,
            )
        else:
            if args.analyzer.resolve() != DEFAULT_ANALYZER.resolve():
                raise BehavioralDeconfoundingRunnerError(
                    "behavior execution forbids a custom analyzer"
                )
            result = run_behavior(result_root=args.result_root)
    except BehavioralDeconfoundingRunnerError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

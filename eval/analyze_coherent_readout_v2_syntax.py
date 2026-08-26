"""Analyze the frozen non-biological coherent-readout v2 syntax bank.

The analyzer independently reconstructs every retained quantity from the mandatory
little-endian float32 full-vocabulary sidecar.  It validates the exact 256-request
topology, applies the preregistered fixed-bank gates, and gives selection authority
only to the frozen ``syntax_selection`` model role.  It never analyzes biology,
knowledge, or activation claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import coherent_binary_readout as coherent
    from . import run_coherent_readout_v2_syntax as runner
except ImportError:  # pragma: no cover - direct execution from eval/
    import coherent_binary_readout as coherent
    import run_coherent_readout_v2_syntax as runner


ANALYSIS_SCHEMA = "coherent-readout-v2-syntax-analysis-v1"
SELECTED_CANDIDATE_SCHEMA = "coherent-readout-v2-selected-candidate-v1"

DESIGN_KEYS = {
    "schema_version",
    "mode",
    "purpose",
    "analysis_id",
    "model_role",
    "selection_eligible_model",
    "source_fixture_sha256",
    "source_manifest_sha256",
    "source_fixture_canonical_sha256",
    "preregistration_sha256",
    "expected_runner_code_sha256",
    "expected_analyzer_code_sha256",
    "expected_environment",
    "candidate_bank_sha256",
    "candidate_registry",
    "pair_registry",
    "expected_item_ids",
    "expected_model_id",
    "expected_model_revision",
    "expected_model_weights_sha256",
    "expected_tokenizer_id",
    "expected_tokenizer_revision",
    "expected_tokenizer_vocab_size",
    "expected_chat_template_sha256",
    "expected_dtype",
    "expected_device",
    "expected_vocab_size",
    "gate_config",
    "expected_record_count",
    "expected_record_ids",
    "call_plan_sha256",
    "confirmatory_execution_allowed",
    "claim_scope",
}

FORM_IDS = tuple(f"{order}::{mapping}" for order, mapping in runner.FORM_KEYS)
TRUTH_POLARITIES = ("negative", "positive")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = PROJECT_ROOT / "signal/syntax/coherent_readout_v2_syntax_bank.json"
FROZEN_MANIFEST_PATH = (
    PROJECT_ROOT / "signal/syntax/coherent_readout_v2_syntax_bank.manifest.json"
)
FROZEN_PREREG_PATH = PROJECT_ROOT / "docs/COHERENT_READOUT_V2_SYNTAX_SELECTION_PREREG.md"


class SyntaxAnalysisError(ValueError):
    """Raised when a syntax artifact violates the frozen analysis contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SyntaxAnalysisError(
            f"{label} schema mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntaxAnalysisError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SyntaxAnalysisError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    output = _require_string(value, label)
    if len(output) != 64 or any(character not in "0123456789abcdef" for character in output):
        raise SyntaxAnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return output


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SyntaxAnalysisError(f"{label} must be an integer")
    return int(value)


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SyntaxAnalysisError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise SyntaxAnalysisError(f"{label} must be finite")
    return output


@lru_cache(maxsize=1)
def _frozen_fixture_contract() -> dict[str, Any]:
    frozen_files = (
        (FROZEN_FIXTURE_PATH, runner.FROZEN_FIXTURE_SHA256, "fixture"),
        (FROZEN_MANIFEST_PATH, runner.FROZEN_FIXTURE_MANIFEST_SHA256, "fixture manifest"),
        (FROZEN_PREREG_PATH, runner.FROZEN_PREREG_SHA256, "preregistration"),
    )
    for path, expected_sha256, label in frozen_files:
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise SyntaxAnalysisError(f"frozen {label} bytes are unavailable or changed")
    try:
        fixture = json.loads(FROZEN_FIXTURE_PATH.read_text(encoding="utf-8"))
        return runner.validate_fixture(fixture)
    except (json.JSONDecodeError, OSError, runner.SyntaxRunnerError) as error:
        raise SyntaxAnalysisError("frozen syntax fixture is invalid") from error


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = dict(_require_mapping(value, "expected_environment"))
    expected_keys = {
        "python_implementation",
        "python_version",
        "platform",
        "dependencies",
        "execution_source_sha256",
        "environment_sha256",
    }
    _exact_keys(environment, expected_keys, "expected_environment")
    observed_sha256 = _require_sha256(
        environment.pop("environment_sha256"), "environment_sha256"
    )
    for key in ("python_implementation", "python_version", "platform"):
        _require_string(environment[key], f"expected_environment.{key}")
    dependencies = _require_mapping(
        environment["dependencies"], "expected_environment.dependencies"
    )
    dependency_names = {
        "huggingface-hub",
        "numpy",
        "scipy",
        "torch",
        "transformers",
    }
    _exact_keys(dependencies, dependency_names, "expected_environment.dependencies")
    for dependency, version in dependencies.items():
        _require_string(dependency, "expected_environment dependency name")
        _require_string(version, f"expected_environment dependency {dependency}")
    execution_sources = _require_mapping(
        environment["execution_source_sha256"],
        "expected_environment.execution_source_sha256",
    )
    _exact_keys(
        execution_sources,
        {
            "causal_intervention",
            "coherent_binary_readout",
            "run_coherent_binary_readout",
        },
        "expected_environment.execution_source_sha256",
    )
    for source, digest in execution_sources.items():
        _require_sha256(digest, f"execution source {source}")
    if canonical_sha256(environment) != observed_sha256:
        raise SyntaxAnalysisError("expected environment digest is inconsistent")
    environment["environment_sha256"] = observed_sha256
    return environment


def _validate_candidate_registry(
    value: Any, *, vocab_size: int, tokenizer_vocab_size: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(runner.CANDIDATE_ORDER):
        raise SyntaxAnalysisError("candidate registry must contain four candidates")
    definitions = runner.candidate_definitions()
    normalized: list[dict[str, Any]] = []
    for index, (observed, definition) in enumerate(zip(value, definitions, strict=True)):
        row = dict(_require_mapping(observed, f"candidate registry row {index}"))
        expected_keys = {*definition, "candidate_definition_sha256", "x_token_id", "y_token_id"}
        _exact_keys(row, expected_keys, f"candidate registry row {index}")
        for key, expected in definition.items():
            if canonical_json(row[key]) != canonical_json(expected):
                raise SyntaxAnalysisError(
                    f"candidate registry changed frozen field {definition['candidate_id']}::{key}"
                )
        expected_digest = runner.canonical_sha256(definition)
        if row["candidate_definition_sha256"] != expected_digest:
            raise SyntaxAnalysisError("candidate definition digest changed")
        x_token_id = _require_int(row["x_token_id"], "candidate x_token_id")
        y_token_id = _require_int(row["y_token_id"], "candidate y_token_id")
        if x_token_id == y_token_id or min(x_token_id, y_token_id) < 0:
            raise SyntaxAnalysisError("candidate answer token IDs must be distinct and nonnegative")
        if max(x_token_id, y_token_id) >= min(vocab_size, tokenizer_vocab_size):
            raise SyntaxAnalysisError("candidate answer token ID lies outside the frozen vocabulary")
        normalized.append({**row, "x_token_id": x_token_id, "y_token_id": y_token_id})
    if [row["candidate_id"] for row in normalized] != list(runner.CANDIDATE_ORDER):
        raise SyntaxAnalysisError("candidate registry order changed")
    return normalized


def _validate_pair_registry(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != 8:
        raise SyntaxAnalysisError("pair registry must contain eight pairs")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        row = dict(_require_mapping(raw, f"pair registry row {index}"))
        _exact_keys(row, runner.PAIR_KEYS, f"pair registry row {index}")
        pair_id = _require_string(row["pair_id"], "pair_id")
        if pair_id in seen or row["cluster_id"] != pair_id:
            raise SyntaxAnalysisError("pair/cluster registry must be one-to-one")
        seen.add(pair_id)
        positive = _require_string(row["positive_class"], "positive_class")
        negative = _require_string(row["negative_class"], "negative_class")
        if positive == negative:
            raise SyntaxAnalysisError("pair classes must be distinct")
        output.append(
            {
                "pair_id": pair_id,
                "cluster_id": pair_id,
                "positive_class": positive,
                "negative_class": negative,
            }
        )
    return output


def validate_design(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every frozen design dependency before reading outcomes."""

    design = dict(_require_mapping(value, "syntax design"))
    _exact_keys(design, DESIGN_KEYS, "syntax design")
    if design["schema_version"] != runner.DESIGN_SCHEMA:
        raise SyntaxAnalysisError("unsupported syntax design schema")
    if design["mode"] != "development" or design["purpose"] != "syntax_selection_only":
        raise SyntaxAnalysisError("syntax design mode or purpose changed")
    role = design["model_role"]
    if role not in runner.FROZEN_ROLE_LOCKS:
        raise SyntaxAnalysisError("syntax design has an unknown model role")
    selection_eligible = role == "syntax_selection"
    if design["selection_eligible_model"] is not selection_eligible:
        raise SyntaxAnalysisError("selection authority disagrees with the frozen model role")
    role_lock = runner.FROZEN_ROLE_LOCKS[role]
    if any(
        design[field] != role_lock[lock]
        for field, lock in (
            ("expected_model_id", "model_id"),
            ("expected_model_revision", "model_revision"),
            ("expected_model_weights_sha256", "model_weights_sha256"),
        )
    ):
        raise SyntaxAnalysisError("model identity differs from the frozen role lock")
    if design["expected_tokenizer_id"] != design["expected_model_id"] or design[
        "expected_tokenizer_revision"
    ] != design["expected_model_revision"]:
        raise SyntaxAnalysisError("tokenizer ID/revision must equal the model lock")
    if design["expected_dtype"] != "float32" or design["expected_device"] != "mps":
        raise SyntaxAnalysisError("syntax design must use frozen float32 MPS execution")
    if design["source_fixture_sha256"] != runner.FROZEN_FIXTURE_SHA256:
        raise SyntaxAnalysisError("design fixture digest differs from the preregistration")
    if design["source_manifest_sha256"] != runner.FROZEN_FIXTURE_MANIFEST_SHA256:
        raise SyntaxAnalysisError("design fixture-manifest digest differs from the preregistration")
    if design["preregistration_sha256"] != runner.FROZEN_PREREG_SHA256:
        raise SyntaxAnalysisError("design preregistration digest differs from the runner lock")
    frozen_fixture = _frozen_fixture_contract()
    if design["analysis_id"] != frozen_fixture["analysis_id"]:
        raise SyntaxAnalysisError("analysis ID differs from the frozen syntax fixture")
    if (
        design["source_fixture_canonical_sha256"]
        != frozen_fixture["fixture_canonical_sha256"]
    ):
        raise SyntaxAnalysisError("design canonical fixture digest differs from the frozen bank")
    if design["expected_runner_code_sha256"] != file_sha256(Path(runner.__file__)):
        raise SyntaxAnalysisError("runner changed after syntax plan freezing")
    if design["expected_analyzer_code_sha256"] != file_sha256(Path(__file__)):
        raise SyntaxAnalysisError("analyzer differs from the code hash frozen in the plan")
    environment = _validate_environment(design["expected_environment"])
    if canonical_json(environment) != canonical_json(runner.environment_lock()):
        raise SyntaxAnalysisError("analysis environment differs from the frozen execution lock")
    vocab_size = _require_int(design["expected_vocab_size"], "expected_vocab_size")
    tokenizer_vocab_size = _require_int(
        design["expected_tokenizer_vocab_size"], "expected_tokenizer_vocab_size"
    )
    if min(vocab_size, tokenizer_vocab_size) < 2:
        raise SyntaxAnalysisError("frozen vocabulary sizes are invalid")
    candidate_registry = _validate_candidate_registry(
        design["candidate_registry"],
        vocab_size=vocab_size,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )
    if design["candidate_bank_sha256"] != runner.candidate_bank_sha256():
        raise SyntaxAnalysisError("candidate bank changed after preregistration")
    pair_registry = _validate_pair_registry(design["pair_registry"])
    if canonical_json(pair_registry) != canonical_json(frozen_fixture["pair_registry"]):
        raise SyntaxAnalysisError("pair registry differs from the frozen syntax fixture")
    item_ids = design["expected_item_ids"]
    if (
        not isinstance(item_ids, list)
        or len(item_ids) != 16
        or item_ids != sorted(set(item_ids))
        or any(not isinstance(item, str) or not item for item in item_ids)
    ):
        raise SyntaxAnalysisError("expected item registry must be 16 sorted unique IDs")
    frozen_item_ids = [item["item_id"] for item in frozen_fixture["items"]]
    if item_ids != frozen_item_ids:
        raise SyntaxAnalysisError("item registry differs from the frozen syntax fixture")
    if canonical_json(design["gate_config"]) != canonical_json(runner.default_gate_config()):
        raise SyntaxAnalysisError("gate configuration differs from the preregistration")
    expected_record_count = _require_int(
        design["expected_record_count"], "expected_record_count"
    )
    expected_record_ids = design["expected_record_ids"]
    if (
        expected_record_count != 256
        or not isinstance(expected_record_ids, list)
        or len(expected_record_ids) != 256
        or expected_record_ids != sorted(set(expected_record_ids))
    ):
        raise SyntaxAnalysisError("expected record registry must contain 256 unique IDs")
    for digest, label in (
        (design["source_fixture_canonical_sha256"], "source fixture canonical digest"),
        (design["expected_chat_template_sha256"], "chat-template digest"),
        (design["call_plan_sha256"], "call-plan digest"),
    ):
        _require_sha256(digest, label)
    if design["confirmatory_execution_allowed"] is not False:
        raise SyntaxAnalysisError("syntax design cannot authorize confirmation")
    if design["claim_scope"] != "syntax_selection_only_no_biology_knowledge_or_activation_claim":
        raise SyntaxAnalysisError("syntax design claim boundary changed")
    return {
        **design,
        "expected_environment": environment,
        "candidate_registry": candidate_registry,
        "pair_registry": pair_registry,
        "expected_vocab_size": vocab_size,
        "expected_tokenizer_vocab_size": tokenizer_vocab_size,
        "expected_record_count": expected_record_count,
        "expected_item_ids": list(item_ids),
        "expected_record_ids": list(expected_record_ids),
    }


def _candidate_by_id(design: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in design["candidate_registry"]}


def _pair_by_id(design: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {row["pair_id"]: row for row in design["pair_registry"]}


def _validate_record_identity(
    raw: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    record = dict(_require_mapping(raw, "syntax record"))
    _exact_keys(record, runner.RECORD_KEYS, "syntax record")
    if record["schema_version"] != runner.RECORD_SCHEMA:
        raise SyntaxAnalysisError("unsupported syntax record schema")
    if runner.record_id(record) != record["record_id"]:
        raise SyntaxAnalysisError("record ID does not bind the frozen request identity")
    candidate = _candidate_by_id(design).get(record["candidate_id"])
    if candidate is None:
        raise SyntaxAnalysisError("record contains an unknown syntax candidate")
    candidate_fields = {
        "candidate_rank": "candidate_rank",
        "candidate_definition_sha256": "candidate_definition_sha256",
        "render_mode": "render_mode",
        "add_generation_prompt": "add_generation_prompt",
        "continue_final_message": "continue_final_message",
        "enable_thinking": "enable_thinking",
        "x_answer_text": "x_answer_text",
        "y_answer_text": "y_answer_text",
        "x_token_id": "x_token_id",
        "y_token_id": "y_token_id",
    }
    if any(
        canonical_json(record[field]) != canonical_json(candidate[key])
        for field, key in candidate_fields.items()
    ):
        raise SyntaxAnalysisError("record candidate definition differs from the design")
    pair = _pair_by_id(design).get(record["pair_id"])
    if pair is None or record["cluster_id"] != record["pair_id"]:
        raise SyntaxAnalysisError("record pair/cluster identity changed")
    if any(record[key] != pair[key] for key in ("positive_class", "negative_class")):
        raise SyntaxAnalysisError("record class orientation differs from its pair registry")
    polarity = record["truth_polarity"]
    if polarity not in TRUTH_POLARITIES:
        raise SyntaxAnalysisError("record truth polarity is invalid")
    if record["declared_class"] != record[f"{polarity}_class"]:
        raise SyntaxAnalysisError("record declared class disagrees with truth polarity")
    if record["item_id"] not in design["expected_item_ids"]:
        raise SyntaxAnalysisError("record item is outside the frozen item registry")
    frozen_item = next(
        item
        for item in _frozen_fixture_contract()["items"]
        if item["item_id"] == record["item_id"]
    )
    frozen_item_fields = (
        "pair_id",
        "cluster_id",
        "fixture_item_sha256",
        "declaration_sha256",
        "positive_class",
        "negative_class",
        "declared_class",
        "truth_polarity",
    )
    if any(
        canonical_json(record[field]) != canonical_json(frozen_item[field])
        for field in frozen_item_fields
    ):
        raise SyntaxAnalysisError("record item identity differs from the frozen syntax fixture")
    form = (record["order"], record["mapping"])
    if form not in runner.FORM_KEYS:
        raise SyntaxAnalysisError("record contains an unknown order/remapping form")
    messages = runner.render_candidate_messages(
        record["candidate_id"], frozen_item, record["order"], record["mapping"]
    )
    user_content = next(
        message["content"] for message in messages if message["role"] == "user"
    )
    if record["messages_sha256"] != canonical_sha256(messages):
        raise SyntaxAnalysisError("record messages differ from the frozen syntax request")
    if record["user_content_sha256"] != runner.text_sha256(user_content):
        raise SyntaxAnalysisError("record user content differs from the frozen syntax request")
    x_token_id = _require_int(record["x_token_id"], "record x_token_id")
    y_token_id = _require_int(record["y_token_id"], "record y_token_id")
    positive_token_id, negative_token_id = (
        (x_token_id, y_token_id)
        if record["mapping"] == "positive_is_x"
        else (y_token_id, x_token_id)
    )
    if record["positive_token_id"] != positive_token_id or record[
        "negative_token_id"
    ] != negative_token_id:
        raise SyntaxAnalysisError("record remapping token alignment is invalid")
    correct_token_id, wrong_token_id = (
        (positive_token_id, negative_token_id)
        if polarity == "positive"
        else (negative_token_id, positive_token_id)
    )
    if record["correct_token_id"] != correct_token_id or record[
        "wrong_token_id"
    ] != wrong_token_id:
        raise SyntaxAnalysisError("record truth-aligned answer tokens are invalid")
    locked_fields = {
        "model_role": design["model_role"],
        "model_id": design["expected_model_id"],
        "model_revision": design["expected_model_revision"],
        "model_weights_sha256": design["expected_model_weights_sha256"],
        "tokenizer_id": design["expected_tokenizer_id"],
        "tokenizer_revision": design["expected_tokenizer_revision"],
        "chat_template_sha256": design["expected_chat_template_sha256"],
        "dtype": design["expected_dtype"],
        "device": design["expected_device"],
        "vocab_size": design["expected_vocab_size"],
        "runner_code_sha256": design["expected_runner_code_sha256"],
        "analyzer_code_sha256": design["expected_analyzer_code_sha256"],
        "environment_sha256": design["expected_environment"]["environment_sha256"],
        "logits_source": runner.LOGITS_SOURCE,
    }
    if any(record[key] != expected for key, expected in locked_fields.items()):
        raise SyntaxAnalysisError("record runtime/model identity differs from the design")
    for field in (
        "record_id",
        "candidate_definition_sha256",
        "fixture_item_sha256",
        "declaration_sha256",
        "messages_sha256",
        "user_content_sha256",
        "prompt_sha256",
        "execution_input_sha256",
        "model_weights_sha256",
        "chat_template_sha256",
        "runner_code_sha256",
        "analyzer_code_sha256",
        "environment_sha256",
        "full_vocab_logits_sha256",
        "forward_trace_sha256",
    ):
        _require_sha256(record[field], field)
    input_token_count = _require_int(record["input_token_count"], "input_token_count")
    sidecar_row = _require_int(record["full_vocab_logits_row"], "full_vocab_logits_row")
    if input_token_count <= 0 or not 0 <= sidecar_row < 256:
        raise SyntaxAnalysisError("record token count or sidecar row is invalid")
    maximum_token_ids = record["maximum_token_ids"]
    if (
        not isinstance(maximum_token_ids, list)
        or not maximum_token_ids
        or maximum_token_ids != sorted(set(maximum_token_ids))
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < design["expected_vocab_size"]
            for token in maximum_token_ids
        )
    ):
        raise SyntaxAnalysisError("record maximum-token registry is invalid")
    maximum_tie_count = _require_int(record["maximum_tie_count"], "maximum_tie_count")
    if maximum_tie_count != len(maximum_token_ids):
        raise SyntaxAnalysisError("record maximum tie count is inconsistent")
    for field in ("x_logit", "y_logit", "full_vocab_logsumexp", "greedy_logit"):
        _require_finite(record[field], field)
    greedy_token_id = _require_int(record["greedy_token_id"], "greedy_token_id")
    if not 0 <= greedy_token_id < design["expected_vocab_size"]:
        raise SyntaxAnalysisError("record greedy token is outside the vocabulary")
    return record


def _validate_topology(
    records: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> None:
    if len(records) != 256:
        raise SyntaxAnalysisError("syntax analysis requires exactly 256 records")
    observed_ids = sorted(record["record_id"] for record in records)
    if observed_ids != design["expected_record_ids"]:
        missing = sorted(set(design["expected_record_ids"]) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(design["expected_record_ids"]))
        raise SyntaxAnalysisError(
            f"frozen record registry mismatch: missing={missing}, unexpected={extra}"
        )
    if runner.call_plan_sha256(records) != design["call_plan_sha256"]:
        raise SyntaxAnalysisError("records do not reproduce the frozen call-plan digest")
    expected_topology = {
        (candidate, item, order, mapping)
        for candidate in runner.CANDIDATE_ORDER
        for item in design["expected_item_ids"]
        for order, mapping in runner.FORM_KEYS
    }
    observed_topology = {
        (record["candidate_id"], record["item_id"], record["order"], record["mapping"])
        for record in records
    }
    if observed_topology != expected_topology:
        raise SyntaxAnalysisError("candidate/item/form topology is not the exact Cartesian product")
    if len(observed_topology) != len(records):
        raise SyntaxAnalysisError("candidate/item/form topology contains duplicates")
    item_identity_fields = (
        "pair_id",
        "cluster_id",
        "fixture_item_sha256",
        "declaration_sha256",
        "positive_class",
        "negative_class",
        "declared_class",
        "truth_polarity",
    )
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_candidate_item: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_item_form: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_item[record["item_id"]].append(record)
        by_candidate_item[(record["candidate_id"], record["item_id"])].append(record)
        by_item_form[(record["item_id"], record["order"], record["mapping"])].append(record)
    pair_truth_coverage: set[tuple[str, str]] = set()
    for item_id, rows in by_item.items():
        if len(rows) != 16:
            raise SyntaxAnalysisError(f"item {item_id} must contain 16 candidate/form rows")
        if any(len({row[field] for row in rows}) != 1 for field in item_identity_fields):
            raise SyntaxAnalysisError(f"item {item_id} identity changes across candidates/forms")
        pair_truth_coverage.add((rows[0]["pair_id"], rows[0]["truth_polarity"]))
    expected_pair_truth = {
        (pair["pair_id"], truth)
        for pair in design["pair_registry"]
        for truth in TRUTH_POLARITIES
    }
    if pair_truth_coverage != expected_pair_truth:
        raise SyntaxAnalysisError("item registry does not contain both truths for every pair")
    for key, rows in by_candidate_item.items():
        if len(rows) != 4 or {(row["order"], row["mapping"]) for row in rows} != set(
            runner.FORM_KEYS
        ):
            raise SyntaxAnalysisError(f"candidate/item group {key} lacks the four forms")
        for hash_field in (
            "messages_sha256",
            "user_content_sha256",
            "prompt_sha256",
            "execution_input_sha256",
            "forward_trace_sha256",
        ):
            if len({row[hash_field] for row in rows}) != 4:
                raise SyntaxAnalysisError(f"candidate/item group {key} mixes {hash_field}")
    for key, rows in by_item_form.items():
        if len(rows) != 4 or len({row["candidate_id"] for row in rows}) != 4:
            raise SyntaxAnalysisError(f"item/form group {key} mixes syntax candidates")
        if len({row["prompt_sha256"] for row in rows}) != 4:
            raise SyntaxAnalysisError(f"item/form group {key} reuses a candidate prompt")


def _verify_sidecar(
    records: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    full_vocab_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    matrix = np.asarray(full_vocab_logits)
    if matrix.dtype != np.dtype("<f4") or matrix.ndim != 2:
        raise SyntaxAnalysisError("sidecar must be a two-dimensional little-endian float32 matrix")
    expected_shape = (256, design["expected_vocab_size"])
    if matrix.shape != expected_shape:
        raise SyntaxAnalysisError(
            f"sidecar shape mismatch: expected {expected_shape}, observed {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise SyntaxAnalysisError("sidecar contains nonfinite logits")
    sidecar_rows = [record["full_vocab_logits_row"] for record in records]
    if sorted(sidecar_rows) != list(range(256)):
        raise SyntaxAnalysisError("sidecar row registry must be a permutation of 0..255")
    reconstructed: dict[str, np.ndarray] = {}
    for record in records:
        row = matrix[record["full_vocab_logits_row"]]
        diagnostics = coherent.full_vocab_diagnostics(
            row,
            x_token_id=record["x_token_id"],
            y_token_id=record["y_token_id"],
        )
        for field in ("x_logit", "y_logit", "greedy_logit", "full_vocab_logsumexp"):
            if float(record[field]) != float(diagnostics[field]):
                raise SyntaxAnalysisError(f"sidecar does not reconstruct record field {field}")
        if record["greedy_token_id"] != diagnostics["greedy_token_id"]:
            raise SyntaxAnalysisError("sidecar does not reconstruct greedy_token_id")
        if record["full_vocab_logits_sha256"] != diagnostics["full_vocab_logits_sha256"]:
            raise SyntaxAnalysisError("sidecar row digest differs from the raw record")
        maximum = float(np.max(row))
        maximum_token_ids = [int(token) for token in np.flatnonzero(row == maximum)]
        if record["maximum_token_ids"] != maximum_token_ids or record[
            "maximum_tie_count"
        ] != len(maximum_token_ids):
            raise SyntaxAnalysisError("sidecar does not reconstruct the exact argmax tie set")
        if record["forward_trace_sha256"] != runner.forward_trace_sha256(record):
            raise SyntaxAnalysisError("forward trace does not bind the sidecar row identity")
        reconstructed[record["record_id"]] = row
    return matrix, reconstructed


def _form_id(record: Mapping[str, Any]) -> str:
    return f"{record['order']}::{record['mapping']}"


def _score_rows(
    records: Sequence[Mapping[str, Any]], rows: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        row = rows[record["record_id"]]
        correct_logit = float(row[record["correct_token_id"]])
        wrong_logit = float(row[record["wrong_token_id"]])
        energy = correct_logit - wrong_logit
        score = math.tanh(energy / 2.0)
        maximum_ids = record["maximum_token_ids"]
        unique_maximum = len(maximum_ids) == 1
        maximum_token_id = maximum_ids[0] if unique_maximum else None
        native = bool(
            unique_maximum
            and maximum_token_id in {record["x_token_id"], record["y_token_id"]}
        )
        native_correct = bool(unique_maximum and maximum_token_id == record["correct_token_id"])
        x_logit = float(row[record["x_token_id"]])
        y_logit = float(row[record["y_token_id"]])
        two_token_logsumexp = float(np.logaddexp(x_logit, y_logit))
        maximum = float(np.max(row))
        full_vocab_logsumexp = maximum + math.log(
            float(np.exp(row.astype(np.float64) - maximum).sum())
        )
        label_mass = math.exp(two_token_logsumexp - full_vocab_logsumexp)
        if not math.isfinite(label_mass) or label_mass > 1.0 + 1e-12:
            raise SyntaxAnalysisError("sidecar implies invalid two-token probability mass")
        output.append(
            {
                "record_id": record["record_id"],
                "candidate_id": record["candidate_id"],
                "item_id": record["item_id"],
                "pair_id": record["pair_id"],
                "truth_polarity": record["truth_polarity"],
                "order": record["order"],
                "mapping": record["mapping"],
                "form_id": _form_id(record),
                "energy": energy,
                "s": score,
                "native": native,
                "native_correct": native_correct,
                "native_wrong": bool(native and not native_correct),
                "maximum_tie_count": record["maximum_tie_count"],
                "label_mass": min(1.0, label_mass),
            }
        )
    return output


def _pair_equivalence(values: Mapping[str, float], margin: float, alpha: float) -> dict[str, Any]:
    pair_ids = sorted(values)
    summary = coherent.equivalence_summary(
        [values[pair_id] for pair_id in pair_ids], margin=margin, alpha=alpha
    )
    summary["n_pairs"] = summary.pop("n_donors")
    summary["leave_one_pair_out_means"] = summary.pop("leave_one_donor_out_means")
    return {"pair_values": {pair_id: values[pair_id] for pair_id in pair_ids}, **summary}


def _candidate_metrics(
    candidate_id: str,
    scored_rows: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [row for row in scored_rows if row["candidate_id"] == candidate_id]
    if len(rows) != 64:
        raise SyntaxAnalysisError(f"candidate {candidate_id} must contain exactly 64 rows")
    gate = design["gate_config"]
    by_form: dict[str, dict[str, Any]] = {}
    by_form_truth: dict[str, dict[str, int]] = {}
    for form_id in FORM_IDS:
        form_rows = [row for row in rows if row["form_id"] == form_id]
        scores = [float(row["s"]) for row in form_rows]
        by_form[form_id] = {
            "record_count": len(form_rows),
            "native_count": sum(bool(row["native"]) for row in form_rows),
            "native_correct_count": sum(bool(row["native_correct"]) for row in form_rows),
            "positive_s_count": sum(float(row["s"]) > 0.0 for row in form_rows),
            "median_s": float(np.median(np.asarray(scores, dtype=float))),
            "mean_s": float(np.mean(np.asarray(scores, dtype=float))),
        }
        for truth in TRUTH_POLARITIES:
            truth_rows = [row for row in form_rows if row["truth_polarity"] == truth]
            by_form_truth[f"{form_id}::{truth}"] = {
                "record_count": len(truth_rows),
                "native_count": sum(bool(row["native"]) for row in truth_rows),
                "native_correct_count": sum(
                    bool(row["native_correct"]) for row in truth_rows
                ),
            }
    by_item: dict[str, dict[str, Any]] = {}
    item_effects: dict[str, dict[str, float | str]] = {}
    for item_id in design["expected_item_ids"]:
        item_rows = [row for row in rows if row["item_id"] == item_id]
        forms = {(row["order"], row["mapping"]): row for row in item_rows}
        if len(item_rows) != 4 or set(forms) != set(runner.FORM_KEYS):
            raise SyntaxAnalysisError(f"candidate {candidate_id} item {item_id} lacks four forms")
        scores = {key: float(forms[key]["s"]) for key in runner.FORM_KEYS}
        pf_px = scores[("positive_first", "positive_is_x")]
        pf_py = scores[("positive_first", "positive_is_y")]
        nf_px = scores[("negative_first", "positive_is_x")]
        nf_py = scores[("negative_first", "positive_is_y")]
        values = list(scores.values())
        effects = {
            "O": 0.5 * (pf_px + pf_py - nf_px - nf_py),
            "R": 0.5 * (pf_px + nf_px - pf_py - nf_py),
            "I": 0.5 * ((pf_px - pf_py) - (nf_px - nf_py)),
        }
        item_effects[item_id] = {
            "pair_id": str(item_rows[0]["pair_id"]),
            "truth_polarity": str(item_rows[0]["truth_polarity"]),
            **effects,
        }
        score_range = max(values) - min(values)
        by_item[item_id] = {
            "pair_id": item_rows[0]["pair_id"],
            "truth_polarity": item_rows[0]["truth_polarity"],
            "mean_s": float(np.mean(np.asarray(values, dtype=float))),
            "score_range": score_range,
            "range_pass": bool(score_range <= gate["item_range_margin"]),
            **effects,
        }
    pair_estimands: dict[str, dict[str, Any]] = {}
    pair_ids = [pair["pair_id"] for pair in design["pair_registry"]]
    for effect in ("O", "R", "I"):
        mean_values: dict[str, float] = {}
        difference_values: dict[str, float] = {}
        for pair_id in pair_ids:
            positive = next(
                float(value[effect])
                for value in item_effects.values()
                if value["pair_id"] == pair_id and value["truth_polarity"] == "positive"
            )
            negative = next(
                float(value[effect])
                for value in item_effects.values()
                if value["pair_id"] == pair_id and value["truth_polarity"] == "negative"
            )
            mean_values[pair_id] = 0.5 * (positive + negative)
            difference_values[pair_id] = 0.5 * (positive - negative)
        pair_estimands[f"M_{effect}"] = _pair_equivalence(
            mean_values, gate["equivalence_margin"], gate["equivalence_alpha"]
        )
        pair_estimands[f"D_{effect}"] = _pair_equivalence(
            difference_values, gate["equivalence_margin"], gate["equivalence_alpha"]
        )
    total_native = sum(bool(row["native"]) for row in rows)
    total_native_correct = sum(bool(row["native_correct"]) for row in rows)
    failure_reasons: list[str] = []
    if total_native < gate["overall_native_min_count"]:
        failure_reasons.append("overall_native_below_floor")
    for form_id in FORM_IDS:
        if by_form[form_id]["native_count"] < gate["per_form_native_min_count"]:
            failure_reasons.append(f"form_native_below_floor::{form_id}")
    for key in sorted(by_form_truth):
        if by_form_truth[key]["native_count"] < gate["per_form_truth_native_min_count"]:
            failure_reasons.append(f"form_truth_native_below_floor::{key}")
    if total_native_correct < gate["overall_native_correct_min_count"]:
        failure_reasons.append("overall_native_correct_below_floor")
    for form_id in FORM_IDS:
        if by_form[form_id]["native_correct_count"] < gate[
            "per_form_native_correct_min_count"
        ]:
            failure_reasons.append(f"form_native_correct_below_floor::{form_id}")
    for key in sorted(by_form_truth):
        if by_form_truth[key]["native_correct_count"] < gate[
            "per_form_truth_native_correct_min_count"
        ]:
            failure_reasons.append(f"form_truth_native_correct_below_floor::{key}")
    for form_id in FORM_IDS:
        if by_form[form_id]["positive_s_count"] < gate["per_form_positive_s_min_count"]:
            failure_reasons.append(f"form_positive_s_below_floor::{form_id}")
        if by_form[form_id]["median_s"] < gate["per_form_median_s_min"]:
            failure_reasons.append(f"form_median_s_below_floor::{form_id}")
    range_pass_count = sum(bool(value["range_pass"]) for value in by_item.values())
    if range_pass_count < gate["item_range_required_count"]:
        failure_reasons.append("item_range_gate_failed")
    for estimand in gate["pair_estimands"]:
        if not pair_estimands[estimand]["pass"]:
            failure_reasons.append(f"pair_equivalence_failed::{estimand}")
    registry_entry = _candidate_by_id(design)[candidate_id]
    rank_metrics = {
        "min_form_native_correct_count": min(
            value["native_correct_count"] for value in by_form.values()
        ),
        "total_native_correct_count": total_native_correct,
        "min_form_truth_native_correct_count": min(
            value["native_correct_count"] for value in by_form_truth.values()
        ),
        "min_form_native_count": min(value["native_count"] for value in by_form.values()),
        "total_native_count": total_native,
        "max_item_range": max(value["score_range"] for value in by_item.values()),
        "max_abs_mean_pair_estimand": max(
            abs(pair_estimands[estimand]["mean"]) for estimand in gate["pair_estimands"]
        ),
        "min_form_median_s": min(value["median_s"] for value in by_form.values()),
        "candidate_rank": registry_entry["candidate_rank"],
        "candidate_id": candidate_id,
    }
    return {
        "candidate_id": candidate_id,
        "candidate_rank": registry_entry["candidate_rank"],
        "record_count": len(rows),
        "native_count": total_native,
        "native_fraction": total_native / len(rows),
        "native_correct_count": total_native_correct,
        "native_correct_fraction": total_native_correct / len(rows),
        "native_wrong_count": sum(bool(row["native_wrong"]) for row in rows),
        "non_native_count": len(rows) - total_native,
        "tied_maximum_count": sum(row["maximum_tie_count"] > 1 for row in rows),
        "mean_label_mass": float(np.mean([row["label_mass"] for row in rows])),
        "by_form": by_form,
        "by_form_truth": by_form_truth,
        "items": by_item,
        "item_range_pass_count": range_pass_count,
        "pair_estimands": pair_estimands,
        "eligible": not failure_reasons,
        "failure_reasons": failure_reasons,
        "ranking_metrics": rank_metrics,
    }


def _ranking_key(candidate: Mapping[str, Any], ranking: Sequence[Mapping[str, str]]) -> tuple[Any, ...]:
    output: list[Any] = []
    metrics = candidate["ranking_metrics"]
    for rule in ranking:
        metric = rule["metric"]
        direction = rule["direction"]
        value = metrics[metric]
        if isinstance(value, str):
            if direction != "min":
                raise SyntaxAnalysisError("string ranking metrics must use min direction")
            output.append(value)
        elif direction == "max":
            output.append(-value)
        elif direction == "min":
            output.append(value)
        else:
            raise SyntaxAnalysisError("ranking direction must be max or min")
    return tuple(output)


def _selected_projection(
    candidate: Mapping[str, Any],
    design: Mapping[str, Any],
    design_sha256: str,
    records_sha256: str,
    matrix_sha256: str,
) -> dict[str, Any]:
    registry_entry = _candidate_by_id(design)[candidate["candidate_id"]]
    return {
        "schema_version": SELECTED_CANDIDATE_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "candidate_rank": candidate["candidate_rank"],
        "candidate_definition_sha256": registry_entry["candidate_definition_sha256"],
        "candidate_definition": {
            key: registry_entry[key]
            for key in runner.candidate_definitions()[candidate["candidate_rank"]]
        },
        "x_token_id": registry_entry["x_token_id"],
        "y_token_id": registry_entry["y_token_id"],
        "source_design_sha256": design_sha256,
        "source_call_plan_sha256": design["call_plan_sha256"],
        "source_records_canonical_sha256": records_sha256,
        "source_full_vocab_matrix_sha256": matrix_sha256,
        "candidate_bank_sha256": design["candidate_bank_sha256"],
        "gate_config_sha256": canonical_sha256(design["gate_config"]),
        "ranking_metrics": dict(candidate["ranking_metrics"]),
        "model_id": design["expected_model_id"],
        "model_revision": design["expected_model_revision"],
        "model_weights_sha256": design["expected_model_weights_sha256"],
        "tokenizer_id": design["expected_tokenizer_id"],
        "tokenizer_revision": design["expected_tokenizer_revision"],
        "chat_template_sha256": design["expected_chat_template_sha256"],
        "dtype": design["expected_dtype"],
        "device": design["expected_device"],
        "source_fixture_sha256": design["source_fixture_sha256"],
        "source_manifest_sha256": design["source_manifest_sha256"],
        "preregistration_sha256": design["preregistration_sha256"],
        "runner_code_sha256": design["expected_runner_code_sha256"],
        "analyzer_code_sha256": design["expected_analyzer_code_sha256"],
    }


def analyze_syntax_selection(
    records: Iterable[Mapping[str, Any]],
    design: Mapping[str, Any],
    full_vocab_logits: np.ndarray,
) -> dict[str, Any]:
    """Validate, reconstruct, gate, rank, and optionally select one syntax."""

    locked = validate_design(design)
    checked_records = [_validate_record_identity(record, locked) for record in records]
    _validate_topology(checked_records, locked)
    checked_records.sort(key=lambda record: record["record_id"])
    matrix, sidecar_rows = _verify_sidecar(checked_records, locked, full_vocab_logits)
    scored_rows = _score_rows(checked_records, sidecar_rows)
    candidates = {
        candidate_id: _candidate_metrics(candidate_id, scored_rows, locked)
        for candidate_id in runner.CANDIDATE_ORDER
    }
    selection_authorized = locked["selection_eligible_model"]
    if selection_authorized:
        eligible = [candidate for candidate in candidates.values() if candidate["eligible"]]
        ranked = sorted(
            eligible,
            key=lambda candidate: _ranking_key(
                candidate, locked["gate_config"]["ranking"]
            ),
        )
    else:
        ranked = []
    selected: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    projection_sha256: str | None = None
    design_sha256 = canonical_sha256(locked)
    records_sha256 = canonical_sha256(checked_records)
    matrix_sha256 = coherent.full_vocab_matrix_sha256(matrix)
    if not selection_authorized:
        status = "SYNTAX_SMOKE_ANALYSIS_COMPLETE_NO_SELECTION_AUTHORITY"
    elif not ranked:
        status = "SYNTAX_SELECTION_STOP_NO_ELIGIBLE_CONTEXT"
    else:
        status = "SYNTAX_SELECTION_PASS"
        selected = ranked[0]
        projection = _selected_projection(
            selected,
            locked,
            design_sha256,
            records_sha256,
            matrix_sha256,
        )
        projection_sha256 = canonical_sha256(projection)
    ranking_order = [candidate["candidate_id"] for candidate in ranked]
    for candidate_id, candidate in candidates.items():
        if selection_authorized:
            candidate["eligible_rank"] = (
                ranking_order.index(candidate_id) + 1
                if candidate_id in ranking_order
                else None
            )
            candidate["selected"] = bool(
                selected is not None and candidate_id == selected["candidate_id"]
            )
        else:
            diagnostic_summary = dict(candidate.pop("ranking_metrics"))
            diagnostic_summary.pop("candidate_rank")
            diagnostic_summary.pop("candidate_id")
            candidate["diagnostic_summary"] = diagnostic_summary
            candidate["selection_evaluation_performed"] = False
            candidate.pop("eligible")
            candidate.pop("failure_reasons")
    claim_boundary = (
        "This fixed-bank development artifact can select one model-specific answer "
        "syntax only. It does not establish biology, knowledge, natural use, "
        "calibration, latent representation, an activation gap, or a physical law."
        if selection_authorized
        else (
            "This smoke artifact validates software and artifact extraction and reports "
            "diagnostics only. It cannot select, eliminate, rank, or veto a syntax, and "
            "it makes no biology, knowledge, activation-gap, or physical-law claim."
        )
    )
    result = {
        "artifact_type": "groundbench.coherent_readout_v2_syntax_selection",
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "mode": "development",
        "purpose": "syntax_selection_only",
        "model_role": locked["model_role"],
        "selection_authorized": selection_authorized,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selected_candidate_projection": projection,
        "selected_candidate_projection_sha256": projection_sha256,
        "design": locked,
        "design_sha256": design_sha256,
        "gate_config_sha256": canonical_sha256(locked["gate_config"]),
        "raw_records_canonical_sha256": records_sha256,
        "full_vocab_matrix_sha256": matrix_sha256,
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "validation": {
            "expected_records": 256,
            "observed_records": len(checked_records),
            "candidate_count": len(candidates),
            "item_count": len(locked["expected_item_ids"]),
            "pair_count": len(locked["pair_registry"]),
            "forms_per_candidate_item": len(runner.FORM_KEYS),
            "finite_full_vocab_logits": True,
            "sidecar_reconstructed": True,
            "exact_cartesian_topology": True,
        },
        "candidates": candidates,
        "claim_boundary": claim_boundary,
    }
    if selection_authorized:
        result["eligible_ranking_order"] = ranking_order
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Coherent readout v2: syntax-selection report",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Model role: `{result['model_role']}`.",
        "",
        "| candidate | native | native-correct | min form correct | max range | max |M/D mean| | eligible | rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in runner.CANDIDATE_ORDER:
        candidate = result["candidates"][candidate_id]
        metrics = candidate.get("ranking_metrics", candidate.get("diagnostic_summary"))
        selection_result = (
            ("PASS" if candidate["eligible"] else "FAIL")
            if result["selection_authorized"]
            else "N/A"
        )
        eligible_rank = candidate.get("eligible_rank") or "-"
        lines.append(
            f"| `{candidate_id}` | {candidate['native_count']}/64 | "
            f"{candidate['native_correct_count']}/64 | "
            f"{metrics['min_form_native_correct_count']}/16 | "
            f"{metrics['max_item_range']:.6f} | "
            f"{metrics['max_abs_mean_pair_estimand']:.6f} | "
            f"{selection_result} | "
            f"{eligible_rank} |"
        )
    lines.extend(
        [
            "",
            f"Selected candidate: `{result['selected_candidate_id']}`.",
            "",
            (
                "Selected-candidate projection SHA-256: "
                f"`{result['selected_candidate_projection_sha256']}`."
            ),
            "",
            f"Design SHA-256: `{result['design_sha256']}`.",
            "",
            f"Full-vocabulary matrix SHA-256: `{result['full_vocab_matrix_sha256']}`.",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SyntaxAnalysisError(f"JSONL line {line_number} is not an object")
        output.append(value)
    return output


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = value.encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise SyntaxAnalysisError(f"refusing to overwrite a different artifact: {path}")
    path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--full-logits-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        design = json.loads(args.design.read_text(encoding="utf-8"))
        matrix = np.load(args.full_logits_npy, allow_pickle=False)
        result = analyze_syntax_selection(_load_jsonl(args.records), design, matrix)
        _write_text(
            args.output_json,
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        _write_text(args.output_markdown, render_markdown(result))
        print(
            canonical_json(
                {
                    "output": str(args.output_json),
                    "selected_candidate_id": result["selected_candidate_id"],
                    "status": result["status"],
                }
            )
        )
    except (SyntaxAnalysisError, json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

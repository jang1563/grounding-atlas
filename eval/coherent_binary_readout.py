"""Coherent same-forward-pass binary readout contract and Level-0 analyzer.

This module implements the measurement gate in
``docs/COHERENT_BINARY_READOUT_DESIGN.md``.  It consumes only the two opaque-label
logits from one next-token distribution plus minimal full-vocabulary diagnostics.
It does not generate natural-language probabilities and it does not adjudicate
biology, knowledge, or activation claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student_t

RECORD_SCHEMA = "coherent-binary-readout-record-v1"
DESIGN_SCHEMA = "coherent-binary-readout-design-v1"
ANALYSIS_SCHEMA = "coherent-binary-readout-level0-analysis-v1"

ORDERS = ("positive_first", "negative_first")
MAPPINGS = ("positive_is_x", "positive_is_y")
FORM_KEYS = tuple((order, mapping) for order in ORDERS for mapping in MAPPINGS)

RECORD_KEYS = {
    "schema_version",
    "record_id",
    "donor_id",
    "source_item_id",
    "item_id",
    "readout_id",
    "input_family",
    "source_fixture_record_id",
    "gene_sentence_sha256",
    "positive_class",
    "negative_class",
    "order",
    "mapping",
    "x_token_id",
    "y_token_id",
    "positive_token_id",
    "negative_token_id",
    "x_logit",
    "y_logit",
    "full_vocab_logsumexp",
    "full_vocab_logits_sha256",
    "full_vocab_logits_row",
    "forward_trace_sha256",
    "logits_source",
    "vocab_size",
    "greedy_token_id",
    "greedy_logit",
    "user_prompt_sha256",
    "prompt_sha256",
    "execution_input_sha256",
    "input_token_count",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "chat_template_sha256",
    "dtype",
}

DESIGN_KEYS = {
    "schema_version",
    "mode",
    "required_readouts",
    "readout_classes",
    "required_input_families",
    "source_fixture_sha256",
    "source_manifest_sha256",
    "preregistration_sha256",
    "runner_code_sha256",
    "call_plan_sha256",
    "margin_lock_sha256",
    "margin_lock_status",
    "expected_model_id",
    "expected_model_revision",
    "expected_tokenizer_id",
    "expected_tokenizer_revision",
    "expected_chat_template_sha256",
    "expected_dtype",
    "expected_x_token_id",
    "expected_y_token_id",
    "expected_vocab_size",
    "expected_donor_ids",
    "expected_source_items",
    "expected_record_ids",
    "expected_confirmatory_donors",
    "alpha",
    "equivalence_margin",
    "coherence_tolerance",
    "item_range_margin",
    "strong_score_threshold",
    "format_overall_min",
    "format_per_group_min",
    "format_per_donor_min",
    "item_range_pass_min",
}

HEX_DIGITS = frozenset("0123456789abcdef")
SOURCE_ITEM_KEYS = {"source_item_id", "donor_id"}
READOUT_CLASS_KEYS = {"positive_class", "negative_class"}
FULL_LOGIT_ROW_SCHEMA = "coherent-binary-full-logits-f32le-v1"


class CoherentReadoutError(ValueError):
    """Raised when a readout artifact violates the frozen Level-0 contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_full_vocab_row(values: Sequence[float]) -> np.ndarray:
    """Return the frozen finite 1-D little-endian float32 logit representation."""

    row = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    if row.ndim != 1 or len(row) < 2 or not np.isfinite(row).all():
        raise CoherentReadoutError(
            "full-vocabulary logit row must contain at least two finite values"
        )
    return row


def full_vocab_logits_sha256(values: Sequence[float]) -> str:
    row = canonical_full_vocab_row(values)
    header = canonical_json(
        {"schema_version": FULL_LOGIT_ROW_SCHEMA, "length": len(row)}
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + row.tobytes(order="C")).hexdigest()


def full_vocab_matrix_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise CoherentReadoutError("full-vocabulary sidecar must be a finite 2-D matrix")
    header = canonical_json(
        {
            "schema_version": "coherent-binary-full-logits-matrix-f32le-v1",
            "shape": list(matrix.shape),
        }
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + matrix.tobytes(order="C")).hexdigest()


def full_vocab_diagnostics(
    values: Sequence[float], *, x_token_id: int, y_token_id: int
) -> dict[str, Any]:
    row = canonical_full_vocab_row(values)
    for token_id, label in ((x_token_id, "X"), (y_token_id, "Y")):
        if not 0 <= token_id < len(row):
            raise CoherentReadoutError(f"{label} token ID lies outside logit row")
    greedy_token_id = int(np.argmax(row))
    maximum = float(row[greedy_token_id])
    logsumexp = maximum + math.log(
        float(np.exp(row.astype(np.float64) - maximum).sum())
    )
    return {
        "x_logit": float(row[x_token_id]),
        "y_logit": float(row[y_token_id]),
        "greedy_token_id": greedy_token_id,
        "greedy_logit": maximum,
        "full_vocab_logsumexp": logsumexp,
        "full_vocab_logits_sha256": full_vocab_logits_sha256(row),
    }


def default_design(
    *,
    mode: str,
    required_readouts: Sequence[str],
    readout_classes: Mapping[str, Mapping[str, str]],
    required_input_families: Sequence[str],
    source_fixture_sha256: str,
    source_manifest_sha256: str,
    preregistration_sha256: str,
    runner_code_sha256: str,
    call_plan_sha256: str,
    margin_lock_sha256: str,
    margin_lock_status: str,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    chat_template_sha256: str,
    dtype: str,
    x_token_id: int,
    y_token_id: int,
    vocab_size: int,
    expected_donor_ids: Sequence[str],
    expected_source_items: Sequence[Mapping[str, str]],
    expected_record_ids: Sequence[str],
    expected_confirmatory_donors: int | None = None,
) -> dict[str, Any]:
    """Return the exact v1 design object with the preregistered default margins."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "mode": mode,
        "required_readouts": sorted(required_readouts),
        "readout_classes": {
            key: dict(readout_classes[key]) for key in sorted(readout_classes)
        },
        "required_input_families": sorted(required_input_families),
        "source_fixture_sha256": source_fixture_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "preregistration_sha256": preregistration_sha256,
        "runner_code_sha256": runner_code_sha256,
        "call_plan_sha256": call_plan_sha256,
        "margin_lock_sha256": margin_lock_sha256,
        "margin_lock_status": margin_lock_status,
        "expected_model_id": model_id,
        "expected_model_revision": model_revision,
        "expected_tokenizer_id": tokenizer_id,
        "expected_tokenizer_revision": tokenizer_revision,
        "expected_chat_template_sha256": chat_template_sha256,
        "expected_dtype": dtype,
        "expected_x_token_id": x_token_id,
        "expected_y_token_id": y_token_id,
        "expected_vocab_size": vocab_size,
        "expected_donor_ids": sorted(expected_donor_ids),
        "expected_source_items": sorted(
            [dict(value) for value in expected_source_items],
            key=lambda value: value["source_item_id"],
        ),
        "expected_record_ids": sorted(expected_record_ids),
        "expected_confirmatory_donors": expected_confirmatory_donors,
        "alpha": 0.05,
        "equivalence_margin": 0.06,
        "coherence_tolerance": 1e-6,
        "item_range_margin": 0.20,
        "strong_score_threshold": 0.20,
        "format_overall_min": 0.95,
        "format_per_group_min": 0.95,
        "format_per_donor_min": 0.90,
        "item_range_pass_min": 0.95,
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CoherentReadoutError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoherentReadoutError(f"{label} must be a nonempty string")
    return value


def _require_single_line(value: Any, label: str) -> str:
    output = _require_string(value, label)
    if output != output.strip() or "\n" in output or "\r" in output:
        raise CoherentReadoutError(f"{label} must be a stripped single line")
    return output


def _require_sha256(value: Any, label: str) -> str:
    output = _require_string(value, label)
    if len(output) != 64 or any(character not in HEX_DIGITS for character in output):
        raise CoherentReadoutError(f"{label} must be a lowercase SHA-256 digest")
    return output


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CoherentReadoutError(f"{label} must be an integer")
    output = int(value)
    if output < 0:
        raise CoherentReadoutError(f"{label} must be nonnegative")
    return output


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise CoherentReadoutError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise CoherentReadoutError(f"{label} must be finite")
    return output


def _validate_name_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CoherentReadoutError(f"{label} must be a nonempty list")
    result = [_require_string(item, f"{label} item") for item in value]
    if result != sorted(set(result)):
        raise CoherentReadoutError(f"{label} must be unique and sorted")
    return result


def validate_design(design: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(design, DESIGN_KEYS, "design")
    if design["schema_version"] != DESIGN_SCHEMA:
        raise CoherentReadoutError("unsupported design schema")
    mode = design["mode"]
    if mode not in {"development", "confirmatory"}:
        raise CoherentReadoutError("design mode must be development or confirmatory")
    readouts = _validate_name_list(design["required_readouts"], "required_readouts")
    families = _validate_name_list(
        design["required_input_families"], "required_input_families"
    )
    readout_classes_value = design["readout_classes"]
    if not isinstance(readout_classes_value, Mapping) or set(readout_classes_value) != set(
        readouts
    ):
        raise CoherentReadoutError("readout_classes must exactly cover required_readouts")
    readout_classes: dict[str, dict[str, str]] = {}
    for readout in readouts:
        value = readout_classes_value[readout]
        if not isinstance(value, Mapping):
            raise CoherentReadoutError(f"readout class specification {readout} must be an object")
        _require_exact_keys(value, READOUT_CLASS_KEYS, f"readout class specification {readout}")
        positive = _require_single_line(value["positive_class"], f"{readout} positive_class")
        negative = _require_single_line(value["negative_class"], f"{readout} negative_class")
        if positive == negative:
            raise CoherentReadoutError(f"{readout} class strings must be distinct")
        readout_classes[readout] = {
            "positive_class": positive,
            "negative_class": negative,
        }
    expected_n = design["expected_confirmatory_donors"]
    if mode == "development":
        if expected_n is not None:
            raise CoherentReadoutError(
                "development design must not declare expected confirmatory donors"
            )
    else:
        expected_n = _require_int(expected_n, "expected_confirmatory_donors")
        if not 12 <= expected_n <= 20:
            raise CoherentReadoutError("confirmatory donor count must be between 12 and 20")

    output = dict(design)
    output["required_readouts"] = readouts
    output["readout_classes"] = readout_classes
    output["required_input_families"] = families
    for key in (
        "source_fixture_sha256",
        "source_manifest_sha256",
        "preregistration_sha256",
        "runner_code_sha256",
        "call_plan_sha256",
        "margin_lock_sha256",
    ):
        output[key] = _require_sha256(design[key], key)
    margin_lock_status = design["margin_lock_status"]
    if margin_lock_status not in {"candidate_unqualified", "phase0_qualified"}:
        raise CoherentReadoutError(
            "margin_lock_status must be candidate_unqualified or phase0_qualified"
        )
    if mode == "confirmatory" and margin_lock_status != "phase0_qualified":
        raise CoherentReadoutError("confirmatory design requires a Phase-0-qualified margin")
    output["margin_lock_status"] = margin_lock_status
    output["expected_model_id"] = _require_string(
        design["expected_model_id"], "expected_model_id"
    )
    output["expected_model_revision"] = _require_string(
        design["expected_model_revision"], "expected_model_revision"
    )
    output["expected_tokenizer_id"] = _require_string(
        design["expected_tokenizer_id"], "expected_tokenizer_id"
    )
    output["expected_tokenizer_revision"] = _require_string(
        design["expected_tokenizer_revision"], "expected_tokenizer_revision"
    )
    output["expected_chat_template_sha256"] = _require_sha256(
        design["expected_chat_template_sha256"], "expected_chat_template_sha256"
    )
    output["expected_dtype"] = _require_string(
        design["expected_dtype"], "expected_dtype"
    )
    output["expected_x_token_id"] = _require_int(
        design["expected_x_token_id"], "expected_x_token_id"
    )
    output["expected_y_token_id"] = _require_int(
        design["expected_y_token_id"], "expected_y_token_id"
    )
    if output["expected_x_token_id"] == output["expected_y_token_id"]:
        raise CoherentReadoutError("expected opaque token IDs must be distinct")
    output["expected_vocab_size"] = _require_int(
        design["expected_vocab_size"], "expected_vocab_size"
    )
    if output["expected_vocab_size"] < 2 or max(
        output["expected_x_token_id"], output["expected_y_token_id"]
    ) >= output["expected_vocab_size"]:
        raise CoherentReadoutError("expected opaque token IDs lie outside vocabulary")
    output["expected_donor_ids"] = _validate_name_list(
        design["expected_donor_ids"], "expected_donor_ids"
    )
    source_items_value = design["expected_source_items"]
    if not isinstance(source_items_value, list) or not source_items_value:
        raise CoherentReadoutError("expected_source_items must be a nonempty list")
    source_items: list[dict[str, str]] = []
    for index, value in enumerate(source_items_value):
        if not isinstance(value, Mapping):
            raise CoherentReadoutError(f"expected_source_items item {index} must be an object")
        _require_exact_keys(value, SOURCE_ITEM_KEYS, f"expected_source_items item {index}")
        source_items.append(
            {
                "source_item_id": _require_string(
                    value["source_item_id"], f"expected_source_items item {index} source_item_id"
                ),
                "donor_id": _require_string(
                    value["donor_id"], f"expected_source_items item {index} donor_id"
                ),
            }
        )
    if source_items != sorted(source_items, key=lambda value: value["source_item_id"]):
        raise CoherentReadoutError("expected_source_items must be sorted by source_item_id")
    if len({value["source_item_id"] for value in source_items}) != len(source_items):
        raise CoherentReadoutError("expected_source_items source_item_id values must be unique")
    if {value["donor_id"] for value in source_items} != set(output["expected_donor_ids"]):
        raise CoherentReadoutError("expected_source_items must cover every expected donor")
    output["expected_source_items"] = source_items
    record_ids = design["expected_record_ids"]
    if not isinstance(record_ids, list) or not record_ids:
        raise CoherentReadoutError("expected_record_ids must be a nonempty list")
    output["expected_record_ids"] = [
        _require_sha256(value, "expected_record_ids item") for value in record_ids
    ]
    if output["expected_record_ids"] != sorted(set(output["expected_record_ids"])):
        raise CoherentReadoutError("expected_record_ids must be unique and sorted")
    if mode == "confirmatory" and len(output["expected_donor_ids"]) != expected_n:
        raise CoherentReadoutError(
            "expected donor registry does not match expected_confirmatory_donors"
        )
    output["expected_confirmatory_donors"] = expected_n
    for key in (
        "alpha",
        "equivalence_margin",
        "coherence_tolerance",
        "item_range_margin",
        "strong_score_threshold",
        "format_overall_min",
        "format_per_group_min",
        "format_per_donor_min",
        "item_range_pass_min",
    ):
        output[key] = _require_float(design[key], key)
    if not 0.0 < output["alpha"] < 0.5:
        raise CoherentReadoutError("alpha must lie strictly between 0 and 0.5")
    if output["alpha"] != 0.05:
        raise CoherentReadoutError("design v1 freezes alpha at 0.05")
    for key in ("equivalence_margin", "item_range_margin", "strong_score_threshold"):
        if not 0.0 < output[key] < 2.0:
            raise CoherentReadoutError(f"{key} must lie strictly between 0 and 2")
    if not 0.0 < output["coherence_tolerance"] <= 1e-6:
        raise CoherentReadoutError("coherence_tolerance must lie in (0, 1e-6]")
    for key in (
        "format_overall_min",
        "format_per_group_min",
        "format_per_donor_min",
        "item_range_pass_min",
    ):
        if not 0.0 <= output[key] <= 1.0:
            raise CoherentReadoutError(f"{key} must lie in [0,1]")
    if output["equivalence_margin"] > 0.06:
        raise CoherentReadoutError("equivalence_margin cannot exceed the frozen 0.06")
    if output["item_range_margin"] > 0.20:
        raise CoherentReadoutError("item_range_margin cannot exceed the frozen 0.20")
    if output["strong_score_threshold"] > 0.20:
        raise CoherentReadoutError("strong_score_threshold cannot exceed the frozen 0.20")
    if output["format_overall_min"] < 0.95 or output["format_per_group_min"] < 0.95:
        raise CoherentReadoutError("overall and group format minima cannot be below 0.95")
    if output["format_per_donor_min"] < 0.90:
        raise CoherentReadoutError("per-donor format minimum cannot be below 0.90")
    if output["item_range_pass_min"] < 0.95:
        raise CoherentReadoutError("item range pass minimum cannot be below 0.95")
    return output


def render_binary_prompt(
    *,
    gene_sentence: str,
    positive_class: str,
    negative_class: str,
    order: str,
    mapping: str,
    x_label: str = "X",
    y_label: str = "Y",
) -> str:
    """Render one symmetric order-by-remapping prompt before a chat template."""

    if order not in ORDERS:
        raise CoherentReadoutError(f"unknown order: {order}")
    if mapping not in MAPPINGS:
        raise CoherentReadoutError(f"unknown mapping: {mapping}")
    gene_sentence = _require_single_line(gene_sentence, "gene_sentence")
    positive_class = _require_single_line(positive_class, "positive_class")
    negative_class = _require_single_line(negative_class, "negative_class")
    x_label = _require_single_line(x_label, "x_label")
    y_label = _require_single_line(y_label, "y_label")
    if positive_class == negative_class:
        raise CoherentReadoutError("positive and negative class strings must be distinct")
    if x_label == y_label or len(x_label) != len(y_label):
        raise CoherentReadoutError("opaque labels must be distinct and equal-length")

    positive_label, negative_label = (
        (x_label, y_label) if mapping == "positive_is_x" else (y_label, x_label)
    )
    positive_line = f"label {positive_label} means {positive_class}"
    negative_line = f"label {negative_label} means {negative_class}"
    line_1, line_2 = (
        (positive_line, negative_line)
        if order == "positive_first"
        else (negative_line, positive_line)
    )
    return (
        "Classify the cell using exactly one label.\n"
        f"{line_1}\n{line_2}\n"
        f"Genes, highest expression rank first: {gene_sentence}\n"
        "Label:"
    )


def contextual_single_token_id(tokenizer: Any, context: str, answer_text: str) -> int:
    """Resolve one answer token by exact contextual suffix tokenization."""

    context = _require_string(context, "context")
    answer_text = _require_string(answer_text, "answer_text")
    base_ids = list(tokenizer.encode(context, add_special_tokens=False))
    answer_ids = list(tokenizer.encode(context + answer_text, add_special_tokens=False))
    if answer_ids[: len(base_ids)] != base_ids or len(answer_ids) != len(base_ids) + 1:
        raise CoherentReadoutError(
            f"answer text {answer_text!r} is not one context-stable token"
        )
    return _require_int(answer_ids[-1], "answer token ID")


def _record_identity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "donor_id",
        "source_item_id",
        "item_id",
        "readout_id",
        "input_family",
        "source_fixture_record_id",
        "gene_sentence_sha256",
        "positive_class",
        "negative_class",
        "order",
        "mapping",
        "x_token_id",
        "y_token_id",
        "positive_token_id",
        "negative_token_id",
        "user_prompt_sha256",
        "prompt_sha256",
        "execution_input_sha256",
        "input_token_count",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "chat_template_sha256",
        "dtype",
        "logits_source",
        "vocab_size",
        "full_vocab_logits_row",
    )
    return {key: record[key] for key in keys}


def record_id(record: Mapping[str, Any]) -> str:
    return canonical_sha256(_record_identity_payload(record))


def call_plan_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash the complete pre-forward request plan without any model outcomes."""

    entries = [
        {
            "record_id": record_id(record),
            "request_identity": _record_identity_payload(record),
        }
        for record in records
    ]
    entries.sort(key=lambda value: value["record_id"])
    if not entries or len({value["record_id"] for value in entries}) != len(entries):
        raise CoherentReadoutError("call plan must contain unique nonempty request identities")
    return canonical_sha256(
        {"schema_version": "coherent-binary-call-plan-v1", "entries": entries}
    )


def forward_trace_sha256(record: Mapping[str, Any]) -> str:
    """Bind one raw full-vocabulary tensor to its exact model input context."""

    return canonical_sha256(
        {
            "schema_version": "coherent-binary-forward-trace-v1",
            "prompt_sha256": record["prompt_sha256"],
            "execution_input_sha256": record["execution_input_sha256"],
            "input_token_count": record["input_token_count"],
            "full_vocab_logits_sha256": record["full_vocab_logits_sha256"],
            "full_vocab_logits_row": record["full_vocab_logits_row"],
            "vocab_size": record["vocab_size"],
            "model_id": record["model_id"],
            "model_revision": record["model_revision"],
            "tokenizer_id": record["tokenizer_id"],
            "tokenizer_revision": record["tokenizer_revision"],
            "dtype": record["dtype"],
            "logits_source": record["logits_source"],
        }
    )


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(record, RECORD_KEYS, "record")
    if record["schema_version"] != RECORD_SCHEMA:
        raise CoherentReadoutError("unsupported record schema")
    output = dict(record)
    for key in (
        "record_id",
        "donor_id",
        "source_item_id",
        "item_id",
        "readout_id",
        "input_family",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dtype",
    ):
        output[key] = _require_string(record[key], key)
    output["record_id"] = _require_sha256(record["record_id"], "record_id")
    for key in (
        "user_prompt_sha256",
        "prompt_sha256",
        "execution_input_sha256",
        "chat_template_sha256",
        "source_fixture_record_id",
        "gene_sentence_sha256",
        "full_vocab_logits_sha256",
        "forward_trace_sha256",
    ):
        output[key] = _require_sha256(record[key], key)
    output["positive_class"] = _require_single_line(
        record["positive_class"], "positive_class"
    )
    output["negative_class"] = _require_single_line(
        record["negative_class"], "negative_class"
    )
    if output["positive_class"] == output["negative_class"]:
        raise CoherentReadoutError("record class strings must be distinct")
    if record["logits_source"] != "raw_model_output_before_processors":
        raise CoherentReadoutError(
            "logits_source must be raw_model_output_before_processors"
        )
    output["logits_source"] = record["logits_source"]
    if record["order"] not in ORDERS or record["mapping"] not in MAPPINGS:
        raise CoherentReadoutError("record has an unknown order or mapping")
    for key in (
        "x_token_id",
        "y_token_id",
        "positive_token_id",
        "negative_token_id",
        "greedy_token_id",
        "input_token_count",
        "vocab_size",
        "full_vocab_logits_row",
    ):
        output[key] = _require_int(record[key], key)
    if output["input_token_count"] < 1:
        raise CoherentReadoutError("input_token_count must be positive")
    if output["vocab_size"] < 2:
        raise CoherentReadoutError("vocab_size must be at least two")
    if any(
        output[key] >= output["vocab_size"]
        for key in ("x_token_id", "y_token_id", "greedy_token_id")
    ):
        raise CoherentReadoutError("retained token ID lies outside the frozen vocabulary")
    if output["x_token_id"] == output["y_token_id"]:
        raise CoherentReadoutError("opaque labels must map to distinct token IDs")
    expected_positive, expected_negative = (
        (output["x_token_id"], output["y_token_id"])
        if record["mapping"] == "positive_is_x"
        else (output["y_token_id"], output["x_token_id"])
    )
    if (
        output["positive_token_id"] != expected_positive
        or output["negative_token_id"] != expected_negative
    ):
        raise CoherentReadoutError("aligned token IDs do not match opaque remapping")
    for key in ("x_logit", "y_logit", "full_vocab_logsumexp", "greedy_logit"):
        output[key] = _require_float(record[key], key)
    retained_lse = float(np.logaddexp(output["x_logit"], output["y_logit"]))
    scale = max(1.0, abs(retained_lse), abs(output["full_vocab_logsumexp"]))
    tolerance = 32.0 * np.finfo(float).eps * scale
    if output["greedy_logit"] + tolerance < max(
        output["x_logit"], output["y_logit"]
    ):
        raise CoherentReadoutError("greedy logit is below a retained opaque-token logit")
    if output["full_vocab_logsumexp"] + tolerance < max(
        retained_lse, output["greedy_logit"]
    ):
        raise CoherentReadoutError(
            "full-vocabulary logsumexp is inconsistent with retained logits"
        )
    if output["full_vocab_logsumexp"] > (
        output["greedy_logit"] + math.log(output["vocab_size"]) + tolerance
    ):
        raise CoherentReadoutError(
            "full-vocabulary logsumexp exceeds the greedy-logit/vocabulary bound"
        )
    if output["greedy_token_id"] == output["x_token_id"] and not math.isclose(
        output["greedy_logit"], output["x_logit"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise CoherentReadoutError("greedy X token does not match x_logit")
    if output["greedy_token_id"] == output["y_token_id"] and not math.isclose(
        output["greedy_logit"], output["y_logit"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise CoherentReadoutError("greedy Y token does not match y_logit")
    if output["record_id"] != record_id(output):
        raise CoherentReadoutError("record_id does not bind the exact request identity")
    if output["forward_trace_sha256"] != forward_trace_sha256(output):
        raise CoherentReadoutError("forward_trace_sha256 does not bind the raw tensor trace")
    return output


def make_record(**values: Any) -> dict[str, Any]:
    """Construct and validate a record, deriving its immutable identity."""

    record = {"schema_version": RECORD_SCHEMA, **values}
    record["record_id"] = record_id(record)
    if "forward_trace_sha256" not in record:
        record["forward_trace_sha256"] = forward_trace_sha256(record)
    return validate_record(record)


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_record(record)
    positive = (
        checked["x_logit"]
        if checked["positive_token_id"] == checked["x_token_id"]
        else checked["y_logit"]
    )
    negative = (
        checked["x_logit"]
        if checked["negative_token_id"] == checked["x_token_id"]
        else checked["y_logit"]
    )
    delta = positive - negative
    if not math.isfinite(delta):
        raise CoherentReadoutError("aligned opaque-token logit difference is nonfinite")
    if delta >= 0.0:
        exp_negative_delta = math.exp(-delta)
        q_positive = 1.0 / (1.0 + exp_negative_delta)
    else:
        exp_delta = math.exp(delta)
        q_positive = exp_delta / (1.0 + exp_delta)
    q_negative = 1.0 - q_positive
    two_token_logsumexp = float(np.logaddexp(positive, negative))
    permitted_mass = math.exp(two_token_logsumexp - checked["full_vocab_logsumexp"])
    if not math.isfinite(permitted_mass) or permitted_mass > 1.0 + 1e-12:
        raise CoherentReadoutError("invalid two-token full-vocabulary probability mass")
    permitted_mass = min(1.0, permitted_mass)
    return {
        **checked,
        "delta": delta,
        "q_positive": q_positive,
        "q_negative": q_negative,
        "coherence_residual": abs(q_positive + q_negative - 1.0),
        "s": math.tanh(delta / 2.0),
        "two_token_probability_mass": permitted_mass,
        "format_adherent": checked["greedy_token_id"]
        in {checked["x_token_id"], checked["y_token_id"]},
    }


def _t_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise CoherentReadoutError("donor vector must contain at least two finite values")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def _one_sided_t_p(values: np.ndarray, *, null: float, direction: str) -> float:
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    centered_mean = float(values.mean() - null)
    if standard_error == 0.0:
        statistic = 0.0 if centered_mean == 0.0 else math.copysign(math.inf, centered_mean)
    else:
        statistic = centered_mean / standard_error
    if direction == "greater":
        return float(student_t.sf(statistic, len(values) - 1))
    if direction == "less":
        return float(student_t.cdf(statistic, len(values) - 1))
    raise CoherentReadoutError(f"unknown direction: {direction}")


def _comparison_tolerance(values: np.ndarray, *, null: float = 0.0) -> float:
    """Return a scale-aware floating-point comparison tolerance.

    The tolerance only protects arithmetic comparisons. It never turns equality at
    a scientific boundary into a success: boundary ties remain failures.
    """

    scale = max(1.0, abs(null), float(np.sum(np.abs(values))))
    return 64.0 * np.finfo(float).eps * scale


def exact_sign_flip_p(
    values: Sequence[float],
    *,
    direction: str,
    null: float = 0.0,
    chunk_size: int = 65_536,
) -> float:
    """Exact one-sided Rademacher p-value, conditional on sign symmetry."""

    vector = np.asarray(values, dtype=float) - null
    if vector.ndim != 1 or not 2 <= len(vector) <= 20 or not np.isfinite(vector).all():
        raise CoherentReadoutError("exact sign-flip requires 2-20 finite donor values")
    observed = float(vector.sum())
    total = 1 << len(vector)
    extreme = 0
    bit_positions = np.arange(len(vector), dtype=np.uint64)
    tolerance = _comparison_tolerance(vector)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        indices = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (((indices >> bit_positions) & 1).astype(float) * 2.0) - 1.0
        signed_sums = signs @ vector
        if direction == "greater":
            extreme += int(np.sum(signed_sums >= observed - tolerance))
        elif direction == "less":
            extreme += int(np.sum(signed_sums <= observed + tolerance))
        else:
            raise CoherentReadoutError(f"unknown direction: {direction}")
    return extreme / total


def _sign_test(values: np.ndarray, *, direction: str, null: float) -> dict[str, Any]:
    tolerance = _comparison_tolerance(values, null=null)
    if direction == "greater":
        successes = int(np.sum(values > null + tolerance))
    elif direction == "less":
        successes = int(np.sum(values < null - tolerance))
    else:
        raise CoherentReadoutError(f"unknown direction: {direction}")
    return {
        "successes": successes,
        "trials": len(values),
        "ties_counted_as_failures": int(np.sum(np.abs(values - null) <= tolerance)),
        "p_value": float(
            binomtest(successes, len(values), 0.5, alternative="greater").pvalue
        ),
    }


def equivalence_summary(
    values: Sequence[float], *, margin: float, alpha: float = 0.05
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not 2 <= len(vector) <= 20 or not np.isfinite(vector).all():
        raise CoherentReadoutError("equivalence requires 2-20 finite donor values")
    if not 0.0 < margin < 2.0:
        raise CoherentReadoutError("equivalence margin must lie strictly between 0 and 2")
    if not 0.0 < alpha < 0.5:
        raise CoherentReadoutError("equivalence alpha must lie strictly between 0 and 0.5")
    ci90_lower, ci90_upper = _t_interval(vector, 0.90)
    lodo = [float(np.delete(vector, index).mean()) for index in range(len(vector))]
    lower_exact = exact_sign_flip_p(vector, direction="greater", null=-margin)
    upper_exact = exact_sign_flip_p(vector, direction="less", null=margin)
    lower_t = _one_sided_t_p(vector, null=-margin, direction="greater")
    upper_t = _one_sided_t_p(vector, null=margin, direction="less")
    output = {
        "n_donors": len(vector),
        "mean": float(vector.mean()),
        "standard_deviation": float(vector.std(ddof=1)),
        "margin": margin,
        "alpha": alpha,
        "ci90_lower": ci90_lower,
        "ci90_upper": ci90_upper,
        "ci90_strictly_inside_margin": bool(
            ci90_lower > -margin and ci90_upper < margin
        ),
        "lower_shift_exact_sign_flip_p": lower_exact,
        "upper_shift_exact_sign_flip_p": upper_exact,
        "exact_shifted_tost_pass": bool(lower_exact < alpha and upper_exact < alpha),
        "lower_shift_student_t_p": lower_t,
        "upper_shift_student_t_p": upper_t,
        "student_t_tost_pass": bool(lower_t < alpha and upper_t < alpha),
        "lower_exact_sign_test": _sign_test(vector, direction="greater", null=-margin),
        "upper_exact_sign_test": _sign_test(vector, direction="less", null=margin),
        "leave_one_donor_out_means": lodo,
        "all_lodo_strictly_inside_margin": bool(
            all(-margin < value < margin for value in lodo)
        ),
        "sign_symmetry_assumption": "required_unverified",
        "pass_is_conditional_on_sign_symmetry": True,
    }
    output["pass"] = bool(
        output["ci90_strictly_inside_margin"]
        and output["exact_shifted_tost_pass"]
        and output["student_t_tost_pass"]
        and output["all_lodo_strictly_inside_margin"]
    )
    return output


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise CoherentReadoutError("fraction denominator must be positive")
    return numerator / denominator


def verify_full_vocab_sidecar(
    records: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    full_vocab_logits: np.ndarray,
) -> str:
    """Recompute every retained diagnostic from the raw full-vocabulary sidecar."""

    matrix = np.asarray(full_vocab_logits)
    if matrix.dtype != np.dtype("<f4") or matrix.ndim != 2:
        raise CoherentReadoutError(
            "full-vocabulary sidecar must be a two-dimensional little-endian float32 array"
        )
    expected_shape = (len(records), int(design["expected_vocab_size"]))
    if matrix.shape != expected_shape:
        raise CoherentReadoutError(
            f"full-vocabulary sidecar shape mismatch: expected {expected_shape}, observed {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise CoherentReadoutError("full-vocabulary sidecar contains nonfinite logits")
    rows = [int(record["full_vocab_logits_row"]) for record in records]
    if sorted(rows) != list(range(len(records))):
        raise CoherentReadoutError(
            "full_vocab_logits_row must be a permutation of the sidecar rows"
        )
    for record in records:
        if record["vocab_size"] != design["expected_vocab_size"]:
            raise CoherentReadoutError("record vocabulary size does not match design")
        row = matrix[int(record["full_vocab_logits_row"])]
        diagnostics = full_vocab_diagnostics(
            row,
            x_token_id=int(record["x_token_id"]),
            y_token_id=int(record["y_token_id"]),
        )
        for key in ("x_logit", "y_logit", "greedy_logit", "full_vocab_logsumexp"):
            if not math.isclose(
                float(record[key]),
                float(diagnostics[key]),
                rel_tol=1e-12,
                abs_tol=1e-7,
            ):
                raise CoherentReadoutError(f"sidecar does not reproduce record field {key}")
        if int(record["greedy_token_id"]) != diagnostics["greedy_token_id"]:
            raise CoherentReadoutError("sidecar does not reproduce greedy_token_id")
        if record["full_vocab_logits_sha256"] != diagnostics[
            "full_vocab_logits_sha256"
        ]:
            raise CoherentReadoutError("sidecar row digest does not match record")
    return full_vocab_matrix_sha256(matrix)


def analyze_level0(
    records: Iterable[Mapping[str, Any]],
    design: Mapping[str, Any],
    full_vocab_logits: np.ndarray,
) -> dict[str, Any]:
    """Validate raw records and adjudicate every registered Level-0 group."""

    locked = validate_design(design)
    validated = [validate_record(record) for record in records]
    if not validated:
        raise CoherentReadoutError("no records supplied")
    full_vocab_sidecar_sha256 = verify_full_vocab_sidecar(
        validated, locked, full_vocab_logits
    )
    checked = [score_record(record) for record in validated]
    if len({record["record_id"] for record in checked}) != len(checked):
        raise CoherentReadoutError("duplicate record_id")
    observed_record_ids = {record["record_id"] for record in checked}
    expected_record_ids = set(locked["expected_record_ids"])
    if observed_record_ids != expected_record_ids:
        raise CoherentReadoutError(
            "frozen call-plan coverage mismatch: "
            f"missing={sorted(expected_record_ids - observed_record_ids)}, "
            f"unexpected={sorted(observed_record_ids - expected_record_ids)}"
        )
    if call_plan_sha256(checked) != locked["call_plan_sha256"]:
        raise CoherentReadoutError("records do not reproduce the frozen call-plan hash")
    if {record["model_id"] for record in checked} != {
        locked["expected_model_id"]
    }:
        raise CoherentReadoutError("record model ID does not match design")
    if {record["model_revision"] for record in checked} != {
        locked["expected_model_revision"]
    }:
        raise CoherentReadoutError("record model revision does not match design")
    if {record["tokenizer_id"] for record in checked} != {
        locked["expected_tokenizer_id"]
    }:
        raise CoherentReadoutError("record tokenizer ID does not match design")
    if {record["tokenizer_revision"] for record in checked} != {
        locked["expected_tokenizer_revision"]
    }:
        raise CoherentReadoutError("record tokenizer revision does not match design")
    if {record["chat_template_sha256"] for record in checked} != {
        locked["expected_chat_template_sha256"]
    }:
        raise CoherentReadoutError("record chat-template hash does not match design")
    if {record["dtype"] for record in checked} != {locked["expected_dtype"]}:
        raise CoherentReadoutError("record dtype does not match design")
    if {record["x_token_id"] for record in checked} != {
        locked["expected_x_token_id"]
    } or {record["y_token_id"] for record in checked} != {
        locked["expected_y_token_id"]
    }:
        raise CoherentReadoutError("record opaque-token IDs do not match design")
    for record in checked:
        expected_classes = locked["readout_classes"].get(record["readout_id"])
        if expected_classes is None or any(
            record[key] != expected_classes[key] for key in READOUT_CLASS_KEYS
        ):
            raise CoherentReadoutError("record class strings do not match the design")

    expected_groups = {
        (readout, family)
        for readout in locked["required_readouts"]
        for family in locked["required_input_families"]
    }
    observed_groups = {
        (record["readout_id"], record["input_family"]) for record in checked
    }
    if observed_groups != expected_groups:
        raise CoherentReadoutError(
            "readout/input-family coverage mismatch: "
            f"missing={sorted(expected_groups-observed_groups)}, "
            f"extra={sorted(observed_groups-expected_groups)}"
        )

    expected_source_donor = {
        value["source_item_id"]: value["donor_id"]
        for value in locked["expected_source_items"]
    }
    item_donor: dict[str, str] = {}
    source_donor: dict[str, str] = {}
    by_item: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in checked:
        prior = item_donor.setdefault(record["item_id"], record["donor_id"])
        if prior != record["donor_id"]:
            raise CoherentReadoutError("item_id occurs under multiple donors")
        source_prior = source_donor.setdefault(
            record["source_item_id"], record["donor_id"]
        )
        if source_prior != record["donor_id"]:
            raise CoherentReadoutError("source_item_id occurs under multiple donors")
        if expected_source_donor.get(record["source_item_id"]) != record["donor_id"]:
            raise CoherentReadoutError("record source item/donor does not match design")
        key = (
            record["donor_id"],
            record["source_item_id"],
            record["item_id"],
            record["readout_id"],
            record["input_family"],
        )
        by_item[key].append(record)

    expected_topology = {
        (source_item, family, readout)
        for source_item in expected_source_donor
        for family in locked["required_input_families"]
        for readout in locked["required_readouts"]
    }
    observed_topology: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    item_ids_by_source_family: dict[tuple[str, str], set[str]] = defaultdict(set)
    gene_hashes_by_source_family: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in checked:
        topology = (
            record["source_item_id"],
            record["input_family"],
            record["readout_id"],
        )
        observed_topology[topology].add(record["item_id"])
        source_family = (record["source_item_id"], record["input_family"])
        item_ids_by_source_family[source_family].add(record["item_id"])
        gene_hashes_by_source_family[source_family].add(record["gene_sentence_sha256"])
    if set(observed_topology) != expected_topology or any(
        len(item_ids) != 1 for item_ids in observed_topology.values()
    ):
        raise CoherentReadoutError("source-item/readout/input-family topology mismatch")
    if any(len(item_ids) != 1 for item_ids in item_ids_by_source_family.values()):
        raise CoherentReadoutError("readouts do not share one concrete item per source/family")
    if any(len(hashes) != 1 for hashes in gene_hashes_by_source_family.values()):
        raise CoherentReadoutError("readouts do not share one gene sentence per source/family")

    item_rows: list[dict[str, Any]] = []
    for (donor, source_item, item, readout, family), rows in sorted(by_item.items()):
        forms = {(row["order"], row["mapping"]): row for row in rows}
        if len(rows) != 4 or set(forms) != set(FORM_KEYS):
            raise CoherentReadoutError(
                f"item {item}/{readout}/{family} does not contain each form once"
            )
        for hash_field in (
            "user_prompt_sha256",
            "prompt_sha256",
            "execution_input_sha256",
            "forward_trace_sha256",
        ):
            if len({row[hash_field] for row in rows}) != 4:
                raise CoherentReadoutError(
                    f"item {item}/{readout}/{family} does not have four distinct {hash_field} values"
                )
        if len({row["source_fixture_record_id"] for row in rows}) != 1:
            raise CoherentReadoutError("one form group spans multiple fixture records")
        scores = {key: float(forms[key]["s"]) for key in FORM_KEYS}
        pf_px = scores[("positive_first", "positive_is_x")]
        pf_py = scores[("positive_first", "positive_is_y")]
        nf_px = scores[("negative_first", "positive_is_x")]
        nf_py = scores[("negative_first", "positive_is_y")]
        mean_s = float(np.mean(list(scores.values())))
        score_range = max(scores.values()) - min(scores.values())
        score_values = np.asarray(list(scores.values()), dtype=float)
        score_tolerance = _comparison_tolerance(score_values)
        strong = abs(mean_s) + score_tolerance >= locked["strong_score_threshold"]
        if mean_s > score_tolerance:
            same_sign = all(value > score_tolerance for value in scores.values())
        elif mean_s < -score_tolerance:
            same_sign = all(value < -score_tolerance for value in scores.values())
        else:
            same_sign = False
        item_rows.append(
            {
                "donor_id": donor,
                "source_item_id": source_item,
                "item_id": item,
                "readout_id": readout,
                "input_family": family,
                "mean_s": mean_s,
                "O": 0.5 * (pf_px + pf_py - nf_px - nf_py),
                "R": 0.5 * (pf_px + nf_px - pf_py - nf_py),
                "I": 0.5 * ((pf_px - pf_py) - (nf_px - nf_py)),
                "score_range": score_range,
                "range_pass": bool(
                    score_range <= locked["item_range_margin"] + score_tolerance
                ),
                "strong_item": bool(strong),
                "strong_item_same_sign": bool((not strong) or same_sign),
                "format_adherent_records": sum(
                    bool(row["format_adherent"]) for row in rows
                ),
                "record_count": len(rows),
                "max_coherence_residual": max(
                    float(row["coherence_residual"]) for row in rows
                ),
                "mean_two_token_probability_mass": float(
                    np.mean([row["two_token_probability_mass"] for row in rows])
                ),
            }
        )

    donors = sorted({row["donor_id"] for row in item_rows})
    if len(donors) < 2 or len(donors) > 20:
        raise CoherentReadoutError("analysis requires 2-20 donors")
    if donors != locked["expected_donor_ids"]:
        raise CoherentReadoutError("observed donor registry does not match design")
    if locked["mode"] == "confirmatory" and len(donors) != locked[
        "expected_confirmatory_donors"
    ]:
        raise CoherentReadoutError("confirmatory donor count does not match design")

    group_donors: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in item_rows:
        group_donors[(row["readout_id"], row["input_family"])].add(row["donor_id"])
    if any(group_donors[group] != set(donors) for group in expected_groups):
        raise CoherentReadoutError("each readout/input family must contain every donor")

    results: dict[str, Any] = {}
    for readout, family in sorted(expected_groups):
        rows = [
            row
            for row in item_rows
            if row["readout_id"] == readout and row["input_family"] == family
        ]
        donor_effects: dict[str, dict[str, float]] = {}
        for donor in donors:
            donor_rows = [row for row in rows if row["donor_id"] == donor]
            donor_effects[donor] = {
                estimand: float(np.mean([row[estimand] for row in donor_rows]))
                for estimand in ("O", "R", "I")
            }
        equivalence = {
            estimand: equivalence_summary(
                [donor_effects[donor][estimand] for donor in donors],
                margin=locked["equivalence_margin"],
                alpha=locked["alpha"],
            )
            for estimand in ("O", "R", "I")
        }

        record_count = sum(row["record_count"] for row in rows)
        format_count = sum(row["format_adherent_records"] for row in rows)
        per_donor_format: dict[str, float] = {}
        for donor in donors:
            donor_rows = [row for row in rows if row["donor_id"] == donor]
            per_donor_format[donor] = _fraction(
                sum(row["format_adherent_records"] for row in donor_rows),
                sum(row["record_count"] for row in donor_rows),
            )
        overall_format = _fraction(format_count, record_count)
        range_pass_count = int(sum(bool(row["range_pass"]) for row in rows))
        range_required_count = math.ceil(locked["item_range_pass_min"] * len(rows))
        range_fraction = _fraction(range_pass_count, len(rows))
        strong_rows = [row for row in rows if row["strong_item"]]
        strong_sign_pass = all(row["strong_item_same_sign"] for row in strong_rows)
        extraction_pass = all(
            row["max_coherence_residual"] <= locked["coherence_tolerance"]
            for row in rows
        )
        format_pass = bool(
            overall_format >= locked["format_per_group_min"]
            and all(
                fraction >= locked["format_per_donor_min"]
                for fraction in per_donor_format.values()
            )
        )
        item_guardrail_pass = bool(
            range_pass_count >= range_required_count and strong_sign_pass
        )
        group_pass = bool(
            extraction_pass
            and format_pass
            and item_guardrail_pass
            and all(summary["pass"] for summary in equivalence.values())
        )
        results[f"{readout}::{family}"] = {
            "readout_id": readout,
            "input_family": family,
            "n_donors": len(donors),
            "n_items": len(rows),
            "n_records": record_count,
            "extraction_coherence": {
                "max_residual": max(row["max_coherence_residual"] for row in rows),
                "pass": extraction_pass,
            },
            "format_adherence": {
                "adherent_records": format_count,
                "fraction": overall_format,
                "required_group": locked["format_per_group_min"],
                "per_donor_fraction": per_donor_format,
                "required_per_donor": locked["format_per_donor_min"],
                "mean_two_token_probability_mass": float(
                    np.mean([row["mean_two_token_probability_mass"] for row in rows])
                ),
                "pass": format_pass,
            },
            "nuisance_equivalence": equivalence,
            "item_guardrail": {
                "range_pass_fraction": range_fraction,
                "range_pass_count": range_pass_count,
                "range_required_count": range_required_count,
                "required_fraction": locked["item_range_pass_min"],
                "strong_item_count": len(strong_rows),
                "all_strong_items_same_sign": strong_sign_pass,
                "pass": item_guardrail_pass,
            },
            "donor_effects": donor_effects,
            "pass": group_pass,
        }

    global_format_count = sum(bool(record["format_adherent"]) for record in checked)
    global_format_fraction = _fraction(global_format_count, len(checked))
    global_format_pass = global_format_fraction >= locked["format_overall_min"]
    all_group_format_pass = all(
        group["format_adherence"]["pass"] for group in results.values()
    )
    level0_pass = bool(
        global_format_pass and all(group["pass"] for group in results.values())
    )
    if locked["mode"] == "development":
        if level0_pass:
            status = (
                "DEVELOPMENT_LEVEL0_PASS_NOT_CONFIRMATORY"
                if locked["margin_lock_status"] == "phase0_qualified"
                else "DEVELOPMENT_LEVEL0_CANDIDATE_PASS_MARGIN_NOT_QUALIFIED"
            )
        elif not global_format_pass or not all_group_format_pass:
            status = "DEVELOPMENT_READOUT_FORMAT_INVALID"
        else:
            status = "DEVELOPMENT_LEVEL0_FAIL"
    else:
        if level0_pass:
            status = "LEVEL0_PASS"
        elif not global_format_pass or not all_group_format_pass:
            status = "READOUT_FORMAT_INVALID"
        else:
            status = "READOUT_INVALID"
    sorted_records = sorted(validated, key=lambda row: row["record_id"])
    return {
        "artifact_type": "groundbench.level0_coherent_binary_readout",
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "mode": locked["mode"],
        "margin_lock_status": locked["margin_lock_status"],
        "level0_pass": level0_pass,
        "design": locked,
        "design_sha256": canonical_sha256(locked),
        "raw_records_sha256": canonical_sha256(sorted_records),
        "full_vocab_sidecar_sha256": full_vocab_sidecar_sha256,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_records": len(checked),
        "n_items": len(item_rows),
        "n_donors": len(donors),
        "donor_ids": donors,
        "validation": {
            "expected_records": len(expected_record_ids),
            "observed_records": len(observed_record_ids),
            "missing_records": [],
            "unexpected_records": [],
            "finite_logits": True,
            "max_complement_error": max(
                float(record["coherence_residual"]) for record in checked
            ),
        },
        "global_format_adherence": {
            "adherent_records": global_format_count,
            "record_count": len(checked),
            "fraction": global_format_fraction,
            "required": locked["format_overall_min"],
            "pass": global_format_pass,
        },
        "groups": results,
        "claim_boundary": (
            "Level 0 measures only whether the two-token interface satisfies the "
            "registered conditional measurement gates. It does not adjudicate "
            "calibration, biological validity, knowledge, integration, gain, activation, "
            "or a physical law. Exact sign-flip conclusions remain conditional on the "
            "required but unverified donor-effect sign-symmetry assumption."
        ),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Coherent binary readout: Level-0 report",
        "",
        f"Status: **{result['status']}**",
        "",
        (
            f"Mode `{result['mode']}`; {result['n_donors']} donors, "
            f"{result['n_items']} items, {result['n_records']} records."
        ),
        "",
        (
            "Global full-vocabulary format adherence: "
            f"**{result['global_format_adherence']['fraction']:.3f}** "
            f"(required >= {result['global_format_adherence']['required']:.3f})."
        ),
        "",
        "| readout / family | format | O eq | R eq | I eq | item guardrail | result |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in sorted(result["groups"].items()):
        eq = group["nuisance_equivalence"]
        lines.append(
            f"| `{name}` | {group['format_adherence']['fraction']:.3f} | "
            f"{'PASS' if eq['O']['pass'] else 'FAIL'} | "
            f"{'PASS' if eq['R']['pass'] else 'FAIL'} | "
            f"{'PASS' if eq['I']['pass'] else 'FAIL'} | "
            f"{'PASS' if group['item_guardrail']['pass'] else 'FAIL'} | "
            f"**{'PASS' if group['pass'] else 'FAIL'}** |"
        )
    lines.extend(
        [
            "",
            f"Design SHA-256: `{result['design_sha256']}`.",
            "",
            f"Raw-record canonical SHA-256: `{result['raw_records_sha256']}`.",
            "",
            f"Analyzer code SHA-256: `{result['analysis_code_sha256']}`.",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CoherentReadoutError(f"JSONL line {line_number} is not an object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--full-logits-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    full_logits = np.load(args.full_logits_npy, allow_pickle=False)
    result = analyze_level0(_load_jsonl(args.records), design, full_logits)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()

"""Analyze the prospective V4 behavioral-deconfounding experiment.

The analyzer is deliberately behavior-only.  It executes no model forward,
selects no activation layer or direction, and grants no downstream authority.
It replays the runner's immutable artifact contract before applying the frozen
component-localization decision tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import run_coherent_readout_v4_behavioral_deconfounding as runner
except ImportError:  # direct execution from eval/, or runner landing concurrently
    try:
        import run_coherent_readout_v4_behavioral_deconfounding as runner
    except ImportError:
        runner = None  # type: ignore[assignment]


ANALYSIS_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-analysis-v1"
ANALYSIS_MANIFEST_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-analysis-manifest-v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260805
MODEL_VOCAB_SIZE = 151_936
WORLD_IDS = tuple(f"behavior_world_{index:03d}" for index in range(1, 9))

FAMILY_ORDER = ("property_retrieval", "codebook_lookup", "composition")
FAMILY_TOTALS = {
    "property_retrieval": 64,
    "codebook_lookup": 128,
    "composition": 256,
}
ROWS_PER_WORLD = {
    "property_retrieval": 8,
    "codebook_lookup": 16,
    "composition": 32,
}
FAMILY_ANSWER_LABELS = {
    "property_retrieval": ("P", "Q"),
    "codebook_lookup": ("X", "Y"),
    "composition": ("X", "Y"),
}
FACTOR_LEVELS = {
    "target_property": ("P", "Q"),
    "mapping_id": ("identity", "swapped"),
    "target_fact_order": ("target_first", "target_second"),
    "rule_order": ("p_rule_first", "q_rule_first"),
    "xy_option_order": ("x_then_y", "y_then_x"),
    "pq_option_order": ("p_then_q", "q_then_p"),
}
STRATUM_FACTORS = {
    "property_retrieval": ("target_property", "target_fact_order", "option_order"),
    "codebook_lookup": ("target_property", "mapping_id", "rule_order", "option_order"),
    "composition": (
        "target_property",
        "mapping_id",
        "target_fact_order",
        "rule_order",
        "option_order",
    ),
}
BUNDLE_FACTORS = {
    "property_retrieval": ("world_id", "target_property"),
    "codebook_lookup": ("world_id", "target_property", "mapping_id"),
    "composition": ("world_id", "target_property", "mapping_id"),
}
BUNDLE_PERMUTATIONS = {
    "property_retrieval": 4,
    "codebook_lookup": 4,
    "composition": 8,
}

FAMILY_OVERALL_THRESHOLD = 0.95
COMPONENT_STRATUM_THRESHOLD = 0.95
COMPOSITION_STRATUM_THRESHOLD = 0.90
BUNDLE_THRESHOLD = 0.95
WORLD_THRESHOLD = 0.90
BOOTSTRAP_LOWER_THRESHOLD = 0.90
CHANNEL_TOP1_THRESHOLD = 0.95
CHANNEL_MASS_THRESHOLD = 0.95
DOMINANT_MATCH_THRESHOLD = 0.90
DOMINANT_ADVANTAGE_THRESHOLD = 0.10

STATUS_ENGINEERING_INVALID = "V4_ENGINEERING_INVALID"
STATUS_BOTH_COMPONENTS_FAIL = "V4_RETRIEVAL_AND_LOOKUP_COMPONENTS_FAIL"
STATUS_RETRIEVAL_FAIL = "V4_RETRIEVAL_COMPONENT_FAIL"
STATUS_LOOKUP_FAIL = "V4_LOOKUP_COMPONENT_FAIL"
STATUS_COMPOSITION_FAIL = "V4_COMPOSITION_FAIL_COMPONENTS_PASS"
STATUS_QUALIFIED = "V4_BEHAVIORAL_COMPOSITION_QUALIFIED"
FINAL_STATUSES = (
    STATUS_ENGINEERING_INVALID,
    STATUS_BOTH_COMPONENTS_FAIL,
    STATUS_RETRIEVAL_FAIL,
    STATUS_LOOKUP_FAIL,
    STATUS_COMPOSITION_FAIL,
    STATUS_QUALIFIED,
)

POLICY_INTENDED = "intended_compositional_rule"
POLICY_V3 = "frozen_v3_heuristic"
POLICY_LAST_OPTION = "last_displayed_option"
POLICY_FIRST_RULE = "first_displayed_codebook_rule_output"
POLICY_CONSTANT_Y = "constant_y"
POLICY_CONSTANT_X = "constant_x"
POLICY_ORDER = (
    POLICY_V3,
    POLICY_LAST_OPTION,
    POLICY_FIRST_RULE,
    POLICY_CONSTANT_Y,
    POLICY_CONSTANT_X,
)

CLAIM_BOUNDARIES = {
    "supported_scope": "synthetic_symbolic_behavior_in_one_locked_model",
    "component_localization_only": True,
    "causal_mechanism_inference": "forbidden",
    "activation_gap_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "biological_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
    "heuristic_dominance_is_mechanistic_evidence": False,
    "activation_axis_or_layer_selection": "not_performed",
    "v3_claim_reinterpretation": "forbidden",
}

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    runner.RESULT_ROOT
    if runner is not None and hasattr(runner, "RESULT_ROOT")
    else ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v4_behavioral_deconfounding"
    / "qwen2.5-1.5b-instruct"
)
DEFAULT_ANALYSIS = RESULT_ROOT / "behavior_analysis.json"
DEFAULT_MARKDOWN = RESULT_ROOT / "analysis.md"
DEFAULT_ANALYSIS_MANIFEST = RESULT_ROOT / "analysis_manifest.json"


class BehavioralDeconfoundingAnalysisError(ValueError):
    """Raised when frozen artifacts or the V4 analysis contract do not replay."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BehavioralDeconfoundingAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BehavioralDeconfoundingAnalysisError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BehavioralDeconfoundingAnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BehavioralDeconfoundingAnalysisError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping_answer(target_property: str, mapping_id: str) -> str:
    if target_property not in FACTOR_LEVELS["target_property"]:
        raise BehavioralDeconfoundingAnalysisError("target_property changed")
    if mapping_id not in FACTOR_LEVELS["mapping_id"]:
        raise BehavioralDeconfoundingAnalysisError("mapping_id changed")
    if mapping_id == "identity":
        return "X" if target_property == "P" else "Y"
    return "Y" if target_property == "P" else "X"


def _expected_options(family: str, option_order: str) -> tuple[str, str]:
    if family == "property_retrieval":
        if option_order == "p_then_q":
            return ("P", "Q")
        if option_order == "q_then_p":
            return ("Q", "P")
    else:
        if option_order == "x_then_y":
            return ("X", "Y")
        if option_order == "y_then_x":
            return ("Y", "X")
    raise BehavioralDeconfoundingAnalysisError("option_order changed")


def _first_rule_output(mapping_id: str, rule_order: str) -> str:
    if rule_order == "p_rule_first":
        return _mapping_answer("P", mapping_id)
    if rule_order == "q_rule_first":
        return _mapping_answer("Q", mapping_id)
    raise BehavioralDeconfoundingAnalysisError("rule_order changed")


def _v3_heuristic(target_property: str, mapping_id: str, target_fact_order: str) -> str:
    return "X" if target_property == "P" and mapping_id == "identity" and target_fact_order == "target_first" else "Y"


def _expected_registry_ids(
    *,
    family: str,
    world_id: str,
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
    rule_order: str | None,
    option_order: str,
) -> tuple[str, str]:
    property_slug = target_property.lower()
    if family == "property_retrieval":
        return (
            f"retrieval:{world_id}:p-{property_slug}",
            f"retrieval:p-{property_slug}:f-{target_fact_order}:o-{option_order}",
        )
    if family == "codebook_lookup":
        return (
            f"lookup:{world_id}:p-{property_slug}:m-{mapping_id}",
            f"lookup:p-{property_slug}:m-{mapping_id}:r-{rule_order}:o-{option_order}",
        )
    return (
        f"composition:{world_id}:p-{property_slug}:m-{mapping_id}",
        (f"composition:p-{property_slug}:m-{mapping_id}:f-{target_fact_order}:r-{rule_order}:o-{option_order}"),
    )


def _coerce_label_token_map(
    answer_labels: tuple[str, str],
    raw_token_ids: Any,
) -> dict[str, int]:
    if isinstance(raw_token_ids, Mapping):
        if set(raw_token_ids) != set(answer_labels):
            raise BehavioralDeconfoundingAnalysisError("label_token_ids keys changed")
        result = {label: _integer(raw_token_ids[label], f"label_token_ids[{label}]") for label in answer_labels}
    elif isinstance(raw_token_ids, Sequence) and not isinstance(raw_token_ids, (str, bytes)):
        if len(raw_token_ids) != 2:
            raise BehavioralDeconfoundingAnalysisError("label_token_ids length changed")
        result = {
            label: _integer(token_id, f"label_token_ids[{label}]")
            for label, token_id in zip(answer_labels, raw_token_ids, strict=True)
        }
    else:
        raise BehavioralDeconfoundingAnalysisError("label_token_ids changed")
    if len(set(result.values())) != 2:
        raise BehavioralDeconfoundingAnalysisError("label token IDs must be distinct")
    return result


def _normalize_diagnostics(
    raw: Mapping[str, Any],
    *,
    answer_labels: tuple[str, str],
    label_token_ids: Mapping[str, int],
    correct_answer: str,
) -> dict[str, Any]:
    required = {
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
        "full_vocab_logits_sha256",
    }
    if not required.issubset(raw):
        missing = sorted(required - set(raw))
        raise BehavioralDeconfoundingAnalysisError(f"diagnostics missing fields: {missing}")
    raw_logits = raw["label_logits"]
    if not isinstance(raw_logits, Sequence) or isinstance(raw_logits, (str, bytes)) or len(raw_logits) != 2:
        raise BehavioralDeconfoundingAnalysisError("label_logits must have length two")
    logits = tuple(_finite(value, "label_logits") for value in raw_logits)
    raw_by_text = raw["label_logit_by_text"]
    if not isinstance(raw_by_text, Mapping) or set(raw_by_text) != set(answer_labels):
        raise BehavioralDeconfoundingAnalysisError("label_logit_by_text keys changed")
    by_text = {label: _finite(raw_by_text[label], f"logit[{label}]") for label in answer_labels}
    if any(abs(by_text[label] - logits[index]) > 1e-7 for index, label in enumerate(answer_labels)):
        raise BehavioralDeconfoundingAnalysisError("label logit representations disagree")
    first_margin = _finite(raw["first_minus_second_margin"], "first_minus_second_margin")
    if abs(first_margin - (logits[0] - logits[1])) > 1e-7:
        raise BehavioralDeconfoundingAnalysisError("first-minus-second margin does not reconstruct")
    incorrect_answer = answer_labels[1] if correct_answer == answer_labels[0] else answer_labels[0]
    correct_margin = _finite(raw["correct_minus_incorrect_margin"], "correct_minus_incorrect_margin")
    if abs(correct_margin - (by_text[correct_answer] - by_text[incorrect_answer])) > 1e-7:
        raise BehavioralDeconfoundingAnalysisError("correct-minus-incorrect margin does not reconstruct")
    logsumexp = _finite(raw["full_vocab_logsumexp"], "full_vocab_logsumexp")
    label_mass = _finite(raw["label_probability_mass"], "label_probability_mass")
    expected_mass = math.exp(float(np.logaddexp(logits[0], logits[1])) - logsumexp)
    if not 0.0 <= label_mass <= 1.0 or abs(label_mass - expected_mass) > 1e-10:
        raise BehavioralDeconfoundingAnalysisError("label probability mass does not reconstruct")
    greedy_token_id = _integer(raw["greedy_token_id"], "greedy_token_id")
    greedy_logit = _finite(raw["greedy_logit"], "greedy_logit")
    maximum_ids = raw["maximum_token_ids"]
    if (
        not isinstance(maximum_ids, list)
        or not maximum_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in maximum_ids)
        or maximum_ids != sorted(set(maximum_ids))
    ):
        raise BehavioralDeconfoundingAnalysisError("maximum_token_ids changed")
    tie_count = _integer(raw["maximum_tie_count"], "maximum_tie_count", minimum=1)
    if tie_count != len(maximum_ids) or greedy_token_id != maximum_ids[0]:
        raise BehavioralDeconfoundingAnalysisError("global maximum metadata is inconsistent")
    if logsumexp + 1e-7 < greedy_logit or greedy_logit + 1e-7 < max(logits):
        raise BehavioralDeconfoundingAnalysisError("global logit ordering is inconsistent")
    for label, token_id in label_token_ids.items():
        is_global_max = by_text[label] == greedy_logit
        if (token_id in maximum_ids) != is_global_max:
            raise BehavioralDeconfoundingAnalysisError("label/global-maximum membership is inconsistent")
    digest = _sha256(raw["full_vocab_logits_sha256"], "full_vocab_logits_sha256")
    return {
        "label_logits": list(logits),
        "label_logit_by_text": by_text,
        "first_minus_second_margin": first_margin,
        "correct_minus_incorrect_margin": correct_margin,
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": label_mass,
        "greedy_token_id": greedy_token_id,
        "greedy_logit": greedy_logit,
        "maximum_token_ids": list(maximum_ids),
        "maximum_tie_count": tie_count,
        "full_vocab_logits_sha256": digest,
    }


def _diagnostics_from_full_vocab(
    raw_row: np.ndarray,
    *,
    answer_labels: Sequence[str],
    label_token_ids: Sequence[int],
    correct_answer: str,
) -> dict[str, Any]:
    """Independently reconstruct runner diagnostics from one raw f32 row."""

    value = np.asarray(raw_row)
    if value.dtype != np.dtype("<f4") or value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise BehavioralDeconfoundingAnalysisError("raw full-vocabulary row changed")
    labels = list(answer_labels)
    token_ids = list(label_token_ids)
    if tuple(labels) not in {("X", "Y"), ("P", "Q")}:
        raise BehavioralDeconfoundingAnalysisError("raw diagnostic label pair changed")
    if (
        len(token_ids) != 2
        or len(set(token_ids)) != 2
        or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids)
        or any(token_id < 0 or token_id >= MODEL_VOCAB_SIZE for token_id in token_ids)
        or correct_answer not in labels
    ):
        raise BehavioralDeconfoundingAnalysisError("raw diagnostic token registry changed")
    label_logits = [float(value[token_id]) for token_id in token_ids]
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    row64 = value.astype(np.float64)
    peak = float(row64.max())
    logsumexp = peak + math.log(float(np.exp(row64 - peak).sum()))
    label_logsumexp = float(np.logaddexp(label_logits[0], label_logits[1]))
    correct_index = labels.index(correct_answer)
    incorrect_index = 1 - correct_index
    correct_token_id = token_ids[correct_index]
    greedy_token_id = maximum_ids[0]
    return {
        "label_logits": label_logits,
        "label_logit_by_text": {label: label_logits[index] for index, label in enumerate(labels)},
        "first_minus_second_margin": label_logits[0] - label_logits[1],
        "correct_minus_incorrect_margin": (label_logits[correct_index] - label_logits[incorrect_index]),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "greedy_token_id": greedy_token_id,
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "correct_is_global_maximum": correct_token_id in maximum_ids,
        "unique_global_argmax_is_correct": (len(maximum_ids) == 1 and maximum_ids[0] == correct_token_id),
        "greedy_matches_correct": greedy_token_id == correct_token_id,
        "full_vocab_logits_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest(),
    }


def _merged_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Merge an optional frozen-cell subobject without hiding disagreements."""

    merged: dict[str, Any] = {}
    cell = raw.get("cell")
    if cell is not None:
        if not isinstance(cell, Mapping):
            raise BehavioralDeconfoundingAnalysisError("record cell must be an object")
        merged.update(cell)
    for key, value in raw.items():
        if key == "cell":
            continue
        if key in merged and merged[key] != value:
            raise BehavioralDeconfoundingAnalysisError(f"record/cell field disagreement: {key}")
        merged[key] = value
    return merged


def _normalize_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = _merged_record(raw)
    required = {
        "cell_id",
        "world_id",
        "family_id",
        "stratum_id",
        "target_property",
        "mapping_id",
        "target_fact_order",
        "rule_order",
        "option_order",
        "answer_labels",
        "displayed_options",
        "correct_answer",
        "correct_option_position",
        "v3_heuristic_answer",
        "last_option_heuristic_answer",
        "first_rule_output_heuristic_answer",
        "semantic_bundle_id",
        "permutation_index",
        "label_token_ids",
        "diagnostics",
    }
    if not required.issubset(row):
        missing = sorted(required - set(row))
        raise BehavioralDeconfoundingAnalysisError(f"record missing fields: {missing}")
    family = row["family_id"]
    if family not in FAMILY_ORDER:
        raise BehavioralDeconfoundingAnalysisError("family_id changed")
    if not isinstance(row["cell_id"], str) or not row["cell_id"]:
        raise BehavioralDeconfoundingAnalysisError("cell_id changed")
    if row["world_id"] not in WORLD_IDS:
        raise BehavioralDeconfoundingAnalysisError("world_id changed")
    target_property = row["target_property"]
    if target_property not in FACTOR_LEVELS["target_property"]:
        raise BehavioralDeconfoundingAnalysisError("target_property changed")
    mapping_id = row["mapping_id"]
    target_fact_order = row["target_fact_order"]
    rule_order = row["rule_order"]
    option_order = row["option_order"]
    if family == "property_retrieval":
        if mapping_id is not None or rule_order is not None:
            raise BehavioralDeconfoundingAnalysisError("retrieval includes forbidden mapping/rule factors")
        if target_fact_order not in FACTOR_LEVELS["target_fact_order"]:
            raise BehavioralDeconfoundingAnalysisError("retrieval fact order changed")
        if option_order not in FACTOR_LEVELS["pq_option_order"]:
            raise BehavioralDeconfoundingAnalysisError("retrieval option order changed")
    elif family == "codebook_lookup":
        if target_fact_order is not None:
            raise BehavioralDeconfoundingAnalysisError("lookup includes a forbidden fact-order factor")
        if mapping_id not in FACTOR_LEVELS["mapping_id"] or rule_order not in FACTOR_LEVELS["rule_order"]:
            raise BehavioralDeconfoundingAnalysisError("lookup mapping/rule factors changed")
        if option_order not in FACTOR_LEVELS["xy_option_order"]:
            raise BehavioralDeconfoundingAnalysisError("lookup option order changed")
    else:
        if (
            mapping_id not in FACTOR_LEVELS["mapping_id"]
            or target_fact_order not in FACTOR_LEVELS["target_fact_order"]
            or rule_order not in FACTOR_LEVELS["rule_order"]
            or option_order not in FACTOR_LEVELS["xy_option_order"]
        ):
            raise BehavioralDeconfoundingAnalysisError("composition factor levels changed")
    raw_labels = row["answer_labels"]
    raw_options = row["displayed_options"]
    if not isinstance(raw_labels, list) or tuple(raw_labels) != FAMILY_ANSWER_LABELS[family]:
        raise BehavioralDeconfoundingAnalysisError("answer_labels changed")
    answer_labels = tuple(raw_labels)
    if not isinstance(raw_options, list) or tuple(raw_options) != _expected_options(family, option_order):
        raise BehavioralDeconfoundingAnalysisError("displayed_options do not reconstruct")
    displayed_options = tuple(raw_options)
    correct_answer = row["correct_answer"]
    expected_answer = (
        target_property if family == "property_retrieval" else _mapping_answer(target_property, mapping_id)
    )
    if correct_answer != expected_answer:
        raise BehavioralDeconfoundingAnalysisError("correct answer violates the intended ledger")
    expected_position = "first" if displayed_options[0] == correct_answer else "last"
    if row["correct_option_position"] != expected_position:
        raise BehavioralDeconfoundingAnalysisError("correct_option_position does not reconstruct")
    expected_v3 = _v3_heuristic(target_property, mapping_id, target_fact_order) if family == "composition" else None
    expected_first_rule = (
        _first_rule_output(mapping_id, rule_order) if family in {"codebook_lookup", "composition"} else None
    )
    if row["v3_heuristic_answer"] != expected_v3:
        raise BehavioralDeconfoundingAnalysisError("stored V3 heuristic answer does not reconstruct")
    if row["last_option_heuristic_answer"] != displayed_options[-1]:
        raise BehavioralDeconfoundingAnalysisError("stored last-option heuristic does not reconstruct")
    if row["first_rule_output_heuristic_answer"] != expected_first_rule:
        raise BehavioralDeconfoundingAnalysisError("stored first-rule heuristic does not reconstruct")
    expected_bundle_id, expected_stratum_id = _expected_registry_ids(
        family=family,
        world_id=row["world_id"],
        target_property=target_property,
        mapping_id=mapping_id,
        target_fact_order=target_fact_order,
        rule_order=rule_order,
        option_order=option_order,
    )
    if row["semantic_bundle_id"] != expected_bundle_id:
        raise BehavioralDeconfoundingAnalysisError("semantic_bundle_id does not reconstruct")
    if row["stratum_id"] != expected_stratum_id:
        raise BehavioralDeconfoundingAnalysisError("stratum_id does not reconstruct")
    permutation_index = _integer(row["permutation_index"], "permutation_index")
    label_token_ids = _coerce_label_token_map(answer_labels, row["label_token_ids"])
    diagnostics = row["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise BehavioralDeconfoundingAnalysisError("diagnostics must be an object")
    normalized_diagnostics = _normalize_diagnostics(
        diagnostics,
        answer_labels=answer_labels,
        label_token_ids=label_token_ids,
        correct_answer=correct_answer,
    )
    return {
        "cell_id": row["cell_id"],
        "world_id": row["world_id"],
        "family_id": family,
        "stratum_id": expected_stratum_id,
        "target_property": target_property,
        "mapping_id": mapping_id,
        "target_fact_order": target_fact_order,
        "rule_order": rule_order,
        "option_order": option_order,
        "answer_labels": list(answer_labels),
        "displayed_options": list(displayed_options),
        "correct_answer": correct_answer,
        "correct_option_position": expected_position,
        "v3_heuristic_answer": expected_v3,
        "last_option_heuristic_answer": displayed_options[-1],
        "first_rule_output_heuristic_answer": expected_first_rule,
        "semantic_bundle_id": expected_bundle_id,
        "permutation_index": permutation_index,
        "label_token_ids": label_token_ids,
        "diagnostics": normalized_diagnostics,
    }


def _stratum_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in STRATUM_FACTORS[row["family_id"]])


def _stratum_id(row: Mapping[str, Any]) -> str:
    return "__".join(f"{field}={row[field]}" for field in STRATUM_FACTORS[row["family_id"]])


def _bundle_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in BUNDLE_FACTORS[row["family_id"]])


def _validate_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 448:
        raise BehavioralDeconfoundingAnalysisError("behavior row count must equal 448")
    cell_ids = [row["cell_id"] for row in rows]
    if len(set(cell_ids)) != len(cell_ids):
        raise BehavioralDeconfoundingAnalysisError("behavior cell IDs are duplicated")
    if {row["world_id"] for row in rows} != set(WORLD_IDS):
        raise BehavioralDeconfoundingAnalysisError("behavior world registry changed")
    family_counts = Counter(row["family_id"] for row in rows)
    if dict(family_counts) != FAMILY_TOTALS:
        raise BehavioralDeconfoundingAnalysisError("family totals changed")
    per_world: dict[str, dict[str, int]] = {}
    full_strata: dict[str, dict[str, int]] = {}
    bundle_counts: dict[str, dict[str, int]] = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family_id"] == family]
        world_counts = Counter(row["world_id"] for row in family_rows)
        if set(world_counts) != set(WORLD_IDS) or set(world_counts.values()) != {ROWS_PER_WORLD[family]}:
            raise BehavioralDeconfoundingAnalysisError(f"{family} per-world coverage changed")
        per_world[family] = dict(sorted(world_counts.items()))
        stratum_counts = Counter(_stratum_tuple(row) for row in family_rows)
        expected_n_strata = ROWS_PER_WORLD[family]
        if len(stratum_counts) != expected_n_strata or set(stratum_counts.values()) != {len(WORLD_IDS)}:
            raise BehavioralDeconfoundingAnalysisError(f"{family} full-factorial strata changed")
        for world_id in WORLD_IDS:
            world_strata = Counter(_stratum_tuple(row) for row in family_rows if row["world_id"] == world_id)
            if set(world_strata) != set(stratum_counts) or set(world_strata.values()) != {1}:
                raise BehavioralDeconfoundingAnalysisError(f"{family} world factorial is incomplete")
        full_strata[family] = {
            _stratum_id(next(row for row in family_rows if _stratum_tuple(row) == key)): count
            for key, count in sorted(stratum_counts.items())
        }
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in family_rows:
            grouped[_bundle_tuple(row)].append(row)
        expected_bundles = FAMILY_TOTALS[family] // BUNDLE_PERMUTATIONS[family]
        if len(grouped) != expected_bundles or {len(values) for values in grouped.values()} != {
            BUNDLE_PERMUTATIONS[family]
        }:
            raise BehavioralDeconfoundingAnalysisError(f"{family} semantic bundles changed")
        stored_to_reconstructed: dict[str, tuple[Any, ...]] = {}
        for reconstructed, members in grouped.items():
            stored_ids = {str(member["semantic_bundle_id"]) for member in members}
            permutation_indices = [member["permutation_index"] for member in members]
            if len(stored_ids) != 1 or set(permutation_indices) != set(range(BUNDLE_PERMUTATIONS[family])):
                raise BehavioralDeconfoundingAnalysisError(f"{family} bundle metadata changed")
            stored_id = next(iter(stored_ids))
            if stored_id in stored_to_reconstructed and stored_to_reconstructed[stored_id] != reconstructed:
                raise BehavioralDeconfoundingAnalysisError("semantic bundle ID aliases multiple bundles")
            stored_to_reconstructed[stored_id] = reconstructed
        bundle_counts[family] = {
            stored_id: BUNDLE_PERMUTATIONS[family] for stored_id in sorted(stored_to_reconstructed)
        }
    return {
        "expected_total_rows": 448,
        "observed_total_rows": len(rows),
        "expected_world_ids": list(WORLD_IDS),
        "family_counts": dict(family_counts),
        "per_world_family_counts": per_world,
        "full_factorial_stratum_counts": full_strata,
        "semantic_bundle_counts": bundle_counts,
        "exact_contract_pass": True,
    }


def _global_prediction(row: Mapping[str, Any]) -> str | None:
    diagnostics = row["diagnostics"]
    maximum_ids = diagnostics["maximum_token_ids"]
    if len(maximum_ids) != 1:
        return None
    inverse = {token_id: label for label, token_id in row["label_token_ids"].items()}
    return inverse.get(maximum_ids[0])


def _two_label_prediction(row: Mapping[str, Any]) -> str | None:
    first, second = row["answer_labels"]
    logits = row["diagnostics"]["label_logit_by_text"]
    if logits[first] == logits[second]:
        return None
    return first if logits[first] > logits[second] else second


def _correct(row: Mapping[str, Any]) -> bool:
    return _global_prediction(row) == row["correct_answer"]


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    if not values:
        raise BehavioralDeconfoundingAnalysisError("cannot summarize an empty Boolean vector")
    correct = int(sum(values))
    return {"n": len(values), "count": correct, "rate": correct / len(values)}


def _common_bootstrap_indices() -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        len(WORLD_IDS),
        size=(BOOTSTRAP_DRAWS, len(WORLD_IDS)),
    )


def _bootstrap_world_mean(
    world_values: Mapping[str, float],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    if list(world_values) != list(WORLD_IDS):
        raise BehavioralDeconfoundingAnalysisError("bootstrap world order changed")
    values = np.asarray([_finite(world_values[world], f"world[{world}]") for world in WORLD_IDS])
    if bootstrap_indices.shape != (BOOTSTRAP_DRAWS, len(WORLD_IDS)):
        raise BehavioralDeconfoundingAnalysisError("bootstrap index registry changed")
    distribution = values[bootstrap_indices].mean(axis=1)
    return {
        "unit": "world_id",
        "n_worlds": len(WORLD_IDS),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "point_mean": float(values.mean()),
        "lower_95": float(np.quantile(distribution, 0.025)),
        "upper_95": float(np.quantile(distribution, 0.975)),
    }


def _family_accuracy(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    members = [row for row in rows if row["family_id"] == family]
    correct = [_correct(row) for row in members]
    two_label_correct = [_two_label_prediction(row) == row["correct_answer"] for row in members]
    by_label = {
        label: _rate([_correct(row) for row in members if row["correct_answer"] == label])
        for label in FAMILY_ANSWER_LABELS[family]
    }
    strata: dict[str, list[bool]] = defaultdict(list)
    worlds: dict[str, list[bool]] = defaultdict(list)
    for row in members:
        strata[_stratum_id(row)].append(_correct(row))
        worlds[row["world_id"]].append(_correct(row))
    by_stratum = {key: _rate(values) for key, values in sorted(strata.items())}
    by_world = {world: _rate(worlds[world]) for world in WORLD_IDS}
    world_values = {world: by_world[world]["rate"] for world in WORLD_IDS}

    bundles: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in members:
        bundles[_bundle_tuple(row)].append(row)
    bundle_all_correct = [all(_correct(row) for row in bundle) for bundle in bundles.values()]
    bundle_prediction_invariant = [
        len({tuple(row["diagnostics"]["maximum_token_ids"]) for row in bundle}) == 1 for bundle in bundles.values()
    ]
    bundle_label_prediction_invariant = []
    for bundle in bundles.values():
        predictions = [_global_prediction(row) for row in bundle]
        bundle_label_prediction_invariant.append(
            all(prediction is not None for prediction in predictions) and len(set(predictions)) == 1
        )

    order_factors = tuple(
        factor for factor in STRATUM_FACTORS[family] if factor in {"target_fact_order", "rule_order", "option_order"}
    )
    order_levels: dict[str, dict[str, dict[str, Any]]] = {}
    for factor in order_factors:
        level_metrics = {
            str(level): _rate([_correct(row) for row in members if row[factor] == level])
            for level in sorted({row[factor] for row in members})
        }
        rates = [metric["rate"] for metric in level_metrics.values()]
        order_levels[factor] = {
            "levels": level_metrics,
            "max_minus_min_accuracy": max(rates) - min(rates),
        }

    overall = _rate(correct)
    descriptive_two_label = _rate(two_label_correct)
    bootstrap = _bootstrap_world_mean(world_values, bootstrap_indices)
    stratum_threshold = COMPOSITION_STRATUM_THRESHOLD if family == "composition" else COMPONENT_STRATUM_THRESHOLD
    gates = {
        "overall_accuracy": overall["rate"] >= FAMILY_OVERALL_THRESHOLD,
        "every_full_factorial_stratum": all(metric["rate"] >= stratum_threshold for metric in by_stratum.values()),
        "semantic_bundle_all_permutations_correct": (_rate(bundle_all_correct)["rate"] >= BUNDLE_THRESHOLD),
        "every_world_accuracy": all(metric["rate"] >= WORLD_THRESHOLD for metric in by_world.values()),
        "world_bootstrap_lower_95": bootstrap["lower_95"] >= BOOTSTRAP_LOWER_THRESHOLD,
    }
    if family == "composition":
        gates["native_x_accuracy"] = by_label["X"]["rate"] >= FAMILY_OVERALL_THRESHOLD
        gates["native_y_accuracy"] = by_label["Y"]["rate"] >= FAMILY_OVERALL_THRESHOLD
    gates["pass"] = all(gates.values())
    return {
        "overall_confirmatory_full_vocab_accuracy": overall,
        "descriptive_two_label_preference_accuracy": descriptive_two_label,
        "accuracy_by_native_label": by_label,
        "accuracy_by_full_factorial_stratum": by_stratum,
        "accuracy_by_world": by_world,
        "world_cluster_bootstrap": bootstrap,
        "order_invariance": {
            "accuracy_by_order_factor_and_level": order_levels,
            "semantic_bundle_global_argmax_invariant": _rate(bundle_prediction_invariant),
            "semantic_bundle_family_label_prediction_invariant": _rate(bundle_label_prediction_invariant),
            "semantic_bundle_all_permutations_correct": _rate(bundle_all_correct),
        },
        "thresholds": {
            "overall_accuracy": FAMILY_OVERALL_THRESHOLD,
            "full_factorial_stratum_accuracy": stratum_threshold,
            "semantic_bundle_all_permutations_correct_rate": BUNDLE_THRESHOLD,
            "every_world_accuracy": WORLD_THRESHOLD,
            "world_bootstrap_lower_95": BOOTSTRAP_LOWER_THRESHOLD,
            "native_label_accuracy": FAMILY_OVERALL_THRESHOLD if family == "composition" else None,
        },
        "gates": gates,
    }


def _channel_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        members = [row for row in rows if row["family_id"] == family]
        no_ties = [row["diagnostics"]["maximum_tie_count"] == 1 for row in members]
        top1_in_labels = [
            len(row["diagnostics"]["maximum_token_ids"]) == 1
            and row["diagnostics"]["maximum_token_ids"][0] in set(row["label_token_ids"].values())
            for row in members
        ]
        masses = [row["diagnostics"]["label_probability_mass"] for row in members]
        family_checks = {
            "finite_full_vocabulary_diagnostics": True,
            "no_global_argmax_ties": _rate(no_ties),
            "unique_global_argmax_in_family_labels": _rate(top1_in_labels),
            "mean_family_label_probability_mass": float(np.mean(masses)),
        }
        gates = {
            "finite_full_vocabulary_diagnostics": True,
            "no_global_argmax_ties": all(no_ties),
            "unique_global_argmax_in_family_labels_rate": (
                family_checks["unique_global_argmax_in_family_labels"]["rate"] >= CHANNEL_TOP1_THRESHOLD
            ),
            "mean_family_label_probability_mass": (
                family_checks["mean_family_label_probability_mass"] >= CHANNEL_MASS_THRESHOLD
            ),
        }
        family_checks["gates"] = {**gates, "pass": all(gates.values())}
        by_family[family] = family_checks
    return {
        "by_family": by_family,
        "thresholds": {
            "unique_global_argmax_in_family_labels_rate": CHANNEL_TOP1_THRESHOLD,
            "mean_family_label_probability_mass": CHANNEL_MASS_THRESHOLD,
            "global_argmax_ties_allowed": 0,
        },
        "pass": all(by_family[family]["gates"]["pass"] for family in FAMILY_ORDER),
    }


def _policy_answer(row: Mapping[str, Any], policy: str) -> str:
    if row["family_id"] != "composition":
        raise BehavioralDeconfoundingAnalysisError("composition policy applied outside composition")
    if policy == POLICY_INTENDED:
        return str(row["correct_answer"])
    if policy == POLICY_V3:
        return _v3_heuristic(
            str(row["target_property"]),
            str(row["mapping_id"]),
            str(row["target_fact_order"]),
        )
    if policy == POLICY_LAST_OPTION:
        return str(row["displayed_options"][-1])
    if policy == POLICY_FIRST_RULE:
        return _first_rule_output(str(row["mapping_id"]), str(row["rule_order"]))
    if policy == POLICY_CONSTANT_Y:
        return "Y"
    if policy == POLICY_CONSTANT_X:
        return "X"
    raise BehavioralDeconfoundingAnalysisError("unregistered composition policy")


def _heuristic_comparison(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    *,
    composition_pass: bool,
) -> dict[str, Any]:
    members = [row for row in rows if row["family_id"] == "composition"]
    policies = (POLICY_INTENDED, *POLICY_ORDER)
    policy_metrics: dict[str, Any] = {}
    world_vectors: dict[str, np.ndarray] = {}
    for policy in policies:
        matches = [_global_prediction(row) == _policy_answer(row, policy) for row in members]
        by_world = {
            world: _rate(
                [_global_prediction(row) == _policy_answer(row, policy) for row in members if row["world_id"] == world]
            )
            for world in WORLD_IDS
        }
        world_values = {world: by_world[world]["rate"] for world in WORLD_IDS}
        vector = np.asarray([world_values[world] for world in WORLD_IDS])
        world_vectors[policy] = vector
        policy_agreement = [_policy_answer(row, policy) == row["correct_answer"] for row in members]
        policy_metrics[policy] = {
            "model_match": _rate(matches),
            "model_match_by_world": by_world,
            "world_cluster_bootstrap": _bootstrap_world_mean(world_values, bootstrap_indices),
            "policy_agreement_with_intended_ledger": _rate(policy_agreement),
        }
    intended_vector = world_vectors[POLICY_INTENDED]
    intended_rate = policy_metrics[POLICY_INTENDED]["model_match"]["rate"]
    for policy in POLICY_ORDER:
        differences = world_vectors[policy] - intended_vector
        distribution = differences[bootstrap_indices].mean(axis=1)
        metric = policy_metrics[policy]
        metric["match_rate_minus_intended"] = metric["model_match"]["rate"] - intended_rate
        metric["difference_world_cluster_bootstrap"] = {
            "unit": "world_id",
            "n_worlds": len(WORLD_IDS),
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "point_mean": float(differences.mean()),
            "lower_95": float(np.quantile(distribution, 0.025)),
            "upper_95": float(np.quantile(distribution, 0.975)),
        }

    qualifying = [
        policy
        for policy in POLICY_ORDER
        if policy_metrics[policy]["model_match"]["rate"] >= DOMINANT_MATCH_THRESHOLD
        and policy_metrics[policy]["match_rate_minus_intended"] >= DOMINANT_ADVANTAGE_THRESHOLD
    ]
    if composition_pass or not qualifying:
        dominance = "NO_REGISTERED_HEURISTIC_DOMINANT"
        selected_policy = None
    else:
        counts = {policy: policy_metrics[policy]["model_match"]["count"] for policy in qualifying}
        highest = max(counts.values())
        tied = [policy for policy in POLICY_ORDER if policy in counts and counts[policy] == highest]
        if len(tied) > 1:
            dominance = "MULTIPLE_REGISTERED_HEURISTICS"
            selected_policy = None
        else:
            selected_policy = tied[0]
            dominance = {
                POLICY_V3: "V3_HEURISTIC_DOMINANT",
                POLICY_LAST_OPTION: "LAST_DISPLAYED_OPTION_HEURISTIC_DOMINANT",
                POLICY_FIRST_RULE: "FIRST_DISPLAYED_RULE_OUTPUT_HEURISTIC_DOMINANT",
                POLICY_CONSTANT_Y: "CONSTANT_Y_HEURISTIC_DOMINANT",
                POLICY_CONSTANT_X: "CONSTANT_X_HEURISTIC_DOMINANT",
            }[selected_policy]
    return {
        "registered_policy_order": [POLICY_INTENDED, *POLICY_ORDER],
        "policies": policy_metrics,
        "dominance_rule": {
            "applied_only_when_composition_fails": True,
            "minimum_model_match_rate": DOMINANT_MATCH_THRESHOLD,
            "minimum_advantage_over_intended": DOMINANT_ADVANTAGE_THRESHOLD,
            "exact_highest_match_tie_status": "MULTIPLE_REGISTERED_HEURISTICS",
            "fixed_priority": list(POLICY_ORDER),
        },
        "qualifying_failure_heuristics": qualifying if not composition_pass else [],
        "dominant_failure_classification": dominance,
        "selected_dominant_policy": selected_policy,
    }


def _terminal_status(*, engineering: bool, retrieval: bool, lookup: bool, composition: bool) -> str:
    if not engineering:
        return STATUS_ENGINEERING_INVALID
    if not retrieval and not lookup:
        return STATUS_BOTH_COMPONENTS_FAIL
    if not retrieval:
        return STATUS_RETRIEVAL_FAIL
    if not lookup:
        return STATUS_LOOKUP_FAIL
    if not composition:
        return STATUS_COMPOSITION_FAIL
    return STATUS_QUALIFIED


def analyze_records(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    artifact_validation: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize validated runner rows and apply the frozen V4 decision tree."""

    normalized = [_normalize_record(raw) for raw in raw_rows]
    coverage = _validate_coverage(normalized)
    normalized.sort(key=lambda row: (row["family_id"], row["world_id"], row["cell_id"]))
    bootstrap_indices = _common_bootstrap_indices()
    accuracy = {family: _family_accuracy(normalized, family, bootstrap_indices) for family in FAMILY_ORDER}
    channel = _channel_checks(normalized)
    artifact_pass = True
    if artifact_validation is not None:
        if artifact_validation.get("pass") is not True:
            artifact_pass = False
    engineering_pass = bool(coverage["exact_contract_pass"] and channel["pass"] and artifact_pass)
    retrieval_pass = accuracy["property_retrieval"]["gates"]["pass"] is True
    lookup_pass = accuracy["codebook_lookup"]["gates"]["pass"] is True
    composition_pass = accuracy["composition"]["gates"]["pass"] is True
    status = _terminal_status(
        engineering=engineering_pass,
        retrieval=retrieval_pass,
        lookup=lookup_pass,
        composition=composition_pass,
    )
    heuristic = _heuristic_comparison(
        normalized,
        bootstrap_indices,
        composition_pass=composition_pass,
    )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": status,
        "claim_authority": "terminal_behavioral_component_localization_only",
        "maximum_claim": (
            "performance localization among output engineering, property retrieval, "
            "codebook lookup, and their synthetic composition in one locked model"
        ),
        "claim_boundaries": CLAIM_BOUNDARIES,
        "artifact_validation": dict(artifact_validation or {"pass": True, "mode": "in_memory_test"}),
        "coverage": coverage,
        "accuracy": accuracy,
        "channel_checks": channel,
        "composition_policy_comparison": heuristic,
        "component_gates": {
            "engineering": engineering_pass,
            "property_retrieval": retrieval_pass,
            "codebook_lookup": lookup_pass,
            "composition": composition_pass,
        },
        "decision_precedence": [
            "engineering",
            "property_retrieval_and_codebook_lookup",
            "property_retrieval",
            "codebook_lookup",
            "composition",
            "qualified",
        ],
        "terminal_no_downstream_authorization": True,
        "provenance": dict(provenance or {}),
        "normalized_row_registry_canonical_sha256": canonical_sha256(normalized),
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }


def _validated_runner_bundle(result_root: Path) -> Mapping[str, Any]:
    if runner is None:
        raise BehavioralDeconfoundingAnalysisError("V4 runner module is unavailable")
    validator = getattr(runner, "validate_behavior_artifacts", None)
    if validator is None:
        raise BehavioralDeconfoundingAnalysisError("V4 runner validation API is unavailable")
    try:
        bundle = validator(result_root=result_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise BehavioralDeconfoundingAnalysisError(str(error)) from error
    if not isinstance(bundle, Mapping):
        raise BehavioralDeconfoundingAnalysisError("runner validation bundle must be an object")
    return bundle


def _enrich_records_from_cells(
    raw_records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells = plan.get("cell_registry")
    if not isinstance(cells, list) or any(not isinstance(cell, Mapping) for cell in cells):
        raise BehavioralDeconfoundingAnalysisError("validated plan cell registry changed")
    cell_by_id = {cell.get("cell_id"): cell for cell in cells}
    if len(cell_by_id) != 448 or None in cell_by_id:
        raise BehavioralDeconfoundingAnalysisError("validated plan cell IDs changed")
    enriched = []
    for record in raw_records:
        cell = cell_by_id.get(record.get("cell_id"))
        if cell is None:
            raise BehavioralDeconfoundingAnalysisError("record does not resolve to a frozen cell")
        value = dict(record)
        for field, field_value in cell.items():
            if field in value and value[field] != field_value:
                raise BehavioralDeconfoundingAnalysisError(f"record differs from frozen cell field: {field}")
            value.setdefault(field, field_value)
        enriched.append(value)
    return enriched


def _bundle_records(bundle: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw_records: Sequence[Mapping[str, Any]] | None = None
    for key in ("records", "behavior_records"):
        value = bundle.get(key)
        if value is not None:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise BehavioralDeconfoundingAnalysisError("runner records bundle changed")
            if any(not isinstance(row, Mapping) for row in value):
                raise BehavioralDeconfoundingAnalysisError("runner records must be objects")
            raw_records = value
            break
    if raw_records is None:
        raise BehavioralDeconfoundingAnalysisError("runner validation bundle omitted records")
    plan = bundle.get("plan_manifest")
    if not isinstance(plan, Mapping):
        raise BehavioralDeconfoundingAnalysisError("runner validation bundle omitted plan_manifest")
    enriched = _enrich_records_from_cells(raw_records, plan)
    raw_matrix = bundle.get("full_vocab_logits")
    if not isinstance(raw_matrix, np.ndarray):
        raise BehavioralDeconfoundingAnalysisError("runner bundle omitted raw full-vocabulary matrix")
    if (
        raw_matrix.dtype != np.dtype("<f4")
        or raw_matrix.shape != (448, MODEL_VOCAB_SIZE)
        or not np.isfinite(raw_matrix).all()
    ):
        raise BehavioralDeconfoundingAnalysisError("raw full-vocabulary matrix changed")
    observed_rows = []
    for expected_index, record in enumerate(enriched):
        row_index = record.get("full_vocab_logits_row")
        if row_index != expected_index:
            raise BehavioralDeconfoundingAnalysisError("full-vocabulary row order changed")
        computed = _diagnostics_from_full_vocab(
            raw_matrix[row_index],
            answer_labels=record["answer_labels"],
            label_token_ids=record["label_token_ids"],
            correct_answer=record["correct_answer"],
        )
        stored = record.get("diagnostics")
        if not isinstance(stored, Mapping) or dict(stored) != computed:
            raise BehavioralDeconfoundingAnalysisError(
                f"record diagnostics do not independently reconstruct at row {row_index}"
            )
        value = dict(record)
        value["diagnostics"] = computed
        observed_rows.append(value)
    return observed_rows


def _artifact_bindings(result_root: Path) -> dict[str, Any]:
    filenames = (
        "plan_manifest.json",
        "design.json",
        "tokenization_receipt.json",
        "dependency_lock.json",
        "behavior_attempt.json",
        "behavior_records.jsonl",
        "behavior_full_vocab_logits.npy",
        "behavior_execution_manifest.json",
    )
    bindings = {}
    for filename in filenames:
        path = result_root / filename
        if not path.is_file():
            raise BehavioralDeconfoundingAnalysisError(f"validated input artifact disappeared: {filename}")
        bindings[filename] = {"path": str(path), "file_sha256": file_sha256(path)}
    return bindings


def _render_bilingual_markdown(analysis: Mapping[str, Any]) -> str:
    gates = analysis["component_gates"]
    heuristic = analysis["composition_policy_comparison"]
    lines = [
        "# Coherent Readout V4 behavioral-deconfounding analysis",
        "",
        "## English",
        "",
        f"Terminal status: `{analysis['status']}`.",
        "",
        f"- Output engineering: **{'pass' if gates['engineering'] else 'fail'}**",
        f"- Property retrieval: **{'pass' if gates['property_retrieval'] else 'fail'}**",
        f"- Codebook lookup: **{'pass' if gates['codebook_lookup'] else 'fail'}**",
        f"- Composition: **{'pass' if gates['composition'] else 'fail'}**",
        (f"- Registered failure-policy classification: `{heuristic['dominant_failure_classification']}`"),
        "",
        (
            "This is a terminal synthetic behavioral diagnostic. It does not establish a causal "
            "mechanism, latent knowledge, an activation gap, biology, a physical law, or model-family "
            "generalization."
        ),
        "",
        "## 한국어",
        "",
        f"최종 상태: `{analysis['status']}`.",
        "",
        f"- 출력 engineering: **{'통과' if gates['engineering'] else '실패'}**",
        f"- 속성 retrieval: **{'통과' if gates['property_retrieval'] else '실패'}**",
        f"- codebook lookup: **{'통과' if gates['codebook_lookup'] else '실패'}**",
        f"- composition: **{'통과' if gates['composition'] else '실패'}**",
        (f"- 등록된 실패 정책 분류: `{heuristic['dominant_failure_classification']}`"),
        "",
        (
            "이 결과는 합성 과제의 종결형 행동 진단이다. 인과 메커니즘, 잠재지식, activation "
            "gap, 생물학, 물리 법칙, 또는 모델 계열 일반화를 증명하지 않는다."
        ),
        "",
    ]
    return "\n".join(lines)


def _write_frozen_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise BehavioralDeconfoundingAnalysisError(f"refusing to overwrite different artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_frozen_bundle(payloads: Mapping[Path, bytes]) -> None:
    """Preflight every output before writing, preventing a partial mixed bundle."""

    for path, payload in payloads.items():
        if path.exists() and path.read_bytes() != payload:
            raise BehavioralDeconfoundingAnalysisError(f"refusing to overwrite different artifact: {path}")
    for path, payload in payloads.items():
        _write_frozen_bytes(path, payload)


def _write_outputs(result_root: Path, analysis: Mapping[str, Any]) -> None:
    analysis_path = result_root / DEFAULT_ANALYSIS.name
    markdown_path = result_root / DEFAULT_MARKDOWN.name
    manifest_path = result_root / DEFAULT_ANALYSIS_MANIFEST.name
    analysis_bytes = (json.dumps(analysis, indent=2, sort_keys=True) + "\n").encode("utf-8")
    markdown_bytes = _render_bilingual_markdown(analysis).encode("utf-8")
    manifest = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "status": analysis["status"],
        "analysis": {
            "path": str(analysis_path),
            "file_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "markdown": {
            "path": str(markdown_path),
            "file_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
        },
        "analyzer": {"path": str(Path(__file__)), "file_sha256": file_sha256(Path(__file__))},
        "input_artifacts": _artifact_bindings(result_root),
        "model_forwards_executed_by_analyzer": 0,
        "biological_model_calls": 0,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_frozen_bundle(
        {
            analysis_path: analysis_bytes,
            markdown_path: markdown_bytes,
            manifest_path: manifest_bytes,
        }
    )


def analyze_behavior(
    *,
    result_root: Path = RESULT_ROOT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Replay runner artifacts, analyze exactly 448 rows, and optionally freeze outputs."""

    result_root = Path(result_root)
    bundle = _validated_runner_bundle(result_root)
    bindings = _artifact_bindings(result_root)
    plan = bundle.get("plan_manifest")
    execution_manifest = bundle.get("execution_manifest")
    if not isinstance(plan, Mapping) or not isinstance(execution_manifest, Mapping):
        raise BehavioralDeconfoundingAnalysisError("runner validation bundle omitted plan or execution manifest")
    artifact_validation = {
        "pass": True,
        "validator": "runner.validate_behavior_artifacts",
        "input_artifacts": bindings,
    }
    provenance = {
        "result_root": str(result_root),
        "runner_module": str(getattr(runner, "__file__", "")),
        "runner_file_sha256": (
            file_sha256(Path(runner.__file__)) if runner is not None and getattr(runner, "__file__", None) else None
        ),
        "call_plan_sha256": plan.get("call_plan_sha256"),
        "plan_manifest_canonical_sha256": plan.get("canonical_sha256"),
        "execution_manifest_file_sha256": bindings["behavior_execution_manifest.json"]["file_sha256"],
        "execution_status": execution_manifest.get("status"),
    }
    analysis = analyze_records(
        _bundle_records(bundle),
        artifact_validation=artifact_validation,
        provenance=provenance,
    )
    if write_outputs:
        _write_outputs(result_root, analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()
    analysis = analyze_behavior(result_root=arguments.result_root, write_outputs=not arguments.no_write)
    print(json.dumps({"status": analysis["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()

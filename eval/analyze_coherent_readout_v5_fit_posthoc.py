#!/usr/bin/env python3
"""Descriptive post-hoc decomposition of the stopped V5 fit baseline.

This module never authorizes another model phase and never changes the frozen
V5 status.  It summarizes serial-position sensitivities already present in the
completed fit records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v5_positional_activation"
    / "qwen2.5-1.5b-instruct"
)
FIXTURE = ROOT / "signal" / "syntax" / "coherent_readout_v5_positional_activation_bank.json"
RECORDS = RESULT_DIR / "fit_baseline_records.jsonl"
EXECUTION_MANIFEST = RESULT_DIR / "fit_baseline_execution_manifest.json"
FIT_ANALYSIS = RESULT_DIR / "fit_analysis.json"
PLAN_MANIFEST = RESULT_DIR / "plan_manifest.json"
DEFAULT_OUTPUT = RESULT_DIR / "fit_posthoc_analysis.json"
SCHEMA = "coherent-readout-v5-fit-posthoc-v1"


class V5PosthocError(RuntimeError):
    """Raised when the frozen V5 inputs do not match the stopped fit run."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V5PosthocError(f"expected an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise V5PosthocError(f"record {line_number} is not an object")
        rows.append(value)
    return rows


def _correct(row: Mapping[str, Any]) -> bool:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not isinstance(
        diagnostics.get("answer_correct"), bool
    ):
        raise V5PosthocError("record lacks a Boolean answer_correct diagnostic")
    return bool(diagnostics["answer_correct"])


def _predicted(row: Mapping[str, Any]) -> str:
    diagnostics = row.get("diagnostics")
    predicted = diagnostics.get("predicted_answer") if isinstance(diagnostics, Mapping) else None
    if not isinstance(predicted, str) or not predicted:
        raise V5PosthocError("record lacks a unique predicted answer")
    return predicted


def _margin(row: Mapping[str, Any]) -> float:
    diagnostics = row.get("diagnostics")
    value = (
        diagnostics.get("expected_minus_distractor_margin")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise V5PosthocError("record margin is not finite")
    return float(value)


def _factor_product(row: Mapping[str, Any], names: Sequence[str]) -> int:
    factors = row.get("factors")
    if not isinstance(factors, Mapping):
        raise V5PosthocError("record factors are missing")
    product = 1
    for name in names:
        value = factors.get(name)
        if value not in (-1, 1):
            raise V5PosthocError(f"factor {name} is not signed")
        product *= int(value)
    return product


def factorial_term_summary(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    if not rows:
        raise V5PosthocError("factorial term has no rows")
    by_world: dict[str, list[float]] = defaultdict(list)
    choice_by_world: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        world = row.get("world_id")
        if not isinstance(world, str) or not world:
            raise V5PosthocError("world_id is invalid")
        sign = _factor_product(row, names)
        by_world[world].append(_margin(row) * sign)
        choice_by_world[world].append((1.0 if _correct(row) else -1.0) * sign)
    world_values = {
        world: float(np.mean(values)) for world, values in sorted(by_world.items())
    }
    choice_values = {
        world: float(np.mean(choice_by_world[world])) for world in sorted(choice_by_world)
    }
    values = list(world_values.values())
    return {
        "term": "*".join(names),
        "margin_beta": float(np.mean(values)),
        "choice_beta": float(np.mean(list(choice_values.values()))),
        "world_margin_betas": world_values,
        "positive_worlds": sum(value > 0.0 for value in values),
        "negative_worlds": sum(value < 0.0 for value in values),
        "zero_worlds": sum(value == 0.0 for value in values),
        "n_worlds": len(values),
    }


def exact_mcnemar_two_sided(first_only: int, second_only: int) -> float:
    if min(first_only, second_only) < 0:
        raise V5PosthocError("discordant counts cannot be negative")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_order_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_pair_ids: set[str] | None = None,
) -> dict[str, Any]:
    pairs: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("pair_id")
        world_id = row.get("world_id")
        factors = row.get("factors")
        order = factors.get("o") if isinstance(factors, Mapping) else None
        if not isinstance(pair_id, str) or not isinstance(world_id, str) or order not in (-1, 1):
            raise V5PosthocError("composition pair identity is invalid")
        if allowed_pair_ids is not None and pair_id not in allowed_pair_ids:
            continue
        pairs[(world_id, pair_id)][int(order)] = row
    if not pairs or any(set(pair) != {-1, 1} for pair in pairs.values()):
        raise V5PosthocError("composition order pairs are incomplete")

    outcomes: Counter[str] = Counter()
    differences: list[float] = []
    for pair in pairs.values():
        first = pair[-1]
        second = pair[1]
        first_correct = _correct(first)
        second_correct = _correct(second)
        if first_correct and second_correct:
            outcomes["both_correct"] += 1
        elif first_correct:
            outcomes["target_first_only"] += 1
        elif second_correct:
            outcomes["target_second_only"] += 1
        else:
            outcomes["neither_correct"] += 1
        differences.append(_margin(first) - _margin(second))

    first_only = outcomes["target_first_only"]
    second_only = outcomes["target_second_only"]
    return {
        "n_pairs": len(pairs),
        "paired_outcomes": dict(sorted(outcomes.items())),
        "target_first_minus_target_second_margin_mean": float(np.mean(differences)),
        "positive_pair_count": sum(value > 0.0 for value in differences),
        "negative_pair_count": sum(value < 0.0 for value in differences),
        "zero_pair_count": sum(value == 0.0 for value in differences),
        "exact_mcnemar_two_sided_fixed_panel_diagnostic": exact_mcnemar_two_sided(
            first_only, second_only
        ),
        "p_value_scope": "fixed_deterministic_fit_panel_not_population_inference",
    }


def composition_cue_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not members:
            raise V5PosthocError("composition cue cell is empty")
        return {
            "n": len(members),
            "correct": sum(_correct(row) for row in members),
            "accuracy": float(np.mean([_correct(row) for row in members])),
            "margin_mean": float(np.mean([_margin(row) for row in members])),
        }

    cues = {
        "target_rule_first": [row for row in rows if _factor_product(row, ("p", "r")) == 1],
        "target_rule_last": [row for row in rows if _factor_product(row, ("p", "r")) == -1],
        "correct_option_first": [
            row for row in rows if _factor_product(row, ("p", "m", "v")) == -1
        ],
        "correct_option_last": [
            row for row in rows if _factor_product(row, ("p", "m", "v")) == 1
        ],
        "target_fact_first": [row for row in rows if _factor_product(row, ("o",)) == -1],
        "target_fact_second": [row for row in rows if _factor_product(row, ("o",)) == 1],
        "all_early_cues": [
            row
            for row in rows
            if _factor_product(row, ("p", "r")) == 1
            and _factor_product(row, ("p", "m", "v")) == -1
            and _factor_product(row, ("o",)) == -1
        ],
        "all_late_cues": [
            row
            for row in rows
            if _factor_product(row, ("p", "r")) == -1
            and _factor_product(row, ("p", "m", "v")) == 1
            and _factor_product(row, ("o",)) == 1
        ],
    }
    return {name: summarize(members) for name, members in cues.items()}


def composition_additive_recency_model(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise V5PosthocError("composition recency model has no rows")
    design = np.asarray(
        [
            [
                1.0,
                float(_factor_product(row, ("o",))),
                float(-_factor_product(row, ("p", "r"))),
                float(_factor_product(row, ("p", "m", "v"))),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    margins = np.asarray([_margin(row) for row in rows], dtype=np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(design, margins, rcond=None)
    if rank != design.shape[1]:
        raise V5PosthocError("composition recency design lost rank")
    fitted = design @ coefficients
    residual_sum_squares = float(np.square(margins - fitted).sum())
    total_sum_squares = float(np.square(margins - margins.mean()).sum())
    if total_sum_squares <= 0.0:
        raise V5PosthocError("composition margins have zero variance")
    observed_sign = margins > 0.0
    if any(_correct(row) != bool(value) for row, value in zip(rows, observed_sign, strict=True)):
        raise V5PosthocError("answer correctness disagrees with margin sign")
    predicted_sign = fitted > 0.0
    return {
        "formula": "margin ~ 1 + target_second + target_rule_last + correct_option_last",
        "coefficients": {
            "intercept": float(coefficients[0]),
            "target_second": float(coefficients[1]),
            "target_rule_last": float(coefficients[2]),
            "correct_option_last": float(coefficients[3]),
        },
        "r_squared": 1.0 - residual_sum_squares / total_sum_squares,
        "sign_accuracy": float(np.mean(predicted_sign == observed_sign)),
        "sign_correct": int(np.sum(predicted_sign == observed_sign)),
        "n": len(rows),
        "fit_scope": "descriptive_same_fit_panel_not_cross_validated",
    }


def _rule_rhs(line: str) -> str:
    marker = " maps to "
    if marker not in line or not line.endswith("."):
        raise V5PosthocError("rule line changed")
    return line.split(marker, 1)[1][:-1]


def _fact_property(line: str) -> str:
    marker = " has property "
    if marker not in line or not line.endswith("."):
        raise V5PosthocError("fact line changed")
    return line.split(marker, 1)[1][:-1]


def heuristic_summary(
    rows: Sequence[Mapping[str, Any]], cells: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    matches: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        cell_id = row.get("cell_id")
        cell = cells.get(cell_id) if isinstance(cell_id, str) else None
        if not isinstance(cell, Mapping):
            raise V5PosthocError("record does not resolve to a fixture cell")
        predicted = _predicted(row)
        options = cell.get("answer_options")
        if not isinstance(options, list) or len(options) != 2:
            raise V5PosthocError("fixture answer options changed")
        matches["last_option"].append(predicted == options[-1])
        matches["first_option"].append(predicted == options[0])

        rules = cell.get("rule_lines")
        if isinstance(rules, list) and rules:
            matches["last_rule_rhs"].append(predicted == _rule_rhs(str(rules[-1])))
            matches["first_rule_rhs"].append(predicted == _rule_rhs(str(rules[0])))

        facts = cell.get("fact_lines")
        codebook = cell.get("codebook")
        if isinstance(facts, list) and facts and isinstance(codebook, Mapping):
            first_property = _fact_property(str(facts[0]))
            last_property = _fact_property(str(facts[-1]))
            matches["first_fact_mapped_code"].append(predicted == codebook[first_property])
            matches["last_fact_mapped_code"].append(predicted == codebook[last_property])

    return {
        name: {
            "matches": sum(values),
            "n": len(values),
            "fraction": sum(values) / len(values),
        }
        for name, values in sorted(matches.items())
    }


def family_output_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    masses: list[float] = []
    valid_global = 0
    unique_global = 0
    for row in rows:
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise V5PosthocError("diagnostics are missing")
        mass = diagnostics.get("label_probability_mass")
        if not isinstance(mass, (int, float)) or isinstance(mass, bool):
            raise V5PosthocError("label probability mass is invalid")
        masses.append(float(mass))
        if diagnostics.get("greedy_token_id") in {
            row.get("expected_token_id"),
            row.get("distractor_token_id"),
        }:
            valid_global += 1
        if diagnostics.get("maximum_tie_count") == 1:
            unique_global += 1
    return {
        "n": len(rows),
        "accuracy": float(np.mean([_correct(row) for row in rows])),
        "valid_pair_global_argmax_count": valid_global,
        "unique_global_argmax_count": unique_global,
        "label_probability_mass_mean": float(np.mean(masses)),
        "label_probability_mass_min": min(masses),
        "label_probability_mass_max": max(masses),
    }


def matched_lookup_composition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        factors = row.get("factors")
        if not isinstance(factors, Mapping):
            raise V5PosthocError("matched factors are missing")
        return (
            row.get("world_id"),
            factors.get("p"),
            factors.get("m"),
            factors.get("r"),
            factors.get("v"),
        )

    lookup = {key(row): row for row in rows if row.get("family") == "lookup"}
    composition = [row for row in rows if row.get("family") == "composition"]
    if len(lookup) != 128 or len(composition) != 256:
        raise V5PosthocError("matched lookup/composition counts changed")
    result: dict[str, Any] = {}
    for order, label in ((-1, "target_first"), (1, "target_second")):
        members = [row for row in composition if _factor_product(row, ("o",)) == order]
        agreement = 0
        transitions: Counter[str] = Counter()
        for row in members:
            lookup_row = lookup.get(key(row))
            if lookup_row is None:
                raise V5PosthocError("composition row lacks a matched lookup row")
            agreement += _predicted(row) == _predicted(lookup_row)
            lookup_correct = _correct(lookup_row)
            composition_correct = _correct(row)
            transition = (
                f"lookup_{'correct' if lookup_correct else 'wrong'}__"
                f"composition_{'correct' if composition_correct else 'wrong'}"
            )
            transitions[transition] += 1
        result[label] = {
            "n": len(members),
            "prediction_agreement_with_lookup": agreement,
            "prediction_agreement_fraction": agreement / len(members),
            "transitions": dict(sorted(transitions.items())),
        }
    return result


def analyze() -> dict[str, Any]:
    fixture = load_json(FIXTURE)
    execution = load_json(EXECUTION_MANIFEST)
    fit_analysis = load_json(FIT_ANALYSIS)
    plan_manifest = load_json(PLAN_MANIFEST)
    rows = load_jsonl(RECORDS)
    fixture_cells_raw = fixture.get("cells")
    if not isinstance(fixture_cells_raw, list):
        raise V5PosthocError("fixture cells are missing")
    cells = {
        cell["cell_id"]: cell
        for cell in fixture_cells_raw
        if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)
    }

    if fit_analysis.get("status") != "V5_FIT_COMPONENT_ADMISSION_FAIL":
        raise V5PosthocError("post-hoc analysis requires the frozen fit-stop status")
    if execution.get("status") != "EXECUTION_COMPLETE_NOT_ANALYZED":
        raise V5PosthocError("fit execution manifest status changed")
    if execution.get("phase_model_calls") != 448 or len(rows) != 448:
        raise V5PosthocError("fit call or record count changed")
    if len({row.get("record_id") for row in rows}) != len(rows):
        raise V5PosthocError("fit record IDs are duplicated")
    call_plan_sha256 = execution.get("call_plan_sha256")
    if not isinstance(call_plan_sha256, str) or any(
        row.get("call_plan_sha256") != call_plan_sha256 for row in rows
    ):
        raise V5PosthocError("call-plan binding changed")
    for row in rows:
        cell = cells.get(row.get("cell_id"))
        if cell is None or cell.get("role") != "fit":
            raise V5PosthocError("fit record does not resolve to a fit fixture cell")
        if row.get("expected_answer") != cell.get("correct_answer"):
            raise V5PosthocError("expected-answer ledger disagrees with the fixture")

    families = {
        family: [row for row in rows if row.get("family") == family]
        for family in ("retrieval", "lookup", "composition")
    }
    if {family: len(value) for family, value in families.items()} != {
        "retrieval": 64,
        "lookup": 128,
        "composition": 256,
    }:
        raise V5PosthocError("fit family counts changed")

    term_registry = {
        "retrieval": (("p", "v"),),
        "lookup": (("p", "r"), ("p", "m", "v")),
        "composition": (("p", "r"), ("p", "m", "v"), ("o",)),
    }
    direct_prerequisites = [row for row in rows if row.get("intervention_prerequisite") is True]
    if len(direct_prerequisites) != 128:
        raise V5PosthocError("direct-prerequisite panel changed")
    plan = plan_manifest.get("plan")
    world_registry = plan.get("world_registry") if isinstance(plan, Mapping) else None
    if not isinstance(world_registry, list):
        raise V5PosthocError("plan world registry is missing")
    intervention_pair_ids = {
        pair["pair_id"]
        for world in world_registry
        if isinstance(world, Mapping) and world.get("role") == "fit"
        for pair in world.get("intervention_pairs", [])
        if isinstance(pair, Mapping) and isinstance(pair.get("pair_id"), str)
    }
    if len(intervention_pair_ids) != 64:
        raise V5PosthocError("fit intervention-pair panel changed")

    return {
        "schema_version": SCHEMA,
        "analysis_type": "exploratory_posthoc_no_new_model_calls",
        "frozen_confirmatory_status": fit_analysis["status"],
        "bindings": {
            "call_plan_sha256": call_plan_sha256,
            "fixture_file_sha256": file_sha256(FIXTURE),
            "records_file_sha256": file_sha256(RECORDS),
            "execution_manifest_file_sha256": file_sha256(EXECUTION_MANIFEST),
            "fit_analysis_file_sha256": file_sha256(FIT_ANALYSIS),
            "plan_manifest_file_sha256": file_sha256(PLAN_MANIFEST),
        },
        "integrity": {
            "records": len(rows),
            "unique_record_ids": len({row["record_id"] for row in rows}),
            "expected_answer_ledger_matches_fixture": True,
            "new_model_calls": 0,
        },
        "registered_behavior": fit_analysis["behavior"],
        "direct_prerequisites": {
            "correct": sum(_correct(row) for row in direct_prerequisites),
            "n": len(direct_prerequisites),
        },
        "family_output": {
            family: family_output_summary(members) for family, members in families.items()
        },
        "factorial_recency_terms": {
            family: [factorial_term_summary(families[family], term) for term in terms]
            for family, terms in term_registry.items()
        },
        "heuristic_matches": {
            family: heuristic_summary(members, cells) for family, members in families.items()
        },
        "composition_cues": composition_cue_summary(families["composition"]),
        "composition_additive_recency_model": composition_additive_recency_model(
            families["composition"]
        ),
        "composition_order_pairs": {
            "full_factorial": paired_order_summary(families["composition"]),
            "fixed_intervention_panel": paired_order_summary(
                families["composition"], allowed_pair_ids=intervention_pair_ids
            ),
        },
        "matched_lookup_to_composition": matched_lookup_composition_summary(rows),
        "interpretation_boundary": {
            "confirmatory_status_changed": False,
            "localization_or_patch_authorized": False,
            "causal_activation_mediation_supported": False,
            "latent_knowledge_supported": False,
            "biology_or_physical_law_supported": False,
            "supported_description": (
                "fixed-fit-panel serial-position sensitivity with a fully reversed "
                "TARGET-order effect"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["frozen_confirmatory_status"],
                "new_model_calls": 0,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

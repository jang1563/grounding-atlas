"""Localize the completed PBMC68k module-comparison discovery.

This is a descriptive post-hoc analysis of the frozen raw checkpoint. It does
not alter the preregistered aggregate gate and does not support gene-specific
or biological causal claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "benchmark" / "single_cell" / "module_factorial" / "claude-haiku-4-5-20251001_raw.jsonl"
DATA_PATH = ROOT / "signal" / "single_cell" / "pbmc68k_cd8_nk_module_factorial.csv"
OUT_ROOT = RAW_PATH.parent

EXPECTED_RAW_SHA256 = "2a68e04ead9a55b28f348e7790e06b24cf273b8ddc9ae53c7263f919c9fdb0d7"
EXPECTED_DATA_SHA256 = "fadd90b9aa1249c2691943287962a381329bcba8f91f0b392636ba80ce9f1d9b"
EXPECTED_PLAN_SHA256 = "25d083894f54b17ee0dca72a04b19a30f03e076403dde7f9ae05d05c7c42ae96"
EXPECTED_PREREG_SHA256 = "a7f38447ceaae932b87492b15d667af02c21b006fedb7beb7469d7448b17e605"
EXPECTED_CODE_SHA256 = "6168ea801d0f77a1c9216468aa00d0119ca6a09b3bc4f53ae99797bc85d15141"

MODULES = ("T_TCR_CD8", "cytotoxic_effector")
FORMS = ("ab_pa", "ab_pb", "ba_pa", "ba_pb")
EXACT_PROBABILITY = re.compile(r"(?:0(?:\.\d*)?|1(?:\.0*)?|\.\d+)")


class DiscoveryAnalysisError(ValueError):
    """Raised when the frozen discovery inputs fail validation."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _equal_class(values: np.ndarray, labels: np.ndarray) -> float:
    return float(0.5 * (values[labels == 1].mean() + values[labels == 0].mean()))


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    tolerance = 1e-12
    return {
        "positive": int((values > tolerance).sum()),
        "zero": int((np.abs(values) <= tolerance).sum()),
        "negative": int((values < -tolerance).sum()),
    }


def _parse_raw_probability(record: dict[str, Any]) -> float:
    token = str(record["raw_output"]).strip()
    if EXACT_PROBABILITY.fullmatch(token) is None:
        raise DiscoveryAnalysisError(f"noncompliant raw output: {record['request_id']}")
    probability = float(token)
    if not 0.0 <= probability <= 1.0:
        raise DiscoveryAnalysisError(f"out-of-range raw output: {record['request_id']}")
    expected_aligned = probability if record["queried_target"] == "A" else 1.0 - probability
    if (
        not record["parsed"]
        or not np.isclose(
            float(record["reported_probability"]),
            probability,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.isclose(
            float(record["aligned_probability_a"]),
            expected_aligned,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise DiscoveryAnalysisError(f"raw/derived probability mismatch: {record['request_id']}")
    return expected_aligned


def analyze() -> dict[str, Any]:
    raw_sha256 = _sha256(RAW_PATH)
    data_sha256 = _sha256(DATA_PATH)
    if raw_sha256 != EXPECTED_RAW_SHA256:
        raise DiscoveryAnalysisError("raw checkpoint hash mismatch")
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise DiscoveryAnalysisError("factorial CSV hash mismatch")

    records = [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line]
    if (
        len(records) != 1256
        or len({record["request_id"] for record in records}) != 1256
        or {record["planned_index"] for record in records} != set(range(1256))
    ):
        raise DiscoveryAnalysisError("raw request count, IDs, or planned indices changed")
    for field, expected in (
        ("call_plan_sha256", EXPECTED_PLAN_SHA256),
        ("preregistration_sha256", EXPECTED_PREREG_SHA256),
        ("execution_code_sha256", EXPECTED_CODE_SHA256),
    ):
        if {record[field] for record in records} != {expected}:
            raise DiscoveryAnalysisError(f"raw provenance mismatch: {field}")

    intervention_rows = {
        (row["entity_id"], row["module"]): row
        for row in csv.DictReader(DATA_PATH.open(newline="", encoding="utf-8"))
        if row["row_type"] == "intervention"
    }
    logical: dict[tuple[str, str, str, str], float] = {}
    labels: dict[str, int] = {}
    annotations: dict[str, str] = {}
    raw_grid = Counter()
    logical_count = 0
    for record in records:
        aligned = _parse_raw_probability(record)
        raw_grid[str(record["raw_output"]).strip()] += 1
        entity_id = record["entity_id"]
        labels[entity_id] = int(record["label_a"])
        annotations[entity_id] = record["reference_annotation"]
        for assignment in record["assignments"]:
            key = (
                entity_id,
                assignment["module"],
                assignment["condition"],
                record["form"],
            )
            if key in logical:
                raise DiscoveryAnalysisError(f"duplicate logical assignment: {key}")
            logical[key] = aligned
            logical_count += 1
    if logical_count != 1300:
        raise DiscoveryAnalysisError(f"expected 1300 logical assignments, observed {logical_count}")

    entity_ids = sorted(labels)
    label_array = np.asarray(
        [labels[entity_id] for entity_id in entity_ids],
        dtype=int,
    )
    if Counter(label_array.tolist()) != Counter({1: 55, 0: 10}):
        raise DiscoveryAnalysisError("frozen class counts changed")

    matrices: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}
    unmasked = np.asarray(
        [
            [
                logical[
                    (
                        entity_id,
                        "none",
                        "unmasked",
                        form,
                    )
                ]
                for form in FORMS
            ]
            for entity_id in entity_ids
        ],
        dtype=float,
    )
    for module in MODULES:
        for condition in ("module_mask", "neutral_mask"):
            matrices[(module, condition)] = np.asarray(
                [
                    [
                        logical[
                            (
                                entity_id,
                                module,
                                condition,
                                form,
                            )
                        ]
                        for form in FORMS
                    ]
                    for entity_id in entity_ids
                ],
                dtype=float,
            )

    effects = {module: (matrices[(module, "neutral_mask")] - matrices[(module, "module_mask")]) for module in MODULES}
    cell_effects = {module: effect.mean(axis=1) for module, effect in effects.items()}
    annotation_array = np.asarray(
        [annotations[entity_id] for entity_id in entity_ids],
        dtype=object,
    )

    module_results: dict[str, Any] = {}
    for module in MODULES:
        cell_effect = cell_effects[module]
        aggregate = _equal_class(cell_effect, label_array)
        by_annotation = []
        for annotation in sorted(set(annotation_array.tolist())):
            selected = annotation_array == annotation
            by_annotation.append(
                {
                    "annotation": annotation,
                    "n": int(selected.sum()),
                    "mean_effect_on_p_cd8": float(cell_effect[selected].mean()),
                    "sign_counts": _sign_counts(cell_effect[selected]),
                }
            )

        target_results = []
        target_by_entity = {
            entity_id: intervention_rows[(entity_id, module)]["target_genes"] for entity_id in entity_ids
        }
        for gene in sorted(set(target_by_entity.values())):
            selected = np.asarray(
                [target_by_entity[entity_id] == gene for entity_id in entity_ids],
                dtype=bool,
            )
            contribution = float(
                0.5
                * (
                    (cell_effect[selected & (label_array == 1)].sum() / int((label_array == 1).sum()))
                    + (cell_effect[selected & (label_array == 0)].sum() / int((label_array == 0).sum()))
                )
            )
            mask_matrix = matrices[(module, "module_mask")]
            neutral_matrix = matrices[(module, "neutral_mask")]
            target_pairs = [
                (
                    neutral_matrix[row_index, form_index],
                    mask_matrix[row_index, form_index],
                )
                for row_index in np.flatnonzero(selected)
                for form_index in range(len(FORMS))
            ]
            if module == "T_TCR_CD8":
                regime_switches = int(sum(neutral >= 0.75 and masked <= 0.25 for neutral, masked in target_pairs))
            else:
                regime_switches = int(sum(neutral <= 0.25 and masked >= 0.75 for neutral, masked in target_pairs))
            per_annotation = []
            for annotation in sorted(set(annotation_array[selected].tolist())):
                localized = selected & (annotation_array == annotation)
                per_annotation.append(
                    {
                        "annotation": annotation,
                        "n": int(localized.sum()),
                        "mean_effect_on_p_cd8": float(cell_effect[localized].mean()),
                    }
                )
            target_results.append(
                {
                    "gene": gene,
                    "n_cells": int(selected.sum()),
                    "pooled_mean_effect_on_p_cd8": float(cell_effect[selected].mean()),
                    "equal_class_contribution": contribution,
                    "fraction_of_aggregate_effect": (contribution / aggregate),
                    "sign_counts": _sign_counts(cell_effect[selected]),
                    "per_annotation": per_annotation,
                    "form_level_pairs": len(target_pairs),
                    "expected_direction_regime_switches": regime_switches,
                    "expected_direction_regime_switch_rate": float(regime_switches / len(target_pairs)),
                }
            )
        if not np.isclose(
            sum(record["equal_class_contribution"] for record in target_results),
            aggregate,
            rtol=0.0,
            atol=1e-15,
        ):
            raise DiscoveryAnalysisError(f"target contributions do not sum for {module}")

        module_results[module] = {
            "equal_class_effect_on_p_cd8": aggregate,
            "cd8_effect_on_p_cd8": float(cell_effect[label_array == 1].mean()),
            "nk_effect_on_p_cd8": float(cell_effect[label_array == 0].mean()),
            "cell_sign_counts": _sign_counts(cell_effect),
            "by_annotation": by_annotation,
            "by_masked_target": target_results,
        }

    condition_probabilities = {
        "unmasked": _equal_class(
            unmasked.mean(axis=1),
            label_array,
        )
    }
    for module in MODULES:
        for condition in ("module_mask", "neutral_mask"):
            condition_probabilities[f"{module}:{condition}"] = _equal_class(
                matrices[(module, condition)].mean(axis=1),
                label_array,
            )

    t_effect = cell_effects["T_TCR_CD8"]
    c_effect = cell_effects["cytotoxic_effector"]
    response_grid = {
        token: count
        for token, count in sorted(
            raw_grid.items(),
            key=lambda item: float(item[0]),
        )
    }
    mass_at_015_or_085 = sum(
        count for token, count in raw_grid.items() if np.isclose(float(token), 0.15) or np.isclose(float(token), 0.85)
    ) / len(records)

    return {
        "analysis": ("post_hoc_descriptive_localization_of_preregistered_module_comparison"),
        "primary_gate_unchanged": True,
        "n_cells": len(entity_ids),
        "class_counts": {
            "CD8": int((label_array == 1).sum()),
            "NK": int((label_array == 0).sum()),
        },
        "unique_model_calls": len(records),
        "logical_observations": logical_count,
        "condition_equal_class_p_cd8": condition_probabilities,
        "module_results": module_results,
        "cell_level_summary": {
            "positive_t_minus_cytotoxic_separation": int(((t_effect - c_effect) > 1e-12).sum()),
            "both_module_effects_expected_direction": int(((t_effect > 1e-12) & (c_effect < -1e-12)).sum()),
        },
        "response_quantization": {
            "unique_raw_probabilities": len(raw_grid),
            "raw_probability_counts": response_grid,
            "fraction_exactly_0.15_or_0.85": (mass_at_015_or_085),
        },
        "discovery_claim": (
            "In this mixed-marker subset, Haiku 4.5 shows opposing "
            "token-deletion sensitivities: a modest distributed "
            "CD3/TCR-category sensitivity and a larger context-dependent "
            "sensitivity concentrated in cells whose selected NK-associated "
            "cue was GNLY or NKG7. The localization does not support a "
            "generic cytotoxic-program interpretation."
        ),
        "boundary": (
            "Post-hoc and context-confounded localization in one "
            "label-informed reduced panel, one donor, and one model. "
            "GNLY and NKG7 are NK associated but are not NK-lineage-specific, "
            "and the selected-target strata do not isolate either gene from "
            "its cellular context. The result does not establish "
            "individual-gene causality, a biological pathway mechanism, "
            "hidden-state activation, latent knowledge, prompt invariance, "
            "or a physical law."
        ),
        "provenance": {
            "raw_checkpoint": str(RAW_PATH.relative_to(ROOT)),
            "raw_checkpoint_sha256": raw_sha256,
            "factorial_csv": str(DATA_PATH.relative_to(ROOT)),
            "factorial_csv_sha256": data_sha256,
            "call_plan_sha256": EXPECTED_PLAN_SHA256,
            "preregistration_sha256": EXPECTED_PREREG_SHA256,
            "execution_code_sha256": EXPECTED_CODE_SHA256,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    t_result = result["module_results"]["T_TCR_CD8"]
    c_result = result["module_results"]["cytotoxic_effector"]
    cytotoxic_targets = {record["gene"]: record for record in c_result["by_masked_target"]}
    lines = [
        "# PBMC68k module comparison: post-hoc discovery localization",
        "",
        "> Descriptive post-hoc localization. The preregistered aggregate",
        "> gate is unchanged; gene and subtype results are hypothesis-generating.",
        "> Terminology correction: the immutable primary report's “Primary",
        "> mechanistic result” heading should read “Primary intervention result”;",
        "> no internal mechanism was measured.",
        "",
        "## Descriptive localization",
        "",
        (
            "The intervention shows two asymmetric token-deletion "
            "sensitivities in this mixed-marker subset: a modest, distributed "
            f"CD3/TCR-category sensitivity (`{_fmt(t_result['equal_class_effect_on_p_cd8'])}`) "
            "and a larger frozen cytotoxic/NK-enriched-category sensitivity "
            f"(`{_fmt(c_result['equal_class_effect_on_p_cd8'])}`)."
        ),
        "",
        (
            "The localization does not support a generic cytotoxic-program "
            "interpretation. Cells whose selected target was `GNLY` "
            "descriptively account for "
            f"`{cytotoxic_targets['GNLY']['fraction_of_aggregate_effect']:.1%}` "
            "of the signed equal-class aggregate, and the `NKG7`-targeted "
            "stratum accounts for "
            f"`{cytotoxic_targets['NKG7']['fraction_of_aggregate_effect']:.1%}`. "
            "Small opposing contributions in the `CCL5`- and `GZMK`-targeted "
            "contexts make the signed shares sum above 100%. These are "
            "context-confounded strata, not isolated gene effects."
        ),
        "",
        "## Bidirectional condition shift",
        "",
        "| condition | equal-class P(CD8) |",
        "|---|---:|",
    ]
    labels = {
        "unmasked": "unmasked",
        "T_TCR_CD8:module_mask": "TCR/CD8 token masked",
        "T_TCR_CD8:neutral_mask": "T-specific neutral masked",
        ("cytotoxic_effector:module_mask"): "cytotoxic/NK-enriched token masked",
        ("cytotoxic_effector:neutral_mask"): "cytotoxic-specific neutral masked",
    }
    for key in (
        "unmasked",
        "T_TCR_CD8:module_mask",
        "T_TCR_CD8:neutral_mask",
        "cytotoxic_effector:module_mask",
        "cytotoxic_effector:neutral_mask",
    ):
        lines.append(f"| {labels[key]} | {result['condition_equal_class_p_cd8'][key]:.3f} |")
    lines.extend(
        [
            "",
            "## Cytotoxic/NK-enriched target localization",
            "",
            (
                "| masked target | cells | equal-class contribution | "
                "signed share of aggregate | strong form-level switches |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in sorted(
        c_result["by_masked_target"],
        key=lambda item: -abs(item["equal_class_contribution"]),
    ):
        lines.append(
            f"| {record['gene']} | {record['n_cells']} | "
            f"{_fmt(record['equal_class_contribution'])} | "
            f"{record['fraction_of_aggregate_effect']:.1%} | "
            f"{record['expected_direction_regime_switches']}/"
            f"{record['form_level_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## State dependence",
            "",
            "| annotation | n | cytotoxic/NK-enriched effect on P(CD8) |",
            "|---|---:|---:|",
        ]
    )
    for record in c_result["by_annotation"]:
        lines.append(f"| {record['annotation']} | {record['n']} | {_fmt(record['mean_effect_on_p_cd8'])} |")
    gnly_annotations = {record["annotation"]: record for record in cytotoxic_targets["GNLY"]["per_annotation"]}
    t_signs = t_result["cell_sign_counts"]
    c_signs = c_result["cell_sign_counts"]
    cell_summary = result["cell_level_summary"]
    lines.extend(
        [
            "",
            (
                "The `GNLY`-targeted stratum is itself context dependent: "
                f"CD56+ NK `{_fmt(gnly_annotations['CD56+ NK']['mean_effect_on_p_cd8'])}` "
                f"(n={gnly_annotations['CD56+ NK']['n']}), cytotoxic CD8 "
                f"`{_fmt(gnly_annotations['CD8+ Cytotoxic T']['mean_effect_on_p_cd8'])}` "
                f"(n={gnly_annotations['CD8+ Cytotoxic T']['n']}), and naive CD8 "
                f"`{_fmt(gnly_annotations['CD8+/CD45RA+ Naive Cytotoxic']['mean_effect_on_p_cd8'])}` "
                f"(n={gnly_annotations['CD8+/CD45RA+ Naive Cytotoxic']['n']})."
            ),
            "",
            "## Cell-level heterogeneity",
            "",
            (
                "The TCR/CD8 effect had the expected positive direction in "
                f"`{t_signs['positive']}`/65 cells (`{t_signs['zero']}` zero, "
                f"`{t_signs['negative']}` opposite); the cytotoxic/NK-enriched "
                f"effect had the expected negative direction in `{c_signs['negative']}`/65 "
                f"(`{c_signs['zero']}` zero, `{c_signs['positive']}` opposite). "
                f"Only `{cell_summary['both_module_effects_expected_direction']}`/65 "
                "cells had both expected signs, while T-minus-cytotoxic "
                f"separation was positive in "
                f"`{cell_summary['positive_t_minus_cytotoxic_separation']}`/65. "
                "The aggregate and class-mean sign gates therefore do not imply "
                "universal per-cell behavior."
            ),
        ]
    )
    quantization = result["response_quantization"]
    lines.extend(
        [
            "",
            "## Output regime",
            "",
            (
                f"Only `{quantization['unique_raw_probabilities']}` raw "
                "probability values were emitted; "
                f"`{quantization['fraction_exactly_0.15_or_0.85']:.1%}` "
                "of calls were exactly `0.15` or `0.85`. In the "
                "`GNLY`-targeted stratum, "
                f"`{cytotoxic_targets['GNLY']['expected_direction_regime_switches']}`/"
                f"`{cytotoxic_targets['GNLY']['form_level_pairs']}` prompt-form "
                "pairs switched from low (≤0.25) to high (≥0.75) P(CD8) "
                "after masking. This looks more like cue-triggered category "
                "switching than calibrated additive evidence integration."
            ),
            "",
            "## Claim boundary",
            "",
            result["boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=OUT_ROOT / "discovery_localization.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=OUT_ROOT / "discovery_localization.md",
    )
    args = parser.parse_args()

    result = analyze()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "markdown_out": str(args.markdown_out),
                "discovery_claim": result["discovery_claim"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

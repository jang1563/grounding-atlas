"""Recompute bounded post-hoc localization for the GSE96583 donor experiment.

This script binds the completed raw checkpoint and reconstructs every
cell/module/form effect before describing target-token strata. Target strata
were not confirmatory hypotheses and cannot support individual-gene biological
causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import single_cell_donor_context_factorial as factorial  # noqa: E402

RAW_PATH = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "donor_context_factorial"
    / "claude-haiku-4-5-20251001_raw.jsonl"
)
PRIMARY_RESULT_PATH = RAW_PATH.with_name("claude-haiku-4-5-20251001.json")
DEFAULT_JSON_OUT = RAW_PATH.with_name("discovery_localization.json")
DEFAULT_MARKDOWN_OUT = RAW_PATH.with_name("discovery_localization.md")
EXPECTED_RAW_SHA256 = (
    "aaf048375f4fa8f5b972a406185bfe15618323b8eacdff1b25db938ea2e63aea"
)
EXPECTED_PRIMARY_RESULT_SHA256 = (
    "f2d68cdf11615f616df3f7be3089caf569e8ee1e7d967309ca02bcbe43c40e4b"
)
EXPECTED_RUNNER_SHA256 = (
    "782ceb97226a97cee4acf9dab19e8c6f4166876d41d088dead34cb5ca45f24f2"
)
SENTINELS = {"GNLY", "NKG7", "CCL5"}


class GSE96583DiscoveryError(ValueError):
    """Raised when completed artifacts do not satisfy the frozen contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_effect_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    for path, expected, label in (
        (RAW_PATH, EXPECTED_RAW_SHA256, "raw checkpoint"),
        (
            PRIMARY_RESULT_PATH,
            EXPECTED_PRIMARY_RESULT_SHA256,
            "primary result",
        ),
        (Path(factorial.__file__), EXPECTED_RUNNER_SHA256, "execution runner"),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise GSE96583DiscoveryError(
                f"{label} missing or differs from its frozen SHA-256"
            )

    base_rows, intervention_rows, _ = factorial._load_inputs()
    plan, plan_sha256 = factorial._plan(
        base_rows,
        intervention_rows,
        factorial.EXPECTED_MODEL,
    )
    checkpoint = factorial._read_checkpoint(RAW_PATH)
    plan_by_id = {item["request_id"]: item for item in plan}
    if set(checkpoint) != set(plan_by_id):
        raise GSE96583DiscoveryError("raw checkpoint does not match the plan")

    base_by_entity, intervention_by_key = factorial._metadata_maps(
        base_rows,
        intervention_rows,
    )
    values: dict[tuple[str, str, str], np.ndarray] = defaultdict(
        lambda: np.full(len(factorial.FORMS), np.nan)
    )
    records: list[dict[str, Any]] = []
    for request_id, record in checkpoint.items():
        planned = plan_by_id[request_id]
        factorial._validate_checkpoint_record(
            record,
            planned,
            model=factorial.EXPECTED_MODEL,
            base=base_by_entity[planned["entity_id"]],
            intervention_by_key=intervention_by_key,
            data_sha256=factorial.EXPECTED_DATA_SHA256,
            manifest_sha256=factorial.EXPECTED_MANIFEST_SHA256,
            prereg_sha256=factorial.EXPECTED_PREREG_SHA256,
            plan_sha256=plan_sha256,
            execution_code_sha256=EXPECTED_RUNNER_SHA256,
        )
        form_index = list(factorial.FORMS).index(record["form"])
        aligned = float(record["aligned_probability_a"])
        for assignment in record["assignments"]:
            values[
                (
                    record["entity_id"],
                    assignment["module"],
                    assignment["condition"],
                )
            ][form_index] = aligned
        records.append(record)

    effect_rows = []
    for (entity_id, module), intervention in intervention_by_key.items():
        module_values = values[(entity_id, module, "module_mask")]
        neutral_values = values[(entity_id, module, "neutral_mask")]
        if not np.isfinite(module_values).all() or not np.isfinite(
            neutral_values
        ).all():
            raise GSE96583DiscoveryError("an intervention lacks four forms")
        base = base_by_entity[entity_id]
        form_effects = neutral_values - module_values
        effect_rows.append(
            {
                "entity_id": entity_id,
                "donor_id": base["donor_id"],
                "sampling_context": base["sampling_context"],
                "reference_annotation": base["reference_annotation"],
                "module": module,
                "target_gene": intervention["target_genes"],
                "effect_on_p_cd8": float(form_effects.mean()),
                "form_effects": {
                    form: float(form_effects[index])
                    for index, form in enumerate(factorial.FORMS)
                },
            }
        )
    return effect_rows, base_rows, intervention_rows, records


def _component_localization(
    effect_rows: list[dict[str, Any]],
    context: str,
    module: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in effect_rows
        if row["sampling_context"] == context and row["module"] == module
    ]
    donors = sorted(
        {row["donor_id"] for row in selected},
        key=lambda value: int(value),
    )
    total_cells_by_donor = Counter(row["donor_id"] for row in selected)
    component_mean = float(
        np.mean(
            [
                np.mean(
                    [
                        row["effect_on_p_cd8"]
                        for row in selected
                        if row["donor_id"] == donor
                    ]
                )
                for donor in donors
            ]
        )
    )
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_target[row["target_gene"]].append(row)

    strata = []
    for target, rows in sorted(
        by_target.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        donor_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            donor_values[row["donor_id"]].append(row["effect_on_p_cd8"])
        contribution = float(
            np.mean(
                [
                    sum(donor_values.get(donor, []))
                    / total_cells_by_donor[donor]
                    for donor in donors
                ]
            )
        )
        values = np.asarray(
            [row["effect_on_p_cd8"] for row in rows],
            dtype=float,
        )
        strata.append(
            {
                "target_gene": target,
                "n_cells": len(rows),
                "n_donors": len(donor_values),
                "cell_mean_effect_on_p_cd8": float(values.mean()),
                "donor_equal_within_stratum_mean_effect_on_p_cd8": float(
                    np.mean(
                        [
                            np.mean(donor_values[donor])
                            for donor in sorted(
                                donor_values,
                                key=lambda value: int(value),
                            )
                        ]
                    )
                ),
                "contribution_to_component_donor_mean": contribution,
                "signed_share_of_component_mean": (
                    contribution / component_mean
                    if component_mean != 0.0
                    else None
                ),
                "cell_sign_counts": {
                    "negative": int((values < 0).sum()),
                    "zero": int((values == 0).sum()),
                    "positive": int((values > 0).sum()),
                },
            }
        )
    if not np.isclose(
        sum(row["contribution_to_component_donor_mean"] for row in strata),
        component_mean,
        rtol=0.0,
        atol=1e-15,
    ):
        raise GSE96583DiscoveryError("target contributions do not decompose")
    return {
        "sampling_context": context,
        "module": module,
        "n_cells": len(selected),
        "n_donors": len(donors),
        "component_donor_mean": component_mean,
        "target_strata": strata,
    }


def analyze() -> dict[str, Any]:
    effect_rows, base_rows, intervention_rows, records = (
        _validated_effect_rows()
    )
    components = [
        _component_localization(effect_rows, context, module)
        for context, modules in factorial.CONTEXT_MODULES.items()
        for module in modules
    ]

    triple_support = Counter()
    for row in intervention_rows:
        if (
            row["sampling_context"] == factorial.RECEPTOR_CONTEXT
            and row["module"] == "cytotoxic_effector"
            and SENTINELS <= set(row["module_hits"].split(";"))
        ):
            triple_support[row["donor_id"]] += 1
    if len(triple_support) != factorial.EXPECTED_DONORS:
        raise GSE96583DiscoveryError(
            "the prospective sentinel factorial lacks an eligible donor"
        )

    raw_frequency = Counter(str(record["raw_output"]).strip() for record in records)
    top_four = {"0.15", "0.25", "0.75", "0.85"}
    return {
        "analysis_id": (
            "gse96583-cd8-nk-context-module-donor-replication-v1:"
            "posthoc-target-localization"
        ),
        "status": "post_hoc_descriptive",
        "components": components,
        "raw_output_quantization": {
            "distinct_raw_outputs": len(raw_frequency),
            "frequency": dict(
                sorted(
                    raw_frequency.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "top_four_values": sorted(top_four),
            "top_four_count": sum(
                count
                for value, count in raw_frequency.items()
                if value in top_four
            ),
            "top_four_fraction": (
                sum(
                    count
                    for value, count in raw_frequency.items()
                    if value in top_four
                )
                / len(records)
            ),
        },
        "prospective_sentinel_factorial_feasibility": {
            "required_top50_tokens": sorted(SENTINELS),
            "eligible_selected_receptor_context_cells_by_donor": dict(
                sorted(
                    triple_support.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "minimum_per_donor": min(triple_support.values()),
            "feasible_for_one_cell_per_donor": True,
        },
        "boundary": (
            "Target identity was selected from expression rank and inspected "
            "after the primary run. These strata motivate prospective "
            "single-token hypotheses but do not isolate biological gene "
            "effects, homogeneous pathways, latent knowledge, hidden-state "
            "activation, or a physical law."
        ),
        "provenance": {
            "raw_checkpoint": str(RAW_PATH.relative_to(ROOT)),
            "raw_checkpoint_sha256": EXPECTED_RAW_SHA256,
            "primary_result": str(PRIMARY_RESULT_PATH.relative_to(ROOT)),
            "primary_result_sha256": EXPECTED_PRIMARY_RESULT_SHA256,
            "execution_runner_sha256": EXPECTED_RUNNER_SHA256,
            "input_csv_sha256": factorial.EXPECTED_DATA_SHA256,
            "input_manifest_sha256": factorial.EXPECTED_MANIFEST_SHA256,
            "preregistration_sha256": factorial.EXPECTED_PREREG_SHA256,
            "base_cells": len(base_rows),
            "interventions": len(intervention_rows),
            "raw_records": len(records),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GSE96583 post-hoc target-token localization",
        "",
        "This artifact recomputes target-token strata from the hash-bound raw",
        "checkpoint. It is descriptive and was not a confirmatory target-gene",
        "test.",
        "",
    ]
    for component in result["components"]:
        lines.extend(
            [
                (
                    f"## {component['sampling_context']} / "
                    f"{component['module']}"
                ),
                "",
                (
                    "Donor-level component mean: "
                    f"`{component['component_donor_mean']:+.6f}`."
                ),
                "",
                "| target | cells | donors | donor-equal stratum mean | contribution | signed share | negative / zero / positive |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in component["target_strata"]:
            share = row["signed_share_of_component_mean"]
            signs = row["cell_sign_counts"]
            lines.append(
                f"| {row['target_gene']} | {row['n_cells']} "
                f"| {row['n_donors']} "
                f"| {row['donor_equal_within_stratum_mean_effect_on_p_cd8']:+.6f} "
                f"| {row['contribution_to_component_donor_mean']:+.6f} "
                f"| {share:.1%} "
                f"| {signs['negative']} / {signs['zero']} / "
                f"{signs['positive']} |"
            )
        lines.append("")
    quantization = result["raw_output_quantization"]
    feasibility = result["prospective_sentinel_factorial_feasibility"]
    lines.extend(
        [
            "## Quantization and next-test feasibility",
            "",
            (
                f"- Distinct raw outputs: `{quantization['distinct_raw_outputs']}`; "
                f"the four values 0.15/0.25/0.75/0.85 account for "
                f"`{quantization['top_four_fraction']:.1%}` of calls."
            ),
            (
                "- Selected receptor-context cells containing GNLY, NKG7, "
                "and CCL5 in the top 50 exist in all eight donors; minimum "
                f"per donor: `{feasibility['minimum_per_donor']}`."
            ),
            "",
            "## Boundary",
            "",
            result["boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=DEFAULT_MARKDOWN_OUT,
    )
    args = parser.parse_args()
    result = analyze()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
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
                "raw_records": result["provenance"]["raw_records"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

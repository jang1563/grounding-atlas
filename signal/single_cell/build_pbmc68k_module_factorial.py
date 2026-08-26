"""Build a within-frame label-blind PBMC68k two-category masking comparison.

The frozen parent marker list is repartitioned into three operational,
disjoint candidate modules:

* TCR/CD8-associated markers;
* NK receptor/identity-associated markers; and
* shared cytotoxic-effector markers.

The reduced target artifact censors the NK-receptor module, so only TCR/CD8
and cytotoxic-effector modules are executed.  Within the label-defined CD8/NK
task frame, the common-support population contains every cell with at least
one top-50 hit from both executed modules.  Target labels do not select
common-support inclusion or either intervention.  The experiment masks the
highest-ranked hit from each module (k=1), paired with a separately matched
source-neutral control.  When both matching problems choose the same neutral
input, execution reuses its single model response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import build_pbmc68k_program_mask as prior
import numpy as np

ANALYSIS_ID = "pbmc68k-cd8-nk-module-factorial-v1"
DEFAULT_OUT = Path(__file__).with_name("pbmc68k_cd8_nk_module_factorial.csv")
DEFAULT_MANIFEST = Path(__file__).with_name("pbmc68k_cd8_nk_module_factorial.manifest.json")

T_TCR_CD8_MODULE = frozenset("CD3D CD3E CD3G CD8A CD8B".split())
NK_RECEPTOR_MODULE = frozenset("KLRD1 KLRF1 KLRC1 KLRC2 NCAM1 FCGR3A NCR1".split())
CYTOTOXIC_EFFECTOR_MODULE = frozenset("GNLY NKG7 PRF1 GZMA GZMB GZMH GZMK CTSW CCL5 FGFBP2 XCL1 XCL2".split())
AMBIGUOUS_EXCLUDED = frozenset("CD247 KLRB1 IL7R TYROBP CST7 SPON2".split())
MODULES = {
    "T_TCR_CD8": T_TCR_CD8_MODULE,
    "cytotoxic_effector": CYTOTOXIC_EFFECTOR_MODULE,
}
DEFERRED_MODULES = {
    "NK_receptor_identity": NK_RECEPTOR_MODULE,
}
ALL_CANDIDATE_MODULES = {**MODULES, **DEFERRED_MODULES}
LABEL_MAP = {
    "CD8+ Cytotoxic T": 1,
    "CD8+/CD45RA+ Naive Cytotoxic": 1,
    "CD56+ NK": 0,
}

if len(set().union(*ALL_CANDIDATE_MODULES.values(), AMBIGUOUS_EXCLUDED)) != len(prior.FAMOUS_PARENT):
    raise RuntimeError("module partition contains overlapping parent genes")
if set().union(*ALL_CANDIDATE_MODULES.values(), AMBIGUOUS_EXCLUDED) != prior.FAMOUS_PARENT:
    raise RuntimeError("module partition does not reconstruct the frozen parent marker set")


class ModuleFactorialBuildError(ValueError):
    """Raised when a frozen input or derived module mask violates the design."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialize(values: list[Any]) -> str:
    return ";".join(str(value) for value in values)


def _source_neutral_controls(
    source: ad.AnnData,
    universe: np.ndarray,
) -> tuple[set[str], dict[str, float]]:
    source_gene_names = np.asarray(
        list(map(str, source.raw.var_names if source.raw is not None else source.var_names)),
        dtype=object,
    )
    source_universe_mask = np.asarray([gene in set(universe) for gene in source_gene_names], dtype=bool)
    annotations = source.obs["louvain"].astype(str)
    selected_mask = annotations.isin(["CD8 T cells", "NK cells"]).to_numpy()
    expression = source.raw.X if source.raw is not None else source.X
    expression = prior._dense(expression[selected_mask][:, source_universe_mask])
    source_universe = source_gene_names[source_universe_mask].astype(str)
    ranked = [prior._rank_cell(source_universe, expression[row]) for row in range(expression.shape[0])]
    labels = (annotations[selected_mask].to_numpy() == "CD8 T cells").astype(int)

    gaps = {}
    for gene in universe:
        cd8_prevalence = np.mean([gene in ranked[row][0] for row in np.flatnonzero(labels == 1)])
        nk_prevalence = np.mean([gene in ranked[row][0] for row in np.flatnonzero(labels == 0)])
        gaps[gene] = float(abs(cd8_prevalence - nk_prevalence))
    controls = {gene for gene, gap in gaps.items() if gap <= prior.CONTROL_MAX_SOURCE_PREVALENCE_GAP}
    return controls, gaps


def _base_row(
    cell_id: str,
    annotation: str,
    label: int,
    genes: list[str],
) -> dict[str, Any]:
    sentence = " ".join(genes)
    return {
        "row_type": "base",
        "entity_id": f"pbmc68k:{cell_id}",
        "cell_barcode": cell_id,
        "technical_group": cell_id.rsplit("-", 1)[-1],
        "reference_annotation": annotation,
        "label_a": label,
        "module": "none",
        "dose_k": 0,
        "module_hits": "",
        "module_hit_count": 0,
        "target_genes": "",
        "target_values": "",
        "control_genes": "",
        "control_values": "",
        "control_rank_distance": "",
        "control_expression_distance": "",
        "control_prevalence_distance": "",
        "control_token_length_distance": "",
        "control_total_cost": "",
        "original_sentence": sentence,
        "module_mask_sentence": sentence,
        "control_mask_sentence": sentence,
    }


def build(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    source_sha256 = prior._verify_hash(
        source_path,
        prior.EXPECTED_SOURCE_SHA256,
        "PBMC3k source",
    )
    target_sha256 = prior._verify_hash(
        target_path,
        prior.EXPECTED_TARGET_SHA256,
        "PBMC68k target",
    )

    source = ad.read_h5ad(source_path, backed="r")
    target = ad.read_h5ad(target_path)
    source_gene_names = set(map(str, source.raw.var_names if source.raw is not None else source.var_names))
    target_gene_names = np.asarray(
        list(map(str, target.raw.var_names if target.raw is not None else target.var_names)),
        dtype=object,
    )
    universe_mask = np.asarray(
        [gene in source_gene_names and prior.HOUSEKEEPING.match(gene) is None for gene in target_gene_names],
        dtype=bool,
    )
    universe = target_gene_names[universe_mask].astype(str)
    if len(universe) != 750:
        raise ModuleFactorialBuildError(
            f"frozen common non-housekeeping universe changed: expected 750, observed {len(universe)}"
        )

    source_neutral_controls, source_prevalence_gap = _source_neutral_controls(
        source,
        universe,
    )
    annotations = target.obs["bulk_labels"].astype(str)
    selected_mask = annotations.isin(LABEL_MAP).to_numpy()
    selected_obs = target.obs.loc[selected_mask].copy()
    selected_annotations = annotations[selected_mask].to_numpy()
    expression = target.raw.X if target.raw is not None else target.X
    expression = prior._dense(expression[selected_mask][:, universe_mask])
    ranked = [prior._rank_cell(universe, expression[row]) for row in range(len(selected_obs))]
    prevalence_counts = Counter(gene for genes, _ in ranked for gene in genes)
    prevalence = {gene: count / len(ranked) for gene, count in prevalence_counts.items()}

    rows: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    candidate_support: dict[str, Counter[str]] = {module: Counter() for module in ALL_CANDIDATE_MODULES}
    module_gene_frequencies: dict[str, Counter[str]] = {module: Counter() for module in ALL_CANDIDATE_MODULES}

    for row_index, (cell_id, annotation) in enumerate(zip(selected_obs.index.astype(str), selected_annotations)):
        label = LABEL_MAP[annotation]
        genes, values = ranked[row_index]
        hits_by_module: dict[str, list[str]] = {}

        for module_name, module_genes in ALL_CANDIDATE_MODULES.items():
            hits = [gene for gene in genes if gene in module_genes]
            hits_by_module[module_name] = hits
            candidate_support[module_name]["all_cells"] += 1
            candidate_support[module_name][f"{annotation}:all"] += 1
            candidate_support[module_name][f"{annotation}:ge1"] += int(len(hits) >= 1)
            candidate_support[module_name][f"{annotation}:ge2"] += int(len(hits) >= 2)
            candidate_support[module_name][f"{annotation}:ge3"] += int(len(hits) >= 3)
            module_gene_frequencies[module_name].update(hits)

        if not all(hits_by_module[module] for module in MODULES):
            continue

        rows.append(_base_row(cell_id, annotation, label, genes))
        for module_name in MODULES:
            hits = hits_by_module[module_name]
            target_genes = hits[:1]
            control_genes, control_values, matches = prior._match_control_genes(
                genes,
                values,
                target_genes,
                prevalence,
                source_neutral_controls,
            )
            for match_index, match in enumerate(matches):
                match_records.append(
                    {
                        "entity_id": f"pbmc68k:{cell_id}",
                        "module": module_name,
                        "matched_pair_index": match_index + 1,
                        **match,
                    }
                )

            target_values = [values[genes.index(gene)] for gene in target_genes]
            rows.append(
                {
                    "row_type": "intervention",
                    "entity_id": f"pbmc68k:{cell_id}",
                    "cell_barcode": cell_id,
                    "technical_group": cell_id.rsplit("-", 1)[-1],
                    "reference_annotation": annotation,
                    "label_a": label,
                    "module": module_name,
                    "dose_k": 1,
                    "module_hits": _serialize(hits),
                    "module_hit_count": len(hits),
                    "target_genes": _serialize(target_genes),
                    "target_values": _serialize([f"{value:.6g}" for value in target_values]),
                    "control_genes": _serialize(control_genes),
                    "control_values": _serialize([f"{value:.6g}" for value in control_values]),
                    "control_rank_distance": matches[0]["rank_distance"],
                    "control_expression_distance": f"{float(matches[0]['expression_distance']):.6g}",
                    "control_prevalence_distance": f"{float(matches[0]['prevalence_distance']):.6g}",
                    "control_token_length_distance": matches[0]["token_length_distance"],
                    "control_total_cost": f"{float(matches[0]['total_cost']):.6g}",
                    "original_sentence": " ".join(genes),
                    "module_mask_sentence": prior._mask_sentence(
                        genes,
                        target_genes,
                    ),
                    "control_mask_sentence": prior._mask_sentence(
                        genes,
                        control_genes,
                    ),
                }
            )

    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [row for row in rows if row["row_type"] == "intervention"]
    base_counts = Counter(int(row["label_a"]) for row in base_rows)
    if len(base_rows) != 65 or base_counts != Counter({1: 55, 0: 10}):
        raise ModuleFactorialBuildError(
            f"frozen common-support cell set changed: n={len(base_rows)}, counts={dict(base_counts)}"
        )
    expected_interventions = 130
    if len(intervention_rows) != expected_interventions:
        raise ModuleFactorialBuildError(
            f"frozen intervention count changed: expected {expected_interventions}, observed {len(intervention_rows)}"
        )
    logical_condition_form_observations = len(base_rows) * 4 + len(intervention_rows) * 2 * 4
    if logical_condition_form_observations != 1300:
        raise ModuleFactorialBuildError(
            f"expected 1300 logical condition-form observations, observed {logical_condition_form_observations}"
        )
    unique_cell_inputs = {(row["entity_id"], row["original_sentence"]) for row in base_rows}
    unique_cell_inputs.update((row["entity_id"], row["module_mask_sentence"]) for row in intervention_rows)
    unique_cell_inputs.update((row["entity_id"], row["control_mask_sentence"]) for row in intervention_rows)
    expected_unique_api_calls = len(unique_cell_inputs) * 4
    shared_neutral_control_cells = sum(
        len({row["control_mask_sentence"] for row in intervention_rows if row["entity_id"] == entity_id}) == 1
        for entity_id in {row["entity_id"] for row in base_rows}
    )
    if expected_unique_api_calls != 1256 or shared_neutral_control_cells != 11:
        raise ModuleFactorialBuildError(
            "frozen request de-duplication changed: "
            f"unique_calls={expected_unique_api_calls}, "
            f"shared_neutral_cells={shared_neutral_control_cells}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rank_distances = np.asarray(
        [float(record["rank_distance"]) for record in match_records],
        dtype=float,
    )
    expression_distances = np.asarray(
        [float(record["expression_distance"]) for record in match_records],
        dtype=float,
    )
    prevalence_distances = np.asarray(
        [float(record["prevalence_distance"]) for record in match_records],
        dtype=float,
    )
    module_dose_counts = Counter(
        (
            row["module"],
            int(row["dose_k"]),
            row["reference_annotation"],
        )
        for row in intervention_rows
    )
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "freeze_date": "2026-07-27",
        "claim_scope": (
            "single_model_single_external_cohort_label_blind_text_input_"
            "module_ablation_not_hidden_state_or_biological_causality"
        ),
        "source_pbmc3k": {
            "path": str(source_path),
            "sha256": source_sha256,
            "role": "frozen gene universe, parent marker set, and neutral-control filter",
        },
        "target_pbmc68k": {
            "path": str(target_path),
            "sha256": target_sha256,
            "cohort": "10x Fresh 68k PBMC Donor A; reduced annotated Scanpy artifact",
            "scanpy_documentation": (
                "https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc68k_reduced.html"
            ),
            "primary_source_doi": "10.1038/ncomms14049",
        },
        "module_provenance": {
            "parent_file": "signal/single_cell/build_cd8t_nk.py",
            "parent_git_commit": "20f94201c60878b83dbae182b8447248666ede29",
            "parent_set": sorted(prior.FAMOUS_PARENT),
            "T_TCR_CD8": sorted(T_TCR_CD8_MODULE),
            "NK_receptor_identity_deferred": sorted(NK_RECEPTOR_MODULE),
            "cytotoxic_effector": sorted(CYTOTOXIC_EFFECTOR_MODULE),
            "ambiguous_excluded": sorted(AMBIGUOUS_EXCLUDED),
            "executed_modules": sorted(MODULES),
            "deferred_modules": sorted(DEFERRED_MODULES),
            "deferred_reason": (
                "PBMC68k-reduced has no k>=2 support for the canonical "
                "NK-receptor module and only FCGR3A contributes at k=1"
            ),
            "operational_boundary": (
                "fixed marker categories for a token-ablation experiment; "
                "not purified pathways or claims of gene-specific biological causality"
            ),
        },
        "construction": {
            "top_k": prior.TOP_K,
            "mask_token": prior.MASK_TOKEN,
            "all_target_cells": len(ranked),
            "common_support_cells": len(base_rows),
            "common_support_class_counts": {
                "CD8": base_counts[1],
                "NK": base_counts[0],
            },
            "intervention_rows": len(intervention_rows),
            "logical_condition_form_observations": logical_condition_form_observations,
            "expected_unique_api_calls": expected_unique_api_calls,
            "shared_neutral_control_cells": shared_neutral_control_cells,
            "common_non_housekeeping_gene_count": len(universe),
            "common_gene_universe_sha256": hashlib.sha256(("\n".join(sorted(universe)) + "\n").encode()).hexdigest(),
            "dose_policy": {
                "primary": (
                    "k=1 highest-ranked hit from each executed module on the two-module common-support population"
                ),
                "higher_dose": (
                    "not executed because common support falls to 36 cells at "
                    "k=2 and 9 cells at k=3, including only 2 and 1 NK cells"
                ),
                "NK_receptor_identity": ("not executed because the reduced feature panel censors the canonical module"),
            },
            "candidate_module_support": {
                module: dict(sorted(counts.items())) for module, counts in candidate_support.items()
            },
            "module_dose_annotation_counts": [
                {
                    "module": module,
                    "dose_k": dose,
                    "reference_annotation": annotation,
                    "n": count,
                }
                for (module, dose, annotation), count in sorted(module_dose_counts.items())
            ],
            "module_gene_top50_frequencies": {
                module: [
                    {"gene": gene, "cells": count}
                    for gene, count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ]
                for module, counts in module_gene_frequencies.items()
            },
            "control_matching": {
                "method": (
                    "within-cell linear-sum assignment on rank, expression, "
                    "global top-50 prevalence, and token length; all frozen "
                    "parent markers and source-PBMC3k class-associated genes excluded"
                ),
                "source_neutrality_rule": (
                    "absolute CD8-vs-NK top-50 prevalence gap in PBMC3k "
                    f"<= {prior.CONTROL_MAX_SOURCE_PREVALENCE_GAP:.2f}"
                ),
                "source_neutral_control_gene_count": len(source_neutral_controls),
                "independently_matched_controls_may_coincide": True,
                "source_prevalence_gap_sha256": hashlib.sha256(
                    (
                        "\n".join(
                            f"{gene}\t{source_prevalence_gap[gene]:.12g}" for gene in sorted(source_prevalence_gap)
                        )
                        + "\n"
                    ).encode()
                ).hexdigest(),
                "matched_gene_pairs": len(match_records),
                "median_rank_distance": float(np.median(rank_distances)),
                "max_rank_distance": float(rank_distances.max()),
                "median_expression_distance": float(np.median(expression_distances)),
                "max_expression_distance": float(expression_distances.max()),
                "median_prevalence_distance": float(np.median(prevalence_distances)),
                "max_prevalence_distance": float(prevalence_distances.max()),
            },
        },
        "artifacts": {
            "builder_sha256": _sha256(Path(__file__)),
            "csv_path": str(output_path.relative_to(prior.ROOT)),
            "csv_sha256": _sha256(output_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-h5ad",
        type=Path,
        default=prior.DEFAULT_SOURCE,
    )
    parser.add_argument(
        "--target-h5ad",
        type=Path,
        default=prior.DEFAULT_TARGET,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build(
        args.source_h5ad.resolve(),
        args.target_h5ad.resolve(),
        args.out.resolve(),
        args.manifest.resolve(),
    )
    print(
        json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "csv": manifest["artifacts"]["csv_path"],
                "csv_sha256": manifest["artifacts"]["csv_sha256"],
                "common_support_cells": manifest["construction"]["common_support_cells"],
                "intervention_rows": manifest["construction"]["intervention_rows"],
                "logical_condition_form_observations": manifest["construction"]["logical_condition_form_observations"],
                "expected_unique_api_calls": manifest["construction"]["expected_unique_api_calls"],
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

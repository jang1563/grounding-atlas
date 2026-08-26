"""Build the frozen PBMC68k CD8/NK gene-program masking intervention.

The biological programs are a disjoint partition of the CD8/NK marker set that
was committed in ``build_cd8t_nk.py`` on 2026-06-19, before the model outputs
used in this experiment existed.  The source feature universe is PBMC3k; the
evaluation cells come from Scanpy's independently annotated PBMC68k Donor A
artifact.

For every eligible evaluation cell, the script masks up to three highest-ranked
genes from the program matching its reference label.  A control arm masks the
same number of non-program genes, jointly matched within the cell on expression
rank, log-normalized expression, global top-50 prevalence, and token length.
No LLM output is used to select cells, programs, or controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "pbmc3k_processed.h5ad"
DEFAULT_TARGET = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "scanpy"
    / "datasets"
    / "10x_pbmc68k_reduced.h5ad"
)
DEFAULT_OUT = Path(__file__).with_name("pbmc68k_cd8_nk_program_mask.csv")
DEFAULT_MANIFEST = Path(__file__).with_name("pbmc68k_cd8_nk_program_mask.manifest.json")

EXPECTED_SOURCE_SHA256 = "0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38"
EXPECTED_TARGET_SHA256 = "863e19914ab2d4ba97edc9623ac3a343c0461f1e40b121bfb5fa92638b22e9bd"
HOUSEKEEPING = re.compile(r"^(RP[LS]|MRP[LS]|MT-|MALAT1)")
MASK_TOKEN = "MASKED_GENE"
TOP_K = 50
MAX_MASK = 3
CONTROL_MAX_SOURCE_PREVALENCE_GAP = 0.10

# Frozen parent set from build_cd8t_nk.py commit 20f94201c60878b83dbae182b8447248666ede29.
FAMOUS_PARENT = frozenset(
    (
        "CD8A CD8B GZMB GZMA GZMH GZMK NKG7 GNLY KLRD1 KLRF1 KLRB1 "
        "KLRC1 KLRC2 NCAM1 FCGR3A PRF1 CCL5 NCR1 CD3D CD3E CD3G CD247 "
        "IL7R FGFBP2 SPON2 CST7 CTSW XCL1 XCL2 TYROBP"
    ).split()
)
CD8_IDENTITY_PROGRAM = frozenset("CD8A CD8B GZMK KLRB1 CD3D CD3E CD3G CD247 IL7R".split())
NK_IDENTITY_PROGRAM = frozenset(
    ("GZMB GZMH NKG7 GNLY KLRD1 KLRF1 KLRC1 KLRC2 NCAM1 FCGR3A PRF1 NCR1 FGFBP2 SPON2 XCL1 XCL2 TYROBP").split()
)
SHARED_EXCLUDED = FAMOUS_PARENT - CD8_IDENTITY_PROGRAM - NK_IDENTITY_PROGRAM
if CD8_IDENTITY_PROGRAM & NK_IDENTITY_PROGRAM:
    raise RuntimeError("frozen CD8 and NK programs must be disjoint")
if CD8_IDENTITY_PROGRAM | NK_IDENTITY_PROGRAM | SHARED_EXCLUDED != FAMOUS_PARENT:
    raise RuntimeError("frozen program partition does not reconstruct the parent marker set")

LABEL_MAP = {
    "CD8+ Cytotoxic T": (1, "CD8_identity"),
    "CD8+/CD45RA+ Naive Cytotoxic": (1, "CD8_identity"),
    "CD56+ NK": (0, "NK_identity"),
}
PROGRAMS = {
    "CD8_identity": CD8_IDENTITY_PROGRAM,
    "NK_identity": NK_IDENTITY_PROGRAM,
}


class ProgramMaskBuildError(ValueError):
    """Raised when a frozen input or derived mask violates the design."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ProgramMaskBuildError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")
    return observed


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _rank_cell(genes: np.ndarray, values: np.ndarray) -> tuple[list[str], list[float]]:
    order = np.lexsort((genes, -values))
    order = order[values[order] > 0][:TOP_K]
    if len(order) != TOP_K:
        raise ProgramMaskBuildError(
            f"cell has only {len(order)} expressed genes in the frozen universe; expected {TOP_K}"
        )
    return genes[order].tolist(), values[order].astype(float).tolist()


def _mask_sentence(genes: list[str], masked_genes: list[str]) -> str:
    mask = set(masked_genes)
    if len(mask) != len(masked_genes):
        raise ProgramMaskBuildError("a mask cannot contain duplicate genes")
    if not mask <= set(genes):
        raise ProgramMaskBuildError("a mask contains a gene absent from the cell sentence")
    return " ".join(MASK_TOKEN if gene in mask else gene for gene in genes)


def _match_control_genes(
    genes: list[str],
    values: list[float],
    target_genes: list[str],
    prevalence: dict[str, float],
    source_neutral_controls: set[str],
) -> tuple[list[str], list[float], list[dict[str, float | int | str]]]:
    positions = {gene: rank for rank, gene in enumerate(genes)}
    candidates = [gene for gene in genes if gene not in FAMOUS_PARENT and gene in source_neutral_controls]
    if len(candidates) < len(target_genes):
        raise ProgramMaskBuildError("insufficient non-program genes for the matched control")

    cost = np.empty((len(target_genes), len(candidates)), dtype=float)
    components: dict[tuple[int, int], dict[str, float | int | str]] = {}
    for row, target in enumerate(target_genes):
        target_rank = positions[target]
        target_value = values[target_rank]
        for column, candidate in enumerate(candidates):
            candidate_rank = positions[candidate]
            candidate_value = values[candidate_rank]
            rank_distance = abs(target_rank - candidate_rank)
            expression_distance = abs(target_value - candidate_value)
            prevalence_distance = abs(prevalence.get(target, 0.0) - prevalence.get(candidate, 0.0))
            token_length_distance = abs(len(target) - len(candidate))
            total = (
                rank_distance / 3.0
                + expression_distance / 0.5
                + prevalence_distance / 0.10
                + token_length_distance / 3.0
            )
            cost[row, column] = total
            components[(row, column)] = {
                "target_gene": target,
                "control_gene": candidate,
                "target_rank_1based": target_rank + 1,
                "control_rank_1based": candidate_rank + 1,
                "rank_distance": rank_distance,
                "expression_distance": expression_distance,
                "prevalence_distance": prevalence_distance,
                "token_length_distance": token_length_distance,
                "total_cost": total,
            }

    rows, columns = linear_sum_assignment(cost)
    if list(rows) != list(range(len(target_genes))):
        raise ProgramMaskBuildError("control matching did not cover every target gene")
    matches = [components[(int(row), int(column))] for row, column in zip(rows, columns)]
    control_genes = [str(match["control_gene"]) for match in matches]
    control_values = [values[positions[gene]] for gene in control_genes]
    return control_genes, control_values, matches


def _serialize(values: list[Any]) -> str:
    return ";".join(str(value) for value in values)


def build(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    source_sha256 = _verify_hash(source_path, EXPECTED_SOURCE_SHA256, "PBMC3k source")
    target_sha256 = _verify_hash(target_path, EXPECTED_TARGET_SHA256, "PBMC68k target")

    source = ad.read_h5ad(source_path, backed="r")
    target = ad.read_h5ad(target_path)
    source_gene_names = np.asarray(
        list(map(str, source.raw.var_names if source.raw is not None else source.var_names)),
        dtype=object,
    )
    source_genes = set(source_gene_names)
    target_gene_names = np.asarray(
        list(map(str, target.raw.var_names if target.raw is not None else target.var_names)),
        dtype=object,
    )
    universe_mask = np.asarray(
        [gene in source_genes and HOUSEKEEPING.match(gene) is None for gene in target_gene_names],
        dtype=bool,
    )
    universe = target_gene_names[universe_mask].astype(str)
    if len(universe) != 750:
        raise ProgramMaskBuildError(
            f"frozen common non-housekeeping universe changed: expected 750, observed {len(universe)}"
        )

    source_universe_mask = np.asarray(
        [gene in set(universe) for gene in source_gene_names],
        dtype=bool,
    )
    source_annotations = source.obs["louvain"].astype(str)
    source_selected_mask = source_annotations.isin(["CD8 T cells", "NK cells"]).to_numpy()
    source_expression = source.raw.X if source.raw is not None else source.X
    source_expression = _dense(source_expression[source_selected_mask][:, source_universe_mask])
    source_universe = source_gene_names[source_universe_mask].astype(str)
    source_ranked = [_rank_cell(source_universe, source_expression[row]) for row in range(source_expression.shape[0])]
    source_labels = (source_annotations[source_selected_mask].to_numpy() == "CD8 T cells").astype(int)
    source_prevalence_gap: dict[str, float] = {}
    for gene in universe:
        cd8_prevalence = np.mean([gene in source_ranked[row][0] for row in np.flatnonzero(source_labels == 1)])
        nk_prevalence = np.mean([gene in source_ranked[row][0] for row in np.flatnonzero(source_labels == 0)])
        source_prevalence_gap[gene] = float(abs(cd8_prevalence - nk_prevalence))
    source_neutral_controls = {
        gene for gene, gap in source_prevalence_gap.items() if gap <= CONTROL_MAX_SOURCE_PREVALENCE_GAP
    }

    annotations = target.obs["bulk_labels"].astype(str)
    selected_mask = annotations.isin(LABEL_MAP).to_numpy()
    selected_obs = target.obs.loc[selected_mask].copy()
    selected_annotations = annotations[selected_mask].to_numpy()
    expression = target.raw.X if target.raw is not None else target.X
    expression = _dense(expression[selected_mask][:, universe_mask])
    ranked = [_rank_cell(universe, expression[row]) for row in range(len(selected_obs))]
    prevalence_counts = Counter(gene for genes, _ in ranked for gene in genes)
    prevalence = {gene: count / len(ranked) for gene, count in prevalence_counts.items()}

    rows: list[dict[str, Any]] = []
    zero_hit_by_class = Counter()
    match_records: list[dict[str, Any]] = []
    for row_index, (cell_id, annotation) in enumerate(zip(selected_obs.index.astype(str), selected_annotations)):
        label, program_name = LABEL_MAP[annotation]
        program = PROGRAMS[program_name]
        genes, values = ranked[row_index]
        hits = [gene for gene in genes if gene in program]
        if not hits:
            zero_hit_by_class[program_name] += 1
            continue
        target_genes = hits[: min(MAX_MASK, len(hits))]
        target_values = [values[genes.index(gene)] for gene in target_genes]
        control_genes, control_values, matches = _match_control_genes(
            genes,
            values,
            target_genes,
            prevalence,
            source_neutral_controls,
        )
        for match in matches:
            match_records.append(
                {
                    "entity_id": f"pbmc68k:{cell_id}",
                    "program": program_name,
                    **match,
                }
            )
        rows.append(
            {
                "entity_id": f"pbmc68k:{cell_id}",
                "cell_barcode": cell_id,
                "technical_group": cell_id.rsplit("-", 1)[-1],
                "reference_annotation": annotation,
                "label_a": label,
                "program": program_name,
                "program_hits": _serialize(hits),
                "program_hit_count": len(hits),
                "mask_k": len(target_genes),
                "target_genes": _serialize(target_genes),
                "target_values": _serialize(f"{value:.6g}" for value in target_values),
                "control_genes": _serialize(control_genes),
                "control_values": _serialize(f"{value:.6g}" for value in control_values),
                "original_sentence": " ".join(genes),
                "program_mask_sentence": _mask_sentence(genes, target_genes),
                "control_mask_sentence": _mask_sentence(genes, control_genes),
            }
        )

    expected_counts = {1: 88, 0: 31}
    observed_counts = Counter(int(row["label_a"]) for row in rows)
    if dict(observed_counts) != expected_counts:
        raise ProgramMaskBuildError(
            f"eligible class counts changed: expected {expected_counts}, observed {dict(observed_counts)}"
        )
    if len(rows) != 119:
        raise ProgramMaskBuildError(f"expected 119 eligible cells, observed {len(rows)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rank_distances = np.asarray(
        [float(match["rank_distance"]) for match in match_records],
        dtype=float,
    )
    expression_distances = np.asarray(
        [float(match["expression_distance"]) for match in match_records],
        dtype=float,
    )
    prevalence_distances = np.asarray(
        [float(match["prevalence_distance"]) for match in match_records],
        dtype=float,
    )
    manifest = {
        "analysis_id": "pbmc68k-cd8-nk-program-mask-v1",
        "freeze_date": "2026-07-27",
        "claim_scope": (
            "single_external_cohort_output_level_program_mask_intervention_"
            "not_multi_donor_generalization_or_hidden_state_causality"
        ),
        "source_pbmc3k": {
            "path": str(source_path),
            "sha256": source_sha256,
            "role": "frozen gene-universe and pre-existing marker-set source",
        },
        "target_pbmc68k": {
            "path": str(target_path),
            "sha256": target_sha256,
            "scanpy_documentation": (
                "https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc68k_reduced.html"
            ),
            "primary_source_doi": "10.1038/ncomms14049",
            "cohort": "10x Fresh 68k PBMC Donor A; reduced annotated Scanpy artifact",
        },
        "program_provenance": {
            "parent_file": "signal/single_cell/build_cd8t_nk.py",
            "parent_git_commit": "20f94201c60878b83dbae182b8447248666ede29",
            "parent_set": sorted(FAMOUS_PARENT),
            "cd8_identity": sorted(CD8_IDENTITY_PROGRAM),
            "nk_identity": sorted(NK_IDENTITY_PROGRAM),
            "shared_excluded_from_target_and_control": sorted(SHARED_EXCLUDED),
        },
        "construction": {
            "top_k": TOP_K,
            "max_masked_program_genes": MAX_MASK,
            "mask_token": MASK_TOKEN,
            "common_non_housekeeping_gene_count": len(universe),
            "common_gene_universe_sha256": hashlib.sha256(("\n".join(sorted(universe)) + "\n").encode()).hexdigest(),
            "target_annotation_map": {
                annotation: {"label_a": label, "program": program} for annotation, (label, program) in LABEL_MAP.items()
            },
            "all_target_cd8_nk_cells": len(ranked),
            "eligible_cells": len(rows),
            "eligible_class_counts": {
                "CD8": observed_counts[1],
                "NK": observed_counts[0],
            },
            "zero_hit_cells": dict(zero_hit_by_class),
            "control_matching": {
                "method": (
                    "within-cell linear-sum assignment on rank, expression, "
                    "global top-50 prevalence, and token length; all parent-program "
                    "genes and source-PBMC3k class-associated genes excluded"
                ),
                "source_neutrality_rule": (
                    f"absolute CD8-vs-NK top-50 prevalence gap in PBMC3k <= {CONTROL_MAX_SOURCE_PREVALENCE_GAP:.2f}"
                ),
                "source_neutral_control_gene_count": len(source_neutral_controls),
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
            "csv_path": str(output_path.relative_to(ROOT)),
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
    parser.add_argument("--source-h5ad", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-h5ad", type=Path, default=DEFAULT_TARGET)
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
                "csv": manifest["artifacts"]["csv_path"],
                "csv_sha256": manifest["artifacts"]["csv_sha256"],
                "eligible_cells": manifest["construction"]["eligible_cells"],
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

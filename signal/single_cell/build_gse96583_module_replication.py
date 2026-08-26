"""Build a donor-aware GSE96583 context-paired module masking comparison.

The target cohort is the unstimulated batch-2 arm from Kang et al. GSE96583.
Only demuxlet singlets annotated as CD8 T cells or NK cells enter the task
frame. Within that frame, expression—not the target label—determines whether a
cell has top-50 support for each frozen module and which token is masked.
Deposited labels are retained only for task framing, descriptive composition,
and secondary analysis; they do not select an expression context, sampled
cell, module, target token, or mask within the frame.

The full three-module same-cell design is infeasible across donors because
T-lineage and NK-receptor tokens rarely co-occur. The executable design
therefore uses two expression-defined paired contexts: one requires both
TCR/CD8 and cytotoxic-effector tokens, and one requires both
NK-receptor/identity and cytotoxic-effector tokens. Within each donor, cells
are selected by a frozen hash without using the deposited cell-type label.
Each cell receives both context masks, separately matched source-neutral
deletions, and an unmasked input. This is a text-input intervention, not a
biological perturbation or an isolated-gene experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import build_pbmc68k_module_factorial as pbmc68k
import build_pbmc68k_program_mask as prior
import numpy as np
from scipy import sparse
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "gse96583"
DEFAULT_SOURCE = prior.DEFAULT_SOURCE
DEFAULT_MATRIX = RAW_ROOT / "GSM2560248_2.1.mtx.gz"
DEFAULT_BARCODES = RAW_ROOT / "GSM2560248_barcodes.tsv.gz"
DEFAULT_GENES = RAW_ROOT / "GSE96583_batch2.genes.tsv.gz"
DEFAULT_METADATA = RAW_ROOT / "GSE96583_batch2.total.tsne.df.tsv.gz"
DEFAULT_ARCHIVE = RAW_ROOT / "GSE96583_RAW.tar"
DEFAULT_OUT = Path(__file__).with_name("gse96583_cd8_nk_module_replication.csv")
DEFAULT_MANIFEST = Path(__file__).with_name("gse96583_cd8_nk_module_replication.manifest.json")

ANALYSIS_ID = "gse96583-cd8-nk-context-module-donor-replication-v1"
T_CYTOTOXIC_SAMPLE_PER_DONOR = 4
RECEPTOR_CYTOTOXIC_SAMPLE_PER_DONOR = 3
NORMALIZATION_TARGET_SUM = 10_000.0
EXPECTED_DONORS = ("101", "107", "1015", "1016", "1039", "1244", "1256", "1488")
EXPECTED_SOURCE_SHA256 = prior.EXPECTED_SOURCE_SHA256
EXPECTED_INPUT_SHA256 = {
    "archive": "e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80",
    "matrix": "32add28a0b3397d9ef3f220b7a6a55e98e60fe7b66fe48d0986d634df8ca0013",
    "barcodes": "d58d8d55cbe4a12757207784b3bc9227bf200c9100ca15131176e9f8159c955e",
    "genes": "93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca",
    "metadata": "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09",
}
TRANSITIVELY_LOCKED_NOT_INGESTED = {
    "GSM2560249_2.2.mtx.gz": {
        "bytes": 29_050_932,
        "sha256": "8aecc98a7ac4957bbc2570f87ebe8ce97332a5bcdbf557d40bd5aabfd287bdc5",
    },
    "GSM2560249_barcodes.tsv.gz": {
        "bytes": 52_366,
        "sha256": "9bb38e080dfae81036fbcd9902c6c6254a4466aa85f70705479be3b2d6679d55",
    },
}
EXPECTED_MATRIX_SHAPE = (35_635, 14_619)
EXPECTED_MATRIX_NNZ = 8_732_747
EXPECTED_METADATA_ROWS = 29_065
EXPECTED_CONTROL_ROWS = 14_619

MODULES = {
    "T_TCR_CD8": pbmc68k.T_TCR_CD8_MODULE,
    "NK_receptor_identity": pbmc68k.NK_RECEPTOR_MODULE,
    "cytotoxic_effector": pbmc68k.CYTOTOXIC_EFFECTOR_MODULE,
}
CONTEXT_MODULES = {
    "T_plus_cytotoxic": ("T_TCR_CD8", "cytotoxic_effector"),
    "NK_receptor_plus_cytotoxic": (
        "NK_receptor_identity",
        "cytotoxic_effector",
    ),
}
CELL_TYPE_TO_LABEL = {
    "CD8 T cells": 1,
    "NK cells": 0,
}
METADATA_COLUMNS = (
    "barcode",
    "tsne1",
    "tsne2",
    "donor_id",
    "stimulus",
    "cluster",
    "cell_type",
    "multiplet_status",
)


class GSE96583BuildError(ValueError):
    """Raised when an authenticated source or derived build violates the contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise GSE96583BuildError(f"missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise GSE96583BuildError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")
    return observed


def _verify_transitively_locked_archive_members(
    archive_path: Path,
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name, expected in TRANSITIVELY_LOCKED_NOT_INGESTED.items():
            member = members.get(name)
            if member is None or member.size != expected["bytes"]:
                raise GSE96583BuildError(
                    f"transitively locked archive member missing or changed: {name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise GSE96583BuildError(
                    f"cannot read transitively locked archive member: {name}"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            observed = digest.hexdigest()
            if observed != expected["sha256"]:
                raise GSE96583BuildError(
                    f"transitively locked archive member SHA-256 changed: {name}"
                )
            verified[name] = {
                "bytes": member.size,
                "sha256": observed,
                "status": "transitively_locked_not_ingested",
            }
    return verified


def _read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [line.rstrip("\r\n") for line in handle]


def _read_genes(path: Path) -> tuple[list[str], list[str]]:
    rows = [line.split("\t") for line in _read_lines(path)]
    if len(rows) != EXPECTED_MATRIX_SHAPE[0] or any(len(row) != 2 for row in rows):
        raise GSE96583BuildError("gene map dimensions changed")
    ensembl_ids = [row[0] for row in rows]
    symbols = [row[1] for row in rows]
    if len(set(ensembl_ids)) != len(ensembl_ids) or any(not symbol for symbol in symbols):
        raise GSE96583BuildError("gene map has duplicate Ensembl IDs or empty symbols")
    return ensembl_ids, symbols


def _read_metadata(path: Path) -> list[dict[str, str]]:
    lines = _read_lines(path)
    if not lines:
        raise GSE96583BuildError("metadata is empty")
    header = lines[0].split("\t")
    # The deposited file has seven header tokens but eight fields per data row;
    # its first field is an unlabeled barcode. Bind the positional schema here.
    if header != ["tsne1", "tsne2", "ind", "stim", "cluster", "cell", "multiplets"]:
        raise GSE96583BuildError("deposited metadata header changed")
    parsed = []
    for line_number, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) != len(METADATA_COLUMNS):
            raise GSE96583BuildError(f"metadata row {line_number} has {len(values)} fields; expected 8")
        parsed.append(dict(zip(METADATA_COLUMNS, values)))
    if len(parsed) != EXPECTED_METADATA_ROWS:
        raise GSE96583BuildError(
            f"metadata row count changed: expected {EXPECTED_METADATA_ROWS}, observed {len(parsed)}"
        )
    return parsed


def _load_authenticated_target(
    matrix_path: Path,
    barcode_path: Path,
    gene_path: Path,
    metadata_path: Path,
    archive_path: Path,
) -> tuple[sparse.csc_matrix, list[str], list[str], list[dict[str, str]], dict[str, str]]:
    observed_hashes = {
        "archive": _verify_hash(archive_path, EXPECTED_INPUT_SHA256["archive"], "GEO archive"),
        "matrix": _verify_hash(matrix_path, EXPECTED_INPUT_SHA256["matrix"], "control matrix"),
        "barcodes": _verify_hash(
            barcode_path,
            EXPECTED_INPUT_SHA256["barcodes"],
            "control barcodes",
        ),
        "genes": _verify_hash(gene_path, EXPECTED_INPUT_SHA256["genes"], "batch-2 gene map"),
        "metadata": _verify_hash(
            metadata_path,
            EXPECTED_INPUT_SHA256["metadata"],
            "batch-2 metadata",
        ),
    }
    observed_hashes["archive_members_transitively_locked_not_ingested"] = (
        _verify_transitively_locked_archive_members(archive_path)
    )
    _, gene_symbols = _read_genes(gene_path)
    barcodes = _read_lines(barcode_path)
    if len(barcodes) != EXPECTED_CONTROL_ROWS or len(set(barcodes)) != len(barcodes):
        raise GSE96583BuildError("control barcode count or uniqueness changed")
    metadata = _read_metadata(metadata_path)
    control_metadata = [row for row in metadata if row["stimulus"] == "ctrl"]
    if len(control_metadata) != EXPECTED_CONTROL_ROWS:
        raise GSE96583BuildError("control metadata row count changed")
    if [row["barcode"] for row in control_metadata] != barcodes:
        raise GSE96583BuildError("control barcode order does not match the matrix")

    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle)
    if not sparse.issparse(matrix):
        raise GSE96583BuildError("deposited count matrix is unexpectedly dense")
    matrix = matrix.tocsc()
    if matrix.shape != EXPECTED_MATRIX_SHAPE or matrix.nnz != EXPECTED_MATRIX_NNZ:
        raise GSE96583BuildError(f"matrix dimensions changed: shape={matrix.shape}, nnz={matrix.nnz}")
    if np.any(matrix.data < 0) or np.any(matrix.data != np.floor(matrix.data)):
        raise GSE96583BuildError("matrix contains negative or non-integer counts")
    return matrix, gene_symbols, barcodes, control_metadata, observed_hashes


def _sampling_hash(donor_id: str, barcode: str) -> str:
    """Return the frozen sampling key without reference-label information."""
    payload = f"{ANALYSIS_ID}|{donor_id}|{barcode}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _rank_target_cells(
    matrix: sparse.csc_matrix,
    gene_symbols: list[str],
    metadata: list[dict[str, str]],
    source_gene_names: set[str],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    selected_indices = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row["multiplet_status"] == "singlet" and row["cell_type"] in CELL_TYPE_TO_LABEL
        ],
        dtype=int,
    )
    selected_metadata = [metadata[index] for index in selected_indices]
    if len(selected_metadata) != 1_998:
        raise GSE96583BuildError(
            f"framed control-singlet count changed: expected 1998, observed {len(selected_metadata)}"
        )
    donor_counts = Counter(row["donor_id"] for row in selected_metadata)
    if tuple(sorted(donor_counts, key=int)) != EXPECTED_DONORS:
        raise GSE96583BuildError(f"donor set changed: {sorted(donor_counts, key=int)}")

    retained_rows = [
        index
        for index, symbol in enumerate(gene_symbols)
        if symbol in source_gene_names and prior.HOUSEKEEPING.match(symbol) is None
    ]
    universe = np.asarray(
        sorted({gene_symbols[index] for index in retained_rows}),
        dtype=object,
    )
    universe_index = {gene: index for index, gene in enumerate(universe)}
    aggregation = sparse.csr_matrix(
        (
            np.ones(len(retained_rows), dtype=float),
            (
                [universe_index[gene_symbols[index]] for index in retained_rows],
                retained_rows,
            ),
        ),
        shape=(len(universe), matrix.shape[0]),
    )
    selected_counts = matrix[:, selected_indices]
    library_sizes = np.asarray(selected_counts.sum(axis=0)).ravel()
    if np.any(library_sizes <= 0):
        raise GSE96583BuildError("framed cell has zero total UMI count")
    expression = (aggregation @ selected_counts).tocsc().astype(float)
    expression = expression @ sparse.diags(
        NORMALIZATION_TARGET_SUM / library_sizes,
        format="csc",
    )
    expression.data = np.log1p(expression.data)

    ranked: list[dict[str, Any]] = []
    qc_excluded = Counter()
    for column, row in enumerate(selected_metadata):
        values = np.asarray(expression.getcol(column).toarray()).ravel()
        expressed_genes = int((values > 0).sum())
        if expressed_genes < prior.TOP_K:
            qc_excluded[(row["donor_id"], row["cell_type"])] += 1
            continue
        genes, ranked_values = prior._rank_cell(universe, values)
        hits = {module: [gene for gene in genes if gene in module_genes] for module, module_genes in MODULES.items()}
        ranked.append(
            {
                **row,
                "label_a": CELL_TYPE_TO_LABEL[row["cell_type"]],
                "genes": genes,
                "values": ranked_values,
                "hits": hits,
                "library_size": float(library_sizes[column]),
                "sampling_hash": _sampling_hash(
                    row["donor_id"],
                    row["barcode"],
                ),
            }
        )

    support = {
        "framed_cells_before_renderability_qc": len(selected_metadata),
        "renderable_cells": len(ranked),
        "renderability_qc": ("at least 50 positive genes in the frozen common non-housekeeping universe"),
        "renderability_qc_excluded_by_donor_class": [
            {
                "donor_id": donor,
                "cell_type": cell_type,
                "n": qc_excluded[(donor, cell_type)],
            }
            for donor in EXPECTED_DONORS
            for cell_type in CELL_TYPE_TO_LABEL
        ],
        "universe_genes": len(universe),
        "universe_sha256": hashlib.sha256(("\n".join(map(str, universe)) + "\n").encode()).hexdigest(),
        "duplicate_symbol_rows_aggregated": len(retained_rows) - len(universe),
        "by_module_donor_class": {},
        "by_support_set_donor_class": {},
    }
    for module in MODULES:
        counts = Counter((row["donor_id"], row["cell_type"]) for row in ranked if row["hits"][module])
        support["by_module_donor_class"][module] = [
            {
                "donor_id": donor,
                "cell_type": cell_type,
                "n": counts[(donor, cell_type)],
            }
            for donor in EXPECTED_DONORS
            for cell_type in CELL_TYPE_TO_LABEL
        ]
    support_sets = {
        "T_plus_cytotoxic": ("T_TCR_CD8", "cytotoxic_effector"),
        "T_plus_NK_receptor": ("T_TCR_CD8", "NK_receptor_identity"),
        "NK_receptor_plus_cytotoxic": (
            "NK_receptor_identity",
            "cytotoxic_effector",
        ),
        "all_three": tuple(MODULES),
    }
    for name, required_modules in support_sets.items():
        counts = Counter(
            (row["donor_id"], row["cell_type"])
            for row in ranked
            if all(row["hits"][module] for module in required_modules)
        )
        support["by_support_set_donor_class"][name] = [
            {
                "donor_id": donor,
                "cell_type": cell_type,
                "n": counts[(donor, cell_type)],
            }
            for donor in EXPECTED_DONORS
            for cell_type in CELL_TYPE_TO_LABEL
        ]
    return ranked, universe, support


def _serialize(values: list[Any]) -> str:
    return ";".join(str(value) for value in values)


def _base_row(cell: dict[str, Any]) -> dict[str, Any]:
    sentence = " ".join(cell["genes"])
    return {
        "row_type": "base",
        "entity_id": (f"gse96583:ctrl:{cell['donor_id']}:{cell['barcode']}"),
        "cell_barcode": cell["barcode"],
        "donor_id": cell["donor_id"],
        "technical_group": cell["donor_id"],
        "stimulus": cell["stimulus"],
        "multiplet_status": cell["multiplet_status"],
        "reference_annotation": cell["cell_type"],
        "label_a": cell["label_a"],
        "sampling_context": cell["sampling_context"],
        "sampling_stratum": f"{cell['donor_id']}:{cell['sampling_context']}",
        "sampling_hash": cell["sampling_hash"],
        "library_size": f"{cell['library_size']:.12g}",
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


def _select_expression_contexts(
    ranked: list[dict[str, Any]],
    t_cytotoxic_per_donor: int,
    receptor_cytotoxic_per_donor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if t_cytotoxic_per_donor <= 0 or receptor_cytotoxic_per_donor <= 0:
        raise GSE96583BuildError("per-donor context sample sizes must be positive")
    requests = (
        ("T_plus_cytotoxic", t_cytotoxic_per_donor),
        ("NK_receptor_plus_cytotoxic", receptor_cytotoxic_per_donor),
    )
    selected: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    used_barcodes: set[str] = set()
    for donor in EXPECTED_DONORS:
        for context, requested in requests:
            required_modules = CONTEXT_MODULES[context]
            raw_candidates = sorted(
                (
                    row
                    for row in ranked
                    if row["donor_id"] == donor
                    and all(row["hits"][module] for module in required_modules)
                ),
                key=lambda row: (row["sampling_hash"], row["barcode"]),
            )
            candidates = [
                row for row in raw_candidates if row["barcode"] not in used_barcodes
            ]
            if len(candidates) < requested:
                raise GSE96583BuildError(
                    f"insufficient expression-context support for {donor}/{context}: "
                    f"need {requested}, observed {len(candidates)}"
                )
            chosen = candidates[:requested]
            selected.extend({**row, "sampling_context": context} for row in chosen)
            used_barcodes.update(row["barcode"] for row in chosen)
            selection_audit.append(
                {
                    "donor_id": donor,
                    "sampling_context": context,
                    "required_modules": list(required_modules),
                    "eligible_before_prior_context_exclusion": len(raw_candidates),
                    "eligible_after_prior_context_exclusion": len(candidates),
                    "requested": requested,
                    "selected": len(chosen),
                }
            )
    return selected, selection_audit


def build(
    source_path: Path,
    matrix_path: Path,
    barcode_path: Path,
    gene_path: Path,
    metadata_path: Path,
    archive_path: Path,
    output_path: Path,
    manifest_path: Path,
    t_cytotoxic_per_donor: int = T_CYTOTOXIC_SAMPLE_PER_DONOR,
    receptor_cytotoxic_per_donor: int = RECEPTOR_CYTOTOXIC_SAMPLE_PER_DONOR,
    audit_only: bool = False,
) -> dict[str, Any]:
    source_sha256 = _verify_hash(
        source_path,
        EXPECTED_SOURCE_SHA256,
        "PBMC3k source",
    )
    matrix, gene_symbols, _, metadata, observed_hashes = _load_authenticated_target(
        matrix_path,
        barcode_path,
        gene_path,
        metadata_path,
        archive_path,
    )
    source = ad.read_h5ad(source_path, backed="r")
    source_gene_names = set(
        map(
            str,
            source.raw.var_names if source.raw is not None else source.var_names,
        )
    )
    ranked, universe, support = _rank_target_cells(
        matrix,
        gene_symbols,
        metadata,
        source_gene_names,
    )
    selected, selection_audit = _select_expression_contexts(
        ranked,
        t_cytotoxic_per_donor,
        receptor_cytotoxic_per_donor,
    )
    context_eligible_counts = {
        context: [
            record["eligible_after_prior_context_exclusion"]
            for record in selection_audit
            if record["sampling_context"] == context
        ]
        for context in CONTEXT_MODULES
    }
    audit = {
        "analysis_id": ANALYSIS_ID,
        "requested_cells_per_donor": {
            "T_plus_cytotoxic": t_cytotoxic_per_donor,
            "NK_receptor_plus_cytotoxic": receptor_cytotoxic_per_donor,
        },
        "minimum_context_support_after_prior_context_exclusion": {
            context: min(counts) for context, counts in context_eligible_counts.items()
        },
        "requested_context_sample_feasible": True,
        "context_selection_order": list(CONTEXT_MODULES),
        "context_selection_audit": selection_audit,
        "support": support,
        "source_hashes": {
            "pbmc3k": source_sha256,
            **observed_hashes,
        },
        "sampled_cells": len(selected),
        "sampled_context_counts": dict(
            sorted(Counter(row["sampling_context"] for row in selected).items())
        ),
        "sampled_class_counts_descriptive_only": dict(
            sorted(Counter(row["cell_type"] for row in selected).items())
        ),
        "sampled_context_by_class_descriptive_only": [
            {
                "sampling_context": context,
                "reference_annotation": cell_type,
                "n": sum(
                    row["sampling_context"] == context and row["cell_type"] == cell_type
                    for row in selected
                ),
            }
            for context in CONTEXT_MODULES
            for cell_type in CELL_TYPE_TO_LABEL
        ],
        "sampled_donor_counts": dict(
            sorted(
                Counter(row["donor_id"] for row in selected).items(),
                key=lambda item: int(item[0]),
            )
        ),
    }
    if audit_only:
        return audit

    source_neutral_controls, source_prevalence_gap = pbmc68k._source_neutral_controls(
        source,
        universe,
    )
    prevalence_counts = Counter(gene for cell in ranked for gene in cell["genes"])
    prevalence = {gene: count / len(ranked) for gene, count in prevalence_counts.items()}

    rows: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    for cell in selected:
        rows.append(_base_row(cell))
        for module in CONTEXT_MODULES[cell["sampling_context"]]:
            hits = cell["hits"][module]
            target_genes = hits[:1]
            control_genes, control_values, matches = prior._match_control_genes(
                cell["genes"],
                cell["values"],
                target_genes,
                prevalence,
                source_neutral_controls,
            )
            for match in matches:
                match_records.append(
                    {
                        "entity_id": (f"gse96583:ctrl:{cell['donor_id']}:{cell['barcode']}"),
                        "module": module,
                        **match,
                    }
                )
            target_values = [cell["values"][cell["genes"].index(gene)] for gene in target_genes]
            base = _base_row(cell)
            rows.append(
                {
                    **base,
                    "row_type": "intervention",
                    "module": module,
                    "dose_k": 1,
                    "module_hits": _serialize(hits),
                    "module_hit_count": len(hits),
                    "target_genes": _serialize(target_genes),
                    "target_values": _serialize([f"{value:.12g}" for value in target_values]),
                    "control_genes": _serialize(control_genes),
                    "control_values": _serialize([f"{value:.12g}" for value in control_values]),
                    "control_rank_distance": matches[0]["rank_distance"],
                    "control_expression_distance": (f"{float(matches[0]['expression_distance']):.12g}"),
                    "control_prevalence_distance": (f"{float(matches[0]['prevalence_distance']):.12g}"),
                    "control_token_length_distance": (matches[0]["token_length_distance"]),
                    "control_total_cost": (f"{float(matches[0]['total_cost']):.12g}"),
                    "module_mask_sentence": prior._mask_sentence(
                        cell["genes"],
                        target_genes,
                    ),
                    "control_mask_sentence": prior._mask_sentence(
                        cell["genes"],
                        control_genes,
                    ),
                }
            )

    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [row for row in rows if row["row_type"] == "intervention"]
    expected_cells = len(EXPECTED_DONORS) * (
        t_cytotoxic_per_donor + receptor_cytotoxic_per_donor
    )
    if len(base_rows) != expected_cells or len(intervention_rows) != expected_cells * 2:
        raise GSE96583BuildError("context sample or intervention row count changed")
    unique_inputs = {(row["entity_id"], row["original_sentence"]) for row in base_rows}
    unique_inputs.update((row["entity_id"], row["module_mask_sentence"]) for row in intervention_rows)
    unique_inputs.update((row["entity_id"], row["control_mask_sentence"]) for row in intervention_rows)
    logical_observations = len(base_rows) * 4 + len(intervention_rows) * 2 * 4
    unique_calls = len(unique_inputs) * 4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rank_distances = np.asarray([float(record["rank_distance"]) for record in match_records])
    expression_distances = np.asarray([float(record["expression_distance"]) for record in match_records])
    prevalence_distances = np.asarray([float(record["prevalence_distance"]) for record in match_records])
    module_targets = {
        module: [
            {"gene": gene, "cells": count}
            for gene, count in sorted(
                Counter(row["target_genes"] for row in intervention_rows if row["module"] == module).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        for module in MODULES
    }
    shared_neutral_counts = Counter(
        len({row["control_mask_sentence"] for row in intervention_rows if row["entity_id"] == entity_id})
        for entity_id in {row["entity_id"] for row in base_rows}
    )
    manifest = {
        **audit,
        "freeze_date": "2026-07-27",
        "status": "built_not_executed",
        "claim_scope": (
            "multi_donor_sle_control_mixed_marker_text_input_module_"
            "sensitivity_not_gene_pathway_hidden_state_or_physical_law"
        ),
        "source": {
            "geo_accession": "GSE96583",
            "geo_url": ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583"),
            "paper_doi": "10.1038/nbt.4042",
            "arm": "batch 2 control; 6-hour culture without IFN-beta",
            "population": "eight SLE donors",
            "inclusion": (
                "demuxlet singlets in the deposited CD8 T-cell or NK-cell "
                "task frame with at least 50 renderable common-universe "
                "genes; expression support, not the deposited label, assigns "
                "the T-plus-cytotoxic or NK-receptor-plus-cytotoxic context"
            ),
            "deposited_metadata_header_note": (
                "seven header tokens for eight row fields; first positional field is the unlabeled barcode"
            ),
            "annotation_boundary": (
                "the paper derived deposited cell labels from the same "
                "expression matrix by reference-marker prediction; labels "
                "are therefore not orthogonal biological ground truth and "
                "are used only to define the upstream task frame and for "
                "descriptive or secondary summaries"
            ),
            "transitively_locked_not_ingested": observed_hashes[
                "archive_members_transitively_locked_not_ingested"
            ],
            "stimulated_secondary_boundary": (
                "the stimulated files are locked as archive members but not "
                "ingested; activating that secondary requires direct "
                "extraction and validation, and it remains the same eight "
                "donors rather than eight additional replicates"
            ),
            "redistribution": (
                "no_NCBI_restriction; submitter_rights_caveat; distribute "
                "derived tables and the fetch/hash manifest without "
                "relicensing upstream GEO bytes"
            ),
        },
        "preprocessing": {
            "counts": "deposited integer UMI matrix",
            "duplicate_gene_symbols": ("sum Ensembl rows sharing a symbol before normalization"),
            "normalization": ("per-cell total-count normalization to 10000 followed by log1p"),
            "universe": (
                "intersection with frozen PBMC3k symbols, excluding RP[LS]/MRP[LS]/MT-/MALAT1 housekeeping patterns"
            ),
            "ranking": (
                "top 50 positive values; descending log-normalized expression with gene-symbol lexical tie break"
            ),
        },
        "modules": {module: sorted(genes) for module, genes in MODULES.items()},
        "expression_contexts": {
            context: list(modules) for context, modules in CONTEXT_MODULES.items()
        },
        "sampling": {
            "policy": (
                "for each donor, assign T-plus-cytotoxic first and then "
                "NK-receptor-plus-cytotoxic; within each expression-support "
                "set take the lowest SHA-256 hashes, excluding cells already "
                "assigned to the prior context"
            ),
            "label_boundary": (
                "reference labels define only the upstream CD8/NK task "
                "frame; labels do not enter the sampling hash or select "
                "context, cell, module pair, target token, or mask"
            ),
            "sampling_hash_payload": (f"{ANALYSIS_ID}|<donor_id>|<barcode>"),
        },
        "control_matching": {
            "method": (
                "separate within-cell match for each module on expression "
                "rank, log-normalized expression, framed-cell top-50 "
                "prevalence, and token length"
            ),
            "source_neutrality_rule": (
                "PBMC3k absolute CD8-vs-NK top-50 prevalence gap <= 0.10; all 30 frozen parent markers excluded"
            ),
            "source_neutral_control_gene_count": len(source_neutral_controls),
            "source_prevalence_gap_sha256": hashlib.sha256(
                (
                    "\n".join(f"{gene}\t{source_prevalence_gap[gene]:.12g}" for gene in sorted(source_prevalence_gap))
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
            "distinct_neutral_inputs_per_cell": {
                str(count): cells for count, cells in sorted(shared_neutral_counts.items())
            },
        },
        "execution_budget": {
            "prompt_forms_per_input": 4,
            "logical_condition_form_observations": logical_observations,
            "unique_api_calls_after_input_deduplication": unique_calls,
        },
        "selected_module_targets": module_targets,
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
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--barcodes", type=Path, default=DEFAULT_BARCODES)
    parser.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--t-cytotoxic-per-donor",
        type=int,
        default=T_CYTOTOXIC_SAMPLE_PER_DONOR,
    )
    parser.add_argument(
        "--receptor-cytotoxic-per-donor",
        type=int,
        default=RECEPTOR_CYTOTOXIC_SAMPLE_PER_DONOR,
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = build(
        args.source_h5ad.resolve(),
        args.matrix.resolve(),
        args.barcodes.resolve(),
        args.genes.resolve(),
        args.metadata.resolve(),
        args.archive.resolve(),
        args.out.resolve(),
        args.manifest.resolve(),
        args.t_cytotoxic_per_donor,
        args.receptor_cytotoxic_per_donor,
        args.audit_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

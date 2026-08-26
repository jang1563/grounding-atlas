"""Build a held-out multi-donor GNLY/NKG7/CCL5 masking factorial.

The authenticated GSE96583 control-singlet CD8/NK frame and its frozen top-50
representation come from ``build_gse96583_module_replication.py``.  Every cell
used by the prior 56-cell context experiment is excluded before eligibility is
evaluated.  Among the remaining cells, eligibility requires GNLY, NKG7, and
CCL5 all to occur in the top 50.  One cell per donor is selected by a SHA-256
key salted only by the immutable pre-hypothesis parent CSV digest, followed by
the donor ID and barcode.  The investigator-chosen analysis ID does not enter
selection.

For each selected cell, the three sentinels are jointly assigned to three
distinct PBMC3k reference-prevalence-balanced controls with the frozen
linear-sum cost.  This operational balance rule does not imply semantic
neutrality.  The output contains an unmasked base row plus all seven nonempty
sentinel subsets.  Every subset uses the corresponding subset of the single
joint control assignment, so target and control masks are nested and directly
comparable.  This builder makes no model calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import build_gse96583_module_replication as parent
import build_pbmc68k_module_factorial as source_control
import build_pbmc68k_program_mask as matching
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "gse96583-sentinel-factorial-holdout-v1"
FREEZE_DATE = "2026-07-30"
SENTINELS = ("GNLY", "NKG7", "CCL5")
SUBSETS = tuple(
    subset
    for size in range(1, len(SENTINELS) + 1)
    for subset in itertools.combinations(SENTINELS, size)
)
PROMPT_FORMS_PER_INPUT = 4

DEFAULT_SOURCE = parent.DEFAULT_SOURCE
DEFAULT_MATRIX = parent.DEFAULT_MATRIX
DEFAULT_BARCODES = parent.DEFAULT_BARCODES
DEFAULT_GENES = parent.DEFAULT_GENES
DEFAULT_METADATA = parent.DEFAULT_METADATA
DEFAULT_ARCHIVE = parent.DEFAULT_ARCHIVE
DEFAULT_PARENT_BUILDER = Path(__file__).with_name(
    "build_gse96583_module_replication.py"
)
DEFAULT_PARENT_CSV = Path(__file__).with_name(
    "gse96583_cd8_nk_module_replication.csv"
)
DEFAULT_PARENT_MANIFEST = Path(__file__).with_name(
    "gse96583_cd8_nk_module_replication.manifest.json"
)
DEFAULT_OUT = Path(__file__).with_name("gse96583_sentinel_factorial.csv")
DEFAULT_MANIFEST = Path(__file__).with_name(
    "gse96583_sentinel_factorial.manifest.json"
)

EXPECTED_PARENT_SHA256 = {
    "builder": "b089e91231042db7c8a7e17fd6dbe1e32169bf1f05361cd6e5e8041c395f00bf",
    "csv": "f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321",
    "manifest": "3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f",
}
EXPECTED_PARENT_BASE_CELLS = 56
EXPECTED_RENDERABLE_FRAME_CELLS = 1_997
EXPECTED_ALL_TRIPLE_SUPPORT = {
    "101": 39,
    "107": 14,
    "1015": 135,
    "1016": 194,
    "1039": 8,
    "1244": 58,
    "1256": 161,
    "1488": 69,
}
EXPECTED_UNUSED_TRIPLE_SUPPORT = {
    "101": 37,
    "107": 11,
    "1015": 132,
    "1016": 193,
    "1039": 5,
    "1244": 55,
    "1256": 157,
    "1488": 66,
}


class SentinelFactorialBuildError(ValueError):
    """Raised when a frozen input or derived factorial violates its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise SentinelFactorialBuildError(f"missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise SentinelFactorialBuildError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _sampling_hash(donor_id: str, barcode: str) -> str:
    """Return the immutable-parent-salted, label-independent sampling key."""
    payload = f"{EXPECTED_PARENT_SHA256['csv']}|{donor_id}|{barcode}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _serialize(values: list[Any] | tuple[Any, ...]) -> str:
    return ";".join(str(value) for value in values)


def _read_parent_exclusions(
    builder_path: Path,
    csv_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, str]]:
    observed = {
        "builder": _verify_hash(
            builder_path,
            EXPECTED_PARENT_SHA256["builder"],
            "parent builder",
        ),
        "csv": _verify_hash(
            csv_path,
            EXPECTED_PARENT_SHA256["csv"],
            "parent CSV",
        ),
        "manifest": _verify_hash(
            manifest_path,
            EXPECTED_PARENT_SHA256["manifest"],
            "parent manifest",
        ),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != parent.ANALYSIS_ID:
        raise SentinelFactorialBuildError("parent manifest analysis ID changed")
    if manifest.get("status") != "built_not_executed":
        raise SentinelFactorialBuildError("parent manifest status changed")
    if manifest.get("artifacts", {}).get("builder_sha256") != observed["builder"]:
        raise SentinelFactorialBuildError("parent manifest does not bind its builder")
    if manifest.get("artifacts", {}).get("csv_sha256") != observed["csv"]:
        raise SentinelFactorialBuildError("parent manifest does not bind its CSV")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    base_rows = [row for row in rows if row["row_type"] == "base"]
    if len(base_rows) != EXPECTED_PARENT_BASE_CELLS:
        raise SentinelFactorialBuildError(
            "parent exclusion population changed: "
            f"expected {EXPECTED_PARENT_BASE_CELLS}, observed {len(base_rows)}"
        )
    if len({row["cell_barcode"] for row in base_rows}) != len(base_rows):
        raise SentinelFactorialBuildError("parent base-cell barcodes are not unique")
    donor_counts = Counter(row["donor_id"] for row in base_rows)
    if set(donor_counts) != set(parent.EXPECTED_DONORS) or any(
        donor_counts[donor] != 7 for donor in parent.EXPECTED_DONORS
    ):
        raise SentinelFactorialBuildError(
            f"parent donor exclusions changed: {dict(donor_counts)}"
        )
    return base_rows, manifest, observed


def _support_record(
    donor_id: str,
    all_eligible: list[dict[str, Any]],
    unused_eligible: list[dict[str, Any]],
    excluded_barcodes: set[str],
) -> dict[str, Any]:
    all_donor = sorted(
        (cell for cell in all_eligible if cell["donor_id"] == donor_id),
        key=lambda cell: (cell["barcode"], cell["cell_type"]),
    )
    unused_donor = sorted(
        (cell for cell in unused_eligible if cell["donor_id"] == donor_id),
        key=lambda cell: (_sampling_hash(donor_id, cell["barcode"]), cell["barcode"]),
    )
    excluded_eligible = [
        cell for cell in all_donor if cell["barcode"] in excluded_barcodes
    ]
    return {
        "donor_id": donor_id,
        "eligible_in_full_ranked_frame": len(all_donor),
        "eligible_excluded_by_parent_holdout": len(excluded_eligible),
        "eligible_after_parent_holdout": len(unused_donor),
        "eligible_excluded_barcodes": sorted(
            cell["barcode"] for cell in excluded_eligible
        ),
        "selected_barcode": unused_donor[0]["barcode"],
        "selected_sampling_hash": _sampling_hash(
            donor_id,
            unused_donor[0]["barcode"],
        ),
    }


def _base_row(cell: dict[str, Any]) -> dict[str, Any]:
    sentence = " ".join(cell["genes"])
    sentinel_values = [
        cell["values"][cell["genes"].index(gene)] for gene in SENTINELS
    ]
    return {
        "row_type": "base",
        "entity_id": f"gse96583:ctrl:{cell['donor_id']}:{cell['barcode']}",
        "cell_barcode": cell["barcode"],
        "donor_id": cell["donor_id"],
        "technical_group": cell["donor_id"],
        "stimulus": cell["stimulus"],
        "multiplet_status": cell["multiplet_status"],
        "reference_annotation": cell["cell_type"],
        "label_a": cell["label_a"],
        "sampling_context": "held_out_triple_sentinel",
        "sampling_stratum": f"{cell['donor_id']}:held_out_triple_sentinel",
        "sampling_hash": cell["sampling_hash"],
        "library_size": f"{cell['library_size']:.12g}",
        "module": "none",
        "subset_id": "none",
        "subset_size": 0,
        "subset_mask": "000",
        "sentinel_genes": _serialize(SENTINELS),
        "sentinel_values": _serialize(
            [f"{value:.12g}" for value in sentinel_values]
        ),
        "joint_control_genes": _serialize(cell["joint_control_genes"]),
        "joint_control_values": _serialize(
            [f"{value:.12g}" for value in cell["joint_control_values"]]
        ),
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


def _intervention_row(
    cell: dict[str, Any],
    subset: tuple[str, ...],
) -> dict[str, Any]:
    base = _base_row(cell)
    match_by_target = {
        str(record["target_gene"]): record for record in cell["joint_matches"]
    }
    target_values = [
        cell["values"][cell["genes"].index(gene)] for gene in subset
    ]
    controls = [
        str(match_by_target[target]["control_gene"]) for target in subset
    ]
    control_values = [
        cell["values"][cell["genes"].index(gene)] for gene in controls
    ]
    subset_bits = "".join("1" if gene in subset else "0" for gene in SENTINELS)
    matches = [match_by_target[target] for target in subset]
    return {
        **base,
        "row_type": "intervention",
        "module": "GNLY_NKG7_CCL5_sentinel_factorial",
        "subset_id": "+".join(subset),
        "subset_size": len(subset),
        "subset_mask": subset_bits,
        "target_genes": _serialize(subset),
        "target_values": _serialize(
            [f"{value:.12g}" for value in target_values]
        ),
        "control_genes": _serialize(controls),
        "control_values": _serialize(
            [f"{value:.12g}" for value in control_values]
        ),
        "control_rank_distance": _serialize(
            [int(record["rank_distance"]) for record in matches]
        ),
        "control_expression_distance": _serialize(
            [f"{float(record['expression_distance']):.12g}" for record in matches]
        ),
        "control_prevalence_distance": _serialize(
            [f"{float(record['prevalence_distance']):.12g}" for record in matches]
        ),
        "control_token_length_distance": _serialize(
            [int(record["token_length_distance"]) for record in matches]
        ),
        "control_total_cost": _serialize(
            [f"{float(record['total_cost']):.12g}" for record in matches]
        ),
        "module_mask_sentence": matching._mask_sentence(
            cell["genes"],
            list(subset),
        ),
        "control_mask_sentence": matching._mask_sentence(
            cell["genes"],
            controls,
        ),
    }


def _validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [
        row for row in rows if row["row_type"] == "intervention"
    ]
    if len(base_rows) != len(parent.EXPECTED_DONORS):
        raise SentinelFactorialBuildError("expected one base cell per donor")
    if len(intervention_rows) != len(parent.EXPECTED_DONORS) * len(SUBSETS):
        raise SentinelFactorialBuildError("expected seven subset rows per donor")

    expected_subset_ids = {"+".join(subset) for subset in SUBSETS}
    for base in base_rows:
        entity_id = base["entity_id"]
        entity_rows = [
            row for row in intervention_rows if row["entity_id"] == entity_id
        ]
        if {str(row["subset_id"]) for row in entity_rows} != expected_subset_ids:
            raise SentinelFactorialBuildError(
                f"subset coverage changed for {entity_id}"
            )
        joint_controls = str(base["joint_control_genes"]).split(";")
        if len(joint_controls) != len(SENTINELS) or len(set(joint_controls)) != len(
            joint_controls
        ):
            raise SentinelFactorialBuildError(
                f"joint controls are not one-to-one for {entity_id}"
            )
        mapping = dict(zip(SENTINELS, joint_controls))
        for row in entity_rows:
            targets = str(row["target_genes"]).split(";")
            controls = str(row["control_genes"]).split(";")
            if controls != [mapping[target] for target in targets]:
                raise SentinelFactorialBuildError(
                    f"subset controls are not nested for {entity_id}/{row['subset_id']}"
                )
            if str(row["module_mask_sentence"]).split().count(
                matching.MASK_TOKEN
            ) != len(targets):
                raise SentinelFactorialBuildError("target mask dose changed")
            if str(row["control_mask_sentence"]).split().count(
                matching.MASK_TOKEN
            ) != len(controls):
                raise SentinelFactorialBuildError("control mask dose changed")

    inputs: list[dict[str, str]] = []
    for row in base_rows:
        inputs.append(
            {
                "entity_id": str(row["entity_id"]),
                "condition": "unmasked",
                "subset_id": "none",
                "sentence": str(row["original_sentence"]),
            }
        )
    for row in intervention_rows:
        for condition, column in (
            ("target_mask", "module_mask_sentence"),
            ("control_mask", "control_mask_sentence"),
        ):
            inputs.append(
                {
                    "entity_id": str(row["entity_id"]),
                    "condition": condition,
                    "subset_id": str(row["subset_id"]),
                    "sentence": str(row[column]),
                }
            )
    expected_unique_inputs = len(parent.EXPECTED_DONORS) * (
        1 + 2 * len(SUBSETS)
    )
    if len(inputs) != expected_unique_inputs:
        raise SentinelFactorialBuildError("logical input count changed")
    if len({record["sentence"] for record in inputs}) != expected_unique_inputs:
        raise SentinelFactorialBuildError(
            "factorial contains duplicate input sentences"
        )
    logical_observations = len(inputs) * PROMPT_FORMS_PER_INPUT
    unique_calls = (
        len({record["sentence"] for record in inputs}) * PROMPT_FORMS_PER_INPUT
    )
    if logical_observations != 480 or unique_calls != 480:
        raise SentinelFactorialBuildError(
            "frozen execution budget changed: "
            f"logical={logical_observations}, unique={unique_calls}"
        )
    sentence_plan = [
        {
            **{key: record[key] for key in ("entity_id", "condition", "subset_id")},
            "sentence_sha256": hashlib.sha256(
                record["sentence"].encode()
            ).hexdigest(),
        }
        for record in inputs
    ]
    return {
        "base_rows": len(base_rows),
        "factorial_subset_rows": len(intervention_rows),
        "unique_input_sentences": len({record["sentence"] for record in inputs}),
        "prompt_forms_per_input": PROMPT_FORMS_PER_INPUT,
        "logical_condition_form_observations": logical_observations,
        "unique_api_calls_after_input_deduplication": unique_calls,
        "input_sentence_plan_sha256": _canonical_sha256(sentence_plan),
    }


def build(
    source_path: Path,
    matrix_path: Path,
    barcode_path: Path,
    gene_path: Path,
    metadata_path: Path,
    archive_path: Path,
    parent_builder_path: Path,
    parent_csv_path: Path,
    parent_manifest_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    parent_base_rows, parent_manifest, parent_hashes = _read_parent_exclusions(
        parent_builder_path,
        parent_csv_path,
        parent_manifest_path,
    )
    source_sha256 = parent._verify_hash(
        source_path,
        parent.EXPECTED_SOURCE_SHA256,
        "PBMC3k source",
    )
    matrix, gene_symbols, _, metadata, raw_hashes = parent._load_authenticated_target(
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
    ranked, universe, frame_support = parent._rank_target_cells(
        matrix,
        gene_symbols,
        metadata,
        source_gene_names,
    )
    if len(ranked) != EXPECTED_RENDERABLE_FRAME_CELLS:
        raise SentinelFactorialBuildError(
            f"renderable frame changed: expected {EXPECTED_RENDERABLE_FRAME_CELLS}, "
            f"observed {len(ranked)}"
        )

    ranked_by_barcode = {cell["barcode"]: cell for cell in ranked}
    excluded_barcodes = {row["cell_barcode"] for row in parent_base_rows}
    if len(excluded_barcodes) != EXPECTED_PARENT_BASE_CELLS:
        raise SentinelFactorialBuildError("holdout exclusion set changed")
    for row in parent_base_rows:
        cell = ranked_by_barcode.get(row["cell_barcode"])
        if cell is None:
            raise SentinelFactorialBuildError(
                f"parent exclusion absent from ranked frame: {row['cell_barcode']}"
            )
        if " ".join(cell["genes"]) != row["original_sentence"]:
            raise SentinelFactorialBuildError(
                f"parent top-50 reconstruction changed: {row['cell_barcode']}"
            )
        if parent._sampling_hash(cell["donor_id"], cell["barcode"]) != row[
            "sampling_hash"
        ]:
            raise SentinelFactorialBuildError(
                f"parent sampling hash changed: {row['cell_barcode']}"
            )

    sentinel_set = set(SENTINELS)
    all_eligible = [
        cell for cell in ranked if sentinel_set <= set(cell["genes"])
    ]
    unused_eligible = [
        cell for cell in all_eligible if cell["barcode"] not in excluded_barcodes
    ]
    all_counts = Counter(cell["donor_id"] for cell in all_eligible)
    unused_counts = Counter(cell["donor_id"] for cell in unused_eligible)
    if {
        donor: all_counts[donor] for donor in parent.EXPECTED_DONORS
    } != EXPECTED_ALL_TRIPLE_SUPPORT:
        raise SentinelFactorialBuildError(
            f"full-frame triple support changed: {dict(all_counts)}"
        )
    if {
        donor: unused_counts[donor] for donor in parent.EXPECTED_DONORS
    } != EXPECTED_UNUSED_TRIPLE_SUPPORT:
        raise SentinelFactorialBuildError(
            f"post-holdout triple support changed: {dict(unused_counts)}"
        )

    support_by_donor = [
        _support_record(
            donor,
            all_eligible,
            unused_eligible,
            excluded_barcodes,
        )
        for donor in parent.EXPECTED_DONORS
    ]
    selected: list[dict[str, Any]] = []
    for donor in parent.EXPECTED_DONORS:
        candidates = sorted(
            (cell for cell in unused_eligible if cell["donor_id"] == donor),
            key=lambda cell: (
                _sampling_hash(donor, cell["barcode"]),
                cell["barcode"],
            ),
        )
        if not candidates:
            raise SentinelFactorialBuildError(
                f"no held-out triple-positive cell for donor {donor}"
            )
        chosen = candidates[0]
        selected.append(
            {
                **chosen,
                "sampling_hash": _sampling_hash(donor, chosen["barcode"]),
            }
        )
    if any(cell["barcode"] in excluded_barcodes for cell in selected):
        raise SentinelFactorialBuildError("a selected cell leaks from the parent cohort")

    prevalence_counts = Counter(
        gene for cell in ranked for gene in cell["genes"]
    )
    prevalence = {
        gene: count / len(ranked) for gene, count in prevalence_counts.items()
    }
    source_neutral_controls, source_prevalence_gap = (
        source_control._source_neutral_controls(source, universe)
    )

    matched_cells: list[dict[str, Any]] = []
    exact_matches: list[dict[str, Any]] = []
    for cell in selected:
        controls, control_values, matches = matching._match_control_genes(
            cell["genes"],
            cell["values"],
            list(SENTINELS),
            prevalence,
            source_neutral_controls,
        )
        if len(controls) != len(SENTINELS) or len(set(controls)) != len(controls):
            raise SentinelFactorialBuildError(
                f"joint matching is not one-to-one for {cell['barcode']}"
            )
        if [str(record["target_gene"]) for record in matches] != list(SENTINELS):
            raise SentinelFactorialBuildError(
                f"joint matching target order changed for {cell['barcode']}"
            )
        if any(control not in source_neutral_controls for control in controls):
            raise SentinelFactorialBuildError(
                "control outside the PBMC3k reference-prevalence-balanced "
                f"candidate set for {cell['barcode']}"
            )
        if any(control in matching.FAMOUS_PARENT for control in controls):
            raise SentinelFactorialBuildError(
                f"parent marker selected as a control for {cell['barcode']}"
            )
        matched_cell = {
            **cell,
            "joint_control_genes": controls,
            "joint_control_values": control_values,
            "joint_matches": matches,
        }
        matched_cells.append(matched_cell)
        for record in matches:
            exact_matches.append(
                {
                    "donor_id": cell["donor_id"],
                    "cell_barcode": cell["barcode"],
                    "target_gene": str(record["target_gene"]),
                    "control_gene": str(record["control_gene"]),
                    "target_rank_1based": int(record["target_rank_1based"]),
                    "control_rank_1based": int(record["control_rank_1based"]),
                    "rank_distance": int(record["rank_distance"]),
                    "expression_distance": float(record["expression_distance"]),
                    "prevalence_distance": float(record["prevalence_distance"]),
                    "token_length_distance": int(
                        record["token_length_distance"]
                    ),
                    "total_cost": float(record["total_cost"]),
                }
            )

    rows: list[dict[str, Any]] = []
    for cell in matched_cells:
        rows.append(_base_row(cell))
        rows.extend(_intervention_row(cell, subset) for subset in SUBSETS)
    budget = _validate_rows(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_cells = [
        {
            "donor_id": cell["donor_id"],
            "cell_barcode": cell["barcode"],
            "sampling_hash": cell["sampling_hash"],
            "reference_annotation_descriptive_only": cell["cell_type"],
            "sentinel_ranks_1based": {
                gene: cell["genes"].index(gene) + 1 for gene in SENTINELS
            },
            "sentinel_values": {
                gene: cell["values"][cell["genes"].index(gene)]
                for gene in SENTINELS
            },
            "joint_control_genes": list(cell["joint_control_genes"]),
        }
        for cell in matched_cells
    ]
    exclusion_records = sorted(
        (
            {
                "donor_id": row["donor_id"],
                "cell_barcode": row["cell_barcode"],
            }
            for row in parent_base_rows
        ),
        key=lambda record: (int(record["donor_id"]), record["cell_barcode"]),
    )
    match_costs = np.asarray(
        [float(record["total_cost"]) for record in exact_matches],
        dtype=float,
    )
    rank_distances = np.asarray(
        [int(record["rank_distance"]) for record in exact_matches],
        dtype=float,
    )
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "status": "built_not_executed",
        "claim_scope": (
            "held_out_multi_donor_sle_control_output_level_three_token_"
            "factorial_not_gene_pathway_hidden_state_activation_gap_or_physical_law"
        ),
        "source": {
            "geo_accession": "GSE96583",
            "arm": "batch 2 control; 6-hour culture without IFN-beta",
            "population": "eight SLE donors",
            "pbmc3k_source_sha256": source_sha256,
            "authenticated_gse96583_hashes": raw_hashes,
            "renderable_cd8_nk_frame_cells": len(ranked),
            "common_non_housekeeping_universe_genes": len(universe),
            "common_universe_sha256": frame_support["universe_sha256"],
            "annotation_boundary": (
                "deposited CD8/NK labels define only the upstream task frame "
                "and are descriptive after selection; they do not enter the "
                "eligibility gate, holdout exclusion rule, sampling hash, "
                "sentinel subset, or control assignment"
            ),
        },
        "parent_holdout_exclusion": {
            "analysis_id": parent_manifest["analysis_id"],
            "builder_path": str(parent_builder_path.relative_to(ROOT)),
            "builder_sha256": parent_hashes["builder"],
            "csv_path": str(parent_csv_path.relative_to(ROOT)),
            "csv_sha256": parent_hashes["csv"],
            "manifest_path": str(parent_manifest_path.relative_to(ROOT)),
            "manifest_sha256": parent_hashes["manifest"],
            "excluded_base_cells": len(exclusion_records),
            "exclusion_records_sha256": _canonical_sha256(exclusion_records),
            "rule": (
                "exclude every barcode represented by a base row in the "
                "frozen 56-cell parent CSV before sentinel eligibility and sampling"
            ),
        },
        "eligibility_and_sampling": {
            "required_top50_sentinels": list(SENTINELS),
            "full_frame_eligible_cells": len(all_eligible),
            "eligible_after_parent_holdout": len(unused_eligible),
            "support_by_donor": support_by_donor,
            "minimum_unused_support_per_donor": min(unused_counts.values()),
            "sampling_policy": (
                "within each donor, take the eligible non-parent cell with "
                "the lowest SHA-256 key salted by the immutable parent CSV "
                "digest; barcode is the deterministic tie break"
            ),
            "sampling_hash_payload": (
                f"{EXPECTED_PARENT_SHA256['csv']}|<donor_id>|<barcode>"
            ),
            "sampling_salt_provenance": (
                "immutable pre-hypothesis parent CSV SHA-256; the analysis ID "
                "is retained for artifact identity but excluded from selection"
            ),
            "selected_cells": selected_cells,
            "selected_cells_sha256": _canonical_sha256(selected_cells),
            "selected_reference_annotation_counts_descriptive_only": dict(
                sorted(Counter(cell["cell_type"] for cell in matched_cells).items())
            ),
        },
        "factorial": {
            "sentinel_order": list(SENTINELS),
            "nonempty_subsets": [
                {
                    "subset_id": "+".join(subset),
                    "subset_size": len(subset),
                    "subset_mask": "".join(
                        "1" if gene in subset else "0" for gene in SENTINELS
                    ),
                }
                for subset in SUBSETS
            ],
            "mask_token": matching.MASK_TOKEN,
            "nesting_rule": (
                "jointly assign one distinct control to each sentinel once "
                "per cell; every subset masks its targets and the corresponding "
                "subset of that fixed control assignment"
            ),
        },
        "control_matching": {
            "method": (
                "within-cell linear-sum assignment for all three sentinels "
                "jointly, minimizing the frozen total cost over distinct controls"
            ),
            "cost": (
                "rank_distance/3 + abs_log_expression_distance/0.5 + "
                "top50_prevalence_distance/0.10 + token_length_distance/3"
            ),
            "prevalence_population": (
                "all 1997 renderable cells in the frozen CD8/NK task frame"
            ),
            "framed_prevalence_sha256": _canonical_sha256(
                {gene: prevalence[gene] for gene in sorted(prevalence)}
            ),
            "pbmc3k_reference_prevalence_balance_rule": (
                "PBMC3k absolute CD8-vs-NK top-50 prevalence gap <= 0.10; "
                "all 30 frozen parent markers excluded from controls"
            ),
            "semantic_boundary": (
                "reference-prevalence-balanced is an operational matching "
                "criterion and does not imply that a control token is "
                "biologically, linguistically, or causally neutral"
            ),
            "pbmc3k_reference_prevalence_balanced_control_gene_count": len(
                source_neutral_controls
            ),
            "pbmc3k_reference_prevalence_gap_sha256": hashlib.sha256(
                (
                    "\n".join(
                        f"{gene}\t{source_prevalence_gap[gene]:.12g}"
                        for gene in sorted(source_prevalence_gap)
                    )
                    + "\n"
                ).encode()
            ).hexdigest(),
            "matched_pairs": len(exact_matches),
            "exact_matches": exact_matches,
            "exact_matches_sha256": _canonical_sha256(exact_matches),
            "median_total_cost": float(np.median(match_costs)),
            "max_total_cost": float(match_costs.max()),
            "median_rank_distance": float(np.median(rank_distances)),
            "max_rank_distance": float(rank_distances.max()),
        },
        "execution_budget": {
            **budget,
            "formula": "8 cells * (1 unmasked + 7 subsets * 2 masks) * 4 forms = 480",
            "model_calls_made_by_builder": 0,
        },
        "artifacts": {
            "builder_path": str(Path(__file__).relative_to(ROOT)),
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
    parser.add_argument(
        "--parent-builder",
        type=Path,
        default=DEFAULT_PARENT_BUILDER,
    )
    parser.add_argument("--parent-csv", type=Path, default=DEFAULT_PARENT_CSV)
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        default=DEFAULT_PARENT_MANIFEST,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build(
        args.source_h5ad.resolve(),
        args.matrix.resolve(),
        args.barcodes.resolve(),
        args.genes.resolve(),
        args.metadata.resolve(),
        args.archive.resolve(),
        args.parent_builder.resolve(),
        args.parent_csv.resolve(),
        args.parent_manifest.resolve(),
        args.out.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

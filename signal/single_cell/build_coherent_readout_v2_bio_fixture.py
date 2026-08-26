"""Build a disjoint GSE96583 v2 biological development fixture.

The source is the authenticated batch-2 control-singlet CD8/NK task frame used by
the existing GSE96583 builders.  Deposited cell labels are consulted only by that
frozen renderer to define the pre-existing task frame.  They never enter candidate
hashes, selected-cell records, fixture fields, or the manifest's selected payload.

Before selection, every barcode or entity present anywhere in either the frozen
module-replication CSV or sentinel-factorial CSV is excluded.  One remaining
renderable top-50 cell is hash-selected per donor without using a source label,
then crossed with the lineage and cytotoxic-state readouts.  The resulting bank is
permanent development-only material.  This builder makes zero model calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_DIR = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "gse96583"

ANALYSIS_ID = "gse96583-coherent-readout-v2-biological-development-fixture-v1"
FIXTURE_SCHEMA = "coherent-readout-v2-biological-development-fixture-v1"
RECORD_SCHEMA = "coherent-readout-v2-biological-development-item-v1"
MANIFEST_SCHEMA = (
    "coherent-readout-v2-biological-development-fixture-manifest-v1"
)
FREEZE_DATE = "2026-08-02"
OUTCOME_EXPOSURE = "project_record_qwen_outcome_unexposed_at_freeze"

EXPECTED_DONORS = ("101", "107", "1015", "1016", "1039", "1244", "1256", "1488")
INPUT_FAMILIES = ("unmodified",)
READOUTS = {
    "cytotoxic_state": {
        "negative_class": "cytotoxic-low",
        "positive_class": "cytotoxic-high",
    },
    "lineage": {
        "negative_class": "CD8 T cell",
        "positive_class": "NK cell",
    },
}

FIREWALL = {
    "partition": "permanent_development_only",
    "permanent": True,
    "confirmatory_eligibility": "prohibited",
    "promotion_to_confirmation": "forbidden",
    "outcome_exposure": OUTCOME_EXPOSURE,
    "claim_scope": (
        "readout_engineering_only_no_biology_knowledge_or_activation_inference"
    ),
}

DEFAULT_RENDERER_BUILDER = SIGNAL_DIR / "build_gse96583_module_replication.py"
DEFAULT_SOURCE = ROOT / "data" / "pbmc3k_processed.h5ad"
DEFAULT_MATRIX = RAW_ROOT / "GSM2560248_2.1.mtx.gz"
DEFAULT_BARCODES = RAW_ROOT / "GSM2560248_barcodes.tsv.gz"
DEFAULT_GENES = RAW_ROOT / "GSE96583_batch2.genes.tsv.gz"
DEFAULT_METADATA = RAW_ROOT / "GSE96583_batch2.total.tsne.df.tsv.gz"
DEFAULT_ARCHIVE = RAW_ROOT / "GSE96583_RAW.tar"
DEFAULT_RESULTS_ROOT = ROOT / "results"

DEFAULT_MODULE_BUILDER = DEFAULT_RENDERER_BUILDER
DEFAULT_MODULE_CSV = SIGNAL_DIR / "gse96583_cd8_nk_module_replication.csv"
DEFAULT_MODULE_MANIFEST = SIGNAL_DIR / "gse96583_cd8_nk_module_replication.manifest.json"
DEFAULT_SENTINEL_BUILDER = SIGNAL_DIR / "build_gse96583_sentinel_factorial.py"
DEFAULT_SENTINEL_CSV = SIGNAL_DIR / "gse96583_sentinel_factorial.csv"
DEFAULT_SENTINEL_MANIFEST = SIGNAL_DIR / "gse96583_sentinel_factorial.manifest.json"

DEFAULT_OUT = SIGNAL_DIR / "coherent_readout_v2_bio_fixture.json"
DEFAULT_MANIFEST = SIGNAL_DIR / "coherent_readout_v2_bio_fixture.manifest.json"

EXPECTED_RENDERER_SHA256 = (
    "b089e91231042db7c8a7e17fd6dbe1e32169bf1f05361cd6e5e8041c395f00bf"
)
EXPECTED_SOURCE_SHA256 = (
    "0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38"
)
EXPECTED_RAW_SHA256 = {
    "archive": "e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80",
    "matrix": "32add28a0b3397d9ef3f220b7a6a55e98e60fe7b66fe48d0986d634df8ca0013",
    "barcodes": "d58d8d55cbe4a12757207784b3bc9227bf200c9100ca15131176e9f8159c955e",
    "genes": "93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca",
    "metadata": "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09",
}
EXPECTED_COMMON_UNIVERSE_GENES = 13_503
EXPECTED_COMMON_UNIVERSE_SHA256 = (
    "145e849b9860115d455226de456935ba3b29366915e659dd2a58637916cb5506"
)
EXPECTED_RENDERABLE_FRAME_CELLS = 1_997
EXPECTED_EXCLUDED_CELLS = 64
EXPECTED_ELIGIBLE_BY_DONOR = {
    "101": 136,
    "107": 52,
    "1015": 348,
    "1016": 656,
    "1039": 36,
    "1244": 155,
    "1256": 384,
    "1488": 166,
}

EXCLUSION_SPECS = (
    {
        "name": "module_replication",
        "analysis_id": "gse96583-cd8-nk-context-module-donor-replication-v1",
        "builder": DEFAULT_MODULE_BUILDER,
        "builder_sha256": EXPECTED_RENDERER_SHA256,
        "csv": DEFAULT_MODULE_CSV,
        "csv_sha256": "f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321",
        "manifest": DEFAULT_MODULE_MANIFEST,
        "manifest_sha256": "3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f",
        "rows": 168,
        "unique_cells": 56,
    },
    {
        "name": "sentinel_factorial",
        "analysis_id": "gse96583-sentinel-factorial-holdout-v1",
        "builder": DEFAULT_SENTINEL_BUILDER,
        "builder_sha256": "9336e633763b91c8d0c983d75b67004da9ee6f681c1b4f7dd2d4ba92b07f8992",
        "csv": DEFAULT_SENTINEL_CSV,
        "csv_sha256": "673db8c8bcd6ba923e62891de6cd5f04f97967706ac3ade8a6e44ad2d14a4b95",
        "manifest": DEFAULT_SENTINEL_MANIFEST,
        "manifest_sha256": "9f0035469c81852b11e2a36651b7892bf2dad4d30f8049b37cd1f655ca9bf0c4",
        "rows": 64,
        "unique_cells": 8,
    },
)

FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "mode",
    "outcome_exposure",
    "firewall",
    "donor_ids",
    "input_families",
    "readouts",
    "items",
    "model_calls_made_by_builder",
}
ITEM_KEYS = {
    "schema_version",
    "fixture_record_id",
    "item_id",
    "donor_id",
    "source_entity_id",
    "source_cell_barcode",
    "input_family",
    "readout_id",
    "positive_class",
    "negative_class",
    "gene_sentence",
    "gene_sentence_sha256",
    "source_projection_sha256",
    "selection_record_sha256",
    "outcome_exposure",
    "firewall",
}
MANIFEST_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "freeze_date",
    "mode",
    "outcome_exposure",
    "claim_scope",
    "firewall",
    "label_boundary",
    "source",
    "exclusions",
    "selection",
    "outcome_exposure_audit",
    "contract",
    "artifacts",
}
FORBIDDEN_LABEL_FIELDS = {
    "cell_type",
    "reference_annotation",
    "label",
    "label_a",
    "ground_truth",
    "deposited_cell_type",
    "cytotoxic_state_label",
    "state_label",
}
RESULT_REGISTRY_SUFFIXES = (".csv", ".json", ".jsonl")


class BioFixtureError(ValueError):
    """Raised when a v2 biological fixture source or artifact violates its lock."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise BioFixtureError(f"missing {label}: {path}")
    observed = _file_sha256(path)
    if observed != expected:
        raise BioFixtureError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BioFixtureError(
            f"{label} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _load_renderer() -> Any:
    _verify_hash(
        DEFAULT_RENDERER_BUILDER,
        EXPECTED_RENDERER_SHA256,
        "authenticated GSE96583 renderer",
    )
    signal_path = str(SIGNAL_DIR)
    if signal_path not in sys.path:
        sys.path.insert(0, signal_path)
    try:
        import build_gse96583_module_replication as renderer
    except ImportError as error:
        raise BioFixtureError(
            "authenticated raw rendering requires the optional full dependencies"
        ) from error
    if Path(renderer.__file__).resolve() != DEFAULT_RENDERER_BUILDER.resolve():
        raise BioFixtureError("imported GSE96583 renderer from an unexpected path")
    return renderer


def _read_exclusions() -> tuple[set[str], set[str], dict[str, Any]]:
    by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    registry_summaries: list[dict[str, Any]] = []
    for spec in EXCLUSION_SPECS:
        observed_builder = _verify_hash(
            spec["builder"], spec["builder_sha256"], f"{spec['name']} builder"
        )
        observed_csv = _verify_hash(
            spec["csv"], spec["csv_sha256"], f"{spec['name']} CSV"
        )
        observed_manifest = _verify_hash(
            spec["manifest"],
            spec["manifest_sha256"],
            f"{spec['name']} manifest",
        )
        manifest = json.loads(spec["manifest"].read_text(encoding="utf-8"))
        if manifest.get("analysis_id") != spec["analysis_id"]:
            raise BioFixtureError(f"{spec['name']} manifest analysis ID changed")
        artifacts = manifest.get("artifacts", {})
        if artifacts.get("builder_sha256") != observed_builder:
            raise BioFixtureError(f"{spec['name']} manifest does not bind its builder")
        if artifacts.get("csv_sha256") != observed_csv:
            raise BioFixtureError(f"{spec['name']} manifest does not bind its CSV")

        with spec["csv"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != spec["rows"]:
            raise BioFixtureError(f"{spec['name']} row count changed")
        required = {"cell_barcode", "entity_id", "donor_id"}
        if not rows or not required <= set(rows[0]):
            raise BioFixtureError(f"{spec['name']} exclusion fields changed")
        identities = {(row["cell_barcode"], row["entity_id"]) for row in rows}
        if len(identities) != spec["unique_cells"]:
            raise BioFixtureError(f"{spec['name']} unique-cell count changed")
        for row in rows:
            expected_entity = (
                f"gse96583:ctrl:{row['donor_id']}:{row['cell_barcode']}"
            )
            if row["entity_id"] != expected_entity:
                raise BioFixtureError(f"{spec['name']} identity topology changed")
            by_identity[(row["cell_barcode"], row["entity_id"])].add(spec["name"])
        registry_summaries.append(
            {
                "name": spec["name"],
                "analysis_id": spec["analysis_id"],
                "builder_path": _relative_path(spec["builder"]),
                "builder_sha256": observed_builder,
                "csv_path": _relative_path(spec["csv"]),
                "csv_sha256": observed_csv,
                "manifest_path": _relative_path(spec["manifest"]),
                "manifest_sha256": observed_manifest,
                "rows": len(rows),
                "unique_barcodes": len({row["cell_barcode"] for row in rows}),
                "unique_entities": len({row["entity_id"] for row in rows}),
            }
        )

    exclusion_records = [
        {
            "cell_barcode": barcode,
            "entity_id": entity_id,
            "registries": sorted(registries),
        }
        for (barcode, entity_id), registries in sorted(by_identity.items())
    ]
    if len(exclusion_records) != EXPECTED_EXCLUDED_CELLS:
        raise BioFixtureError("combined exclusion set changed")
    barcodes = {record["cell_barcode"] for record in exclusion_records}
    entities = {record["entity_id"] for record in exclusion_records}
    if len(barcodes) != EXPECTED_EXCLUDED_CELLS or len(entities) != EXPECTED_EXCLUDED_CELLS:
        raise BioFixtureError("exclusion barcode/entity mapping is not one-to-one")
    audit = {
        "rule": (
            "exclude every barcode or entity occurring in any row of both frozen "
            "source registries before donor-wise hash selection"
        ),
        "registries": registry_summaries,
        "excluded_unique_barcodes": len(barcodes),
        "excluded_unique_entities": len(entities),
        "exclusion_records_sha256": _canonical_sha256(exclusion_records),
    }
    return barcodes, entities, audit


def _load_authenticated_frame() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    renderer = _load_renderer()
    source_sha256 = _verify_hash(
        DEFAULT_SOURCE, EXPECTED_SOURCE_SHA256, "PBMC3k rendering-universe source"
    )
    matrix, gene_symbols, _, metadata, raw_hashes = renderer._load_authenticated_target(
        DEFAULT_MATRIX,
        DEFAULT_BARCODES,
        DEFAULT_GENES,
        DEFAULT_METADATA,
        DEFAULT_ARCHIVE,
    )
    for key, expected in EXPECTED_RAW_SHA256.items():
        if raw_hashes.get(key) != expected:
            raise BioFixtureError(f"authenticated raw {key} digest changed")
    try:
        import anndata as ad
    except ImportError as error:
        raise BioFixtureError(
            "authenticated raw rendering requires the optional anndata dependency"
        ) from error
    source = ad.read_h5ad(DEFAULT_SOURCE, backed="r")
    try:
        source_gene_names = set(
            map(
                str,
                source.raw.var_names if source.raw is not None else source.var_names,
            )
        )
    finally:
        source.file.close()
    ranked, universe, support = renderer._rank_target_cells(
        matrix,
        gene_symbols,
        metadata,
        source_gene_names,
    )
    if len(ranked) != EXPECTED_RENDERABLE_FRAME_CELLS:
        raise BioFixtureError("renderable CD8/NK frame count changed")
    if len(universe) != EXPECTED_COMMON_UNIVERSE_GENES:
        raise BioFixtureError("common rendering universe size changed")
    if support["universe_sha256"] != EXPECTED_COMMON_UNIVERSE_SHA256:
        raise BioFixtureError("common rendering universe digest changed")
    if tuple(sorted({cell["donor_id"] for cell in ranked}, key=int)) != EXPECTED_DONORS:
        raise BioFixtureError("renderable frame donor set changed")
    source_audit = {
        "geo_accession": "GSE96583",
        "arm": "batch 2 control; 6-hour culture without IFN-beta",
        "population": "eight SLE donors",
        "renderer_builder_path": _relative_path(DEFAULT_RENDERER_BUILDER),
        "renderer_builder_sha256": EXPECTED_RENDERER_SHA256,
        "pbmc3k_rendering_universe_path": _relative_path(DEFAULT_SOURCE),
        "pbmc3k_rendering_universe_sha256": source_sha256,
        "authenticated_raw_paths": {
            "archive": _relative_path(DEFAULT_ARCHIVE),
            "matrix": _relative_path(DEFAULT_MATRIX),
            "barcodes": _relative_path(DEFAULT_BARCODES),
            "genes": _relative_path(DEFAULT_GENES),
            "metadata": _relative_path(DEFAULT_METADATA),
        },
        "authenticated_raw_sha256": raw_hashes,
        "renderable_frame_cells": len(ranked),
        "common_non_housekeeping_universe_genes": len(universe),
        "common_universe_sha256": support["universe_sha256"],
        "annotation_boundary": (
            "deposited CD8/NK labels are used only inside the authenticated "
            "renderer to define the pre-existing task frame; no deposited label "
            "enters selection, fixture fields, model fields, or exported audits"
        ),
    }
    return ranked, source_audit


def _selection_hash(
    donor_id: str,
    barcode: str,
    gene_sentence_sha256: str,
    matrix_sha256: str,
    exclusion_records_sha256: str,
) -> str:
    payload = (
        f"{ANALYSIS_ID}|{matrix_sha256}|{exclusion_records_sha256}|"
        f"{donor_id}|{barcode}|{gene_sentence_sha256}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_cells(
    ranked: Sequence[Mapping[str, Any]],
    excluded_barcodes: set[str],
    excluded_entities: set[str],
    matrix_sha256: str,
    exclusion_records_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_by_donor: dict[str, list[dict[str, Any]]] = {
        donor: [] for donor in EXPECTED_DONORS
    }
    for cell in ranked:
        donor_id = str(cell["donor_id"])
        barcode = str(cell["barcode"])
        entity_id = f"gse96583:ctrl:{donor_id}:{barcode}"
        if barcode in excluded_barcodes or entity_id in excluded_entities:
            continue
        genes = list(map(str, cell["genes"]))
        if len(genes) != 50 or len(set(genes)) != 50:
            raise BioFixtureError("candidate is not a distinct top-50 gene sentence")
        gene_sentence = " ".join(genes)
        sentence_sha256 = hashlib.sha256(gene_sentence.encode("utf-8")).hexdigest()
        selection_hash = _selection_hash(
            donor_id,
            barcode,
            sentence_sha256,
            matrix_sha256,
            exclusion_records_sha256,
        )
        projection = {
            "donor_id": donor_id,
            "source_cell_barcode": barcode,
            "source_entity_id": entity_id,
            "gene_sentence": gene_sentence,
            "gene_sentence_sha256": sentence_sha256,
            "selection_hash": selection_hash,
        }
        projection["selection_record_sha256"] = _canonical_sha256(projection)
        candidates_by_donor[donor_id].append(projection)

    observed_counts = {
        donor: len(candidates_by_donor[donor]) for donor in EXPECTED_DONORS
    }
    if observed_counts != EXPECTED_ELIGIBLE_BY_DONOR:
        raise BioFixtureError(
            f"eligible support changed: expected {EXPECTED_ELIGIBLE_BY_DONOR}, "
            f"observed {observed_counts}"
        )
    candidate_registry = sorted(
        (
            {
                key: candidate[key]
                for key in (
                    "donor_id",
                    "source_cell_barcode",
                    "source_entity_id",
                    "gene_sentence_sha256",
                    "selection_hash",
                    "selection_record_sha256",
                )
            }
            for candidates in candidates_by_donor.values()
            for candidate in candidates
        ),
        key=lambda row: (int(row["donor_id"]), row["selection_hash"], row["source_cell_barcode"]),
    )
    selected = [
        min(
            candidates_by_donor[donor],
            key=lambda row: (row["selection_hash"], row["source_cell_barcode"]),
        )
        for donor in EXPECTED_DONORS
    ]
    if any(
        row["source_cell_barcode"] in excluded_barcodes
        or row["source_entity_id"] in excluded_entities
        for row in selected
    ):
        raise BioFixtureError("selected cell overlaps an exclusion registry")
    selection_audit = {
        "policy": (
            "within each donor after the complete barcode-or-entity exclusion, "
            "select the lowest SHA-256 key with barcode as deterministic tie break"
        ),
        "sampling_hash_payload": (
            f"{ANALYSIS_ID}|<matrix_sha256>|<exclusion_records_sha256>|"
            "<donor_id>|<barcode>|<gene_sentence_sha256>"
        ),
        "sampling_fields_exclude_deposited_labels": True,
        "eligible_after_exclusion_by_donor": observed_counts,
        "candidate_registry_records": len(candidate_registry),
        "candidate_registry_sha256": _canonical_sha256(candidate_registry),
        "selected_cells": [
            {
                key: row[key]
                for key in (
                    "donor_id",
                    "source_cell_barcode",
                    "source_entity_id",
                    "gene_sentence_sha256",
                    "selection_hash",
                    "selection_record_sha256",
                )
            }
            for row in selected
        ],
    }
    selection_audit["selected_cells_sha256"] = _canonical_sha256(
        selection_audit["selected_cells"]
    )
    return selected, selection_audit


def _scan_qwen_result_registries(
    selected: Sequence[Mapping[str, Any]],
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    if not results_root.is_dir():
        raise BioFixtureError(f"project results root does not exist: {results_root}")
    registry_files: list[dict[str, Any]] = []
    hits: list[dict[str, str]] = []
    candidate_files = sorted(
        path
        for path in results_root.rglob("*")
        if path.is_file() and path.suffix.lower() in RESULT_REGISTRY_SUFFIXES
    )
    for path in candidate_files:
        raw = path.read_bytes()
        relative = _relative_path(path)
        if "qwen" not in relative.lower() and b"qwen" not in raw.lower():
            continue
        registry_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
        for row in selected:
            queries = {
                "source_cell_barcode": str(row["source_cell_barcode"]),
                "source_entity_id": str(row["source_entity_id"]),
                "gene_sentence": str(row["gene_sentence"]),
            }
            for field, query in queries.items():
                if query.encode("utf-8") in raw:
                    hits.append(
                        {
                            "path": relative,
                            "donor_id": str(row["donor_id"]),
                            "field": field,
                        }
                    )
    if hits:
        raise BioFixtureError(f"selected identity is already in a Qwen result registry: {hits}")
    return {
        "status": OUTCOME_EXPOSURE,
        "bounded_scope": (
            "exact selected barcode, entity, and top-50 sentence were absent from "
            "structured project result files associated with Qwen at freeze"
        ),
        "results_root": _relative_path(results_root),
        "included_suffixes": list(RESULT_REGISTRY_SUFFIXES),
        "qwen_registry_files": registry_files,
        "qwen_registry_files_sha256": _canonical_sha256(registry_files),
        "exact_identity_hits": 0,
        "global_or_external_exposure_claimed": False,
    }


def _build_record(cell: Mapping[str, Any], readout_id: str) -> dict[str, Any]:
    if readout_id not in READOUTS:
        raise BioFixtureError(f"unknown readout: {readout_id}")
    source_projection = {
        key: cell[key]
        for key in (
            "donor_id",
            "source_cell_barcode",
            "source_entity_id",
            "gene_sentence_sha256",
            "selection_hash",
            "selection_record_sha256",
        )
    }
    item_id = (
        f"gse96583:v2dev:{cell['donor_id']}:"
        f"{cell['source_cell_barcode']}:unmodified"
    )
    record = {
        "schema_version": RECORD_SCHEMA,
        "item_id": item_id,
        "donor_id": cell["donor_id"],
        "source_entity_id": cell["source_entity_id"],
        "source_cell_barcode": cell["source_cell_barcode"],
        "input_family": "unmodified",
        "readout_id": readout_id,
        **READOUTS[readout_id],
        "gene_sentence": cell["gene_sentence"],
        "gene_sentence_sha256": cell["gene_sentence_sha256"],
        "source_projection_sha256": _canonical_sha256(source_projection),
        "selection_record_sha256": cell["selection_record_sha256"],
        "outcome_exposure": OUTCOME_EXPOSURE,
        "firewall": FIREWALL,
    }
    record["fixture_record_id"] = _canonical_sha256(record)
    return record


def _validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(fixture, FIXTURE_KEYS, "fixture")
    if fixture["schema_version"] != FIXTURE_SCHEMA:
        raise BioFixtureError("fixture schema changed")
    if fixture["analysis_id"] != ANALYSIS_ID or fixture["mode"] != "development":
        raise BioFixtureError("fixture identity or mode changed")
    if fixture["outcome_exposure"] != OUTCOME_EXPOSURE:
        raise BioFixtureError("fixture outcome-exposure boundary changed")
    if fixture["firewall"] != FIREWALL:
        raise BioFixtureError("fixture firewall changed")
    if fixture["donor_ids"] != list(EXPECTED_DONORS):
        raise BioFixtureError("fixture donor order changed")
    if fixture["input_families"] != list(INPUT_FAMILIES):
        raise BioFixtureError("fixture input families changed")
    if fixture["readouts"] != READOUTS:
        raise BioFixtureError("fixture readouts changed")
    if fixture["model_calls_made_by_builder"] != 0:
        raise BioFixtureError("fixture claims model execution")
    items = fixture["items"]
    if not isinstance(items, list) or len(items) != 16:
        raise BioFixtureError("fixture must contain exactly 16 rows")
    item_ids = [str(row.get("fixture_record_id")) for row in items]
    if item_ids != sorted(item_ids) or len(set(item_ids)) != len(item_ids):
        raise BioFixtureError("fixture records must be uniquely digest-sorted")
    coverage = Counter()
    cells_by_donor: dict[str, tuple[str, str, str]] = {}
    for row in items:
        _exact_keys(row, ITEM_KEYS, f"item {row.get('fixture_record_id')}")
        if set(row) & FORBIDDEN_LABEL_FIELDS:
            raise BioFixtureError("deposited label field entered the fixture")
        if row["schema_version"] != RECORD_SCHEMA:
            raise BioFixtureError("record schema changed")
        if row["firewall"] != FIREWALL or row["outcome_exposure"] != OUTCOME_EXPOSURE:
            raise BioFixtureError("record escaped the development firewall")
        readout_id = str(row["readout_id"])
        if readout_id not in READOUTS or any(
            row[key] != value for key, value in READOUTS[readout_id].items()
        ):
            raise BioFixtureError("record readout orientation changed")
        genes = str(row["gene_sentence"]).split()
        if len(genes) != 50 or len(set(genes)) != 50:
            raise BioFixtureError("record is not a distinct top-50 sentence")
        if row["gene_sentence_sha256"] != hashlib.sha256(
            row["gene_sentence"].encode("utf-8")
        ).hexdigest():
            raise BioFixtureError("record gene-sentence digest mismatch")
        record_without_id = dict(row)
        observed_record_id = record_without_id.pop("fixture_record_id")
        if observed_record_id != _canonical_sha256(record_without_id):
            raise BioFixtureError("fixture record identity mismatch")
        donor = str(row["donor_id"])
        identity = (
            str(row["source_cell_barcode"]),
            str(row["source_entity_id"]),
            str(row["gene_sentence_sha256"]),
        )
        if donor in cells_by_donor and cells_by_donor[donor] != identity:
            raise BioFixtureError("one donor spans multiple source cells")
        cells_by_donor[donor] = identity
        coverage[(donor, readout_id)] += 1
    expected_coverage = Counter(
        (donor, readout) for donor in EXPECTED_DONORS for readout in READOUTS
    )
    if coverage != expected_coverage or set(cells_by_donor) != set(EXPECTED_DONORS):
        raise BioFixtureError("fixture donor/readout coverage changed")
    return dict(fixture)


def assemble_fixture(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the label-free 16-row fixture from eight selected source projections."""

    if [str(row["donor_id"]) for row in selected] != list(EXPECTED_DONORS):
        raise BioFixtureError("selected cells must be ordered one per frozen donor")
    items = sorted(
        (
            _build_record(cell, readout)
            for cell in selected
            for readout in sorted(READOUTS)
        ),
        key=lambda row: row["fixture_record_id"],
    )
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "mode": "development",
        "outcome_exposure": OUTCOME_EXPOSURE,
        "firewall": FIREWALL,
        "donor_ids": list(EXPECTED_DONORS),
        "input_families": list(INPUT_FAMILIES),
        "readouts": READOUTS,
        "items": items,
        "model_calls_made_by_builder": 0,
    }
    return _validate_fixture(fixture)


def build_fixture_and_provenance() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the authenticated frame, exclusions, selection, and audit."""

    excluded_barcodes, excluded_entities, exclusions = _read_exclusions()
    ranked, source = _load_authenticated_frame()
    ranked_barcodes = {str(cell["barcode"]) for cell in ranked}
    ranked_entities = {
        f"gse96583:ctrl:{cell['donor_id']}:{cell['barcode']}" for cell in ranked
    }
    if not excluded_barcodes <= ranked_barcodes or not excluded_entities <= ranked_entities:
        raise BioFixtureError("an exclusion identity is absent from the renderable frame")
    selected, selection = _select_cells(
        ranked,
        excluded_barcodes,
        excluded_entities,
        EXPECTED_RAW_SHA256["matrix"],
        exclusions["exclusion_records_sha256"],
    )
    outcome_audit = _scan_qwen_result_registries(selected)
    fixture = assemble_fixture(selected)
    provenance = {
        "source": source,
        "exclusions": exclusions,
        "selection": selection,
        "outcome_exposure_audit": outcome_audit,
    }
    return fixture, provenance


def _validate_manifest(
    manifest: Mapping[str, Any], fixture: Mapping[str, Any], fixture_path: Path
) -> dict[str, Any]:
    _exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise BioFixtureError("manifest schema changed")
    if manifest["analysis_id"] != ANALYSIS_ID or manifest["mode"] != "development":
        raise BioFixtureError("manifest identity or mode changed")
    if manifest["status"] != "built_not_executed":
        raise BioFixtureError("manifest status changed")
    if manifest["outcome_exposure"] != OUTCOME_EXPOSURE:
        raise BioFixtureError("manifest outcome-exposure boundary changed")
    if manifest["firewall"] != FIREWALL:
        raise BioFixtureError("manifest firewall changed")
    if manifest["contract"]["model_calls_made_by_builder"] != 0:
        raise BioFixtureError("manifest claims model execution")
    if manifest["artifacts"]["builder_sha256"] != _file_sha256(Path(__file__)):
        raise BioFixtureError("manifest does not bind the builder")
    if manifest["artifacts"]["fixture_sha256"] != _file_sha256(fixture_path):
        raise BioFixtureError("manifest does not bind fixture bytes")
    if manifest["artifacts"]["fixture_canonical_sha256"] != _canonical_sha256(
        fixture
    ):
        raise BioFixtureError("manifest does not bind fixture content")
    return dict(manifest)


def build_manifest(
    fixture: Mapping[str, Any],
    provenance: Mapping[str, Any],
    fixture_path: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Build the frozen manifest for an on-disk validated fixture."""

    locked = _validate_fixture(fixture)
    if not fixture_path.is_file():
        raise BioFixtureError(f"fixture artifact does not exist: {fixture_path}")
    if json.loads(fixture_path.read_text(encoding="utf-8")) != locked:
        raise BioFixtureError("fixture path does not contain the supplied fixture")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "built_not_executed",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "outcome_exposure": OUTCOME_EXPOSURE,
        "claim_scope": (
            "qwen_readout_development_only_not_confirmation_biology_knowledge_"
            "activation_gap_or_physical_law"
        ),
        "firewall": FIREWALL,
        "label_boundary": {
            "source_labels_used_only_for_pre_existing_frame": True,
            "source_labels_used_for_selection": False,
            "source_labels_exported": False,
            "state_or_ground_truth_label_present": False,
            "permitted_interpretation": "readout_engineering_only",
        },
        "source": provenance["source"],
        "exclusions": provenance["exclusions"],
        "selection": provenance["selection"],
        "outcome_exposure_audit": provenance["outcome_exposure_audit"],
        "contract": {
            "donors": len(EXPECTED_DONORS),
            "source_cells": len(EXPECTED_DONORS),
            "readouts": READOUTS,
            "input_families": list(INPUT_FAMILIES),
            "items": len(locked["items"]),
            "model_calls_made_by_builder": 0,
            "confirmatory_eligibility": "prohibited",
        },
        "artifacts": {
            "builder_path": _relative_path(Path(__file__)),
            "builder_sha256": _file_sha256(Path(__file__)),
            "fixture_path": _relative_path(fixture_path),
            "fixture_sha256": _file_sha256(fixture_path),
            "fixture_canonical_sha256": _canonical_sha256(locked),
        },
    }
    return _validate_manifest(manifest, locked, fixture_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_and_write(
    *, out: Path = DEFAULT_OUT, manifest_out: Path = DEFAULT_MANIFEST
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write deterministic fixture and provenance artifacts without model execution."""

    if out.resolve() == manifest_out.resolve():
        raise BioFixtureError("fixture and manifest paths must be distinct")
    fixture, provenance = build_fixture_and_provenance()
    _write_json(out, fixture)
    manifest = build_manifest(fixture, provenance, out)
    _write_json(manifest_out, manifest)
    return fixture, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture, manifest = build_and_write(out=args.out, manifest_out=args.manifest)
    print(
        f"wrote {args.out} ({len(fixture['items'])} rows) and {args.manifest}; "
        f"outcome_exposure={manifest['outcome_exposure']}; model_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

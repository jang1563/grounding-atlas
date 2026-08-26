"""Build the outcome-exposed GSE96583 Level-0 development fixture.

This fixture exists only to engineer and falsify the coherent two-token readout.
It reuses the eight donor-explicit base cells from the completed GSE96583
sentinel experiment.  Consequently every item is outcome-exposed and is
permanently barred from confirmatory inference.  The only model input copied
from the source CSV is ``original_sentence``; legacy target/control masks and
generated model outputs never enter the fixture.

The first fixture release contains only the unmodified input family.  It crosses
each donor item with two frozen readouts: NK-versus-CD8 lineage and
cytotoxic-high-versus-cytotoxic-low state.  This builder makes no model calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ID = "gse96583-coherent-readout-development-fixture-v1"
FIXTURE_SCHEMA = "coherent-readout-development-fixture-v1"
RECORD_SCHEMA = "coherent-readout-development-item-v1"
MANIFEST_SCHEMA = "coherent-readout-development-fixture-manifest-v1"
FREEZE_DATE = "2026-08-02"

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
SOURCE_PROJECTION_FIELDS = (
    "row_type",
    "entity_id",
    "cell_barcode",
    "donor_id",
    "original_sentence",
)

DEFAULT_SOURCE_BUILDER = Path(__file__).with_name(
    "build_gse96583_sentinel_factorial.py"
)
DEFAULT_SOURCE_CSV = Path(__file__).with_name("gse96583_sentinel_factorial.csv")
DEFAULT_SOURCE_MANIFEST = Path(__file__).with_name(
    "gse96583_sentinel_factorial.manifest.json"
)
DEFAULT_PRIOR_RAW = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "sentinel_factorial"
    / "claude-haiku-4-5-20251001_raw.jsonl"
)
DEFAULT_PRIOR_RESULT = DEFAULT_PRIOR_RAW.with_name("claude-haiku-4-5-20251001.json")
DEFAULT_OUT = Path(__file__).with_name("coherent_readout_dev_fixture.json")
DEFAULT_MANIFEST = Path(__file__).with_name(
    "coherent_readout_dev_fixture.manifest.json"
)

EXPECTED_SOURCE_SHA256 = {
    "builder": "9336e633763b91c8d0c983d75b67004da9ee6f681c1b4f7dd2d4ba92b07f8992",
    "csv": "673db8c8bcd6ba923e62891de6cd5f04f97967706ac3ade8a6e44ad2d14a4b95",
    "manifest": "9f0035469c81852b11e2a36651b7892bf2dad4d30f8049b37cd1f655ca9bf0c4",
}
EXPECTED_PRIOR_OUTCOME_SHA256 = {
    "raw": "b625d8cf8f65e15863d19f8eb3b8300c8d0f84be6709eaac7edd94339dfede33",
    "result": "14d0f0c04b98c9e94650c6f966feb0f80ccf6877c612fee8c0d65110180e8df2",
}
EXPECTED_SOURCE_ANALYSIS_ID = "gse96583-sentinel-factorial-holdout-v1"
EXPECTED_PRIOR_MODEL = "claude-haiku-4-5-20251001"
EXPECTED_PRIOR_CALLS = 480

FIREWALL = {
    "biological_effect_inference": "forbidden",
    "confirmatory_eligibility": "prohibited",
    "partition": "development_only_outcome_exposed",
    "promotion_to_confirmation": "forbidden",
    "purpose": "level0_readout_engineering_only",
    "source_outcome_exposure": "prior_model_outputs_exist_for_every_source_item",
}


class CoherentReadoutFixtureError(ValueError):
    """Raised when a source or derived development fixture violates its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise CoherentReadoutFixtureError(f"missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise CoherentReadoutFixtureError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_source(
    source_builder: Path,
    source_csv: Path,
    source_manifest: Path,
    prior_raw: Path,
    prior_result: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    observed_source = {
        "builder": _verify_hash(
            source_builder,
            EXPECTED_SOURCE_SHA256["builder"],
            "source builder",
        ),
        "csv": _verify_hash(source_csv, EXPECTED_SOURCE_SHA256["csv"], "source CSV"),
        "manifest": _verify_hash(
            source_manifest,
            EXPECTED_SOURCE_SHA256["manifest"],
            "source manifest",
        ),
    }
    observed_outcomes = {
        "raw": _verify_hash(
            prior_raw,
            EXPECTED_PRIOR_OUTCOME_SHA256["raw"],
            "prior raw model output",
        ),
        "result": _verify_hash(
            prior_result,
            EXPECTED_PRIOR_OUTCOME_SHA256["result"],
            "prior analyzed result",
        ),
    }

    source_lock = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_lock.get("analysis_id") != EXPECTED_SOURCE_ANALYSIS_ID:
        raise CoherentReadoutFixtureError("source manifest analysis ID changed")
    source_artifacts = source_lock.get("artifacts", {})
    if source_artifacts.get("builder_sha256") != observed_source["builder"]:
        raise CoherentReadoutFixtureError("source manifest does not bind its builder")
    if source_artifacts.get("csv_sha256") != observed_source["csv"]:
        raise CoherentReadoutFixtureError("source manifest does not bind its CSV")

    prior = json.loads(prior_result.read_text(encoding="utf-8"))
    if prior.get("analysis_id") != EXPECTED_SOURCE_ANALYSIS_ID:
        raise CoherentReadoutFixtureError("prior result analysis ID changed")
    if prior.get("model") != EXPECTED_PRIOR_MODEL:
        raise CoherentReadoutFixtureError("prior result model changed")
    if prior.get("model_calls") != EXPECTED_PRIOR_CALLS:
        raise CoherentReadoutFixtureError("prior result call count changed")
    raw_line_count = sum(
        1
        for line in prior_raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if raw_line_count != EXPECTED_PRIOR_CALLS:
        raise CoherentReadoutFixtureError("prior raw model-output count changed")

    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise CoherentReadoutFixtureError(f"source row count changed: {len(rows)}")
    base_rows = [row for row in rows if row.get("row_type") == "base"]
    if len(base_rows) != len(EXPECTED_DONORS):
        raise CoherentReadoutFixtureError("source must contain exactly one base row per donor")
    donor_counts = Counter(row.get("donor_id") for row in base_rows)
    if donor_counts != Counter({donor: 1 for donor in EXPECTED_DONORS}):
        raise CoherentReadoutFixtureError(
            f"source donor identities or counts changed: {dict(donor_counts)}"
        )
    if len({row.get("entity_id") for row in base_rows}) != len(base_rows):
        raise CoherentReadoutFixtureError("source base entity IDs are not unique")
    return base_rows, observed_source, observed_outcomes


def _build_record(source_row: Mapping[str, str], readout_id: str) -> dict[str, Any]:
    if readout_id not in READOUTS:
        raise CoherentReadoutFixtureError(f"unknown readout: {readout_id}")
    donor_id = source_row["donor_id"]
    barcode = source_row["cell_barcode"]
    sentence = source_row["original_sentence"]
    genes = sentence.split()
    if len(genes) != 50 or len(set(genes)) != 50:
        raise CoherentReadoutFixtureError(
            f"{donor_id}/{barcode} does not contain 50 distinct ranked genes"
        )
    if "MASKED_GENE" in genes:
        raise CoherentReadoutFixtureError("legacy MASKED_GENE entered the fixture")
    if any(not gene.strip() for gene in genes):
        raise CoherentReadoutFixtureError("empty gene token entered the fixture")

    item_id = f"gse96583:dev:{donor_id}:{barcode}:unmodified"
    record = {
        "schema_version": RECORD_SCHEMA,
        "item_id": item_id,
        "donor_id": donor_id,
        "source_entity_id": source_row["entity_id"],
        "source_cell_barcode": barcode,
        "input_family": "unmodified",
        "readout_id": readout_id,
        **READOUTS[readout_id],
        "gene_sentence": sentence,
        "gene_sentence_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "source_projection_sha256": _canonical_sha256(
            {field: source_row[field] for field in SOURCE_PROJECTION_FIELDS}
        ),
        "firewall_partition": FIREWALL["partition"],
        "confirmatory_eligibility": FIREWALL["confirmatory_eligibility"],
        "source_outcome_exposure": FIREWALL["source_outcome_exposure"],
    }
    record["fixture_record_id"] = _canonical_sha256(record)
    return record


def _validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise CoherentReadoutFixtureError("fixture schema changed")
    if fixture.get("analysis_id") != ANALYSIS_ID or fixture.get("mode") != "development":
        raise CoherentReadoutFixtureError("fixture identity or mode changed")
    if fixture.get("firewall") != FIREWALL:
        raise CoherentReadoutFixtureError("development firewall changed")
    if fixture.get("donor_ids") != list(EXPECTED_DONORS):
        raise CoherentReadoutFixtureError("fixture donor order changed")
    if fixture.get("input_families") != list(INPUT_FAMILIES):
        raise CoherentReadoutFixtureError("fixture input families changed")
    if fixture.get("readouts") != READOUTS:
        raise CoherentReadoutFixtureError("fixture readout class text changed")

    items = fixture.get("items")
    if not isinstance(items, list) or len(items) != len(EXPECTED_DONORS) * len(READOUTS):
        raise CoherentReadoutFixtureError("fixture record count changed")
    expected_pairs = {
        (donor, readout)
        for donor in EXPECTED_DONORS
        for readout in sorted(READOUTS)
    }
    observed_pairs = {(row.get("donor_id"), row.get("readout_id")) for row in items}
    if observed_pairs != expected_pairs:
        raise CoherentReadoutFixtureError("fixture donor/readout coverage changed")
    if any(row.get("input_family") != "unmodified" for row in items):
        raise CoherentReadoutFixtureError("non-unmodified input entered v1")
    if any(row.get("firewall_partition") != FIREWALL["partition"] for row in items):
        raise CoherentReadoutFixtureError("record escaped the development firewall")
    if any(row.get("confirmatory_eligibility") != "prohibited" for row in items):
        raise CoherentReadoutFixtureError("record became confirmatory-eligible")

    item_donors: dict[str, str] = {}
    sentences_by_item: dict[str, set[str]] = {}
    for row in items:
        expected_classes = READOUTS.get(str(row.get("readout_id")))
        if expected_classes is None or any(
            row.get(key) != value for key, value in expected_classes.items()
        ):
            raise CoherentReadoutFixtureError("record readout class text changed")
        sentence = str(row.get("gene_sentence", ""))
        if len(sentence.split()) != 50 or "MASKED_GENE" in sentence.split():
            raise CoherentReadoutFixtureError("record is not an unmodified top-50 sentence")
        if row.get("gene_sentence_sha256") != hashlib.sha256(
            sentence.encode("utf-8")
        ).hexdigest():
            raise CoherentReadoutFixtureError("record sentence digest mismatch")
        record_without_id = dict(row)
        observed_id = record_without_id.pop("fixture_record_id", None)
        if observed_id != _canonical_sha256(record_without_id):
            raise CoherentReadoutFixtureError("fixture record identity mismatch")
        item_id = str(row.get("item_id"))
        donor = str(row.get("donor_id"))
        prior_donor = item_donors.setdefault(item_id, donor)
        if prior_donor != donor:
            raise CoherentReadoutFixtureError("one item ID spans multiple donors")
        sentences_by_item.setdefault(item_id, set()).add(sentence)
    if any(len(sentences) != 1 for sentences in sentences_by_item.values()):
        raise CoherentReadoutFixtureError("readouts do not share an item sentence")
    if len(item_donors) != len(EXPECTED_DONORS):
        raise CoherentReadoutFixtureError("fixture does not contain one item per donor")
    return dict(fixture)


def build_fixture(
    source_builder: Path = DEFAULT_SOURCE_BUILDER,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    prior_raw: Path = DEFAULT_PRIOR_RAW,
    prior_result: Path = DEFAULT_PRIOR_RESULT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the validated fixture and verified source provenance."""

    base_rows, observed_source, observed_outcomes = _read_source(
        source_builder,
        source_csv,
        source_manifest,
        prior_raw,
        prior_result,
    )
    rows_by_donor = {row["donor_id"]: row for row in base_rows}
    items = [
        _build_record(rows_by_donor[donor], readout)
        for donor in EXPECTED_DONORS
        for readout in sorted(READOUTS)
    ]
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "mode": "development",
        "firewall": FIREWALL,
        "donor_ids": list(EXPECTED_DONORS),
        "input_families": list(INPUT_FAMILIES),
        "readouts": READOUTS,
        "items": items,
    }
    provenance = {
        "source": observed_source,
        "prior_outcomes": observed_outcomes,
    }
    return _validate_fixture(fixture), provenance


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_and_write(
    *,
    source_builder: Path = DEFAULT_SOURCE_BUILDER,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    prior_raw: Path = DEFAULT_PRIOR_RAW,
    prior_result: Path = DEFAULT_PRIOR_RESULT,
    out: Path = DEFAULT_OUT,
    manifest_out: Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and write deterministic fixture and provenance manifest artifacts."""

    fixture, provenance = build_fixture(
        source_builder,
        source_csv,
        source_manifest,
        prior_raw,
        prior_result,
    )
    _write_json(out, fixture)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "built_not_executed",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "claim_scope": (
            "outcome_exposed_level0_readout_engineering_only_not_biology_"
            "knowledge_activation_gap_or_confirmation"
        ),
        "firewall": FIREWALL,
        "label_boundary": {
            "deposited_cell_type_label_copied": False,
            "ground_truth_label_in_fixture": False,
            "orthogonal_cytotoxic_state_label_present": False,
            "permitted_interpretation": "prompt_form_and_readout_coherence_only",
        },
        "source": {
            "geo_accession": "GSE96583",
            "source_analysis_id": EXPECTED_SOURCE_ANALYSIS_ID,
            "source_builder_path": _relative_path(source_builder),
            "source_builder_sha256": provenance["source"]["builder"],
            "source_csv_path": _relative_path(source_csv),
            "source_csv_sha256": provenance["source"]["csv"],
            "source_manifest_path": _relative_path(source_manifest),
            "source_manifest_sha256": provenance["source"]["manifest"],
            "source_rows": "eight base rows; exactly one per explicit donor",
            "source_projection_fields_used": list(SOURCE_PROJECTION_FIELDS),
            "source_field_used_for_model_input": "original_sentence",
            "source_fields_forbidden_from_model_input": [
                "control_mask_sentence",
                "module_mask_sentence",
            ],
        },
        "outcome_exposure": {
            "status": "outcome_exposed",
            "prior_model": EXPECTED_PRIOR_MODEL,
            "prior_model_calls": EXPECTED_PRIOR_CALLS,
            "prior_raw_path": _relative_path(prior_raw),
            "prior_raw_sha256": provenance["prior_outcomes"]["raw"],
            "prior_result_path": _relative_path(prior_result),
            "prior_result_sha256": provenance["prior_outcomes"]["result"],
            "interpretation": (
                "all source cells were previously evaluated; no record may be "
                "re-labeled as held-out or promoted into confirmation"
            ),
        },
        "contract": {
            "donor_ids": list(EXPECTED_DONORS),
            "explicit_donor_count": len(EXPECTED_DONORS),
            "input_families": list(INPUT_FAMILIES),
            "readouts": READOUTS,
            "items": len(fixture["items"]),
            "unique_items": len(EXPECTED_DONORS),
            "model_calls_made_by_builder": 0,
        },
        "artifacts": {
            "builder_path": _relative_path(Path(__file__)),
            "builder_sha256": _sha256(Path(__file__)),
            "fixture_path": _relative_path(out),
            "fixture_sha256": _sha256(out),
        },
    }
    _write_json(manifest_out, manifest)
    return fixture, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-builder", type=Path, default=DEFAULT_SOURCE_BUILDER)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--prior-raw", type=Path, default=DEFAULT_PRIOR_RAW)
    parser.add_argument("--prior-result", type=Path, default=DEFAULT_PRIOR_RESULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture, manifest = build_and_write(
        source_builder=args.source_builder,
        source_csv=args.source_csv,
        source_manifest=args.source_manifest,
        prior_raw=args.prior_raw,
        prior_result=args.prior_result,
        out=args.out,
        manifest_out=args.manifest,
    )
    print(
        f"wrote {args.out} ({len(fixture['items'])} items) and "
        f"{args.manifest}; firewall={manifest['firewall']['partition']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

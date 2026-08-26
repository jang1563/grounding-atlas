from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "signal" / "dms" / "mavedb_candidate_registry.v2.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
URN_PATTERN = re.compile(r"^urn:mavedb:[0-9]{8}-(?:[a-z]+|0)-[0-9]+$")
EXPECTED_CANONICAL_SHA256 = "be3b3aef32ae2c5c62db58ba2a64015c71848f2702257abc3092db1b4a1b63cd"
EXPECTED_COUNT_LOCKS = {
    "urn:mavedb:00000005-a-5",
    "urn:mavedb:00000097-0-2",
}
CANONICAL_AMINO_ACIDS_THREE_LETTER = [
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
]
EXPECTED_MISSENSE_EXCEPTIONS = [
    {
        "coordinate_out_of_bounds_count": 0,
        "reference_residue_mismatch_count": 0,
        "reference_ter_excluded_count": 20,
        "strict_unique_missense_count": 12464,
        "syntactic_ref_not_alt_non_stop_gain_count": 12484,
        "urn": "urn:mavedb:00000049-a-6",
    },
    {
        "coordinate_out_of_bounds_count": 796,
        "reference_residue_mismatch_count": 1207,
        "reference_ter_excluded_count": 0,
        "strict_unique_missense_count": 2132,
        "syntactic_ref_not_alt_non_stop_gain_count": 4135,
        "urn": "urn:mavedb:00001251-a-1",
    },
]
EXPECTED_NO_TARGET_HGVS_URNS = {
    "urn:mavedb:00000097-0-2",
    "urn:mavedb:00001225-a-1",
    "urn:mavedb:00001259-a-2",
}
EXPECTED_ZERO_EXCEPTION_URNS = {
    "urn:mavedb:00000005-a-5",
    "urn:mavedb:00000013-b-1",
    "urn:mavedb:00000050-a-1",
    "urn:mavedb:00000054-a-1",
    "urn:mavedb:00000055-b-1",
    "urn:mavedb:00000059-a-1",
    "urn:mavedb:00000094-a-2",
    "urn:mavedb:00000095-a-1",
    "urn:mavedb:00000096-a-1",
    "urn:mavedb:00000108-a-1",
    "urn:mavedb:00001205-a-1",
    "urn:mavedb:00001216-a-1",
    "urn:mavedb:00001266-a-1",
    "urn:mavedb:00001269-b-1",
    "urn:mavedb:00001275-a-1",
}


def _load_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_registry_identity_and_candidate_counts_are_frozen() -> None:
    registry = _load_registry()

    assert set(registry) == {
        "access_date",
        "api",
        "artifact_type",
        "body_hash_basis",
        "candidate_counts",
        "confirmatory_eligible",
        "family_disjointness_statement",
        "missense_count_definition",
        "missense_validation",
        "official_documentation",
        "record_order",
        "records",
        "schema_version",
    }
    assert registry["schema_version"] == 2
    assert registry["artifact_type"] == "groundbench.dms_mavedb_candidate_registry"
    assert registry["access_date"] == "2026-07-25"
    assert registry["record_order"] == "score_set_urn_lexicographic"
    assert registry["confirmatory_eligible"] is False
    assert registry["candidate_counts"] == {
        "conditional": 6,
        "core": 14,
        "total": 20,
    }
    assert _canonical_sha256(registry) == EXPECTED_CANONICAL_SHA256


def test_registry_records_remain_candidate_only_and_urn_sorted() -> None:
    registry = _load_registry()
    records = registry["records"]
    urns = [record["urn"] for record in records]

    assert len(records) == 20
    assert urns == sorted(urns)
    assert len(set(urns)) == len(urns)
    assert all(URN_PATTERN.fullmatch(urn) for urn in urns)
    assert sum(record["candidate_tier"] == "core" for record in records) == 14
    assert sum(record["candidate_tier"] == "conditional" for record in records) == 6

    for record in records:
        assert record["admission_status"] == "candidate_not_ingested"
        assert record["blockers"]
        assert record["provisional_family"]["evidence_status"] == ("inferred_not_release_group")
        assert record["provisional_family"]["label"]
        assert record["score_orientation"]["evidence"]
        assert record["score_orientation"]["evidence_status"]
        assert record["score_orientation"]["retained_function_transform"]
        assert record["score_row_count"] > 0
        assert 0 <= record["unique_missense_hgvs_pro_count"] <= record["score_row_count"]
        assert record["license"]["short_name"] == "CC0"
        assert record["license"]["url"].startswith("https://")


def test_strict_missense_validation_and_exceptions_are_frozen() -> None:
    registry = _load_registry()
    records_by_urn = {record["urn"]: record for record in registry["records"]}
    validation = registry["missense_validation"]

    assert validation["canonical_amino_acids_three_letter"] == (CANONICAL_AMINO_ACIDS_THREE_LETTER)
    assert validation["coordinate_base"] == 1
    assert validation["dna_translation_table"] == 1
    assert validation["hgvs_pro_pattern"] == (r"^p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2})$")
    assert "in-range coordinate" in validation["target_sequence_policy"]
    assert "source-reference-residue match" in validation["target_sequence_policy"]
    assert validation["exception_summaries"] == EXPECTED_MISSENSE_EXCEPTIONS

    exception_urns = {item["urn"] for item in validation["exception_summaries"]}
    no_target_urns = set(validation["no_hgvs_pro_target_unavailable_urns"])
    zero_exception_urns = set(validation["validated_zero_exception_urns"])
    all_urns = set(records_by_urn)
    assert no_target_urns == EXPECTED_NO_TARGET_HGVS_URNS
    assert zero_exception_urns == EXPECTED_ZERO_EXCEPTION_URNS
    assert exception_urns.isdisjoint(no_target_urns)
    assert exception_urns.isdisjoint(zero_exception_urns)
    assert no_target_urns.isdisjoint(zero_exception_urns)
    assert exception_urns | no_target_urns | zero_exception_urns == all_urns

    for summary in validation["exception_summaries"]:
        excluded = (
            summary["reference_ter_excluded_count"]
            + summary["coordinate_out_of_bounds_count"]
            + summary["reference_residue_mismatch_count"]
        )
        assert summary["strict_unique_missense_count"] == (
            summary["syntactic_ref_not_alt_non_stop_gain_count"] - excluded
        )
        assert (
            records_by_urn[summary["urn"]]["unique_missense_hgvs_pro_count"] == summary["strict_unique_missense_count"]
        )

    assert records_by_urn["urn:mavedb:00000049-a-6"]["unique_missense_hgvs_pro_count"] == 12464
    for urn in no_target_urns:
        assert records_by_urn[urn]["unique_missense_hgvs_pro_count"] == 0


def test_identity_and_count_lineage_ceiling_are_fail_closed() -> None:
    records_by_urn = {record["urn"]: record for record in _load_registry()["records"]}
    count_lineage_urns = {
        "urn:mavedb:00000005-a-5",
        "urn:mavedb:00000097-0-2",
    }

    for urn in count_lineage_urns:
        record = records_by_urn[urn]
        assert record["deposited_data_availability"] == (
            "substantive_counts_and_replicate_lineage_deposited_replay_unresolved"
        )
        assert record["source_readiness_ceiling"] == "COUNT_LINEAGE_PARTIAL"
        assert "replicate_count_readiness" not in record
        blockers = " ".join(record["blockers"])
        assert "no explicit WT observation" in blockers
        assert "replay" in blockers

    cbs_blockers = " ".join(records_by_urn["urn:mavedb:00000005-a-5"]["blockers"])
    assert "TileSeq control-count subtraction" in cbs_blockers
    brca1_blockers = " ".join(records_by_urn["urn:mavedb:00000097-0-2"]["blockers"])
    assert "26 child score sets" in brca1_blockers
    assert "LOESS correction" in brca1_blockers

    tpk1 = records_by_urn["urn:mavedb:00001251-a-1"]
    assert tpk1["candidate_tier"] == "conditional"
    assert tpk1["source_readiness_ceiling"] == "IDENTITY_BLOCKED"
    assert tpk1["unique_missense_hgvs_pro_count"] == 2132
    tpk1_blockers = " ".join(tpk1["blockers"])
    assert "796 syntactic missense strings are out of bounds" in tpk1_blockers
    assert "1207 source reference residues disagree" in tpk1_blockers
    assert "construct-to-Q9H3S4 mapping" in tpk1_blockers

    assert {
        record["urn"] for record in records_by_urn.values() if "source_readiness_ceiling" in record
    } == count_lineage_urns | {"urn:mavedb:00001251-a-1"}


def test_exact_body_locks_fail_closed_when_counts_are_unverified() -> None:
    records = _load_registry()["records"]

    for record in records:
        urn = record["urn"]
        for artifact_name in ("metadata", "scores"):
            artifact = record[artifact_name]
            assert artifact["body_verified"] is True
            assert artifact["body_bytes"] > 0
            assert SHA256_PATTERN.fullmatch(artifact["body_sha256"])
            assert artifact["url"].startswith("https://api.mavedb.org/api/v1/")

        counts = record["counts"]
        assert counts["url"].startswith("https://api.mavedb.org/api/v1/")
        if urn in EXPECTED_COUNT_LOCKS:
            assert counts["body_verified"] is True
            assert counts["body_bytes"] > 0
            assert SHA256_PATTERN.fullmatch(counts["body_sha256"])
            assert counts["metadata_count_columns"]
        else:
            assert counts["body_verified"] is False
            assert counts["body_bytes"] is None
            assert counts["body_sha256"] is None
            assert any("counts response body was not locked" in item for item in record["blockers"])

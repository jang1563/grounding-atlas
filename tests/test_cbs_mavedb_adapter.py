from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from eval import cbs_mavedb_adapter as cbs
from eval import mavedb_source_lock as msl

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "signal" / "dms" / "mavedb_candidate_registry.v2.json"
STATUS_PATH = ROOT / "signal" / "dms" / "cbs_adapter_status.v1.json"

METADATA_URL = f"{msl.DEFAULT_API_BASE_URL}/score-sets/{cbs.CBS_URN}"
SCORES_URL = f"{METADATA_URL}/scores"
COUNTS_URL = f"{METADATA_URL}/counts"
MAPPED_URL = f"{METADATA_URL}/mapped-variants"


class MockTransport:
    def __init__(self, responses):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls = defaultdict(int)

    def __call__(self, url, headers):
        del headers
        self.calls[url] += 1
        if url not in self.responses or not self.responses[url]:
            raise AssertionError(f"unexpected transport URL: {url}")
        return self.responses[url][0]


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _response(body, content_type):
    if isinstance(body, str):
        body = body.encode()
    return msl.HttpResponse(
        status=200,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
        body=body,
    )


def _production_registry():
    return json.loads(REGISTRY_PATH.read_text())


def _cbs_record(registry):
    return next(record for record in registry["records"] if record["urn"] == cbs.CBS_URN)


def _metadata():
    return {
        "urn": cbs.CBS_URN,
        "title": cbs.CBS_TITLE,
        "numVariants": 2,
        "private": False,
        "processingState": "success",
        "supersededScoreSet": None,
        "supersedingScoreSet": None,
        "license": {
            "id": 1,
            "shortName": "CC0",
            "longName": "CC0 (Public domain)",
            "active": True,
            "link": "https://creativecommons.org/publicdomain/zero/1.0/",
            "version": "1.0",
            "recordType": "ShortLicense",
        },
        "dataUsagePolicy": None,
        "datasetColumns": {
            "scoreColumns": ["score", "sd", "se"],
            "countColumns": list(cbs.EXPECTED_COUNT_COLUMNS),
            "recordType": "DatasetColumns",
        },
        "scoreCalibrations": [],
        "targetGenes": [
            {
                "name": cbs.CBS_GENE,
                "mappedHgncName": cbs.CBS_GENE,
                "externalIdentifiers": [
                    {
                        "identifier": "P35520",
                        "dbName": "UniProt",
                        "offset": 0,
                    }
                ],
                "targetSequence": {
                    "sequenceType": "dna",
                    "sequence": "ATG" * 552,
                    "recordType": "TargetSequence",
                },
                "targetAccession": None,
            }
        ],
    }


def _scores():
    header = [*msl.FIXED_VARIANT_COLUMNS, "score", "sd", "se"]
    rows = [
        [f"{cbs.CBS_URN}#1", "c.1A>G", "NA", "p.Met1Val", "0.8", "0.1", "0.05"],
        [f"{cbs.CBS_URN}#2", "c.4A>G", "NA", "p.Lys2Glu", "0.2", "0.2", "0.10"],
    ]
    return ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()


def _counts():
    header = [*msl.FIXED_VARIANT_COLUMNS, *cbs.EXPECTED_COUNT_COLUMNS]
    rows = []
    for index in range(2):
        fixed = [
            f"{cbs.CBS_URN}#{index + 1}",
            f"c.{1 + index * 3}A>G",
            "NA",
            ("p.Met1Val", "p.Lys2Glu")[index],
        ]
        values = [f"{(10 + index + column_index) / 10:.1f}" for column_index in range(32)]
        rows.append(fixed + values)
    rows[1][4] = "NA"
    return ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()


def _mappings():
    return [
        {
            "variantUrn": f"{cbs.CBS_URN}#1",
            "current": True,
            "preMapped": {"id": "pre-1"},
            "postMapped": {"id": "post-1"},
            "errorMessage": None,
            "alignmentLevel": "protein",
            "mappingApiVersion": "2026.2.1",
            "vrsVersion": "2",
            "atMismatchedLocus": False,
            "nearGap": False,
        },
        {
            "variantUrn": f"{cbs.CBS_URN}#2",
            "current": True,
            "preMapped": {"id": "pre-2"},
            "postMapped": {"id": "post-2"},
            "errorMessage": None,
            "alignmentLevel": "protein",
            "mappingApiVersion": "2026.2.1",
            "vrsVersion": "2",
            "atMismatchedLocus": False,
            "nearGap": False,
        },
    ]


def _refresh_source_lock(lock):
    lock["metadata_tabular_license_binding_sha256"] = msl.canonical_sha256(msl._license_binding_payload(lock))
    lock["source_bundle_sha256"] = msl.canonical_sha256(msl._source_bundle_payload(lock))
    return lock


def _bind_source_body(lock, name, body):
    artifact = lock["source_artifacts"][name]
    digest = hashlib.sha256(body).hexdigest()
    artifact["declared_content_length"] = len(body)
    artifact["wire_byte_count"] = len(body)
    artifact["wire_sha256"] = digest
    artifact["decoded_byte_count"] = len(body)
    artifact["sha256"] = digest
    return _refresh_source_lock(lock)


def _bind_registry_body(registry, name, body, monkeypatch):
    record = _cbs_record(registry)
    body_lock = record[name]
    body_lock["body_bytes"] = len(body)
    body_lock["body_sha256"] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CBS_RECORD_SHA256",
        cbs.canonical_sha256(record),
    )
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CANDIDATE_REGISTRY_SHA256",
        cbs.canonical_sha256(registry),
    )
    return registry


def _synthetic_inputs(monkeypatch):
    metadata_body = _json_bytes(_metadata())
    scores_body = _scores()
    counts_body = _counts()
    mapped_body = _json_bytes(_mappings())
    responses = {
        msl.DEFAULT_OPENAPI_URL: [
            _response(
                _json_bytes({"openapi": "3.1.0", "info": {"version": "2026.2.7"}}),
                "application/json",
            )
        ],
        METADATA_URL: [_response(metadata_body, "application/json")],
        SCORES_URL: [_response(scores_body, "text/csv; charset=utf-8")],
        COUNTS_URL: [_response(counts_body, "text/csv; charset=utf-8")],
        MAPPED_URL: [_response(mapped_body, "application/json")],
    }
    config = msl.SourceLockConfig(
        urn=cbs.CBS_URN,
        readiness=msl.Readiness.COUNT_LINEAGE_PARTIAL,
        readiness_evidence=msl.ReadinessEvidence(
            aggregate_score_column="score",
            count_lineage_columns=cbs.EXPECTED_COUNT_COLUMNS,
        ),
        expected_api_version="2026.2.7",
    )
    lock = msl.build_candidate_source_lock(
        config,
        transport=MockTransport(responses),
    )

    registry = deepcopy(_production_registry())
    record = _cbs_record(registry)
    record["score_row_count"] = 2
    record["unique_missense_hgvs_pro_count"] = 2
    for name, body in (
        ("metadata", metadata_body),
        ("scores", scores_body),
        ("counts", counts_body),
    ):
        record[name]["body_bytes"] = len(body)
        record[name]["body_sha256"] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(cbs, "CBS_SCORE_ROW_COUNT", 2)
    monkeypatch.setattr(cbs, "CBS_UNIQUE_MISSENSE_COUNT", 2)
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CBS_RECORD_SHA256",
        cbs.canonical_sha256(record),
    )
    monkeypatch.setattr(cbs, "EXPECTED_METADATA_BODY_BYTES", len(metadata_body))
    monkeypatch.setattr(
        cbs,
        "EXPECTED_METADATA_BODY_SHA256",
        hashlib.sha256(metadata_body).hexdigest(),
    )
    monkeypatch.setattr(cbs, "EXPECTED_SCORES_BODY_BYTES", len(scores_body))
    monkeypatch.setattr(
        cbs,
        "EXPECTED_SCORES_BODY_SHA256",
        hashlib.sha256(scores_body).hexdigest(),
    )
    monkeypatch.setattr(cbs, "EXPECTED_COUNTS_BODY_BYTES", len(counts_body))
    monkeypatch.setattr(
        cbs,
        "EXPECTED_COUNTS_BODY_SHA256",
        hashlib.sha256(counts_body).hexdigest(),
    )
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CANDIDATE_REGISTRY_SHA256",
        cbs.canonical_sha256(registry),
    )
    return cbs.CbsLowB6AdapterInputs(
        candidate_registry=registry,
        materialized_source_lock=lock,
        metadata_body=metadata_body,
        scores_body=scores_body,
        counts_body=counts_body,
        mapped_variants_body=mapped_body,
    )


def _production_inputs():
    return cbs.CbsLowB6AdapterInputs(candidate_registry=_production_registry())


def test_production_registry_builds_candidate_only_status():
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())

    assert status["source_readiness"] == "COUNT_LINEAGE_PARTIAL"
    assert status["ingestion_status"] == "not_ingested"
    assert status["outcome_status"] == "not_derived"
    assert status["confirmatory_eligible"] is False
    assert status["automatic_promotion"] is False
    assert status["offline_validation"]["decoded_source_bodies_status"] == "not_supplied"
    assert cbs.SOURCE_BUNDLE_BLOCKER in status["active_blocker_codes"]
    assert status["schema_2_compatibility"]["compatible"] is False
    assert status["count_measurement_semantics"] == {
        "measurement_scale": "relative_read_frequency_per_1M_total_reads",
        "value_contract": "finite_nonnegative_decimal_or_explicit_missing",
        "missing_tokens": ["NA"],
        "raw_read_count_claim": False,
        "controlNS_controlS_role": cbs.CONTROL_CHANNEL_ROLE,
        "functional_wt_baseline_claim": False,
    }
    evidence = status["external_scientific_status_evidence"]
    assert evidence["raw_read_counts_lowB6_workbook_row_count"] == 22_536
    assert evidence["experimental_lowB6_workbook_row_count"] == 11_478
    assert evidence["published_functional_classification_rule"] == {
        "classification": "deleterious_if_upper_95_percent_ci_below_threshold",
        "upper_95_percent_ci_threshold": 0.60,
        "fdr": 0.05,
        "complement_classification": "not_defined_as_neutral_or_retained_function",
    }
    assert evidence["published_b6_remediability_rule"] == {
        "eligibility": "classified_deleterious_under_low_b6_rule",
        "classification": ("remediable_if_lower_95_percent_ci_of_high_minus_low_above_threshold"),
        "lower_95_percent_ci_threshold": 0.22,
        "fdr": 0.05,
    }
    assert evidence["published_well_measured_rule"] == {
        "preselection_allele_frequency_operator": ">",
        "preselection_allele_frequency_percent_threshold": 0.005,
        "standard_error_operator": "<",
        "standard_error_threshold": 0.2,
    }
    assert evidence["score_uncertainty_channel_evidence"] == {
        "low_b6_sd_se_squared_ratio": 8,
        "high_b6_sd_se_squared_ratio": 4,
        "interpretation": ("consistent_with_eight_low_b6_and_four_high_b6_observation_channels"),
        "replicate_count_claim": False,
        "evidence_scope": "external_scientific_status_only_sample_map_missing",
    }
    assert status["status_sha256"] == cbs.adapter_status_sha256(status)


def test_persisted_status_is_exact_deterministic_build():
    persisted = json.loads(STATUS_PATH.read_text())
    built = cbs.build_cbs_low_b6_adapter_status(_production_inputs())

    assert persisted == built
    assert STATUS_PATH.read_bytes() == cbs.canonical_json_bytes(built) + b"\n"
    cbs.validate_cbs_adapter_status(persisted)


def test_registry_hash_tamper_is_rejected_before_semantic_use():
    registry = _production_registry()
    _cbs_record(registry)["gene"] = "CBS2"

    with pytest.raises(cbs.CbsAdapterError, match="registry SHA-256"):
        cbs.validate_cbs_candidate_registry(registry)


def test_registry_semantic_tamper_is_rejected_even_if_rehashed(monkeypatch):
    registry = _production_registry()
    record = _cbs_record(registry)
    record["gene"] = "CBS2"
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CANDIDATE_REGISTRY_SHA256",
        cbs.canonical_sha256(registry),
    )
    monkeypatch.setattr(
        cbs,
        "EXPECTED_CBS_RECORD_SHA256",
        cbs.canonical_sha256(record),
    )

    with pytest.raises(cbs.CbsAdapterError, match=r"record\.gene differs"):
        cbs.validate_cbs_candidate_registry(registry)


def test_complete_synthetic_source_bundle_validates_exact_lineage(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    status = cbs.build_cbs_low_b6_adapter_status(inputs)
    offline = status["offline_validation"]

    assert offline["materialized_source_lock_status"] == "validated_structural_only"
    assert offline["decoded_source_bodies_status"] == "validated_structural_only"
    assert offline["mapped_variants_authentication_status"] == "unanchored_structural_validation_only"
    assert offline["openapi_authentication_status"] == "unanchored_source_lock_summary_only"
    assert offline["score_columns"] == ["score", "sd", "se"]
    assert offline["count_columns"] == list(cbs.EXPECTED_COUNT_COLUMNS)
    assert offline["score_row_count"] == 2
    assert offline["count_row_count"] == 2
    assert offline["score_count_accession_join_status"] == "validated_exact_order"
    assert offline["mapped_current_accession_join_status"] == "validated_exact_set"
    assert offline["nonmissing_count_cell_count"] == 63
    assert offline["missing_count_cell_count"] == 1
    assert cbs.SOURCE_BUNDLE_BLOCKER in status["active_blocker_codes"]
    assert status["source_readiness"] == "COUNT_LINEAGE_PARTIAL"
    assert status["ingestion_status"] == "not_ingested"
    assert status["outcome_status"] == "not_derived"


def test_optional_source_bundle_is_all_or_none(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    incomplete = replace(inputs, mapped_variants_body=None)

    with pytest.raises(cbs.CbsAdapterError, match="must be supplied together"):
        cbs.build_cbs_low_b6_adapter_status(incomplete)


def test_unbound_body_tamper_is_rejected(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    tampered = replace(inputs, scores_body=inputs.scores_body + b"\n")

    with pytest.raises(cbs.CbsAdapterError, match="materialized source lock"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_registry_cross_binding_rejects_coherently_relocked_body(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    counts_body = inputs.counts_body + b"\n"
    _bind_source_body(lock, "counts", counts_body)
    tampered = replace(
        inputs,
        materialized_source_lock=lock,
        counts_body=counts_body,
    )

    with pytest.raises(cbs.CbsAdapterError, match="frozen candidate registry"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_score_count_accession_order_mismatch_is_rejected(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    registry = deepcopy(inputs.candidate_registry)
    lock = deepcopy(inputs.materialized_source_lock)
    counts_body = inputs.counts_body.replace(
        f"{cbs.CBS_URN}#2".encode(),
        f"{cbs.CBS_URN}#3".encode(),
        1,
    )
    _bind_source_body(lock, "counts", counts_body)
    _bind_registry_body(registry, "counts", counts_body, monkeypatch)
    tampered = replace(
        inputs,
        candidate_registry=registry,
        materialized_source_lock=lock,
        counts_body=counts_body,
    )

    with pytest.raises(cbs.CbsAdapterError, match="accession order differs"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_mapped_current_accession_mismatch_is_rejected(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    mapped_body = inputs.mapped_variants_body.replace(
        f"{cbs.CBS_URN}#2".encode(),
        f"{cbs.CBS_URN}#3".encode(),
        1,
    )
    _bind_source_body(lock, "mapped_variants", mapped_body)
    tampered = replace(
        inputs,
        materialized_source_lock=lock,
        mapped_variants_body=mapped_body,
    )

    with pytest.raises(cbs.CbsAdapterError, match="mapped-variant accessions differ"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_coherently_relocked_mapping_remains_explicitly_unauthenticated(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    mapped = json.loads(inputs.mapped_variants_body)
    mapped[0]["mappingApiVersion"] = "forged-but-structurally-valid"
    mapped_body = _json_bytes(mapped)
    _bind_source_body(lock, "mapped_variants", mapped_body)
    lock["mapping_contract"] = cbs._mapping_summary(
        mapped_body,
        expected_accessions=[
            f"{cbs.CBS_URN}#1",
            f"{cbs.CBS_URN}#2",
        ],
    )
    _refresh_source_lock(lock)
    tampered = replace(
        inputs,
        materialized_source_lock=lock,
        mapped_variants_body=mapped_body,
    )

    status = cbs.build_cbs_low_b6_adapter_status(tampered)

    assert (
        status["offline_validation"]["mapped_variants_authentication_status"] == "unanchored_structural_validation_only"
    )
    assert "CBS_MAPPED_VARIANTS_BODY_UNAUTHENTICATED" in status["active_blocker_codes"]
    assert cbs.SOURCE_BUNDLE_BLOCKER in status["active_blocker_codes"]
    assert status["confirmatory_eligible"] is False
    assert status["outcome_status"] == "not_derived"


@pytest.mark.parametrize(
    "bad_value",
    ["-1", "+1", "inf", "NaN", "na", "None", "null", "", "1 ", " 1", "--1"],
)
def test_count_cells_require_finite_nonnegative_decimal_or_exact_missing(
    monkeypatch,
    bad_value,
):
    inputs = _synthetic_inputs(monkeypatch)
    registry = deepcopy(inputs.candidate_registry)
    lock = deepcopy(inputs.materialized_source_lock)
    counts_body = inputs.counts_body.replace(b",1.0,", f",{bad_value},".encode(), 1)
    _bind_source_body(lock, "counts", counts_body)
    _bind_registry_body(registry, "counts", counts_body, monkeypatch)
    tampered = replace(
        inputs,
        candidate_registry=registry,
        materialized_source_lock=lock,
        counts_body=counts_body,
    )

    with pytest.raises(
        cbs.CbsAdapterError,
        match="finite nonnegative decimal or an exact missing token",
    ):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_count_header_permutation_is_rejected(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    registry = deepcopy(inputs.candidate_registry)
    lock = deepcopy(inputs.materialized_source_lock)
    header, remainder = inputs.counts_body.split(b"\n", 1)
    columns = header.split(b",")
    columns[-1], columns[-2] = columns[-2], columns[-1]
    counts_body = b",".join(columns) + b"\n" + remainder
    _bind_source_body(lock, "counts", counts_body)
    _bind_registry_body(registry, "counts", counts_body, monkeypatch)
    tampered = replace(
        inputs,
        candidate_registry=registry,
        materialized_source_lock=lock,
        counts_body=counts_body,
    )

    with pytest.raises(cbs.CbsAdapterError, match="header differs from the exact schema"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_truncated_csv_row_is_rejected(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    registry = deepcopy(inputs.candidate_registry)
    lock = deepcopy(inputs.materialized_source_lock)
    lines = inputs.counts_body.decode().splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0]
    counts_body = ("\n".join(lines) + "\n").encode()
    _bind_source_body(lock, "counts", counts_body)
    _bind_registry_body(registry, "counts", counts_body, monkeypatch)
    tampered = replace(
        inputs,
        candidate_registry=registry,
        materialized_source_lock=lock,
        counts_body=counts_body,
    )

    with pytest.raises(cbs.CbsAdapterError, match="truncated or overlong"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_source_lock_cannot_claim_a_different_readiness(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    lock["readiness"]["state"] = "AGGREGATE_ONLY"
    _refresh_source_lock(lock)
    tampered = replace(inputs, materialized_source_lock=lock)

    with pytest.raises(cbs.CbsAdapterError, match="COUNT_LINEAGE_PARTIAL"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_source_lock_cannot_substitute_another_aggregate_score_column(monkeypatch):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    lock["readiness"]["evidence"]["aggregate_score_column"] = "sd"
    _refresh_source_lock(lock)
    tampered = replace(inputs, materialized_source_lock=lock)

    with pytest.raises(cbs.CbsAdapterError, match="score/count readiness evidence"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("processed_replicate_columns", ["sd", "se"]),
        ("transformation_spec_sha256", "0" * 64),
        ("wt_control_spec_sha256", "1" * 64),
        ("identity_resolution_spec_sha256", "2" * 64),
        ("confirmatory_preregistration_sha256", "3" * 64),
        ("independent_replication_spec_sha256", "4" * 64),
        ("identity_block_reason", "not applicable but forged"),
        ("rejection_reason", "not applicable but forged"),
    ],
)
def test_source_lock_cannot_claim_unfrozen_cbs_specs_or_replicates(
    monkeypatch,
    field,
    value,
):
    inputs = _synthetic_inputs(monkeypatch)
    lock = deepcopy(inputs.materialized_source_lock)
    lock["readiness"]["evidence"][field] = value
    _refresh_source_lock(lock)
    tampered = replace(inputs, materialized_source_lock=lock)

    with pytest.raises(cbs.CbsAdapterError, match="score/count readiness evidence"):
        cbs.build_cbs_low_b6_adapter_status(tampered)


def test_status_self_hash_and_registry_hash_are_verified():
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    status["active_blocker_codes"] = status["active_blocker_codes"][:-1]

    with pytest.raises(cbs.CbsAdapterError):
        cbs.validate_cbs_adapter_status(status)

    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    status["candidate_registry_sha256"] = "0" * 64
    status["status_sha256"] = cbs.adapter_status_sha256(status)
    with pytest.raises(cbs.CbsAdapterError, match="candidate registry hash differs"):
        cbs.validate_cbs_adapter_status(status)


def test_status_candidate_record_and_registry_expectations_resist_coherent_rehash():
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    status["candidate_registry_record_sha256"] = "0" * 64
    status["status_sha256"] = cbs.adapter_status_sha256(status)
    with pytest.raises(cbs.CbsAdapterError, match="candidate record hash differs"):
        cbs.validate_cbs_adapter_status(status)

    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    status["registry_expectations"]["counts"]["body_bytes"] += 1
    status["status_sha256"] = cbs.adapter_status_sha256(status)
    with pytest.raises(cbs.CbsAdapterError, match="expectation for counts differs"):
        cbs.validate_cbs_adapter_status(status)


def test_status_replicate_count_claim_requires_boolean_false():
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    status["external_scientific_status_evidence"]["score_uncertainty_channel_evidence"]["replicate_count_claim"] = 0
    status["status_sha256"] = cbs.adapter_status_sha256(status)

    with pytest.raises(cbs.CbsAdapterError, match="replicate_count_claim must be false"):
        cbs.validate_cbs_adapter_status(status)


def test_standalone_status_cannot_forge_validated_source_bundle():
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    offline = status["offline_validation"]
    offline.update(
        {
            "materialized_source_lock_status": "validated_structural_only",
            "decoded_source_bodies_status": "validated_structural_only",
            "source_lock_sha256": "1" * 64,
            "source_bundle_sha256": "2" * 64,
            "metadata_sha256": cbs.EXPECTED_METADATA_BODY_SHA256,
            "scores_sha256": cbs.EXPECTED_SCORES_BODY_SHA256,
            "counts_sha256": cbs.EXPECTED_COUNTS_BODY_SHA256,
            "mapped_variants_sha256": "3" * 64,
            "mapped_variants_authentication_status": ("unanchored_structural_validation_only"),
            "openapi_authentication_status": ("unanchored_source_lock_summary_only"),
            "score_columns": list(cbs.EXPECTED_SCORE_COLUMNS),
            "score_row_count": cbs.CBS_SCORE_ROW_COUNT,
            "count_row_count": cbs.CBS_SCORE_ROW_COUNT,
            "score_count_accession_join_status": "validated_exact_order",
            "mapped_current_accession_join_status": "validated_exact_set",
            "nonmissing_count_cell_count": (cbs.CBS_SCORE_ROW_COUNT * len(cbs.EXPECTED_COUNT_COLUMNS)),
            "missing_count_cell_count": 0,
        }
    )
    status["status_sha256"] = cbs.adapter_status_sha256(status)

    with pytest.raises(cbs.CbsAdapterError, match="exact source inputs"):
        cbs.validate_cbs_adapter_status(status)


def test_assay_bundle_rejects_control_counts_as_functional_wt():
    with pytest.raises(cbs.CbsAdapterError, match="functional wild-type baseline"):
        cbs.build_cbs_low_b6_assay_bundle(
            _production_inputs(),
            treat_control_counts_as_functional_wt=True,
        )


def test_assay_bundle_rejects_aggregate_score_as_raw_replicate():
    with pytest.raises(cbs.CbsAdapterError, match="aggregate score"):
        cbs.build_cbs_low_b6_assay_bundle(
            _production_inputs(),
            treat_aggregate_score_as_raw_replicate=True,
        )


def test_assay_bundle_rejects_relative_frequency_as_raw_read_count():
    with pytest.raises(cbs.CbsAdapterError, match="cannot be treated as raw read counts"):
        cbs.build_cbs_low_b6_assay_bundle(
            _production_inputs(),
            treat_relative_frequency_as_raw_read_count=True,
        )


def test_assay_bundle_always_requires_future_authenticated_native_replay():
    with pytest.raises(cbs.CbsAdapterError, match="future authenticated native"):
        cbs.build_cbs_low_b6_assay_bundle(_production_inputs())
    with pytest.raises(cbs.CbsAdapterError, match="cannot authenticate future"):
        cbs.build_cbs_low_b6_assay_bundle(
            _production_inputs(),
            native_replay_artifact={"claimed": True},
        )


def test_atomic_writer_is_exclusive_and_replace_safe(tmp_path):
    status = cbs.build_cbs_low_b6_adapter_status(_production_inputs())
    path = tmp_path / "cbs-status.json"

    assert cbs.write_cbs_adapter_status(path, status) == path
    assert path.read_bytes() == cbs.canonical_json_bytes(status) + b"\n"
    assert cbs.load_cbs_adapter_status(path) == status
    with pytest.raises(cbs.CbsAdapterError, match="already exists"):
        cbs.write_cbs_adapter_status(path, status)

    assert cbs.write_cbs_adapter_status(path, status, replace=True) == path
    assert cbs.load_cbs_adapter_status(path) == status

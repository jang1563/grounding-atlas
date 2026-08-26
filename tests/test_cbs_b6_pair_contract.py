from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from eval import cbs_b6_pair_contract as pair
from eval import mavedb_source_lock as msl

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "signal" / "dms" / "cbs_b6_pair_registry.v1.json"
HIGH_STATUS_PATH = ROOT / "signal" / "dms" / "cbs_high_b6_adapter_status.v1.json"
PAIR_STATUS_PATH = ROOT / "signal" / "dms" / "cbs_b6_pair_status.v1.json"
LOW_SOURCE_LOCK_PATH = ROOT / "signal" / "dms" / "cbs_low_b6_source_lock.v1.json"
HIGH_SOURCE_LOCK_PATH = ROOT / "signal" / "dms" / "cbs_high_b6_source_lock.v1.json"


@dataclass
class SyntheticPair:
    registry: dict
    low_body: bytes
    high_metadata_body: bytes
    high_body: bytes
    high_counts_body: bytes

    @property
    def inputs(self) -> pair.CbsB6PairInputs:
        return pair.CbsB6PairInputs(
            pair_registry=self.registry,
            low_scores_body=self.low_body,
            high_metadata_body=self.high_metadata_body,
            high_scores_body=self.high_body,
            high_counts_body=self.high_counts_body,
        )

    @property
    def high_inputs(self) -> pair.CbsHighB6AdapterInputs:
        return pair.CbsHighB6AdapterInputs(
            pair_registry=self.registry,
            high_metadata_body=self.high_metadata_body,
            high_scores_body=self.high_body,
            high_counts_body=self.high_counts_body,
        )


def _production_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def _csv_body(urn: str, rows: list[list[str]]) -> bytes:
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, lineterminator="\r\n")
    writer.writerow([*msl.FIXED_VARIANT_COLUMNS, *pair.SCORE_COLUMNS])
    for index, row in enumerate(rows, start=1):
        writer.writerow([f"{urn}#{index}", *row])
    return destination.getvalue().encode()


def _metadata_body(sequence: str) -> bytes:
    value = {
        "urn": pair.HIGH_URN,
        "title": pair.HIGH_TITLE,
        "numVariants": 3,
        "private": False,
        "processingState": "success",
        "supersededScoreSet": None,
        "supersedingScoreSet": None,
        "dataUsagePolicy": None,
        "license": {
            "id": 1,
            "shortName": "CC0",
            "longName": "CC0 (Public domain)",
            "active": True,
            "link": "https://creativecommons.org/publicdomain/zero/1.0/",
            "version": "1.0",
            "recordType": "ShortLicense",
        },
        "datasetColumns": {
            "scoreColumns": list(pair.SCORE_COLUMNS),
            "countColumns": list(pair.HIGH_COUNT_COLUMNS),
            "recordType": "DatasetColumns",
        },
        "scoreCalibrations": [],
        "targetGenes": [
            {
                "name": pair.GENE,
                "mappedHgncName": pair.GENE,
                "targetAccession": None,
                "targetSequence": {
                    "sequenceType": pair.TARGET_SEQUENCE_TYPE,
                    "sequence": sequence,
                    "recordType": "TargetSequence",
                },
                "externalIdentifiers": [
                    {
                        "identifier": {
                            "dbName": "UniProt",
                            "identifier": pair.TARGET_UNIPROT_ID,
                            "recordType": "ExternalGeneIdentifier",
                            "url": (f"http://purl.uniprot.org/uniprot/{pair.TARGET_UNIPROT_ID}"),
                        },
                        "offset": 0,
                        "recordType": "ExternalGeneIdentifierOffset",
                    }
                ],
            }
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _count_body(score_body: bytes) -> bytes:
    score_records = _csv_records(score_body)
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, lineterminator="\r\n")
    writer.writerow([*msl.FIXED_VARIANT_COLUMNS, *pair.HIGH_COUNT_COLUMNS])
    for row_index, score_record in enumerate(score_records[1:], start=1):
        fixed = score_record[: len(msl.FIXED_VARIANT_COLUMNS)]
        counts = [f"{row_index + column_index / 10:.1f}" for column_index in range(len(pair.HIGH_COUNT_COLUMNS))]
        writer.writerow([*fixed, *counts])
    return destination.getvalue().encode()


def _csv_records(body: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(body.decode(), newline="")))


def _replace_cell(body: bytes, row_index: int, column: str, value: str) -> bytes:
    records = _csv_records(body)
    column_index = records[0].index(column)
    records[row_index][column_index] = value
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, lineterminator="\r\n")
    writer.writerows(records)
    return destination.getvalue().encode()


def _bind_synthetic_registry(
    monkeypatch: pytest.MonkeyPatch,
    synthetic: SyntheticPair,
) -> None:
    low_digest = hashlib.sha256(synthetic.low_body).hexdigest()
    high_metadata_digest = hashlib.sha256(synthetic.high_metadata_body).hexdigest()
    high_digest = hashlib.sha256(synthetic.high_body).hexdigest()
    high_counts_digest = hashlib.sha256(synthetic.high_counts_body).hexdigest()
    metadata = json.loads(synthetic.high_metadata_body)
    sequence = metadata["targetGenes"][0]["targetSequence"]["sequence"]
    sequence_digest = hashlib.sha256(sequence.encode()).hexdigest()
    monkeypatch.setattr(pair, "LOW_ROW_COUNT", 4)
    monkeypatch.setattr(pair, "HIGH_ROW_COUNT", 3)
    monkeypatch.setattr(pair, "LOW_SCORES_BODY_BYTES", len(synthetic.low_body))
    monkeypatch.setattr(pair, "LOW_SCORES_BODY_SHA256", low_digest)
    monkeypatch.setattr(pair, "HIGH_SCORES_BODY_BYTES", len(synthetic.high_body))
    monkeypatch.setattr(pair, "HIGH_SCORES_BODY_SHA256", high_digest)
    monkeypatch.setattr(
        pair,
        "HIGH_METADATA_BODY_BYTES",
        len(synthetic.high_metadata_body),
    )
    monkeypatch.setattr(
        pair,
        "HIGH_METADATA_BODY_SHA256",
        high_metadata_digest,
    )
    monkeypatch.setattr(
        pair,
        "HIGH_COUNTS_BODY_BYTES",
        len(synthetic.high_counts_body),
    )
    monkeypatch.setattr(pair, "HIGH_COUNTS_BODY_SHA256", high_counts_digest)
    monkeypatch.setattr(pair, "TARGET_DNA_LENGTH", len(sequence))
    monkeypatch.setattr(pair, "TARGET_DNA_SHA256", sequence_digest)
    monkeypatch.setattr(
        pair,
        "EXPECTED_STRUCTURAL_OVERLAP",
        {
            "native_join_row_count": 2,
            "low_only_row_count": 2,
            "high_only_row_count": 1,
            "native_join_unique_hgvs_pro_count": 1,
            "condition_set_intersection_unique_hgvs_pro_count": 2,
        },
    )

    low = synthetic.registry["conditions"]["low"]
    high = synthetic.registry["conditions"]["high"]
    low["row_count"] = 4
    high["row_count"] = 3
    low["scores"]["body_bytes"] = len(synthetic.low_body)
    low["scores"]["body_sha256"] = low_digest
    high["scores"]["body_bytes"] = len(synthetic.high_body)
    high["scores"]["body_sha256"] = high_digest
    high["metadata"]["body_bytes"] = len(synthetic.high_metadata_body)
    high["metadata"]["body_sha256"] = high_metadata_digest
    high["counts"]["body_bytes"] = len(synthetic.high_counts_body)
    high["counts"]["body_sha256"] = high_counts_digest
    synthetic.registry["target"]["dna_sequence_length"] = len(sequence)
    synthetic.registry["target"]["dna_sequence_sha256"] = sequence_digest
    synthetic.registry["pair_contract"]["expected_structural_overlap"] = dict(pair.EXPECTED_STRUCTURAL_OVERLAP)
    monkeypatch.setattr(
        pair,
        "EXPECTED_PAIR_REGISTRY_SHA256",
        pair.canonical_sha256(synthetic.registry),
    )


@pytest.fixture
def synthetic_pair(monkeypatch: pytest.MonkeyPatch) -> SyntheticPair:
    low_body = _csv_body(
        pair.LOW_URN,
        [
            ["c.1A>G", "", "p.Ala1Gly", "0.1", "0.2", "0.1"],
            ["c.2C>T", "", "p.Ala1Gly", "0.1", "0.2", "0.1"],
            ["c.3G>A", "", "p.Gly2Asp", "NA", "NA", "NA"],
            ["c.4T>C", "", "p.Cys3Arg", "-0.1", "0.3", "0.15"],
        ],
    )
    high_body = _csv_body(
        pair.HIGH_URN,
        [
            ["c.1A>G", "", "p.Ala1Gly", "0.5", "0.3", "0.15"],
            ["c.2C>T", "", "p.Ala1Gly", "0.5", "0.3", "0.15"],
            ["c.5A>T", "", "p.Gly2Asp", "0.7", "0.4", "0.2"],
        ],
    )
    high_metadata_body = _metadata_body("ATGCGTATGCGT")
    high_counts_body = _count_body(high_body)
    synthetic = SyntheticPair(
        registry=deepcopy(_production_registry()),
        low_body=low_body,
        high_metadata_body=high_metadata_body,
        high_body=high_body,
        high_counts_body=high_counts_body,
    )
    _bind_synthetic_registry(monkeypatch, synthetic)
    return synthetic


def _rebind_changed_body(
    monkeypatch: pytest.MonkeyPatch,
    synthetic: SyntheticPair,
    *,
    condition: str,
    body: bytes,
) -> SyntheticPair:
    if condition == "low":
        changed = replace(synthetic, low_body=body)
    else:
        changed = replace(
            synthetic,
            high_body=body,
            high_counts_body=_count_body(body),
        )
    _bind_synthetic_registry(monkeypatch, changed)
    return changed


def test_production_registry_is_exact_and_builds_candidate_only_statuses() -> None:
    registry = _production_registry()
    assert pair.validate_cbs_b6_pair_registry(registry) == registry

    high = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(registry))
    paired = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(registry))
    for status in (high, paired):
        assert status["admission_status"] == "candidate_not_ingested"
        assert status["ingestion_status"] == "not_ingested"
        assert status["outcome_status"] == "not_derived"
        assert status["confirmatory_eligible"] is False
        assert status["automatic_promotion"] is False
    assert paired["replicate_pairing_claim"] is False
    assert paired["contrast_unit"] == pair.CONTRAST_UNIT
    assert paired["delta_orientation"] == pair.DELTA_ORIENTATION
    assert "not_sufficient_for_remediability" in paired["delta_orientation"]

    pair.validate_cbs_high_b6_adapter_status(high)
    pair.validate_cbs_b6_pair_status(paired)


def test_production_registry_hash_and_semantics_resist_tamper() -> None:
    registry = _production_registry()
    registry["pair_contract"]["delta_orientation"] = "low_b6_score_minus_high_b6_score"
    with pytest.raises(pair.CbsB6PairError, match="canonical SHA-256"):
        pair.validate_cbs_b6_pair_registry(registry)


@pytest.mark.parametrize(
    ("condition", "path", "bundle_sha256", "mapping_error_count"),
    [
        (
            "low",
            LOW_SOURCE_LOCK_PATH,
            pair.LOW_SOURCE_BUNDLE_SHA256,
            pair.LOW_MAPPING_ERROR_COUNT,
        ),
        (
            "high",
            HIGH_SOURCE_LOCK_PATH,
            pair.HIGH_SOURCE_BUNDLE_SHA256,
            pair.HIGH_MAPPING_ERROR_COUNT,
        ),
    ],
)
def test_production_source_locks_are_exactly_registry_pinned(
    condition: str,
    path: Path,
    bundle_sha256: str,
    mapping_error_count: int,
) -> None:
    registry = _production_registry()
    lock = pair.validate_cbs_b6_source_lock_bytes(
        path.read_bytes(),
        condition=condition,
        pair_registry=registry,
    )
    assert lock["source_bundle_sha256"] == bundle_sha256
    assert lock["mapping_contract"]["current_error_count"] == mapping_error_count
    assert lock["readiness"]["state"] == "COUNT_LINEAGE_PARTIAL"
    assert lock["ingestion_status"] == "not_ingested"
    assert lock["outcome_status"] == "not_derived"
    assert (
        pair.load_cbs_b6_source_lock(
            path,
            condition=condition,
            pair_registry=registry,
        )
        == lock
    )


def _rebind_source_lock_artifact(
    monkeypatch: pytest.MonkeyPatch,
    registry: dict,
    *,
    condition: str,
    body: bytes,
) -> None:
    expectation = registry["conditions"][condition]["source_lock"]
    expectation["artifact_bytes"] = len(body)
    expectation["artifact_sha256"] = hashlib.sha256(body).hexdigest()
    prefix = condition.upper()
    monkeypatch.setattr(
        pair,
        f"{prefix}_SOURCE_LOCK_ARTIFACT_BYTES",
        expectation["artifact_bytes"],
    )
    monkeypatch.setattr(
        pair,
        f"{prefix}_SOURCE_LOCK_ARTIFACT_SHA256",
        expectation["artifact_sha256"],
    )
    monkeypatch.setattr(
        pair,
        "EXPECTED_PAIR_REGISTRY_SHA256",
        pair.canonical_sha256(registry),
    )


@pytest.mark.parametrize("condition", ["low", "high"])
def test_source_lock_truncation_or_wrong_python_type_fails_closed(
    condition: str,
) -> None:
    registry = _production_registry()
    path = LOW_SOURCE_LOCK_PATH if condition == "low" else HIGH_SOURCE_LOCK_PATH
    body = path.read_bytes()
    with pytest.raises(pair.CbsB6PairError, match="differs from the registry"):
        pair.validate_cbs_b6_source_lock_bytes(
            body[:-1],
            condition=condition,
            pair_registry=registry,
        )
    with pytest.raises(pair.CbsB6PairError, match="must be exact bytes"):
        pair.validate_cbs_b6_source_lock_bytes(
            body.decode(),
            condition=condition,
            pair_registry=registry,
        )


def test_coherently_relocked_reformatted_source_lock_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _production_registry()
    lock = json.loads(HIGH_SOURCE_LOCK_PATH.read_text())
    body = json.dumps(lock, indent=2, sort_keys=True).encode() + b"\n"
    _rebind_source_lock_artifact(
        monkeypatch,
        registry,
        condition="high",
        body=body,
    )
    with pytest.raises(pair.CbsB6PairError, match="exact canonical JSON"):
        pair.validate_cbs_b6_source_lock_bytes(
            body,
            condition="high",
            pair_registry=registry,
        )


def test_coherently_relocked_duplicate_key_source_lock_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _production_registry()
    original = HIGH_SOURCE_LOCK_PATH.read_bytes()
    body = b'{"schema_version":1,' + original[1:]
    _rebind_source_lock_artifact(
        monkeypatch,
        registry,
        condition="high",
        body=body,
    )
    with pytest.raises(pair.CbsB6PairError, match="duplicate key"):
        pair.validate_cbs_b6_source_lock_bytes(
            body,
            condition="high",
            pair_registry=registry,
        )


def test_source_lock_registry_tamper_and_resolved_blockers_fail_closed() -> None:
    registry = _production_registry()
    registry["conditions"]["high"]["source_lock"]["mapped_variants"]["decoded_byte_count"] += 1
    with pytest.raises(pair.CbsB6PairError, match="canonical SHA-256"):
        pair.validate_cbs_b6_pair_registry(registry)

    canonical_registry = _production_registry()
    high_status = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(canonical_registry))
    pair_status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(canonical_registry))
    assert "CBS_HIGH_B6_COMPLETE_SOURCE_LOCK_MISSING" not in high_status["active_blocker_codes"]
    assert "CBS_HIGH_B6_MAPPED_VARIANTS_BODY_UNAUTHENTICATED" not in high_status["active_blocker_codes"]
    assert "CBS_B6_PAIR_FULL_SOURCE_BUNDLES_UNAUTHENTICATED" not in pair_status["active_blocker_codes"]
    assert high_status["source_readiness"] == "COUNT_LINEAGE_PARTIAL"
    assert high_status["outcome_status"] == "not_derived"
    assert pair_status["outcome_status"] == "not_derived"


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version_int_to_float",
        "registry_row_count_int_to_float",
        "semantic_false_to_zero",
    ],
)
def test_high_status_rejects_json_scalar_type_confusion_after_coherent_rehash(
    synthetic_pair: SyntheticPair,
    mutation: str,
) -> None:
    status = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(synthetic_pair.registry))
    if mutation == "schema_version_int_to_float":
        status["schema_version"] = 1.0
    elif mutation == "registry_row_count_int_to_float":
        status["registry_expectations"]["row_count"] = float(status["registry_expectations"]["row_count"])
    else:
        status["count_measurement_semantics"]["raw_read_count_claim"] = 0
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="differs"):
        pair.validate_cbs_high_b6_adapter_status(status)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version_int_to_float",
        "contract_false_to_zero",
        "overlap_count_int_to_float",
    ],
)
def test_pair_status_rejects_json_scalar_type_confusion_after_coherent_rehash(
    synthetic_pair: SyntheticPair,
    mutation: str,
) -> None:
    status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
    if mutation == "schema_version_int_to_float":
        status["schema_version"] = 1.0
    elif mutation == "contract_false_to_zero":
        status["registry_pair_contract"]["imputation_allowed"] = 0
    else:
        overlap = status["registry_pair_contract"]["expected_structural_overlap"]
        overlap["native_join_row_count"] = float(overlap["native_join_row_count"])
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="differs"):
        pair.validate_cbs_b6_pair_status(status)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("count_measurement_semantics", "raw_read_count_claim", True),
        ("count_measurement_semantics", "functional_wt_baseline_claim", True),
        ("score_measurement_semantics", "score_sd_se_as_replicates", True),
        ("score_measurement_semantics", "replicate_count_claim", True),
        ("score_measurement_semantics", "sd_se_squared_ratio", 8),
    ],
)
def test_high_status_measurement_semantics_resist_coherent_rehash(
    synthetic_pair: SyntheticPair,
    section: str,
    field: str,
    value: object,
) -> None:
    status = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(synthetic_pair.registry))
    status[section][field] = value
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="measurement semantics|replicate"):
        pair.validate_cbs_high_b6_adapter_status(status)


def test_high_status_prohibited_reinterpretations_are_exact(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(synthetic_pair.registry))
    status["prohibited_reinterpretations"].remove("normalized_relative_read_frequency_as_raw_read_count")
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="prohibited reinterpretations"):
        pair.validate_cbs_high_b6_adapter_status(status)


def test_pair_status_cannot_describe_conditions_as_paired_replicates(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
    status["replicate_pairing_claim"] = True
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="replicate_pairing_claim|boolean"):
        pair.validate_cbs_b6_pair_status(status)


def test_synthetic_exact_bodies_validate_accession_independent_join(
    synthetic_pair: SyntheticPair,
) -> None:
    _, summary = pair.validate_cbs_b6_pair_inputs(synthetic_pair.inputs)
    assert summary["native_join_status"].startswith("validated_accession_independent")
    assert summary["accession_join_used"] is False
    assert (
        summary["target_identity_status"]
        == "registry_audited_both_conditions_high_runtime_validated_low_runtime_not_replayed"
    )
    assert summary["native_join_row_count"] == 2
    assert summary["low_only_row_count"] == 2
    assert summary["high_only_row_count"] == 1
    assert summary["native_join_unique_hgvs_pro_count"] == 1
    assert summary["condition_set_intersection_unique_hgvs_pro_count"] == 2
    assert summary["complete_paired_native_row_count"] == 2
    assert summary["complete_paired_unique_hgvs_pro_count"] == 1
    assert summary["imputation_performed"] is False
    assert summary["delta_values_materialized"] is False
    assert summary["missingness_policy"].endswith("qc_admissibility_not_evaluated")


def test_structural_statuses_require_source_input_replay(
    synthetic_pair: SyntheticPair,
) -> None:
    high = pair.build_cbs_high_b6_adapter_status(synthetic_pair.high_inputs)
    paired = pair.build_cbs_b6_pair_status(synthetic_pair.inputs)

    with pytest.raises(pair.CbsB6PairError, match="exact metadata, score, and count"):
        pair.validate_cbs_high_b6_adapter_status(high)
    with pytest.raises(pair.CbsB6PairError, match="exact low scores and high"):
        pair.validate_cbs_b6_pair_status(paired)

    pair.validate_cbs_high_b6_adapter_status(
        high,
        source_inputs=synthetic_pair.high_inputs,
    )
    pair.validate_cbs_b6_pair_status(
        paired,
        source_inputs=synthetic_pair.inputs,
    )


def test_pair_score_bodies_are_all_or_none(synthetic_pair: SyntheticPair) -> None:
    incomplete = pair.CbsB6PairInputs(
        pair_registry=synthetic_pair.registry,
        low_scores_body=synthetic_pair.low_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="must be supplied together"):
        pair.validate_cbs_b6_pair_inputs(incomplete)


def test_high_adapter_core_bodies_are_all_or_none(
    synthetic_pair: SyntheticPair,
) -> None:
    incomplete = pair.CbsHighB6AdapterInputs(
        pair_registry=synthetic_pair.registry,
        high_scores_body=synthetic_pair.high_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="must be supplied together"):
        pair.build_cbs_high_b6_adapter_status(incomplete)


def test_high_core_metadata_score_and_count_contracts_validate(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_high_b6_adapter_status(synthetic_pair.high_inputs)
    offline = status["offline_validation"]
    assert offline["core_bodies_status"] == "validated_structural_only"
    assert offline["metadata_contract_status"].startswith("validated_active_public")
    assert offline["score_count_accession_order_status"].startswith("validated_exact_order")
    assert offline["count_value_status"].startswith("validated_finite_nonnegative")
    assert offline["nonmissing_count_cell_count"] + offline["missing_count_cell_count"] == 3 * len(
        pair.HIGH_COUNT_COLUMNS
    )


def test_coherently_relocked_metadata_semantic_tamper_is_rejected(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = json.loads(synthetic_pair.high_metadata_body)
    metadata["title"] = "forged high-B6"
    body = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    changed = replace(synthetic_pair, high_metadata_body=body)
    _bind_synthetic_registry(monkeypatch, changed)
    with pytest.raises(pair.CbsB6PairError, match="metadata body.title"):
        pair.build_cbs_high_b6_adapter_status(changed.high_inputs)


@pytest.mark.parametrize("value", ["-0.1", "NaN", "inf", ""])
def test_high_count_cells_reject_negative_nonfinite_or_implicit_missing(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    body = _replace_cell(
        synthetic_pair.high_counts_body,
        1,
        pair.HIGH_COUNT_COLUMNS[0],
        value,
    )
    changed = replace(synthetic_pair, high_counts_body=body)
    _bind_synthetic_registry(monkeypatch, changed)
    with pytest.raises(pair.CbsB6PairError, match="finite nonnegative decimal or exact NA"):
        pair.build_cbs_high_b6_adapter_status(changed.high_inputs)


def test_high_count_exact_na_is_missing_not_imputed(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _replace_cell(
        synthetic_pair.high_counts_body,
        1,
        pair.HIGH_COUNT_COLUMNS[0],
        "NA",
    )
    changed = replace(synthetic_pair, high_counts_body=body)
    _bind_synthetic_registry(monkeypatch, changed)
    status = pair.build_cbs_high_b6_adapter_status(changed.high_inputs)
    assert status["offline_validation"]["missing_count_cell_count"] == 1
    assert status["offline_validation"]["imputation_performed"] is False


def test_high_score_count_accession_order_mismatch_is_rejected(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _replace_cell(
        synthetic_pair.high_counts_body,
        2,
        "accession",
        f"{pair.HIGH_URN}#999",
    )
    changed = replace(synthetic_pair, high_counts_body=body)
    _bind_synthetic_registry(monkeypatch, changed)
    with pytest.raises(pair.CbsB6PairError, match="accession order differs"):
        pair.build_cbs_high_b6_adapter_status(changed.high_inputs)


def test_unbound_score_body_tamper_is_rejected(synthetic_pair: SyntheticPair) -> None:
    tampered = replace(
        synthetic_pair.inputs,
        high_scores_body=synthetic_pair.high_body + b"\n",
    )
    with pytest.raises(pair.CbsB6PairError, match="hash-pinned"):
        pair.validate_cbs_b6_pair_inputs(tampered)


def test_duplicate_native_join_key_is_rejected(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(
        synthetic_pair.high_body,
        2,
        "hgvs_nt",
        "c.1A>G",
    )
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="native join keys must be unique"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_matched_native_rows_require_exact_protein_identity(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(
        synthetic_pair.high_body,
        1,
        "hgvs_pro",
        "p.Ala1Val",
    )
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="disagree in hgvs_splice or hgvs_pro"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_matched_hgvs_nt_rows_require_exact_splice_annotation(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(
        synthetic_pair.high_body,
        1,
        "hgvs_splice",
        "c.1+1G>A",
    )
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="disagree in hgvs_splice or hgvs_pro"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_codon_duplicate_values_must_agree_exactly(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(synthetic_pair.high_body, 2, "score", "0.500")
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="codon duplicates.*exact missingness"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_codon_duplicate_missingness_must_agree_exactly(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(synthetic_pair.high_body, 2, "se", "NA")
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="exact missingness"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sd", "-0.1", "cannot be negative"),
        ("se", "-0.1", "cannot be negative"),
        ("sd", "NaN", "finite decimal"),
        ("se", "inf", "finite decimal"),
        ("score", "-inf", "finite decimal"),
    ],
)
def test_nonfinite_or_negative_uncertainty_is_rejected(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: str,
    message: str,
) -> None:
    changed_body = _replace_cell(synthetic_pair.high_body, 3, column, value)
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match=message):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_missing_values_are_excluded_without_codon_propagation_or_imputation(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(synthetic_pair.low_body, 1, "se", "NA")
    changed_body = _replace_cell(changed_body, 2, "se", "NA")
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="low",
        body=changed_body,
    )
    _, summary = pair.validate_cbs_b6_pair_inputs(changed.inputs)
    assert summary["native_join_row_count"] == 2
    assert summary["complete_paired_native_row_count"] == 0
    assert summary["complete_paired_unique_hgvs_pro_count"] == 0
    assert summary["imputation_performed"] is False
    assert summary["delta_values_materialized"] is False


def test_expected_overlap_is_frozen_not_inferred_from_supplied_bodies(
    synthetic_pair: SyntheticPair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_body = _replace_cell(
        synthetic_pair.high_body,
        2,
        "hgvs_nt",
        "c.20C>T",
    )
    changed = _rebind_changed_body(
        monkeypatch,
        synthetic_pair,
        condition="high",
        body=changed_body,
    )
    with pytest.raises(pair.CbsB6PairError, match="structural overlap differs"):
        pair.validate_cbs_b6_pair_inputs(changed.inputs)


def test_pair_status_rejects_reversed_orientation_even_if_rehashed(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
    status["delta_orientation"] = "low_minus_high"
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="delta_orientation"):
        pair.validate_cbs_b6_pair_status(status)


def test_pair_status_rejects_confirmatory_or_outcome_promotion(
    synthetic_pair: SyntheticPair,
) -> None:
    for field, value in (
        ("confirmatory_eligible", True),
        ("outcome_status", "derived"),
        ("ingestion_status", "ingested"),
    ):
        status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
        status[field] = value
        status["status_sha256"] = pair.status_sha256(status)
        with pytest.raises(pair.CbsB6PairError, match=field):
            pair.validate_cbs_b6_pair_status(status)


def test_forged_validated_pair_status_cannot_omit_replay(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
    offline = status["offline_pair_validation"]
    offline["high_core_bodies_status"] = "validated_structural_only"
    offline["low_score_body_status"] = "validated_structural_only"
    offline["high_score_body_status"] = "validated_structural_only"
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="exact low scores and high"):
        pair.validate_cbs_b6_pair_status(status)


def test_forged_validated_high_status_cannot_omit_replay(
    synthetic_pair: SyntheticPair,
) -> None:
    status = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(synthetic_pair.registry))
    status["offline_validation"]["core_bodies_status"] = "validated_structural_only"
    status["status_sha256"] = pair.status_sha256(status)
    with pytest.raises(pair.CbsB6PairError, match="exact metadata, score, and count"):
        pair.validate_cbs_high_b6_adapter_status(status)


def test_outcome_materialization_always_rejects(
    synthetic_pair: SyntheticPair,
) -> None:
    with pytest.raises(pair.CbsB6PairError, match="covariance or joint-bootstrap"):
        pair.materialize_cbs_b6_pair_outcomes(synthetic_pair.inputs)
    with pytest.raises(pair.CbsB6PairError, match="cannot authenticate"):
        pair.materialize_cbs_b6_pair_outcomes(
            synthetic_pair.inputs,
            covariance_or_joint_bootstrap_artifact={"status": "claimed"},
        )


def test_outcome_materialization_rejects_reverse_orientation_and_imputation(
    synthetic_pair: SyntheticPair,
) -> None:
    with pytest.raises(pair.CbsB6PairError, match="orientation is frozen"):
        pair.materialize_cbs_b6_pair_outcomes(
            synthetic_pair.inputs,
            delta_orientation="low_minus_high",
        )
    with pytest.raises(pair.CbsB6PairError, match="cannot be imputed"):
        pair.materialize_cbs_b6_pair_outcomes(
            synthetic_pair.inputs,
            impute_missing=True,
        )


def test_atomic_status_writers_and_loaders(
    synthetic_pair: SyntheticPair,
    tmp_path: Path,
) -> None:
    high = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(synthetic_pair.registry))
    paired = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(synthetic_pair.registry))
    high_path = tmp_path / "high.json"
    pair_path = tmp_path / "pair.json"
    assert pair.write_cbs_high_b6_adapter_status(high_path, high) == high_path
    assert pair.write_cbs_b6_pair_status(pair_path, paired) == pair_path
    assert pair.load_cbs_high_b6_adapter_status(high_path) == high
    assert pair.load_cbs_b6_pair_status(pair_path) == paired
    with pytest.raises(pair.CbsB6PairError, match="already exists"):
        pair.write_cbs_b6_pair_status(pair_path, paired)


def test_module_cli_writes_canonical_candidate_only_statuses(tmp_path: Path) -> None:
    high_path = tmp_path / "high.json"
    pair_path = tmp_path / "pair.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.cbs_b6_pair_contract",
            "--registry",
            str(REGISTRY_PATH),
            "--high-status-out",
            str(high_path),
            "--pair-status-out",
            str(pair_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["outcome_status"] == "not_derived"
    assert report["confirmatory_eligible"] is False
    pair.load_cbs_high_b6_adapter_status(high_path)
    pair.load_cbs_b6_pair_status(pair_path)


def test_persisted_production_statuses_are_deterministic_when_present() -> None:
    if not HIGH_STATUS_PATH.exists() or not PAIR_STATUS_PATH.exists():
        pytest.skip("production statuses are generated after module integration")
    registry = _production_registry()
    expected_high = pair.build_cbs_high_b6_adapter_status(pair.CbsHighB6AdapterInputs(registry))
    expected_pair = pair.build_cbs_b6_pair_status(pair.CbsB6PairInputs(registry))
    assert json.loads(HIGH_STATUS_PATH.read_text()) == expected_high
    assert json.loads(PAIR_STATUS_PATH.read_text()) == expected_pair
    pair.load_cbs_high_b6_adapter_status(HIGH_STATUS_PATH)
    pair.load_cbs_b6_pair_status(PAIR_STATUS_PATH)

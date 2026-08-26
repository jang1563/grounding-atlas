"""Fail-closed CBS low/high-vitamin-B6 pairing contract.

This module freezes the candidate-stage relationship between the public
MaveDB CBS low-B6 and high-B6 score sets.  It can authenticate the exact pair
registry and, when both exact score bodies are supplied, replay only structural
properties of the accession-independent HGVS join.  It never emits score
differences, confidence intervals, labels, or benchmark outcomes.

The deposited accessions are score-set-specific, so pairing uses the MaveDB
primary ``hgvs_nt`` index and then requires ``hgvs_splice`` and ``hgvs_pro``
annotations to agree exactly (including exact missingness).  Codon duplicates
may be collapsed only when ``score``, ``sd``, and ``se`` missingness and every
nonmissing value agree exactly within each condition.  Missing values are never
imputed or propagated.  Even after a successful structural replay, a
covariance-aware native assay replay remains required before ``high - low``
uncertainty can be reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

from eval import cbs_mavedb_adapter as low_adapter
from eval import mavedb_source_lock as msl

SCHEMA_VERSION = 1
PAIR_REGISTRY_ARTIFACT_TYPE = "groundbench.dms_cbs_b6_pair_registry"
HIGH_STATUS_ARTIFACT_TYPE = "groundbench.dms_cbs_high_b6_adapter_status"
PAIR_STATUS_ARTIFACT_TYPE = "groundbench.dms_cbs_b6_pair_status"

REGISTRY_CLAIM_SCOPE = "exact_candidate_pair_source_expectations_only_no_ingestion_or_outcome_claim"
HIGH_STATUS_CLAIM_SCOPE = "candidate_high_b6_source_expectations_only_no_ingestion_or_outcome_claim"
PAIR_STATUS_CLAIM_SCOPE = "candidate_pair_structural_validation_only_no_ingestion_or_outcome_claim"

HIGH_ADAPTER_ID = "mavedb-cbs-high-b6-pair-candidate-v1"
PAIR_CONTRACT_ID = "mavedb-cbs-low-high-b6-pair-contract-v1"

GENE = "CBS"
LOW_URN = "urn:mavedb:00000005-a-5"
HIGH_URN = "urn:mavedb:00000005-a-6"
LOW_TITLE = "CBS low-B6"
HIGH_TITLE = "CBS high-B6"
LOW_DOSE_DEFINITION = "average_of_0_and_1_ng_per_mL_pyridoxine_conditions"
HIGH_DOSE_DEFINITION = "400_ng_per_mL_pyridoxine_condition"

LOW_ROW_COUNT = 11_478
HIGH_ROW_COUNT = 10_802
SCORE_COLUMNS = ("score", "sd", "se")
LOW_COUNT_COLUMNS = low_adapter.EXPECTED_COUNT_COLUMNS
HIGH_COUNT_COLUMNS = (
    "nonselect1",
    "nonselect2",
    "nonselect3",
    "nonselect4",
    "select1",
    "select2",
    "select3",
    "select4",
    "controlNS1",
    "controlNS2",
    "controlNS3",
    "controlNS4",
    "controlS1",
    "controlS2",
    "controlS3",
    "controlS4",
)

TARGET_SEQUENCE_TYPE = "dna"
TARGET_DNA_LENGTH = 1_656
TARGET_DNA_SHA256 = "0ed0d6e9099a2c31e49d9a00b1711a42be68816b798e3616fb072625d7d96f0f"
TARGET_UNIPROT_ID = "P35520"
TARGET_PROTEIN_LENGTH = 551
TARGET_PROTEIN_SHA256 = "1e7cf97e465052ebc9f35a7bc9c299a509c5a73495086332a256b67cca3e5f78"

LOW_METADATA_BODY_BYTES = low_adapter.EXPECTED_METADATA_BODY_BYTES
LOW_METADATA_BODY_SHA256 = low_adapter.EXPECTED_METADATA_BODY_SHA256
LOW_SCORES_BODY_BYTES = low_adapter.EXPECTED_SCORES_BODY_BYTES
LOW_SCORES_BODY_SHA256 = low_adapter.EXPECTED_SCORES_BODY_SHA256
LOW_COUNTS_BODY_BYTES = low_adapter.EXPECTED_COUNTS_BODY_BYTES
LOW_COUNTS_BODY_SHA256 = low_adapter.EXPECTED_COUNTS_BODY_SHA256

HIGH_METADATA_BODY_BYTES = 29_327
HIGH_METADATA_BODY_SHA256 = "153dda901c68a3e28a9aef02e41008e6ee8ed19d5f05c518190d5a304a33933d"
HIGH_SCORES_BODY_BYTES = 1_260_675
HIGH_SCORES_BODY_SHA256 = "2fa0775fb5cd887102c786a8ecf89d3299e7d3e2eeae220b0d6572a03140adc8"
HIGH_COUNTS_BODY_BYTES = 2_568_937
HIGH_COUNTS_BODY_SHA256 = "247ed53735d626eb46fb1af162ee57e084ec596e6808438e51aef2b297da9f46"

OPENAPI_VERSION = "2026.2.7"
OPENAPI_BODY_BYTES = 388_236
OPENAPI_BODY_SHA256 = "cb51201fb966601f1dc7bb6297f0022fca6a4cfa5ba22299b6b22523c2efd408"

LOW_SOURCE_LOCK_ARTIFACT_BYTES = 8_058
LOW_SOURCE_LOCK_ARTIFACT_SHA256 = "1028035c19b989dc9b5d2e0b4e66339e8a768effe7a0ca4db80c411c4f1c1f93"
LOW_SOURCE_LOCK_CANONICAL_SHA256 = "f3463c28d06e9d75d19d36e075578545ec41930944a497df329e40a02f3b2413"
LOW_SOURCE_BUNDLE_SHA256 = "a2e3d05b342f5c08245fb0de3e46577cf6f83896409cab2a644616c28fed9407"
LOW_MAPPING_CONTRACT_SHA256 = "de1e8ae271e94817559647a6f78f2ac7551aa90bf9d3a72924f4f6e85a3d10f3"
LOW_MAPPING_ERROR_COUNT = 152
LOW_MAPPED_VARIANTS_BODY_BYTES = 67_236_243
LOW_MAPPED_VARIANTS_BODY_SHA256 = "0281defc8da4b9b2a3046c7642a1d69f9cc4e5ad3533de954019b6438edaffa6"

HIGH_SOURCE_LOCK_ARTIFACT_BYTES = 6_970
HIGH_SOURCE_LOCK_ARTIFACT_SHA256 = "20e15da150d56279cc0893d9d0cba235e2cf1d6a8ac86b5c79b400a300c48846"
HIGH_SOURCE_LOCK_CANONICAL_SHA256 = "36a23a59b0c0b2b5599ebf4a760d51d06c8160bef3a8bb0a3701b989e00541c9"
HIGH_SOURCE_BUNDLE_SHA256 = "c587b79111f612dcdbcf20964b77ef9129f59025e7560516ff8eec2ae9940cb0"
HIGH_MAPPING_CONTRACT_SHA256 = "cd5eef0fc44aa7ace92f5111dff3b82ecf5bb58ef60aaef9ca0d9277914611a3"
HIGH_MAPPING_ERROR_COUNT = 150
HIGH_MAPPED_VARIANTS_BODY_BYTES = 63_266_281
HIGH_MAPPED_VARIANTS_BODY_SHA256 = "d4a587c2f1b100d546c84f6212a3b392c476b55a30e5303b5ba6711d1da9429d"

EXPECTED_PAIR_REGISTRY_SHA256 = "b27b1bc4e5f8e2a9af826e2a1892ef162e9759759b3ef218ef40589e81af5bfa"

CONTRAST_UNIT = "variant_aligned_two_condition_contrast_not_paired_replicates"
DELTA_ORIENTATION = (
    "high_b6_score_minus_low_b6_score_positive_indicates_higher_high_b6_fitness_not_sufficient_for_remediability"
)
NATIVE_JOIN_KEY = ("hgvs_nt",)
PROTEIN_GROUP_KEY = "hgvs_pro"
CODON_DUPLICATE_POLICY = "collapse_only_when_score_sd_se_missingness_and_nonmissing_values_are_exactly_equal_never_average_or_count_as_replicates"
MISSINGNESS_POLICY = "structural_pair_requires_nonmissing_score_sd_se_in_both_conditions_qc_admissibility_not_evaluated"
PRIMARY_ENDPOINT = "continuous_high_minus_low_fitness_difference"
UNCERTAINTY_RULE = "variance_high_minus_low_equals_variance_high_plus_variance_low_minus_two_covariance_high_low"
COVARIANCE_STATUS = "required_not_authenticated"
OUTCOME_MATERIALIZATION_RULE = (
    "forbidden_until_native_replay_qc_codon_collapse_covariance_ci_and_external_registration_are_authenticated"
)

EXPECTED_STRUCTURAL_OVERLAP = {
    "native_join_row_count": 10_098,
    "low_only_row_count": 1_380,
    "high_only_row_count": 704,
    "native_join_unique_hgvs_pro_count": 7_365,
    "condition_set_intersection_unique_hgvs_pro_count": 7_395,
}

PUBLISHED_LOW_B6_FUNCTIONAL_RULE = {
    "classification": "deleterious_if_upper_95_percent_ci_strictly_below_threshold",
    "upper_95_percent_ci_threshold": 0.6,
    "fdr": 0.05,
    "complement_classification": "not_defined_as_neutral_or_retained_function",
}
PUBLISHED_B6_REMEDIABILITY_RULE = {
    "eligibility": "classified_deleterious_under_low_b6_rule",
    "classification": "remediable_if_lower_95_percent_ci_of_high_minus_low_strictly_above_threshold",
    "lower_95_percent_ci_threshold": 0.22,
    "fdr": 0.05,
}

REGISTRY_BLOCKERS = (
    "The low- and high-B6 native TileSeq replay, sample map, QC, codon collapse, and functional anchors are not authenticated.",
    "The low/high covariance or a joint-bootstrap artifact is absent, so delta confidence intervals cannot be reconstructed.",
    "The paired target has not been externally registered and cannot be represented by the current scalar schema-2 replay.",
)

HIGH_BLOCKER_CODES = (
    "CBS_HIGH_B6_NATIVE_TILESEQ_REPLAY_UNAUTHENTICATED",
    "CBS_HIGH_B6_SCHEMA2_MULTICHANNEL_REPLAY_UNSUPPORTED",
    "CBS_HIGH_B6_OUTCOME_MATERIALIZATION_PROHIBITED",
)
HIGH_COUNT_MEASUREMENT_SEMANTICS = {
    "measurement_scale": "relative_read_frequency_per_1M_total_reads",
    "value_contract": "finite_nonnegative_decimal_or_exact_NA",
    "raw_read_count_claim": False,
    "controlNS_controlS_role": (
        "nonmutagenized_wild_type_amplicon_sequencing_error_control_not_functional_wt_baseline"
    ),
    "functional_wt_baseline_claim": False,
}
HIGH_SCORE_MEASUREMENT_SEMANTICS = {
    "measurement": "aggregate_normalized_fitness_score",
    "columns": list(SCORE_COLUMNS),
    "raw_replicate_columns": [],
    "score_sd_se_as_replicates": False,
    "replicate_count_claim": False,
    "sd_se_squared_ratio": 4,
    "ratio_evidence_scope": "external_status_only_sample_map_missing",
}
HIGH_PROHIBITED_REINTERPRETATIONS = (
    "controlNS_or_controlS_as_functional_wild_type",
    "deposited_aggregate_score_as_raw_replicate",
    "high_low_conditions_as_paired_replicates",
    "normalized_relative_read_frequency_as_raw_read_count",
    "score_sd_or_se_as_replicate",
    "sd_se_squared_ratio_as_authenticated_replicate_count",
)
PAIR_BLOCKER_CODES = (
    "CBS_B6_PAIR_COVARIANCE_OR_JOINT_BOOTSTRAP_MISSING",
    "CBS_B6_PAIR_EXTERNAL_REGISTRATION_MISSING",
    "CBS_B6_PAIR_NATIVE_TILESEQ_REPLAY_UNAUTHENTICATED",
    "CBS_B6_PAIR_OUTCOME_MATERIALIZATION_PROHIBITED",
    "CBS_B6_PAIR_SCHEMA2_EXTENSION_MISSING",
)

REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "claim_scope",
        "access_date",
        "gene",
        "target",
        "conditions",
        "pair_contract",
        "admission_status",
        "outcome_status",
        "confirmatory_eligible",
        "blockers",
    }
)
TARGET_FIELDS = frozenset(
    {
        "gene",
        "sequence_type",
        "dna_sequence_length",
        "dna_sequence_sha256",
        "uniprot_id",
        "protein_sequence_length",
        "protein_sequence_sha256",
    }
)
CONDITION_FIELDS = frozenset(
    {
        "urn",
        "title",
        "dose_definition",
        "row_count",
        "score_columns",
        "count_columns",
        "metadata",
        "scores",
        "counts",
        "source_lock",
    }
)
BODY_LOCK_FIELDS = frozenset({"url", "body_bytes", "body_sha256"})
SOURCE_LOCK_FIELDS = frozenset(
    {
        "artifact_type",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_json_sha256",
        "source_bundle_sha256",
        "readiness_state",
        "mapping_contract_sha256",
        "current_mapping_error_count",
        "mapped_variants",
        "openapi",
    }
)
MAPPED_VARIANTS_LOCK_FIELDS = frozenset({"decoded_byte_count", "sha256"})
OPENAPI_LOCK_FIELDS = frozenset({"version", "decoded_byte_count", "sha256"})
PAIR_CONTRACT_FIELDS = frozenset(
    {
        "contrast_unit",
        "replicate_pairing_claim",
        "delta_orientation",
        "native_join_key",
        "protein_group_key",
        "codon_duplicate_policy",
        "missingness_policy",
        "imputation_allowed",
        "primary_endpoint",
        "published_low_b6_functional_rule",
        "published_b6_remediability_rule",
        "expected_structural_overlap",
        "uncertainty_rule",
        "covariance_status",
        "outcome_materialization",
    }
)
EXPECTED_OVERLAP_FIELDS = frozenset(EXPECTED_STRUCTURAL_OVERLAP)

HIGH_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "adapter_id",
        "claim_scope",
        "urn",
        "gene",
        "title",
        "dose_definition",
        "pair_registry_sha256",
        "source_readiness",
        "admission_status",
        "ingestion_status",
        "outcome_status",
        "confirmatory_eligible",
        "automatic_promotion",
        "registry_expectations",
        "offline_validation",
        "count_measurement_semantics",
        "score_measurement_semantics",
        "active_blocker_codes",
        "prohibited_reinterpretations",
        "status_sha256",
    }
)
HIGH_OFFLINE_FIELDS = frozenset(
    {
        "pair_registry_status",
        "core_bodies_status",
        "metadata_body_sha256",
        "score_body_sha256",
        "count_body_sha256",
        "score_row_count",
        "count_row_count",
        "score_columns",
        "count_columns",
        "complete_measured_row_count",
        "incomplete_measured_row_count",
        "codon_duplicate_status",
        "metadata_contract_status",
        "score_count_accession_order_status",
        "count_value_status",
        "nonmissing_count_cell_count",
        "missing_count_cell_count",
        "imputation_performed",
        "outcomes_materialized",
    }
)

PAIR_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "contract_id",
        "claim_scope",
        "gene",
        "pair_registry_sha256",
        "conditions",
        "contrast_unit",
        "replicate_pairing_claim",
        "delta_orientation",
        "primary_endpoint",
        "admission_status",
        "ingestion_status",
        "outcome_status",
        "confirmatory_eligible",
        "automatic_promotion",
        "registry_pair_contract",
        "high_adapter_status_sha256",
        "offline_pair_validation",
        "active_blocker_codes",
        "status_sha256",
    }
)
PAIR_CONDITION_STATUS_FIELDS = frozenset({"low_urn", "high_urn", "low_dose", "high_dose"})
PAIR_OFFLINE_FIELDS = frozenset(
    {
        "pair_registry_status",
        "target_identity_status",
        "high_core_bodies_status",
        "low_score_body_status",
        "high_score_body_status",
        "low_score_body_sha256",
        "high_score_body_sha256",
        "low_score_row_count",
        "high_score_row_count",
        "native_join_key",
        "accession_join_used",
        "native_join_status",
        "native_join_row_count",
        "low_only_row_count",
        "high_only_row_count",
        "native_join_unique_hgvs_pro_count",
        "condition_set_intersection_unique_hgvs_pro_count",
        "complete_paired_native_row_count",
        "complete_paired_unique_hgvs_pro_count",
        "codon_duplicate_status",
        "missingness_policy",
        "imputation_performed",
        "delta_values_materialized",
        "uncertainty_status",
    }
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_PATTERN = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
SCORE_MISSING_TOKENS = frozenset({"", "NA"})
COUNT_MISSING_TOKENS = frozenset({"NA"})


class CbsB6PairError(ValueError):
    """Raised when the CBS B6 pair contract fails closed."""


@dataclass(frozen=True)
class CbsHighB6AdapterInputs:
    """Exact registry and optional all-or-none high-B6 core bodies."""

    pair_registry: Mapping[str, Any]
    high_metadata_body: bytes | None = None
    high_scores_body: bytes | None = None
    high_counts_body: bytes | None = None


@dataclass(frozen=True)
class CbsB6PairInputs:
    """Exact registry and optional low score plus high-B6 core bodies."""

    pair_registry: Mapping[str, Any]
    low_scores_body: bytes | None = None
    high_metadata_body: bytes | None = None
    high_scores_body: bytes | None = None
    high_counts_body: bytes | None = None


@dataclass(frozen=True)
class _ScoreRow:
    native_key: str
    hgvs_splice: str | None
    hgvs_pro: str | None
    raw_values: tuple[str | None, str | None, str | None]
    values: tuple[Decimal | None, Decimal | None, Decimal | None]


@dataclass(frozen=True)
class _ScoreTable:
    rows: tuple[_ScoreRow, ...]
    by_native_key: Mapping[str, _ScoreRow]
    accessions: tuple[str, ...]
    complete_measured_row_count: int
    incomplete_measured_row_count: int


@dataclass(frozen=True)
class _CountTable:
    accessions: tuple[str, ...]
    row_count: int
    nonmissing_count_cell_count: int
    missing_count_cell_count: int


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository-wide canonical JSON representation."""

    return msl.canonical_json_bytes(value)


def canonical_sha256(value: Any) -> str:
    """Return a canonical JSON SHA-256 digest."""

    return msl.canonical_sha256(value)


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CbsB6PairError(f"{context} must be a JSON object")
    return value


def _exact_mapping(
    value: Any,
    expected_fields: frozenset[str] | set[str],
    context: str,
) -> Mapping[str, Any]:
    result = _mapping(value, context)
    observed = set(result)
    expected = set(expected_fields)
    if observed != expected:
        raise CbsB6PairError(
            f"{context} must use the exact schema; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return result


def _require_exact_json(observed: Any, expected: Any, context: str) -> None:
    """Require canonical JSON identity, including JSON scalar types."""

    try:
        observed_bytes = canonical_json_bytes(observed)
        expected_bytes = canonical_json_bytes(expected)
    except msl.SourceLockError as exc:
        raise CbsB6PairError(f"{context} must be canonical JSON") from exc
    if observed_bytes != expected_bytes:
        raise CbsB6PairError(f"{context} differs")


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CbsB6PairError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 1:
        raise CbsB6PairError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise CbsB6PairError(f"{context} must be a nonnegative integer")
    return value


def _json_object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CbsB6PairError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_file(path: str | Path, context: str) -> Mapping[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CbsB6PairError(f"cannot read {context}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object_hook)
    except CbsB6PairError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CbsB6PairError(f"{context} must be duplicate-free UTF-8 JSON") from exc
    return _mapping(value, context)


def _parse_json_bytes(value: bytes, context: str) -> Any:
    if type(value) is not bytes:
        raise CbsB6PairError(f"{context} must be exact decoded bytes")
    try:
        return json.loads(value.decode("utf-8"), object_pairs_hook=_json_object_hook)
    except CbsB6PairError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CbsB6PairError(f"{context} must be duplicate-free UTF-8 JSON") from exc


def _condition_body_lock(
    urn: str,
    name: str,
    body_bytes: int,
    body_sha256: str,
) -> dict[str, Any]:
    return {
        "url": f"{msl.DEFAULT_API_BASE_URL}/score-sets/{urn}" + ("" if name == "metadata" else f"/{name}"),
        "body_bytes": body_bytes,
        "body_sha256": body_sha256,
    }


def _source_lock_expectation(condition: str) -> dict[str, Any]:
    if condition == "low":
        artifact_bytes = LOW_SOURCE_LOCK_ARTIFACT_BYTES
        artifact_sha256 = LOW_SOURCE_LOCK_ARTIFACT_SHA256
        canonical_sha256_value = LOW_SOURCE_LOCK_CANONICAL_SHA256
        source_bundle_sha256 = LOW_SOURCE_BUNDLE_SHA256
        mapping_contract_sha256 = LOW_MAPPING_CONTRACT_SHA256
        current_mapping_error_count = LOW_MAPPING_ERROR_COUNT
        mapped_body_bytes = LOW_MAPPED_VARIANTS_BODY_BYTES
        mapped_body_sha256 = LOW_MAPPED_VARIANTS_BODY_SHA256
    elif condition == "high":
        artifact_bytes = HIGH_SOURCE_LOCK_ARTIFACT_BYTES
        artifact_sha256 = HIGH_SOURCE_LOCK_ARTIFACT_SHA256
        canonical_sha256_value = HIGH_SOURCE_LOCK_CANONICAL_SHA256
        source_bundle_sha256 = HIGH_SOURCE_BUNDLE_SHA256
        mapping_contract_sha256 = HIGH_MAPPING_CONTRACT_SHA256
        current_mapping_error_count = HIGH_MAPPING_ERROR_COUNT
        mapped_body_bytes = HIGH_MAPPED_VARIANTS_BODY_BYTES
        mapped_body_sha256 = HIGH_MAPPED_VARIANTS_BODY_SHA256
    else:
        raise AssertionError(f"unknown CBS B6 condition: {condition}")
    return {
        "artifact_type": msl.ARTIFACT_TYPE,
        "artifact_bytes": artifact_bytes,
        "artifact_sha256": artifact_sha256,
        "canonical_json_sha256": canonical_sha256_value,
        "source_bundle_sha256": source_bundle_sha256,
        "readiness_state": msl.Readiness.COUNT_LINEAGE_PARTIAL.value,
        "mapping_contract_sha256": mapping_contract_sha256,
        "current_mapping_error_count": current_mapping_error_count,
        "mapped_variants": {
            "decoded_byte_count": mapped_body_bytes,
            "sha256": mapped_body_sha256,
        },
        "openapi": {
            "version": OPENAPI_VERSION,
            "decoded_byte_count": OPENAPI_BODY_BYTES,
            "sha256": OPENAPI_BODY_SHA256,
        },
    }


def _expected_condition(condition: str) -> dict[str, Any]:
    if condition == "low":
        urn = LOW_URN
        return {
            "urn": urn,
            "title": LOW_TITLE,
            "dose_definition": LOW_DOSE_DEFINITION,
            "row_count": LOW_ROW_COUNT,
            "score_columns": list(SCORE_COLUMNS),
            "count_columns": list(LOW_COUNT_COLUMNS),
            "metadata": _condition_body_lock(
                urn,
                "metadata",
                LOW_METADATA_BODY_BYTES,
                LOW_METADATA_BODY_SHA256,
            ),
            "scores": _condition_body_lock(
                urn,
                "scores",
                LOW_SCORES_BODY_BYTES,
                LOW_SCORES_BODY_SHA256,
            ),
            "counts": _condition_body_lock(
                urn,
                "counts",
                LOW_COUNTS_BODY_BYTES,
                LOW_COUNTS_BODY_SHA256,
            ),
            "source_lock": _source_lock_expectation("low"),
        }
    if condition == "high":
        urn = HIGH_URN
        return {
            "urn": urn,
            "title": HIGH_TITLE,
            "dose_definition": HIGH_DOSE_DEFINITION,
            "row_count": HIGH_ROW_COUNT,
            "score_columns": list(SCORE_COLUMNS),
            "count_columns": list(HIGH_COUNT_COLUMNS),
            "metadata": _condition_body_lock(
                urn,
                "metadata",
                HIGH_METADATA_BODY_BYTES,
                HIGH_METADATA_BODY_SHA256,
            ),
            "scores": _condition_body_lock(
                urn,
                "scores",
                HIGH_SCORES_BODY_BYTES,
                HIGH_SCORES_BODY_SHA256,
            ),
            "counts": _condition_body_lock(
                urn,
                "counts",
                HIGH_COUNTS_BODY_BYTES,
                HIGH_COUNTS_BODY_SHA256,
            ),
            "source_lock": _source_lock_expectation("high"),
        }
    raise AssertionError(f"unknown CBS B6 condition: {condition}")


def _expected_pair_contract() -> dict[str, Any]:
    return {
        "contrast_unit": CONTRAST_UNIT,
        "replicate_pairing_claim": False,
        "delta_orientation": DELTA_ORIENTATION,
        "native_join_key": list(NATIVE_JOIN_KEY),
        "protein_group_key": PROTEIN_GROUP_KEY,
        "codon_duplicate_policy": CODON_DUPLICATE_POLICY,
        "missingness_policy": MISSINGNESS_POLICY,
        "imputation_allowed": False,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "published_low_b6_functional_rule": dict(PUBLISHED_LOW_B6_FUNCTIONAL_RULE),
        "published_b6_remediability_rule": dict(PUBLISHED_B6_REMEDIABILITY_RULE),
        "expected_structural_overlap": dict(EXPECTED_STRUCTURAL_OVERLAP),
        "uncertainty_rule": UNCERTAINTY_RULE,
        "covariance_status": COVARIANCE_STATUS,
        "outcome_materialization": OUTCOME_MATERIALIZATION_RULE,
    }


def _expected_registry() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PAIR_REGISTRY_ARTIFACT_TYPE,
        "claim_scope": REGISTRY_CLAIM_SCOPE,
        "access_date": "2026-07-26",
        "gene": GENE,
        "target": {
            "gene": GENE,
            "sequence_type": TARGET_SEQUENCE_TYPE,
            "dna_sequence_length": TARGET_DNA_LENGTH,
            "dna_sequence_sha256": TARGET_DNA_SHA256,
            "uniprot_id": TARGET_UNIPROT_ID,
            "protein_sequence_length": TARGET_PROTEIN_LENGTH,
            "protein_sequence_sha256": TARGET_PROTEIN_SHA256,
        },
        "conditions": {
            "low": _expected_condition("low"),
            "high": _expected_condition("high"),
        },
        "pair_contract": _expected_pair_contract(),
        "admission_status": "candidate_not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "blockers": list(REGISTRY_BLOCKERS),
    }


def validate_cbs_b6_pair_registry(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the exact, hash-pinned CBS B6 pair registry."""

    registry = _exact_mapping(registry, REGISTRY_FIELDS, "CBS B6 pair registry")
    observed_hash = canonical_sha256(registry)
    if observed_hash != EXPECTED_PAIR_REGISTRY_SHA256:
        raise CbsB6PairError("CBS B6 pair registry canonical SHA-256 differs")

    _exact_mapping(registry["target"], TARGET_FIELDS, "CBS B6 pair registry.target")
    conditions = _exact_mapping(
        registry["conditions"],
        {"low", "high"},
        "CBS B6 pair registry.conditions",
    )
    for condition in ("low", "high"):
        record = _exact_mapping(
            conditions[condition],
            CONDITION_FIELDS,
            f"CBS B6 pair registry.conditions.{condition}",
        )
        for name in ("metadata", "scores", "counts"):
            body = _exact_mapping(
                record[name],
                BODY_LOCK_FIELDS,
                f"CBS B6 pair registry.conditions.{condition}.{name}",
            )
            _positive_int(
                body["body_bytes"],
                f"CBS B6 pair registry.conditions.{condition}.{name}.body_bytes",
            )
            _sha256(
                body["body_sha256"],
                f"CBS B6 pair registry.conditions.{condition}.{name}.body_sha256",
            )
        source_lock = _exact_mapping(
            record["source_lock"],
            SOURCE_LOCK_FIELDS,
            f"CBS B6 pair registry.conditions.{condition}.source_lock",
        )
        _positive_int(
            source_lock["artifact_bytes"],
            f"CBS B6 pair registry.conditions.{condition}.source_lock.artifact_bytes",
        )
        for field in (
            "artifact_sha256",
            "canonical_json_sha256",
            "source_bundle_sha256",
            "mapping_contract_sha256",
        ):
            _sha256(
                source_lock[field],
                f"CBS B6 pair registry.conditions.{condition}.source_lock.{field}",
            )
        _nonnegative_int(
            source_lock["current_mapping_error_count"],
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.current_mapping_error_count"),
        )
        mapped = _exact_mapping(
            source_lock["mapped_variants"],
            MAPPED_VARIANTS_LOCK_FIELDS,
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.mapped_variants"),
        )
        _positive_int(
            mapped["decoded_byte_count"],
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.mapped_variants.decoded_byte_count"),
        )
        _sha256(
            mapped["sha256"],
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.mapped_variants.sha256"),
        )
        openapi = _exact_mapping(
            source_lock["openapi"],
            OPENAPI_LOCK_FIELDS,
            f"CBS B6 pair registry.conditions.{condition}.source_lock.openapi",
        )
        _positive_int(
            openapi["decoded_byte_count"],
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.openapi.decoded_byte_count"),
        )
        _sha256(
            openapi["sha256"],
            (f"CBS B6 pair registry.conditions.{condition}.source_lock.openapi.sha256"),
        )
    contract = _exact_mapping(
        registry["pair_contract"],
        PAIR_CONTRACT_FIELDS,
        "CBS B6 pair registry.pair_contract",
    )
    _exact_mapping(
        contract["expected_structural_overlap"],
        EXPECTED_OVERLAP_FIELDS,
        "CBS B6 pair registry.pair_contract.expected_structural_overlap",
    )
    _require_exact_json(
        registry,
        _expected_registry(),
        "CBS B6 pair registry exact production contract",
    )
    return registry


def load_cbs_b6_pair_registry(path: str | Path) -> Mapping[str, Any]:
    """Load and validate an exact pair registry JSON file."""

    registry = _load_json_file(path, "CBS B6 pair registry")
    validate_cbs_b6_pair_registry(registry)
    return registry


def validate_cbs_b6_source_lock_bytes(
    source_lock_bytes: bytes,
    *,
    condition: str,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate one exact, registry-pinned low/high CBS source-lock artifact."""

    registry = validate_cbs_b6_pair_registry(pair_registry)
    if condition not in {"low", "high"}:
        raise CbsB6PairError("CBS B6 source-lock condition must be low or high")
    if type(source_lock_bytes) is not bytes:
        raise CbsB6PairError("CBS B6 source-lock artifact must be exact bytes")

    condition_record = _mapping(
        _mapping(registry["conditions"], "CBS B6 pair registry.conditions")[condition],
        f"CBS B6 pair registry.conditions.{condition}",
    )
    expectation = _mapping(
        condition_record["source_lock"],
        f"CBS B6 pair registry.conditions.{condition}.source_lock",
    )
    if (
        len(source_lock_bytes) != expectation["artifact_bytes"]
        or _bytes_sha256(source_lock_bytes) != expectation["artifact_sha256"]
    ):
        raise CbsB6PairError(f"CBS {condition}-B6 source-lock artifact differs from the registry")

    lock = _mapping(
        _parse_json_bytes(
            source_lock_bytes,
            f"CBS {condition}-B6 source-lock artifact",
        ),
        f"CBS {condition}-B6 source-lock artifact",
    )
    if canonical_json_bytes(lock) + b"\n" != source_lock_bytes:
        raise CbsB6PairError(f"CBS {condition}-B6 source-lock artifact must use exact canonical JSON")
    try:
        msl.validate_source_lock(lock)
    except msl.SourceLockError as exc:
        raise CbsB6PairError(f"CBS {condition}-B6 source-lock artifact is invalid") from exc

    _require_exact_json(
        {
            "artifact_type": lock["artifact_type"],
            "canonical_json_sha256": canonical_sha256(lock),
            "source_bundle_sha256": lock["source_bundle_sha256"],
            "urn": lock["urn"],
            "readiness_state": _mapping(
                lock["readiness"],
                f"CBS {condition}-B6 source-lock readiness",
            )["state"],
            "ingestion_status": lock["ingestion_status"],
            "outcome_status": lock["outcome_status"],
        },
        {
            "artifact_type": expectation["artifact_type"],
            "canonical_json_sha256": expectation["canonical_json_sha256"],
            "source_bundle_sha256": expectation["source_bundle_sha256"],
            "urn": condition_record["urn"],
            "readiness_state": expectation["readiness_state"],
            "ingestion_status": "not_ingested",
            "outcome_status": "not_derived",
        },
        f"CBS {condition}-B6 source-lock identity",
    )

    source_artifacts = _mapping(
        lock["source_artifacts"],
        f"CBS {condition}-B6 source-lock source_artifacts",
    )
    for name in ("metadata", "scores", "counts"):
        artifact = _mapping(
            source_artifacts[name],
            f"CBS {condition}-B6 source-lock source_artifacts.{name}",
        )
        body_lock = _mapping(
            condition_record[name],
            f"CBS B6 pair registry.conditions.{condition}.{name}",
        )
        _require_exact_json(
            {
                "decoded_byte_count": artifact["decoded_byte_count"],
                "sha256": artifact["sha256"],
            },
            {
                "decoded_byte_count": body_lock["body_bytes"],
                "sha256": body_lock["body_sha256"],
            },
            f"CBS {condition}-B6 source-lock {name} body identity",
        )

    mapped_artifact = _mapping(
        source_artifacts["mapped_variants"],
        f"CBS {condition}-B6 source-lock source_artifacts.mapped_variants",
    )
    openapi_artifact = _mapping(
        _mapping(lock["api"], f"CBS {condition}-B6 source-lock api")["openapi_artifact"],
        f"CBS {condition}-B6 source-lock api.openapi_artifact",
    )
    _require_exact_json(
        {
            "decoded_byte_count": mapped_artifact["decoded_byte_count"],
            "sha256": mapped_artifact["sha256"],
        },
        expectation["mapped_variants"],
        f"CBS {condition}-B6 mapped-variants identity",
    )
    _require_exact_json(
        {
            "version": _mapping(
                lock["api"],
                f"CBS {condition}-B6 source-lock api",
            )["openapi_version"],
            "decoded_byte_count": openapi_artifact["decoded_byte_count"],
            "sha256": openapi_artifact["sha256"],
        },
        expectation["openapi"],
        f"CBS {condition}-B6 OpenAPI identity",
    )

    mapping_contract = _mapping(
        lock["mapping_contract"],
        f"CBS {condition}-B6 source-lock mapping_contract",
    )
    _require_exact_json(
        {
            "mapping_contract_sha256": canonical_sha256(mapping_contract),
            "current_mapping_error_count": mapping_contract["current_error_count"],
            "current_record_count": mapping_contract["current_record_count"],
        },
        {
            "mapping_contract_sha256": expectation["mapping_contract_sha256"],
            "current_mapping_error_count": expectation["current_mapping_error_count"],
            "current_record_count": condition_record["row_count"],
        },
        f"CBS {condition}-B6 mapping contract",
    )
    metadata_contract = _mapping(
        lock["metadata_contract"],
        f"CBS {condition}-B6 source-lock metadata_contract",
    )
    targets = metadata_contract.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        raise CbsB6PairError(f"CBS {condition}-B6 source-lock must contain exactly one target")
    target = _mapping(
        targets[0],
        f"CBS {condition}-B6 source-lock target",
    )
    registry_target = _mapping(registry["target"], "CBS B6 pair registry.target")
    _require_exact_json(
        {
            "title": metadata_contract["title"],
            "num_variants": metadata_contract["num_variants"],
            "count_columns": metadata_contract["dataset_columns"]["countColumns"],
            "score_columns": metadata_contract["dataset_columns"]["scoreColumns"],
            "target_name": target["name"],
            "sequence_length": target["sequence_length"],
            "sequence_sha256": target["sequence_sha256"],
        },
        {
            "title": condition_record["title"],
            "num_variants": condition_record["row_count"],
            "count_columns": condition_record["count_columns"],
            "score_columns": condition_record["score_columns"],
            "target_name": registry_target["gene"],
            "sequence_length": registry_target["dna_sequence_length"],
            "sequence_sha256": registry_target["dna_sequence_sha256"],
        },
        f"CBS {condition}-B6 source-lock registry cross-binding",
    )
    return lock


def load_cbs_b6_source_lock(
    path: str | Path,
    *,
    condition: str,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load one exact low/high CBS source-lock artifact."""

    try:
        source_lock_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise CbsB6PairError(f"cannot read CBS {condition}-B6 source-lock artifact") from exc
    return validate_cbs_b6_source_lock_bytes(
        source_lock_bytes,
        condition=condition,
        pair_registry=pair_registry,
    )


def _normalized_hgvs(
    raw: str,
    *,
    context: str,
    required: bool,
) -> str | None:
    if raw not in SCORE_MISSING_TOKENS and raw != raw.strip():
        raise CbsB6PairError(f"{context} cannot contain surrounding whitespace")
    if raw in SCORE_MISSING_TOKENS:
        if required:
            raise CbsB6PairError(f"{context} cannot be missing")
        return None
    if not raw:
        if required:
            raise CbsB6PairError(f"{context} cannot be empty")
        return None
    return raw


def _parse_numeric_cell(
    raw: str,
    *,
    context: str,
    nonnegative: bool,
) -> tuple[str | None, Decimal | None]:
    if raw in SCORE_MISSING_TOKENS:
        return None, None
    if raw != raw.strip() or DECIMAL_PATTERN.fullmatch(raw) is None:
        raise CbsB6PairError(f"{context} must be a finite decimal or exact missing token")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise CbsB6PairError(f"{context} must be a finite decimal or exact missing token") from exc
    if not value.is_finite():
        raise CbsB6PairError(f"{context} cannot be nonfinite")
    if nonnegative and value < 0:
        raise CbsB6PairError(f"{context} cannot be negative")
    return raw, value


def _validate_codon_duplicates(
    rows: Sequence[_ScoreRow],
    *,
    condition: str,
) -> None:
    grouped: dict[str, list[_ScoreRow]] = defaultdict(list)
    for row in rows:
        if row.hgvs_pro is not None:
            grouped[row.hgvs_pro].append(row)
    for protein, group in grouped.items():
        if len(group) < 2:
            continue
        first = group[0].raw_values
        for row in group[1:]:
            if row.raw_values != first:
                differing = [
                    column
                    for column, left, right in zip(
                        SCORE_COLUMNS,
                        first,
                        row.raw_values,
                        strict=True,
                    )
                    if left != right
                ]
                raise CbsB6PairError(
                    f"{condition} codon duplicates for {protein!r} disagree "
                    f"in exact missingness or values for {differing}"
                )


def _parse_score_body(
    body: bytes,
    *,
    condition: str,
    urn: str,
    expected_rows: int,
) -> _ScoreTable:
    if type(body) is not bytes:
        raise CbsB6PairError(f"{condition} score body must be exact decoded bytes")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CbsB6PairError(f"{condition} score body must be UTF-8 CSV") from exc
    if "\x00" in text:
        raise CbsB6PairError(f"{condition} score body cannot contain NUL bytes")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CbsB6PairError(f"{condition} score body cannot be empty") from exc
    expected_header = [*msl.FIXED_VARIANT_COLUMNS, *SCORE_COLUMNS]
    if header != expected_header:
        raise CbsB6PairError(f"{condition} score header differs; observed={header}, expected={expected_header}")
    if len(header) != len(set(header)):
        raise CbsB6PairError(f"{condition} score header cannot contain duplicates")

    rows: list[_ScoreRow] = []
    accessions: list[str] = []
    accession_set: set[str] = set()
    native_keys: set[str] = set()
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise CbsB6PairError(
                f"{condition} score row {row_number} is truncated or overlong: "
                f"observed={len(values)}, expected={len(header)}"
            )
        record = dict(zip(header, values, strict=True))
        accession = record["accession"]
        if not accession.startswith(f"{urn}#"):
            raise CbsB6PairError(f"{condition} score row {row_number} has an invalid accession")
        if accession in accession_set:
            raise CbsB6PairError(f"{condition} score accessions must be unique")
        accession_set.add(accession)
        accessions.append(accession)

        hgvs_nt = _normalized_hgvs(
            record["hgvs_nt"],
            context=f"{condition} score row {row_number}.hgvs_nt",
            required=True,
        )
        assert hgvs_nt is not None
        hgvs_splice = _normalized_hgvs(
            record["hgvs_splice"],
            context=f"{condition} score row {row_number}.hgvs_splice",
            required=False,
        )
        hgvs_pro = _normalized_hgvs(
            record["hgvs_pro"],
            context=f"{condition} score row {row_number}.hgvs_pro",
            required=False,
        )
        native_key = hgvs_nt
        if native_key in native_keys:
            raise CbsB6PairError(f"{condition} score native join keys must be unique; duplicate={native_key!r}")
        native_keys.add(native_key)

        raw_score, score = _parse_numeric_cell(
            record["score"],
            context=f"{condition} score row {row_number}.score",
            nonnegative=False,
        )
        raw_sd, sd = _parse_numeric_cell(
            record["sd"],
            context=f"{condition} score row {row_number}.sd",
            nonnegative=True,
        )
        raw_se, se = _parse_numeric_cell(
            record["se"],
            context=f"{condition} score row {row_number}.se",
            nonnegative=True,
        )
        rows.append(
            _ScoreRow(
                native_key=native_key,
                hgvs_splice=hgvs_splice,
                hgvs_pro=hgvs_pro,
                raw_values=(raw_score, raw_sd, raw_se),
                values=(score, sd, se),
            )
        )

    if len(rows) != expected_rows:
        raise CbsB6PairError(f"{condition} score row count differs: observed={len(rows)}, expected={expected_rows}")
    _validate_codon_duplicates(rows, condition=condition)
    complete_count = sum(all(value is not None for value in row.values) for row in rows)
    return _ScoreTable(
        rows=tuple(rows),
        by_native_key={row.native_key: row for row in rows},
        accessions=tuple(accessions),
        complete_measured_row_count=complete_count,
        incomplete_measured_row_count=len(rows) - complete_count,
    )


def _validate_exact_score_body(
    body: bytes,
    *,
    condition: str,
) -> _ScoreTable:
    if condition == "low":
        expected_bytes = LOW_SCORES_BODY_BYTES
        expected_sha256 = LOW_SCORES_BODY_SHA256
        urn = LOW_URN
        row_count = LOW_ROW_COUNT
    elif condition == "high":
        expected_bytes = HIGH_SCORES_BODY_BYTES
        expected_sha256 = HIGH_SCORES_BODY_SHA256
        urn = HIGH_URN
        row_count = HIGH_ROW_COUNT
    else:
        raise AssertionError(f"unknown CBS B6 condition: {condition}")
    if type(body) is not bytes:
        raise CbsB6PairError(f"{condition} score body must be exact decoded bytes")
    if len(body) != expected_bytes or _bytes_sha256(body) != expected_sha256:
        raise CbsB6PairError(f"{condition} score body differs from the hash-pinned pair registry")
    return _parse_score_body(
        body,
        condition=condition,
        urn=urn,
        expected_rows=row_count,
    )


def _validate_high_metadata_body(body: bytes) -> None:
    if type(body) is not bytes:
        raise CbsB6PairError("high metadata body must be exact decoded bytes")
    if len(body) != HIGH_METADATA_BODY_BYTES or _bytes_sha256(body) != HIGH_METADATA_BODY_SHA256:
        raise CbsB6PairError("high metadata body differs from the hash-pinned pair registry")
    metadata = _mapping(
        _parse_json_bytes(body, "high metadata body"),
        "high metadata body",
    )
    exact_identity = {
        "urn": HIGH_URN,
        "title": HIGH_TITLE,
        "numVariants": HIGH_ROW_COUNT,
        "private": False,
        "processingState": "success",
        "supersededScoreSet": None,
        "supersedingScoreSet": None,
        "dataUsagePolicy": None,
    }
    for field, expected in exact_identity.items():
        if metadata.get(field) != expected:
            raise CbsB6PairError(f"high metadata body.{field} differs")

    dataset_columns = _exact_mapping(
        metadata.get("datasetColumns"),
        msl.DATASET_COLUMNS_FIELDS,
        "high metadata body.datasetColumns",
    )
    if dataset_columns != {
        "scoreColumns": list(SCORE_COLUMNS),
        "countColumns": list(HIGH_COUNT_COLUMNS),
        "recordType": "DatasetColumns",
    }:
        raise CbsB6PairError("high metadata dataset columns differ")

    expected_license = {
        "id": 1,
        "shortName": "CC0",
        "longName": "CC0 (Public domain)",
        "active": True,
        "link": "https://creativecommons.org/publicdomain/zero/1.0/",
        "version": "1.0",
        "recordType": "ShortLicense",
    }
    license_value = _exact_mapping(
        metadata.get("license"),
        msl.LICENSE_FIELDS,
        "high metadata body.license",
    )
    if license_value != expected_license:
        raise CbsB6PairError("high metadata license differs")

    calibrations = metadata.get("scoreCalibrations")
    if not isinstance(calibrations, list):
        raise CbsB6PairError("high metadata score calibrations must be a list")
    targets = metadata.get("targetGenes")
    if not isinstance(targets, list) or len(targets) != 1:
        raise CbsB6PairError("high metadata must contain exactly one target")
    target = _mapping(targets[0], "high metadata targetGenes[0]")
    target_sequence = _mapping(
        target.get("targetSequence"),
        "high metadata targetGenes[0].targetSequence",
    )
    sequence = target_sequence.get("sequence")
    if (
        target.get("name") != GENE
        or target.get("mappedHgncName") != GENE
        or target.get("targetAccession") is not None
        or target_sequence.get("sequenceType") != TARGET_SEQUENCE_TYPE
        or not isinstance(sequence, str)
        or len(sequence) != TARGET_DNA_LENGTH
    ):
        raise CbsB6PairError("high metadata target identity differs")
    try:
        sequence_sha256 = _bytes_sha256(sequence.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise CbsB6PairError("high metadata target DNA must be ASCII") from exc
    if sequence_sha256 != TARGET_DNA_SHA256:
        raise CbsB6PairError("high metadata target DNA hash differs")
    identifiers = target.get("externalIdentifiers")
    if not isinstance(identifiers, list):
        raise CbsB6PairError("high metadata target external identifiers must be a list")
    uniprot_matches = []
    for index, raw_entry in enumerate(identifiers):
        entry = _mapping(
            raw_entry,
            f"high metadata targetGenes[0].externalIdentifiers[{index}]",
        )
        identifier = _mapping(
            entry.get("identifier"),
            f"high metadata targetGenes[0].externalIdentifiers[{index}].identifier",
        )
        if (
            identifier.get("dbName") == "UniProt"
            and identifier.get("identifier") == TARGET_UNIPROT_ID
            and entry.get("offset") == 0
        ):
            uniprot_matches.append(entry)
    if len(uniprot_matches) != 1:
        raise CbsB6PairError("high metadata UniProt identity differs")


def _validate_high_count_body(
    body: bytes,
    *,
    scores: _ScoreTable,
) -> _CountTable:
    if type(body) is not bytes:
        raise CbsB6PairError("high count body must be exact decoded bytes")
    if len(body) != HIGH_COUNTS_BODY_BYTES or _bytes_sha256(body) != HIGH_COUNTS_BODY_SHA256:
        raise CbsB6PairError("high count body differs from the hash-pinned pair registry")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CbsB6PairError("high count body must be UTF-8 CSV") from exc
    if "\x00" in text:
        raise CbsB6PairError("high count body cannot contain NUL bytes")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CbsB6PairError("high count body cannot be empty") from exc
    expected_header = [*msl.FIXED_VARIANT_COLUMNS, *HIGH_COUNT_COLUMNS]
    if header != expected_header:
        raise CbsB6PairError(f"high count header differs; observed={header}, expected={expected_header}")
    if len(header) != len(set(header)):
        raise CbsB6PairError("high count header cannot contain duplicates")

    accessions: list[str] = []
    finite_by_column = {column: 0 for column in HIGH_COUNT_COLUMNS}
    missing_count = 0
    nonmissing_count = 0
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise CbsB6PairError(
                f"high count row {row_number} is truncated or overlong: observed={len(values)}, expected={len(header)}"
            )
        record = dict(zip(header, values, strict=True))
        row_index = row_number - 2
        if row_index >= len(scores.rows):
            raise CbsB6PairError("high count row count exceeds high score row count")
        accession = record["accession"]
        if accession != scores.accessions[row_index]:
            raise CbsB6PairError(f"high score/count accession order differs at row {row_number}")
        accessions.append(accession)
        score_row = scores.rows[row_index]
        count_native_key = _normalized_hgvs(
            record["hgvs_nt"],
            context=f"high count row {row_number}.hgvs_nt",
            required=True,
        )
        count_hgvs_splice = _normalized_hgvs(
            record["hgvs_splice"],
            context=f"high count row {row_number}.hgvs_splice",
            required=False,
        )
        count_hgvs_pro = _normalized_hgvs(
            record["hgvs_pro"],
            context=f"high count row {row_number}.hgvs_pro",
            required=False,
        )
        if (
            count_native_key != score_row.native_key
            or count_hgvs_splice != score_row.hgvs_splice
            or count_hgvs_pro != score_row.hgvs_pro
        ):
            raise CbsB6PairError(f"high score/count fixed variant identity differs at row {row_number}")

        for column in HIGH_COUNT_COLUMNS:
            raw = record[column]
            if raw in COUNT_MISSING_TOKENS:
                missing_count += 1
                continue
            if raw != raw.strip() or DECIMAL_PATTERN.fullmatch(raw) is None:
                raise CbsB6PairError(
                    f"high count row {row_number}.{column} must be a finite nonnegative decimal or exact NA"
                )
            try:
                numeric = Decimal(raw)
            except InvalidOperation as exc:
                raise CbsB6PairError(
                    f"high count row {row_number}.{column} must be a finite nonnegative decimal or exact NA"
                ) from exc
            if not numeric.is_finite() or numeric < 0:
                raise CbsB6PairError(
                    f"high count row {row_number}.{column} must be a finite nonnegative decimal or exact NA"
                )
            finite_by_column[column] += 1
            nonmissing_count += 1

    if len(accessions) != HIGH_ROW_COUNT or len(accessions) != len(scores.rows):
        raise CbsB6PairError("high count row count must equal the frozen high score row count")
    empty_columns = sorted(column for column, count in finite_by_column.items() if count == 0)
    if empty_columns:
        raise CbsB6PairError(f"high count columns have no observed decimal values: {empty_columns}")
    return _CountTable(
        accessions=tuple(accessions),
        row_count=len(accessions),
        nonmissing_count_cell_count=nonmissing_count,
        missing_count_cell_count=missing_count,
    )


def _empty_high_offline() -> dict[str, Any]:
    return {
        "pair_registry_status": "validated_exact_canonical_hash",
        "core_bodies_status": "not_supplied",
        "metadata_body_sha256": None,
        "score_body_sha256": None,
        "count_body_sha256": None,
        "score_row_count": None,
        "count_row_count": None,
        "score_columns": list(SCORE_COLUMNS),
        "count_columns": list(HIGH_COUNT_COLUMNS),
        "complete_measured_row_count": None,
        "incomplete_measured_row_count": None,
        "codon_duplicate_status": "not_evaluated",
        "metadata_contract_status": "not_evaluated",
        "score_count_accession_order_status": "not_evaluated",
        "count_value_status": "not_evaluated",
        "nonmissing_count_cell_count": None,
        "missing_count_cell_count": None,
        "imputation_performed": False,
        "outcomes_materialized": False,
    }


def _validate_high_inputs(
    inputs: CbsHighB6AdapterInputs,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(inputs, CbsHighB6AdapterInputs):
        raise TypeError("inputs must be CbsHighB6AdapterInputs")
    registry = validate_cbs_b6_pair_registry(inputs.pair_registry)
    optional_bodies = (
        inputs.high_metadata_body,
        inputs.high_scores_body,
        inputs.high_counts_body,
    )
    supplied = tuple(body is not None for body in optional_bodies)
    if any(supplied) and not all(supplied):
        raise CbsB6PairError("high-B6 metadata, scores, and counts bodies must be supplied together")
    if not any(supplied):
        return registry, _empty_high_offline()
    assert inputs.high_metadata_body is not None
    assert inputs.high_scores_body is not None
    assert inputs.high_counts_body is not None
    _validate_high_metadata_body(inputs.high_metadata_body)
    scores = _validate_exact_score_body(inputs.high_scores_body, condition="high")
    counts = _validate_high_count_body(inputs.high_counts_body, scores=scores)
    return registry, {
        "pair_registry_status": "validated_exact_canonical_hash",
        "core_bodies_status": "validated_structural_only",
        "metadata_body_sha256": _bytes_sha256(inputs.high_metadata_body),
        "score_body_sha256": _bytes_sha256(inputs.high_scores_body),
        "count_body_sha256": _bytes_sha256(inputs.high_counts_body),
        "score_row_count": len(scores.rows),
        "count_row_count": counts.row_count,
        "score_columns": list(SCORE_COLUMNS),
        "count_columns": list(HIGH_COUNT_COLUMNS),
        "complete_measured_row_count": scores.complete_measured_row_count,
        "incomplete_measured_row_count": scores.incomplete_measured_row_count,
        "codon_duplicate_status": ("validated_exact_score_sd_se_missingness_and_values_within_hgvs_pro"),
        "metadata_contract_status": ("validated_active_public_cc0_exact_schema_target_and_sequence"),
        "score_count_accession_order_status": ("validated_exact_order_and_fixed_variant_identity"),
        "count_value_status": "validated_finite_nonnegative_decimal_or_exact_NA",
        "nonmissing_count_cell_count": counts.nonmissing_count_cell_count,
        "missing_count_cell_count": counts.missing_count_cell_count,
        "imputation_performed": False,
        "outcomes_materialized": False,
    }


def _status_payload(status: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(status)
    payload.pop("status_sha256", None)
    return payload


def status_sha256(status: Mapping[str, Any]) -> str:
    """Hash a status without its self-hash field."""

    return canonical_sha256(_status_payload(status))


def build_cbs_high_b6_adapter_status(
    inputs: CbsHighB6AdapterInputs,
) -> dict[str, Any]:
    """Build a deterministic, candidate-only high-B6 adapter status."""

    _, offline = _validate_high_inputs(inputs)
    structural = offline["core_bodies_status"] == "validated_structural_only"
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": HIGH_STATUS_ARTIFACT_TYPE,
        "adapter_id": HIGH_ADAPTER_ID,
        "claim_scope": HIGH_STATUS_CLAIM_SCOPE,
        "urn": HIGH_URN,
        "gene": GENE,
        "title": HIGH_TITLE,
        "dose_definition": HIGH_DOSE_DEFINITION,
        "pair_registry_sha256": EXPECTED_PAIR_REGISTRY_SHA256,
        "source_readiness": "COUNT_LINEAGE_PARTIAL",
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "registry_expectations": _expected_condition("high"),
        "offline_validation": dict(offline),
        "count_measurement_semantics": dict(HIGH_COUNT_MEASUREMENT_SEMANTICS),
        "score_measurement_semantics": dict(HIGH_SCORE_MEASUREMENT_SEMANTICS),
        "active_blocker_codes": list(HIGH_BLOCKER_CODES),
        "prohibited_reinterpretations": list(HIGH_PROHIBITED_REINTERPRETATIONS),
        "status_sha256": "",
    }
    status["status_sha256"] = status_sha256(status)
    validate_cbs_high_b6_adapter_status(
        status,
        source_inputs=inputs if structural else None,
    )
    return status


def validate_cbs_high_b6_adapter_status(
    status: Mapping[str, Any],
    *,
    source_inputs: CbsHighB6AdapterInputs | None = None,
) -> None:
    """Validate a high-B6 status, replaying exact bytes for structural claims."""

    status = _exact_mapping(status, HIGH_STATUS_FIELDS, "CBS high-B6 adapter status")
    exact_values = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": HIGH_STATUS_ARTIFACT_TYPE,
        "adapter_id": HIGH_ADAPTER_ID,
        "claim_scope": HIGH_STATUS_CLAIM_SCOPE,
        "urn": HIGH_URN,
        "gene": GENE,
        "title": HIGH_TITLE,
        "dose_definition": HIGH_DOSE_DEFINITION,
        "pair_registry_sha256": EXPECTED_PAIR_REGISTRY_SHA256,
        "source_readiness": "COUNT_LINEAGE_PARTIAL",
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
    }
    for field, expected in exact_values.items():
        _require_exact_json(
            status[field],
            expected,
            f"CBS high-B6 adapter status.{field}",
        )
    if status["confirmatory_eligible"] is not False or status["automatic_promotion"] is not False:
        raise CbsB6PairError("CBS high-B6 boolean claim boundary differs")
    _require_exact_json(
        status["registry_expectations"],
        _expected_condition("high"),
        "CBS high-B6 registry expectations",
    )
    _require_exact_json(
        status["active_blocker_codes"],
        list(HIGH_BLOCKER_CODES),
        "CBS high-B6 blocker codes",
    )
    _require_exact_json(
        status["count_measurement_semantics"],
        HIGH_COUNT_MEASUREMENT_SEMANTICS,
        "CBS high-B6 count measurement semantics",
    )
    count_semantics = _mapping(
        status["count_measurement_semantics"],
        "CBS high-B6 count measurement semantics",
    )
    if (
        count_semantics.get("raw_read_count_claim") is not False
        or count_semantics.get("functional_wt_baseline_claim") is not False
    ):
        raise CbsB6PairError("CBS high-B6 count claims must be boolean false")
    _require_exact_json(
        status["score_measurement_semantics"],
        HIGH_SCORE_MEASUREMENT_SEMANTICS,
        "CBS high-B6 score measurement semantics",
    )
    score_semantics = _mapping(
        status["score_measurement_semantics"],
        "CBS high-B6 score measurement semantics",
    )
    if (
        score_semantics.get("score_sd_se_as_replicates") is not False
        or score_semantics.get("replicate_count_claim") is not False
        or type(score_semantics.get("sd_se_squared_ratio")) is not int
    ):
        raise CbsB6PairError("CBS high-B6 score replicate claims differ")
    _require_exact_json(
        status["prohibited_reinterpretations"],
        list(HIGH_PROHIBITED_REINTERPRETATIONS),
        "CBS high-B6 prohibited reinterpretations",
    )

    offline = _exact_mapping(
        status["offline_validation"],
        HIGH_OFFLINE_FIELDS,
        "CBS high-B6 adapter status.offline_validation",
    )
    if (
        offline["pair_registry_status"] != "validated_exact_canonical_hash"
        or offline["score_columns"] != list(SCORE_COLUMNS)
        or offline["count_columns"] != list(HIGH_COUNT_COLUMNS)
        or offline["imputation_performed"] is not False
        or offline["outcomes_materialized"] is not False
    ):
        raise CbsB6PairError("CBS high-B6 offline validation boundary differs")
    if offline["core_bodies_status"] == "not_supplied":
        if source_inputs is not None and any(
            body is not None
            for body in (
                source_inputs.high_metadata_body,
                source_inputs.high_scores_body,
                source_inputs.high_counts_body,
            )
        ):
            raise CbsB6PairError("unsupplied high core status conflicts with supplied bytes")
        _require_exact_json(
            offline,
            _empty_high_offline(),
            "unsupplied high core status",
        )
    elif offline["core_bodies_status"] == "validated_structural_only":
        if source_inputs is None or any(
            body is None
            for body in (
                source_inputs.high_metadata_body,
                source_inputs.high_scores_body,
                source_inputs.high_counts_body,
            )
        ):
            raise CbsB6PairError(
                "validated high core status requires exact metadata, score, and count bytes for replay"
            )
        _, replayed = _validate_high_inputs(source_inputs)
        _require_exact_json(
            offline,
            replayed,
            "high-B6 offline status replayed source bytes",
        )
        if (
            offline["metadata_body_sha256"] != HIGH_METADATA_BODY_SHA256
            or offline["score_body_sha256"] != HIGH_SCORES_BODY_SHA256
            or offline["count_body_sha256"] != HIGH_COUNTS_BODY_SHA256
        ):
            raise CbsB6PairError("validated high core body digests differ")
        if offline["score_row_count"] != HIGH_ROW_COUNT or offline["count_row_count"] != HIGH_ROW_COUNT:
            raise CbsB6PairError("validated high score/count row counts differ")
        complete = _nonnegative_int(
            offline["complete_measured_row_count"],
            "CBS high-B6 complete_measured_row_count",
        )
        incomplete = _nonnegative_int(
            offline["incomplete_measured_row_count"],
            "CBS high-B6 incomplete_measured_row_count",
        )
        if complete + incomplete != HIGH_ROW_COUNT:
            raise CbsB6PairError("high-B6 measured-row totals differ")
        nonmissing_counts = _nonnegative_int(
            offline["nonmissing_count_cell_count"],
            "CBS high-B6 nonmissing_count_cell_count",
        )
        missing_counts = _nonnegative_int(
            offline["missing_count_cell_count"],
            "CBS high-B6 missing_count_cell_count",
        )
        if nonmissing_counts + missing_counts != HIGH_ROW_COUNT * len(HIGH_COUNT_COLUMNS):
            raise CbsB6PairError("high-B6 count-cell totals differ")
        expected_statuses = {
            "codon_duplicate_status": ("validated_exact_score_sd_se_missingness_and_values_within_hgvs_pro"),
            "metadata_contract_status": ("validated_active_public_cc0_exact_schema_target_and_sequence"),
            "score_count_accession_order_status": ("validated_exact_order_and_fixed_variant_identity"),
            "count_value_status": ("validated_finite_nonnegative_decimal_or_exact_NA"),
        }
        for field, expected in expected_statuses.items():
            if offline[field] != expected:
                raise CbsB6PairError(f"high-B6 {field} differs")
    else:
        raise CbsB6PairError("high core_bodies_status must be not_supplied or validated_structural_only")
    if _sha256(status["status_sha256"], "CBS high-B6 status_sha256") != status_sha256(status):
        raise CbsB6PairError("CBS high-B6 adapter status self-hash differs")


def _empty_pair_offline() -> dict[str, Any]:
    return {
        "pair_registry_status": "validated_exact_canonical_hash",
        "target_identity_status": ("registry_audited_both_conditions_runtime_not_replayed"),
        "high_core_bodies_status": "not_supplied",
        "low_score_body_status": "not_supplied",
        "high_score_body_status": "not_supplied",
        "low_score_body_sha256": None,
        "high_score_body_sha256": None,
        "low_score_row_count": None,
        "high_score_row_count": None,
        "native_join_key": list(NATIVE_JOIN_KEY),
        "accession_join_used": False,
        "native_join_status": "not_evaluated",
        "native_join_row_count": None,
        "low_only_row_count": None,
        "high_only_row_count": None,
        "native_join_unique_hgvs_pro_count": None,
        "condition_set_intersection_unique_hgvs_pro_count": None,
        "complete_paired_native_row_count": None,
        "complete_paired_unique_hgvs_pro_count": None,
        "codon_duplicate_status": "not_evaluated",
        "missingness_policy": MISSINGNESS_POLICY,
        "imputation_performed": False,
        "delta_values_materialized": False,
        "uncertainty_status": "not_evaluated_covariance_required",
    }


def validate_cbs_b6_pair_inputs(
    inputs: CbsB6PairInputs,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the registry and optionally replay the exact score-body join."""

    if not isinstance(inputs, CbsB6PairInputs):
        raise TypeError("inputs must be CbsB6PairInputs")
    registry = validate_cbs_b6_pair_registry(inputs.pair_registry)
    supplied = (
        inputs.low_scores_body is not None,
        inputs.high_metadata_body is not None,
        inputs.high_scores_body is not None,
        inputs.high_counts_body is not None,
    )
    if any(supplied) and not all(supplied):
        raise CbsB6PairError("low-B6 scores and high-B6 metadata, scores, and counts bodies must be supplied together")
    if not any(supplied):
        return registry, _empty_pair_offline()

    assert inputs.low_scores_body is not None
    assert inputs.high_metadata_body is not None
    assert inputs.high_scores_body is not None
    assert inputs.high_counts_body is not None
    _, high_offline = _validate_high_inputs(
        CbsHighB6AdapterInputs(
            pair_registry=inputs.pair_registry,
            high_metadata_body=inputs.high_metadata_body,
            high_scores_body=inputs.high_scores_body,
            high_counts_body=inputs.high_counts_body,
        )
    )
    low = _validate_exact_score_body(inputs.low_scores_body, condition="low")
    high = _validate_exact_score_body(inputs.high_scores_body, condition="high")

    low_keys = set(low.by_native_key)
    high_keys = set(high.by_native_key)
    overlap_keys = low_keys & high_keys
    low_only = low_keys - high_keys
    high_only = high_keys - low_keys
    low_proteins = {row.hgvs_pro for row in low.rows if row.hgvs_pro is not None}
    high_proteins = {row.hgvs_pro for row in high.rows if row.hgvs_pro is not None}
    condition_protein_intersection = low_proteins & high_proteins
    overlap_proteins: set[str] = set()
    complete_pair_keys: set[str] = set()
    complete_pair_proteins: set[str] = set()
    for native_key in overlap_keys:
        low_row = low.by_native_key[native_key]
        high_row = high.by_native_key[native_key]
        if low_row.hgvs_splice != high_row.hgvs_splice or low_row.hgvs_pro != high_row.hgvs_pro:
            raise CbsB6PairError(
                "matched hgvs_nt rows disagree in hgvs_splice or hgvs_pro; "
                f"key={native_key!r}, "
                f"low_splice={low_row.hgvs_splice!r}, "
                f"high_splice={high_row.hgvs_splice!r}, "
                f"low_pro={low_row.hgvs_pro!r}, high_pro={high_row.hgvs_pro!r}"
            )
        if low_row.hgvs_pro is not None:
            overlap_proteins.add(low_row.hgvs_pro)
        if all(value is not None for value in (*low_row.values, *high_row.values)):
            complete_pair_keys.add(native_key)
            if low_row.hgvs_pro is not None:
                complete_pair_proteins.add(low_row.hgvs_pro)

    observed_overlap = {
        "native_join_row_count": len(overlap_keys),
        "low_only_row_count": len(low_only),
        "high_only_row_count": len(high_only),
        "native_join_unique_hgvs_pro_count": len(overlap_proteins),
        "condition_set_intersection_unique_hgvs_pro_count": len(condition_protein_intersection),
    }
    if observed_overlap != EXPECTED_STRUCTURAL_OVERLAP:
        raise CbsB6PairError(
            "replayed CBS B6 structural overlap differs from the frozen registry; "
            f"observed={observed_overlap}, expected={EXPECTED_STRUCTURAL_OVERLAP}"
        )

    return registry, {
        "pair_registry_status": "validated_exact_canonical_hash",
        "target_identity_status": ("registry_audited_both_conditions_high_runtime_validated_low_runtime_not_replayed"),
        "high_core_bodies_status": high_offline["core_bodies_status"],
        "low_score_body_status": "validated_structural_only",
        "high_score_body_status": "validated_structural_only",
        "low_score_body_sha256": _bytes_sha256(inputs.low_scores_body),
        "high_score_body_sha256": _bytes_sha256(inputs.high_scores_body),
        "low_score_row_count": len(low.rows),
        "high_score_row_count": len(high.rows),
        "native_join_key": list(NATIVE_JOIN_KEY),
        "accession_join_used": False,
        "native_join_status": ("validated_accession_independent_primary_hgvs_nt_intersection_with_exact_annotations"),
        "native_join_row_count": len(overlap_keys),
        "low_only_row_count": len(low_only),
        "high_only_row_count": len(high_only),
        "native_join_unique_hgvs_pro_count": len(overlap_proteins),
        "condition_set_intersection_unique_hgvs_pro_count": len(condition_protein_intersection),
        "complete_paired_native_row_count": len(complete_pair_keys),
        "complete_paired_unique_hgvs_pro_count": len(complete_pair_proteins),
        "codon_duplicate_status": ("validated_exact_score_sd_se_missingness_and_values_within_each_condition_hgvs_pro"),
        "missingness_policy": MISSINGNESS_POLICY,
        "imputation_performed": False,
        "delta_values_materialized": False,
        "uncertainty_status": "not_derived_covariance_or_joint_bootstrap_required",
    }


def build_cbs_b6_pair_status(inputs: CbsB6PairInputs) -> dict[str, Any]:
    """Build a deterministic, non-outcome CBS low/high-B6 pair status."""

    _, offline = validate_cbs_b6_pair_inputs(inputs)
    structural = offline["native_join_status"] != "not_evaluated"
    high_inputs = CbsHighB6AdapterInputs(
        pair_registry=inputs.pair_registry,
        high_metadata_body=inputs.high_metadata_body,
        high_scores_body=inputs.high_scores_body,
        high_counts_body=inputs.high_counts_body,
    )
    high_status = build_cbs_high_b6_adapter_status(high_inputs)
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PAIR_STATUS_ARTIFACT_TYPE,
        "contract_id": PAIR_CONTRACT_ID,
        "claim_scope": PAIR_STATUS_CLAIM_SCOPE,
        "gene": GENE,
        "pair_registry_sha256": EXPECTED_PAIR_REGISTRY_SHA256,
        "conditions": {
            "low_urn": LOW_URN,
            "high_urn": HIGH_URN,
            "low_dose": LOW_DOSE_DEFINITION,
            "high_dose": HIGH_DOSE_DEFINITION,
        },
        "contrast_unit": CONTRAST_UNIT,
        "replicate_pairing_claim": False,
        "delta_orientation": DELTA_ORIENTATION,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "registry_pair_contract": _expected_pair_contract(),
        "high_adapter_status_sha256": high_status["status_sha256"],
        "offline_pair_validation": dict(offline),
        "active_blocker_codes": list(PAIR_BLOCKER_CODES),
        "status_sha256": "",
    }
    status["status_sha256"] = status_sha256(status)
    validate_cbs_b6_pair_status(
        status,
        source_inputs=inputs if structural else None,
    )
    return status


def validate_cbs_b6_pair_status(
    status: Mapping[str, Any],
    *,
    source_inputs: CbsB6PairInputs | None = None,
) -> None:
    """Validate a pair status, requiring byte replay for structural claims."""

    status = _exact_mapping(status, PAIR_STATUS_FIELDS, "CBS B6 pair status")
    exact_values = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PAIR_STATUS_ARTIFACT_TYPE,
        "contract_id": PAIR_CONTRACT_ID,
        "claim_scope": PAIR_STATUS_CLAIM_SCOPE,
        "gene": GENE,
        "pair_registry_sha256": EXPECTED_PAIR_REGISTRY_SHA256,
        "contrast_unit": CONTRAST_UNIT,
        "replicate_pairing_claim": False,
        "delta_orientation": DELTA_ORIENTATION,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
    }
    for field, expected in exact_values.items():
        _require_exact_json(
            status[field],
            expected,
            f"CBS B6 pair status.{field}",
        )
    if (
        status["replicate_pairing_claim"] is not False
        or status["confirmatory_eligible"] is not False
        or status["automatic_promotion"] is not False
    ):
        raise CbsB6PairError("CBS B6 pair status boolean claim boundary differs")
    conditions = _exact_mapping(
        status["conditions"],
        PAIR_CONDITION_STATUS_FIELDS,
        "CBS B6 pair status.conditions",
    )
    _require_exact_json(
        conditions,
        {
            "low_urn": LOW_URN,
            "high_urn": HIGH_URN,
            "low_dose": LOW_DOSE_DEFINITION,
            "high_dose": HIGH_DOSE_DEFINITION,
        },
        "CBS B6 pair status conditions",
    )
    _require_exact_json(
        status["registry_pair_contract"],
        _expected_pair_contract(),
        "CBS B6 pair status registry contract",
    )
    if (
        _mapping(
            status["registry_pair_contract"],
            "CBS B6 pair status registry contract",
        ).get("replicate_pairing_claim")
        is not False
    ):
        raise CbsB6PairError("CBS B6 pair cannot claim paired replicates")
    _require_exact_json(
        status["active_blocker_codes"],
        list(PAIR_BLOCKER_CODES),
        "CBS B6 pair status blocker codes",
    )

    offline = _exact_mapping(
        status["offline_pair_validation"],
        PAIR_OFFLINE_FIELDS,
        "CBS B6 pair status.offline_pair_validation",
    )
    if (
        offline["pair_registry_status"] != "validated_exact_canonical_hash"
        or offline["native_join_key"] != list(NATIVE_JOIN_KEY)
        or offline["accession_join_used"] is not False
        or offline["missingness_policy"] != MISSINGNESS_POLICY
        or offline["imputation_performed"] is not False
        or offline["delta_values_materialized"] is not False
    ):
        raise CbsB6PairError("CBS B6 pair offline claim boundary differs")

    body_statuses = (
        offline["high_core_bodies_status"],
        offline["low_score_body_status"],
        offline["high_score_body_status"],
    )
    if body_statuses == ("not_supplied", "not_supplied", "not_supplied"):
        if source_inputs is not None and any(
            body is not None
            for body in (
                source_inputs.low_scores_body,
                source_inputs.high_metadata_body,
                source_inputs.high_scores_body,
                source_inputs.high_counts_body,
            )
        ):
            raise CbsB6PairError("unsupplied pair status conflicts with supplied bytes")
        _require_exact_json(
            offline,
            _empty_pair_offline(),
            "unsupplied pair status",
        )
        expected_high = build_cbs_high_b6_adapter_status(
            CbsHighB6AdapterInputs(
                pair_registry=_expected_registry(),
                high_metadata_body=None,
                high_scores_body=None,
                high_counts_body=None,
            )
        )
    elif body_statuses == (
        "validated_structural_only",
        "validated_structural_only",
        "validated_structural_only",
    ):
        if source_inputs is None or any(
            body is None
            for body in (
                source_inputs.low_scores_body,
                source_inputs.high_metadata_body,
                source_inputs.high_scores_body,
                source_inputs.high_counts_body,
            )
        ):
            raise CbsB6PairError(
                "validated pair status requires exact low scores and high "
                "metadata, scores, and counts bodies for replay"
            )
        _, replayed = validate_cbs_b6_pair_inputs(source_inputs)
        _require_exact_json(
            offline,
            replayed,
            "pair offline status replayed score bodies",
        )
        if (
            offline["low_score_body_sha256"] != LOW_SCORES_BODY_SHA256
            or offline["high_score_body_sha256"] != HIGH_SCORES_BODY_SHA256
            or offline["low_score_row_count"] != LOW_ROW_COUNT
            or offline["high_score_row_count"] != HIGH_ROW_COUNT
        ):
            raise CbsB6PairError("validated pair source identities differ")
        for field, expected in EXPECTED_STRUCTURAL_OVERLAP.items():
            _require_exact_json(
                offline[field],
                expected,
                f"validated pair overlap {field}",
            )
        complete_rows = _nonnegative_int(
            offline["complete_paired_native_row_count"],
            "complete_paired_native_row_count",
        )
        complete_proteins = _nonnegative_int(
            offline["complete_paired_unique_hgvs_pro_count"],
            "complete_paired_unique_hgvs_pro_count",
        )
        if (
            complete_rows > offline["native_join_row_count"]
            or complete_proteins > offline["native_join_unique_hgvs_pro_count"]
        ):
            raise CbsB6PairError("complete paired counts exceed structural overlap")
        if (
            offline["target_identity_status"]
            != "registry_audited_both_conditions_high_runtime_validated_low_runtime_not_replayed"
            or offline["native_join_status"]
            != "validated_accession_independent_primary_hgvs_nt_intersection_with_exact_annotations"
            or offline["codon_duplicate_status"]
            != "validated_exact_score_sd_se_missingness_and_values_within_each_condition_hgvs_pro"
            or offline["uncertainty_status"] != "not_derived_covariance_or_joint_bootstrap_required"
        ):
            raise CbsB6PairError("validated pair structural statuses differ")
        expected_high = build_cbs_high_b6_adapter_status(
            CbsHighB6AdapterInputs(
                pair_registry=source_inputs.pair_registry,
                high_metadata_body=source_inputs.high_metadata_body,
                high_scores_body=source_inputs.high_scores_body,
                high_counts_body=source_inputs.high_counts_body,
            )
        )
    else:
        raise CbsB6PairError("low/high score-body statuses must be identical and exact")

    if status["high_adapter_status_sha256"] != expected_high["status_sha256"]:
        raise CbsB6PairError("pair status high-adapter binding differs")
    _sha256(status["high_adapter_status_sha256"], "high_adapter_status_sha256")
    if _sha256(status["status_sha256"], "CBS B6 pair status_sha256") != status_sha256(status):
        raise CbsB6PairError("CBS B6 pair status self-hash differs")


def materialize_cbs_b6_pair_outcomes(
    inputs: CbsB6PairInputs,
    *,
    delta_orientation: str = DELTA_ORIENTATION,
    impute_missing: bool = False,
    covariance_or_joint_bootstrap_artifact: Mapping[str, Any] | None = None,
) -> NoReturn:
    """Reject all outcome materialization at the current candidate stage."""

    validate_cbs_b6_pair_inputs(inputs)
    if delta_orientation != DELTA_ORIENTATION:
        raise CbsB6PairError("CBS B6 delta orientation is frozen as high-B6 score minus low-B6 score")
    if type(impute_missing) is not bool:
        raise TypeError("impute_missing must be boolean")
    if impute_missing:
        raise CbsB6PairError("CBS B6 pair missing values cannot be imputed")
    if covariance_or_joint_bootstrap_artifact is None:
        raise CbsB6PairError("CBS B6 pair outcomes require an authenticated covariance or joint-bootstrap artifact")
    _mapping(
        covariance_or_joint_bootstrap_artifact,
        "CBS B6 covariance or joint-bootstrap artifact",
    )
    raise CbsB6PairError(
        "this candidate-only contract cannot authenticate covariance artifacts or materialize outcomes"
    )


def build_cbs_b6_pair_outcomes(
    inputs: CbsB6PairInputs,
    **kwargs: Any,
) -> NoReturn:
    """Backward-friendly explicit alias for the always-rejecting materializer."""

    return materialize_cbs_b6_pair_outcomes(inputs, **kwargs)


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    replace: bool,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise CbsB6PairError("CBS B6 status artifacts must use .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary_name, destination)
        else:
            try:
                os.link(temporary_name, destination)
            except FileExistsError as exc:
                raise CbsB6PairError(f"CBS B6 status already exists: {destination}") from exc
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_cbs_high_b6_adapter_status(
    path: str | Path,
    status: Mapping[str, Any],
    *,
    replace: bool = False,
    source_inputs: CbsHighB6AdapterInputs | None = None,
) -> Path:
    """Atomically write a validated high-B6 adapter status."""

    validate_cbs_high_b6_adapter_status(status, source_inputs=source_inputs)
    return _atomic_write_json(path, status, replace=replace)


def load_cbs_high_b6_adapter_status(
    path: str | Path,
    *,
    source_inputs: CbsHighB6AdapterInputs | None = None,
) -> Mapping[str, Any]:
    """Load and validate a high-B6 adapter status."""

    status = _load_json_file(path, "CBS high-B6 adapter status")
    validate_cbs_high_b6_adapter_status(status, source_inputs=source_inputs)
    return status


def write_cbs_b6_pair_status(
    path: str | Path,
    status: Mapping[str, Any],
    *,
    replace: bool = False,
    source_inputs: CbsB6PairInputs | None = None,
) -> Path:
    """Atomically write a validated pair status."""

    validate_cbs_b6_pair_status(status, source_inputs=source_inputs)
    return _atomic_write_json(path, status, replace=replace)


def load_cbs_b6_pair_status(
    path: str | Path,
    *,
    source_inputs: CbsB6PairInputs | None = None,
) -> Mapping[str, Any]:
    """Load and validate a pair status."""

    status = _load_json_file(path, "CBS B6 pair status")
    validate_cbs_b6_pair_status(status, source_inputs=source_inputs)
    return status


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the exact CBS B6 pair registry and write deterministic "
            "candidate-only high-B6 and pair status artifacts"
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--high-status-out", type=Path, required=True)
    parser.add_argument("--pair-status-out", type=Path, required=True)
    parser.add_argument("--low-scores-body", type=Path)
    parser.add_argument("--high-metadata-body", type=Path)
    parser.add_argument("--high-scores-body", type=Path)
    parser.add_argument("--high-counts-body", type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace existing status files",
    )
    arguments = parser.parse_args()
    optional_paths = (
        arguments.low_scores_body,
        arguments.high_metadata_body,
        arguments.high_scores_body,
        arguments.high_counts_body,
    )
    if any(path is not None for path in optional_paths) and not all(path is not None for path in optional_paths):
        parser.error(
            "--low-scores-body, --high-metadata-body, --high-scores-body, "
            "and --high-counts-body must be supplied together"
        )

    registry = load_cbs_b6_pair_registry(arguments.registry)
    low_body = arguments.low_scores_body.read_bytes() if arguments.low_scores_body is not None else None
    high_metadata_body = arguments.high_metadata_body.read_bytes() if arguments.high_metadata_body is not None else None
    high_body = arguments.high_scores_body.read_bytes() if arguments.high_scores_body is not None else None
    high_counts_body = arguments.high_counts_body.read_bytes() if arguments.high_counts_body is not None else None
    high_inputs = CbsHighB6AdapterInputs(
        pair_registry=registry,
        high_metadata_body=high_metadata_body,
        high_scores_body=high_body,
        high_counts_body=high_counts_body,
    )
    pair_inputs = CbsB6PairInputs(
        pair_registry=registry,
        low_scores_body=low_body,
        high_metadata_body=high_metadata_body,
        high_scores_body=high_body,
        high_counts_body=high_counts_body,
    )
    high_status = build_cbs_high_b6_adapter_status(high_inputs)
    pair_status = build_cbs_b6_pair_status(pair_inputs)
    write_cbs_high_b6_adapter_status(
        arguments.high_status_out,
        high_status,
        replace=arguments.replace,
        source_inputs=high_inputs if high_body is not None else None,
    )
    write_cbs_b6_pair_status(
        arguments.pair_status_out,
        pair_status,
        replace=arguments.replace,
        source_inputs=pair_inputs if high_body is not None else None,
    )
    print(
        json.dumps(
            {
                "high_status_path": str(arguments.high_status_out),
                "high_status_sha256": high_status["status_sha256"],
                "pair_status_path": str(arguments.pair_status_out),
                "pair_status_sha256": pair_status["status_sha256"],
                "outcome_status": pair_status["outcome_status"],
                "confirmatory_eligible": pair_status["confirmatory_eligible"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()

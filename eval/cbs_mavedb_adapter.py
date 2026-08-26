"""Candidate-only offline validation for the CBS low-vitamin-B6 MaveDB assay.

This module deliberately stops before schema-2 ingestion.  The deposited CBS
TileSeq ``counts`` table contains selected/nonselected variant-library relative
read frequencies per one million total reads and corresponding non-mutagenized
amplicon error-control channels.  Those normalized channels are neither raw
read counts nor scalar mutant-versus-functional-WT observations, and the
deposited aggregate score is not a raw replicate.  Until a native count replay
and its assay semantics are externally authenticated, this adapter can validate
source lineage only and must remain ``COUNT_LINEAGE_PARTIAL``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn

from eval import mavedb_source_lock as msl

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "groundbench.dms_cbs_low_b6_adapter_status"
ADAPTER_ID = "mavedb-cbs-low-b6-count-lineage-v1"
CLAIM_SCOPE = "candidate_count_lineage_validation_only_no_ingestion_or_outcome_claim"

CBS_URN = "urn:mavedb:00000005-a-5"
CBS_GENE = "CBS"
CBS_TITLE = "CBS low-B6"
CBS_READINESS = "COUNT_LINEAGE_PARTIAL"
CBS_SCORE_ROW_COUNT = 11_478
CBS_UNIQUE_MISSENSE_COUNT = 7_217
CBS_TARGET_SEQUENCE_TYPE = "dna"
CBS_TARGET_SEQUENCE_LENGTH = 1_656
EXPECTED_CANDIDATE_REGISTRY_SHA256 = "be3b3aef32ae2c5c62db58ba2a64015c71848f2702257abc3092db1b4a1b63cd"
EXPECTED_CBS_RECORD_SHA256 = "accbdcd60f89ba27a130e620e15ea6781cbd038dc00c5ae4533e9ebe44b4c329"
EXPECTED_SCORE_COLUMNS = ("score", "sd", "se")
EXPECTED_METADATA_BODY_BYTES = 30_355
EXPECTED_METADATA_BODY_SHA256 = "87e4f67a88175e6b6d3f3657d5737875d0bbaff599566a5fb301b70b2963ec03"
EXPECTED_SCORES_BODY_BYTES = 1_336_001
EXPECTED_SCORES_BODY_SHA256 = "47d2da4bd1c1368b5664c5f854444ca579cadea8a716eb549cfb59c12853fde5"
EXPECTED_COUNTS_BODY_BYTES = 4_616_367
EXPECTED_COUNTS_BODY_SHA256 = "b1bc332e11a2584cd199cbfcf2070b7cc7628d0d4d95a2e39938cd06851026c9"
COUNT_MEASUREMENT_SCALE = "relative_read_frequency_per_1M_total_reads"
COUNT_VALUE_CONTRACT = "finite_nonnegative_decimal_or_explicit_missing"
CONTROL_CHANNEL_ROLE = "nonmutagenized_wild_type_amplicon_sequencing_error_control_not_functional_wt_baseline"

EXPECTED_COUNT_COLUMNS = (
    "nonselect1",
    "nonselect2",
    "nonselect3",
    "nonselect4",
    "nonselect5",
    "nonselect6",
    "nonselect7",
    "nonselect8",
    "select1",
    "select2",
    "select3",
    "select4",
    "select5",
    "select6",
    "select7",
    "select8",
    "controlNS1",
    "controlNS2",
    "controlNS3",
    "controlNS4",
    "controlNS5",
    "controlNS6",
    "controlNS7",
    "controlNS8",
    "controlS1",
    "controlS2",
    "controlS3",
    "controlS4",
    "controlS5",
    "controlS6",
    "controlS7",
    "controlS8",
)

EXPECTED_REGISTRY_BLOCKERS = (
    "The yeast low-vitamin-B6 complementation context and codon-duplicate policy must be frozen before admission.",
    "The deposited count table has no explicit WT observation or frozen admissible control-baseline definition.",
    "The TileSeq control-count subtraction, selection/non-selection aggregation, normalization, and QC replay are not frozen, so the deposited counts support only COUNT_LINEAGE_PARTIAL.",
)

BASE_BLOCKER_CODES = (
    "CBS_CODON_TO_PROTEIN_COLLAPSE_MAP_MISSING",
    "CBS_EXTERNAL_ADAPTER_REGISTRATION_MISSING",
    "CBS_EXPLICIT_FUNCTIONAL_WT_BASELINE_ABSENT",
    "CBS_FUNCTIONAL_ANCHOR_MANIFEST_MISSING",
    "CBS_MAPPED_VARIANTS_BODY_UNAUTHENTICATED",
    "CBS_NATIVE_REPLAY_ARTIFACT_UNAUTHENTICATED",
    "CBS_OPENAPI_BODY_UNAUTHENTICATED",
    "CBS_QC_SCALING_AND_CI_RULE_MISSING",
    "CBS_RAW_READ_COUNT_WORKBOOK_ARTIFACT_UNAUTHENTICATED",
    "CBS_SAMPLE_ROLE_AND_SEQUENCING_DEPTH_MAP_MISSING",
    "CBS_SCHEMA2_MULTICHANNEL_REPLAY_UNSUPPORTED",
    "CBS_TILESEQ_PARAMETER_SHEET_MISSING",
    "CBS_TILESEQ_SOFTWARE_REVISION_MISSING",
)
SOURCE_BUNDLE_BLOCKER = "CBS_MATERIALIZED_SOURCE_BUNDLE_NOT_VALIDATED"

REQUIRED_UPSTREAM_ARTIFACTS = (
    "codon_to_protein_collapse_map",
    "control_subtraction_qc_scaling_and_ci_rule",
    "externally_authenticated_adapter_registration",
    "final_paper_additional_file_3_raw_read_counts_lowB6_sheet",
    "sample_role_and_sequencing_depth_map",
    "schema2_native_count_replay_extension",
    "synonymous_and_nonsense_functional_anchor_manifest",
    "tileseq_parameter_sheet",
    "tileseq_software_revision_and_source_archive",
)

SCHEMA_2_INCOMPATIBILITY_CODES = (
    "CI_AWARE_LABEL_RULE_UNREPRESENTED",
    "CODON_TO_PROTEIN_COLLAPSE_UNREPRESENTED",
    "MULTICHANNEL_SELECTED_NONSELECTED_ERROR_CONTROL_COUNTS",
    "NO_EXPLICIT_FUNCTIONAL_WT_OBSERVATION",
    "NORMALIZED_RELATIVE_FREQUENCY_NOT_RAW_READ_COUNTS",
    "REGULARIZED_UNCERTAINTY_UNREPRESENTED",
    "SYNONYMOUS_NONSENSE_MEDIAN_ANCHORS_UNREPRESENTED",
)

PROHIBITED_REINTERPRETATIONS = (
    "controlNS_or_controlS_as_functional_wild_type",
    "deposited_aggregate_score_as_raw_replicate",
    "normalized_relative_read_frequency_as_raw_read_count",
)

COUNT_MISSING_TOKENS = frozenset({"NA"})
DECIMAL_PATTERN = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REGISTRY_FIELDS = frozenset(
    {
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
)
CBS_RECORD_FIELDS = frozenset(
    {
        "admission_status",
        "blockers",
        "candidate_tier",
        "counts",
        "deposited_data_availability",
        "gene",
        "license",
        "metadata",
        "provisional_family",
        "score_orientation",
        "score_row_count",
        "scores",
        "source_readiness_ceiling",
        "target_evidence",
        "title",
        "unique_missense_hgvs_pro_count",
        "urn",
    }
)
BODY_LOCK_FIELDS = frozenset({"body_bytes", "body_sha256", "body_verified", "url"})
COUNT_BODY_LOCK_FIELDS = BODY_LOCK_FIELDS | {"metadata_count_columns"}

STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "adapter_id",
        "claim_scope",
        "urn",
        "gene",
        "title",
        "candidate_registry_sha256",
        "candidate_registry_record_sha256",
        "source_readiness",
        "admission_status",
        "ingestion_status",
        "outcome_status",
        "confirmatory_eligible",
        "automatic_promotion",
        "registry_expectations",
        "offline_validation",
        "count_measurement_semantics",
        "external_scientific_status_evidence",
        "schema_2_compatibility",
        "active_blocker_codes",
        "required_upstream_artifacts",
        "prohibited_reinterpretations",
        "status_sha256",
    }
)
REGISTRY_EXPECTATION_FIELDS = frozenset(
    {
        "candidate_tier",
        "metadata",
        "scores",
        "counts",
        "score_row_count",
        "unique_missense_hgvs_pro_count",
        "score_columns",
        "score_columns_status",
        "count_columns",
        "target_sequence_type",
        "target_sequence_length",
    }
)
OFFLINE_VALIDATION_FIELDS = frozenset(
    {
        "candidate_registry_status",
        "materialized_source_lock_status",
        "decoded_source_bodies_status",
        "source_lock_sha256",
        "source_bundle_sha256",
        "metadata_sha256",
        "scores_sha256",
        "counts_sha256",
        "mapped_variants_sha256",
        "mapped_variants_authentication_status",
        "openapi_authentication_status",
        "score_columns",
        "count_columns",
        "score_row_count",
        "count_row_count",
        "score_count_accession_join_status",
        "mapped_current_accession_join_status",
        "nonmissing_count_cell_count",
        "missing_count_cell_count",
    }
)
SCHEMA_2_COMPATIBILITY_FIELDS = frozenset({"compatible", "status", "reason_codes"})
COUNT_MEASUREMENT_SEMANTICS_FIELDS = frozenset(
    {
        "measurement_scale",
        "value_contract",
        "missing_tokens",
        "raw_read_count_claim",
        "controlNS_controlS_role",
        "functional_wt_baseline_claim",
    }
)
EXTERNAL_SCIENTIFIC_STATUS_FIELDS = frozenset(
    {
        "evidence_scope",
        "source",
        "raw_read_counts_lowB6_workbook_row_count",
        "experimental_lowB6_workbook_row_count",
        "raw_read_count_artifact_authentication_status",
        "low_b6_fitness_condition_rule",
        "published_functional_classification_rule",
        "published_b6_remediability_rule",
        "published_well_measured_rule",
        "score_uncertainty_channel_evidence",
        "obsolete_parameters_not_used",
    }
)
PUBLISHED_FUNCTIONAL_CLASSIFICATION_RULE_FIELDS = frozenset(
    {
        "classification",
        "upper_95_percent_ci_threshold",
        "fdr",
        "complement_classification",
    }
)
PUBLISHED_B6_REMEDIABILITY_RULE_FIELDS = frozenset(
    {
        "eligibility",
        "classification",
        "lower_95_percent_ci_threshold",
        "fdr",
    }
)
PUBLISHED_WELL_MEASURED_RULE_FIELDS = frozenset(
    {
        "preselection_allele_frequency_operator",
        "preselection_allele_frequency_percent_threshold",
        "standard_error_operator",
        "standard_error_threshold",
    }
)
SCORE_UNCERTAINTY_CHANNEL_EVIDENCE_FIELDS = frozenset(
    {
        "low_b6_sd_se_squared_ratio",
        "high_b6_sd_se_squared_ratio",
        "interpretation",
        "replicate_count_claim",
        "evidence_scope",
    }
)


class CbsAdapterError(ValueError):
    """Raised when CBS source lineage or candidate-only boundaries fail."""


@dataclass(frozen=True)
class CbsLowB6AdapterInputs:
    """Exact offline inputs for the candidate-only CBS validation stage.

    The materialized source lock and four decoded response bodies are an
    all-or-none group.  No network access is performed by this module.
    """

    candidate_registry: Mapping[str, Any]
    materialized_source_lock: Mapping[str, Any] | None = None
    metadata_body: bytes | None = None
    scores_body: bytes | None = None
    counts_body: bytes | None = None
    mapped_variants_body: bytes | None = None


@dataclass(frozen=True)
class _CsvTable:
    header: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]
    accessions: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    return msl.canonical_json_bytes(value)


def canonical_sha256(value: Any) -> str:
    return msl.canonical_sha256(value)


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CbsAdapterError(f"{context} must be a JSON object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CbsAdapterError(f"{context} must be a JSON array")
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
        raise CbsAdapterError(
            f"{context} must use the exact schema; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return result


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CbsAdapterError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CbsAdapterError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 1:
        raise CbsAdapterError(f"{context} must be a positive integer")
    return value


def _string_list(value: Any, context: str) -> list[str]:
    result = _list(value, context)
    for index, item in enumerate(result):
        _nonempty_string(item, f"{context}[{index}]")
    if len(result) != len(set(result)):
        raise CbsAdapterError(f"{context} cannot contain duplicates")
    return result


def _json_object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CbsAdapterError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_json_bytes(value: bytes, context: str) -> Any:
    if type(value) is not bytes:
        raise CbsAdapterError(f"{context} must be exact decoded bytes")
    try:
        text = value.decode("utf-8")
        return json.loads(text, object_pairs_hook=_json_object_hook)
    except CbsAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CbsAdapterError(f"{context} is not valid duplicate-free UTF-8 JSON") from exc


def _load_json_file(path: str | Path, context: str) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = _parse_json_bytes(source.read_bytes(), context)
    except OSError as exc:
        raise CbsAdapterError(f"cannot read {context}") from exc
    return _mapping(value, context)


def _validate_body_lock(
    value: Any,
    *,
    context: str,
    expected_fields: frozenset[str],
) -> Mapping[str, Any]:
    body = _exact_mapping(value, expected_fields, context)
    if body["body_verified"] is not True:
        raise CbsAdapterError(f"{context}.body_verified must be true")
    _positive_int(body["body_bytes"], f"{context}.body_bytes")
    _sha256(body["body_sha256"], f"{context}.body_sha256")
    url = _nonempty_string(body["url"], f"{context}.url")
    if not url.startswith("https://api.mavedb.org/api/v1/"):
        raise CbsAdapterError(f"{context}.url must use the official MaveDB API")
    return body


def validate_cbs_candidate_registry(
    registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the exact frozen candidate registry and return the CBS record."""

    registry = _exact_mapping(registry, REGISTRY_FIELDS, "MaveDB candidate registry")
    observed_registry_sha256 = canonical_sha256(registry)
    if observed_registry_sha256 != EXPECTED_CANDIDATE_REGISTRY_SHA256:
        raise CbsAdapterError("candidate registry SHA-256 differs from the frozen production registry")
    if registry["schema_version"] != 2:
        raise CbsAdapterError("candidate registry schema_version must be 2")
    if registry["artifact_type"] != "groundbench.dms_mavedb_candidate_registry":
        raise CbsAdapterError("candidate registry artifact_type is invalid")
    if registry["confirmatory_eligible"] is not False:
        raise CbsAdapterError("candidate registry cannot claim confirmatory eligibility")
    if registry["record_order"] != "score_set_urn_lexicographic":
        raise CbsAdapterError("candidate registry record order is invalid")
    if registry["api"] != {
        "base_url": msl.DEFAULT_API_BASE_URL,
        "version": "2026.2.7",
    }:
        raise CbsAdapterError("candidate registry API identity differs")
    if registry["candidate_counts"] != {
        "conditional": 6,
        "core": 14,
        "total": 20,
    }:
        raise CbsAdapterError("candidate registry candidate counts differ")

    records = _list(registry["records"], "candidate registry.records")
    urns = [
        _nonempty_string(
            _mapping(record, f"candidate registry.records[{index}]").get("urn"),
            f"candidate registry.records[{index}].urn",
        )
        for index, record in enumerate(records)
    ]
    if urns != sorted(urns) or len(urns) != len(set(urns)):
        raise CbsAdapterError("candidate registry records must be sorted by unique URN")
    matches = [record for record in records if record["urn"] == CBS_URN]
    if len(matches) != 1:
        raise CbsAdapterError("candidate registry must contain exactly one CBS low-B6 record")
    record = _exact_mapping(matches[0], CBS_RECORD_FIELDS, "CBS candidate record")
    if canonical_sha256(record) != EXPECTED_CBS_RECORD_SHA256:
        raise CbsAdapterError("CBS candidate record SHA-256 differs")

    exact_facts = {
        "urn": CBS_URN,
        "gene": CBS_GENE,
        "title": CBS_TITLE,
        "candidate_tier": "core",
        "admission_status": "candidate_not_ingested",
        "deposited_data_availability": ("substantive_counts_and_replicate_lineage_deposited_replay_unresolved"),
        "source_readiness_ceiling": CBS_READINESS,
        "score_row_count": CBS_SCORE_ROW_COUNT,
        "unique_missense_hgvs_pro_count": CBS_UNIQUE_MISSENSE_COUNT,
    }
    for field, expected in exact_facts.items():
        if record[field] != expected:
            raise CbsAdapterError(f"CBS candidate record.{field} differs")
    if tuple(record["blockers"]) != EXPECTED_REGISTRY_BLOCKERS:
        raise CbsAdapterError("CBS candidate registry blockers differ")
    if record["license"] != {
        "long_name": "CC0 (Public domain)",
        "short_name": "CC0",
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "version": "1.0",
    }:
        raise CbsAdapterError("CBS candidate license differs")
    if record["score_orientation"] != {
        "evidence": (
            "API method text states that the fitness values were normalized so "
            "median nonsense is 0 and median synonymous is 1."
        ),
        "evidence_status": "api_method_explicit",
        "retained_function_transform": "identity",
    }:
        raise CbsAdapterError("CBS candidate score orientation differs")
    target = _mapping(record["target_evidence"], "CBS candidate target_evidence")
    if (
        target.get("api_target_name") != CBS_GENE
        or target.get("mapped_hgnc_name") != CBS_GENE
        or target.get("mapped_uniprot_id") != "P35520"
        or target.get("sequence_present") is not True
        or target.get("sequence_type") != CBS_TARGET_SEQUENCE_TYPE
        or target.get("sequence_length") != CBS_TARGET_SEQUENCE_LENGTH
    ):
        raise CbsAdapterError("CBS candidate target evidence differs")

    metadata = _validate_body_lock(
        record["metadata"],
        context="CBS candidate metadata",
        expected_fields=BODY_LOCK_FIELDS,
    )
    scores = _validate_body_lock(
        record["scores"],
        context="CBS candidate scores",
        expected_fields=BODY_LOCK_FIELDS,
    )
    counts = _validate_body_lock(
        record["counts"],
        context="CBS candidate counts",
        expected_fields=COUNT_BODY_LOCK_FIELDS,
    )
    if metadata["url"] != f"{msl.DEFAULT_API_BASE_URL}/score-sets/{CBS_URN}":
        raise CbsAdapterError("CBS candidate metadata URL differs")
    if scores["url"] != f"{metadata['url']}/scores":
        raise CbsAdapterError("CBS candidate scores URL differs")
    if counts["url"] != f"{metadata['url']}/counts":
        raise CbsAdapterError("CBS candidate counts URL differs")
    if tuple(counts["metadata_count_columns"]) != EXPECTED_COUNT_COLUMNS:
        raise CbsAdapterError("CBS candidate count columns differ from the frozen 32-column schema")
    return record


def _parse_csv_table(
    body: bytes,
    *,
    context: str,
    expected_header: Sequence[str],
    expected_rows: int,
) -> _CsvTable:
    if type(body) is not bytes:
        raise CbsAdapterError(f"{context} must be exact decoded bytes")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CbsAdapterError(f"{context} must be UTF-8 CSV") from exc
    if "\x00" in text:
        raise CbsAdapterError(f"{context} cannot contain NUL bytes")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CbsAdapterError(f"{context} cannot be empty") from exc
    if header != list(expected_header):
        raise CbsAdapterError(
            f"{context} header differs from the exact schema; observed={header}, expected={list(expected_header)}"
        )
    if len(header) != len(set(header)):
        raise CbsAdapterError(f"{context} header cannot contain duplicates")

    records: list[Mapping[str, str]] = []
    accessions: list[str] = []
    for row_index, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise CbsAdapterError(
                f"{context} row {row_index} is truncated or overlong: observed={len(row)}, expected={len(header)}"
            )
        record = dict(zip(header, row, strict=True))
        accession = record["accession"]
        if not accession.startswith(f"{CBS_URN}#"):
            raise CbsAdapterError(f"{context} row {row_index} has an invalid accession")
        records.append(record)
        accessions.append(accession)
    if len(records) != expected_rows:
        raise CbsAdapterError(f"{context} row count differs: observed={len(records)}, expected={expected_rows}")
    if len(accessions) != len(set(accessions)):
        raise CbsAdapterError(f"{context} accessions must be unique")
    return _CsvTable(
        header=tuple(header),
        rows=tuple(records),
        accessions=tuple(accessions),
    )


def _is_source_missing(value: str) -> bool:
    return value.strip().lower() in msl.MISSING_TABLE_VALUES


def _score_numeric_summaries(
    table: _CsvTable,
    columns: Sequence[str],
) -> tuple[dict[str, int], dict[str, int]]:
    finite: dict[str, int] = {}
    invalid: dict[str, int] = {}
    for column in columns:
        finite_count = 0
        invalid_count = 0
        for row in table.rows:
            raw = row[column]
            if _is_source_missing(raw):
                continue
            try:
                value = float(raw)
            except ValueError:
                invalid_count += 1
                continue
            if math.isfinite(value):
                finite_count += 1
            else:
                invalid_count += 1
        finite[column] = finite_count
        invalid[column] = invalid_count
    return finite, invalid


def _validate_count_cells(
    table: _CsvTable,
    count_columns: Sequence[str],
) -> tuple[dict[str, int], int, int]:
    finite_counts = {column: 0 for column in count_columns}
    missing_count = 0
    nonmissing_count = 0
    for row_index, row in enumerate(table.rows, start=2):
        for column in count_columns:
            raw = row[column]
            if raw in COUNT_MISSING_TOKENS:
                missing_count += 1
                continue
            try:
                value = Decimal(raw)
            except InvalidOperation as exc:
                raise CbsAdapterError(
                    f"CBS counts row {row_index} column {column!r} must be a finite "
                    "nonnegative decimal or an exact missing token"
                ) from exc
            if DECIMAL_PATTERN.fullmatch(raw) is None or not value.is_finite() or value < 0:
                raise CbsAdapterError(
                    f"CBS counts row {row_index} column {column!r} must be a finite "
                    "nonnegative decimal or an exact missing token"
                )
            finite_counts[column] += 1
            nonmissing_count += 1
    if any(count == 0 for count in finite_counts.values()):
        empty = sorted(column for column, count in finite_counts.items() if count == 0)
        raise CbsAdapterError(f"CBS count columns have no observed decimal values: {empty}")
    return finite_counts, nonmissing_count, missing_count


def _duplicate_nonmissing_group_count(table: _CsvTable, column: str) -> int:
    counts = Counter(row[column] for row in table.rows if not _is_source_missing(row[column]))
    return sum(count > 1 for count in counts.values())


def _validate_metadata_body(
    metadata_body: bytes,
    *,
    source_lock: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[str]]:
    metadata = _mapping(
        _parse_json_bytes(metadata_body, "CBS MaveDB metadata body"),
        "CBS MaveDB metadata body",
    )
    if metadata.get("urn") != CBS_URN:
        raise CbsAdapterError("CBS metadata body URN differs")
    if metadata.get("title") != CBS_TITLE:
        raise CbsAdapterError("CBS metadata body title differs")
    if metadata.get("numVariants") != CBS_SCORE_ROW_COUNT:
        raise CbsAdapterError("CBS metadata body numVariants differs")
    if (
        metadata.get("private") is not False
        or metadata.get("processingState") != "success"
        or metadata.get("supersededScoreSet") is not None
        or metadata.get("supersedingScoreSet") is not None
    ):
        raise CbsAdapterError("CBS metadata body is not the active public successful score set")

    dataset_columns = _exact_mapping(
        metadata.get("datasetColumns"),
        msl.DATASET_COLUMNS_FIELDS,
        "CBS metadata datasetColumns",
    )
    if dataset_columns["recordType"] != "DatasetColumns":
        raise CbsAdapterError("CBS metadata datasetColumns.recordType differs")
    score_columns = _string_list(
        dataset_columns["scoreColumns"],
        "CBS metadata scoreColumns",
    )
    count_columns = _string_list(
        dataset_columns["countColumns"],
        "CBS metadata countColumns",
    )
    if tuple(score_columns) != EXPECTED_SCORE_COLUMNS:
        raise CbsAdapterError("CBS metadata scoreColumns differ from the assay-specific schema")
    if tuple(count_columns) != EXPECTED_COUNT_COLUMNS:
        raise CbsAdapterError("CBS metadata countColumns differ from the frozen 32-column schema")
    if set(msl.FIXED_VARIANT_COLUMNS) & (set(score_columns) | set(count_columns)):
        raise CbsAdapterError("CBS custom columns collide with fixed MaveDB columns")

    lock_metadata = source_lock["metadata_contract"]
    if lock_metadata["title"] != metadata["title"]:
        raise CbsAdapterError("CBS metadata title differs from source lock")
    if lock_metadata["num_variants"] != metadata["numVariants"]:
        raise CbsAdapterError("CBS metadata numVariants differs from source lock")
    if lock_metadata["dataset_columns"] != dataset_columns:
        raise CbsAdapterError("CBS metadata datasetColumns differ from source lock")
    if lock_metadata["license"] != metadata.get("license"):
        raise CbsAdapterError("CBS metadata license differs from source lock")
    if lock_metadata["data_usage_policy"] != metadata.get("dataUsagePolicy"):
        raise CbsAdapterError("CBS metadata dataUsagePolicy differs from source lock")
    calibrations = metadata.get("scoreCalibrations")
    if calibrations is None:
        calibrations = []
    if not isinstance(calibrations, list):
        raise CbsAdapterError("CBS metadata scoreCalibrations must be a list or null")
    if lock_metadata["calibration_count"] != len(calibrations):
        raise CbsAdapterError("CBS calibration count differs from source lock")
    if lock_metadata["calibrations_sha256"] != canonical_sha256(calibrations):
        raise CbsAdapterError("CBS calibrations differ from source lock")

    targets = _list(metadata.get("targetGenes"), "CBS metadata targetGenes")
    if len(targets) != 1:
        raise CbsAdapterError("CBS metadata must contain exactly one target")
    target = _mapping(targets[0], "CBS metadata targetGenes[0]")
    target_sequence = _mapping(
        target.get("targetSequence"),
        "CBS metadata targetGenes[0].targetSequence",
    )
    sequence = target_sequence.get("sequence")
    if (
        target.get("name") != CBS_GENE
        or target.get("mappedHgncName") != CBS_GENE
        or target.get("targetAccession") is not None
        or target_sequence.get("sequenceType") != CBS_TARGET_SEQUENCE_TYPE
        or not isinstance(sequence, str)
        or len(sequence) != CBS_TARGET_SEQUENCE_LENGTH
    ):
        raise CbsAdapterError("CBS metadata embedded DNA target differs")
    lock_targets = lock_metadata["targets"]
    if len(lock_targets) != 1:
        raise CbsAdapterError("CBS source lock must contain exactly one target")
    lock_target = lock_targets[0]
    try:
        sequence_sha256 = _bytes_sha256(sequence.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise CbsAdapterError("CBS metadata embedded DNA target must be ASCII") from exc
    if (
        lock_target["target_kind"] != "embedded_sequence"
        or lock_target["name"] != CBS_GENE
        or lock_target["mapped_hgnc_name"] != CBS_GENE
        or lock_target["sequence_type"] != CBS_TARGET_SEQUENCE_TYPE
        or lock_target["sequence_length"] != CBS_TARGET_SEQUENCE_LENGTH
        or lock_target["sequence_sha256"] != sequence_sha256
        or lock_target["external_identifiers_sha256"] != canonical_sha256(target.get("externalIdentifiers", []))
    ):
        raise CbsAdapterError("CBS metadata target differs from source lock")
    return metadata, score_columns


def _mapping_summary(
    mapped_body: bytes,
    *,
    expected_accessions: Sequence[str],
) -> Mapping[str, Any]:
    records = _list(
        _parse_json_bytes(mapped_body, "CBS MaveDB mapped-variants body"),
        "CBS MaveDB mapped-variants body",
    )
    current: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"CBS mapped-variants[{index}]")
        variant_urn = record.get("variantUrn")
        if not isinstance(variant_urn, str) or not variant_urn.startswith(f"{CBS_URN}#"):
            raise CbsAdapterError(f"CBS mapped-variants[{index}] has an invalid variantUrn")
        if type(record.get("current")) is not bool:
            raise CbsAdapterError(f"CBS mapped-variants[{index}].current must be boolean")
        if record["current"]:
            current.append(record)

    current_accessions = [str(record["variantUrn"]) for record in current]
    current_counts = Counter(current_accessions)
    duplicates = sorted(accession for accession, count in current_counts.items() if count != 1)
    expected_set = set(expected_accessions)
    current_set = set(current_accessions)
    missing = sorted(expected_set - current_set)
    extra = sorted(current_set - expected_set)
    if duplicates or missing or extra:
        raise CbsAdapterError(
            "CBS current mapped-variant accessions differ from score/count accessions; "
            f"duplicates={duplicates[:5]}, missing={missing[:5]}, extra={extra[:5]}"
        )

    def _nonnull_values(field: str) -> list[str]:
        return sorted({str(record[field]) for record in current if record.get(field) is not None})

    current_errors = [
        record
        for record in current
        if record.get("postMapped") is None
        or (record.get("errorMessage") is not None and str(record.get("errorMessage")).strip())
    ]
    canonical_current = sorted(
        (dict(record) for record in current),
        key=lambda record: record["variantUrn"],
    )
    return {
        "history_record_count": len(records),
        "current_record_count": len(current),
        "current_unique_variant_count": len(current_set),
        "current_duplicate_variant_count": 0,
        "current_missing_variant_count": 0,
        "current_extra_variant_count": 0,
        "current_error_count": len(current_errors),
        "current_post_mapped_null_count": sum(record.get("postMapped") is None for record in current),
        "current_pre_mapped_null_count": sum(record.get("preMapped") is None for record in current),
        "current_at_mismatched_locus_count": sum(record.get("atMismatchedLocus") is True for record in current),
        "current_near_gap_count": sum(record.get("nearGap") is True for record in current),
        "alignment_levels": _nonnull_values("alignmentLevel"),
        "mapping_api_versions": _nonnull_values("mappingApiVersion"),
        "vrs_versions": _nonnull_values("vrsVersion"),
        "current_accessions_sha256": canonical_sha256(sorted(current_set)),
        "current_mappings_sha256": canonical_sha256(canonical_current),
    }


def _validate_source_lock_identity(
    source_lock: Mapping[str, Any],
    registry_record: Mapping[str, Any],
) -> None:
    try:
        msl.validate_source_lock(source_lock)
    except (msl.SourceLockError, TypeError) as exc:
        raise CbsAdapterError("materialized CBS MaveDB source lock is invalid") from exc
    if source_lock["urn"] != CBS_URN:
        raise CbsAdapterError("materialized source lock is not the CBS low-B6 score set")
    if (
        source_lock["claim_scope"] != msl.CLAIM_SCOPE
        or source_lock["ingestion_status"] != "not_ingested"
        or source_lock["outcome_status"] != "not_derived"
    ):
        raise CbsAdapterError("materialized source lock exceeds candidate-only claims")
    if source_lock["api"]["base_url"] != msl.DEFAULT_API_BASE_URL:
        raise CbsAdapterError("materialized source lock uses a different MaveDB API base")
    if source_lock["api"]["openapi_version"] != "2026.2.7":
        raise CbsAdapterError("materialized source lock API version differs")
    readiness = source_lock["readiness"]
    if readiness["state"] != CBS_READINESS or readiness["automatic_promotion"] is not False:
        raise CbsAdapterError("materialized source lock must remain COUNT_LINEAGE_PARTIAL without promotion")
    evidence = readiness["evidence"]
    expected_readiness_evidence = {
        "aggregate_score_column": "score",
        "processed_replicate_columns": [],
        "count_lineage_columns": list(EXPECTED_COUNT_COLUMNS),
        "identity_resolution_spec_sha256": None,
        "transformation_spec_sha256": None,
        "wt_control_spec_sha256": None,
        "confirmatory_preregistration_sha256": None,
        "independent_replication_spec_sha256": None,
        "identity_block_reason": None,
        "rejection_reason": None,
    }
    if evidence != expected_readiness_evidence:
        raise CbsAdapterError("materialized source lock score/count readiness evidence differs")

    dataset_columns = source_lock["metadata_contract"]["dataset_columns"]
    if tuple(dataset_columns["countColumns"]) != EXPECTED_COUNT_COLUMNS:
        raise CbsAdapterError("materialized source lock metadata count columns differ")
    score_columns = dataset_columns["scoreColumns"]
    if tuple(score_columns) != EXPECTED_SCORE_COLUMNS:
        raise CbsAdapterError("materialized source lock score columns differ")
    tabular = source_lock["tabular_contract"]
    if (
        tabular["score_row_count"] != CBS_SCORE_ROW_COUNT
        or tabular["count_row_count"] != CBS_SCORE_ROW_COUNT
        or tabular["counts_status"] != "substantive"
        or tabular["substantive_count_column_count"] != len(EXPECTED_COUNT_COLUMNS)
    ):
        raise CbsAdapterError("materialized source lock tabular CBS facts differ")

    license_value = source_lock["metadata_contract"]["license"]
    registry_license = registry_record["license"]
    if (
        license_value["shortName"] != registry_license["short_name"]
        or license_value["longName"] != registry_license["long_name"]
        or license_value["link"] != registry_license["url"]
        or license_value["version"] != registry_license["version"]
    ):
        raise CbsAdapterError("materialized source lock license differs from registry")

    source_artifacts = source_lock["source_artifacts"]
    for name in ("metadata", "scores", "counts"):
        if source_artifacts[name]["url"] != registry_record[name]["url"]:
            raise CbsAdapterError(f"materialized source lock {name} URL differs from registry")


def _validate_exact_body_binding(
    *,
    name: str,
    body: bytes,
    source_lock: Mapping[str, Any],
    registry_body_lock: Mapping[str, Any] | None,
) -> str:
    observed_sha256 = _bytes_sha256(body)
    artifact = source_lock["source_artifacts"][name]
    if artifact["sha256"] != observed_sha256 or artifact["decoded_byte_count"] != len(body):
        raise CbsAdapterError(f"CBS {name} decoded body differs from the materialized source lock")
    if registry_body_lock is not None and (
        registry_body_lock["body_sha256"] != observed_sha256 or registry_body_lock["body_bytes"] != len(body)
    ):
        raise CbsAdapterError(f"CBS {name} decoded body differs from the frozen candidate registry")
    return observed_sha256


def _validate_tabular_contract_against_bodies(
    *,
    source_lock: Mapping[str, Any],
    scores: _CsvTable,
    counts: _CsvTable,
    score_columns: Sequence[str],
    count_finite_counts: Mapping[str, int],
    nonmissing_count_cells: int,
) -> None:
    tabular = source_lock["tabular_contract"]
    if tabular["fixed_variant_columns"] != list(msl.FIXED_VARIANT_COLUMNS):
        raise CbsAdapterError("source lock fixed variant columns differ")
    if tabular["score_header"] != list(scores.header):
        raise CbsAdapterError("score body header differs from source lock")
    if tabular["count_header"] != list(counts.header):
        raise CbsAdapterError("count body header differs from source lock")
    if tabular["score_row_count"] != len(scores.rows) or tabular["count_row_count"] != len(counts.rows):
        raise CbsAdapterError("source body row counts differ from source lock")
    if tabular["accessions_sha256"] != canonical_sha256(list(scores.accessions)):
        raise CbsAdapterError("score/count accession order differs from source lock")
    if tabular["sorted_accessions_sha256"] != canonical_sha256(sorted(scores.accessions)):
        raise CbsAdapterError("sorted score/count accessions differ from source lock")

    score_finite, score_invalid = _score_numeric_summaries(scores, score_columns)
    if tabular["score_finite_value_counts"] != score_finite:
        raise CbsAdapterError("score finite-value counts differ from source lock")
    if tabular["score_invalid_value_counts"] != score_invalid:
        raise CbsAdapterError("score invalid-value counts differ from source lock")
    if tabular["count_finite_value_counts"] != dict(count_finite_counts):
        raise CbsAdapterError("count finite-value counts differ from source lock")
    if tabular["count_invalid_value_counts"] != {column: 0 for column in EXPECTED_COUNT_COLUMNS}:
        raise CbsAdapterError("source lock claims invalid CBS count-cell summaries")
    if tabular["substantive_count_value_count"] != nonmissing_count_cells:
        raise CbsAdapterError("substantive count-cell total differs from source lock")
    if tabular["native_hgvs_nt_duplicate_group_count"] != (_duplicate_nonmissing_group_count(scores, "hgvs_nt")):
        raise CbsAdapterError("native hgvs_nt duplicate summary differs from source lock")
    if tabular["native_hgvs_pro_duplicate_group_count"] != (_duplicate_nonmissing_group_count(scores, "hgvs_pro")):
        raise CbsAdapterError("native hgvs_pro duplicate summary differs from source lock")


def _empty_offline_validation() -> dict[str, Any]:
    return {
        "candidate_registry_status": "validated",
        "materialized_source_lock_status": "not_supplied",
        "decoded_source_bodies_status": "not_supplied",
        "source_lock_sha256": None,
        "source_bundle_sha256": None,
        "metadata_sha256": None,
        "scores_sha256": None,
        "counts_sha256": None,
        "mapped_variants_sha256": None,
        "mapped_variants_authentication_status": ("not_supplied_and_no_registry_expected_digest"),
        "openapi_authentication_status": ("not_supplied_and_no_registry_expected_digest"),
        "score_columns": None,
        "count_columns": list(EXPECTED_COUNT_COLUMNS),
        "score_row_count": None,
        "count_row_count": None,
        "score_count_accession_join_status": "not_evaluated",
        "mapped_current_accession_join_status": "not_evaluated",
        "nonmissing_count_cell_count": None,
        "missing_count_cell_count": None,
    }


def validate_cbs_low_b6_adapter_inputs(
    inputs: CbsLowB6AdapterInputs,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate registry and, when supplied, the complete offline source bundle."""

    if not isinstance(inputs, CbsLowB6AdapterInputs):
        raise TypeError("inputs must be CbsLowB6AdapterInputs")
    registry_record = validate_cbs_candidate_registry(inputs.candidate_registry)
    optional_values = (
        inputs.materialized_source_lock,
        inputs.metadata_body,
        inputs.scores_body,
        inputs.counts_body,
        inputs.mapped_variants_body,
    )
    supplied = [value is not None for value in optional_values]
    if any(supplied) and not all(supplied):
        raise CbsAdapterError(
            "materialized source lock and metadata/scores/counts/mapped-variants "
            "decoded bodies must be supplied together"
        )
    if not any(supplied):
        return registry_record, _empty_offline_validation()

    source_lock = _mapping(
        inputs.materialized_source_lock,
        "materialized CBS MaveDB source lock",
    )
    bodies = {
        "metadata": inputs.metadata_body,
        "scores": inputs.scores_body,
        "counts": inputs.counts_body,
        "mapped_variants": inputs.mapped_variants_body,
    }
    for name, body in bodies.items():
        if type(body) is not bytes:
            raise CbsAdapterError(f"CBS {name} body must be exact decoded bytes")

    _validate_source_lock_identity(source_lock, registry_record)
    observed_hashes = {
        name: _validate_exact_body_binding(
            name=name,
            body=body,
            source_lock=source_lock,
            registry_body_lock=(registry_record[name] if name in {"metadata", "scores", "counts"} else None),
        )
        for name, body in bodies.items()
    }

    _, score_columns = _validate_metadata_body(
        bodies["metadata"],
        source_lock=source_lock,
    )
    scores = _parse_csv_table(
        bodies["scores"],
        context="CBS scores",
        expected_header=(*msl.FIXED_VARIANT_COLUMNS, *score_columns),
        expected_rows=CBS_SCORE_ROW_COUNT,
    )
    counts = _parse_csv_table(
        bodies["counts"],
        context="CBS counts",
        expected_header=(*msl.FIXED_VARIANT_COLUMNS, *EXPECTED_COUNT_COLUMNS),
        expected_rows=CBS_SCORE_ROW_COUNT,
    )
    if counts.accessions != scores.accessions:
        raise CbsAdapterError("CBS score and count accession order differs")
    count_finite_counts, nonmissing_count_cells, missing_count_cells = _validate_count_cells(
        counts, EXPECTED_COUNT_COLUMNS
    )
    _validate_tabular_contract_against_bodies(
        source_lock=source_lock,
        scores=scores,
        counts=counts,
        score_columns=score_columns,
        count_finite_counts=count_finite_counts,
        nonmissing_count_cells=nonmissing_count_cells,
    )
    observed_mapping_summary = _mapping_summary(
        bodies["mapped_variants"],
        expected_accessions=scores.accessions,
    )
    if observed_mapping_summary != source_lock["mapping_contract"]:
        raise CbsAdapterError("mapped-variants body summary differs from source lock")

    return registry_record, {
        "candidate_registry_status": "validated",
        "materialized_source_lock_status": "validated_structural_only",
        "decoded_source_bodies_status": "validated_structural_only",
        "source_lock_sha256": canonical_sha256(source_lock),
        "source_bundle_sha256": source_lock["source_bundle_sha256"],
        "metadata_sha256": observed_hashes["metadata"],
        "scores_sha256": observed_hashes["scores"],
        "counts_sha256": observed_hashes["counts"],
        "mapped_variants_sha256": observed_hashes["mapped_variants"],
        "mapped_variants_authentication_status": ("unanchored_structural_validation_only"),
        "openapi_authentication_status": ("unanchored_source_lock_summary_only"),
        "score_columns": list(score_columns),
        "count_columns": list(EXPECTED_COUNT_COLUMNS),
        "score_row_count": len(scores.rows),
        "count_row_count": len(counts.rows),
        "score_count_accession_join_status": "validated_exact_order",
        "mapped_current_accession_join_status": "validated_exact_set",
        "nonmissing_count_cell_count": nonmissing_count_cells,
        "missing_count_cell_count": missing_count_cells,
    }


def _status_payload(status: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(status)
    payload.pop("status_sha256", None)
    return payload


def adapter_status_sha256(status: Mapping[str, Any]) -> str:
    return canonical_sha256(_status_payload(status))


def build_cbs_low_b6_adapter_status(
    inputs: CbsLowB6AdapterInputs,
) -> dict[str, Any]:
    """Build the deterministic candidate-only CBS adapter status."""

    registry_record, offline_validation = validate_cbs_low_b6_adapter_inputs(inputs)
    source_replayed = offline_validation["decoded_source_bodies_status"] == "validated_structural_only"
    blockers = [*BASE_BLOCKER_CODES, SOURCE_BUNDLE_BLOCKER]
    blockers.sort()

    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "adapter_id": ADAPTER_ID,
        "claim_scope": CLAIM_SCOPE,
        "urn": CBS_URN,
        "gene": CBS_GENE,
        "title": CBS_TITLE,
        "candidate_registry_sha256": canonical_sha256(inputs.candidate_registry),
        "candidate_registry_record_sha256": canonical_sha256(registry_record),
        "source_readiness": CBS_READINESS,
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "registry_expectations": {
            "candidate_tier": "core",
            "metadata": {
                "body_bytes": registry_record["metadata"]["body_bytes"],
                "body_sha256": registry_record["metadata"]["body_sha256"],
                "url": registry_record["metadata"]["url"],
            },
            "scores": {
                "body_bytes": registry_record["scores"]["body_bytes"],
                "body_sha256": registry_record["scores"]["body_sha256"],
                "url": registry_record["scores"]["url"],
            },
            "counts": {
                "body_bytes": registry_record["counts"]["body_bytes"],
                "body_sha256": registry_record["counts"]["body_sha256"],
                "url": registry_record["counts"]["url"],
            },
            "score_row_count": CBS_SCORE_ROW_COUNT,
            "unique_missense_hgvs_pro_count": CBS_UNIQUE_MISSENSE_COUNT,
            "score_columns": None,
            "score_columns_status": ("not_frozen_in_candidate_registry_resolve_from_exact_metadata"),
            "count_columns": list(EXPECTED_COUNT_COLUMNS),
            "target_sequence_type": CBS_TARGET_SEQUENCE_TYPE,
            "target_sequence_length": CBS_TARGET_SEQUENCE_LENGTH,
        },
        "offline_validation": dict(offline_validation),
        "count_measurement_semantics": {
            "measurement_scale": COUNT_MEASUREMENT_SCALE,
            "value_contract": COUNT_VALUE_CONTRACT,
            "missing_tokens": sorted(COUNT_MISSING_TOKENS),
            "raw_read_count_claim": False,
            "controlNS_controlS_role": CONTROL_CHANNEL_ROLE,
            "functional_wt_baseline_claim": False,
        },
        "external_scientific_status_evidence": {
            "evidence_scope": ("external_scientific_status_only_not_source_lock_or_replay_authentication"),
            "source": "final_paper_additional_file_3",
            "raw_read_counts_lowB6_workbook_row_count": 22_536,
            "experimental_lowB6_workbook_row_count": 11_478,
            "raw_read_count_artifact_authentication_status": ("not_supplied_or_hash_authenticated_by_this_adapter"),
            "low_b6_fitness_condition_rule": ("average_0_and_1_ng_per_mL_conditions_due_to_high_agreement"),
            "published_functional_classification_rule": {
                "classification": ("deleterious_if_upper_95_percent_ci_below_threshold"),
                "upper_95_percent_ci_threshold": 0.60,
                "fdr": 0.05,
                "complement_classification": ("not_defined_as_neutral_or_retained_function"),
            },
            "published_b6_remediability_rule": {
                "eligibility": "classified_deleterious_under_low_b6_rule",
                "classification": ("remediable_if_lower_95_percent_ci_of_high_minus_low_above_threshold"),
                "lower_95_percent_ci_threshold": 0.22,
                "fdr": 0.05,
            },
            "published_well_measured_rule": {
                "preselection_allele_frequency_operator": ">",
                "preselection_allele_frequency_percent_threshold": 0.005,
                "standard_error_operator": "<",
                "standard_error_threshold": 0.2,
            },
            "score_uncertainty_channel_evidence": {
                "low_b6_sd_se_squared_ratio": 8,
                "high_b6_sd_se_squared_ratio": 4,
                "interpretation": ("consistent_with_eight_low_b6_and_four_high_b6_observation_channels"),
                "replicate_count_claim": False,
                "evidence_scope": ("external_scientific_status_only_sample_map_missing"),
            },
            "obsolete_parameters_not_used": [
                "fitness_threshold_0.45",
                "delta_0.29",
            ],
        },
        "schema_2_compatibility": {
            "compatible": False,
            "status": "INCOMPATIBLE_WITH_CURRENT_SCALAR_REPLAY",
            "reason_codes": list(SCHEMA_2_INCOMPATIBILITY_CODES),
        },
        "active_blocker_codes": blockers,
        "required_upstream_artifacts": list(REQUIRED_UPSTREAM_ARTIFACTS),
        "prohibited_reinterpretations": list(PROHIBITED_REINTERPRETATIONS),
        "status_sha256": "",
    }
    status["status_sha256"] = adapter_status_sha256(status)
    validate_cbs_adapter_status(
        status,
        source_inputs=inputs if source_replayed else None,
    )
    return status


def validate_cbs_adapter_status(
    status: Mapping[str, Any],
    *,
    source_inputs: CbsLowB6AdapterInputs | None = None,
) -> None:
    status = _exact_mapping(status, STATUS_FIELDS, "CBS adapter status")
    exact_values = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "adapter_id": ADAPTER_ID,
        "claim_scope": CLAIM_SCOPE,
        "urn": CBS_URN,
        "gene": CBS_GENE,
        "title": CBS_TITLE,
        "source_readiness": CBS_READINESS,
        "admission_status": "candidate_not_ingested",
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
    }
    for field, expected in exact_values.items():
        if status[field] != expected:
            raise CbsAdapterError(f"CBS adapter status.{field} differs")
    registry_sha256 = _sha256(
        status["candidate_registry_sha256"],
        "CBS adapter status.candidate_registry_sha256",
    )
    if registry_sha256 != EXPECTED_CANDIDATE_REGISTRY_SHA256:
        raise CbsAdapterError("CBS adapter status candidate registry hash differs")
    record_sha256 = _sha256(
        status["candidate_registry_record_sha256"],
        "CBS adapter status.candidate_registry_record_sha256",
    )
    if record_sha256 != EXPECTED_CBS_RECORD_SHA256:
        raise CbsAdapterError("CBS adapter status candidate record hash differs")

    expectations = _exact_mapping(
        status["registry_expectations"],
        REGISTRY_EXPECTATION_FIELDS,
        "CBS adapter status.registry_expectations",
    )
    if (
        expectations["candidate_tier"] != "core"
        or expectations["score_row_count"] != CBS_SCORE_ROW_COUNT
        or expectations["unique_missense_hgvs_pro_count"] != CBS_UNIQUE_MISSENSE_COUNT
        or expectations["score_columns"] is not None
        or expectations["score_columns_status"] != "not_frozen_in_candidate_registry_resolve_from_exact_metadata"
        or tuple(expectations["count_columns"]) != EXPECTED_COUNT_COLUMNS
        or expectations["target_sequence_type"] != CBS_TARGET_SEQUENCE_TYPE
        or expectations["target_sequence_length"] != CBS_TARGET_SEQUENCE_LENGTH
    ):
        raise CbsAdapterError("CBS adapter registry expectations differ")
    expected_body_contracts = {
        "metadata": {
            "body_bytes": EXPECTED_METADATA_BODY_BYTES,
            "body_sha256": EXPECTED_METADATA_BODY_SHA256,
            "url": f"{msl.DEFAULT_API_BASE_URL}/score-sets/{CBS_URN}",
        },
        "scores": {
            "body_bytes": EXPECTED_SCORES_BODY_BYTES,
            "body_sha256": EXPECTED_SCORES_BODY_SHA256,
            "url": f"{msl.DEFAULT_API_BASE_URL}/score-sets/{CBS_URN}/scores",
        },
        "counts": {
            "body_bytes": EXPECTED_COUNTS_BODY_BYTES,
            "body_sha256": EXPECTED_COUNTS_BODY_SHA256,
            "url": f"{msl.DEFAULT_API_BASE_URL}/score-sets/{CBS_URN}/counts",
        },
    }
    for name in ("metadata", "scores", "counts"):
        body = _exact_mapping(
            expectations[name],
            {"body_bytes", "body_sha256", "url"},
            f"CBS adapter status.registry_expectations.{name}",
        )
        _positive_int(body["body_bytes"], f"registry_expectations.{name}.body_bytes")
        _sha256(body["body_sha256"], f"registry_expectations.{name}.body_sha256")
        _nonempty_string(body["url"], f"registry_expectations.{name}.url")
        if body != expected_body_contracts[name]:
            raise CbsAdapterError(f"CBS adapter registry expectation for {name} differs")

    offline = _exact_mapping(
        status["offline_validation"],
        OFFLINE_VALIDATION_FIELDS,
        "CBS adapter status.offline_validation",
    )
    if offline["candidate_registry_status"] != "validated":
        raise CbsAdapterError("CBS candidate registry must be validated")
    body_status = offline["decoded_source_bodies_status"]
    source_status = offline["materialized_source_lock_status"]
    if body_status == "not_supplied":
        if source_status != "not_supplied":
            raise CbsAdapterError("source-lock/body validation statuses are inconsistent")
        if (
            offline["mapped_variants_authentication_status"] != "not_supplied_and_no_registry_expected_digest"
            or offline["openapi_authentication_status"] != "not_supplied_and_no_registry_expected_digest"
        ):
            raise CbsAdapterError("unsupplied source authentication statuses differ")
        nullable_fields = (
            "source_lock_sha256",
            "source_bundle_sha256",
            "metadata_sha256",
            "scores_sha256",
            "counts_sha256",
            "mapped_variants_sha256",
            "score_columns",
            "score_row_count",
            "count_row_count",
            "nonmissing_count_cell_count",
            "missing_count_cell_count",
        )
        if any(offline[field] is not None for field in nullable_fields):
            raise CbsAdapterError("unsupplied source validation cannot claim observed values")
        if (
            offline["score_count_accession_join_status"] != "not_evaluated"
            or offline["mapped_current_accession_join_status"] != "not_evaluated"
        ):
            raise CbsAdapterError("unsupplied source validation cannot claim joins")
    elif body_status == "validated_structural_only":
        if source_inputs is None:
            raise CbsAdapterError(
                "validated source-body status requires the exact source inputs for independent replay"
            )
        if source_status != "validated_structural_only":
            raise CbsAdapterError("validated source bodies require a validated source lock")
        if (
            offline["mapped_variants_authentication_status"] != "unanchored_structural_validation_only"
            or offline["openapi_authentication_status"] != "unanchored_source_lock_summary_only"
        ):
            raise CbsAdapterError("structurally validated source authentication statuses differ")
        for field in (
            "source_lock_sha256",
            "source_bundle_sha256",
            "metadata_sha256",
            "scores_sha256",
            "counts_sha256",
            "mapped_variants_sha256",
        ):
            _sha256(offline[field], f"offline_validation.{field}")
        score_columns = _string_list(
            offline["score_columns"],
            "offline_validation.score_columns",
        )
        if tuple(score_columns) != EXPECTED_SCORE_COLUMNS:
            raise CbsAdapterError("validated CBS score columns differ")
        for field in (
            "score_row_count",
            "count_row_count",
            "nonmissing_count_cell_count",
        ):
            _positive_int(offline[field], f"offline_validation.{field}")
        if type(offline["missing_count_cell_count"]) is not int or offline["missing_count_cell_count"] < 0:
            raise CbsAdapterError("offline_validation.missing_count_cell_count must be nonnegative")
        if (
            offline["score_row_count"] != CBS_SCORE_ROW_COUNT
            or offline["count_row_count"] != CBS_SCORE_ROW_COUNT
            or offline["nonmissing_count_cell_count"] + offline["missing_count_cell_count"]
            != CBS_SCORE_ROW_COUNT * len(EXPECTED_COUNT_COLUMNS)
        ):
            raise CbsAdapterError("validated CBS row or count-cell totals differ")
        expected_observed_hashes = {
            "metadata_sha256": EXPECTED_METADATA_BODY_SHA256,
            "scores_sha256": EXPECTED_SCORES_BODY_SHA256,
            "counts_sha256": EXPECTED_COUNTS_BODY_SHA256,
        }
        for field, expected in expected_observed_hashes.items():
            if offline[field] != expected:
                raise CbsAdapterError(f"offline_validation.{field} differs from the frozen registry")
        if (
            offline["score_count_accession_join_status"] != "validated_exact_order"
            or offline["mapped_current_accession_join_status"] != "validated_exact_set"
        ):
            raise CbsAdapterError("validated source bodies require exact accession joins")
        _, replayed_offline = validate_cbs_low_b6_adapter_inputs(source_inputs)
        if dict(offline) != dict(replayed_offline):
            raise CbsAdapterError("validated offline status differs from replayed source inputs")
    else:
        raise CbsAdapterError("decoded_source_bodies_status must be not_supplied or validated_structural_only")
    if tuple(offline["count_columns"]) != EXPECTED_COUNT_COLUMNS:
        raise CbsAdapterError("offline validation count columns differ")

    count_semantics = _exact_mapping(
        status["count_measurement_semantics"],
        COUNT_MEASUREMENT_SEMANTICS_FIELDS,
        "CBS adapter status.count_measurement_semantics",
    )
    if count_semantics != {
        "measurement_scale": COUNT_MEASUREMENT_SCALE,
        "value_contract": COUNT_VALUE_CONTRACT,
        "missing_tokens": sorted(COUNT_MISSING_TOKENS),
        "raw_read_count_claim": False,
        "controlNS_controlS_role": CONTROL_CHANNEL_ROLE,
        "functional_wt_baseline_claim": False,
    }:
        raise CbsAdapterError("CBS count measurement semantics differ")

    scientific_evidence = _exact_mapping(
        status["external_scientific_status_evidence"],
        EXTERNAL_SCIENTIFIC_STATUS_FIELDS,
        "CBS adapter status.external_scientific_status_evidence",
    )
    functional_rule = _exact_mapping(
        scientific_evidence["published_functional_classification_rule"],
        PUBLISHED_FUNCTIONAL_CLASSIFICATION_RULE_FIELDS,
        "CBS adapter status.external_scientific_status_evidence.published_functional_classification_rule",
    )
    remediability_rule = _exact_mapping(
        scientific_evidence["published_b6_remediability_rule"],
        PUBLISHED_B6_REMEDIABILITY_RULE_FIELDS,
        "CBS adapter status.external_scientific_status_evidence.published_b6_remediability_rule",
    )
    well_measured_rule = _exact_mapping(
        scientific_evidence["published_well_measured_rule"],
        PUBLISHED_WELL_MEASURED_RULE_FIELDS,
        "CBS adapter status.external_scientific_status_evidence.published_well_measured_rule",
    )
    uncertainty_evidence = _exact_mapping(
        scientific_evidence["score_uncertainty_channel_evidence"],
        SCORE_UNCERTAINTY_CHANNEL_EVIDENCE_FIELDS,
        "CBS adapter status.external_scientific_status_evidence.score_uncertainty_channel_evidence",
    )
    if uncertainty_evidence["replicate_count_claim"] is not False:
        raise CbsAdapterError("CBS uncertainty evidence replicate_count_claim must be false")
    expected_scientific_evidence = {
        "evidence_scope": ("external_scientific_status_only_not_source_lock_or_replay_authentication"),
        "source": "final_paper_additional_file_3",
        "raw_read_counts_lowB6_workbook_row_count": 22_536,
        "experimental_lowB6_workbook_row_count": 11_478,
        "raw_read_count_artifact_authentication_status": ("not_supplied_or_hash_authenticated_by_this_adapter"),
        "low_b6_fitness_condition_rule": ("average_0_and_1_ng_per_mL_conditions_due_to_high_agreement"),
        "published_functional_classification_rule": {
            "classification": ("deleterious_if_upper_95_percent_ci_below_threshold"),
            "upper_95_percent_ci_threshold": 0.60,
            "fdr": 0.05,
            "complement_classification": ("not_defined_as_neutral_or_retained_function"),
        },
        "published_b6_remediability_rule": {
            "eligibility": "classified_deleterious_under_low_b6_rule",
            "classification": ("remediable_if_lower_95_percent_ci_of_high_minus_low_above_threshold"),
            "lower_95_percent_ci_threshold": 0.22,
            "fdr": 0.05,
        },
        "published_well_measured_rule": {
            "preselection_allele_frequency_operator": ">",
            "preselection_allele_frequency_percent_threshold": 0.005,
            "standard_error_operator": "<",
            "standard_error_threshold": 0.2,
        },
        "score_uncertainty_channel_evidence": {
            "low_b6_sd_se_squared_ratio": 8,
            "high_b6_sd_se_squared_ratio": 4,
            "interpretation": ("consistent_with_eight_low_b6_and_four_high_b6_observation_channels"),
            "replicate_count_claim": False,
            "evidence_scope": ("external_scientific_status_only_sample_map_missing"),
        },
        "obsolete_parameters_not_used": [
            "fitness_threshold_0.45",
            "delta_0.29",
        ],
    }
    if (
        scientific_evidence != expected_scientific_evidence
        or functional_rule != expected_scientific_evidence["published_functional_classification_rule"]
        or remediability_rule != expected_scientific_evidence["published_b6_remediability_rule"]
        or well_measured_rule != expected_scientific_evidence["published_well_measured_rule"]
        or uncertainty_evidence != expected_scientific_evidence["score_uncertainty_channel_evidence"]
    ):
        raise CbsAdapterError("CBS external scientific status evidence differs")

    compatibility = _exact_mapping(
        status["schema_2_compatibility"],
        SCHEMA_2_COMPATIBILITY_FIELDS,
        "CBS adapter status.schema_2_compatibility",
    )
    if compatibility != {
        "compatible": False,
        "status": "INCOMPATIBLE_WITH_CURRENT_SCALAR_REPLAY",
        "reason_codes": list(SCHEMA_2_INCOMPATIBILITY_CODES),
    }:
        raise CbsAdapterError("CBS schema-2 compatibility boundary differs")
    expected_blockers = [*BASE_BLOCKER_CODES, SOURCE_BUNDLE_BLOCKER]
    expected_blockers.sort()
    if status["active_blocker_codes"] != expected_blockers:
        raise CbsAdapterError("CBS active blocker codes differ")
    if status["required_upstream_artifacts"] != list(REQUIRED_UPSTREAM_ARTIFACTS):
        raise CbsAdapterError("CBS required upstream artifacts differ")
    if status["prohibited_reinterpretations"] != list(PROHIBITED_REINTERPRETATIONS):
        raise CbsAdapterError("CBS prohibited reinterpretations differ")
    if _sha256(status["status_sha256"], "CBS adapter status.status_sha256") != (adapter_status_sha256(status)):
        raise CbsAdapterError("CBS adapter status self-hash differs")


def build_cbs_low_b6_assay_bundle(
    inputs: CbsLowB6AdapterInputs,
    *,
    native_replay_artifact: Mapping[str, Any] | None = None,
    treat_control_counts_as_functional_wt: bool = False,
    treat_aggregate_score_as_raw_replicate: bool = False,
    treat_relative_frequency_as_raw_read_count: bool = False,
) -> NoReturn:
    """Refuse schema-2 emission until a future native replay is authenticated."""

    validate_cbs_low_b6_adapter_inputs(inputs)
    if type(treat_control_counts_as_functional_wt) is not bool:
        raise TypeError("treat_control_counts_as_functional_wt must be boolean")
    if type(treat_aggregate_score_as_raw_replicate) is not bool:
        raise TypeError("treat_aggregate_score_as_raw_replicate must be boolean")
    if type(treat_relative_frequency_as_raw_read_count) is not bool:
        raise TypeError("treat_relative_frequency_as_raw_read_count must be boolean")
    if treat_control_counts_as_functional_wt:
        raise CbsAdapterError(
            "controlNS/controlS are error-control counts and cannot be treated as a functional wild-type baseline"
        )
    if treat_aggregate_score_as_raw_replicate:
        raise CbsAdapterError("the deposited aggregate score cannot be treated as a raw replicate")
    if treat_relative_frequency_as_raw_read_count:
        raise CbsAdapterError(
            "the deposited normalized relative read frequencies per one million "
            "total reads cannot be treated as raw read counts"
        )
    if native_replay_artifact is None:
        raise CbsAdapterError(
            "CBS assay-bundle emission requires a future authenticated native "
            "count-replay artifact and schema extension"
        )
    _mapping(native_replay_artifact, "CBS native replay artifact")
    raise CbsAdapterError("this candidate-only adapter cannot authenticate future native replay artifacts")


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    replace: bool,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise CbsAdapterError("CBS adapter status artifacts must use .json")
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
                raise CbsAdapterError(f"CBS adapter status already exists: {destination}") from exc
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_cbs_adapter_status(
    path: str | Path,
    status: Mapping[str, Any],
    *,
    replace: bool = False,
    source_inputs: CbsLowB6AdapterInputs | None = None,
) -> Path:
    """Atomically create or explicitly replace a validated status artifact."""

    validate_cbs_adapter_status(status, source_inputs=source_inputs)
    return _atomic_write_json(path, status, replace=replace)


def load_cbs_adapter_status(
    path: str | Path,
    *,
    source_inputs: CbsLowB6AdapterInputs | None = None,
) -> Mapping[str, Any]:
    status = _load_json_file(path, "CBS adapter status")
    validate_cbs_adapter_status(status, source_inputs=source_inputs)
    return status


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the offline candidate-only CBS low-B6 source lineage and "
            "write a non-ingested, non-outcome adapter status"
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--status-out", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--metadata-body", type=Path)
    parser.add_argument("--scores-body", type=Path)
    parser.add_argument("--counts-body", type=Path)
    parser.add_argument("--mapped-variants-body", type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing status file",
    )
    arguments = parser.parse_args()

    optional_paths = (
        arguments.source_lock,
        arguments.metadata_body,
        arguments.scores_body,
        arguments.counts_body,
        arguments.mapped_variants_body,
    )
    if any(path is not None for path in optional_paths) and not all(path is not None for path in optional_paths):
        parser.error("--source-lock and all four decoded body paths must be supplied together")

    source_lock = (
        _load_json_file(arguments.source_lock, "CBS MaveDB source lock") if arguments.source_lock is not None else None
    )
    inputs = CbsLowB6AdapterInputs(
        candidate_registry=_load_json_file(
            arguments.registry,
            "MaveDB candidate registry",
        ),
        materialized_source_lock=source_lock,
        metadata_body=(arguments.metadata_body.read_bytes() if arguments.metadata_body is not None else None),
        scores_body=(arguments.scores_body.read_bytes() if arguments.scores_body is not None else None),
        counts_body=(arguments.counts_body.read_bytes() if arguments.counts_body is not None else None),
        mapped_variants_body=(
            arguments.mapped_variants_body.read_bytes() if arguments.mapped_variants_body is not None else None
        ),
    )
    status = build_cbs_low_b6_adapter_status(inputs)
    destination = write_cbs_adapter_status(
        arguments.status_out,
        status,
        replace=arguments.replace,
        source_inputs=inputs if source_lock is not None else None,
    )
    print(
        json.dumps(
            {
                "path": str(destination),
                "source_readiness": status["source_readiness"],
                "confirmatory_eligible": status["confirmatory_eligible"],
                "outcome_status": status["outcome_status"],
                "status_sha256": status["status_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()

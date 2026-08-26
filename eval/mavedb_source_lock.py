"""Deterministic, candidate-stage source locks for public MaveDB score sets.

This module snapshots and validates source bytes only.  A successful lock does
not mean that a score set has been ingested, that an assay-specific outcome has
been derived, or that a confirmatory analysis is authorized.

The transport layer validates the wire response before decoding gzip.  The
contract layer then validates complete JSON/CSV payloads, binds the active
license-bearing metadata to all tabular and mapping bytes, and records only
hashes and structural summaries in the lock.  Readiness is supplied by the
caller and can only be accepted when its minimum evidence is present; the
module never promotes a source to a higher readiness state.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "groundbench.dms_mavedb_candidate_source_lock"
CLAIM_SCOPE = "candidate_source_snapshot_only_no_ingestion_or_outcome_claim"
DEFAULT_API_BASE_URL = "https://api.mavedb.org/api/v1"
DEFAULT_OPENAPI_URL = "https://api.mavedb.org/openapi.json"

FIXED_VARIANT_COLUMNS = (
    "accession",
    "hgvs_nt",
    "hgvs_splice",
    "hgvs_pro",
)
RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
EXPECTED_HASH_KEYS = frozenset(
    {
        "openapi",
        "metadata",
        "scores",
        "counts",
        "mapped_variants",
        "metadata_tabular_license_binding",
    }
)
MISSING_TABLE_VALUES = frozenset({"", "na", "nan", "none", "null"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
URN_PATTERN = re.compile(r"^urn:mavedb:[0-9]{8}-(?:[a-z]+|0)-[0-9]+$")
SEQUENCE_PATTERN = re.compile(r"^[A-Z*.-]+$")

FETCHED_ARTIFACT_FIELDS = frozenset(
    {
        "url",
        "status",
        "attempts",
        "content_type",
        "transfer_encoding",
        "content_encoding",
        "declared_content_length",
        "wire_byte_count",
        "wire_sha256",
        "decoded_byte_count",
        "sha256",
    }
)
READINESS_FIELDS = frozenset({"state", "caller_configured", "automatic_promotion", "evidence"})
READINESS_EVIDENCE_FIELDS = frozenset(
    {
        "aggregate_score_column",
        "processed_replicate_columns",
        "count_lineage_columns",
        "identity_resolution_spec_sha256",
        "transformation_spec_sha256",
        "wt_control_spec_sha256",
        "confirmatory_preregistration_sha256",
        "independent_replication_spec_sha256",
        "identity_block_reason",
        "rejection_reason",
    }
)
API_FIELDS = frozenset({"base_url", "openapi_url", "openapi_version", "openapi_artifact"})
METADATA_CONTRACT_FIELDS = frozenset(
    {
        "title",
        "num_variants",
        "license",
        "license_sha256",
        "data_usage_policy",
        "private",
        "processing_state",
        "superseded_score_set",
        "superseding_score_set",
        "dataset_columns",
        "dataset_columns_sha256",
        "calibration_count",
        "calibrations_sha256",
        "targets",
    }
)
LICENSE_FIELDS = frozenset(
    {
        "id",
        "shortName",
        "longName",
        "active",
        "link",
        "version",
        "recordType",
    }
)
DATASET_COLUMNS_FIELDS = frozenset({"scoreColumns", "countColumns", "recordType"})
TARGET_COMMON_FIELDS = frozenset(
    {
        "name",
        "mapped_hgnc_name",
        "external_identifiers_sha256",
        "target_kind",
        "accession",
        "sequence_type",
        "sequence_length",
        "sequence_sha256",
        "reference_artifact",
    }
)
TARGET_ACCESSION_METADATA_FIELDS = frozenset({"accession", "assembly", "gene", "isBaseEditor", "recordType"})
TABULAR_CONTRACT_FIELDS = frozenset(
    {
        "fixed_variant_columns",
        "score_header",
        "count_header",
        "score_row_count",
        "count_row_count",
        "accessions_sha256",
        "sorted_accessions_sha256",
        "counts_status",
        "substantive_count_column_count",
        "substantive_count_value_count",
        "score_finite_value_counts",
        "count_finite_value_counts",
        "score_invalid_value_counts",
        "count_invalid_value_counts",
        "native_hgvs_nt_duplicate_group_count",
        "native_hgvs_pro_duplicate_group_count",
    }
)
MAPPING_CONTRACT_FIELDS = frozenset(
    {
        "history_record_count",
        "current_record_count",
        "current_unique_variant_count",
        "current_duplicate_variant_count",
        "current_missing_variant_count",
        "current_extra_variant_count",
        "current_error_count",
        "current_post_mapped_null_count",
        "current_pre_mapped_null_count",
        "current_at_mismatched_locus_count",
        "current_near_gap_count",
        "alignment_levels",
        "mapping_api_versions",
        "vrs_versions",
        "current_accessions_sha256",
        "current_mappings_sha256",
    }
)


class SourceLockError(ValueError):
    """Raised when a candidate source cannot be locked fail-closed."""


class Readiness(str, Enum):
    REJECTED = "REJECTED"
    IDENTITY_BLOCKED = "IDENTITY_BLOCKED"
    AGGREGATE_ONLY = "AGGREGATE_ONLY"
    PROCESSED_REPLICATES = "PROCESSED_REPLICATES"
    COUNT_LINEAGE_PARTIAL = "COUNT_LINEAGE_PARTIAL"
    COUNT_RECOMPUTABLE = "COUNT_RECOMPUTABLE"
    CONFIRMATORY_READY = "CONFIRMATORY_READY"


@dataclass(frozen=True)
class ReadinessEvidence:
    """Caller-declared evidence that is checked against fetched source bytes."""

    aggregate_score_column: str | None = None
    processed_replicate_columns: tuple[str, ...] = ()
    count_lineage_columns: tuple[str, ...] = ()
    identity_resolution_spec_sha256: str | None = None
    transformation_spec_sha256: str | None = None
    wt_control_spec_sha256: str | None = None
    confirmatory_preregistration_sha256: str | None = None
    independent_replication_spec_sha256: str | None = None
    identity_block_reason: str | None = None
    rejection_reason: str | None = None

    def as_lock(self) -> dict[str, Any]:
        return {
            "aggregate_score_column": self.aggregate_score_column,
            "processed_replicate_columns": list(self.processed_replicate_columns),
            "count_lineage_columns": list(self.count_lineage_columns),
            "identity_resolution_spec_sha256": self.identity_resolution_spec_sha256,
            "transformation_spec_sha256": self.transformation_spec_sha256,
            "wt_control_spec_sha256": self.wt_control_spec_sha256,
            "confirmatory_preregistration_sha256": self.confirmatory_preregistration_sha256,
            "independent_replication_spec_sha256": self.independent_replication_spec_sha256,
            "identity_block_reason": self.identity_block_reason,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class SourceLockConfig:
    urn: str
    readiness: Readiness
    readiness_evidence: ReadinessEvidence
    api_base_url: str = DEFAULT_API_BASE_URL
    openapi_url: str = DEFAULT_OPENAPI_URL
    expected_api_version: str | None = None
    expected_sha256: Mapping[str, str] = field(default_factory=dict)
    expected_reference_sha256: Mapping[str, str] = field(default_factory=dict)
    max_attempts: int = 3


@dataclass(frozen=True)
class HttpResponse:
    """Raw, not transparently decompressed, HTTP response."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str | None = None


Transport = Callable[[str, Mapping[str, str]], HttpResponse]


@dataclass(frozen=True)
class _FetchedArtifact:
    url: str
    status: int
    attempts: int
    content_type: str
    transfer_encoding: str | None
    content_encoding: str | None
    declared_content_length: int | None
    wire_body: bytes
    decoded_body: bytes

    def as_lock(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status,
            "attempts": self.attempts,
            "content_type": self.content_type,
            "transfer_encoding": self.transfer_encoding,
            "content_encoding": self.content_encoding,
            "declared_content_length": self.declared_content_length,
            "wire_byte_count": len(self.wire_body),
            "wire_sha256": _bytes_sha256(self.wire_body),
            "decoded_byte_count": len(self.decoded_body),
            "sha256": _bytes_sha256(self.decoded_body),
        }


@dataclass(frozen=True)
class _CsvTable:
    header: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    accessions: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceLockError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return _bytes_sha256(canonical_json_bytes(value))


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise SourceLockError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, context)


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLockError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise SourceLockError(f"{context} must be a list")
    result = [_require_nonempty_string(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise SourceLockError(f"{context} must not contain duplicates")
    return result


def _require_exact_mapping(
    value: Any,
    expected_fields: frozenset[str] | set[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected_fields):
        raise SourceLockError(f"{context} must use the exact nested schema")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 1:
        raise SourceLockError(f"{context} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SourceLockError(f"{context} must be a nonnegative integer")
    return value


def _require_optional_nonempty_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, context)


def _require_https_url(value: Any, context: str) -> str:
    url = _require_nonempty_string(value, context)
    if not url.startswith("https://"):
        raise SourceLockError(f"{context} must use HTTPS")
    return url


def _require_sorted_string_list(value: Any, context: str) -> list[str]:
    result = _require_string_list(value, context)
    if result != sorted(result):
        raise SourceLockError(f"{context} must be sorted")
    return result


def _canonical_copy(value: Any, context: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # pragma: no cover
        raise SourceLockError(f"{context} is not JSON-compatible") from exc


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value).strip()
    return None


def _urllib_transport(url: str, headers: Mapping[str, str]) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=180) as response:  # noqa: S310
            return HttpResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
                url=response.geturl(),
            )
    except HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            body=exc.read(),
            url=exc.geturl(),
        )


def _decode_http_200(
    response: HttpResponse,
    *,
    url: str,
    attempts: int,
    expected_content_types: frozenset[str],
) -> _FetchedArtifact:
    if not isinstance(response.body, bytes):
        raise SourceLockError(f"{url} response body must be bytes")
    content_length_value = _header(response.headers, "content-length")
    transfer_encoding_value = _header(response.headers, "transfer-encoding")
    transfer_encoding = transfer_encoding_value.lower() if transfer_encoding_value else None
    if content_length_value is not None and transfer_encoding is not None:
        raise SourceLockError(f"{url} ambiguously provides both HTTP framing headers")
    if content_length_value is None:
        if transfer_encoding != "chunked":
            raise SourceLockError(f"{url} is missing supported HTTP response framing")
        declared_content_length = None
    else:
        try:
            declared_content_length = int(content_length_value)
        except ValueError as exc:
            raise SourceLockError(f"{url} has an invalid Content-Length") from exc
        if declared_content_length < 0 or declared_content_length != len(response.body):
            raise SourceLockError(
                f"{url} Content-Length mismatch: declared={declared_content_length}, received={len(response.body)}"
            )

    raw_content_type = _header(response.headers, "content-type")
    if raw_content_type is None:
        raise SourceLockError(f"{url} is missing Content-Type")
    content_type = raw_content_type.split(";", 1)[0].strip().lower()
    if content_type not in expected_content_types:
        raise SourceLockError(
            f"{url} has Content-Type {content_type!r}; expected one of {sorted(expected_content_types)}"
        )

    content_encoding_value = _header(response.headers, "content-encoding")
    content_encoding = content_encoding_value.lower() if content_encoding_value else None
    if content_encoding is None or content_encoding == "identity":
        decoded_body = response.body
        content_encoding = None
    elif content_encoding == "gzip":
        try:
            decoded_body = gzip.decompress(response.body)
        except (EOFError, OSError) as exc:
            raise SourceLockError(f"{url} has an incomplete gzip body") from exc
    else:
        raise SourceLockError(f"{url} uses unsupported Content-Encoding {content_encoding!r}")
    if not decoded_body:
        raise SourceLockError(f"{url} returned an empty decoded body")
    return _FetchedArtifact(
        url=url,
        status=200,
        attempts=attempts,
        content_type=content_type,
        transfer_encoding=transfer_encoding,
        content_encoding=content_encoding,
        declared_content_length=declared_content_length,
        wire_body=response.body,
        decoded_body=decoded_body,
    )


def _fetch_exact(
    url: str,
    *,
    transport: Transport,
    max_attempts: int,
    expected_content_types: frozenset[str],
) -> _FetchedArtifact:
    if max_attempts < 1:
        raise SourceLockError("max_attempts must be at least 1")
    headers = {
        "Accept-Encoding": "gzip",
        "Accept": ", ".join(sorted(expected_content_types)),
        "User-Agent": "groundbench-mavedb-source-lock/1",
    }
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = transport(url, headers)
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_attempts:
                continue
            raise SourceLockError(f"{url} transport failed after {attempt} attempts") from exc
        if not isinstance(response, HttpResponse):
            raise SourceLockError("transport must return HttpResponse")
        if response.status != 200:
            last_error = SourceLockError(f"{url} returned HTTP {response.status}")
            if response.status in RETRYABLE_HTTP_STATUSES and attempt < max_attempts:
                continue
            raise last_error
        try:
            return _decode_http_200(
                response,
                url=url,
                attempts=attempt,
                expected_content_types=expected_content_types,
            )
        except SourceLockError as exc:
            last_error = exc
            if attempt < max_attempts:
                continue
            raise
    raise SourceLockError(f"{url} fetch failed") from last_error  # pragma: no cover


def _parse_json_bytes(value: bytes, context: str) -> Any:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceLockError(f"{context} is not valid UTF-8") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceLockError(f"{context} is incomplete or invalid JSON") from exc


def _parse_csv_bytes(
    value: bytes,
    *,
    context: str,
    expected_header: Sequence[str],
    expected_rows: int,
    urn: str,
) -> _CsvTable:
    try:
        text = value.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceLockError(f"{context} is not valid UTF-8") from exc
    if "\x00" in text:
        raise SourceLockError(f"{context} contains NUL bytes")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = tuple(next(reader))
        raw_rows = list(reader)
    except (csv.Error, StopIteration) as exc:
        raise SourceLockError(f"{context} is incomplete or invalid CSV") from exc
    if header != tuple(expected_header):
        raise SourceLockError(
            f"{context} schema differs: observed={list(header)!r}, expected={list(expected_header)!r}"
        )
    if len(raw_rows) != expected_rows:
        raise SourceLockError(f"{context} row count differs: observed={len(raw_rows)}, expected={expected_rows}")
    records: list[dict[str, str]] = []
    accessions: list[str] = []
    for index, row in enumerate(raw_rows):
        if len(row) != len(header):
            raise SourceLockError(
                f"{context} row {index + 2} is truncated: observed={len(row)}, expected={len(header)}"
            )
        record = dict(zip(header, row, strict=True))
        accession = record["accession"]
        if not accession.startswith(f"{urn}#"):
            raise SourceLockError(f"{context} row {index + 2} has an invalid accession")
        records.append(record)
        accessions.append(accession)
    if len(set(accessions)) != len(accessions):
        raise SourceLockError(f"{context} accessions must be unique")
    return _CsvTable(
        header=header,
        rows=tuple(records),
        accessions=tuple(accessions),
    )


def _is_missing_table_value(value: str) -> bool:
    return value.strip().lower() in MISSING_TABLE_VALUES


def _finite_numeric_count(
    table: _CsvTable,
    column: str,
    *,
    context: str,
    nonnegative: bool = False,
) -> int:
    count = 0
    for index, row in enumerate(table.rows):
        raw = row[column]
        if _is_missing_table_value(raw):
            continue
        try:
            number = float(raw)
        except ValueError as exc:
            raise SourceLockError(f"{context} row {index + 2} is not numeric") from exc
        if not math.isfinite(number):
            raise SourceLockError(f"{context} row {index + 2} is not finite")
        if nonnegative and number < 0:
            raise SourceLockError(f"{context} row {index + 2} is negative")
        count += 1
    return count


def _finite_numeric_counts_by_column(
    table: _CsvTable,
    columns: Sequence[str],
) -> dict[str, int]:
    """Count finite numeric cells without treating annotation text as an error."""

    result: dict[str, int] = {}
    for column in columns:
        count = 0
        for row in table.rows:
            raw = row[column]
            if _is_missing_table_value(raw):
                continue
            try:
                number = float(raw)
            except ValueError:
                continue
            if math.isfinite(number):
                count += 1
        result[column] = count
    return result


def _invalid_numeric_counts_by_column(
    table: _CsvTable,
    columns: Sequence[str],
) -> dict[str, int]:
    """Count nonmissing cells rejected by the strict numeric parser."""

    result: dict[str, int] = {}
    for column in columns:
        count = 0
        for row in table.rows:
            raw = row[column]
            if _is_missing_table_value(raw):
                continue
            try:
                number = float(raw)
            except ValueError:
                count += 1
                continue
            if not math.isfinite(number):
                count += 1
        result[column] = count
    return result


def _duplicate_nonmissing_values(table: _CsvTable, column: str) -> dict[str, int]:
    counts = Counter(row[column] for row in table.rows if not _is_missing_table_value(row[column]))
    duplicated = {value: count for value, count in counts.items() if count > 1}
    return dict(sorted(duplicated.items()))


def _validate_metadata(metadata: Any, urn: str) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise SourceLockError("MaveDB metadata must be a JSON object")
    if metadata.get("urn") != urn:
        raise SourceLockError("MaveDB metadata URN differs from the configured URN")
    if metadata.get("private") is not False:
        raise SourceLockError("MaveDB score set must be explicitly public/non-private")
    if metadata.get("processingState") != "success":
        raise SourceLockError("MaveDB processingState must be success")
    if metadata.get("supersededScoreSet") is not None:
        raise SourceLockError("superseded MaveDB score sets are rejected")
    if metadata.get("supersedingScoreSet") is not None:
        raise SourceLockError("MaveDB score sets with a superseding record are rejected")

    license_value = _require_exact_mapping(
        metadata.get("license"),
        LICENSE_FIELDS,
        "MaveDB license",
    )
    _require_positive_int(license_value.get("id"), "MaveDB license.id")
    if license_value.get("active") is not True:
        raise SourceLockError("MaveDB license must be active")
    for field_name in ("shortName", "longName", "link", "version", "recordType"):
        _require_nonempty_string(
            license_value.get(field_name),
            f"MaveDB license.{field_name}",
        )
    if license_value["recordType"] != "ShortLicense":
        raise SourceLockError("MaveDB license.recordType must be ShortLicense")
    if not str(license_value["link"]).startswith("https://"):
        raise SourceLockError("MaveDB license.link must use HTTPS")

    num_variants = metadata.get("numVariants")
    if type(num_variants) is not int or num_variants < 1:
        raise SourceLockError("MaveDB numVariants must be a positive integer")
    dataset_columns = _require_exact_mapping(
        metadata.get("datasetColumns"),
        DATASET_COLUMNS_FIELDS,
        "MaveDB datasetColumns",
    )
    if dataset_columns.get("recordType") != "DatasetColumns":
        raise SourceLockError("MaveDB datasetColumns.recordType must be DatasetColumns")
    score_columns = _require_string_list(
        dataset_columns.get("scoreColumns"),
        "MaveDB datasetColumns.scoreColumns",
    )
    count_columns = _require_string_list(
        dataset_columns.get("countColumns"),
        "MaveDB datasetColumns.countColumns",
    )
    if "score" not in score_columns:
        raise SourceLockError("MaveDB scoreColumns must contain the required score column")
    overlap = set(FIXED_VARIANT_COLUMNS) & (set(score_columns) | set(count_columns))
    if overlap:
        raise SourceLockError(f"MaveDB custom columns collide with fixed columns: {sorted(overlap)}")

    calibrations = metadata.get("scoreCalibrations")
    if calibrations is None:
        calibrations = []
    if not isinstance(calibrations, list):
        raise SourceLockError("MaveDB scoreCalibrations must be a list or null")
    target_genes = metadata.get("targetGenes")
    if not isinstance(target_genes, list) or not target_genes:
        raise SourceLockError("MaveDB metadata must contain at least one target gene")
    data_usage_policy = metadata.get("dataUsagePolicy")
    if data_usage_policy is not None:
        _require_nonempty_string(data_usage_policy, "MaveDB dataUsagePolicy")

    return {
        "title": _require_nonempty_string(metadata.get("title"), "MaveDB title"),
        "num_variants": num_variants,
        "license": _canonical_copy(license_value, "MaveDB license"),
        "data_usage_policy": data_usage_policy,
        "private": False,
        "processing_state": "success",
        "superseded_score_set": None,
        "superseding_score_set": None,
        "dataset_columns": _canonical_copy(dataset_columns, "MaveDB datasetColumns"),
        "score_columns": score_columns,
        "count_columns": count_columns,
        "calibrations": _canonical_copy(calibrations, "MaveDB scoreCalibrations"),
        "target_genes": target_genes,
    }


def _normalize_sequence(value: bytes | str, context: str) -> str:
    if isinstance(value, bytes):
        try:
            sequence = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceLockError(f"{context} must be ASCII") from exc
    elif isinstance(value, str):
        sequence = value
    else:
        raise SourceLockError(f"{context} must be bytes or a string")
    if not sequence or sequence != sequence.strip() or any(character.isspace() for character in sequence):
        raise SourceLockError(f"{context} must be a non-empty whitespace-free sequence")
    if sequence != sequence.upper() or SEQUENCE_PATTERN.fullmatch(sequence) is None:
        raise SourceLockError(f"{context} must be an uppercase biological sequence")
    return sequence


def _target_contracts(
    target_genes: Sequence[Any],
    *,
    config: SourceLockConfig,
    transport: Transport,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    consumed_expected_reference_keys: set[str] = set()
    api_base = config.api_base_url.rstrip("/")
    for index, target_raw in enumerate(target_genes):
        context = f"MaveDB targetGenes[{index}]"
        if not isinstance(target_raw, Mapping):
            raise SourceLockError(f"{context} must be an object")
        name = _require_nonempty_string(target_raw.get("name"), f"{context}.name")
        target_sequence = target_raw.get("targetSequence")
        target_accession = target_raw.get("targetAccession")
        if (target_sequence is None) == (target_accession is None):
            raise SourceLockError(f"{context} must have exactly one target sequence or accession")

        target: dict[str, Any] = {
            "name": name,
            "mapped_hgnc_name": target_raw.get("mappedHgncName"),
            "external_identifiers_sha256": canonical_sha256(target_raw.get("externalIdentifiers", [])),
        }
        if target_sequence is not None:
            if not isinstance(target_sequence, Mapping):
                raise SourceLockError(f"{context}.targetSequence must be an object")
            sequence = _normalize_sequence(
                target_sequence.get("sequence"),
                f"{context}.targetSequence.sequence",
            )
            sequence_type = _require_nonempty_string(
                target_sequence.get("sequenceType"),
                f"{context}.targetSequence.sequenceType",
            )
            target.update(
                {
                    "target_kind": "embedded_sequence",
                    "accession": None,
                    "sequence_type": sequence_type,
                    "sequence_length": len(sequence),
                    "sequence_sha256": _bytes_sha256(sequence.encode("ascii")),
                    "reference_artifact": None,
                }
            )
            expected_key = f"target:{name}"
            expected_reference = config.expected_reference_sha256.get(expected_key)
            if expected_reference is not None:
                consumed_expected_reference_keys.add(expected_key)
                if target["sequence_sha256"] != _require_sha256(
                    expected_reference,
                    f"expected reference SHA-256 for {expected_key}",
                ):
                    raise SourceLockError(f"MaveDB embedded target sequence hash mismatch for {name}")
        else:
            if not isinstance(target_accession, Mapping):
                raise SourceLockError(f"{context}.targetAccession must be an object")
            accession = _require_nonempty_string(
                target_accession.get("accession"),
                f"{context}.targetAccession.accession",
            )
            reference_url = f"{api_base}/refget/sequence/{quote(accession, safe='._-')}"
            fetched = _fetch_exact(
                reference_url,
                transport=transport,
                max_attempts=config.max_attempts,
                expected_content_types=frozenset({"text/plain"}),
            )
            sequence = _normalize_sequence(
                fetched.decoded_body,
                f"MaveDB Refget sequence {accession}",
            )
            sequence_sha256 = _bytes_sha256(sequence.encode("ascii"))
            expected_reference = config.expected_reference_sha256.get(accession)
            if expected_reference is not None:
                consumed_expected_reference_keys.add(accession)
                if sequence_sha256 != _require_sha256(
                    expected_reference,
                    f"expected reference SHA-256 for {accession}",
                ):
                    raise SourceLockError(f"MaveDB reference sequence hash mismatch for {accession}")
            target.update(
                {
                    "target_kind": "accession",
                    "accession": accession,
                    "sequence_type": "accession_resolved",
                    "sequence_length": len(sequence),
                    "sequence_sha256": sequence_sha256,
                    "target_accession_metadata": _canonical_copy(
                        target_accession,
                        f"{context}.targetAccession",
                    ),
                    "reference_artifact": fetched.as_lock(),
                }
            )
        result.append(target)
    unused_expected_reference_keys = set(config.expected_reference_sha256) - consumed_expected_reference_keys
    if unused_expected_reference_keys:
        raise SourceLockError(f"unused expected reference SHA-256 keys: {sorted(unused_expected_reference_keys)}")
    return result


def _mapping_summary(
    mapped_value: Any,
    *,
    urn: str,
    expected_accessions: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(mapped_value, list):
        raise SourceLockError("MaveDB mapped-variants body must be a JSON array")
    current: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(mapped_value):
        if not isinstance(raw_record, Mapping):
            raise SourceLockError(f"MaveDB mapped-variants[{index}] must be an object")
        variant_urn = raw_record.get("variantUrn")
        if not isinstance(variant_urn, str) or not variant_urn.startswith(f"{urn}#"):
            raise SourceLockError(f"MaveDB mapped-variants[{index}] has an invalid variantUrn")
        if type(raw_record.get("current")) is not bool:
            raise SourceLockError(f"MaveDB mapped-variants[{index}].current must be boolean")
        if raw_record["current"]:
            current.append(raw_record)

    current_accessions = [str(record["variantUrn"]) for record in current]
    current_counts = Counter(current_accessions)
    duplicate_current = sorted(accession for accession, count in current_counts.items() if count != 1)
    expected_set = set(expected_accessions)
    current_set = set(current_accessions)
    missing = sorted(expected_set - current_set)
    extra = sorted(current_set - expected_set)
    if duplicate_current:
        raise SourceLockError(f"MaveDB mapped-variants has duplicate current mappings: {duplicate_current[:5]}")
    if missing or extra:
        raise SourceLockError(
            f"MaveDB current mapping coverage differs from score accessions: missing={missing[:5]}, extra={extra[:5]}"
        )

    def _nonnull_string_values(field_name: str) -> list[str]:
        return sorted({str(record[field_name]) for record in current if record.get(field_name) is not None})

    current_errors = [
        record
        for record in current
        if record.get("postMapped") is None
        or (record.get("errorMessage") is not None and str(record.get("errorMessage")).strip())
    ]
    canonical_current = sorted(
        (_canonical_copy(record, "current mapped variant") for record in current),
        key=lambda record: record["variantUrn"],
    )
    return {
        "history_record_count": len(mapped_value),
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
        "alignment_levels": _nonnull_string_values("alignmentLevel"),
        "mapping_api_versions": _nonnull_string_values("mappingApiVersion"),
        "vrs_versions": _nonnull_string_values("vrsVersion"),
        "current_accessions_sha256": canonical_sha256(sorted(current_set)),
        "current_mappings_sha256": canonical_sha256(canonical_current),
    }


def _validate_readiness(
    readiness: Readiness,
    evidence: ReadinessEvidence,
    *,
    score_columns: Sequence[str],
    count_columns: Sequence[str],
    scores: _CsvTable,
    counts: _CsvTable,
    counts_status: str,
    mapping_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness is Readiness.CONFIRMATORY_READY:
        raise SourceLockError(
            "candidate-stage MaveDB source locks reject CONFIRMATORY_READY; "
            "caller-supplied hashes cannot authorize confirmatory execution"
        )
    evidence_lock = evidence.as_lock()
    for field_name in (
        "identity_resolution_spec_sha256",
        "transformation_spec_sha256",
        "wt_control_spec_sha256",
        "confirmatory_preregistration_sha256",
        "independent_replication_spec_sha256",
    ):
        evidence_lock[field_name] = _optional_sha256(
            evidence_lock[field_name],
            f"readiness evidence.{field_name}",
        )

    aggregate_states = {
        Readiness.AGGREGATE_ONLY,
        Readiness.PROCESSED_REPLICATES,
        Readiness.COUNT_LINEAGE_PARTIAL,
        Readiness.COUNT_RECOMPUTABLE,
        Readiness.CONFIRMATORY_READY,
    }
    if readiness is Readiness.REJECTED:
        _require_nonempty_string(evidence.rejection_reason, "readiness rejection_reason")
    if readiness is Readiness.IDENTITY_BLOCKED:
        _require_nonempty_string(
            evidence.identity_block_reason,
            "readiness identity_block_reason",
        )

    aggregate_column = evidence.aggregate_score_column
    if readiness in aggregate_states:
        aggregate_column = _require_nonempty_string(
            aggregate_column,
            "readiness aggregate_score_column",
        )
    if aggregate_column is not None:
        if aggregate_column not in score_columns:
            raise SourceLockError(f"readiness aggregate score column {aggregate_column!r} is absent")
        if (
            _finite_numeric_count(
                scores,
                aggregate_column,
                context=f"MaveDB scores.{aggregate_column}",
            )
            == 0
        ):
            raise SourceLockError("readiness aggregate score column has no finite values")

    replicate_columns = list(evidence.processed_replicate_columns)
    if len(set(replicate_columns)) != len(replicate_columns):
        raise SourceLockError("readiness processed replicate columns contain duplicates")
    for column in replicate_columns:
        if column not in score_columns or column == aggregate_column:
            raise SourceLockError(f"readiness replicate score column {column!r} is invalid")
        if (
            _finite_numeric_count(
                scores,
                column,
                context=f"MaveDB scores.{column}",
            )
            == 0
        ):
            raise SourceLockError(f"readiness replicate score column {column!r} has no values")
    if readiness in {Readiness.PROCESSED_REPLICATES, Readiness.CONFIRMATORY_READY} and not (replicate_columns):
        raise SourceLockError(f"{readiness.value} requires processed replicate columns")

    declared_count_columns = list(evidence.count_lineage_columns)
    if len(set(declared_count_columns)) != len(declared_count_columns):
        raise SourceLockError("readiness count lineage columns contain duplicates")
    for column in declared_count_columns:
        if column not in count_columns:
            raise SourceLockError(f"readiness count lineage column {column!r} is absent")
        if (
            _finite_numeric_count(
                counts,
                column,
                context=f"MaveDB counts.{column}",
                nonnegative=True,
            )
            == 0
        ):
            raise SourceLockError(f"readiness count lineage column {column!r} has no finite values")
    count_states = {
        Readiness.COUNT_LINEAGE_PARTIAL,
        Readiness.COUNT_RECOMPUTABLE,
        Readiness.CONFIRMATORY_READY,
    }
    if readiness in count_states:
        if counts_status != "substantive":
            raise SourceLockError(f"{readiness.value} requires substantive count columns")
        if set(declared_count_columns) != set(count_columns):
            raise SourceLockError(f"{readiness.value} must bind every MaveDB count lineage column")

    if readiness in {Readiness.COUNT_RECOMPUTABLE, Readiness.CONFIRMATORY_READY}:
        for field_name in (
            "identity_resolution_spec_sha256",
            "transformation_spec_sha256",
            "wt_control_spec_sha256",
        ):
            if evidence_lock[field_name] is None:
                raise SourceLockError(f"{readiness.value} requires {field_name}")
        if mapping_summary["current_error_count"] != 0:
            raise SourceLockError(f"{readiness.value} requires error-free current mappings")

    return {
        "state": readiness.value,
        "caller_configured": True,
        "automatic_promotion": False,
        "evidence": evidence_lock,
    }


def _validate_expected_hashes(config: SourceLockConfig) -> dict[str, str]:
    expected = dict(config.expected_sha256)
    unknown = set(expected) - EXPECTED_HASH_KEYS
    if unknown:
        raise SourceLockError(f"unknown expected SHA-256 keys: {sorted(unknown)}")
    return {key: _require_sha256(value, f"expected_sha256.{key}") for key, value in expected.items()}


def _check_artifact_hash(
    name: str,
    artifact: _FetchedArtifact,
    expected_hashes: Mapping[str, str],
) -> None:
    expected = expected_hashes.get(name)
    observed = _bytes_sha256(artifact.decoded_body)
    if expected is not None and observed != expected:
        raise SourceLockError(f"MaveDB {name} hash mismatch: observed={observed}, expected={expected}")


def _license_binding_payload(lock: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = lock["source_artifacts"]
    metadata_contract = lock["metadata_contract"]
    return {
        "urn": lock["urn"],
        "metadata_sha256": artifacts["metadata"]["sha256"],
        "scores_sha256": artifacts["scores"]["sha256"],
        "counts_sha256": artifacts["counts"]["sha256"],
        "mapped_variants_sha256": artifacts["mapped_variants"]["sha256"],
        "license": metadata_contract["license"],
        "private": metadata_contract["private"],
        "data_usage_policy": metadata_contract["data_usage_policy"],
    }


def _source_bundle_payload(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lock.items() if key != "source_bundle_sha256"}


def build_candidate_source_lock(
    config: SourceLockConfig,
    *,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Fetch, validate, and summarize one configured public MaveDB score set."""

    if not isinstance(config, SourceLockConfig):
        raise TypeError("config must be SourceLockConfig")
    if URN_PATTERN.fullmatch(config.urn) is None:
        raise SourceLockError("configured MaveDB URN is invalid")
    if not isinstance(config.readiness, Readiness):
        raise SourceLockError("readiness must be a Readiness enum value")
    if not isinstance(config.readiness_evidence, ReadinessEvidence):
        raise SourceLockError("readiness_evidence must be ReadinessEvidence")
    if config.max_attempts < 1:
        raise SourceLockError("max_attempts must be at least 1")
    expected_hashes = _validate_expected_hashes(config)
    resolved_transport = transport or _urllib_transport
    api_base = config.api_base_url.rstrip("/")
    encoded_urn = quote(config.urn, safe=":")
    urls = {
        "metadata": f"{api_base}/score-sets/{encoded_urn}",
        "scores": f"{api_base}/score-sets/{encoded_urn}/scores",
        "counts": f"{api_base}/score-sets/{encoded_urn}/counts",
        "mapped_variants": f"{api_base}/score-sets/{encoded_urn}/mapped-variants",
    }

    openapi_artifact = _fetch_exact(
        config.openapi_url,
        transport=resolved_transport,
        max_attempts=config.max_attempts,
        expected_content_types=frozenset({"application/json"}),
    )
    _check_artifact_hash("openapi", openapi_artifact, expected_hashes)
    openapi = _parse_json_bytes(openapi_artifact.decoded_body, "MaveDB OpenAPI")
    if not isinstance(openapi, Mapping):
        raise SourceLockError("MaveDB OpenAPI must be a JSON object")
    info = openapi.get("info")
    if not isinstance(info, Mapping):
        raise SourceLockError("MaveDB OpenAPI is missing info")
    api_version = _require_nonempty_string(info.get("version"), "MaveDB OpenAPI info.version")
    if config.expected_api_version is not None and api_version != config.expected_api_version:
        raise SourceLockError(
            f"MaveDB API version mismatch: observed={api_version}, expected={config.expected_api_version}"
        )

    fetched: dict[str, _FetchedArtifact] = {}
    for name in ("metadata", "scores", "counts", "mapped_variants"):
        expected_types = (
            frozenset({"application/json"}) if name in {"metadata", "mapped_variants"} else frozenset({"text/csv"})
        )
        artifact = _fetch_exact(
            urls[name],
            transport=resolved_transport,
            max_attempts=config.max_attempts,
            expected_content_types=expected_types,
        )
        _check_artifact_hash(name, artifact, expected_hashes)
        fetched[name] = artifact

    metadata_value = _parse_json_bytes(
        fetched["metadata"].decoded_body,
        "MaveDB metadata",
    )
    metadata = _validate_metadata(metadata_value, config.urn)
    score_columns = metadata["score_columns"]
    count_columns = metadata["count_columns"]
    num_variants = metadata["num_variants"]

    scores = _parse_csv_bytes(
        fetched["scores"].decoded_body,
        context="MaveDB scores",
        expected_header=(*FIXED_VARIANT_COLUMNS, *score_columns),
        expected_rows=num_variants,
        urn=config.urn,
    )
    counts = _parse_csv_bytes(
        fetched["counts"].decoded_body,
        context="MaveDB counts",
        expected_header=(*FIXED_VARIANT_COLUMNS, *count_columns),
        expected_rows=num_variants,
        urn=config.urn,
    )
    if counts.accessions != scores.accessions:
        raise SourceLockError("MaveDB score and count accession order differs")
    if (
        _finite_numeric_count(
            scores,
            "score",
            context="MaveDB scores.score",
        )
        == 0
    ):
        raise SourceLockError("MaveDB scores.score has no finite values")
    score_finite_value_counts = _finite_numeric_counts_by_column(
        scores,
        score_columns,
    )
    score_invalid_value_counts = _invalid_numeric_counts_by_column(
        scores,
        score_columns,
    )

    substantive_count_values = 0
    count_finite_value_counts: dict[str, int] = {}
    count_invalid_value_counts: dict[str, int] = {}
    for column in count_columns:
        finite_count = _finite_numeric_count(
            counts,
            column,
            context=f"MaveDB counts.{column}",
            nonnegative=True,
        )
        count_finite_value_counts[column] = finite_count
        count_invalid_value_counts[column] = 0
        substantive_count_values += finite_count
    if count_columns and substantive_count_values == 0:
        raise SourceLockError("MaveDB declares count columns but all count values are missing")
    counts_status = "substantive" if count_columns else "identifier_only"

    mapped_value = _parse_json_bytes(
        fetched["mapped_variants"].decoded_body,
        "MaveDB mapped-variants",
    )
    mapping_summary = _mapping_summary(
        mapped_value,
        urn=config.urn,
        expected_accessions=scores.accessions,
    )
    targets = _target_contracts(
        metadata["target_genes"],
        config=config,
        transport=resolved_transport,
    )
    readiness = _validate_readiness(
        config.readiness,
        config.readiness_evidence,
        score_columns=score_columns,
        count_columns=count_columns,
        scores=scores,
        counts=counts,
        counts_status=counts_status,
        mapping_summary=mapping_summary,
    )

    source_artifacts = {
        "metadata": fetched["metadata"].as_lock(),
        "scores": fetched["scores"].as_lock(),
        "counts": fetched["counts"].as_lock(),
        "mapped_variants": fetched["mapped_variants"].as_lock(),
    }
    lock: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_scope": CLAIM_SCOPE,
        "ingestion_status": "not_ingested",
        "outcome_status": "not_derived",
        "urn": config.urn,
        "readiness": readiness,
        "api": {
            "base_url": api_base,
            "openapi_url": config.openapi_url,
            "openapi_version": api_version,
            "openapi_artifact": openapi_artifact.as_lock(),
        },
        "source_artifacts": source_artifacts,
        "metadata_contract": {
            "title": metadata["title"],
            "num_variants": num_variants,
            "license": metadata["license"],
            "license_sha256": canonical_sha256(metadata["license"]),
            "data_usage_policy": metadata["data_usage_policy"],
            "private": metadata["private"],
            "processing_state": metadata["processing_state"],
            "superseded_score_set": metadata["superseded_score_set"],
            "superseding_score_set": metadata["superseding_score_set"],
            "dataset_columns": metadata["dataset_columns"],
            "dataset_columns_sha256": canonical_sha256(metadata["dataset_columns"]),
            "calibration_count": len(metadata["calibrations"]),
            "calibrations_sha256": canonical_sha256(metadata["calibrations"]),
            "targets": targets,
        },
        "tabular_contract": {
            "fixed_variant_columns": list(FIXED_VARIANT_COLUMNS),
            "score_header": list(scores.header),
            "count_header": list(counts.header),
            "score_row_count": len(scores.rows),
            "count_row_count": len(counts.rows),
            "accessions_sha256": canonical_sha256(list(scores.accessions)),
            "sorted_accessions_sha256": canonical_sha256(sorted(scores.accessions)),
            "counts_status": counts_status,
            "substantive_count_column_count": len(count_columns),
            "substantive_count_value_count": substantive_count_values,
            "score_finite_value_counts": score_finite_value_counts,
            "count_finite_value_counts": count_finite_value_counts,
            "score_invalid_value_counts": score_invalid_value_counts,
            "count_invalid_value_counts": count_invalid_value_counts,
            "native_hgvs_nt_duplicate_group_count": len(_duplicate_nonmissing_values(scores, "hgvs_nt")),
            "native_hgvs_pro_duplicate_group_count": len(_duplicate_nonmissing_values(scores, "hgvs_pro")),
        },
        "mapping_contract": mapping_summary,
        "metadata_tabular_license_binding_sha256": "",
        "source_bundle_sha256": "",
    }
    lock["metadata_tabular_license_binding_sha256"] = canonical_sha256(_license_binding_payload(lock))
    expected_binding = expected_hashes.get("metadata_tabular_license_binding")
    if expected_binding is not None and lock["metadata_tabular_license_binding_sha256"] != expected_binding:
        raise SourceLockError("MaveDB metadata-tabular license binding hash mismatch")
    lock["source_bundle_sha256"] = canonical_sha256(_source_bundle_payload(lock))
    validate_source_lock(lock)
    return lock


def _validate_persisted_artifact(
    value: Any,
    *,
    context: str,
    expected_content_type: str,
    expected_url: str,
) -> Mapping[str, Any]:
    artifact = _require_exact_mapping(
        value,
        FETCHED_ARTIFACT_FIELDS,
        context,
    )
    url = _require_https_url(artifact["url"], f"{context}.url")
    if url != expected_url:
        raise SourceLockError(f"{context}.url differs from its locked endpoint")
    if type(artifact["status"]) is not int or artifact["status"] != 200:
        raise SourceLockError(f"{context}.status must be HTTP 200")
    _require_positive_int(artifact["attempts"], f"{context}.attempts")
    if artifact["content_type"] != expected_content_type:
        raise SourceLockError(f"{context}.content_type must be {expected_content_type}")

    transfer_encoding = artifact["transfer_encoding"]
    if transfer_encoding not in {None, "chunked"}:
        raise SourceLockError(f"{context} has unsupported Transfer-Encoding")
    wire_byte_count = _require_positive_int(
        artifact["wire_byte_count"],
        f"{context}.wire_byte_count",
    )
    declared_content_length = artifact["declared_content_length"]
    if transfer_encoding == "chunked":
        if declared_content_length is not None:
            raise SourceLockError(f"{context} ambiguously records both HTTP framing headers")
    else:
        if type(declared_content_length) is not int or declared_content_length < 1:
            raise SourceLockError(f"{context} is missing supported HTTP response framing")
        if declared_content_length != wire_byte_count:
            raise SourceLockError(f"{context} Content-Length mismatch")

    content_encoding = artifact["content_encoding"]
    if content_encoding not in {None, "gzip"}:
        raise SourceLockError(f"{context} has unsupported Content-Encoding")
    decoded_byte_count = _require_positive_int(
        artifact["decoded_byte_count"],
        f"{context}.decoded_byte_count",
    )
    wire_sha256 = _require_sha256(
        artifact["wire_sha256"],
        f"{context}.wire_sha256",
    )
    decoded_sha256 = _require_sha256(
        artifact["sha256"],
        f"{context}.sha256",
    )
    if content_encoding is None and (wire_byte_count != decoded_byte_count or wire_sha256 != decoded_sha256):
        raise SourceLockError(f"{context} identity-encoded wire and decoded summaries differ")
    return artifact


def _validate_persisted_target(
    value: Any,
    *,
    context: str,
    api_base_url: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceLockError(f"{context} must be an object")
    target_kind = value.get("target_kind")
    expected_fields = TARGET_COMMON_FIELDS
    if target_kind == "accession":
        expected_fields = expected_fields | {"target_accession_metadata"}
    elif target_kind != "embedded_sequence":
        raise SourceLockError(f"{context}.target_kind is invalid")
    target = _require_exact_mapping(value, expected_fields, context)

    _require_nonempty_string(target["name"], f"{context}.name")
    _require_optional_nonempty_string(
        target["mapped_hgnc_name"],
        f"{context}.mapped_hgnc_name",
    )
    _require_sha256(
        target["external_identifiers_sha256"],
        f"{context}.external_identifiers_sha256",
    )
    sequence_length = _require_positive_int(
        target["sequence_length"],
        f"{context}.sequence_length",
    )
    sequence_sha256 = _require_sha256(
        target["sequence_sha256"],
        f"{context}.sequence_sha256",
    )

    if target_kind == "embedded_sequence":
        if target["accession"] is not None or target["reference_artifact"] is not None:
            raise SourceLockError(f"{context} embedded target cannot contain accession evidence")
        _require_nonempty_string(
            target["sequence_type"],
            f"{context}.sequence_type",
        )
        return target

    accession = _require_nonempty_string(
        target["accession"],
        f"{context}.accession",
    )
    if target["sequence_type"] != "accession_resolved":
        raise SourceLockError(f"{context}.sequence_type must be accession_resolved")
    accession_metadata = _require_exact_mapping(
        target["target_accession_metadata"],
        TARGET_ACCESSION_METADATA_FIELDS,
        f"{context}.target_accession_metadata",
    )
    if accession_metadata["accession"] != accession:
        raise SourceLockError(f"{context} accession metadata differs")
    for field_name in ("assembly", "gene", "recordType"):
        _require_nonempty_string(
            accession_metadata[field_name],
            f"{context}.target_accession_metadata.{field_name}",
        )
    if accession_metadata["recordType"] != "TargetAccession":
        raise SourceLockError(f"{context}.target_accession_metadata.recordType is invalid")
    if type(accession_metadata["isBaseEditor"]) is not bool:
        raise SourceLockError(f"{context}.target_accession_metadata.isBaseEditor must be boolean")
    reference_url = f"{api_base_url}/refget/sequence/{quote(accession, safe='._-')}"
    reference = _validate_persisted_artifact(
        target["reference_artifact"],
        context=f"{context}.reference_artifact",
        expected_content_type="text/plain",
        expected_url=reference_url,
    )
    if reference["sha256"] != sequence_sha256 or reference["decoded_byte_count"] != sequence_length:
        raise SourceLockError(f"{context} reference sequence summary differs")
    return target


def _validate_persisted_metadata_contract(
    value: Any,
    *,
    api_base_url: str,
) -> tuple[Mapping[str, Any], list[str], list[str], int]:
    metadata = _require_exact_mapping(
        value,
        METADATA_CONTRACT_FIELDS,
        "MaveDB metadata_contract",
    )
    _require_nonempty_string(
        metadata["title"],
        "metadata_contract.title",
    )
    num_variants = _require_positive_int(
        metadata["num_variants"],
        "metadata_contract.num_variants",
    )

    license_value = _require_exact_mapping(
        metadata["license"],
        LICENSE_FIELDS,
        "metadata_contract.license",
    )
    _require_positive_int(
        license_value["id"],
        "metadata_contract.license.id",
    )
    if license_value["active"] is not True:
        raise SourceLockError("MaveDB source lock must retain an active license")
    for field_name in (
        "shortName",
        "longName",
        "link",
        "version",
        "recordType",
    ):
        _require_nonempty_string(
            license_value[field_name],
            f"metadata_contract.license.{field_name}",
        )
    if license_value["recordType"] != "ShortLicense":
        raise SourceLockError("metadata_contract.license.recordType must be ShortLicense")
    _require_https_url(
        license_value["link"],
        "metadata_contract.license.link",
    )
    if canonical_sha256(license_value) != metadata["license_sha256"]:
        raise SourceLockError("MaveDB source lock license hash differs")

    _require_optional_nonempty_string(
        metadata["data_usage_policy"],
        "metadata_contract.data_usage_policy",
    )
    if metadata["private"] is not False:
        raise SourceLockError("MaveDB source lock must remain public/non-private")
    if metadata["processing_state"] != "success":
        raise SourceLockError("MaveDB source lock processing_state must remain success")
    if metadata["superseded_score_set"] is not None:
        raise SourceLockError("persisted superseded MaveDB score sets are rejected")
    if metadata["superseding_score_set"] is not None:
        raise SourceLockError("persisted MaveDB score sets with a superseding record are rejected")

    dataset_columns = _require_exact_mapping(
        metadata["dataset_columns"],
        DATASET_COLUMNS_FIELDS,
        "metadata_contract.dataset_columns",
    )
    if dataset_columns["recordType"] != "DatasetColumns":
        raise SourceLockError("metadata_contract.dataset_columns.recordType is invalid")
    score_columns = _require_string_list(
        dataset_columns["scoreColumns"],
        "metadata_contract.dataset_columns.scoreColumns",
    )
    count_columns = _require_string_list(
        dataset_columns["countColumns"],
        "metadata_contract.dataset_columns.countColumns",
    )
    if "score" not in score_columns:
        raise SourceLockError("metadata_contract scoreColumns must contain score")
    overlap = set(FIXED_VARIANT_COLUMNS) & (set(score_columns) | set(count_columns))
    if overlap:
        raise SourceLockError("metadata_contract custom columns collide with fixed columns")
    if canonical_sha256(dataset_columns) != metadata["dataset_columns_sha256"]:
        raise SourceLockError("MaveDB source lock dataset-columns hash differs")

    _require_nonnegative_int(
        metadata["calibration_count"],
        "metadata_contract.calibration_count",
    )
    _require_sha256(
        metadata["calibrations_sha256"],
        "metadata_contract.calibrations_sha256",
    )
    targets = metadata["targets"]
    if not isinstance(targets, list) or not targets:
        raise SourceLockError("metadata_contract.targets must be a nonempty list")
    target_names: list[str] = []
    for index, target in enumerate(targets):
        validated_target = _validate_persisted_target(
            target,
            context=f"metadata_contract.targets[{index}]",
            api_base_url=api_base_url,
        )
        target_names.append(str(validated_target["name"]))
    if len(set(target_names)) != len(target_names):
        raise SourceLockError("metadata_contract target names must be unique")
    return metadata, score_columns, count_columns, num_variants


def _validate_finite_value_counts(
    value: Any,
    *,
    columns: Sequence[str],
    row_count: int,
    context: str,
) -> dict[str, int]:
    summary = _require_exact_mapping(value, set(columns), context)
    result: dict[str, int] = {}
    for column in columns:
        count = _require_nonnegative_int(
            summary[column],
            f"{context}.{column}",
        )
        if count > row_count:
            raise SourceLockError(f"{context}.{column} exceeds the row count")
        result[column] = count
    return result


def _validate_persisted_tabular_contract(
    value: Any,
    *,
    score_columns: Sequence[str],
    count_columns: Sequence[str],
    num_variants: int,
) -> tuple[
    Mapping[str, Any],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    tabular = _require_exact_mapping(
        value,
        TABULAR_CONTRACT_FIELDS,
        "MaveDB tabular_contract",
    )
    fixed_columns = _require_string_list(
        tabular["fixed_variant_columns"],
        "tabular_contract.fixed_variant_columns",
    )
    if fixed_columns != list(FIXED_VARIANT_COLUMNS):
        raise SourceLockError("tabular_contract fixed columns differ")
    score_header = _require_string_list(
        tabular["score_header"],
        "tabular_contract.score_header",
    )
    count_header = _require_string_list(
        tabular["count_header"],
        "tabular_contract.count_header",
    )
    if score_header != [*FIXED_VARIANT_COLUMNS, *score_columns]:
        raise SourceLockError("tabular_contract score header differs")
    if count_header != [*FIXED_VARIANT_COLUMNS, *count_columns]:
        raise SourceLockError("tabular_contract count header differs")
    for field_name in ("score_row_count", "count_row_count"):
        if tabular[field_name] != num_variants:
            raise SourceLockError(f"tabular_contract.{field_name} differs from num_variants")
    _require_sha256(
        tabular["accessions_sha256"],
        "tabular_contract.accessions_sha256",
    )
    _require_sha256(
        tabular["sorted_accessions_sha256"],
        "tabular_contract.sorted_accessions_sha256",
    )

    expected_counts_status = "substantive" if count_columns else "identifier_only"
    if tabular["counts_status"] != expected_counts_status:
        raise SourceLockError("tabular_contract counts_status is inconsistent")
    if tabular["substantive_count_column_count"] != len(count_columns):
        raise SourceLockError("tabular_contract substantive count-column count differs")
    score_finite_counts = _validate_finite_value_counts(
        tabular["score_finite_value_counts"],
        columns=score_columns,
        row_count=num_variants,
        context="tabular_contract.score_finite_value_counts",
    )
    count_finite_counts = _validate_finite_value_counts(
        tabular["count_finite_value_counts"],
        columns=count_columns,
        row_count=num_variants,
        context="tabular_contract.count_finite_value_counts",
    )
    score_invalid_counts = _validate_finite_value_counts(
        tabular["score_invalid_value_counts"],
        columns=score_columns,
        row_count=num_variants,
        context="tabular_contract.score_invalid_value_counts",
    )
    count_invalid_counts = _validate_finite_value_counts(
        tabular["count_invalid_value_counts"],
        columns=count_columns,
        row_count=num_variants,
        context="tabular_contract.count_invalid_value_counts",
    )
    for column in score_columns:
        if score_finite_counts[column] + score_invalid_counts[column] > num_variants:
            raise SourceLockError(f"tabular_contract score summaries exceed rows for {column}")
    for column in count_columns:
        if count_finite_counts[column] + count_invalid_counts[column] > num_variants:
            raise SourceLockError(f"tabular_contract count summaries exceed rows for {column}")
        if count_invalid_counts[column] != 0:
            raise SourceLockError(f"tabular_contract count column {column!r} has invalid values")
    if score_finite_counts["score"] < 1:
        raise SourceLockError("tabular_contract score column has no finite values")
    if score_invalid_counts["score"] != 0:
        raise SourceLockError("tabular_contract score column has invalid values")
    substantive_count_value_count = _require_nonnegative_int(
        tabular["substantive_count_value_count"],
        "tabular_contract.substantive_count_value_count",
    )
    if substantive_count_value_count != sum(count_finite_counts.values()):
        raise SourceLockError("tabular_contract substantive count-value total differs")
    if count_columns and substantive_count_value_count < 1:
        raise SourceLockError("tabular_contract substantive counts have no finite values")
    for field_name in (
        "native_hgvs_nt_duplicate_group_count",
        "native_hgvs_pro_duplicate_group_count",
    ):
        duplicate_count = _require_nonnegative_int(
            tabular[field_name],
            f"tabular_contract.{field_name}",
        )
        if duplicate_count > num_variants:
            raise SourceLockError(f"tabular_contract.{field_name} exceeds the row count")
    return (
        tabular,
        score_finite_counts,
        count_finite_counts,
        score_invalid_counts,
        count_invalid_counts,
    )


def _validate_persisted_mapping_contract(
    value: Any,
    *,
    num_variants: int,
    sorted_accessions_sha256: str,
) -> Mapping[str, Any]:
    mapping = _require_exact_mapping(
        value,
        MAPPING_CONTRACT_FIELDS,
        "MaveDB mapping_contract",
    )
    integer_fields = (
        "history_record_count",
        "current_record_count",
        "current_unique_variant_count",
        "current_duplicate_variant_count",
        "current_missing_variant_count",
        "current_extra_variant_count",
        "current_error_count",
        "current_post_mapped_null_count",
        "current_pre_mapped_null_count",
        "current_at_mismatched_locus_count",
        "current_near_gap_count",
    )
    counts = {
        field_name: _require_nonnegative_int(
            mapping[field_name],
            f"mapping_contract.{field_name}",
        )
        for field_name in integer_fields
    }
    if counts["history_record_count"] < counts["current_record_count"]:
        raise SourceLockError("mapping_contract history count is below current count")
    if counts["current_record_count"] != num_variants or counts["current_unique_variant_count"] != num_variants:
        raise SourceLockError("mapping_contract current coverage differs from num_variants")
    for field_name in (
        "current_duplicate_variant_count",
        "current_missing_variant_count",
        "current_extra_variant_count",
    ):
        if counts[field_name] != 0:
            raise SourceLockError(f"mapping_contract.{field_name} must remain zero")
    for field_name in (
        "current_error_count",
        "current_post_mapped_null_count",
        "current_pre_mapped_null_count",
        "current_at_mismatched_locus_count",
        "current_near_gap_count",
    ):
        if counts[field_name] > num_variants:
            raise SourceLockError(f"mapping_contract.{field_name} exceeds current coverage")
    if counts["current_error_count"] < counts["current_post_mapped_null_count"]:
        raise SourceLockError("mapping_contract error count is below post-mapped null count")
    for field_name in (
        "alignment_levels",
        "mapping_api_versions",
        "vrs_versions",
    ):
        _require_sorted_string_list(
            mapping[field_name],
            f"mapping_contract.{field_name}",
        )
    current_accessions_sha256 = _require_sha256(
        mapping["current_accessions_sha256"],
        "mapping_contract.current_accessions_sha256",
    )
    if current_accessions_sha256 != sorted_accessions_sha256:
        raise SourceLockError("mapping_contract current accessions differ from tabular accessions")
    _require_sha256(
        mapping["current_mappings_sha256"],
        "mapping_contract.current_mappings_sha256",
    )
    return mapping


def _validate_persisted_readiness(
    value: Any,
    *,
    score_columns: Sequence[str],
    count_columns: Sequence[str],
    score_finite_counts: Mapping[str, int],
    count_finite_counts: Mapping[str, int],
    score_invalid_counts: Mapping[str, int],
    count_invalid_counts: Mapping[str, int],
    counts_status: str,
    mapping_contract: Mapping[str, Any],
) -> None:
    readiness = _require_exact_mapping(
        value,
        READINESS_FIELDS,
        "MaveDB readiness",
    )
    try:
        state = Readiness(readiness["state"])
    except (TypeError, ValueError) as exc:
        raise SourceLockError("MaveDB source lock readiness is invalid") from exc
    if state is Readiness.CONFIRMATORY_READY:
        raise SourceLockError("candidate-stage MaveDB source locks reject CONFIRMATORY_READY")
    if readiness["caller_configured"] is not True or readiness["automatic_promotion"] is not False:
        raise SourceLockError("MaveDB readiness must be caller-configured without promotion")
    evidence = _require_exact_mapping(
        readiness["evidence"],
        READINESS_EVIDENCE_FIELDS,
        "MaveDB readiness.evidence",
    )
    for field_name in (
        "identity_resolution_spec_sha256",
        "transformation_spec_sha256",
        "wt_control_spec_sha256",
        "confirmatory_preregistration_sha256",
        "independent_replication_spec_sha256",
    ):
        _optional_sha256(
            evidence[field_name],
            f"readiness.evidence.{field_name}",
        )
    for field_name in ("identity_block_reason", "rejection_reason"):
        _require_optional_nonempty_string(
            evidence[field_name],
            f"readiness.evidence.{field_name}",
        )
    if state is Readiness.REJECTED:
        _require_nonempty_string(
            evidence["rejection_reason"],
            "readiness.evidence.rejection_reason",
        )
    if state is Readiness.IDENTITY_BLOCKED:
        _require_nonempty_string(
            evidence["identity_block_reason"],
            "readiness.evidence.identity_block_reason",
        )

    aggregate_states = {
        Readiness.AGGREGATE_ONLY,
        Readiness.PROCESSED_REPLICATES,
        Readiness.COUNT_LINEAGE_PARTIAL,
        Readiness.COUNT_RECOMPUTABLE,
    }
    aggregate_column = evidence["aggregate_score_column"]
    if state in aggregate_states:
        aggregate_column = _require_nonempty_string(
            aggregate_column,
            "readiness.evidence.aggregate_score_column",
        )
    else:
        aggregate_column = _require_optional_nonempty_string(
            aggregate_column,
            "readiness.evidence.aggregate_score_column",
        )
    if aggregate_column is not None:
        if aggregate_column not in score_columns:
            raise SourceLockError("readiness aggregate score column is absent")
        if score_finite_counts[aggregate_column] < 1:
            raise SourceLockError("readiness aggregate score column has no finite values")
        if score_invalid_counts[aggregate_column] != 0:
            raise SourceLockError("readiness aggregate score column has invalid values")

    replicate_columns = _require_string_list(
        evidence["processed_replicate_columns"],
        "readiness.evidence.processed_replicate_columns",
    )
    for column in replicate_columns:
        if column not in score_columns or column == aggregate_column:
            raise SourceLockError(f"readiness replicate score column {column!r} is invalid")
        if score_finite_counts[column] < 1:
            raise SourceLockError(f"readiness replicate score column {column!r} has no values")
        if score_invalid_counts[column] != 0:
            raise SourceLockError(f"readiness replicate score column {column!r} has invalid values")
    if state is Readiness.PROCESSED_REPLICATES and not replicate_columns:
        raise SourceLockError("PROCESSED_REPLICATES requires processed replicate columns")

    declared_count_columns = _require_string_list(
        evidence["count_lineage_columns"],
        "readiness.evidence.count_lineage_columns",
    )
    for column in declared_count_columns:
        if column not in count_columns:
            raise SourceLockError(f"readiness count lineage column {column!r} is absent")
        if count_finite_counts[column] < 1:
            raise SourceLockError(f"readiness count lineage column {column!r} has no finite values")
        if count_invalid_counts[column] != 0:
            raise SourceLockError(f"readiness count lineage column {column!r} has invalid values")
    count_states = {
        Readiness.COUNT_LINEAGE_PARTIAL,
        Readiness.COUNT_RECOMPUTABLE,
    }
    if state in count_states:
        if counts_status != "substantive":
            raise SourceLockError(f"{state.value} requires substantive count columns")
        if set(declared_count_columns) != set(count_columns):
            raise SourceLockError(f"{state.value} must bind every MaveDB count lineage column")

    if state is Readiness.COUNT_RECOMPUTABLE:
        for field_name in (
            "identity_resolution_spec_sha256",
            "transformation_spec_sha256",
            "wt_control_spec_sha256",
        ):
            if evidence[field_name] is None:
                raise SourceLockError(f"COUNT_RECOMPUTABLE requires {field_name}")
        if mapping_contract["current_error_count"] != 0:
            raise SourceLockError("COUNT_RECOMPUTABLE requires error-free current mappings")


def validate_source_lock(lock: Mapping[str, Any]) -> None:
    if not isinstance(lock, Mapping):
        raise SourceLockError("MaveDB source lock must be an object")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "claim_scope",
        "ingestion_status",
        "outcome_status",
        "urn",
        "readiness",
        "api",
        "source_artifacts",
        "metadata_contract",
        "tabular_contract",
        "mapping_contract",
        "metadata_tabular_license_binding_sha256",
        "source_bundle_sha256",
    }
    if set(lock) != expected_fields:
        raise SourceLockError("MaveDB source lock must use the exact top-level schema")
    if lock["schema_version"] != SCHEMA_VERSION or lock["artifact_type"] != ARTIFACT_TYPE:
        raise SourceLockError("unsupported MaveDB source-lock schema")
    if (
        lock["claim_scope"] != CLAIM_SCOPE
        or lock["ingestion_status"] != "not_ingested"
        or lock["outcome_status"] != "not_derived"
    ):
        raise SourceLockError("MaveDB source lock cannot claim ingestion or outcomes")
    if URN_PATTERN.fullmatch(str(lock["urn"])) is None:
        raise SourceLockError("MaveDB source lock URN is invalid")

    api = _require_exact_mapping(
        lock["api"],
        API_FIELDS,
        "MaveDB api",
    )
    api_base_url = _require_https_url(api["base_url"], "api.base_url")
    if api_base_url.endswith("/"):
        raise SourceLockError("api.base_url must not end with a slash")
    openapi_url = _require_https_url(
        api["openapi_url"],
        "api.openapi_url",
    )
    _require_nonempty_string(
        api["openapi_version"],
        "api.openapi_version",
    )
    _validate_persisted_artifact(
        api["openapi_artifact"],
        context="api.openapi_artifact",
        expected_content_type="application/json",
        expected_url=openapi_url,
    )

    artifacts = _require_exact_mapping(
        lock["source_artifacts"],
        {
            "metadata",
            "scores",
            "counts",
            "mapped_variants",
        },
        "MaveDB source_artifacts",
    )
    encoded_urn = quote(str(lock["urn"]), safe=":")
    endpoint_urls = {
        "metadata": f"{api_base_url}/score-sets/{encoded_urn}",
        "scores": f"{api_base_url}/score-sets/{encoded_urn}/scores",
        "counts": f"{api_base_url}/score-sets/{encoded_urn}/counts",
        "mapped_variants": (f"{api_base_url}/score-sets/{encoded_urn}/mapped-variants"),
    }
    expected_content_types = {
        "metadata": "application/json",
        "scores": "text/csv",
        "counts": "text/csv",
        "mapped_variants": "application/json",
    }
    for name in ("metadata", "scores", "counts", "mapped_variants"):
        _validate_persisted_artifact(
            artifacts[name],
            context=f"source_artifacts.{name}",
            expected_content_type=expected_content_types[name],
            expected_url=endpoint_urls[name],
        )

    metadata_contract, score_columns, count_columns, num_variants = _validate_persisted_metadata_contract(
        lock["metadata_contract"],
        api_base_url=api_base_url,
    )
    (
        tabular_contract,
        score_finite_counts,
        count_finite_counts,
        score_invalid_counts,
        count_invalid_counts,
    ) = _validate_persisted_tabular_contract(
        lock["tabular_contract"],
        score_columns=score_columns,
        count_columns=count_columns,
        num_variants=num_variants,
    )
    mapping_contract = _validate_persisted_mapping_contract(
        lock["mapping_contract"],
        num_variants=num_variants,
        sorted_accessions_sha256=tabular_contract["sorted_accessions_sha256"],
    )
    _validate_persisted_readiness(
        lock["readiness"],
        score_columns=score_columns,
        count_columns=count_columns,
        score_finite_counts=score_finite_counts,
        count_finite_counts=count_finite_counts,
        score_invalid_counts=score_invalid_counts,
        count_invalid_counts=count_invalid_counts,
        counts_status=str(tabular_contract["counts_status"]),
        mapping_contract=mapping_contract,
    )

    observed_binding = canonical_sha256(_license_binding_payload(lock))
    declared_binding = _require_sha256(
        lock["metadata_tabular_license_binding_sha256"],
        "metadata_tabular_license_binding_sha256",
    )
    if observed_binding != declared_binding:
        raise SourceLockError("MaveDB metadata-tabular license binding differs")
    declared_bundle = _require_sha256(
        lock["source_bundle_sha256"],
        "source_bundle_sha256",
    )
    if canonical_sha256(_source_bundle_payload(lock)) != declared_bundle:
        raise SourceLockError("MaveDB source bundle hash differs")


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise SourceLockError("MaveDB source locks must use the .json format")
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
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_source_lock(path: str | Path, lock: Mapping[str, Any]) -> Path:
    validate_source_lock(lock)
    return _atomic_write_json(path, lock)


def load_source_lock(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise SourceLockError("MaveDB source locks must use the .json format")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceLockError("invalid MaveDB source-lock JSON") from exc
    validate_source_lock(value)
    return value

"""Fail-closed schema-v2 contract for raw, replicate-resolved DMS assays.

This module is deliberately independent of :mod:`eval.dms_contract`.  Schema v1
is an aggregate ProteinGym pilot; schema v2 starts from an authoritative raw-data
lock and derives every outcome from matched mutant and wild-type/control
replicates.

The contract is tamper-evident, not a remote-attestation system.  In particular,
an external registry must authenticate ``registration.receipt_sha256`` and the
source owner must authenticate the source-lock digests.  The local validator can
prove that all downstream artifacts remain bound to those trust anchors and that
the declared deterministic transformation was replayed exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RAW_DMS_SCHEMA_VERSION = 2
MINIMUM_CONFIRMATORY_TARGET_FAMILIES = 8
LABEL_1_SEMANTICS = "retained_or_neutral_function"
LABEL_0_SEMANTICS = "impaired_or_loss_of_function"
UNCERTAINTY_ESTIMATOR = "standard_error_across_complete_matched_sets"
UNCERTAINTY_UNAVAILABLE_REASON = "fewer_than_two_complete_matched_sets"

SOURCE_LOCK_ARTIFACT_TYPE = "dms_raw_authoritative_source_lock"
FAMILY_MAP_ARTIFACT_TYPE = "dms_raw_family_map"
INPUT_MANIFEST_ARTIFACT_TYPE = "dms_raw_label_free_input_manifest"
RAW_REPLICATE_MANIFEST_ARTIFACT_TYPE = "dms_raw_replicate_manifest"
TRANSFORMATION_SPECIFICATION_ARTIFACT_TYPE = "dms_raw_transformation_specification"
EXCLUSION_LEDGER_ARTIFACT_TYPE = "dms_raw_exclusion_ledger"
OUTCOME_MANIFEST_ARTIFACT_TYPE = "dms_raw_derived_outcome_manifest"

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DATE_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
TIMESTAMP_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "label",
        "target_label",
        "outcome",
        "effect",
        "oriented_effect",
        "scaled_effect",
        "raw_contrast",
        "raw_value",
        "score",
        "fitness",
        "function_score",
        "qc_status",
        "qc_reason_codes",
        "replicate",
        "replicates",
    }
)

SOURCE_METADATA_NAMES = frozenset(
    {
        "assay_definition",
        "license",
        "provenance",
        "qc_definition",
        "raw_replicates",
        "reference_sequence",
    }
)

SOURCE_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_id",
        "dataset_name",
        "dataset_version",
        "dataset_revision",
        "source_uri",
        "access_date",
        "license_id",
        "redistribution_status",
        "raw_archive_sha256",
        "record_schema_sha256",
        "metadata_status",
        "assays",
    }
)
METADATA_ENTRY_FIELDS = frozenset({"status", "evidence_sha256", "reason"})
ASSAY_LOCK_FIELDS = frozenset(
    {
        "assay_id",
        "protein_id",
        "construct_id",
        "reference_sequence",
        "reference_sequence_sha256",
        "raw_value_name",
        "raw_value_unit",
        "wild_type_definition",
        "control_definition",
        "source_assay_sha256",
        "source_metadata_sha256",
        "raw_records_sha256",
        "expected_input_item_count",
        "expected_raw_replicate_count",
    }
)

FAMILY_MAP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "family_map_id",
        "source_lock_sha256",
        "authority_name",
        "authority_version",
        "authority_uri",
        "mapping_file_sha256",
        "metadata_status",
        "records",
    }
)
FAMILY_MAP_RECORD_FIELDS = frozenset(
    {
        "protein_id",
        "reference_sequence_sha256",
        "family_id",
        "evidence_id",
    }
)

INPUT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_sha256",
        "family_map_sha256",
        "records",
    }
)
INPUT_RECORD_FIELDS = frozenset(
    {
        "item_id",
        "analysis_partition",
        "assay_id",
        "source_record_id",
        "protein_id",
        "construct_id",
        "reference_sequence_sha256",
        "variant_id",
        "mutation",
        "mutant_sequence_sha256",
        "representation_kind",
        "representation_sha256",
        "family_id",
        "split_group_id",
        "intervention_pair_id",
    }
)

RAW_REPLICATE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_sha256",
        "input_manifest_sha256",
        "mutant_records",
        "baseline_pools",
        "baseline_records",
        "baseline_links",
    }
)
MUTANT_REPLICATE_RECORD_FIELDS = frozenset(
    {
        "replicate_id",
        "source_row_id",
        "source_row_sha256",
        "item_id",
        "assay_id",
        "protein_id",
        "variant_id",
        "intervention_pair_id",
        "condition_role",
        "matched_set_id",
        "biological_replicate_id",
        "technical_replicate_id",
        "assay_batch_id",
        "raw_value",
        "raw_unit",
        "qc_status",
        "qc_reason_codes",
    }
)
BASELINE_POOL_FIELDS = frozenset(
    {
        "baseline_pool_id",
        "assay_id",
        "protein_id",
        "construct_id",
        "condition_role",
        "matched_set_id",
        "assay_batch_id",
        "reuse_policy",
        "maximum_item_links_per_observation",
    }
)
BASELINE_RECORD_FIELDS = frozenset(
    {
        "baseline_observation_id",
        "source_row_id",
        "source_row_sha256",
        "baseline_pool_id",
        "assay_id",
        "protein_id",
        "construct_id",
        "condition_role",
        "matched_set_id",
        "biological_replicate_id",
        "technical_replicate_id",
        "assay_batch_id",
        "raw_value",
        "raw_unit",
        "qc_status",
        "qc_reason_codes",
    }
)
BASELINE_LINK_FIELDS = frozenset(
    {
        "link_id",
        "item_id",
        "intervention_pair_id",
        "baseline_pool_id",
        "baseline_observation_id",
        "assay_id",
        "matched_set_id",
        "assay_batch_id",
    }
)

TRANSFORMATION_SPECIFICATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_sha256",
        "family_map_sha256",
        "input_manifest_sha256",
        "specification",
        "registration",
    }
)
TRANSFORMATION_CORE_FIELDS = frozenset(
    {
        "specification_id",
        "assay_transformations",
    }
)
ASSAY_TRANSFORMATION_FIELDS = frozenset(
    {
        "assay_id",
        "baseline_role",
        "value_transform",
        "within_role_aggregation",
        "contrast",
        "across_match_aggregation",
        "orientation",
        "scale",
        "threshold",
        "uncertainty",
        "minimum_pass_replicates_per_role",
        "minimum_complete_matched_sets",
    }
)
VALUE_TRANSFORM_FIELDS = frozenset({"operation", "pseudocount"})
CONTRAST_FIELDS = frozenset({"operation", "pseudocount"})
ORIENTATION_FIELDS = frozenset({"operation", "endpoint_semantics"})
SCALE_FIELDS = frozenset({"operation", "multiplier", "offset"})
THRESHOLD_FIELDS = frozenset(
    {
        "operation",
        "cutoff",
        "label_1_semantics",
        "label_0_semantics",
    }
)
UNCERTAINTY_FIELDS = frozenset(
    {
        "estimator",
        "minimum_complete_matched_sets",
        "unavailable_reason",
    }
)
REGISTRATION_FIELDS = frozenset(
    {
        "status",
        "locked_payload_sha256",
        "registered_at",
        "registry_id",
        "receipt_sha256",
    }
)

EXCLUSION_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_sha256",
        "family_map_sha256",
        "input_manifest_sha256",
        "raw_replicate_manifest_sha256",
        "transformation_specification_sha256",
        "records",
    }
)
EXCLUSION_RECORD_FIELDS = frozenset(
    {
        "item_id",
        "assay_id",
        "intervention_pair_id",
        "reason_codes",
        "evidence_mutant_replicate_ids",
        "evidence_baseline_observation_ids",
        "evidence_baseline_link_ids",
        "transformation_sha256",
    }
)

OUTCOME_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_lock_sha256",
        "family_map_sha256",
        "input_manifest_sha256",
        "raw_replicate_manifest_sha256",
        "transformation_specification_sha256",
        "exclusion_ledger_sha256",
        "records",
    }
)
OUTCOME_RECORD_FIELDS = frozenset(
    {
        "item_id",
        "analysis_partition",
        "assay_id",
        "protein_id",
        "variant_id",
        "family_id",
        "intervention_pair_id",
        "qualified_mutant_replicate_ids",
        "qualified_baseline_observation_ids",
        "baseline_pool_ids",
        "baseline_link_ids",
        "matched_set_ids",
        "mutant_replicate_count",
        "baseline_replicate_count",
        "mutant_biological_replicate_count",
        "baseline_biological_replicate_count",
        "mutant_aggregate",
        "baseline_aggregate",
        "raw_contrast",
        "oriented_effect",
        "scaled_effect",
        "effect_uncertainty",
        "uncertainty_status",
        "uncertainty_reason",
        "uncertainty_matched_set_count",
        "target_label",
        "target_label_semantics",
        "transformation_sha256",
        "raw_replicate_subset_sha256",
        "qc_status",
    }
)

RELEASE_FIELDS = frozenset(
    {
        "source_lock",
        "family_map",
        "input_manifest",
        "raw_replicate_manifest",
        "transformation_specification",
        "exclusion_ledger",
        "outcome_manifest",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON byte representation used by schema v2."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _exact_fields(
    value: Any,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    obj = _mapping(value, context)
    observed = set(obj)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{context} must use the exact schema; missing={missing}, extra={extra}")
    return obj


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _strict_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _sorted_unique_strings(value: Any, context: str) -> list[str]:
    items = _list(value, context)
    for index, item in enumerate(items):
        _string(item, f"{context}[{index}]")
    if items != sorted(items) or len(items) != len(set(items)):
        raise ValueError(f"{context} must contain sorted unique strings")
    return items


def _assert_no_label_leakage(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"{context} is label-free and cannot contain outcome key {key!r}")
            _assert_no_label_leakage(nested, context)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_label_leakage(nested, context)


def _parse_mutation(value: Any, context: str) -> tuple[str, int, str]:
    mutation = _string(value, context)
    match = MUTATION_PATTERN.fullmatch(mutation)
    if match is None:
        raise ValueError(f"{context} must be a canonical single-amino-acid substitution")
    reference, position, alternate = (
        match.group(1),
        int(match.group(2)),
        match.group(3),
    )
    if reference == alternate:
        raise ValueError(f"{context} cannot be synonymous")
    return reference, position, alternate


def _validate_metadata_entry(value: Any, context: str) -> str:
    entry = _exact_fields(value, METADATA_ENTRY_FIELDS, context)
    status = entry["status"]
    if status not in {"verified", "missing"}:
        raise ValueError(f"{context}.status must be verified or missing")
    if status == "verified":
        _sha256(entry["evidence_sha256"], f"{context}.evidence_sha256")
        if entry["reason"] is not None:
            raise ValueError(f"{context}.reason must be null when verified")
    else:
        if entry["evidence_sha256"] is not None:
            raise ValueError(f"{context}.evidence_sha256 must be null when missing")
        _string(entry["reason"], f"{context}.reason")
    return status


def _require_binding(
    observed: Any,
    expected_artifact: Mapping[str, Any],
    context: str,
) -> None:
    if _sha256(observed, context) != canonical_sha256(expected_artifact):
        raise ValueError(f"{context} is bound to different canonical bytes")


def raw_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the raw row content covered by ``source_row_sha256``."""

    payload = dict(record)
    payload.pop("source_row_sha256", None)
    return payload


def raw_record_sha256(record: Mapping[str, Any]) -> str:
    return canonical_sha256(raw_record_payload(record))


def assay_raw_records_sha256(
    raw_components: Mapping[str, Sequence[Mapping[str, Any]]],
    assay_id: str,
) -> str:
    """Hash one assay's physical observations, baseline pools, and exact links."""

    keys_and_ids = (
        ("mutant_records", "replicate_id"),
        ("baseline_pools", "baseline_pool_id"),
        ("baseline_records", "baseline_observation_id"),
        ("baseline_links", "link_id"),
    )
    payload: dict[str, list[dict[str, Any]]] = {}
    for key, identity_field in keys_and_ids:
        values = raw_components.get(key, ())
        selected = [dict(record) for record in values if record.get("assay_id") == assay_id]
        selected.sort(key=lambda record: str(record.get(identity_field, "")))
        payload[key] = selected
    return canonical_sha256(payload)


def validate_source_lock(artifact: Mapping[str, Any]) -> None:
    artifact = _exact_fields(
        artifact,
        SOURCE_LOCK_FIELDS,
        "raw DMS authoritative source lock",
    )
    if artifact["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS source-lock schema version")
    if artifact["artifact_type"] != SOURCE_LOCK_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS source-lock artifact type")

    for field in (
        "source_lock_id",
        "dataset_name",
        "dataset_version",
        "source_uri",
        "redistribution_status",
    ):
        _string(artifact[field], f"source lock.{field}")
    _optional_string(artifact["dataset_revision"], "source lock.dataset_revision")
    access_date = _string(artifact["access_date"], "source lock.access_date")
    if DATE_PATTERN.fullmatch(access_date) is None:
        raise ValueError("source lock.access_date must use YYYY-MM-DD")
    _sha256(artifact["raw_archive_sha256"], "source lock.raw_archive_sha256")
    _sha256(artifact["record_schema_sha256"], "source lock.record_schema_sha256")

    metadata = _exact_fields(
        artifact["metadata_status"],
        SOURCE_METADATA_NAMES,
        "source lock.metadata_status",
    )
    states = {
        name: _validate_metadata_entry(
            metadata[name],
            f"source lock.metadata_status.{name}",
        )
        for name in sorted(SOURCE_METADATA_NAMES)
    }
    license_id = _optional_string(artifact["license_id"], "source lock.license_id")
    if states["license"] == "verified":
        if license_id is None or artifact["redistribution_status"] == "unresolved":
            raise ValueError("verified license metadata requires license_id and resolved redistribution")
    elif license_id is not None or artifact["redistribution_status"] != "unresolved":
        raise ValueError("missing license metadata requires null license_id and unresolved redistribution")

    assays = _list(artifact["assays"], "source lock.assays")
    if not assays:
        raise ValueError("source lock requires at least one assay")
    assay_ids: list[str] = []
    construct_ids: set[str] = set()
    for index, raw_assay in enumerate(assays):
        context = f"source lock.assays[{index}]"
        assay = _exact_fields(raw_assay, ASSAY_LOCK_FIELDS, context)
        assay_id = _string(assay["assay_id"], f"{context}.assay_id")
        assay_ids.append(assay_id)
        for field in (
            "protein_id",
            "construct_id",
            "raw_value_name",
            "raw_value_unit",
            "wild_type_definition",
        ):
            _string(assay[field], f"{context}.{field}")
        if assay["construct_id"] in construct_ids:
            raise ValueError("source lock construct_id values must be unique")
        construct_ids.add(assay["construct_id"])
        _optional_string(assay["control_definition"], f"{context}.control_definition")
        sequence = _string(assay["reference_sequence"], f"{context}.reference_sequence")
        if any(residue not in CANONICAL_AMINO_ACIDS for residue in sequence):
            raise ValueError(f"{context}.reference_sequence is not canonical protein")
        if sequence_sha256(sequence) != _sha256(
            assay["reference_sequence_sha256"],
            f"{context}.reference_sequence_sha256",
        ):
            raise ValueError(f"{context}.reference_sequence checksum differs")
        for field in (
            "source_assay_sha256",
            "source_metadata_sha256",
            "raw_records_sha256",
        ):
            _sha256(assay[field], f"{context}.{field}")
        _strict_int(
            assay["expected_input_item_count"],
            f"{context}.expected_input_item_count",
            minimum=1,
        )
        _strict_int(
            assay["expected_raw_replicate_count"],
            f"{context}.expected_raw_replicate_count",
            minimum=1,
        )
    if assay_ids != sorted(assay_ids) or len(assay_ids) != len(set(assay_ids)):
        raise ValueError("source lock assays must be sorted by unique assay_id")


def source_lock_sha256(artifact: Mapping[str, Any]) -> str:
    validate_source_lock(artifact)
    return canonical_sha256(artifact)


def validate_family_map(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
) -> None:
    validate_source_lock(source_lock)
    artifact = _exact_fields(artifact, FAMILY_MAP_FIELDS, "raw DMS family map")
    if artifact["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS family-map schema version")
    if artifact["artifact_type"] != FAMILY_MAP_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS family-map artifact type")
    _require_binding(
        artifact["source_lock_sha256"],
        source_lock,
        "family map.source_lock_sha256",
    )
    for field in (
        "family_map_id",
        "authority_name",
        "authority_version",
        "authority_uri",
    ):
        _string(artifact[field], f"family map.{field}")
    _sha256(artifact["mapping_file_sha256"], "family map.mapping_file_sha256")
    _validate_metadata_entry(artifact["metadata_status"], "family map.metadata_status")

    expected_keys = {(assay["protein_id"], assay["reference_sequence_sha256"]) for assay in source_lock["assays"]}
    records = _list(artifact["records"], "family map.records")
    observed_keys: list[tuple[str, str]] = []
    protein_to_family: dict[str, str] = {}
    reference_to_family: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        context = f"family map.records[{index}]"
        record = _exact_fields(raw_record, FAMILY_MAP_RECORD_FIELDS, context)
        protein_id = _string(record["protein_id"], f"{context}.protein_id")
        reference_sha = _sha256(
            record["reference_sequence_sha256"],
            f"{context}.reference_sequence_sha256",
        )
        family_id = _string(record["family_id"], f"{context}.family_id")
        _string(record["evidence_id"], f"{context}.evidence_id")
        previous = protein_to_family.setdefault(protein_id, family_id)
        if previous != family_id:
            raise ValueError("one protein_id cannot map to multiple families")
        previous_reference_family = reference_to_family.setdefault(
            reference_sha,
            family_id,
        )
        if previous_reference_family != family_id:
            raise ValueError("one exact reference_sequence_sha256 cannot map to multiple families")
        observed_keys.append((protein_id, reference_sha))
    if observed_keys != sorted(observed_keys) or len(observed_keys) != len(set(observed_keys)):
        raise ValueError("family-map records must be sorted and identity-unique")
    if set(observed_keys) != expected_keys:
        raise ValueError("family map must cover exactly the source-lock protein identities")


def family_map_sha256(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
) -> str:
    validate_family_map(artifact, source_lock)
    return canonical_sha256(artifact)


def validate_input_manifest(
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
) -> None:
    validate_family_map(family_map, source_lock)
    _assert_no_label_leakage(manifest, "raw DMS input manifest")
    manifest = _exact_fields(
        manifest,
        INPUT_MANIFEST_FIELDS,
        "raw DMS label-free input manifest",
    )
    if manifest["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS input-manifest schema version")
    if manifest["artifact_type"] != INPUT_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS input-manifest artifact type")
    _require_binding(
        manifest["source_lock_sha256"],
        source_lock,
        "input manifest.source_lock_sha256",
    )
    _require_binding(
        manifest["family_map_sha256"],
        family_map,
        "input manifest.family_map_sha256",
    )

    assays = {assay["assay_id"]: assay for assay in source_lock["assays"]}
    family_lookup = {
        (record["protein_id"], record["reference_sequence_sha256"]): record["family_id"]
        for record in family_map["records"]
    }
    records = _list(manifest["records"], "input manifest.records")
    if not records:
        raise ValueError("raw DMS input manifest cannot be empty")

    item_ids: list[str] = []
    source_keys: set[tuple[str, str]] = set()
    biological_identities: set[tuple[str, str, str]] = set()
    pair_nesting: dict[str, tuple[str, str, str]] = {}
    assay_partition: dict[str, str] = {}
    family_partitions: dict[str, set[str]] = defaultdict(set)
    protein_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    reference_checksum_partitions: dict[str, set[str]] = defaultdict(set)
    mutant_checksum_partitions: dict[str, set[str]] = defaultdict(set)
    assay_counts: dict[str, int] = defaultdict(int)
    for index, raw_record in enumerate(records):
        context = f"input manifest.records[{index}]"
        record = _exact_fields(raw_record, INPUT_RECORD_FIELDS, context)
        item_id = _string(record["item_id"], f"{context}.item_id")
        partition = record["analysis_partition"]
        if partition not in {"source", "target"}:
            raise ValueError(f"{context}.analysis_partition must be source or target")
        assay_id = _string(record["assay_id"], f"{context}.assay_id")
        if assay_id not in assays:
            raise ValueError(f"{context}.assay_id is not source-locked")
        assay = assays[assay_id]
        source_record_id = _string(
            record["source_record_id"],
            f"{context}.source_record_id",
        )
        for field in ("protein_id", "construct_id", "reference_sequence_sha256"):
            if record[field] != assay[field]:
                raise ValueError(f"{context}.{field} differs from source lock")
        mutation = _string(record["mutation"], f"{context}.mutation")
        if record["variant_id"] != mutation:
            raise ValueError(f"{context}.variant_id must equal canonical mutation")
        reference, position, alternate = _parse_mutation(
            mutation,
            f"{context}.mutation",
        )
        sequence = assay["reference_sequence"]
        if position > len(sequence) or sequence[position - 1] != reference:
            raise ValueError(f"{context}.mutation does not match reference sequence")
        mutant_sequence = sequence[: position - 1] + alternate + sequence[position:]
        mutant_sha = sequence_sha256(mutant_sequence)
        if (
            _sha256(
                record["mutant_sequence_sha256"],
                f"{context}.mutant_sequence_sha256",
            )
            != mutant_sha
        ):
            raise ValueError(f"{context}.mutant sequence checksum differs")
        if record["representation_kind"] != "full_mutant_protein_sequence":
            raise ValueError(f"{context}.representation_kind is invalid")
        if (
            _sha256(
                record["representation_sha256"],
                f"{context}.representation_sha256",
            )
            != mutant_sha
        ):
            raise ValueError(f"{context}.representation checksum differs")

        family_id = family_lookup[(assay["protein_id"], assay["reference_sequence_sha256"])]
        if record["family_id"] != family_id:
            raise ValueError(f"{context}.family_id differs from locked family map")
        if record["split_group_id"] != family_id:
            raise ValueError(f"{context}.split_group_id must equal locked family_id")

        expected_source_record_id = f"{assay_id}:{mutation}"
        expected_item_id = f"{partition}:{assay_id}:{mutation}"
        expected_pair_id = f"{assay_id}:mutant-vs-wt:{mutation}"
        if source_record_id != expected_source_record_id:
            raise ValueError(f"{context}.source_record_id is not identity-bound")
        if item_id != expected_item_id:
            raise ValueError(f"{context}.item_id is not identity-bound")
        if record["intervention_pair_id"] != expected_pair_id:
            raise ValueError(f"{context}.intervention_pair_id is not identity-bound")

        source_key = (assay_id, source_record_id)
        biological_identity = (
            assay["protein_id"],
            assay["reference_sequence_sha256"],
            mutation,
        )
        if source_key in source_keys:
            raise ValueError("input manifest contains duplicate source identities")
        if biological_identity in biological_identities:
            raise ValueError("input manifest contains duplicate biological identities")
        source_keys.add(source_key)
        biological_identities.add(biological_identity)

        nesting = (item_id, family_id, partition)
        previous_nesting = pair_nesting.setdefault(
            record["intervention_pair_id"],
            nesting,
        )
        if previous_nesting != nesting:
            raise ValueError("each intervention_pair_id must nest within one item, family, and partition")
        previous_partition = assay_partition.setdefault(assay_id, partition)
        if previous_partition != partition:
            raise ValueError("one assay cannot cross analysis partitions")
        family_partitions[family_id].add(partition)
        protein_partitions[(assay["protein_id"], assay["reference_sequence_sha256"])].add(partition)
        reference_checksum_partitions[assay["reference_sequence_sha256"]].add(partition)
        mutant_checksum_partitions[mutant_sha].add(partition)
        assay_counts[assay_id] += 1
        item_ids.append(item_id)

    if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("input records must be sorted by unique item_id")
    for assay_id, assay in assays.items():
        if assay_counts[assay_id] != assay["expected_input_item_count"]:
            raise ValueError(f"input item count differs from source lock for assay {assay_id}")
    overlapping_reference_checksums = sorted(
        checksum for checksum, partitions in reference_checksum_partitions.items() if len(partitions) > 1
    )
    if overlapping_reference_checksums:
        raise ValueError(
            "source and target partitions have exact reference-sequence checksum "
            f"overlap: {overlapping_reference_checksums}"
        )
    overlapping_mutant_checksums = sorted(
        checksum for checksum, partitions in mutant_checksum_partitions.items() if len(partitions) > 1
    )
    if overlapping_mutant_checksums:
        raise ValueError(
            f"source and target partitions have exact mutant-sequence checksum overlap: {overlapping_mutant_checksums}"
        )
    overlapping_proteins = sorted(protein for protein, partitions in protein_partitions.items() if len(partitions) > 1)
    if overlapping_proteins:
        raise ValueError(f"source and target partitions have protein/WT identity overlap: {overlapping_proteins}")
    overlapping_families = sorted(family for family, partitions in family_partitions.items() if len(partitions) > 1)
    if overlapping_families:
        raise ValueError(f"source and target partitions have family overlap: {overlapping_families}")


def input_manifest_sha256(
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
) -> str:
    validate_input_manifest(manifest, source_lock, family_map)
    return canonical_sha256(manifest)


def _validate_raw_measurement(
    record: Mapping[str, Any],
    context: str,
    assay: Mapping[str, Any],
) -> tuple[str, str, str]:
    source_row_id = _string(record["source_row_id"], f"{context}.source_row_id")
    biological_id = _string(
        record["biological_replicate_id"],
        f"{context}.biological_replicate_id",
    )
    technical_id = _string(
        record["technical_replicate_id"],
        f"{context}.technical_replicate_id",
    )
    _finite_float(record["raw_value"], f"{context}.raw_value")
    if record["raw_unit"] != assay["raw_value_unit"]:
        raise ValueError(f"{context}.raw_unit differs from source lock")
    qc_status = record["qc_status"]
    if qc_status not in {"pass", "fail"}:
        raise ValueError(f"{context}.qc_status must be pass or fail")
    reasons = _sorted_unique_strings(
        record["qc_reason_codes"],
        f"{context}.qc_reason_codes",
    )
    if qc_status == "pass" and reasons:
        raise ValueError(f"{context} passing QC cannot have reason codes")
    if qc_status == "fail" and not reasons:
        raise ValueError(f"{context} failed QC requires reason codes")
    if _sha256(
        record["source_row_sha256"],
        f"{context}.source_row_sha256",
    ) != raw_record_sha256(record):
        raise ValueError(f"{context}.source-row digest differs")
    return source_row_id, biological_id, technical_id


def validate_raw_replicate_manifest(
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    family_map: Mapping[str, Any],
) -> None:
    validate_input_manifest(input_manifest, source_lock, family_map)
    manifest = _exact_fields(
        manifest,
        RAW_REPLICATE_MANIFEST_FIELDS,
        "raw DMS replicate manifest",
    )
    if manifest["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS replicate-manifest schema version")
    if manifest["artifact_type"] != RAW_REPLICATE_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS replicate-manifest artifact type")
    _require_binding(
        manifest["source_lock_sha256"],
        source_lock,
        "raw replicate manifest.source_lock_sha256",
    )
    _require_binding(
        manifest["input_manifest_sha256"],
        input_manifest,
        "raw replicate manifest.input_manifest_sha256",
    )

    assays = {assay["assay_id"]: assay for assay in source_lock["assays"]}
    inputs = {record["item_id"]: record for record in input_manifest["records"]}
    mutant_records = _list(
        manifest["mutant_records"],
        "raw replicate manifest.mutant_records",
    )
    baseline_pools = _list(
        manifest["baseline_pools"],
        "raw replicate manifest.baseline_pools",
    )
    baseline_records = _list(
        manifest["baseline_records"],
        "raw replicate manifest.baseline_records",
    )
    baseline_links = _list(
        manifest["baseline_links"],
        "raw replicate manifest.baseline_links",
    )
    replicate_ids: list[str] = []
    source_row_ids: set[str] = set()
    replicate_identities: set[tuple[str, ...]] = set()
    assay_counts: dict[str, int] = defaultdict(int)
    item_mutant_match_batches: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for index, raw_record in enumerate(mutant_records):
        context = f"raw replicate manifest.mutant_records[{index}]"
        record = _exact_fields(
            raw_record,
            MUTANT_REPLICATE_RECORD_FIELDS,
            context,
        )
        replicate_id = _string(record["replicate_id"], f"{context}.replicate_id")
        item_id = _string(record["item_id"], f"{context}.item_id")
        if item_id not in inputs:
            raise ValueError(f"{context}.item_id is not in label-free inputs")
        item = inputs[item_id]
        assay = assays[item["assay_id"]]
        for field in (
            "assay_id",
            "protein_id",
            "variant_id",
            "intervention_pair_id",
        ):
            if record[field] != item[field]:
                raise ValueError(f"{context}.{field} is a mismatched mutant identity")
        if record["condition_role"] != "mutant":
            raise ValueError(f"{context}.condition_role must be mutant")
        matched_set_id = _string(
            record["matched_set_id"],
            f"{context}.matched_set_id",
        )
        batch_id = _string(
            record["assay_batch_id"],
            f"{context}.assay_batch_id",
        )
        source_row_id, biological_id, technical_id = _validate_raw_measurement(
            record,
            context,
            assay,
        )

        expected_replicate_id = f"{item_id}:{matched_set_id}:mutant:{biological_id}:{technical_id}"
        if replicate_id != expected_replicate_id:
            raise ValueError(f"{context}.replicate_id is not identity-bound")
        identity = (
            item_id,
            matched_set_id,
            "mutant",
            biological_id,
            technical_id,
            batch_id,
        )
        if identity in replicate_identities:
            raise ValueError("raw replicate manifest contains duplicate identities")
        if source_row_id in source_row_ids:
            raise ValueError("raw replicate manifest contains duplicate source rows")
        replicate_identities.add(identity)
        source_row_ids.add(source_row_id)
        item_mutant_match_batches[item_id].add((matched_set_id, batch_id))
        assay_counts[assay["assay_id"]] += 1
        replicate_ids.append(replicate_id)

    if replicate_ids != sorted(replicate_ids) or len(replicate_ids) != len(set(replicate_ids)):
        raise ValueError("mutant records must be sorted by unique replicate_id")

    pool_ids: list[str] = []
    pools: dict[str, Mapping[str, Any]] = {}
    for index, raw_pool in enumerate(baseline_pools):
        context = f"raw replicate manifest.baseline_pools[{index}]"
        pool = _exact_fields(raw_pool, BASELINE_POOL_FIELDS, context)
        pool_id = _string(pool["baseline_pool_id"], f"{context}.baseline_pool_id")
        assay_id = _string(pool["assay_id"], f"{context}.assay_id")
        if assay_id not in assays:
            raise ValueError(f"{context}.assay_id is not source-locked")
        assay = assays[assay_id]
        for field in ("protein_id", "construct_id"):
            if pool[field] != assay[field]:
                raise ValueError(f"{context}.{field} differs from source lock")
        role = pool["condition_role"]
        if role not in {"wild_type", "control"}:
            raise ValueError(f"{context}.condition_role must be wild_type or control")
        if role == "control" and assay["control_definition"] is None:
            raise ValueError(f"{context} uses an undefined assay control")
        _string(pool["matched_set_id"], f"{context}.matched_set_id")
        _string(pool["assay_batch_id"], f"{context}.assay_batch_id")
        maximum_links = _strict_int(
            pool["maximum_item_links_per_observation"],
            f"{context}.maximum_item_links_per_observation",
            minimum=1,
        )
        reuse_policy = pool["reuse_policy"]
        if reuse_policy == "item_specific":
            if maximum_links != 1:
                raise ValueError(f"{context} item_specific reuse requires maximum links of 1")
        elif reuse_policy == "shared_preregistered":
            if maximum_links < 2:
                raise ValueError(f"{context} shared reuse requires maximum links of at least 2")
        else:
            raise ValueError(f"{context}.reuse_policy must be item_specific or shared_preregistered")
        if pool_id in pools:
            raise ValueError("baseline pools contain duplicate frozen pool IDs")
        pools[pool_id] = pool
        pool_ids.append(pool_id)
    if pool_ids != sorted(pool_ids):
        raise ValueError("baseline pools must be sorted by unique baseline_pool_id")

    observation_ids: list[str] = []
    observations: dict[str, Mapping[str, Any]] = {}
    physical_baseline_identities: set[tuple[str, ...]] = set()
    pool_observation_counts: dict[str, int] = defaultdict(int)
    for index, raw_record in enumerate(baseline_records):
        context = f"raw replicate manifest.baseline_records[{index}]"
        record = _exact_fields(raw_record, BASELINE_RECORD_FIELDS, context)
        observation_id = _string(
            record["baseline_observation_id"],
            f"{context}.baseline_observation_id",
        )
        pool_id = _string(
            record["baseline_pool_id"],
            f"{context}.baseline_pool_id",
        )
        if pool_id not in pools:
            raise ValueError(f"{context}.baseline_pool_id is not frozen")
        pool = pools[pool_id]
        assay = assays[pool["assay_id"]]
        for field in (
            "assay_id",
            "protein_id",
            "construct_id",
            "condition_role",
            "matched_set_id",
            "assay_batch_id",
        ):
            if record[field] != pool[field]:
                raise ValueError(f"{context}.{field} differs from frozen baseline pool")
        source_row_id, biological_id, technical_id = _validate_raw_measurement(
            record,
            context,
            assay,
        )
        expected_observation_id = f"{pool_id}:{biological_id}:{technical_id}"
        if observation_id != expected_observation_id:
            raise ValueError(f"{context}.baseline_observation_id is not identity-bound")
        identity = (
            record["assay_id"],
            record["condition_role"],
            record["matched_set_id"],
            biological_id,
            technical_id,
            record["assay_batch_id"],
        )
        if identity in physical_baseline_identities:
            raise ValueError("baseline records contain cloned physical observation identities")
        if source_row_id in source_row_ids:
            raise ValueError("physical mutant/baseline source rows must be globally unique")
        if observation_id in observations:
            raise ValueError("baseline records contain duplicate observation IDs")
        physical_baseline_identities.add(identity)
        source_row_ids.add(source_row_id)
        observations[observation_id] = record
        observation_ids.append(observation_id)
        pool_observation_counts[pool_id] += 1
        assay_counts[record["assay_id"]] += 1
    if observation_ids != sorted(observation_ids):
        raise ValueError("baseline records must be sorted by unique baseline_observation_id")

    link_ids: list[str] = []
    link_identities: set[tuple[str, str]] = set()
    observation_link_counts: dict[str, int] = defaultdict(int)
    pool_link_items: dict[str, set[str]] = defaultdict(set)
    pool_link_counts: dict[str, int] = defaultdict(int)
    for index, raw_link in enumerate(baseline_links):
        context = f"raw replicate manifest.baseline_links[{index}]"
        link = _exact_fields(raw_link, BASELINE_LINK_FIELDS, context)
        link_id = _string(link["link_id"], f"{context}.link_id")
        item_id = _string(link["item_id"], f"{context}.item_id")
        if item_id not in inputs:
            raise ValueError(f"{context}.item_id is not in label-free inputs")
        item = inputs[item_id]
        pool_id = _string(
            link["baseline_pool_id"],
            f"{context}.baseline_pool_id",
        )
        if pool_id not in pools:
            raise ValueError(f"{context}.baseline_pool_id is not frozen")
        pool = pools[pool_id]
        observation_id = _string(
            link["baseline_observation_id"],
            f"{context}.baseline_observation_id",
        )
        if observation_id not in observations:
            raise ValueError(f"{context}.baseline_observation_id is not a physical observation")
        observation = observations[observation_id]
        if observation["baseline_pool_id"] != pool_id:
            raise ValueError(f"{context} links an observation to the wrong baseline pool")
        if link["intervention_pair_id"] != item["intervention_pair_id"]:
            raise ValueError(f"{context}.intervention_pair_id is not the exact item pair")
        for field in ("assay_id", "matched_set_id", "assay_batch_id"):
            if link[field] != pool[field] or link[field] != observation[field]:
                raise ValueError(f"{context}.{field} is incompatible across pool/observation")
        if link["assay_id"] != item["assay_id"]:
            raise ValueError(f"{context} crosses assay boundaries")
        match_batch = (link["matched_set_id"], link["assay_batch_id"])
        if match_batch not in item_mutant_match_batches[item_id]:
            raise ValueError(f"{context} has no batch/match-compatible mutant observation")
        expected_link_id = f"{item_id}:baseline-link:{observation_id}"
        if link_id != expected_link_id:
            raise ValueError(f"{context}.link_id is not identity-bound")
        link_identity = (item_id, observation_id)
        if link_identity in link_identities:
            raise ValueError("baseline links contain duplicate item-observation links")
        link_identities.add(link_identity)
        observation_link_counts[observation_id] += 1
        pool_link_items[pool_id].add(item_id)
        pool_link_counts[pool_id] += 1
        link_ids.append(link_id)
    if link_ids != sorted(link_ids) or len(link_ids) != len(set(link_ids)):
        raise ValueError("baseline links must be sorted by unique link_id")

    for pool_id, pool in pools.items():
        if pool_observation_counts[pool_id] == 0:
            raise ValueError(f"baseline pool {pool_id} has no physical observations")
        if pool_link_counts[pool_id] == 0:
            raise ValueError(f"baseline pool {pool_id} is unlinked")
        if pool["reuse_policy"] == "item_specific" and len(pool_link_items[pool_id]) != 1:
            raise ValueError(f"item-specific baseline pool {pool_id} must link to exactly one item")
    for observation_id, observation in observations.items():
        pool = pools[observation["baseline_pool_id"]]
        link_count = observation_link_counts[observation_id]
        if link_count == 0:
            raise ValueError(f"baseline observation {observation_id} is unlinked")
        if link_count > pool["maximum_item_links_per_observation"]:
            raise ValueError(f"baseline observation {observation_id} exceeds frozen reuse multiplicity")

    for assay_id, assay in assays.items():
        if assay_counts[assay_id] != assay["expected_raw_replicate_count"]:
            raise ValueError(f"raw replicate count differs from source lock for assay {assay_id}")
        if assay_raw_records_sha256(manifest, assay_id) != assay["raw_records_sha256"]:
            raise ValueError(f"raw replicate digest differs from source lock for assay {assay_id}")


def raw_replicate_manifest_sha256(
    manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    family_map: Mapping[str, Any],
) -> str:
    validate_raw_replicate_manifest(
        manifest,
        source_lock,
        input_manifest,
        family_map,
    )
    return canonical_sha256(manifest)


def transformation_locked_payload(
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact transformation content covered by registration."""

    return {
        "schema_version": specification["schema_version"],
        "artifact_type": specification["artifact_type"],
        "source_lock_sha256": specification["source_lock_sha256"],
        "family_map_sha256": specification["family_map_sha256"],
        "input_manifest_sha256": specification["input_manifest_sha256"],
        "specification": specification["specification"],
    }


def transformation_locked_payload_sha256(
    specification: Mapping[str, Any],
) -> str:
    return canonical_sha256(transformation_locked_payload(specification))


def _validate_assay_transformation(
    raw_transform: Any,
    context: str,
    *,
    assay: Mapping[str, Any],
) -> None:
    transform = _exact_fields(raw_transform, ASSAY_TRANSFORMATION_FIELDS, context)
    if transform["assay_id"] != assay["assay_id"]:
        raise ValueError(f"{context}.assay_id differs from source-lock ordering")
    baseline_role = transform["baseline_role"]
    if baseline_role not in {"wild_type", "control"}:
        raise ValueError(f"{context}.baseline_role must be wild_type or control")
    if baseline_role == "control" and assay["control_definition"] is None:
        raise ValueError(f"{context} selects an undefined control")

    value_transform = _exact_fields(
        transform["value_transform"],
        VALUE_TRANSFORM_FIELDS,
        f"{context}.value_transform",
    )
    operation = value_transform["operation"]
    if operation not in {"identity", "log2", "natural_log"}:
        raise ValueError(f"{context}.value_transform.operation is not declarative")
    pseudocount = _finite_float(
        value_transform["pseudocount"],
        f"{context}.value_transform.pseudocount",
    )
    if pseudocount < 0:
        raise ValueError(f"{context}.value_transform.pseudocount cannot be negative")
    if operation == "identity" and pseudocount != 0:
        raise ValueError(f"{context}.identity transform requires zero pseudocount")

    for field in ("within_role_aggregation", "across_match_aggregation"):
        if transform[field] != "mean":
            raise ValueError(
                f"{context}.{field} must be arithmetic mean so the frozen "
                "standard-error estimator matches the across-match estimand"
            )

    contrast = _exact_fields(
        transform["contrast"],
        CONTRAST_FIELDS,
        f"{context}.contrast",
    )
    if contrast["operation"] not in {"difference", "ratio", "log2_ratio"}:
        raise ValueError(f"{context}.contrast.operation is not declarative")
    contrast_pseudocount = _finite_float(
        contrast["pseudocount"],
        f"{context}.contrast.pseudocount",
    )
    if contrast_pseudocount < 0:
        raise ValueError(f"{context}.contrast.pseudocount cannot be negative")
    if contrast["operation"] == "difference" and contrast_pseudocount != 0:
        raise ValueError(f"{context}.difference contrast requires zero pseudocount")

    orientation = _exact_fields(
        transform["orientation"],
        ORIENTATION_FIELDS,
        f"{context}.orientation",
    )
    if orientation["operation"] not in {"identity", "negate"}:
        raise ValueError(f"{context}.orientation.operation must be identity or negate")
    if orientation["endpoint_semantics"] != "higher_is_more_functional":
        raise ValueError(f"{context}.orientation endpoint must be higher_is_more_functional")

    scale = _exact_fields(
        transform["scale"],
        SCALE_FIELDS,
        f"{context}.scale",
    )
    if scale["operation"] not in {"identity", "affine"}:
        raise ValueError(f"{context}.scale.operation must be identity or affine")
    multiplier = _finite_float(
        scale["multiplier"],
        f"{context}.scale.multiplier",
    )
    offset = _finite_float(scale["offset"], f"{context}.scale.offset")
    if scale["operation"] == "identity" and (multiplier != 1 or offset != 0):
        raise ValueError(f"{context}.identity scale must use multiplier=1, offset=0")
    if multiplier <= 0:
        raise ValueError(f"{context}.scale multiplier must be positive; sign belongs to orientation")

    threshold = _exact_fields(
        transform["threshold"],
        THRESHOLD_FIELDS,
        f"{context}.threshold",
    )
    if threshold["operation"] not in {"greater_equal", "greater_than"}:
        raise ValueError(f"{context}.threshold must respect higher-is-more-functional orientation")
    _finite_float(threshold["cutoff"], f"{context}.threshold.cutoff")
    for field in ("label_1_semantics", "label_0_semantics"):
        _string(threshold[field], f"{context}.threshold.{field}")
    if threshold["label_1_semantics"] != LABEL_1_SEMANTICS:
        raise ValueError(f"{context}.threshold label 1 must mean {LABEL_1_SEMANTICS}")
    if threshold["label_0_semantics"] != LABEL_0_SEMANTICS:
        raise ValueError(f"{context}.threshold label 0 must mean {LABEL_0_SEMANTICS}")

    uncertainty = _exact_fields(
        transform["uncertainty"],
        UNCERTAINTY_FIELDS,
        f"{context}.uncertainty",
    )
    expected_uncertainty = {
        "estimator": UNCERTAINTY_ESTIMATOR,
        "minimum_complete_matched_sets": 2,
        "unavailable_reason": UNCERTAINTY_UNAVAILABLE_REASON,
    }
    if dict(uncertainty) != expected_uncertainty:
        raise ValueError(f"{context}.uncertainty must use the exact schema-v2 estimator")

    _strict_int(
        transform["minimum_pass_replicates_per_role"],
        f"{context}.minimum_pass_replicates_per_role",
        minimum=1,
    )
    _strict_int(
        transform["minimum_complete_matched_sets"],
        f"{context}.minimum_complete_matched_sets",
        minimum=1,
    )


def validate_transformation_specification(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> None:
    validate_input_manifest(input_manifest, source_lock, family_map)
    artifact = _exact_fields(
        artifact,
        TRANSFORMATION_SPECIFICATION_FIELDS,
        "raw DMS transformation specification",
    )
    if artifact["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS transformation schema version")
    if artifact["artifact_type"] != TRANSFORMATION_SPECIFICATION_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS transformation artifact type")
    _require_binding(
        artifact["source_lock_sha256"],
        source_lock,
        "transformation.source_lock_sha256",
    )
    _require_binding(
        artifact["family_map_sha256"],
        family_map,
        "transformation.family_map_sha256",
    )
    _require_binding(
        artifact["input_manifest_sha256"],
        input_manifest,
        "transformation.input_manifest_sha256",
    )

    core = _exact_fields(
        artifact["specification"],
        TRANSFORMATION_CORE_FIELDS,
        "transformation.specification",
    )
    _string(core["specification_id"], "transformation.specification_id")
    transforms = _list(
        core["assay_transformations"],
        "transformation.assay_transformations",
    )
    assays = source_lock["assays"]
    if len(transforms) != len(assays):
        raise ValueError("transformation must cover every source-locked assay exactly")
    for index, (transform, assay) in enumerate(zip(transforms, assays, strict=True)):
        _validate_assay_transformation(
            transform,
            f"transformation.assay_transformations[{index}]",
            assay=assay,
        )

    registration = _exact_fields(
        artifact["registration"],
        REGISTRATION_FIELDS,
        "transformation.registration",
    )
    if _sha256(
        registration["locked_payload_sha256"],
        "transformation.registration.locked_payload_sha256",
    ) != transformation_locked_payload_sha256(artifact):
        raise ValueError(
            "transformation registration does not match locked payload; orientation/threshold may have changed post hoc"
        )
    status = registration["status"]
    if status == "registered_external":
        registered_at = _string(
            registration["registered_at"],
            "transformation.registration.registered_at",
        )
        if TIMESTAMP_PATTERN.fullmatch(registered_at) is None:
            raise ValueError("transformation.registration.registered_at must be UTC second precision")
        _string(registration["registry_id"], "transformation.registration.registry_id")
        _sha256(
            registration["receipt_sha256"],
            "transformation.registration.receipt_sha256",
        )
    elif status == "unregistered_candidate":
        for field in ("registered_at", "registry_id", "receipt_sha256"):
            if registration[field] is not None:
                raise ValueError(f"unregistered transformation requires null {field}")
    else:
        raise ValueError("transformation.registration.status must be registered_external or unregistered_candidate")


def transformation_specification_sha256(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> str:
    validate_transformation_specification(
        artifact,
        source_lock,
        family_map,
        input_manifest,
    )
    return canonical_sha256(artifact)


class _NonFiniteDerivationError(ValueError):
    """Raised before a non-finite derived value can enter an artifact."""


def _finite_derived(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _NonFiniteDerivationError(f"{context} produced a non-finite derived value") from exc
    if not math.isfinite(result):
        raise _NonFiniteDerivationError(f"{context} produced a non-finite derived value")
    return result


def _aggregate(values: Sequence[float], operation: str) -> float:
    if operation != "mean":
        raise AssertionError(f"validated aggregation operation {operation!r} is unknown")
    try:
        result = statistics.fmean(values)
    except OverflowError as exc:
        raise _NonFiniteDerivationError("arithmetic mean produced a non-finite derived value") from exc
    return _finite_derived(result, "arithmetic mean")


def _transform_value(value: float, specification: Mapping[str, Any]) -> float:
    operation = specification["operation"]
    pseudocount = float(specification["pseudocount"])
    if operation == "identity":
        return _finite_derived(value, "identity value transform")
    adjusted = _finite_derived(
        value + pseudocount,
        "value-transform pseudocount addition",
    )
    if adjusted <= 0:
        raise ValueError("value transform domain error")
    if operation == "log2":
        return _finite_derived(math.log2(adjusted), "log2 value transform")
    if operation == "natural_log":
        return _finite_derived(math.log(adjusted), "natural-log value transform")
    raise AssertionError(f"validated value transform {operation!r} is unknown")


def _contrast(
    mutant: float,
    baseline: float,
    specification: Mapping[str, Any],
) -> float:
    operation = specification["operation"]
    pseudocount = float(specification["pseudocount"])
    if operation == "difference":
        return _finite_derived(mutant - baseline, "difference contrast")
    numerator = _finite_derived(
        mutant + pseudocount,
        "contrast numerator",
    )
    denominator = _finite_derived(
        baseline + pseudocount,
        "contrast denominator",
    )
    if denominator == 0:
        raise ValueError("contrast denominator is zero")
    ratio = _finite_derived(numerator / denominator, "ratio contrast")
    if operation == "ratio":
        return ratio
    if operation == "log2_ratio":
        if ratio <= 0:
            raise ValueError("log2-ratio contrast domain error")
        return _finite_derived(math.log2(ratio), "log2-ratio contrast")
    raise AssertionError(f"validated contrast operation {operation!r} is unknown")


def _orient_and_scale(
    effect: float,
    transformation: Mapping[str, Any],
    *,
    context: str,
) -> tuple[float, float]:
    oriented = effect if transformation["orientation"]["operation"] == "identity" else -effect
    oriented = _finite_derived(oriented, f"{context} oriented effect")
    scale = transformation["scale"]
    if scale["operation"] == "identity":
        return oriented, oriented
    scaled_product = _finite_derived(
        oriented * float(scale["multiplier"]),
        f"{context} affine scaled product",
    )
    scaled = _finite_derived(
        scaled_product + float(scale["offset"]),
        f"{context} affine scaled effect",
    )
    return oriented, scaled


def _derive_records(
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_raw_replicate_manifest(
        raw_replicate_manifest,
        source_lock,
        input_manifest,
        family_map,
    )
    validate_transformation_specification(
        transformation_specification,
        source_lock,
        family_map,
        input_manifest,
    )
    transforms = {
        transform["assay_id"]: transform
        for transform in transformation_specification["specification"]["assay_transformations"]
    }
    mutants_by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in raw_replicate_manifest["mutant_records"]:
        mutants_by_item[record["item_id"]].append(record)
    pools = {pool["baseline_pool_id"]: pool for pool in raw_replicate_manifest["baseline_pools"]}
    observations = {record["baseline_observation_id"]: record for record in raw_replicate_manifest["baseline_records"]}
    links_by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for link in raw_replicate_manifest["baseline_links"]:
        links_by_item[link["item_id"]].append(link)

    outcomes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for item in input_manifest["records"]:
        mutant_records = sorted(
            mutants_by_item.get(item["item_id"], []),
            key=lambda record: record["replicate_id"],
        )
        baseline_links = sorted(
            links_by_item.get(item["item_id"], []),
            key=lambda link: link["link_id"],
        )
        transform = transforms[item["assay_id"]]
        baseline_role = transform["baseline_role"]
        selected_links = [
            link
            for link in baseline_links
            if observations[link["baseline_observation_id"]]["condition_role"] == baseline_role
        ]
        baseline_records = sorted(
            [observations[link["baseline_observation_id"]] for link in selected_links],
            key=lambda record: record["baseline_observation_id"],
        )
        links_by_observation = {link["baseline_observation_id"]: link for link in selected_links}
        minimum_replicates = transform["minimum_pass_replicates_per_role"]
        minimum_matches = transform["minimum_complete_matched_sets"]
        transform_sha = canonical_sha256(transform)
        reasons: set[str] = set()
        if not mutant_records and not baseline_links:
            reasons.add("no_raw_replicates")

        if not mutant_records:
            reasons.add("missing_mutant_replicates")
        if not baseline_records:
            reasons.add("missing_matched_baseline_replicates")

        passing_mutants = [record for record in mutant_records if record["qc_status"] == "pass"]
        passing_baselines = [record for record in baseline_records if record["qc_status"] == "pass"]
        if len({record["biological_replicate_id"] for record in passing_mutants}) < (minimum_replicates):
            reasons.add("insufficient_mutant_qc_pass")
        if len({record["biological_replicate_id"] for record in passing_baselines}) < minimum_replicates:
            reasons.add("insufficient_baseline_qc_pass")

        mutants_by_match: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        baselines_by_match: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in mutant_records:
            mutants_by_match[record["matched_set_id"]].append(record)
        for record in baseline_records:
            baselines_by_match[record["matched_set_id"]].append(record)
        matched_set_ids = sorted(set(mutants_by_match) | set(baselines_by_match))
        complete_matches: list[
            tuple[
                str,
                list[Mapping[str, Any]],
                list[Mapping[str, Any]],
                list[Mapping[str, Any]],
            ]
        ] = []
        for matched_set_id in matched_set_ids:
            mutants = [record for record in mutants_by_match[matched_set_id] if record["qc_status"] == "pass"]
            baselines = [record for record in baselines_by_match[matched_set_id] if record["qc_status"] == "pass"]
            mutant_bio = {record["biological_replicate_id"] for record in mutants}
            baseline_bio = {record["biological_replicate_id"] for record in baselines}
            if len(mutant_bio) >= minimum_replicates and len(baseline_bio) >= minimum_replicates:
                links = [links_by_observation[record["baseline_observation_id"]] for record in baselines]
                complete_matches.append((matched_set_id, mutants, baselines, links))
        if len(complete_matches) < minimum_matches:
            reasons.add("insufficient_complete_matched_sets")

        match_mutant_aggregates: list[float] = []
        match_baseline_aggregates: list[float] = []
        match_contrasts: list[float] = []
        if not reasons:
            try:
                for _, mutants, baselines, _ in complete_matches:
                    mutant_values = [
                        _transform_value(
                            float(record["raw_value"]),
                            transform["value_transform"],
                        )
                        for record in mutants
                    ]
                    baseline_values = [
                        _transform_value(
                            float(record["raw_value"]),
                            transform["value_transform"],
                        )
                        for record in baselines
                    ]
                    mutant_aggregate = _aggregate(
                        mutant_values,
                        transform["within_role_aggregation"],
                    )
                    baseline_aggregate = _aggregate(
                        baseline_values,
                        transform["within_role_aggregation"],
                    )
                    match_mutant_aggregates.append(mutant_aggregate)
                    match_baseline_aggregates.append(baseline_aggregate)
                    match_contrasts.append(
                        _contrast(
                            mutant_aggregate,
                            baseline_aggregate,
                            transform["contrast"],
                        )
                    )
            except _NonFiniteDerivationError:
                raise
            except OverflowError as exc:
                raise _NonFiniteDerivationError("transformation produced a non-finite derived value") from exc
            except ValueError:
                reasons.add("transformation_domain_error")

        if reasons:
            exclusions.append(
                {
                    "item_id": item["item_id"],
                    "assay_id": item["assay_id"],
                    "intervention_pair_id": item["intervention_pair_id"],
                    "reason_codes": sorted(reasons),
                    "evidence_mutant_replicate_ids": [record["replicate_id"] for record in mutant_records],
                    "evidence_baseline_observation_ids": [
                        record["baseline_observation_id"] for record in baseline_records
                    ],
                    "evidence_baseline_link_ids": [link["link_id"] for link in selected_links],
                    "transformation_sha256": transform_sha,
                }
            )
            continue

        mutant_aggregate = _aggregate(
            match_mutant_aggregates,
            transform["across_match_aggregation"],
        )
        baseline_aggregate = _aggregate(
            match_baseline_aggregates,
            transform["across_match_aggregation"],
        )
        raw_contrast = _aggregate(
            match_contrasts,
            transform["across_match_aggregation"],
        )
        oriented_effect, scaled_effect = _orient_and_scale(
            raw_contrast,
            transform,
            context="final",
        )
        per_match_scaled_effects: list[float] = []
        for index, match_contrast in enumerate(match_contrasts):
            _, match_scaled = _orient_and_scale(
                match_contrast,
                transform,
                context=f"matched set {index}",
            )
            per_match_scaled_effects.append(match_scaled)
        uncertainty_matched_set_count = len(per_match_scaled_effects)
        if uncertainty_matched_set_count >= 2:
            try:
                uncertainty_numerator = statistics.stdev(per_match_scaled_effects)
            except OverflowError as exc:
                raise _NonFiniteDerivationError("uncertainty estimator produced a non-finite derived value") from exc
            effect_uncertainty = _finite_derived(
                uncertainty_numerator / math.sqrt(uncertainty_matched_set_count),
                "standard error across complete matched sets",
            )
            uncertainty_status = "available"
            uncertainty_reason = None
        else:
            effect_uncertainty = None
            uncertainty_status = "unavailable"
            uncertainty_reason = UNCERTAINTY_UNAVAILABLE_REASON
        threshold = transform["threshold"]
        if threshold["operation"] == "greater_equal":
            target_label = int(scaled_effect >= float(threshold["cutoff"]))
        else:
            target_label = int(scaled_effect > float(threshold["cutoff"]))
        label_semantics = threshold["label_1_semantics" if target_label == 1 else "label_0_semantics"]

        qualified_mutants = sorted(
            [record for _, mutants, _, _ in complete_matches for record in mutants],
            key=lambda record: record["replicate_id"],
        )
        qualified_baselines = sorted(
            [record for _, _, baselines, _ in complete_matches for record in baselines],
            key=lambda record: record["baseline_observation_id"],
        )
        qualified_links = sorted(
            [link for _, _, _, links in complete_matches for link in links],
            key=lambda link: link["link_id"],
        )
        qualified_pool_ids = sorted({record["baseline_pool_id"] for record in qualified_baselines})
        qualified_pools = [pools[pool_id] for pool_id in qualified_pool_ids]
        raw_reference_payload = {
            "mutant_records": qualified_mutants,
            "baseline_pools": qualified_pools,
            "baseline_records": qualified_baselines,
            "baseline_links": qualified_links,
        }
        outcomes.append(
            {
                "item_id": item["item_id"],
                "analysis_partition": item["analysis_partition"],
                "assay_id": item["assay_id"],
                "protein_id": item["protein_id"],
                "variant_id": item["variant_id"],
                "family_id": item["family_id"],
                "intervention_pair_id": item["intervention_pair_id"],
                "qualified_mutant_replicate_ids": [record["replicate_id"] for record in qualified_mutants],
                "qualified_baseline_observation_ids": [
                    record["baseline_observation_id"] for record in qualified_baselines
                ],
                "baseline_pool_ids": qualified_pool_ids,
                "baseline_link_ids": [link["link_id"] for link in qualified_links],
                "matched_set_ids": [matched_set_id for matched_set_id, _, _, _ in complete_matches],
                "mutant_replicate_count": len(qualified_mutants),
                "baseline_replicate_count": len(qualified_baselines),
                "mutant_biological_replicate_count": len(
                    {record["biological_replicate_id"] for record in qualified_mutants}
                ),
                "baseline_biological_replicate_count": len(
                    {record["biological_replicate_id"] for record in qualified_baselines}
                ),
                "mutant_aggregate": float(mutant_aggregate),
                "baseline_aggregate": float(baseline_aggregate),
                "raw_contrast": float(raw_contrast),
                "oriented_effect": float(oriented_effect),
                "scaled_effect": float(scaled_effect),
                "effect_uncertainty": effect_uncertainty,
                "uncertainty_status": uncertainty_status,
                "uncertainty_reason": uncertainty_reason,
                "uncertainty_matched_set_count": uncertainty_matched_set_count,
                "target_label": target_label,
                "target_label_semantics": label_semantics,
                "transformation_sha256": transform_sha,
                "raw_replicate_subset_sha256": canonical_sha256(raw_reference_payload),
                "qc_status": "pass",
            }
        )
    return outcomes, exclusions


def build_exclusion_ledger(
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
) -> dict[str, Any]:
    _, exclusions = _derive_records(
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
    )
    return {
        "schema_version": RAW_DMS_SCHEMA_VERSION,
        "artifact_type": EXCLUSION_LEDGER_ARTIFACT_TYPE,
        "source_lock_sha256": canonical_sha256(source_lock),
        "family_map_sha256": canonical_sha256(family_map),
        "input_manifest_sha256": canonical_sha256(input_manifest),
        "raw_replicate_manifest_sha256": canonical_sha256(raw_replicate_manifest),
        "transformation_specification_sha256": canonical_sha256(transformation_specification),
        "records": exclusions,
    }


def validate_exclusion_ledger(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
) -> None:
    artifact = _exact_fields(
        artifact,
        EXCLUSION_LEDGER_FIELDS,
        "raw DMS exclusion ledger",
    )
    if artifact["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS exclusion-ledger schema version")
    if artifact["artifact_type"] != EXCLUSION_LEDGER_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS exclusion-ledger artifact type")
    bindings = (
        ("source_lock_sha256", source_lock),
        ("family_map_sha256", family_map),
        ("input_manifest_sha256", input_manifest),
        ("raw_replicate_manifest_sha256", raw_replicate_manifest),
        (
            "transformation_specification_sha256",
            transformation_specification,
        ),
    )
    for field, expected in bindings:
        _require_binding(artifact[field], expected, f"exclusion ledger.{field}")
    expected = build_exclusion_ledger(
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
    )
    records = _list(artifact["records"], "exclusion ledger.records")
    for index, record in enumerate(records):
        _exact_fields(
            record,
            EXCLUSION_RECORD_FIELDS,
            f"exclusion ledger.records[{index}]",
        )
    if canonical_json_bytes(artifact) != canonical_json_bytes(expected):
        raise ValueError("exclusion ledger differs from deterministic replicate/QC recomputation")


def exclusion_ledger_sha256(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
) -> str:
    validate_exclusion_ledger(
        artifact,
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
    )
    return canonical_sha256(artifact)


def build_outcome_manifest(
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
    exclusion_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    validate_exclusion_ledger(
        exclusion_ledger,
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
    )
    outcomes, _ = _derive_records(
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
    )
    return {
        "schema_version": RAW_DMS_SCHEMA_VERSION,
        "artifact_type": OUTCOME_MANIFEST_ARTIFACT_TYPE,
        "source_lock_sha256": canonical_sha256(source_lock),
        "family_map_sha256": canonical_sha256(family_map),
        "input_manifest_sha256": canonical_sha256(input_manifest),
        "raw_replicate_manifest_sha256": canonical_sha256(raw_replicate_manifest),
        "transformation_specification_sha256": canonical_sha256(transformation_specification),
        "exclusion_ledger_sha256": canonical_sha256(exclusion_ledger),
        "records": outcomes,
    }


def validate_outcome_manifest(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
    exclusion_ledger: Mapping[str, Any],
) -> None:
    artifact = _exact_fields(
        artifact,
        OUTCOME_MANIFEST_FIELDS,
        "raw DMS derived outcome manifest",
    )
    if artifact["schema_version"] != RAW_DMS_SCHEMA_VERSION:
        raise ValueError("unsupported raw DMS outcome schema version")
    if artifact["artifact_type"] != OUTCOME_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid raw DMS outcome artifact type")
    bindings = (
        ("source_lock_sha256", source_lock),
        ("family_map_sha256", family_map),
        ("input_manifest_sha256", input_manifest),
        ("raw_replicate_manifest_sha256", raw_replicate_manifest),
        (
            "transformation_specification_sha256",
            transformation_specification,
        ),
        ("exclusion_ledger_sha256", exclusion_ledger),
    )
    for field, expected in bindings:
        _require_binding(artifact[field], expected, f"outcome manifest.{field}")

    expected = build_outcome_manifest(
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
        exclusion_ledger,
    )
    records = _list(artifact["records"], "outcome manifest.records")
    expected_by_item = {record["item_id"]: record for record in expected["records"]}
    observed_ids: list[str] = []
    for index, raw_record in enumerate(records):
        context = f"outcome manifest.records[{index}]"
        record = _exact_fields(raw_record, OUTCOME_RECORD_FIELDS, context)
        item_id = _string(record["item_id"], f"{context}.item_id")
        observed_ids.append(item_id)
        if item_id in expected_by_item:
            expected_record = expected_by_item[item_id]
            if record["target_label"] != expected_record["target_label"]:
                raise ValueError(f"{context}.target_label differs from raw-replicate recomputation")
            if record["target_label_semantics"] != expected_record["target_label_semantics"]:
                raise ValueError(f"{context}.target_label_semantics differs from common class semantics")
            for field in ("mutant_aggregate", "baseline_aggregate"):
                if record[field] != expected_record[field]:
                    raise ValueError(f"{context}.{field} differs from raw-replicate replay")
            if record["oriented_effect"] != expected_record["oriented_effect"]:
                raise ValueError(f"{context}.oriented_effect differs from locked orientation")
            for field in (
                "effect_uncertainty",
                "uncertainty_status",
                "uncertainty_reason",
                "uncertainty_matched_set_count",
            ):
                if record[field] != expected_record[field]:
                    raise ValueError(f"{context}.{field} differs from locked uncertainty replay")
    if observed_ids != sorted(observed_ids) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("outcome records must be sorted by unique item_id")
    if canonical_json_bytes(artifact) != canonical_json_bytes(expected):
        raise ValueError("outcome manifest differs from deterministic raw-replicate recomputation")


def outcome_manifest_sha256(
    artifact: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    family_map: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    raw_replicate_manifest: Mapping[str, Any],
    transformation_specification: Mapping[str, Any],
    exclusion_ledger: Mapping[str, Any],
) -> str:
    validate_outcome_manifest(
        artifact,
        source_lock,
        family_map,
        input_manifest,
        raw_replicate_manifest,
        transformation_specification,
        exclusion_ledger,
    )
    return canonical_sha256(artifact)


def validate_release(release: Mapping[str, Any]) -> None:
    release = _exact_fields(release, RELEASE_FIELDS, "raw DMS schema-v2 release")
    source_lock = release["source_lock"]
    family_map = release["family_map"]
    input_manifest = release["input_manifest"]
    raw_replicates = release["raw_replicate_manifest"]
    transformation = release["transformation_specification"]
    exclusions = release["exclusion_ledger"]
    outcomes = release["outcome_manifest"]

    validate_source_lock(source_lock)
    validate_family_map(family_map, source_lock)
    validate_input_manifest(input_manifest, source_lock, family_map)
    validate_raw_replicate_manifest(
        raw_replicates,
        source_lock,
        input_manifest,
        family_map,
    )
    validate_transformation_specification(
        transformation,
        source_lock,
        family_map,
        input_manifest,
    )
    validate_exclusion_ledger(
        exclusions,
        source_lock,
        family_map,
        input_manifest,
        raw_replicates,
        transformation,
    )
    validate_outcome_manifest(
        outcomes,
        source_lock,
        family_map,
        input_manifest,
        raw_replicates,
        transformation,
        exclusions,
    )

    input_ids = {record["item_id"] for record in input_manifest["records"]}
    outcome_ids = {record["item_id"] for record in outcomes["records"]}
    exclusion_ids = {record["item_id"] for record in exclusions["records"]}
    if outcome_ids & exclusion_ids or outcome_ids | exclusion_ids != input_ids:
        raise ValueError("outcomes and exclusions must form an exact disjoint partition of inputs")


def assess_confirmatory_readiness(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess local structure while remaining confirmatory-ineligible.

    Digests and ``verified`` strings inside the release are self-asserted.  They
    can establish deterministic local consistency, but cannot authenticate their
    own source, family authority, license, or registration receipt.  This API
    therefore never returns confirmatory eligibility.  A future, separate API
    must verify these digests against an externally authenticated trust index
    before a confirmatory execution can become eligible.
    """

    validate_release(release)
    source_lock = release["source_lock"]
    family_map = release["family_map"]
    transformation = release["transformation_specification"]
    inputs = release["input_manifest"]["records"]
    outcomes = release["outcome_manifest"]["records"]
    local_missing: list[str] = []

    for name, entry in sorted(source_lock["metadata_status"].items()):
        if entry["status"] != "verified":
            local_missing.append(f"source_metadata:{name}")
    if family_map["metadata_status"]["status"] != "verified":
        local_missing.append("family_map_metadata")
    if transformation["registration"]["status"] != "registered_external":
        local_missing.append("locally_declared_transformation_registration")

    partitions = {record["analysis_partition"] for record in inputs}
    if "source" not in partitions:
        local_missing.append("source_partition")
    if "target" not in partitions:
        local_missing.append("target_partition")

    source_outcomes = [record for record in outcomes if record["analysis_partition"] == "source"]
    source_outcome_families = {record["family_id"] for record in source_outcomes}
    source_labels = {record["target_label"] for record in source_outcomes}
    if len(source_outcomes) < 2:
        local_missing.append(f"minimum_source_outcome_items:{len(source_outcomes)}/2")
    if len(source_outcome_families) < 2:
        local_missing.append(f"minimum_source_outcome_families:{len(source_outcome_families)}/2")
    if source_labels != {0, 1}:
        local_missing.append("source_binary_label_support:" + ",".join(str(label) for label in sorted(source_labels)))
    target_input_families = {record["family_id"] for record in inputs if record["analysis_partition"] == "target"}
    target_outcome_families = {record["family_id"] for record in outcomes if record["analysis_partition"] == "target"}
    target_labels = {record["target_label"] for record in outcomes if record["analysis_partition"] == "target"}
    if len(target_outcome_families) < MINIMUM_CONFIRMATORY_TARGET_FAMILIES:
        local_missing.append(
            f"minimum_evaluable_target_families:{len(target_outcome_families)}/{MINIMUM_CONFIRMATORY_TARGET_FAMILIES}"
        )
    if target_labels != {0, 1}:
        local_missing.append("target_binary_label_support:" + ",".join(str(label) for label in sorted(target_labels)))
    missing_target_families = sorted(target_input_families - target_outcome_families)
    if missing_target_families:
        local_missing.append("target_families_without_outcomes:" + ",".join(missing_target_families))

    for transform in transformation["specification"]["assay_transformations"]:
        if transform["minimum_pass_replicates_per_role"] < 2:
            local_missing.append(f"biological_replicate_minimum_below_two:{transform['assay_id']}")

    local_missing = sorted(set(local_missing))
    local_ready = not local_missing
    missing = sorted(
        {
            *local_missing,
            "externally_authenticated_trust_index",
        }
    )
    return {
        "schema_version": RAW_DMS_SCHEMA_VERSION,
        "status": "NOT_READY_FOR_CONFIRMATORY_EXECUTION",
        "confirmatory_eligible": False,
        "eligible": False,
        "local_structural_status": (
            "READY_AWAITING_EXTERNAL_AUTHENTICATION" if local_ready else "NOT_READY_LOCAL_STRUCTURE"
        ),
        "external_authentication_status": "NOT_AUTHENTICATED",
        "minimum_target_families": MINIMUM_CONFIRMATORY_TARGET_FAMILIES,
        "observed_source_outcome_items": len(source_outcomes),
        "observed_source_outcome_families": len(source_outcome_families),
        "observed_source_labels": sorted(source_labels),
        "observed_target_input_families": len(target_input_families),
        "observed_target_outcome_families": len(target_outcome_families),
        "observed_target_labels": sorted(target_labels),
        "local_missing_requirements": local_missing,
        "missing_requirements": missing,
        "release_sha256": canonical_sha256(release),
    }


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write canonical JSON bytes; callers must validate with the typed API first."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(artifact) + b"\n")
    return destination

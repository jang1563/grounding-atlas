"""Fail-closed schema-1 contract for the aggregate ProteinGym DMS pilot.

The contract deliberately separates label-free model inputs from assay outcomes.
Schema 1 only accepts ``release_status.mode="pilot_nonconfirmatory"`` and is
restricted to ProteinGym's direction-normalized ``DMS_score`` and
``DMS_score_bin`` profile. The bundled builder is pilot-only because the local
source files do not include replicate, license, or protein-family evidence. Raw
MaveDB re-ingestion requires a future assay-specific schema for WT, replicate,
transformation-derived outcomes, and QC; it must not be passed off as this
aggregate schema.

Schema 1 uses the cross-dataset semantic orientation required by the DMS design:
``target_label=1`` means retained/neutral function and is identical to ProteinGym's
``DMS_score_bin``. This intentionally differs from the legacy
``variant_grounding/data/variant_dms*.csv`` tables, where label 1 meant damaging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DMS_SCHEMA_VERSION = 1
PREREGISTRATION_ARTIFACT_TYPE = "dms_preregistration"
INPUT_MANIFEST_ARTIFACT_TYPE = "dms_label_free_input_manifest"
REPLICATE_MANIFEST_ARTIFACT_TYPE = "dms_replicate_manifest"
OUTCOME_MANIFEST_ARTIFACT_TYPE = "dms_outcome_manifest"

PILOT_RELEASE_MODE = "pilot_nonconfirmatory"
CONFIRMATORY_RELEASE_MODE = "confirmatory"
SCHEMA_V1_CONFIRMATORY_REJECTION = (
    "DMS schema version 1 is pilot-only; raw WT/replicate transformation-derived "
    "outcomes require a future assay-specific schema"
)
TRUTH_LEVEL = "T5"

PROTEINGYM_REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
)
PROTEINGYM_ASSAY_URL_TEMPLATE = (
    "https://huggingface.co/datasets/ICML2022/ProteinGym/resolve/main/ProteinGym_substitutions/{assay_id}.csv"
)
LOCAL_PROTEINGYM_ASSAY_IDS = (
    "BRCA1_HUMAN_Findlay_2018",
    "MSH2_HUMAN_Jia_2020",
    "P53_HUMAN_Kotler_2018",
    "PTEN_HUMAN_Mighell_2018",
)

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DATE_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

FORBIDDEN_INPUT_OUTCOME_KEYS = frozenset(
    {
        "dms_score",
        "dms_score_bin",
        "target_label",
        "label",
        "outcome",
        "raw_value",
        "normalized_effect",
        "fitness",
        "function_score",
        "lof_score",
        "correctness",
    }
)

PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "preregistration_id",
        "task_id",
        "task_family_id",
        "biological_question_id",
        "truth_level",
        "release_status",
        "source_provenance",
        "assay_contracts",
        "target_contract",
        "variant_contract",
        "grouping_contract",
        "analysis_contract",
        "transformation_contract",
        "metadata_status",
    }
)
RELEASE_STATUS_FIELDS = frozenset({"mode", "confirmatory_eligible", "missing_confirmatory_metadata"})
SOURCE_PROVENANCE_FIELDS = frozenset(
    {
        "upstream_name",
        "upstream_dataset_version",
        "upstream_revision",
        "reference_catalog_url",
        "reference_catalog_sha256",
        "assay_url_template",
        "access_date",
        "license_id",
        "redistribution_status",
    }
)
ASSAY_CONTRACT_FIELDS = frozenset(
    {
        "assay_id",
        "protein_entry",
        "molecule_name",
        "organism",
        "reference_sequence",
        "reference_sequence_sha256",
        "split_group_id",
        "region_mutated",
        "selection_assay",
        "selection_type",
        "coarse_selection_type",
        "raw_phenotype_name",
        "protein_gym_score_name",
        "protein_gym_score_orientation",
        "raw_score_directionality",
        "binarization_cutoff",
        "binarization_method",
        "doi",
        "publication_title",
        "publication_year",
        "protein_gym_version",
        "source_file_name",
        "source_file_sha256",
        "source_record_count",
    }
)
TARGET_CONTRACT_FIELDS = frozenset(
    {
        "continuous_source_field",
        "continuous_endpoint",
        "continuous_orientation",
        "binary_source_field",
        "binary_functional_value",
        "target_field",
        "target_rule",
        "target_label_0_semantics",
        "target_label_1_semantics",
        "missing_outcome_policy",
    }
)
VARIANT_CONTRACT_FIELDS = frozenset(
    {
        "variant_type",
        "coordinate_system",
        "reference_match_required",
        "allow_multiple_mutants",
        "allow_synonymous",
    }
)
GROUPING_CONTRACT_FIELDS = frozenset(
    {
        "split_group_scope",
        "family_metadata_status",
        "family_map_sha256",
        "intervention_pair_definition",
        "require_non_null_pair",
        "minimum_confirmatory_groups",
    }
)
ANALYSIS_CONTRACT_FIELDS = frozenset(
    {
        "primary_metric",
        "uncertainty_method",
        "minimum_items",
        "minimum_groups",
        "minimum_bootstrap_draws",
    }
)
TRANSFORMATION_CONTRACT_FIELDS = frozenset(
    {
        "code_path",
        "code_sha256",
        "selection_policy",
        "random_seed",
    }
)
METADATA_STATUS_FIELDS = frozenset({"license", "replicates", "protein_family"})
METADATA_ENTRY_FIELDS = frozenset({"status", "reason", "evidence_sha256"})

INPUT_MANIFEST_FIELDS = frozenset({"schema_version", "artifact_type", "preregistration_sha256", "records"})
INPUT_RECORD_FIELDS = frozenset(
    {
        "item_id",
        "entity_id",
        "task_family_id",
        "source_dataset",
        "assay_id",
        "source_record_id",
        "protein_entry",
        "construct_id",
        "reference_sequence_sha256",
        "mutation",
        "mutant_sequence_sha256",
        "representation_kind",
        "representation_sha256",
        "variant_type",
        "condition_id",
        "split_group_id",
        "split_group_scope",
        "intervention_pair_id",
    }
)

REPLICATE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "preregistration_sha256",
        "input_manifest_sha256",
        "status",
        "status_reason",
        "records",
    }
)
REPLICATE_RECORD_FIELDS = frozenset(
    {
        "replicate_id",
        "item_id",
        "intervention_pair_id",
        "role",
        "source_record_id",
        "biological_replicate_id",
        "technical_replicate_id",
        "assay_batch_id",
        "library_id",
        "timepoint",
        "raw_value",
        "raw_unit",
        "transformed_value",
        "qc_status",
        "qc_reason_codes",
    }
)

OUTCOME_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "preregistration_sha256",
        "input_manifest_sha256",
        "replicate_manifest_sha256",
        "target_contract_sha256",
        "records",
    }
)
OUTCOME_RECORD_FIELDS = frozenset(
    {
        "item_id",
        "assay_id",
        "source_record_id",
        "dms_score",
        "dms_score_bin",
        "target_label",
        "qc_status",
        "qc_reason_codes",
        "effect_uncertainty",
        "uncertainty_status",
        "uncertainty_reason",
        "replicate_metadata_status",
        "replicate_metadata_reason",
        "source_file_sha256",
    }
)

RELEASE_FIELDS = frozenset(
    {
        "preregistration",
        "input_manifest",
        "replicate_manifest",
        "outcome_manifest",
    }
)

RAW_PROTEINGYM_FIELDS = ("mutant", "DMS_score", "DMS_score_bin")
REFERENCE_REQUIRED_FIELDS = frozenset(
    {
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "source_organism",
        "target_seq",
        "seq_len",
        "includes_multiple_mutants",
        "DMS_number_single_mutants",
        "DMS_binarization_cutoff",
        "DMS_binarization_method",
        "title",
        "year",
        "jo",
        "region_mutated",
        "molecule_name",
        "selection_assay",
        "selection_type",
        "raw_DMS_phenotype_name",
        "raw_DMS_directionality",
        "ProteinGym_version",
        "coarse_selection_type",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for all DMS hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def preregistration_sha256(artifact: Mapping[str, Any]) -> str:
    validate_preregistration(artifact)
    return canonical_sha256(artifact)


def input_manifest_sha256(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> str:
    validate_input_manifest(manifest, preregistration)
    return canonical_sha256(manifest)


def replicate_manifest_sha256(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> str:
    validate_replicate_manifest(manifest, preregistration, input_manifest)
    return canonical_sha256(manifest)


def outcome_manifest_sha256(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    replicate_manifest: Mapping[str, Any],
) -> str:
    validate_outcome_manifest(
        manifest,
        preregistration,
        input_manifest,
        replicate_manifest,
    )
    return canonical_sha256(manifest)


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


def _strict_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a JSON boolean")
    return value


def _strict_int(
    value: Any,
    context: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _binary_int(value: Any, context: str) -> int:
    result = _strict_int(value, context)
    if result not in {0, 1}:
        raise ValueError(f"{context} must be the integer 0 or 1")
    return result


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


def _optional_sha256(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


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


def _validate_sorted_unique_strings(
    values: Any,
    context: str,
) -> list[str]:
    items = _list(values, context)
    for index, value in enumerate(items):
        _string(value, f"{context}[{index}]")
    if items != sorted(items) or len(set(items)) != len(items):
        raise ValueError(f"{context} must contain sorted unique strings")
    return items


def _validate_metadata_entry(value: Any, context: str) -> str:
    entry = _exact_fields(value, METADATA_ENTRY_FIELDS, context)
    status = entry["status"]
    if status not in {"available", "unavailable"}:
        raise ValueError(f"{context}.status must be available or unavailable")
    reason = entry["reason"]
    evidence = entry["evidence_sha256"]
    if status == "available":
        if reason is not None:
            raise ValueError(f"{context}.reason must be null when metadata is available")
        _sha256(evidence, f"{context}.evidence_sha256")
    else:
        _string(reason, f"{context}.reason")
        if evidence is not None:
            raise ValueError(f"{context}.evidence_sha256 must be null when metadata is unavailable")
    return status


def _assert_no_outcome_keys(value: Any, context: str = "input manifest") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_INPUT_OUTCOME_KEYS:
                raise ValueError(f"{context} is label-free and cannot contain outcome key {key!r}")
            _assert_no_outcome_keys(nested, context)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_outcome_keys(nested, context)


def validate_preregistration(artifact: Mapping[str, Any]) -> None:
    artifact = _exact_fields(
        artifact,
        PREREGISTRATION_FIELDS,
        "DMS preregistration",
    )
    if artifact["schema_version"] != DMS_SCHEMA_VERSION:
        raise ValueError("unsupported DMS preregistration schema version")
    if artifact["artifact_type"] != PREREGISTRATION_ARTIFACT_TYPE:
        raise ValueError("invalid DMS preregistration artifact type")
    for field in (
        "preregistration_id",
        "task_id",
        "task_family_id",
        "biological_question_id",
    ):
        _string(artifact[field], f"DMS preregistration.{field}")
    if artifact["truth_level"] != TRUTH_LEVEL:
        raise ValueError("DMS preregistration truth_level must be T5")

    metadata = _exact_fields(
        artifact["metadata_status"],
        METADATA_STATUS_FIELDS,
        "DMS preregistration.metadata_status",
    )
    metadata_states = {
        name: _validate_metadata_entry(
            metadata[name],
            f"DMS preregistration.metadata_status.{name}",
        )
        for name in sorted(METADATA_STATUS_FIELDS)
    }

    release = _exact_fields(
        artifact["release_status"],
        RELEASE_STATUS_FIELDS,
        "DMS preregistration.release_status",
    )
    mode = release["mode"]
    if mode == CONFIRMATORY_RELEASE_MODE:
        raise ValueError(SCHEMA_V1_CONFIRMATORY_REJECTION)
    if mode != PILOT_RELEASE_MODE:
        raise ValueError("DMS release mode is invalid")
    eligible = _strict_bool(
        release["confirmatory_eligible"],
        "DMS preregistration.release_status.confirmatory_eligible",
    )
    missing = _validate_sorted_unique_strings(
        release["missing_confirmatory_metadata"],
        "DMS preregistration.release_status.missing_confirmatory_metadata",
    )
    expected_missing = sorted(name for name, status in metadata_states.items() if status == "unavailable")
    if missing != expected_missing:
        raise ValueError("release missing_confirmatory_metadata must exactly match unavailable metadata")
    if eligible or not missing:
        raise ValueError("pilot DMS releases must be nonconfirmatory and name missing metadata")

    source = _exact_fields(
        artifact["source_provenance"],
        SOURCE_PROVENANCE_FIELDS,
        "DMS preregistration.source_provenance",
    )
    for field in (
        "upstream_name",
        "upstream_dataset_version",
        "reference_catalog_url",
        "assay_url_template",
        "redistribution_status",
    ):
        _string(source[field], f"DMS preregistration.source_provenance.{field}")
    _sha256(
        source["reference_catalog_sha256"],
        "DMS preregistration.source_provenance.reference_catalog_sha256",
    )
    _optional_string(
        source["upstream_revision"],
        "DMS preregistration.source_provenance.upstream_revision",
    )
    access_date = _optional_string(
        source["access_date"],
        "DMS preregistration.source_provenance.access_date",
    )
    license_id = _optional_string(
        source["license_id"],
        "DMS preregistration.source_provenance.license_id",
    )
    if access_date is not None and DATE_PATTERN.fullmatch(access_date) is None:
        raise ValueError("source access_date must use YYYY-MM-DD")
    if metadata_states["license"] == "available":
        if license_id is None or source["redistribution_status"] == "unresolved":
            raise ValueError("available license metadata requires a license ID and resolved redistribution status")
    elif license_id is not None or source["redistribution_status"] != "unresolved":
        raise ValueError("unavailable license metadata requires null license_id and unresolved redistribution")
    assays = _list(
        artifact["assay_contracts"],
        "DMS preregistration.assay_contracts",
    )
    if not assays:
        raise ValueError("DMS preregistration requires at least one assay contract")
    assay_ids: list[str] = []
    for index, raw_assay in enumerate(assays):
        context = f"DMS preregistration.assay_contracts[{index}]"
        assay = _exact_fields(raw_assay, ASSAY_CONTRACT_FIELDS, context)
        assay_id = _string(assay["assay_id"], f"{context}.assay_id")
        assay_ids.append(assay_id)
        for field in (
            "protein_entry",
            "molecule_name",
            "organism",
            "split_group_id",
            "region_mutated",
            "selection_assay",
            "coarse_selection_type",
            "raw_phenotype_name",
            "protein_gym_score_name",
            "protein_gym_score_orientation",
            "binarization_method",
            "doi",
            "publication_title",
            "protein_gym_version",
            "source_file_name",
        ):
            _string(assay[field], f"{context}.{field}")
        _optional_string(assay["selection_type"], f"{context}.selection_type")
        sequence = _string(assay["reference_sequence"], f"{context}.reference_sequence")
        if any(residue not in CANONICAL_AMINO_ACIDS for residue in sequence):
            raise ValueError(f"{context}.reference_sequence is not a canonical protein")
        if sequence_sha256(sequence) != _sha256(
            assay["reference_sequence_sha256"],
            f"{context}.reference_sequence_sha256",
        ):
            raise ValueError(f"{context}.reference_sequence checksum differs")
        if assay["protein_gym_score_name"] != "DMS_score":
            raise ValueError(f"{context}.protein_gym_score_name must be DMS_score")
        if assay["protein_gym_score_orientation"] != "higher_is_more_functional":
            raise ValueError(f"{context}.protein_gym_score_orientation must be higher_is_more_functional")
        directionality = _strict_int(
            assay["raw_score_directionality"],
            f"{context}.raw_score_directionality",
        )
        if directionality not in {-1, 1}:
            raise ValueError(f"{context}.raw_score_directionality must be -1 or 1")
        _finite_float(assay["binarization_cutoff"], f"{context}.binarization_cutoff")
        _strict_int(
            assay["publication_year"],
            f"{context}.publication_year",
            minimum=1900,
        )
        _sha256(assay["source_file_sha256"], f"{context}.source_file_sha256")
        _strict_int(
            assay["source_record_count"],
            f"{context}.source_record_count",
            minimum=1,
        )
        if Path(assay["source_file_name"]).name != assay["source_file_name"]:
            raise ValueError(f"{context}.source_file_name must be a basename")
    if assay_ids != sorted(assay_ids) or len(set(assay_ids)) != len(assay_ids):
        raise ValueError("assay contracts must be sorted by unique assay_id")

    target = _exact_fields(
        artifact["target_contract"],
        TARGET_CONTRACT_FIELDS,
        "DMS preregistration.target_contract",
    )
    expected_target = {
        "continuous_source_field": "DMS_score",
        "continuous_endpoint": "protein_gym_direction_normalized_function_score",
        "continuous_orientation": "higher_is_more_functional",
        "binary_source_field": "DMS_score_bin",
        "binary_functional_value": 1,
        "target_field": "target_label",
        "target_rule": "DMS_score_bin",
        "target_label_0_semantics": ("damaging_or_loss_of_function_under_assay_threshold"),
        "target_label_1_semantics": "retained_or_neutral_function_under_assay_threshold",
        "missing_outcome_policy": "reject",
    }
    if dict(target) != expected_target:
        raise ValueError("DMS target contract differs from the locked schema-1 estimand")

    variant = _exact_fields(
        artifact["variant_contract"],
        VARIANT_CONTRACT_FIELDS,
        "DMS preregistration.variant_contract",
    )
    expected_variant = {
        "variant_type": "single_amino_acid_substitution",
        "coordinate_system": "protein_1_based",
        "reference_match_required": True,
        "allow_multiple_mutants": False,
        "allow_synonymous": False,
    }
    if dict(variant) != expected_variant:
        raise ValueError("DMS variant contract differs from the exact schema-1 policy")

    grouping = _exact_fields(
        artifact["grouping_contract"],
        GROUPING_CONTRACT_FIELDS,
        "DMS preregistration.grouping_contract",
    )
    scope = grouping["split_group_scope"]
    if scope not in {
        "protein_family",
        "protein_identity_proxy_nonconfirmatory",
    }:
        raise ValueError("DMS split_group_scope is invalid")
    if grouping["family_metadata_status"] != metadata_states["protein_family"]:
        raise ValueError("grouping and protein-family metadata statuses differ")
    family_map_sha256 = _optional_sha256(
        grouping["family_map_sha256"],
        "DMS preregistration.grouping_contract.family_map_sha256",
    )
    if metadata_states["protein_family"] == "available":
        if scope != "protein_family" or family_map_sha256 is None:
            raise ValueError("available family metadata requires protein_family scope and a family-map checksum")
        if metadata["protein_family"]["evidence_sha256"] != family_map_sha256:
            raise ValueError("protein-family evidence and family-map checksums differ")
    elif scope != "protein_identity_proxy_nonconfirmatory" or family_map_sha256 is not None:
        raise ValueError("missing family metadata requires the nonconfirmatory protein-identity proxy")
    if grouping["intervention_pair_definition"] != ("assay_id+single_amino_acid_substitution+assay_wild_type"):
        raise ValueError("DMS intervention-pair definition is invalid")
    if not _strict_bool(
        grouping["require_non_null_pair"],
        "DMS preregistration.grouping_contract.require_non_null_pair",
    ):
        raise ValueError("DMS intervention pairs must be non-null")
    _strict_int(
        grouping["minimum_confirmatory_groups"],
        "DMS preregistration.grouping_contract.minimum_confirmatory_groups",
        minimum=8,
    )

    analysis = _exact_fields(
        artifact["analysis_contract"],
        ANALYSIS_CONTRACT_FIELDS,
        "DMS preregistration.analysis_contract",
    )
    for field in ("primary_metric", "uncertainty_method"):
        _string(analysis[field], f"DMS preregistration.analysis_contract.{field}")
    _strict_int(
        analysis["minimum_items"],
        "DMS preregistration.analysis_contract.minimum_items",
        minimum=1,
    )
    _strict_int(
        analysis["minimum_groups"],
        "DMS preregistration.analysis_contract.minimum_groups",
        minimum=1,
    )
    _strict_int(
        analysis["minimum_bootstrap_draws"],
        "DMS preregistration.analysis_contract.minimum_bootstrap_draws",
        minimum=1,
    )
    if analysis["minimum_groups"] < grouping["minimum_confirmatory_groups"]:
        raise ValueError("analysis minimum_groups cannot weaken the grouping contract")

    transformation = _exact_fields(
        artifact["transformation_contract"],
        TRANSFORMATION_CONTRACT_FIELDS,
        "DMS preregistration.transformation_contract",
    )
    _string(transformation["code_path"], "transformation_contract.code_path")
    _sha256(transformation["code_sha256"], "transformation_contract.code_sha256")
    if transformation["selection_policy"] != "all_valid_source_rows":
        raise ValueError("DMS selection policy must retain all valid source rows")
    if transformation["random_seed"] is not None:
        _strict_int(transformation["random_seed"], "transformation_contract.random_seed")


def validate_input_manifest(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    validate_preregistration(preregistration)
    manifest = _exact_fields(
        manifest,
        INPUT_MANIFEST_FIELDS,
        "DMS label-free input manifest",
    )
    _assert_no_outcome_keys(manifest)
    if manifest["schema_version"] != DMS_SCHEMA_VERSION:
        raise ValueError("unsupported DMS input-manifest schema version")
    if manifest["artifact_type"] != INPUT_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid DMS input-manifest artifact type")
    expected_preregistration_sha256 = preregistration_sha256(preregistration)
    if (
        _sha256(
            manifest["preregistration_sha256"],
            "DMS input manifest.preregistration_sha256",
        )
        != expected_preregistration_sha256
    ):
        raise ValueError("DMS input manifest is bound to a different preregistration")

    assays = {assay["assay_id"]: assay for assay in preregistration["assay_contracts"]}
    grouping = preregistration["grouping_contract"]
    records = _list(manifest["records"], "DMS input manifest.records")
    if not records:
        raise ValueError("DMS input manifest cannot be empty")
    if len(records) != sum(assay["source_record_count"] for assay in assays.values()):
        raise ValueError("all_valid_source_rows input manifest must cover every registered source row")

    item_ids: list[str] = []
    source_keys: set[tuple[str, str]] = set()
    pair_to_group: dict[str, str] = {}
    entity_to_group: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        context = f"DMS input manifest.records[{index}]"
        record = _exact_fields(raw_record, INPUT_RECORD_FIELDS, context)
        item_id = _string(record["item_id"], f"{context}.item_id")
        entity_id = _string(record["entity_id"], f"{context}.entity_id")
        if record["task_family_id"] != preregistration["task_family_id"]:
            raise ValueError(f"{context}.task_family_id differs from preregistration")
        assay_id = _string(record["assay_id"], f"{context}.assay_id")
        source_record_id = _string(
            record["source_record_id"],
            f"{context}.source_record_id",
        )
        if assay_id not in assays:
            raise ValueError(f"{context}.assay_id is not preregistered")
        assay = assays[assay_id]
        if record["source_dataset"] != preregistration["source_provenance"]["upstream_name"]:
            raise ValueError(f"{context}.source_dataset differs from preregistration")
        if record["protein_entry"] != assay["protein_entry"]:
            raise ValueError(f"{context}.protein_entry differs from assay contract")
        expected_construct_id = f"{assay_id}:reference_construct"
        if record["construct_id"] != expected_construct_id:
            raise ValueError(f"{context}.construct_id is not identity-bound")
        if record["reference_sequence_sha256"] != assay["reference_sequence_sha256"]:
            raise ValueError(f"{context}.reference_sequence_sha256 differs from assay contract")
        mutation = _string(record["mutation"], f"{context}.mutation")
        reference, position, alternate = _parse_mutation(
            mutation,
            f"{context}.mutation",
        )
        sequence = assay["reference_sequence"]
        if position > len(sequence) or sequence[position - 1] != reference:
            raise ValueError(f"{context}.mutation does not match the registered reference sequence")
        mutant_sequence = sequence[: position - 1] + alternate + sequence[position:]
        if _sha256(
            record["mutant_sequence_sha256"],
            f"{context}.mutant_sequence_sha256",
        ) != sequence_sha256(mutant_sequence):
            raise ValueError(f"{context}.mutant-sequence checksum differs")
        if record["representation_kind"] != "full_mutant_protein_sequence":
            raise ValueError(f"{context}.representation_kind is invalid")
        if _sha256(
            record["representation_sha256"],
            f"{context}.representation_sha256",
        ) != sequence_sha256(mutant_sequence):
            raise ValueError(f"{context}.representation checksum differs")
        if record["variant_type"] != "single_amino_acid_substitution":
            raise ValueError(f"{context}.variant_type is invalid")
        if record["condition_id"] != "mutant":
            raise ValueError(f"{context}.condition_id must be mutant")
        if record["split_group_scope"] != grouping["split_group_scope"]:
            raise ValueError(f"{context}.split_group_scope differs from preregistration")
        split_group_id = _string(
            record["split_group_id"],
            f"{context}.split_group_id",
        )
        if split_group_id != assay["split_group_id"]:
            raise ValueError(f"{context}.split_group_id differs from assay contract")
        pair_id = _string(
            record["intervention_pair_id"],
            f"{context}.intervention_pair_id",
        )

        expected_item_id = f"{assay_id}:{mutation}"
        expected_entity_id = f"{assay['protein_entry']}:{mutation}"
        expected_pair_id = f"{assay_id}:mutant-vs-wt:{mutation}"
        if item_id != expected_item_id:
            raise ValueError(f"{context}.item_id is not identity-bound")
        if entity_id != expected_entity_id:
            raise ValueError(f"{context}.entity_id is not identity-bound")
        if source_record_id != mutation:
            raise ValueError(f"{context}.source_record_id must equal the source mutation")
        if pair_id != expected_pair_id:
            raise ValueError(f"{context}.intervention_pair_id is not identity-bound")

        source_key = (assay_id, source_record_id)
        if source_key in source_keys:
            raise ValueError("DMS input manifest contains duplicate source records")
        source_keys.add(source_key)
        previous_pair_group = pair_to_group.setdefault(pair_id, split_group_id)
        if previous_pair_group != split_group_id:
            raise ValueError("each DMS intervention_pair_id must nest within one split_group_id")
        previous_entity_group = entity_to_group.setdefault(entity_id, split_group_id)
        if previous_entity_group != split_group_id:
            raise ValueError("each DMS biological entity must nest within one split_group_id")
        item_ids.append(item_id)
    if item_ids != sorted(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("DMS input records must be sorted by unique item_id")


def validate_replicate_manifest(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> None:
    manifest = _exact_fields(
        manifest,
        REPLICATE_MANIFEST_FIELDS,
        "DMS replicate manifest",
    )
    if manifest["schema_version"] != DMS_SCHEMA_VERSION:
        raise ValueError("unsupported DMS replicate-manifest schema version")
    if manifest["artifact_type"] != REPLICATE_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid DMS replicate-manifest artifact type")
    records = _list(manifest["records"], "DMS replicate manifest.records")
    release_mode = (
        preregistration.get("release_status", {}).get("mode") if isinstance(preregistration, Mapping) else None
    )
    if release_mode == CONFIRMATORY_RELEASE_MODE:
        raise ValueError(SCHEMA_V1_CONFIRMATORY_REJECTION)
    validate_input_manifest(input_manifest, preregistration)
    if _sha256(
        manifest["preregistration_sha256"],
        "DMS replicate manifest.preregistration_sha256",
    ) != canonical_sha256(preregistration):
        raise ValueError("DMS replicate manifest is bound to a different preregistration")
    if _sha256(
        manifest["input_manifest_sha256"],
        "DMS replicate manifest.input_manifest_sha256",
    ) != canonical_sha256(input_manifest):
        raise ValueError("DMS replicate manifest is bound to different label-free inputs")

    preregistered_status = preregistration["metadata_status"]["replicates"]["status"]
    status = manifest["status"]
    reason = manifest["status_reason"]
    if status == "unavailable_aggregate_only":
        if preregistered_status != "unavailable":
            raise ValueError("aggregate-only replicate status differs from preregistration")
        _string(reason, "DMS replicate manifest.status_reason")
        if records:
            raise ValueError("aggregate-only replicate manifest must have an empty record list")
        return
    if status != "available":
        raise ValueError("DMS replicate manifest.status is invalid")
    if preregistered_status != "available":
        raise ValueError("available replicate status differs from preregistration")
    if reason is not None:
        raise ValueError("available DMS replicate manifest must have a null status_reason")
    if not records:
        raise ValueError("available DMS replicate manifest must contain replicate lineage")

    inputs = {record["item_id"]: record for record in input_manifest["records"]}
    replicate_ids: list[str] = []
    roles_by_item: dict[str, set[str]] = {item_id: set() for item_id in inputs}
    for index, raw_record in enumerate(records):
        context = f"DMS replicate manifest.records[{index}]"
        record = _exact_fields(raw_record, REPLICATE_RECORD_FIELDS, context)
        replicate_id = _string(
            record["replicate_id"],
            f"{context}.replicate_id",
        )
        item_id = _string(record["item_id"], f"{context}.item_id")
        if item_id not in inputs:
            raise ValueError(f"{context}.item_id is absent from label-free inputs")
        if record["intervention_pair_id"] != inputs[item_id]["intervention_pair_id"]:
            raise ValueError(f"{context}.intervention_pair_id differs from label-free inputs")
        role = record["role"]
        if role not in {"mutant", "wild_type"}:
            raise ValueError(f"{context}.role must be mutant or wild_type")
        roles_by_item[item_id].add(role)
        _string(record["source_record_id"], f"{context}.source_record_id")
        biological_id = _optional_string(
            record["biological_replicate_id"],
            f"{context}.biological_replicate_id",
        )
        technical_id = _optional_string(
            record["technical_replicate_id"],
            f"{context}.technical_replicate_id",
        )
        if biological_id is None and technical_id is None:
            raise ValueError(f"{context} requires a biological or technical replicate ID")
        for field in ("assay_batch_id", "library_id", "timepoint"):
            _optional_string(record[field], f"{context}.{field}")
        _finite_float(record["raw_value"], f"{context}.raw_value")
        _string(record["raw_unit"], f"{context}.raw_unit")
        _finite_float(
            record["transformed_value"],
            f"{context}.transformed_value",
        )
        if record["qc_status"] not in {"PASS", "FAIL"}:
            raise ValueError(f"{context}.qc_status must be PASS or FAIL")
        reasons = _validate_sorted_unique_strings(
            record["qc_reason_codes"],
            f"{context}.qc_reason_codes",
        )
        if record["qc_status"] == "PASS" and reasons:
            raise ValueError(f"{context}.PASS rows cannot have QC reason codes")
        if record["qc_status"] == "FAIL" and not reasons:
            raise ValueError(f"{context}.FAIL rows require QC reason codes")
        replicate_ids.append(replicate_id)
    if replicate_ids != sorted(replicate_ids) or len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("DMS replicate records must be sorted by unique replicate_id")
    incomplete_items = sorted(item_id for item_id, roles in roles_by_item.items() if roles != {"mutant", "wild_type"})
    if incomplete_items:
        raise ValueError(
            "available replicate lineage requires mutant and wild-type records "
            f"for every item; incomplete={incomplete_items[:5]}"
        )


def validate_outcome_manifest(
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    replicate_manifest: Mapping[str, Any],
) -> None:
    validate_replicate_manifest(
        replicate_manifest,
        preregistration,
        input_manifest,
    )
    manifest = _exact_fields(
        manifest,
        OUTCOME_MANIFEST_FIELDS,
        "DMS outcome manifest",
    )
    if manifest["schema_version"] != DMS_SCHEMA_VERSION:
        raise ValueError("unsupported DMS outcome-manifest schema version")
    if manifest["artifact_type"] != OUTCOME_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("invalid DMS outcome-manifest artifact type")
    if _sha256(
        manifest["preregistration_sha256"],
        "DMS outcome manifest.preregistration_sha256",
    ) != canonical_sha256(preregistration):
        raise ValueError("DMS outcome manifest is bound to a different preregistration")
    if _sha256(
        manifest["input_manifest_sha256"],
        "DMS outcome manifest.input_manifest_sha256",
    ) != canonical_sha256(input_manifest):
        raise ValueError("DMS outcome manifest is bound to different label-free inputs")
    if _sha256(
        manifest["replicate_manifest_sha256"],
        "DMS outcome manifest.replicate_manifest_sha256",
    ) != canonical_sha256(replicate_manifest):
        raise ValueError("DMS outcome manifest is bound to different replicate lineage")
    if _sha256(
        manifest["target_contract_sha256"],
        "DMS outcome manifest.target_contract_sha256",
    ) != canonical_sha256(preregistration["target_contract"]):
        raise ValueError("DMS outcome manifest target-contract checksum differs")

    assays = {assay["assay_id"]: assay for assay in preregistration["assay_contracts"]}
    inputs = {record["item_id"]: record for record in input_manifest["records"]}
    records = _list(manifest["records"], "DMS outcome manifest.records")
    item_ids: list[str] = []
    for index, raw_record in enumerate(records):
        context = f"DMS outcome manifest.records[{index}]"
        record = _exact_fields(raw_record, OUTCOME_RECORD_FIELDS, context)
        item_id = _string(record["item_id"], f"{context}.item_id")
        assay_id = _string(record["assay_id"], f"{context}.assay_id")
        source_record_id = _string(
            record["source_record_id"],
            f"{context}.source_record_id",
        )
        if item_id not in inputs:
            raise ValueError(f"{context}.item_id is absent from label-free inputs")
        input_record = inputs[item_id]
        if assay_id != input_record["assay_id"] or source_record_id != input_record["source_record_id"]:
            raise ValueError(f"{context} identity differs from label-free inputs")
        _finite_float(record["dms_score"], f"{context}.dms_score")
        functional_bin = _binary_int(
            record["dms_score_bin"],
            f"{context}.dms_score_bin",
        )
        target_label = _binary_int(
            record["target_label"],
            f"{context}.target_label",
        )
        if target_label != functional_bin:
            raise ValueError(f"{context}.target_label violates the locked binary rule")
        if record["qc_status"] != "PASS":
            raise ValueError(f"{context}.qc_status must be PASS for released rows")
        reasons = _list(record["qc_reason_codes"], f"{context}.qc_reason_codes")
        if reasons:
            raise ValueError(f"{context}.qc_reason_codes must be empty for PASS rows")
        uncertainty_status = record["uncertainty_status"]
        if uncertainty_status not in {"available", "unavailable"}:
            raise ValueError(f"{context}.uncertainty_status is invalid")
        if uncertainty_status == "available":
            uncertainty = _finite_float(
                record["effect_uncertainty"],
                f"{context}.effect_uncertainty",
            )
            if uncertainty < 0:
                raise ValueError(f"{context}.effect_uncertainty cannot be negative")
            if record["uncertainty_reason"] is not None:
                raise ValueError(f"{context}.uncertainty_reason must be null when available")
        else:
            if record["effect_uncertainty"] is not None:
                raise ValueError(f"{context}.effect_uncertainty must be null when unavailable")
            _string(record["uncertainty_reason"], f"{context}.uncertainty_reason")
        replicate_status = record["replicate_metadata_status"]
        preregistered_replicate_status = preregistration["metadata_status"]["replicates"]["status"]
        if replicate_status != preregistered_replicate_status:
            raise ValueError(f"{context}.replicate metadata status differs from preregistration")
        if replicate_status == "available":
            if record["replicate_metadata_reason"] is not None:
                raise ValueError(f"{context}.replicate_metadata_reason must be null when available")
        else:
            _string(
                record["replicate_metadata_reason"],
                f"{context}.replicate_metadata_reason",
            )
        if (
            _sha256(
                record["source_file_sha256"],
                f"{context}.source_file_sha256",
            )
            != assays[assay_id]["source_file_sha256"]
        ):
            raise ValueError(f"{context}.source-file checksum differs")
        item_ids.append(item_id)
    if item_ids != sorted(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("DMS outcome records must be sorted by unique item_id")
    if set(item_ids) != set(inputs):
        raise ValueError("DMS outcomes must identity-join one-to-one with all label-free inputs")


def validate_release(release: Mapping[str, Any]) -> None:
    release = _exact_fields(release, RELEASE_FIELDS, "DMS release")
    preregistration = release["preregistration"]
    input_manifest = release["input_manifest"]
    replicate_manifest = release["replicate_manifest"]
    outcome_manifest = release["outcome_manifest"]
    validate_outcome_manifest(
        outcome_manifest,
        preregistration,
        input_manifest,
        replicate_manifest,
    )


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("DMS contract artifacts must use the .json format")
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


def write_preregistration(
    path: str | Path,
    artifact: Mapping[str, Any],
) -> Path:
    validate_preregistration(artifact)
    return _atomic_write_json(path, artifact)


def write_input_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> Path:
    validate_input_manifest(manifest, preregistration)
    return _atomic_write_json(path, manifest)


def write_replicate_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> Path:
    validate_replicate_manifest(manifest, preregistration, input_manifest)
    return _atomic_write_json(path, manifest)


def write_outcome_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    replicate_manifest: Mapping[str, Any],
) -> Path:
    validate_outcome_manifest(
        manifest,
        preregistration,
        input_manifest,
        replicate_manifest,
    )
    return _atomic_write_json(path, manifest)


def load_contract_artifact(
    path: str | Path,
    *,
    preregistration: Mapping[str, Any] | None = None,
    input_manifest: Mapping[str, Any] | None = None,
    replicate_manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("legacy CSV tables are not DMS contract artifacts; use an exact JSON schema")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read DMS contract artifact: {exc}") from exc
    artifact = _mapping(value, "DMS contract artifact")
    artifact_type = artifact.get("artifact_type")
    if artifact_type == PREREGISTRATION_ARTIFACT_TYPE:
        validate_preregistration(artifact)
    elif artifact_type == INPUT_MANIFEST_ARTIFACT_TYPE:
        if preregistration is None:
            raise ValueError("input-manifest validation requires a preregistration")
        validate_input_manifest(artifact, preregistration)
    elif artifact_type == REPLICATE_MANIFEST_ARTIFACT_TYPE:
        if preregistration is None or input_manifest is None:
            raise ValueError("replicate-manifest validation requires preregistration and label-free inputs")
        validate_replicate_manifest(
            artifact,
            preregistration,
            input_manifest,
        )
    elif artifact_type == OUTCOME_MANIFEST_ARTIFACT_TYPE:
        if preregistration is None or input_manifest is None or replicate_manifest is None:
            raise ValueError(
                "outcome-manifest validation requires preregistration, label-free inputs, and replicate lineage"
            )
        validate_outcome_manifest(
            artifact,
            preregistration,
            input_manifest,
            replicate_manifest,
        )
    else:
        raise ValueError("unknown DMS contract artifact type")
    return artifact


def _load_reference_catalog(path: str | Path) -> dict[str, dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        observed = set(reader.fieldnames or [])
        if not REFERENCE_REQUIRED_FIELDS.issubset(observed):
            missing = sorted(REFERENCE_REQUIRED_FIELDS - observed)
            raise ValueError(f"ProteinGym reference catalog is missing required fields: {missing}")
        records: dict[str, dict[str, str]] = {}
        for row in reader:
            assay_id = row["DMS_id"]
            if assay_id in records:
                raise ValueError(f"ProteinGym reference catalog repeats assay {assay_id!r}")
            records[assay_id] = row
    return records


def _load_raw_proteingym_assay(
    path: str | Path,
    *,
    assay_id: str,
    reference_sequence: str,
) -> list[tuple[str, float, int]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_PROTEINGYM_FIELDS:
            raise ValueError(
                f"{assay_id} must use the exact raw ProteinGym schema "
                f"{list(RAW_PROTEINGYM_FIELDS)}; legacy derived CSVs are rejected"
            )
        records: list[tuple[str, float, int]] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            mutation = row["mutant"]
            reference, position, alternate = _parse_mutation(
                mutation,
                f"{assay_id} row {row_number} mutant",
            )
            if position > len(reference_sequence) or (reference_sequence[position - 1] != reference):
                raise ValueError(f"{assay_id} row {row_number} mutation does not match reference")
            if mutation in seen:
                raise ValueError(f"{assay_id} repeats mutation {mutation!r}")
            seen.add(mutation)
            try:
                score = float(row["DMS_score"])
            except ValueError as exc:
                raise ValueError(f"{assay_id} row {row_number} has invalid DMS_score") from exc
            if not math.isfinite(score):
                raise ValueError(f"{assay_id} row {row_number} has non-finite DMS_score")
            raw_bin = row["DMS_score_bin"]
            if raw_bin not in {"0", "1"}:
                raise ValueError(f"{assay_id} row {row_number} DMS_score_bin must be 0 or 1")
            records.append((mutation, score, int(raw_bin)))
    records.sort(key=lambda value: value[0])
    return records


def build_proteingym_pilot_release(
    *,
    reference_catalog_path: str | Path,
    assay_paths: Mapping[str, str | Path],
) -> dict[str, Mapping[str, Any]]:
    """Build the deterministic four-assay pilot from raw ProteinGym tables.

    The builder uses the four exact local assay CSVs and the local ProteinGym
    reference catalog. It does not read the legacy balanced CSVs, AlphaMissense,
    ClinVar, or model outputs.
    """

    if set(assay_paths) != set(LOCAL_PROTEINGYM_ASSAY_IDS):
        raise ValueError("ProteinGym pilot requires exactly the four preregistered local assay IDs")
    reference_path = Path(reference_catalog_path)
    references = _load_reference_catalog(reference_path)
    missing_references = sorted(set(LOCAL_PROTEINGYM_ASSAY_IDS) - set(references))
    if missing_references:
        raise ValueError(f"ProteinGym reference catalog lacks assays: {missing_references}")

    assay_contracts: list[dict[str, Any]] = []
    raw_records: dict[str, list[tuple[str, float, int]]] = {}
    versions: set[str] = set()
    for assay_id in sorted(LOCAL_PROTEINGYM_ASSAY_IDS):
        reference = references[assay_id]
        sequence = reference["target_seq"].strip().upper()
        if not sequence or any(residue not in CANONICAL_AMINO_ACIDS for residue in sequence):
            raise ValueError(f"{assay_id} has an invalid reference sequence")
        if int(reference["seq_len"]) != len(sequence):
            raise ValueError(f"{assay_id} reference sequence length differs")
        if reference["includes_multiple_mutants"].upper() != "FALSE":
            raise ValueError(f"{assay_id} contains unsupported multiple mutants")
        assay_path = Path(assay_paths[assay_id])
        rows = _load_raw_proteingym_assay(
            assay_path,
            assay_id=assay_id,
            reference_sequence=sequence,
        )
        if assay_path.name != reference["DMS_filename"]:
            raise ValueError(f"{assay_id} source filename differs from reference")
        expected_count = int(reference["DMS_number_single_mutants"])
        if len(rows) != expected_count:
            raise ValueError(f"{assay_id} row count {len(rows)} differs from reference {expected_count}")
        raw_records[assay_id] = rows
        versions.add(reference["ProteinGym_version"])
        assay_contracts.append(
            {
                "assay_id": assay_id,
                "protein_entry": reference["UniProt_ID"],
                "molecule_name": reference["molecule_name"],
                "organism": reference["source_organism"],
                "reference_sequence": sequence,
                "reference_sequence_sha256": sequence_sha256(sequence),
                "split_group_id": (f"protein_identity_proxy:{reference['UniProt_ID']}"),
                "region_mutated": reference["region_mutated"],
                "selection_assay": reference["selection_assay"],
                "selection_type": reference["selection_type"] or None,
                "coarse_selection_type": reference["coarse_selection_type"],
                "raw_phenotype_name": reference["raw_DMS_phenotype_name"],
                "protein_gym_score_name": "DMS_score",
                "protein_gym_score_orientation": "higher_is_more_functional",
                "raw_score_directionality": int(reference["raw_DMS_directionality"]),
                "binarization_cutoff": float(reference["DMS_binarization_cutoff"]),
                "binarization_method": reference["DMS_binarization_method"],
                "doi": reference["jo"],
                "publication_title": reference["title"],
                "publication_year": int(reference["year"]),
                "protein_gym_version": reference["ProteinGym_version"],
                "source_file_name": assay_path.name,
                "source_file_sha256": file_sha256(assay_path),
                "source_record_count": len(rows),
            }
        )
    if len(versions) != 1:
        raise ValueError("four-assay ProteinGym pilot requires one dataset version")

    unavailable_metadata = {
        "license": {
            "status": "unavailable",
            "reason": (
                "No source-specific license or redistribution record is stored with the local ProteinGym assay files."
            ),
            "evidence_sha256": None,
        },
        "replicates": {
            "status": "unavailable",
            "reason": (
                "Local ProteinGym files contain aggregate DMS_score values and no mutant or wild-type replicate ledger."
            ),
            "evidence_sha256": None,
        },
        "protein_family": {
            "status": "unavailable",
            "reason": (
                "No versioned protein-family map is stored locally; protein "
                "identity is only a nonconfirmatory grouping proxy."
            ),
            "evidence_sha256": None,
        },
    }
    preregistration: dict[str, Any] = {
        "schema_version": DMS_SCHEMA_VERSION,
        "artifact_type": PREREGISTRATION_ARTIFACT_TYPE,
        "preregistration_id": "proteingym-four-assay-pilot-v1",
        "task_id": "dms/proteingym_four_assay_pilot",
        "task_family_id": "dms_assay_variant_function",
        "biological_question_id": "assay_specific_single_variant_function",
        "truth_level": TRUTH_LEVEL,
        "release_status": {
            "mode": PILOT_RELEASE_MODE,
            "confirmatory_eligible": False,
            "missing_confirmatory_metadata": [
                "license",
                "protein_family",
                "replicates",
            ],
        },
        "source_provenance": {
            "upstream_name": "ProteinGym",
            "upstream_dataset_version": next(iter(versions)),
            "upstream_revision": None,
            "reference_catalog_url": PROTEINGYM_REFERENCE_URL,
            "reference_catalog_sha256": file_sha256(reference_path),
            "assay_url_template": PROTEINGYM_ASSAY_URL_TEMPLATE,
            "access_date": None,
            "license_id": None,
            "redistribution_status": "unresolved",
        },
        "assay_contracts": assay_contracts,
        "target_contract": {
            "continuous_source_field": "DMS_score",
            "continuous_endpoint": ("protein_gym_direction_normalized_function_score"),
            "continuous_orientation": "higher_is_more_functional",
            "binary_source_field": "DMS_score_bin",
            "binary_functional_value": 1,
            "target_field": "target_label",
            "target_rule": "DMS_score_bin",
            "target_label_0_semantics": ("damaging_or_loss_of_function_under_assay_threshold"),
            "target_label_1_semantics": ("retained_or_neutral_function_under_assay_threshold"),
            "missing_outcome_policy": "reject",
        },
        "variant_contract": {
            "variant_type": "single_amino_acid_substitution",
            "coordinate_system": "protein_1_based",
            "reference_match_required": True,
            "allow_multiple_mutants": False,
            "allow_synonymous": False,
        },
        "grouping_contract": {
            "split_group_scope": "protein_identity_proxy_nonconfirmatory",
            "family_metadata_status": "unavailable",
            "family_map_sha256": None,
            "intervention_pair_definition": ("assay_id+single_amino_acid_substitution+assay_wild_type"),
            "require_non_null_pair": True,
            "minimum_confirmatory_groups": 8,
        },
        "analysis_contract": {
            "primary_metric": "roc_auc",
            "uncertainty_method": "protein_group_bootstrap",
            "minimum_items": 30,
            "minimum_groups": 8,
            "minimum_bootstrap_draws": 1000,
        },
        "transformation_contract": {
            "code_path": "eval/dms_contract.py",
            "code_sha256": file_sha256(Path(__file__)),
            "selection_policy": "all_valid_source_rows",
            "random_seed": None,
        },
        "metadata_status": unavailable_metadata,
    }
    preregistration_digest = preregistration_sha256(preregistration)

    assay_by_id = {assay["assay_id"]: assay for assay in assay_contracts}
    input_records: list[dict[str, Any]] = []
    outcome_records: list[dict[str, Any]] = []
    uncertainty_reason = "Aggregate ProteinGym files do not include per-variant uncertainty."
    replicate_reason = unavailable_metadata["replicates"]["reason"]
    for assay_id in sorted(raw_records):
        assay = assay_by_id[assay_id]
        sequence = assay["reference_sequence"]
        for mutation, score, functional_bin in raw_records[assay_id]:
            _, position, alternate = _parse_mutation(
                mutation,
                f"{assay_id} mutation",
            )
            mutant_sequence = sequence[: position - 1] + alternate + sequence[position:]
            item_id = f"{assay_id}:{mutation}"
            input_records.append(
                {
                    "item_id": item_id,
                    "entity_id": f"{assay['protein_entry']}:{mutation}",
                    "task_family_id": "dms_assay_variant_function",
                    "source_dataset": "ProteinGym",
                    "assay_id": assay_id,
                    "source_record_id": mutation,
                    "protein_entry": assay["protein_entry"],
                    "construct_id": f"{assay_id}:reference_construct",
                    "reference_sequence_sha256": (assay["reference_sequence_sha256"]),
                    "mutation": mutation,
                    "mutant_sequence_sha256": sequence_sha256(mutant_sequence),
                    "representation_kind": "full_mutant_protein_sequence",
                    "representation_sha256": sequence_sha256(mutant_sequence),
                    "variant_type": "single_amino_acid_substitution",
                    "condition_id": "mutant",
                    "split_group_id": assay["split_group_id"],
                    "split_group_scope": ("protein_identity_proxy_nonconfirmatory"),
                    "intervention_pair_id": (f"{assay_id}:mutant-vs-wt:{mutation}"),
                }
            )
            outcome_records.append(
                {
                    "item_id": item_id,
                    "assay_id": assay_id,
                    "source_record_id": mutation,
                    "dms_score": score,
                    "dms_score_bin": functional_bin,
                    "target_label": functional_bin,
                    "qc_status": "PASS",
                    "qc_reason_codes": [],
                    "effect_uncertainty": None,
                    "uncertainty_status": "unavailable",
                    "uncertainty_reason": uncertainty_reason,
                    "replicate_metadata_status": "unavailable",
                    "replicate_metadata_reason": replicate_reason,
                    "source_file_sha256": assay["source_file_sha256"],
                }
            )
    input_records.sort(key=lambda record: record["item_id"])
    outcome_records.sort(key=lambda record: record["item_id"])

    input_manifest: dict[str, Any] = {
        "schema_version": DMS_SCHEMA_VERSION,
        "artifact_type": INPUT_MANIFEST_ARTIFACT_TYPE,
        "preregistration_sha256": preregistration_digest,
        "records": input_records,
    }
    replicate_manifest: dict[str, Any] = {
        "schema_version": DMS_SCHEMA_VERSION,
        "artifact_type": REPLICATE_MANIFEST_ARTIFACT_TYPE,
        "preregistration_sha256": preregistration_digest,
        "input_manifest_sha256": input_manifest_sha256(
            input_manifest,
            preregistration,
        ),
        "status": "unavailable_aggregate_only",
        "status_reason": replicate_reason,
        "records": [],
    }
    outcome_manifest: dict[str, Any] = {
        "schema_version": DMS_SCHEMA_VERSION,
        "artifact_type": OUTCOME_MANIFEST_ARTIFACT_TYPE,
        "preregistration_sha256": preregistration_digest,
        "input_manifest_sha256": input_manifest_sha256(
            input_manifest,
            preregistration,
        ),
        "replicate_manifest_sha256": replicate_manifest_sha256(
            replicate_manifest,
            preregistration,
            input_manifest,
        ),
        "target_contract_sha256": canonical_sha256(preregistration["target_contract"]),
        "records": outcome_records,
    }
    release = {
        "preregistration": preregistration,
        "input_manifest": input_manifest,
        "replicate_manifest": replicate_manifest,
        "outcome_manifest": outcome_manifest,
    }
    validate_release(release)
    return release


def build_local_proteingym_pilot_release(
    repository_root: str | Path | None = None,
) -> dict[str, Mapping[str, Any]]:
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    raw_root = root / "variant_grounding" / "data" / "raw"
    return build_proteingym_pilot_release(
        reference_catalog_path=raw_root / "DMS_substitutions_reference.csv",
        assay_paths={assay_id: raw_root / "dms" / f"{assay_id}.csv" for assay_id in LOCAL_PROTEINGYM_ASSAY_IDS},
    )


def write_release(
    output_directory: str | Path,
    release: Mapping[str, Any],
) -> dict[str, Path]:
    validate_release(release)
    output = Path(output_directory)
    paths = {
        "preregistration": output / "dms_preregistration.v1.json",
        "input_manifest": output / "dms_label_free_inputs.v1.json",
        "replicate_manifest": output / "dms_replicates.v1.json",
        "outcome_manifest": output / "dms_outcomes.v1.json",
    }
    write_preregistration(paths["preregistration"], release["preregistration"])
    write_input_manifest(
        paths["input_manifest"],
        release["input_manifest"],
        release["preregistration"],
    )
    write_replicate_manifest(
        paths["replicate_manifest"],
        release["replicate_manifest"],
        release["preregistration"],
        release["input_manifest"],
    )
    write_outcome_manifest(
        paths["outcome_manifest"],
        release["outcome_manifest"],
        release["preregistration"],
        release["input_manifest"],
        release["replicate_manifest"],
    )
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the four-assay nonconfirmatory ProteinGym DMS pilot")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release = build_local_proteingym_pilot_release(args.repository_root)
    paths = write_release(args.out, release)
    summary = {
        "release_mode": release["preregistration"]["release_status"]["mode"],
        "confirmatory_eligible": False,
        "items": len(release["input_manifest"]["records"]),
        "preregistration_sha256": preregistration_sha256(release["preregistration"]),
        "input_manifest_sha256": input_manifest_sha256(
            release["input_manifest"],
            release["preregistration"],
        ),
        "replicate_manifest_sha256": replicate_manifest_sha256(
            release["replicate_manifest"],
            release["preregistration"],
            release["input_manifest"],
        ),
        "outcome_manifest_sha256": outcome_manifest_sha256(
            release["outcome_manifest"],
            release["preregistration"],
            release["input_manifest"],
            release["replicate_manifest"],
        ),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    _main()

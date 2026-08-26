"""Fail-closed uncertainty prerequisite for the CBS low/high-B6 contrast.

This module locally locks a joint-bootstrap route for reconstructing uncertainty
in ``high B6 - low B6`` fitness.  A direct-covariance route remains documented
as a reference schema, but using it now requires a versioned amendment.

The selected joint bootstrap must replay both conditions together through an
authenticated dependency graph.  Shared experimental ancestors are sampled
once per draw and reused, while condition-specific independent descendants are
resampled independently.  Ordinal spreadsheet columns are never treated as
evidence of biological pairing.

Specification validation is deliberately not evidence authentication.  A
schema-valid specification, provenance declaration, or registration receipt
can never make this candidate ready, promote it, or materialize deltas,
confidence intervals, labels, or benchmark outcomes.  A future claim-bearing
adapter must replay every hash-locked native input and independently
authenticate the external registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

from eval import cbs_b6_pair_contract as pair

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "groundbench.dms_cbs_b6_uncertainty_status"
SPECIFICATION_ARTIFACT_TYPE = "groundbench.dms_cbs_b6_uncertainty_specification"
CONTRACT_ID = "mavedb-cbs-low-high-b6-uncertainty-prerequisite-v1"
CLAIM_SCOPE = "candidate_uncertainty_prerequisite_only_no_evidence_authentication_or_outcome_claim"
SPECIFICATION_CLAIM_SCOPE = "future_uncertainty_method_declaration_only_no_evidence_authentication_or_outcome_claim"

DIRECT_COVARIANCE = "direct_covariance"
JOINT_BOOTSTRAP = "joint_bootstrap"
SUPPORTED_METHODS = (DIRECT_COVARIANCE, JOINT_BOOTSTRAP)
SELECTED_METHOD = JOINT_BOOTSTRAP
METHOD_SELECTION_STATUS = "locally_hash_locked_not_externally_registered"
JOINT_BOOTSTRAP_DRAW_COUNT = 10_000
JOINT_BOOTSTRAP_SEED_DERIVATION = (
    "first_64_bits_of_pair_registry_canonical_sha256_serialized_as_"
    "lowercase_fixed_width_16_hex_and_parsed_base16_to_uint64"
)
JOINT_BOOTSTRAP_SEED = pair.EXPECTED_PAIR_REGISTRY_SHA256[:16]

DELTA_FORMULA = "delta_equals_high_b6_score_minus_low_b6_score"
VARIANCE_FORMULA = "variance_delta_equals_variance_high_plus_variance_low_minus_two_times_covariance_high_low"
CI_RULE = "two_sided_95_percent_interval_from_authenticated_joint_uncertainty"
NO_IMPUTATION_RULE = "exclude_missing_required_measurements_never_impute"
OUTCOME_BOUNDARY = "forbidden_in_this_contract_even_when_a_specification_or_registration_envelope_is_schema_valid"

BLOCKER_CODES = (
    "CBS_B6_FULL_UNCERTAINTY_SPECIFICATION_NOT_SUPPLIED",
    "CBS_B6_POST_COUNT_REPLAY_PROVENANCE_NOT_SUPPLIED",
    "CBS_B6_POST_COUNT_CHANNEL_SAMPLE_ROLE_AND_UPSTREAM_BOUNDARY_NOT_AUTHENTICATED",
    "CBS_B6_SAMPLE_ROLE_AND_RESAMPLING_GRAPH_NOT_AUTHENTICATED",
    "CBS_B6_BOOTSTRAP_RUNTIME_MANIFEST_NOT_AUTHENTICATED",
    "CBS_B6_JOINT_BOOTSTRAP_NOT_AUTHENTICATED",
    "CBS_B6_INDEPENDENT_BIOLOGICAL_BLOCK_MINIMUM_NOT_DEMONSTRATED",
    "CBS_B6_BOOTSTRAP_DEGENERACY_AND_QUANTILE_STABILITY_NOT_DEMONSTRATED",
    "CBS_B6_PUBLISHED_REMEDIABILITY_NULL_MEMBERSHIP_FDR_ADJUSTMENT_AND_CI_METHOD_NOT_AUTHENTICATED",
    "CBS_B6_EXTERNAL_REGISTRATION_NOT_AUTHENTICATED",
    "CBS_B6_UNCERTAINTY_MATERIALIZATION_UNSUPPORTED",
    "CBS_B6_OUTCOME_PROMOTION_PROHIBITED",
)

ESTIMAND_FIELDS = frozenset(
    {
        "gene",
        "conditions",
        "contrast_unit",
        "native_join_key",
        "protein_group_key",
        "delta_orientation",
        "delta_formula",
        "variance_formula",
        "confidence_interval_rule",
        "codon_duplicate_policy",
        "missingness_policy",
        "imputation_allowed",
        "primary_endpoint",
        "replicate_pairing_claim",
    }
)
CONDITION_FIELDS = frozenset({"low_urn", "high_urn", "low_dose", "high_dose"})
ARTIFACT_LOCK_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_role",
        "uri",
        "body_bytes",
        "body_sha256",
        "media_type",
    }
)
NATIVE_PROVENANCE_FIELDS = frozenset(
    {
        "replay_manifest",
        "native_input_manifest",
        "low_source_binding",
        "high_source_binding",
        "sample_role_map",
        "tileseq_software_manifest",
        "bootstrap_runtime_manifest",
        "qc_specification",
        "codon_collapse_manifest",
        "functional_anchor_manifest",
        "analysis_population_manifest",
        "resampling_graph",
    }
)
SOURCE_BINDING_FIELDS = frozenset(
    {
        "condition",
        "urn",
        "source_lock_artifact_bytes",
        "source_lock_artifact_sha256",
        "source_lock_canonical_json_sha256",
        "source_bundle_sha256",
    }
)
DIRECT_FIELDS = frozenset({"method", "contract"})
DIRECT_CONTRACT_FIELDS = frozenset(
    {
        "variant_key",
        "native_score_source",
        "low_b6_composite_rule",
        "score_construction_pipeline",
        "matched_biological_unit_id_column",
        "pairing_rule",
        "paired_unit_independence_assumption",
        "paired_difference_distribution_assumption",
        "estimator",
        "ddof",
        "minimum_complete_matched_units",
        "small_n_threshold_complete_units",
        "small_n_limitation",
        "small_n_release_policy",
        "output_columns",
        "unique_variant_coverage",
        "variance_value_rule",
        "covariance_psd_rule",
        "delta_mean_variance_formula",
        "negative_delta_variance_tolerance",
        "negative_delta_variance_policy",
        "confidence_interval_method",
        "required_diagnostics",
        "confidence_interval_release_gate",
        "missingness_rule",
    }
)
BOOTSTRAP_FIELDS = frozenset({"method", "contract"})
BOOTSTRAP_CONTRACT_FIELDS = frozenset(
    {
        "variant_key",
        "experimental_dependency_block_id_column",
        "dependency_rule",
        "graph_node_types",
        "graph_edge_types",
        "shared_node_draw_rule",
        "independent_node_draw_rule",
        "resampling_graph_sha256",
        "native_input_manifest_sha256",
        "bootstrap_runtime_manifest_sha256",
        "native_input_boundary",
        "replay_scope",
        "upstream_steps_not_replayed",
        "inference_scope",
        "independent_block_definition",
        "minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci",
        "minimum_complete_matched_independent_blocks_for_paired_percentile_ci",
        "n_below_minimum_policy",
        "n_equals_2_policy",
        "draw_count",
        "draw_count_support_rule",
        "seed",
        "rng_seed_decode_rule",
        "rng",
        "rng_version_binding",
        "bootstrap_runtime_manifest_requirements",
        "resample_signature_definition",
        "nested_resample_trace_rule",
        "minimum_unique_resample_signatures",
        "maximum_draws_per_resample_signature",
        "quantile_monte_carlo_interval_method",
        "quantile_stability_max_endpoint_mc_interval_width_basis_points_of_reported_ci_width",
        "quantile_stability_gate",
        "low_b6_composite_rule",
        "per_draw_pipeline",
        "invalid_draw_rule",
        "output_columns",
        "unique_variant_draw_coverage",
        "draw_index_column",
        "resampling_unit",
        "interval_extraction",
        "required_diagnostics",
        "confidence_interval_release_gate",
        "missingness_rule",
    }
)
METHOD_SELECTION_FIELDS = frozenset(
    {
        "method",
        "selection_scope",
        "scientific_rationale",
        "draw_count",
        "seed",
        "seed_derivation",
        "resampling_graph_binding_status",
        "fallback_policy",
    }
)
REGISTRATION_FIELDS = frozenset(
    {
        "registration_authority",
        "immutable_record_uri",
        "registered_at_utc",
        "locked_specification_sha256",
        "receipt",
        "authentication_status",
    }
)
SPECIFICATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "specification_id",
        "claim_scope",
        "pair_registry_sha256",
        "estimand",
        "native_replay_provenance",
        "uncertainty",
        "external_registration",
        "outcome_materialization",
        "confirmatory_eligible",
        "automatic_promotion",
    }
)
STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "contract_id",
        "claim_scope",
        "gene",
        "pair_registry_sha256",
        "estimand",
        "supported_methods",
        "method_contract_definition_sha256",
        "method_selection_status",
        "selected_method",
        "method_selection",
        "specification_status",
        "native_replay_provenance_status",
        "uncertainty_evidence_status",
        "external_registration_status",
        "readiness",
        "admission_status",
        "outcome_status",
        "delta_values_materialized",
        "confidence_intervals_materialized",
        "labels_materialized",
        "confirmatory_eligible",
        "automatic_promotion",
        "active_blocker_codes",
        "status_sha256",
    }
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CbsB6UncertaintyError(ValueError):
    """Raised when the CBS B6 uncertainty prerequisite fails closed."""


def _json_scalar(value: Any, context: str) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    raise CbsB6UncertaintyError(f"{context} must use strict JSON scalar types")


def _strict_json(value: Any, context: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise CbsB6UncertaintyError(f"{context} JSON object keys must be strings")
            _strict_json(item, f"{context}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _strict_json(item, f"{context}[{index}]")
        return
    _json_scalar(value, context)


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict, deterministic canonical JSON bytes."""

    _strict_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CbsB6UncertaintyError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest over strict canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CbsB6UncertaintyError(f"{context} must be a JSON object")
    return value


def _exact_mapping(
    value: Any,
    fields: frozenset[str] | set[str],
    context: str,
) -> Mapping[str, Any]:
    result = _mapping(value, context)
    if not all(type(key) is str for key in result):
        raise CbsB6UncertaintyError(f"{context} keys must be strings")
    observed = set(result)
    expected = set(fields)
    if observed != expected:
        raise CbsB6UncertaintyError(
            f"{context} must use the exact schema; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return result


def _exact(observed: Any, expected: Any, context: str) -> None:
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise CbsB6UncertaintyError(f"{context} differs")


def _string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise CbsB6UncertaintyError(f"{context} must be a nonempty string")
    return value


def _identifier(value: Any, context: str) -> str:
    value = _string(value, context)
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise CbsB6UncertaintyError(f"{context} must be a conservative identifier")
    return value


def _sha256(value: Any, context: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise CbsB6UncertaintyError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 1:
        raise CbsB6UncertaintyError(f"{context} must be a positive integer")
    return value


def _https_uri(value: Any, context: str) -> str:
    value = _string(value, context)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise CbsB6UncertaintyError(f"{context} must be an absolute HTTPS URI without credentials or fragment")
    return value


def _utc(value: Any, context: str) -> str:
    value = _string(value, context)
    if UTC_PATTERN.fullmatch(value) is None:
        raise CbsB6UncertaintyError(f"{context} must be second-precision UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise CbsB6UncertaintyError(f"{context} must be a valid UTC timestamp") from exc
    return value


def _json_object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CbsB6UncertaintyError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, context: str) -> Mapping[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CbsB6UncertaintyError(f"cannot read {context}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object_hook)
    except CbsB6UncertaintyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CbsB6UncertaintyError(f"{context} must be duplicate-free UTF-8 JSON") from exc
    _strict_json(value, context)
    return _mapping(value, context)


def _expected_estimand() -> dict[str, Any]:
    return {
        "gene": pair.GENE,
        "conditions": {
            "low_urn": pair.LOW_URN,
            "high_urn": pair.HIGH_URN,
            "low_dose": pair.LOW_DOSE_DEFINITION,
            "high_dose": pair.HIGH_DOSE_DEFINITION,
        },
        "contrast_unit": pair.CONTRAST_UNIT,
        "native_join_key": list(pair.NATIVE_JOIN_KEY),
        "protein_group_key": pair.PROTEIN_GROUP_KEY,
        "delta_orientation": pair.DELTA_ORIENTATION,
        "delta_formula": DELTA_FORMULA,
        "variance_formula": VARIANCE_FORMULA,
        "confidence_interval_rule": CI_RULE,
        "codon_duplicate_policy": pair.CODON_DUPLICATE_POLICY,
        "missingness_policy": NO_IMPUTATION_RULE,
        "imputation_allowed": False,
        "primary_endpoint": pair.PRIMARY_ENDPOINT,
        "replicate_pairing_claim": False,
    }


def validate_estimand(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the one admissible CBS B6 uncertainty estimand."""

    value = _exact_mapping(value, ESTIMAND_FIELDS, "CBS B6 estimand")
    _exact_mapping(value["conditions"], CONDITION_FIELDS, "CBS B6 estimand.conditions")
    _exact(value, _expected_estimand(), "CBS B6 estimand")
    if value["imputation_allowed"] is not False or value["replicate_pairing_claim"] is not False:
        raise CbsB6UncertaintyError("CBS B6 estimand boolean boundaries differ")
    return value


def _validate_artifact_lock(
    value: Any,
    *,
    expected_role: str,
    context: str,
) -> Mapping[str, Any]:
    value = _exact_mapping(value, ARTIFACT_LOCK_FIELDS, context)
    _identifier(value["artifact_id"], f"{context}.artifact_id")
    _exact(value["artifact_role"], expected_role, f"{context}.artifact_role")
    _https_uri(value["uri"], f"{context}.uri")
    _positive_int(value["body_bytes"], f"{context}.body_bytes")
    _sha256(value["body_sha256"], f"{context}.body_sha256")
    media_type = _string(value["media_type"], f"{context}.media_type")
    if media_type not in {"application/json", "text/csv", "application/x-parquet"}:
        raise CbsB6UncertaintyError(f"{context}.media_type is unsupported")
    return value


def _validate_source_binding(
    value: Any,
    *,
    condition: str,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    context = f"CBS B6 native replay provenance.{condition}_source_binding"
    value = _exact_mapping(value, SOURCE_BINDING_FIELDS, context)
    conditions = _mapping(pair_registry["conditions"], "CBS B6 pair registry.conditions")
    registry_condition = _mapping(
        conditions[condition],
        f"CBS B6 pair registry.conditions.{condition}",
    )
    source_lock = _mapping(
        registry_condition["source_lock"],
        f"CBS B6 pair registry.conditions.{condition}.source_lock",
    )
    _exact(
        value,
        {
            "condition": condition,
            "urn": registry_condition["urn"],
            "source_lock_artifact_bytes": source_lock["artifact_bytes"],
            "source_lock_artifact_sha256": source_lock["artifact_sha256"],
            "source_lock_canonical_json_sha256": source_lock["canonical_json_sha256"],
            "source_bundle_sha256": source_lock["source_bundle_sha256"],
        },
        context,
    )
    _positive_int(
        value["source_lock_artifact_bytes"],
        f"{context}.source_lock_artifact_bytes",
    )
    for field in (
        "source_lock_artifact_sha256",
        "source_lock_canonical_json_sha256",
        "source_bundle_sha256",
    ):
        _sha256(value[field], f"{context}.{field}")
    return value


def validate_native_replay_provenance(
    value: Mapping[str, Any],
    *,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate pre-outcome native-replay declarations and source bindings."""

    value = _exact_mapping(
        value,
        NATIVE_PROVENANCE_FIELDS,
        "CBS B6 native replay provenance",
    )
    _validate_source_binding(
        value["low_source_binding"],
        condition="low",
        pair_registry=pair_registry,
    )
    _validate_source_binding(
        value["high_source_binding"],
        condition="high",
        pair_registry=pair_registry,
    )
    roles = {
        "replay_manifest": "cbs_b6_native_replay_manifest",
        "native_input_manifest": "cbs_b6_native_input_manifest",
        "sample_role_map": "cbs_b6_sample_role_map",
        "tileseq_software_manifest": "cbs_b6_tileseq_software_manifest",
        "bootstrap_runtime_manifest": "cbs_b6_bootstrap_runtime_manifest",
        "qc_specification": "cbs_b6_qc_specification",
        "codon_collapse_manifest": "cbs_b6_codon_collapse_manifest",
        "functional_anchor_manifest": "cbs_b6_functional_anchor_manifest",
        "analysis_population_manifest": "cbs_b6_analysis_population_manifest",
        "resampling_graph": "cbs_b6_resampling_graph",
    }
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    seen_uris: set[str] = set()
    for field, role in roles.items():
        artifact = _validate_artifact_lock(
            value[field],
            expected_role=role,
            context=f"CBS B6 native replay provenance.{field}",
        )
        artifact_id = artifact["artifact_id"]
        digest = artifact["body_sha256"]
        uri = artifact["uri"]
        if artifact_id in seen_ids or digest in seen_digests or uri in seen_uris:
            raise CbsB6UncertaintyError(
                "CBS B6 native replay provenance artifacts must have distinct IDs, URIs, and digests"
            )
        seen_ids.add(artifact_id)
        seen_digests.add(digest)
        seen_uris.add(uri)
    return value


def _expected_direct_contract() -> dict[str, Any]:
    return {
        "variant_key": list(pair.NATIVE_JOIN_KEY),
        "native_score_source": (
            "per_biological_unit_scores_recomputed_from_authenticated_post_count_depth_"
            "normalized_relative_allele_frequency_channels_never_deposited_aggregate_score_sd_se"
        ),
        "low_b6_composite_rule": (
            "recompute_average_of_0_and_1_ng_per_mL_pyridoxine_conditions_within_each_"
            "matched_biological_unit_before_covariance"
        ),
        "score_construction_pipeline": (
            "replay_post_count_tileseq_qc_error_control_codon_collapse_functional_anchor_"
            "and_analysis_population_pipeline_before_covariance"
        ),
        "matched_biological_unit_id_column": "matched_biological_unit_id",
        "pairing_rule": ("exact_authenticated_sample_role_map_low_high_linkage_only_never_infer_pairing"),
        "paired_unit_independence_assumption": (
            "matched_biological_units_are_independent_across_unit_ids_within_each_variant"
        ),
        "paired_difference_distribution_assumption": (
            "complete_matched_unit_high_minus_low_differences_are_approximately_"
            "student_t_compatible_for_two_sided_interval_construction"
        ),
        "estimator": "unbiased_sample_variance_and_covariance_across_complete_matched_units",
        "ddof": 1,
        "minimum_complete_matched_units": 2,
        "small_n_threshold_complete_units": 8,
        "small_n_limitation": (
            "fewer_than_8_complete_matched_units_yields_unstable_variance_distribution_"
            "diagnostics_and_student_t_intervals"
        ),
        "small_n_release_policy": (
            "when_n_complete_is_below_8_require_predeclared_paired_difference_distribution_"
            "diagnostic_to_pass_otherwise_no_interval_or_outcome"
        ),
        "output_columns": [
            "hgvs_nt",
            "n_complete_matched_units",
            "mean_low_b6",
            "mean_high_b6",
            "delta_high_minus_low",
            "sample_variance_low_b6",
            "sample_variance_high_b6",
            "sample_covariance_high_low",
            "variance_delta_mean",
            "standard_error_delta_mean",
            "ci95_lower",
            "ci95_upper",
            "diagnostics_pass",
        ],
        "unique_variant_coverage": ("exactly_one_output_row_for_every_analysis_population_hgvs_nt"),
        "variance_value_rule": "finite_nonnegative_sample_variances_required",
        "covariance_psd_rule": (
            "finite_covariance_and_absolute_covariance_not_greater_than_sqrt_variance_low_times_variance_high"
        ),
        "delta_mean_variance_formula": (
            "variance_delta_mean_equals_sample_variance_high_plus_sample_variance_low_"
            "minus_two_times_sample_covariance_high_low_all_divided_by_n_complete_matched_units"
        ),
        "negative_delta_variance_tolerance": ("absolute_1e-12_normalized_fitness_score_squared"),
        "negative_delta_variance_policy": ("reject_below_negative_tolerance_clip_within_tolerance_to_zero_and_flag"),
        "confidence_interval_method": ("paired_student_t_two_sided_95_percent_df_n_complete_minus_1"),
        "required_diagnostics": [
            "authenticated_sample_map_linkage",
            "authenticated_post_count_native_input_manifest",
            "upstream_nonreplay_scope_disclosed",
            "post_count_pipeline_replayed",
            "paired_unit_independence_from_sample_map",
            "paired_difference_distribution_diagnostic",
            "small_n_limitation_gate",
            "minimum_complete_matched_units",
            "finite_values",
            "nonnegative_variances",
            "covariance_cauchy_psd",
            "nonnegative_delta_mean_variance_after_tolerance_policy",
            "exact_unique_hgvs_nt_coverage",
        ],
        "confidence_interval_release_gate": ("all_required_diagnostics_must_pass_otherwise_no_interval_or_outcome"),
        "missingness_rule": NO_IMPUTATION_RULE,
    }


def _validate_direct_covariance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _exact_mapping(value, DIRECT_FIELDS, "CBS B6 direct covariance specification")
    _exact(value["method"], DIRECT_COVARIANCE, "CBS B6 direct covariance method")
    contract = _exact_mapping(
        value["contract"],
        DIRECT_CONTRACT_FIELDS,
        "CBS B6 direct covariance contract",
    )
    _exact(contract, _expected_direct_contract(), "CBS B6 direct covariance contract")
    return value


def _expected_bootstrap_contract(
    *,
    resampling_graph_sha256: str,
    native_input_manifest_sha256: str,
    bootstrap_runtime_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "resampling_graph_sha256": resampling_graph_sha256,
        "native_input_manifest_sha256": native_input_manifest_sha256,
        "bootstrap_runtime_manifest_sha256": bootstrap_runtime_manifest_sha256,
        "native_input_boundary": "authenticated_post_count_depth_normalized_relative_allele_frequency_channels",
        "replay_scope": "post_count_tileseq_score_reconstruction_only",
        "upstream_steps_not_replayed": [
            "raw_fastq_ingestion",
            "paired_read_agreement",
            "alignment_and_variant_calling",
            "integer_allele_count_generation",
            "sequencing_depth_normalization",
        ],
        "inference_scope": ("assay_condition_response_generalized_over_authenticated_biological_culture_blocks"),
        "independent_block_definition": (
            "connected_component_of_biological_culture_roots_after_collapsing_shared_"
            "ancestry_and_all_technical_library_run_tile_descendants"
        ),
        "minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci": 8,
        "minimum_complete_matched_independent_blocks_for_paired_percentile_ci": 8,
        "n_below_minimum_policy": (
            "NOT_ADJUDICATED_no_ci_no_threshold_label_no_outcome_descriptive_point_estimate_only"
        ),
        "n_equals_2_policy": "categorically_prohibit_claim_bearing_percentile_ci",
        "draw_count": JOINT_BOOTSTRAP_DRAW_COUNT,
        "draw_count_support_rule": ("monte_carlo_draws_never_count_as_independent_experimental_blocks"),
        "seed": JOINT_BOOTSTRAP_SEED,
        "variant_key": list(pair.NATIVE_JOIN_KEY),
        "experimental_dependency_block_id_column": "experimental_dependency_block_id",
        "dependency_rule": (
            "use_only_authenticated_sample_role_and_resampling_graph_edges_never_infer_"
            "cross_condition_pairing_from_ordinal_replicate_labels_or_column_order"
        ),
        "graph_node_types": [
            "biological_culture",
            "sequencing_library",
            "sequencing_run",
            "tile",
            "condition_branch",
            "shared_preselection_pool",
            "wildtype_error_control",
        ],
        "graph_edge_types": [
            "derived_from",
            "technical_repeat_of",
            "biological_replicate_of",
            "shared_across_conditions",
            "condition_specific_branch",
        ],
        "shared_node_draw_rule": (
            "sample_each_authenticated_shared_ancestor_once_per_draw_and_reuse_it_for_all_descendant_condition_branches"
        ),
        "independent_node_draw_rule": (
            "resample_authenticated_condition_specific_independent_descendants_separately_within_the_same_joint_draw"
        ),
        "rng_seed_decode_rule": ("parse_locked_lowercase_fixed_width_16_hex_as_unsigned_base16_64_bit_integer"),
        "rng": "numpy_random_generator_pcg64dxsm",
        "rng_version_binding": "exact_bootstrap_runtime_manifest_body_sha256",
        "bootstrap_runtime_manifest_requirements": [
            "bootstrap_runner_source_sha256",
            "python_implementation_and_exact_version",
            "numpy_exact_version_and_wheel_or_build_sha256",
            "numpy_random_PCG64DXSM",
            "seed_parse_int_fixed_width_hex_base16",
            "exact_generator_construction",
            "sampling_api_arguments_dtype_and_replacement_semantics",
            "canonical_dag_traversal_and_utf8_bytewise_id_tiebreak",
            "draw_consumption_order",
            "output_serialization",
            "platform_architecture_and_endianness",
            "immutable_environment_or_container_digest",
        ],
        "resample_signature_definition": (
            "canonical_sha256_of_top_level_biological_root_multiplicity_vector_in_canonical_root_id_order"
        ),
        "nested_resample_trace_rule": (
            "record_all_library_run_tile_draw_indices_separately_never_use_nested_"
            "diversity_to_satisfy_top_level_root_signature_gate"
        ),
        "minimum_unique_resample_signatures": 1_000,
        "maximum_draws_per_resample_signature": 100,
        "quantile_monte_carlo_interval_method": ("exact_binomial_order_statistic_95_percent_rank_interval"),
        "quantile_stability_max_endpoint_mc_interval_width_basis_points_of_reported_ci_width": 500,
        "quantile_stability_gate": (
            "reported_ci_width_must_be_strictly_positive_and_each_q0.025_q0.975_"
            "rank_interval_width_must_be_at_most_500_basis_points_of_reported_ci_width"
        ),
        "low_b6_composite_rule": (
            "recompute_average_of_0_and_1_ng_per_mL_pyridoxine_condition_branches_"
            "within_every_draw_after_replaying_shared_and_independent_dependencies"
        ),
        "per_draw_pipeline": (
            "replay_post_count_tileseq_qc_error_control_codon_collapse_functional_anchor_"
            "and_analysis_population_pipeline_within_every_draw"
        ),
        "invalid_draw_rule": (
            "reject_entire_variant_interval_if_any_draw_fails_pipeline_or_required_"
            "diagnostics_never_drop_or_replace_invalid_draws"
        ),
        "output_columns": [
            "hgvs_nt",
            "bootstrap_draw",
            "low_b6_score",
            "high_b6_score",
            "delta_high_minus_low",
            "diagnostics_pass",
        ],
        "unique_variant_draw_coverage": ("exactly_draw_count_rows_for_every_analysis_population_hgvs_nt"),
        "draw_index_column": "bootstrap_draw",
        "resampling_unit": ("authenticated_experimental_dependency_block_never_individual_observation_channel"),
        "interval_extraction": ("empirical_two_sided_percentile_95_q0.025_q0.975_linear_interpolation"),
        "required_diagnostics": [
            "authenticated_sample_map_linkage",
            "authenticated_post_count_native_input_manifest",
            "upstream_nonreplay_scope_disclosed",
            "bootstrap_runtime_manifest_hash_binding",
            "authenticated_resampling_graph",
            "acyclic_dependency_graph",
            "complete_node_role_and_edge_coverage",
            "no_inferred_cross_condition_pairing",
            "exact_effective_independent_block_counts_by_condition_branch",
            "biological_root_component_non_aliasing",
            "minimum_effective_independent_block_count_gate",
            "draw_count_not_counted_as_independent_support",
            "shared_ancestors_reused_once_per_draw",
            "independent_descendants_resampled_separately",
            "minimum_unique_resample_signature_gate",
            "maximum_resample_signature_frequency_gate",
            "quantile_endpoint_monte_carlo_stability_gate",
            "post_count_pipeline_replayed_per_draw",
            "all_draws_valid",
            "finite_delta_draws",
            "exact_unique_hgvs_nt_draw_coverage",
        ],
        "confidence_interval_release_gate": ("all_required_diagnostics_must_pass_otherwise_no_interval_or_outcome"),
        "missingness_rule": NO_IMPUTATION_RULE,
    }


def _expected_method_selection() -> dict[str, Any]:
    return {
        "method": SELECTED_METHOD,
        "selection_scope": (
            "local_method_and_rng_lock_only_not_full_specification_evidence_authentication_or_external_registration"
        ),
        "scientific_rationale": (
            "nonlinear_filtering_error_regularization_codon_collapse_anchor_scaling_"
            "and_shared_input_dependencies_require_joint_post_count_full_score_"
            "construction_pipeline_resampling"
        ),
        "draw_count": JOINT_BOOTSTRAP_DRAW_COUNT,
        "seed": JOINT_BOOTSTRAP_SEED,
        "seed_derivation": JOINT_BOOTSTRAP_SEED_DERIVATION,
        "resampling_graph_binding_status": "not_supplied",
        "fallback_policy": (
            "no_silent_fallback_to_direct_covariance_zero_covariance_or_independent_"
            "condition_resampling_versioned_amendment_required"
        ),
    }


def method_contract_definition_sha256() -> str:
    """Hash the exact supported method definitions and dynamic-field rules."""

    definition = {
        "schema_version": SCHEMA_VERSION,
        "selected_method": SELECTED_METHOD,
        "method_selection": _expected_method_selection(),
        "direct_covariance": {
            "contract": _expected_direct_contract(),
            "dynamic_field_rules": {},
            "selection_status": "reference_only_versioned_amendment_required",
        },
        "joint_bootstrap": {
            "contract_template": _expected_bootstrap_contract(
                resampling_graph_sha256="0" * 64,
                native_input_manifest_sha256="0" * 64,
                bootstrap_runtime_manifest_sha256="0" * 64,
            ),
            "dynamic_field_rules": {
                "resampling_graph_sha256": ("exactly_equal_native_replay_provenance_resampling_graph_body_sha256"),
                "native_input_manifest_sha256": (
                    "exactly_equal_native_replay_provenance_native_input_manifest_body_sha256"
                ),
                "bootstrap_runtime_manifest_sha256": (
                    "exactly_equal_native_replay_provenance_bootstrap_runtime_manifest_body_sha256"
                ),
                "draw_count": f"exactly_{JOINT_BOOTSTRAP_DRAW_COUNT}",
                "seed": (
                    "exact_lowercase_fixed_width_16_hex_first_64_bits_of_"
                    "pair_registry_canonical_sha256_parse_base16_to_uint64"
                ),
            },
        },
    }
    return canonical_sha256(definition)


def _validate_joint_bootstrap(value: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _exact_mapping(value, BOOTSTRAP_FIELDS, "CBS B6 joint bootstrap specification")
    _exact(value["method"], JOINT_BOOTSTRAP, "CBS B6 joint bootstrap method")
    contract = _exact_mapping(
        value["contract"],
        BOOTSTRAP_CONTRACT_FIELDS,
        "CBS B6 joint bootstrap contract",
    )
    graph_sha256 = _sha256(
        contract["resampling_graph_sha256"],
        "CBS B6 joint bootstrap contract.resampling_graph_sha256",
    )
    native_input_manifest_sha256 = _sha256(
        contract["native_input_manifest_sha256"],
        "CBS B6 joint bootstrap contract.native_input_manifest_sha256",
    )
    bootstrap_runtime_manifest_sha256 = _sha256(
        contract["bootstrap_runtime_manifest_sha256"],
        "CBS B6 joint bootstrap contract.bootstrap_runtime_manifest_sha256",
    )
    draw_count = _positive_int(
        contract["draw_count"],
        "CBS B6 joint bootstrap contract.draw_count",
    )
    if type(contract["seed"]) is not str or re.fullmatch(r"[0-9a-f]{16}", contract["seed"]) is None:
        raise CbsB6UncertaintyError(
            "CBS B6 joint bootstrap seed must be a lowercase fixed-width 16-character hexadecimal string"
        )
    if draw_count != JOINT_BOOTSTRAP_DRAW_COUNT:
        raise CbsB6UncertaintyError(
            f"CBS B6 joint bootstrap draw_count must equal locked value {JOINT_BOOTSTRAP_DRAW_COUNT}"
        )
    if contract["seed"] != JOINT_BOOTSTRAP_SEED:
        raise CbsB6UncertaintyError(
            "CBS B6 joint bootstrap seed must equal the locked deterministic registry-derived value"
        )
    _exact(
        contract,
        _expected_bootstrap_contract(
            resampling_graph_sha256=graph_sha256,
            native_input_manifest_sha256=native_input_manifest_sha256,
            bootstrap_runtime_manifest_sha256=bootstrap_runtime_manifest_sha256,
        ),
        "CBS B6 joint bootstrap contract",
    )
    return value


def validate_uncertainty_method(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one discriminated uncertainty-method declaration."""

    value = _mapping(value, "CBS B6 uncertainty method")
    method = value.get("method")
    if type(method) is not str:
        raise CbsB6UncertaintyError("CBS B6 uncertainty method discriminator must be a string")
    if method == DIRECT_COVARIANCE:
        return _validate_direct_covariance(value)
    if method == JOINT_BOOTSTRAP:
        return _validate_joint_bootstrap(value)
    raise CbsB6UncertaintyError("unsupported CBS B6 uncertainty method")


def _specification_locked_payload(specification: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(specification))
    registration = dict(_mapping(payload["external_registration"], "external registration"))
    registration["locked_specification_sha256"] = None
    registration["receipt"] = None
    payload["external_registration"] = registration
    return payload


def specification_locked_payload_sha256(specification: Mapping[str, Any]) -> str:
    """Hash a specification with circular registration evidence nulled."""

    return canonical_sha256(_specification_locked_payload(specification))


def _validate_registration(
    value: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _exact_mapping(value, REGISTRATION_FIELDS, "CBS B6 external registration")
    _identifier(
        value["registration_authority"],
        "CBS B6 external registration.registration_authority",
    )
    _https_uri(
        value["immutable_record_uri"],
        "CBS B6 external registration.immutable_record_uri",
    )
    _utc(value["registered_at_utc"], "CBS B6 external registration.registered_at_utc")
    locked = _sha256(
        value["locked_specification_sha256"],
        "CBS B6 external registration.locked_specification_sha256",
    )
    if locked != specification_locked_payload_sha256(specification):
        raise CbsB6UncertaintyError("CBS B6 registration does not bind the locked specification")
    _validate_artifact_lock(
        value["receipt"],
        expected_role="external_registration_receipt",
        context="CBS B6 external registration.receipt",
    )
    _exact(
        value["authentication_status"],
        "declared_not_authenticated_by_this_contract",
        "CBS B6 external registration.authentication_status",
    )
    return value


def _validate_global_artifact_non_aliasing(
    specification: Mapping[str, Any],
) -> None:
    provenance = _mapping(
        specification["native_replay_provenance"],
        "CBS B6 native replay provenance",
    )
    registration = _mapping(
        specification["external_registration"],
        "CBS B6 external registration",
    )
    artifact_fields = (
        "replay_manifest",
        "native_input_manifest",
        "sample_role_map",
        "tileseq_software_manifest",
        "bootstrap_runtime_manifest",
        "qc_specification",
        "codon_collapse_manifest",
        "functional_anchor_manifest",
        "analysis_population_manifest",
        "resampling_graph",
    )
    artifacts = [_mapping(provenance[field], f"CBS B6 native replay provenance.{field}") for field in artifact_fields]
    artifacts.append(
        _mapping(
            registration["receipt"],
            "CBS B6 external registration.receipt",
        )
    )
    for field in ("artifact_id", "uri", "body_sha256"):
        values = [artifact[field] for artifact in artifacts]
        if len(set(values)) != len(values):
            raise CbsB6UncertaintyError(f"CBS B6 provenance and registration receipt {field} values must not alias")
    source_digests: set[str] = set()
    for condition in ("low", "high"):
        binding = _mapping(
            provenance[f"{condition}_source_binding"],
            f"CBS B6 native replay provenance.{condition}_source_binding",
        )
        source_digests.update(
            {
                binding["source_lock_artifact_sha256"],
                binding["source_lock_canonical_json_sha256"],
                binding["source_bundle_sha256"],
            }
        )
    if source_digests & {artifact["body_sha256"] for artifact in artifacts}:
        raise CbsB6UncertaintyError("CBS B6 declared provenance artifacts cannot alias frozen source-lock identities")


def validate_cbs_b6_uncertainty_specification(
    specification: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a future declaration without authenticating its evidence."""

    try:
        pair.validate_cbs_b6_pair_registry(pair_registry)
    except pair.CbsB6PairError as exc:
        raise CbsB6UncertaintyError("CBS B6 pair registry validation failed") from exc
    specification = _exact_mapping(
        specification,
        SPECIFICATION_FIELDS,
        "CBS B6 uncertainty specification",
    )
    _strict_json(specification, "CBS B6 uncertainty specification")
    _exact(specification["schema_version"], SCHEMA_VERSION, "specification.schema_version")
    _exact(
        specification["artifact_type"],
        SPECIFICATION_ARTIFACT_TYPE,
        "specification.artifact_type",
    )
    _identifier(specification["specification_id"], "specification.specification_id")
    _exact(specification["claim_scope"], SPECIFICATION_CLAIM_SCOPE, "specification.claim_scope")
    _exact(
        specification["pair_registry_sha256"],
        pair.EXPECTED_PAIR_REGISTRY_SHA256,
        "specification.pair_registry_sha256",
    )
    validate_estimand(specification["estimand"])
    provenance = validate_native_replay_provenance(
        specification["native_replay_provenance"],
        pair_registry=pair_registry,
    )
    method = validate_uncertainty_method(specification["uncertainty"])
    if method["method"] != SELECTED_METHOD:
        raise CbsB6UncertaintyError(
            "CBS B6 uncertainty method differs from the locally hash-locked joint bootstrap; "
            "a versioned amendment is required"
        )
    contract = _mapping(method["contract"], "CBS B6 joint bootstrap contract")
    resampling_graph = _mapping(
        provenance["resampling_graph"],
        "CBS B6 native replay provenance.resampling_graph",
    )
    if contract["resampling_graph_sha256"] != resampling_graph["body_sha256"]:
        raise CbsB6UncertaintyError("CBS B6 joint bootstrap resampling graph is not bound to provenance")
    native_input_manifest = _mapping(
        provenance["native_input_manifest"],
        "CBS B6 native replay provenance.native_input_manifest",
    )
    if contract["native_input_manifest_sha256"] != native_input_manifest["body_sha256"]:
        raise CbsB6UncertaintyError("CBS B6 joint bootstrap native input manifest is not bound to provenance")
    bootstrap_runtime_manifest = _mapping(
        provenance["bootstrap_runtime_manifest"],
        "CBS B6 native replay provenance.bootstrap_runtime_manifest",
    )
    if contract["bootstrap_runtime_manifest_sha256"] != bootstrap_runtime_manifest["body_sha256"]:
        raise CbsB6UncertaintyError("CBS B6 joint bootstrap runtime manifest is not bound to provenance")
    _validate_registration(specification["external_registration"], specification)
    _validate_global_artifact_non_aliasing(specification)
    _exact(
        specification["outcome_materialization"],
        OUTCOME_BOUNDARY,
        "specification.outcome_materialization",
    )
    if specification["confirmatory_eligible"] is not False:
        raise CbsB6UncertaintyError("uncertainty specification cannot claim confirmatory eligibility")
    if specification["automatic_promotion"] is not False:
        raise CbsB6UncertaintyError("uncertainty specification cannot self-promote")
    return specification


def status_sha256(status: Mapping[str, Any]) -> str:
    """Hash a status without its self-hash field."""

    payload = dict(status)
    payload.pop("status_sha256", None)
    return canonical_sha256(payload)


def build_cbs_b6_uncertainty_status(
    pair_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic method-locked, no-evidence candidate status."""

    try:
        pair.validate_cbs_b6_pair_registry(pair_registry)
    except pair.CbsB6PairError as exc:
        raise CbsB6UncertaintyError("CBS B6 pair registry validation failed") from exc
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "contract_id": CONTRACT_ID,
        "claim_scope": CLAIM_SCOPE,
        "gene": pair.GENE,
        "pair_registry_sha256": pair.EXPECTED_PAIR_REGISTRY_SHA256,
        "estimand": _expected_estimand(),
        "supported_methods": list(SUPPORTED_METHODS),
        "method_contract_definition_sha256": method_contract_definition_sha256(),
        "method_selection_status": METHOD_SELECTION_STATUS,
        "selected_method": SELECTED_METHOD,
        "method_selection": _expected_method_selection(),
        "specification_status": "not_supplied",
        "native_replay_provenance_status": "not_supplied",
        "uncertainty_evidence_status": "not_supplied",
        "external_registration_status": "not_supplied",
        "readiness": False,
        "admission_status": "candidate_not_ingested",
        "outcome_status": "not_derived",
        "delta_values_materialized": False,
        "confidence_intervals_materialized": False,
        "labels_materialized": False,
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "active_blocker_codes": list(BLOCKER_CODES),
        "status_sha256": "",
    }
    status["status_sha256"] = status_sha256(status)
    validate_cbs_b6_uncertainty_status(status, pair_registry=pair_registry)
    return status


def validate_cbs_b6_uncertainty_status(
    status: Mapping[str, Any],
    *,
    pair_registry: Mapping[str, Any],
) -> None:
    """Validate the only status this prerequisite contract can emit."""

    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "contract_id": CONTRACT_ID,
        "claim_scope": CLAIM_SCOPE,
        "gene": pair.GENE,
        "pair_registry_sha256": pair.EXPECTED_PAIR_REGISTRY_SHA256,
        "estimand": _expected_estimand(),
        "supported_methods": list(SUPPORTED_METHODS),
        "method_contract_definition_sha256": method_contract_definition_sha256(),
        "method_selection_status": METHOD_SELECTION_STATUS,
        "selected_method": SELECTED_METHOD,
        "method_selection": _expected_method_selection(),
        "specification_status": "not_supplied",
        "native_replay_provenance_status": "not_supplied",
        "uncertainty_evidence_status": "not_supplied",
        "external_registration_status": "not_supplied",
        "readiness": False,
        "admission_status": "candidate_not_ingested",
        "outcome_status": "not_derived",
        "delta_values_materialized": False,
        "confidence_intervals_materialized": False,
        "labels_materialized": False,
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "active_blocker_codes": list(BLOCKER_CODES),
    }
    try:
        pair.validate_cbs_b6_pair_registry(pair_registry)
    except pair.CbsB6PairError as exc:
        raise CbsB6UncertaintyError("CBS B6 pair registry validation failed") from exc
    status = _exact_mapping(status, STATUS_FIELDS, "CBS B6 uncertainty status")
    _strict_json(status, "CBS B6 uncertainty status")
    _exact_mapping(
        status["method_selection"],
        METHOD_SELECTION_FIELDS,
        "CBS B6 uncertainty status.method_selection",
    )
    for field, expected_value in expected.items():
        _exact(status[field], expected_value, f"CBS B6 uncertainty status.{field}")
    for field in (
        "readiness",
        "delta_values_materialized",
        "confidence_intervals_materialized",
        "labels_materialized",
        "confirmatory_eligible",
        "automatic_promotion",
    ):
        if status[field] is not False:
            raise CbsB6UncertaintyError(f"CBS B6 uncertainty status.{field} must be false")
    observed_hash = _sha256(status["status_sha256"], "CBS B6 uncertainty status.status_sha256")
    if observed_hash != status_sha256(status):
        raise CbsB6UncertaintyError("CBS B6 uncertainty status self-hash differs")


def is_cbs_b6_uncertainty_ready(
    pair_registry: Mapping[str, Any],
    *,
    specification: Mapping[str, Any] | None = None,
    native_inputs: Mapping[str, bytes] | None = None,
) -> bool:
    """Validate declarations when supplied and always return ``False``."""

    try:
        pair.validate_cbs_b6_pair_registry(pair_registry)
    except pair.CbsB6PairError as exc:
        raise CbsB6UncertaintyError("CBS B6 pair registry validation failed") from exc
    if specification is not None:
        validate_cbs_b6_uncertainty_specification(specification, pair_registry)
    if native_inputs is not None:
        if not isinstance(native_inputs, Mapping):
            raise TypeError("native_inputs must be a mapping")
        for key, value in native_inputs.items():
            if type(key) is not str or type(value) is not bytes:
                raise TypeError("native_inputs must map strings to exact bytes")
    return False


def require_cbs_b6_uncertainty_ready(
    pair_registry: Mapping[str, Any],
    *,
    specification: Mapping[str, Any] | None = None,
    native_inputs: Mapping[str, bytes] | None = None,
) -> NoReturn:
    """Always reject readiness in this prerequisite-only contract."""

    is_cbs_b6_uncertainty_ready(
        pair_registry,
        specification=specification,
        native_inputs=native_inputs,
    )
    raise CbsB6UncertaintyError(
        "CBS B6 uncertainty readiness requires a future evidence-authenticating post-count replay adapter"
    )


def materialize_cbs_b6_uncertainty(
    pair_registry: Mapping[str, Any],
    *,
    specification: Mapping[str, Any] | None = None,
    native_inputs: Mapping[str, bytes] | None = None,
) -> NoReturn:
    """Always reject delta, interval, label, and outcome materialization."""

    require_cbs_b6_uncertainty_ready(
        pair_registry,
        specification=specification,
        native_inputs=native_inputs,
    )


def build_cbs_b6_uncertainty_outcomes(
    pair_registry: Mapping[str, Any],
    *,
    specification: Mapping[str, Any] | None = None,
    native_inputs: Mapping[str, bytes] | None = None,
) -> NoReturn:
    """Explicit always-rejecting alias for outcome builders."""

    return materialize_cbs_b6_uncertainty(
        pair_registry,
        specification=specification,
        native_inputs=native_inputs,
    )


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    replace: bool,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise CbsB6UncertaintyError("CBS B6 uncertainty artifacts must use .json")
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
                raise CbsB6UncertaintyError(f"CBS B6 uncertainty artifact already exists: {destination}") from exc
            os.unlink(temporary_name)
        _fsync_parent(destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cbs_b6_uncertainty_status(
    path: str | Path,
    status: Mapping[str, Any],
    *,
    pair_registry: Mapping[str, Any],
    replace: bool = False,
) -> Path:
    """Atomically write an exact uncertainty status."""

    validate_cbs_b6_uncertainty_status(status, pair_registry=pair_registry)
    return _atomic_write_json(path, status, replace=replace)


def load_cbs_b6_uncertainty_status(
    path: str | Path,
    *,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load and validate a duplicate-free uncertainty status."""

    status = _load_json(path, "CBS B6 uncertainty status")
    validate_cbs_b6_uncertainty_status(status, pair_registry=pair_registry)
    return status


def load_cbs_b6_uncertainty_specification(
    path: str | Path,
    *,
    pair_registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load and structurally validate a duplicate-free future specification."""

    specification = _load_json(path, "CBS B6 uncertainty specification")
    return validate_cbs_b6_uncertainty_specification(specification, pair_registry)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Emit the deterministic candidate-only CBS B6 uncertainty status")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--status-out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()

    try:
        registry = pair.load_cbs_b6_pair_registry(arguments.registry)
        status = build_cbs_b6_uncertainty_status(registry)
        write_cbs_b6_uncertainty_status(
            arguments.status_out,
            status,
            pair_registry=registry,
            replace=arguments.replace,
        )
    except (pair.CbsB6PairError, CbsB6UncertaintyError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status_path": str(arguments.status_out),
                "status_sha256": status["status_sha256"],
                "readiness": status["readiness"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()

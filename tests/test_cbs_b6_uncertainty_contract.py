from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval import cbs_b6_pair_contract as pair
from eval import cbs_b6_uncertainty_contract as uncertainty

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "signal" / "dms" / "cbs_b6_pair_registry.v1.json"
STATUS_PATH = ROOT / "signal" / "dms" / "cbs_b6_uncertainty_status.v1.json"


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def _lock(role: str, index: int) -> dict:
    return {
        "artifact_id": f"future-artifact-{index}",
        "artifact_role": role,
        "uri": f"https://evidence.example.org/artifacts/{index}",
        "body_bytes": 100 + index,
        "body_sha256": hashlib.sha256(f"future-artifact-{index}".encode()).hexdigest(),
        "media_type": "application/json",
    }


def _source_binding(registry: dict, condition: str) -> dict:
    record = registry["conditions"][condition]
    source_lock = record["source_lock"]
    return {
        "condition": condition,
        "urn": record["urn"],
        "source_lock_artifact_bytes": source_lock["artifact_bytes"],
        "source_lock_artifact_sha256": source_lock["artifact_sha256"],
        "source_lock_canonical_json_sha256": source_lock["canonical_json_sha256"],
        "source_bundle_sha256": source_lock["source_bundle_sha256"],
    }


def _provenance(registry: dict) -> dict:
    fields = {
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
    result = {field: _lock(role, index) for index, (field, role) in enumerate(fields.items(), start=1)}
    result["low_source_binding"] = _source_binding(registry, "low")
    result["high_source_binding"] = _source_binding(registry, "high")
    return result


def _direct_uncertainty() -> dict:
    return {
        "method": uncertainty.DIRECT_COVARIANCE,
        "contract": uncertainty._expected_direct_contract(),
    }


def _bootstrap_uncertainty(
    resampling_graph_sha256: str,
    native_input_manifest_sha256: str,
    bootstrap_runtime_manifest_sha256: str,
) -> dict:
    return {
        "method": uncertainty.JOINT_BOOTSTRAP,
        "contract": uncertainty._expected_bootstrap_contract(
            resampling_graph_sha256=resampling_graph_sha256,
            native_input_manifest_sha256=native_input_manifest_sha256,
            bootstrap_runtime_manifest_sha256=bootstrap_runtime_manifest_sha256,
        ),
    }


def _specification(
    method: str = uncertainty.JOINT_BOOTSTRAP,
    registry: dict | None = None,
) -> dict:
    registry = _registry() if registry is None else registry
    provenance = _provenance(registry)
    specification = {
        "schema_version": 1,
        "artifact_type": uncertainty.SPECIFICATION_ARTIFACT_TYPE,
        "specification_id": "future-cbs-b6-uncertainty-v1",
        "claim_scope": uncertainty.SPECIFICATION_CLAIM_SCOPE,
        "pair_registry_sha256": pair.EXPECTED_PAIR_REGISTRY_SHA256,
        "estimand": uncertainty._expected_estimand(),
        "native_replay_provenance": provenance,
        "uncertainty": (
            _direct_uncertainty()
            if method == uncertainty.DIRECT_COVARIANCE
            else _bootstrap_uncertainty(
                provenance["resampling_graph"]["body_sha256"],
                provenance["native_input_manifest"]["body_sha256"],
                provenance["bootstrap_runtime_manifest"]["body_sha256"],
            )
        ),
        "external_registration": {
            "registration_authority": "future-external-registry",
            "immutable_record_uri": "https://registry.example.org/records/locked-1",
            "registered_at_utc": "2026-07-26T12:00:00Z",
            "locked_specification_sha256": None,
            "receipt": None,
            "authentication_status": "declared_not_authenticated_by_this_contract",
        },
        "outcome_materialization": uncertainty.OUTCOME_BOUNDARY,
        "confirmatory_eligible": False,
        "automatic_promotion": False,
    }
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )
    specification["external_registration"]["receipt"] = _lock(
        "external_registration_receipt",
        22,
    )
    return specification


def _rehash_registration(specification: dict) -> None:
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )


def test_selected_future_specification_validates_but_never_becomes_ready() -> None:
    registry = _registry()
    specification = _specification()

    assert (
        uncertainty.validate_cbs_b6_uncertainty_specification(
            specification,
            registry,
        )
        is specification
    )
    assert (
        uncertainty.is_cbs_b6_uncertainty_ready(
            registry,
            specification=specification,
            native_inputs={"declared-body": b"not-authenticated"},
        )
        is False
    )
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="future evidence-authenticating"):
        uncertainty.require_cbs_b6_uncertainty_ready(
            registry,
            specification=specification,
            native_inputs={"declared-body": b"not-authenticated"},
        )
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="future evidence-authenticating"):
        uncertainty.materialize_cbs_b6_uncertainty(
            registry,
            specification=specification,
            native_inputs={"declared-body": b"not-authenticated"},
        )
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="future evidence-authenticating"):
        uncertainty.build_cbs_b6_uncertainty_outcomes(
            registry,
            specification=specification,
            native_inputs={"declared-body": b"not-authenticated"},
        )


def test_direct_covariance_requires_a_versioned_method_selection_amendment() -> None:
    registry = _registry()
    specification = _specification(uncertainty.DIRECT_COVARIANCE)

    assert uncertainty.validate_uncertainty_method(specification["uncertainty"]) is specification["uncertainty"]
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="versioned amendment"):
        uncertainty.validate_cbs_b6_uncertainty_specification(
            specification,
            registry,
        )


def test_persisted_candidate_status_is_exact_and_deterministic() -> None:
    registry = _registry()
    built = uncertainty.build_cbs_b6_uncertainty_status(registry)
    persisted = uncertainty.load_cbs_b6_uncertainty_status(
        STATUS_PATH,
        pair_registry=registry,
    )

    assert persisted == built
    assert persisted["readiness"] is False
    assert persisted["method_selection_status"] == uncertainty.METHOD_SELECTION_STATUS
    assert persisted["selected_method"] == uncertainty.JOINT_BOOTSTRAP
    assert persisted["method_selection"] == uncertainty._expected_method_selection()
    assert persisted["method_selection"]["seed"] == pair.EXPECTED_PAIR_REGISTRY_SHA256[:16]
    assert "CBS_B6_UNCERTAINTY_METHOD_NOT_SELECTED" not in persisted["active_blocker_codes"]
    assert "CBS_B6_FULL_UNCERTAINTY_SPECIFICATION_NOT_SUPPLIED" in persisted["active_blocker_codes"]
    assert "CBS_B6_INDEPENDENT_BIOLOGICAL_BLOCK_MINIMUM_NOT_DEMONSTRATED" in (persisted["active_blocker_codes"])
    assert (
        "CBS_B6_PUBLISHED_REMEDIABILITY_NULL_MEMBERSHIP_FDR_ADJUSTMENT_AND_CI_METHOD_NOT_AUTHENTICATED"
        in persisted["active_blocker_codes"]
    )
    assert persisted["specification_status"] == "not_supplied"
    assert persisted["delta_values_materialized"] is False
    assert persisted["confidence_intervals_materialized"] is False
    assert persisted["labels_materialized"] is False
    assert persisted["confirmatory_eligible"] is False
    assert persisted["automatic_promotion"] is False
    assert persisted["status_sha256"] == uncertainty.status_sha256(persisted)


def test_pair_registry_is_exactly_hash_pinned() -> None:
    registry = _registry()
    registry["confirmatory_eligible"] = 0

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="registry validation failed"):
        uncertainty.build_cbs_b6_uncertainty_status(registry)


def test_bootstrap_seed_is_json_interoperable_fixed_width_hex() -> None:
    seed = uncertainty.JOINT_BOOTSTRAP_SEED

    assert isinstance(seed, str)
    assert len(seed) == 16
    assert seed == pair.EXPECTED_PAIR_REGISTRY_SHA256[:16]
    assert int(seed, 16) == 12_860_903_693_372_416_681
    assert int(seed, 16) > 2**53 - 1
    assert uncertainty.canonical_json_bytes({"seed": seed}) == (b'{"seed":"' + seed.encode("ascii") + b'"}')


def test_bootstrap_draws_cannot_inflate_biological_support() -> None:
    specification = _specification()
    contract = specification["uncertainty"]["contract"]

    assert contract["draw_count"] == 10_000
    assert contract["draw_count_support_rule"] == ("monte_carlo_draws_never_count_as_independent_experimental_blocks")
    assert contract["minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci"] == 8
    assert contract["minimum_complete_matched_independent_blocks_for_paired_percentile_ci"] == 8
    assert contract["n_equals_2_policy"] == "categorically_prohibit_claim_bearing_percentile_ci"
    assert contract["minimum_unique_resample_signatures"] == 1_000
    assert contract["maximum_draws_per_resample_signature"] == 100
    assert "top_level_biological_root_multiplicity_vector" in contract["resample_signature_definition"]
    assert "never_use_nested_diversity" in contract["nested_resample_trace_rule"]


def test_coherent_registry_rehash_cannot_hide_type_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    registry["confirmatory_eligible"] = 0
    monkeypatch.setattr(
        pair,
        "EXPECTED_PAIR_REGISTRY_SHA256",
        pair.canonical_sha256(registry),
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="registry validation failed"):
        uncertainty.build_cbs_b6_uncertainty_status(registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readiness", 0),
        ("confirmatory_eligible", 0),
        ("automatic_promotion", 0),
        ("delta_values_materialized", 0),
        ("selected_method", False),
    ],
)
def test_coherent_status_rehash_cannot_change_claim_types(field: str, value: object) -> None:
    registry = _registry()
    status = uncertainty.build_cbs_b6_uncertainty_status(registry)
    status[field] = value
    status["status_sha256"] = uncertainty.status_sha256(status)

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_status(
            status,
            pair_registry=registry,
        )


def test_coherent_status_rehash_cannot_fake_registration_or_promotion() -> None:
    registry = _registry()
    status = uncertainty.build_cbs_b6_uncertainty_status(registry)
    status["method_selection_status"] = "selected"
    status["selected_method"] = uncertainty.DIRECT_COVARIANCE
    status["specification_status"] = "validated"
    status["external_registration_status"] = "authenticated"
    status["readiness"] = True
    status["confirmatory_eligible"] = True
    status["automatic_promotion"] = True
    status["status_sha256"] = uncertainty.status_sha256(status)

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_status(
            status,
            pair_registry=registry,
        )


def test_fake_receipt_digest_or_locked_payload_rejected() -> None:
    registry = _registry()
    specification = _specification()
    specification["external_registration"]["locked_specification_sha256"] = "f" * 64

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="does not bind"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)

    specification = _specification()
    specification["external_registration"]["receipt"]["body_sha256"] = True
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="SHA-256"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_schema_valid_fake_receipt_remains_unauthenticated_and_not_ready() -> None:
    registry = _registry()
    specification = _specification()
    receipt = specification["external_registration"]["receipt"]
    receipt["artifact_id"] = "entirely-self-declared-receipt"
    receipt["uri"] = "https://attacker.example/fake-receipt"
    receipt["body_sha256"] = "a" * 64

    uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)
    assert (
        specification["external_registration"]["authentication_status"] == "declared_not_authenticated_by_this_contract"
    )
    assert uncertainty.is_cbs_b6_uncertainty_ready(registry, specification=specification) is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("uncertainty", "contract", "draw_count"),
            True,
            "positive integer",
        ),
        (
            ("native_replay_provenance", "replay_manifest", "body_bytes"),
            True,
            "positive integer",
        ),
        (
            ("uncertainty", "contract", "seed"),
            False,
            "fixed-width 16-character hexadecimal string",
        ),
    ],
)
def test_specification_rejects_bool_integer_type_confusion(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    registry = _registry()
    specification = _specification(uncertainty.JOINT_BOOTSTRAP)
    target = specification
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match=message):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_method_discriminator_cannot_mix_schemas() -> None:
    registry = _registry()
    specification = _specification()
    specification["uncertainty"]["joint_bootstrap_artifact"] = _lock(
        "cbs_b6_joint_bootstrap_draws",
        30,
    )
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="exact schema"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_preregistered_methods_do_not_lock_future_result_bodies() -> None:
    direct = _specification(uncertainty.DIRECT_COVARIANCE)["uncertainty"]
    bootstrap = _specification(uncertainty.JOINT_BOOTSTRAP)["uncertainty"]

    assert set(direct) == {"method", "contract"}
    assert set(bootstrap) == {"method", "contract"}
    assert "covariance_artifact" not in direct
    assert "joint_bootstrap_artifact" not in bootstrap


@pytest.mark.parametrize(
    "field",
    [
        "urn",
        "source_lock_artifact_bytes",
        "source_lock_artifact_sha256",
        "source_lock_canonical_json_sha256",
        "source_bundle_sha256",
    ],
)
def test_source_bindings_are_exactly_cross_bound_to_pair_registry(field: str) -> None:
    registry = _registry()
    specification = _specification(registry=registry)
    binding = specification["native_replay_provenance"]["low_source_binding"]
    binding[field] = (
        binding[field] + 1 if field == "source_lock_artifact_bytes" else ("f" * 64 if field != "urn" else pair.HIGH_URN)
    )
    _rehash_registration(specification)

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_score_source", "deposited_aggregate_score_sd_se"),
        ("low_b6_composite_rule", "use_deposited_low_b6_aggregate"),
        ("score_construction_pipeline", "skip_error_control_and_functional_anchors"),
        ("matched_biological_unit_id_column", "replicate"),
        ("pairing_rule", "infer_pairing_from_column_order"),
        ("paired_unit_independence_assumption", "assume_all_channels_independent"),
        ("paired_difference_distribution_assumption", "unrestricted"),
        ("estimator", "population_covariance"),
        ("ddof", 0),
        ("minimum_complete_matched_units", 1),
        ("small_n_threshold_complete_units", 2),
        ("small_n_limitation", "none"),
        ("small_n_release_policy", "always_release"),
        ("output_columns", ["hgvs_nt", "covariance"]),
        ("unique_variant_coverage", "allow_partial"),
        ("variance_value_rule", "finite_values"),
        ("covariance_psd_rule", "unchecked"),
        ("delta_mean_variance_formula", uncertainty.VARIANCE_FORMULA),
        ("negative_delta_variance_tolerance", "none"),
        ("negative_delta_variance_policy", "clip_all_negative_values"),
        ("confidence_interval_method", "normal_approximation"),
        ("required_diagnostics", []),
        ("confidence_interval_release_gate", "optional"),
    ],
)
def test_direct_covariance_preregistration_semantics_are_exact(
    field: str,
    value: object,
) -> None:
    method = _direct_uncertainty()
    method["contract"][field] = value

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_uncertainty_method(method)


def test_direct_contract_requires_post_count_pipeline_and_small_n_diagnostics() -> None:
    contract = uncertainty._expected_direct_contract()

    assert "post_count_pipeline_replayed" in contract["required_diagnostics"]
    assert "upstream_nonreplay_scope_disclosed" in contract["required_diagnostics"]
    assert "paired_unit_independence_from_sample_map" in contract["required_diagnostics"]
    assert "paired_difference_distribution_diagnostic" in contract["required_diagnostics"]
    assert "small_n_limitation_gate" in contract["required_diagnostics"]
    assert contract["small_n_threshold_complete_units"] == 8


def test_status_binds_exact_method_contract_definitions() -> None:
    registry = _registry()
    status = uncertainty.build_cbs_b6_uncertainty_status(registry)

    assert status["method_contract_definition_sha256"] == (uncertainty.method_contract_definition_sha256())
    forged = dict(status)
    forged["method_contract_definition_sha256"] = "f" * 64
    forged["status_sha256"] = uncertainty.status_sha256(forged)
    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_status(
            forged,
            pair_registry=registry,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experimental_dependency_block_id_column", "channel", "differs"),
        ("dependency_rule", "infer_pairing_from_column_order", "differs"),
        ("graph_node_types", ["observation_channel"], "differs"),
        ("graph_edge_types", ["paired_by_column_order"], "differs"),
        ("shared_node_draw_rule", "resample_shared_inputs_per_condition", "differs"),
        ("independent_node_draw_rule", "force_cross_condition_pairing", "differs"),
        ("resampling_graph_sha256", "f" * 64, "not bound"),
        ("native_input_manifest_sha256", "f" * 64, "not bound"),
        ("bootstrap_runtime_manifest_sha256", "f" * 64, "not bound"),
        ("native_input_boundary", "raw_fastq_paired_reads", "differs"),
        ("replay_scope", "end_to_end_tileseq_replay", "differs"),
        ("upstream_steps_not_replayed", [], "differs"),
        ("inference_scope", "conditional_on_two_cultures", "differs"),
        (
            "minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci",
            2,
            "differs",
        ),
        (
            "minimum_complete_matched_independent_blocks_for_paired_percentile_ci",
            2,
            "differs",
        ),
        ("n_equals_2_policy", "allow_ci", "differs"),
        ("draw_count", 9_999, "must equal locked value"),
        ("draw_count", 10_001, "must equal locked value"),
        ("seed", -1, "fixed-width 16-character hexadecimal string"),
        ("seed", "0000000000000000", "locked deterministic"),
        ("seed", uncertainty.JOINT_BOOTSTRAP_SEED.upper(), "fixed-width"),
        ("rng", "default_rng", "differs"),
        ("rng_version_binding", "unbound", "differs"),
        ("bootstrap_runtime_manifest_requirements", [], "differs"),
        ("resample_signature_definition", "include_nested_draws", "differs"),
        ("nested_resample_trace_rule", "nested_draws_rescue_root_degeneracy", "differs"),
        ("minimum_unique_resample_signatures", 3, "differs"),
        ("maximum_draws_per_resample_signature", 3_334, "differs"),
        (
            "quantile_stability_max_endpoint_mc_interval_width_basis_points_of_reported_ci_width",
            5_000,
            "differs",
        ),
        ("low_b6_composite_rule", "use_deposited_low_score", "differs"),
        ("per_draw_pipeline", "resample_final_scores_only", "differs"),
        ("invalid_draw_rule", "drop_invalid_draws", "differs"),
        ("output_columns", ["hgvs_nt", "delta"], "differs"),
        ("unique_variant_draw_coverage", "allow_partial", "differs"),
        ("resampling_unit", "observation_channel", "differs"),
        ("interval_extraction", "normal_approximation", "differs"),
        ("required_diagnostics", [], "differs"),
        ("confidence_interval_release_gate", "optional", "differs"),
    ],
)
def test_joint_bootstrap_preregistration_semantics_are_exact(
    field: str,
    value: object,
    message: str,
) -> None:
    registry = _registry()
    specification = _specification(uncertainty.JOINT_BOOTSTRAP, registry)
    specification["uncertainty"]["contract"][field] = value
    _rehash_registration(specification)

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match=message):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


@pytest.mark.parametrize(
    "provenance_field",
    ["sample_role_map", "native_input_manifest", "bootstrap_runtime_manifest"],
)
@pytest.mark.parametrize("field", ["artifact_id", "uri", "body_sha256"])
def test_registration_receipt_cannot_alias_provenance_artifacts(
    field: str,
    provenance_field: str,
) -> None:
    registry = _registry()
    specification = _specification(registry=registry)
    receipt = specification["external_registration"]["receipt"]
    receipt[field] = specification["native_replay_provenance"][provenance_field][field]

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="must not alias"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


@pytest.mark.parametrize(
    "provenance_field",
    ["native_input_manifest", "bootstrap_runtime_manifest"],
)
def test_new_provenance_artifacts_cannot_alias_frozen_source_locks(
    provenance_field: str,
) -> None:
    registry = _registry()
    specification = _specification(registry=registry)
    source_digest = specification["native_replay_provenance"]["low_source_binding"]["source_bundle_sha256"]
    specification["native_replay_provenance"][provenance_field]["body_sha256"] = source_digest
    specification["uncertainty"]["contract"][f"{provenance_field}_sha256"] = source_digest
    _rehash_registration(specification)

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="frozen source-lock"):
        uncertainty.validate_cbs_b6_uncertainty_specification(
            specification,
            registry,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_orientation", "low_minus_high"),
        ("variance_formula", "variance_high_plus_variance_low"),
        ("imputation_allowed", 0),
        ("replicate_pairing_claim", True),
    ],
)
def test_estimand_is_exact_even_after_registration_rehash(
    field: str,
    value: object,
) -> None:
    registry = _registry()
    specification = _specification()
    specification["estimand"][field] = value
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_direct_covariance_cannot_declare_zero_covariance_assumption() -> None:
    registry = _registry()
    specification = _specification()
    specification["uncertainty"]["contract"]["assume_covariance_zero"] = True
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="exact schema"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmatory_eligible", True),
        ("automatic_promotion", True),
    ],
)
def test_schema_valid_specification_cannot_self_promote(
    field: str,
    value: object,
) -> None:
    registry = _registry()
    specification = _specification()
    specification[field] = value
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_registration_cannot_claim_local_authentication() -> None:
    registry = _registry()
    specification = _specification()
    specification["external_registration"]["authentication_status"] = "authenticated"
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="authentication_status"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


@pytest.mark.parametrize(
    ("method", "mutation", "message"),
    [
        (
            uncertainty.DIRECT_COVARIANCE,
            ("contract", "ddof", 0),
            "differs",
        ),
        (
            uncertainty.JOINT_BOOTSTRAP,
            ("contract", "resampling_unit", "observation_channel"),
            "differs",
        ),
        (
            uncertainty.JOINT_BOOTSTRAP,
            ("contract", "draw_count", 9_999),
            "must equal locked value",
        ),
    ],
)
def test_uncertainty_semantics_fail_closed(
    method: str,
    mutation: tuple[str, str, object],
    message: str,
) -> None:
    registry = _registry()
    specification = _specification(method)
    parent, field, value = mutation
    specification["uncertainty"][parent][field] = value
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match=message):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_provenance_artifacts_require_distinct_identity_and_digest() -> None:
    registry = _registry()
    specification = _specification()
    first = specification["native_replay_provenance"]["replay_manifest"]
    second = specification["native_replay_provenance"]["sample_role_map"]
    second["body_sha256"] = first["body_sha256"]
    specification["external_registration"]["locked_specification_sha256"] = (
        uncertainty.specification_locked_payload_sha256(specification)
    )

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="distinct"):
        uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_registration_uri_and_time_are_conservative() -> None:
    registry = _registry()
    for field, value, message in (
        ("immutable_record_uri", "http://registry.example/1", "HTTPS"),
        ("registered_at_utc", "2026-07-26", "UTC"),
        ("registered_at_utc", "2026-99-99T12:00:00Z", "valid UTC"),
    ):
        specification = _specification()
        specification["external_registration"][field] = value
        specification["external_registration"]["locked_specification_sha256"] = (
            uncertainty.specification_locked_payload_sha256(specification)
        )
        with pytest.raises(uncertainty.CbsB6UncertaintyError, match=message):
            uncertainty.validate_cbs_b6_uncertainty_specification(specification, registry)


def test_duplicate_key_status_loader_rejects(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate.json"
    destination.write_text('{"schema_version":1,"schema_version":1}')

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="duplicate key"):
        uncertainty.load_cbs_b6_uncertainty_status(
            destination,
            pair_registry=_registry(),
        )


def test_duplicate_key_specification_loader_rejects(tmp_path: Path) -> None:
    destination = tmp_path / "duplicate-spec.json"
    destination.write_text('{"method":"direct_covariance","method":"joint_bootstrap"}')

    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="duplicate key"):
        uncertainty.load_cbs_b6_uncertainty_specification(
            destination,
            pair_registry=_registry(),
        )


def test_atomic_writer_no_clobber_replace_and_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    status = uncertainty.build_cbs_b6_uncertainty_status(registry)
    destination = tmp_path / "status.json"
    parent_fsync_calls: list[Path] = []
    real_fsync_parent = uncertainty._fsync_parent

    def traced_fsync_parent(path: Path) -> None:
        parent_fsync_calls.append(path)
        real_fsync_parent(path)

    monkeypatch.setattr(uncertainty, "_fsync_parent", traced_fsync_parent)

    uncertainty.write_cbs_b6_uncertainty_status(
        destination,
        status,
        pair_registry=registry,
    )
    original = destination.read_bytes()
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match="already exists"):
        uncertainty.write_cbs_b6_uncertainty_status(
            destination,
            status,
            pair_registry=registry,
        )
    assert destination.read_bytes() == original
    uncertainty.write_cbs_b6_uncertainty_status(
        destination,
        status,
        pair_registry=registry,
        replace=True,
    )
    assert destination.read_bytes() == uncertainty.canonical_json_bytes(status) + b"\n"
    assert not list(tmp_path.glob("*.tmp"))
    assert parent_fsync_calls == [destination, destination]


def test_writer_rejects_wrong_suffix(tmp_path: Path) -> None:
    registry = _registry()
    status = uncertainty.build_cbs_b6_uncertainty_status(registry)
    with pytest.raises(uncertainty.CbsB6UncertaintyError, match=r"\.json"):
        uncertainty.write_cbs_b6_uncertainty_status(
            tmp_path / "status.txt",
            status,
            pair_registry=registry,
        )


def test_native_input_types_are_exact() -> None:
    registry = _registry()
    with pytest.raises(TypeError, match="exact bytes"):
        uncertainty.is_cbs_b6_uncertainty_ready(
            registry,
            native_inputs={"body": bytearray(b"not exact bytes")},
        )
    with pytest.raises(TypeError, match="mapping"):
        uncertainty.is_cbs_b6_uncertainty_ready(
            registry,
            native_inputs=[b"body"],
        )


def test_cli_emits_same_candidate_status(tmp_path: Path) -> None:
    destination = tmp_path / "status.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.cbs_b6_uncertainty_contract",
            "--registry",
            str(REGISTRY_PATH),
            "--status-out",
            str(destination),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    persisted = uncertainty.load_cbs_b6_uncertainty_status(
        destination,
        pair_registry=_registry(),
    )

    assert output["status_sha256"] == persisted["status_sha256"]
    assert output["readiness"] is False
    assert persisted == uncertainty.build_cbs_b6_uncertainty_status(_registry())

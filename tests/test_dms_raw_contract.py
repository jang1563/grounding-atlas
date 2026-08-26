from __future__ import annotations

import math
from copy import deepcopy

import pytest

from eval import dms_raw_contract as raw_dms


def _digest(value: str) -> str:
    return raw_dms.canonical_sha256({"value": value})


def _metadata_entry(name: str, *, verified: bool = True):
    if verified:
        return {
            "status": "verified",
            "evidence_sha256": _digest(f"metadata:{name}"),
            "reason": None,
        }
    return {
        "status": "missing",
        "evidence_sha256": None,
        "reason": f"{name} was not supplied",
    }


def _build_release(
    *,
    target_family_count: int = 8,
    source_family_count: int = 2,
    verified_metadata: bool = True,
    registered: bool = True,
    baseline_role: str = "wild_type",
    failed_target_index: int | None = None,
    failed_source_index: int | None = None,
    shared_control: bool = False,
    shared_item_count: int = 2,
    matched_set_count: int = 1,
    target_all_functional: bool = False,
    reference_alias: bool = False,
    mutant_alias: bool = False,
    numeric_overflow: str | None = None,
    finalize: bool = True,
):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    assay_blueprints = []
    item_blueprints = []
    mutant_records = []
    baseline_pools = []
    baseline_records = []
    baseline_links = []
    assay_count = target_family_count + source_family_count
    first_target_index = source_family_count
    for index in range(assay_count):
        assay_id = f"ASSAY{index:02d}"
        protein_id = f"PROT{index:02d}"
        sequence = amino_acids[index:] + amino_acids[:index]
        family_id = f"FAMILY{index:02d}"
        if reference_alias and index == first_target_index:
            sequence = amino_acids
            family_id = "FAMILY00"
        elif mutant_alias and index == first_target_index:
            sequence = amino_acids[2] + amino_acids[1:]
        reference_sha = raw_dms.sequence_sha256(sequence)
        partition = "source" if index < source_family_count else "target"
        assay_blueprint = {
            "index": index,
            "assay_id": assay_id,
            "protein_id": protein_id,
            "construct_id": f"{assay_id}:construct",
            "sequence": sequence,
            "reference_sha": reference_sha,
            "partition": partition,
            "family_id": family_id,
        }
        assay_blueprints.append(assay_blueprint)

        mutations = [f"{sequence[0]}1{sequence[1]}"]
        if shared_control and index == assay_count - 1:
            mutations.extend(
                f"{sequence[position - 1]}{position}{sequence[position]}"
                for position in range(2, shared_item_count + 1)
            )
        assay_items = []
        for mutation_number, mutation in enumerate(mutations):
            _, position, alternate = raw_dms._parse_mutation(
                mutation,
                "test mutation",
            )
            mutant_sequence = sequence[: position - 1] + alternate + sequence[position:]
            item_id = f"{partition}:{assay_id}:{mutation}"
            pair_id = f"{assay_id}:mutant-vs-wt:{mutation}"
            item = {
                **assay_blueprint,
                "mutation_number": mutation_number,
                "mutation": mutation,
                "mutant_sha": raw_dms.sequence_sha256(mutant_sequence),
                "item_id": item_id,
                "pair_id": pair_id,
            }
            assay_items.append(item)
            item_blueprints.append(item)

        matched_set_id = f"{assay_id}:match-01"
        batch_id = f"{assay_id}:batch-01"
        for item in assay_items:
            functional = (index + item["mutation_number"]) % 2 == 0
            if target_all_functional and partition == "target":
                functional = True
            if numeric_overflow in {"ratio", "affine"}:
                mutant_values = (1e200, 1e200)
            elif numeric_overflow in {"aggregate", "effect"}:
                mutant_values = (1e308, 1e308)
            elif numeric_overflow == "uncertainty":
                mutant_values = (1.3e308, 1.3e308)
            elif numeric_overflow == "baseline_transform":
                mutant_values = (0.0, 0.0)
            else:
                mutant_values = (0.8, 1.0) if functional else (0.1, 0.1)
            for replicate_number, raw_value in enumerate(
                mutant_values,
                start=1,
            ):
                biological_id = f"bio-{replicate_number}"
                technical_id = "tech-1"
                record = {
                    "replicate_id": (f"{item['item_id']}:{matched_set_id}:mutant:{biological_id}:{technical_id}"),
                    "source_row_id": (f"source-row:{assay_id}:{item['mutation']}:mutant:{replicate_number}"),
                    "source_row_sha256": "",
                    "item_id": item["item_id"],
                    "assay_id": assay_id,
                    "protein_id": protein_id,
                    "variant_id": item["mutation"],
                    "intervention_pair_id": item["pair_id"],
                    "condition_role": "mutant",
                    "matched_set_id": matched_set_id,
                    "biological_replicate_id": biological_id,
                    "technical_replicate_id": technical_id,
                    "assay_batch_id": batch_id,
                    "raw_value": raw_value,
                    "raw_unit": "normalized_abundance",
                    "qc_status": (
                        "fail" if numeric_overflow in {"effect", "uncertainty"} and replicate_number == 2 else "pass"
                    ),
                    "qc_reason_codes": (
                        ["overflow-test-unused-replicate"]
                        if numeric_overflow in {"effect", "uncertainty"} and replicate_number == 2
                        else []
                    ),
                }
                record["source_row_sha256"] = raw_dms.raw_record_sha256(record)
                mutant_records.append(record)

        pool_item_groups = [assay_items]
        if not (shared_control and index == assay_count - 1):
            pool_item_groups = [[item] for item in assay_items]
        for group_number, linked_items in enumerate(pool_item_groups):
            is_shared = len(linked_items) > 1
            pool_id = (
                f"{assay_id}:baseline-pool:shared-control"
                if is_shared
                else (f"{assay_id}:baseline-pool:{linked_items[0]['mutation']}")
            )
            pool = {
                "baseline_pool_id": pool_id,
                "assay_id": assay_id,
                "protein_id": protein_id,
                "construct_id": f"{assay_id}:construct",
                "condition_role": baseline_role,
                "matched_set_id": matched_set_id,
                "assay_batch_id": batch_id,
                "reuse_policy": ("shared_preregistered" if is_shared else "item_specific"),
                "maximum_item_links_per_observation": len(linked_items),
            }
            baseline_pools.append(pool)
            should_fail = (failed_target_index == index and partition == "target") or (
                failed_source_index == index and partition == "source"
            )
            baseline_values = (
                (1e-200, 1e-200)
                if numeric_overflow == "ratio"
                else (
                    (-1e308, -1e308)
                    if numeric_overflow == "effect"
                    else (
                        (0.0, 0.0)
                        if numeric_overflow in {"affine", "aggregate", "uncertainty"}
                        else ((1e308, 1e308) if numeric_overflow == "baseline_transform" else (0.2, 0.2))
                    )
                )
            )
            for replicate_number, raw_value in enumerate(
                baseline_values,
                start=1,
            ):
                biological_id = f"bio-{replicate_number}"
                technical_id = "tech-1"
                observation_id = f"{pool_id}:{biological_id}:{technical_id}"
                observation = {
                    "baseline_observation_id": observation_id,
                    "source_row_id": (f"source-row:{assay_id}:baseline:{group_number}:{replicate_number}"),
                    "source_row_sha256": "",
                    "baseline_pool_id": pool_id,
                    "assay_id": assay_id,
                    "protein_id": protein_id,
                    "construct_id": f"{assay_id}:construct",
                    "condition_role": baseline_role,
                    "matched_set_id": matched_set_id,
                    "biological_replicate_id": biological_id,
                    "technical_replicate_id": technical_id,
                    "assay_batch_id": batch_id,
                    "raw_value": raw_value,
                    "raw_unit": "normalized_abundance",
                    "qc_status": (
                        "fail"
                        if should_fail or (numeric_overflow in {"effect", "uncertainty"} and replicate_number == 2)
                        else "pass"
                    ),
                    "qc_reason_codes": (
                        ["low_depth"]
                        if should_fail
                        else (
                            ["overflow-test-unused-replicate"]
                            if numeric_overflow in {"effect", "uncertainty"} and replicate_number == 2
                            else []
                        )
                    ),
                }
                observation["source_row_sha256"] = raw_dms.raw_record_sha256(observation)
                baseline_records.append(observation)
                for item in linked_items:
                    baseline_links.append(
                        {
                            "link_id": (f"{item['item_id']}:baseline-link:{observation_id}"),
                            "item_id": item["item_id"],
                            "intervention_pair_id": item["pair_id"],
                            "baseline_pool_id": pool_id,
                            "baseline_observation_id": observation_id,
                            "assay_id": assay_id,
                            "matched_set_id": matched_set_id,
                            "assay_batch_id": batch_id,
                        }
                    )
    if matched_set_count < 1:
        raise ValueError("matched_set_count must be positive")
    original_mutants = list(mutant_records)
    original_pools = list(baseline_pools)
    original_baselines = list(baseline_records)
    original_links = list(baseline_links)
    for match_number in range(2, matched_set_count + 1):
        old_match_suffix = "match-01"
        new_match_suffix = f"match-{match_number:02d}"
        pool_id_map = {}
        observation_id_map = {}
        for original in original_mutants:
            clone = deepcopy(original)
            clone["matched_set_id"] = clone["matched_set_id"].replace(
                old_match_suffix,
                new_match_suffix,
            )
            clone["replicate_id"] = (
                f"{clone['item_id']}:{clone['matched_set_id']}:mutant:"
                f"{clone['biological_replicate_id']}:"
                f"{clone['technical_replicate_id']}"
            )
            clone["source_row_id"] += f":{new_match_suffix}"
            if numeric_overflow == "uncertainty":
                clone["raw_value"] = -original["raw_value"]
            else:
                clone["raw_value"] += 0.2 * (match_number - 1)
            clone["source_row_sha256"] = raw_dms.raw_record_sha256(clone)
            mutant_records.append(clone)
        for original in original_pools:
            clone = deepcopy(original)
            clone["baseline_pool_id"] = f"{original['baseline_pool_id']}:{new_match_suffix}"
            clone["matched_set_id"] = clone["matched_set_id"].replace(
                old_match_suffix,
                new_match_suffix,
            )
            pool_id_map[original["baseline_pool_id"]] = clone["baseline_pool_id"]
            baseline_pools.append(clone)
        for original in original_baselines:
            clone = deepcopy(original)
            clone["baseline_pool_id"] = pool_id_map[original["baseline_pool_id"]]
            clone["matched_set_id"] = clone["matched_set_id"].replace(
                old_match_suffix,
                new_match_suffix,
            )
            clone["baseline_observation_id"] = (
                f"{clone['baseline_pool_id']}:{clone['biological_replicate_id']}:{clone['technical_replicate_id']}"
            )
            clone["source_row_id"] += f":{new_match_suffix}"
            clone["source_row_sha256"] = raw_dms.raw_record_sha256(clone)
            observation_id_map[original["baseline_observation_id"]] = clone["baseline_observation_id"]
            baseline_records.append(clone)
        for original in original_links:
            clone = deepcopy(original)
            clone["baseline_pool_id"] = pool_id_map[original["baseline_pool_id"]]
            clone["baseline_observation_id"] = observation_id_map[original["baseline_observation_id"]]
            clone["matched_set_id"] = clone["matched_set_id"].replace(
                old_match_suffix,
                new_match_suffix,
            )
            clone["link_id"] = f"{clone['item_id']}:baseline-link:{clone['baseline_observation_id']}"
            baseline_links.append(clone)
    mutant_records.sort(key=lambda record: record["replicate_id"])
    baseline_pools.sort(key=lambda pool: pool["baseline_pool_id"])
    baseline_records.sort(key=lambda record: record["baseline_observation_id"])
    baseline_links.sort(key=lambda link: link["link_id"])
    raw_components = {
        "mutant_records": mutant_records,
        "baseline_pools": baseline_pools,
        "baseline_records": baseline_records,
        "baseline_links": baseline_links,
    }

    source_metadata = {
        name: _metadata_entry(
            name,
            verified=verified_metadata or name != "license",
        )
        for name in sorted(raw_dms.SOURCE_METADATA_NAMES)
    }
    source_lock = {
        "schema_version": raw_dms.RAW_DMS_SCHEMA_VERSION,
        "artifact_type": raw_dms.SOURCE_LOCK_ARTIFACT_TYPE,
        "source_lock_id": "synthetic-source-lock-v2",
        "dataset_name": "synthetic-raw-dms",
        "dataset_version": "2026-07-25",
        "dataset_revision": "test-fixture",
        "source_uri": "https://example.invalid/raw-dms",
        "access_date": "2026-07-25",
        "license_id": "CC0-1.0" if verified_metadata else None,
        "redistribution_status": ("redistributable" if verified_metadata else "unresolved"),
        "raw_archive_sha256": _digest("authoritative raw archive"),
        "record_schema_sha256": _digest("authoritative record schema"),
        "metadata_status": source_metadata,
        "assays": [],
    }
    for blueprint in assay_blueprints:
        assay_items = [item for item in item_blueprints if item["assay_id"] == blueprint["assay_id"]]
        physical_count = sum(
            record["assay_id"] == blueprint["assay_id"] for record in (*mutant_records, *baseline_records)
        )
        source_lock["assays"].append(
            {
                "assay_id": blueprint["assay_id"],
                "protein_id": blueprint["protein_id"],
                "construct_id": blueprint["construct_id"],
                "reference_sequence": blueprint["sequence"],
                "reference_sequence_sha256": blueprint["reference_sha"],
                "raw_value_name": "normalized_abundance",
                "raw_value_unit": "normalized_abundance",
                "wild_type_definition": "matched unedited construct",
                "control_definition": (
                    "matched preregistered neutral-control distribution" if baseline_role == "control" else None
                ),
                "source_assay_sha256": _digest(f"source assay:{blueprint['assay_id']}"),
                "source_metadata_sha256": _digest(f"source metadata:{blueprint['assay_id']}"),
                "raw_records_sha256": raw_dms.assay_raw_records_sha256(
                    raw_components,
                    blueprint["assay_id"],
                ),
                "expected_input_item_count": len(assay_items),
                "expected_raw_replicate_count": physical_count,
            }
        )

    source_sha = raw_dms.canonical_sha256(source_lock)
    family_map = {
        "schema_version": raw_dms.RAW_DMS_SCHEMA_VERSION,
        "artifact_type": raw_dms.FAMILY_MAP_ARTIFACT_TYPE,
        "family_map_id": "synthetic-family-map-v2",
        "source_lock_sha256": source_sha,
        "authority_name": "synthetic-family-authority",
        "authority_version": "1",
        "authority_uri": "https://example.invalid/families",
        "mapping_file_sha256": _digest("family mapping file"),
        "metadata_status": _metadata_entry(
            "family-map",
            verified=verified_metadata,
        ),
        "records": sorted(
            [
                {
                    "protein_id": blueprint["protein_id"],
                    "reference_sequence_sha256": blueprint["reference_sha"],
                    "family_id": blueprint["family_id"],
                    "evidence_id": f"family-evidence:{blueprint['protein_id']}",
                }
                for blueprint in assay_blueprints
            ],
            key=lambda record: (
                record["protein_id"],
                record["reference_sequence_sha256"],
            ),
        ),
    }
    family_sha = raw_dms.canonical_sha256(family_map)

    input_manifest = {
        "schema_version": raw_dms.RAW_DMS_SCHEMA_VERSION,
        "artifact_type": raw_dms.INPUT_MANIFEST_ARTIFACT_TYPE,
        "source_lock_sha256": source_sha,
        "family_map_sha256": family_sha,
        "records": sorted(
            [
                {
                    "item_id": blueprint["item_id"],
                    "analysis_partition": blueprint["partition"],
                    "assay_id": blueprint["assay_id"],
                    "source_record_id": (f"{blueprint['assay_id']}:{blueprint['mutation']}"),
                    "protein_id": blueprint["protein_id"],
                    "construct_id": blueprint["construct_id"],
                    "reference_sequence_sha256": blueprint["reference_sha"],
                    "variant_id": blueprint["mutation"],
                    "mutation": blueprint["mutation"],
                    "mutant_sequence_sha256": blueprint["mutant_sha"],
                    "representation_kind": "full_mutant_protein_sequence",
                    "representation_sha256": blueprint["mutant_sha"],
                    "family_id": blueprint["family_id"],
                    "split_group_id": blueprint["family_id"],
                    "intervention_pair_id": blueprint["pair_id"],
                }
                for blueprint in item_blueprints
            ],
            key=lambda record: record["item_id"],
        ),
    }
    input_sha = raw_dms.canonical_sha256(input_manifest)
    raw_manifest = {
        "schema_version": raw_dms.RAW_DMS_SCHEMA_VERSION,
        "artifact_type": raw_dms.RAW_REPLICATE_MANIFEST_ARTIFACT_TYPE,
        "source_lock_sha256": source_sha,
        "input_manifest_sha256": input_sha,
        **raw_components,
    }

    assay_transformations = []
    for blueprint in assay_blueprints:
        assay_transformations.append(
            {
                "assay_id": blueprint["assay_id"],
                "baseline_role": baseline_role,
                "value_transform": {
                    "operation": ("log2" if numeric_overflow == "baseline_transform" else "identity"),
                    "pseudocount": (1e308 if numeric_overflow == "baseline_transform" else 0.0),
                },
                "within_role_aggregation": "mean",
                "contrast": {
                    "operation": ("ratio" if numeric_overflow == "ratio" else "difference"),
                    "pseudocount": 0.0,
                },
                "across_match_aggregation": "mean",
                "orientation": {
                    "operation": "identity",
                    "endpoint_semantics": "higher_is_more_functional",
                },
                "scale": {
                    "operation": ("affine" if numeric_overflow == "affine" else "identity"),
                    "multiplier": (1e308 if numeric_overflow == "affine" else 1.0),
                    "offset": 0.0,
                },
                "threshold": {
                    "operation": "greater_equal",
                    "cutoff": 0.5,
                    "label_1_semantics": raw_dms.LABEL_1_SEMANTICS,
                    "label_0_semantics": raw_dms.LABEL_0_SEMANTICS,
                },
                "uncertainty": {
                    "estimator": raw_dms.UNCERTAINTY_ESTIMATOR,
                    "minimum_complete_matched_sets": 2,
                    "unavailable_reason": (raw_dms.UNCERTAINTY_UNAVAILABLE_REASON),
                },
                "minimum_pass_replicates_per_role": (1 if numeric_overflow in {"effect", "uncertainty"} else 2),
                "minimum_complete_matched_sets": 1,
            }
        )
    transformation = {
        "schema_version": raw_dms.RAW_DMS_SCHEMA_VERSION,
        "artifact_type": raw_dms.TRANSFORMATION_SPECIFICATION_ARTIFACT_TYPE,
        "source_lock_sha256": source_sha,
        "family_map_sha256": family_sha,
        "input_manifest_sha256": input_sha,
        "specification": {
            "specification_id": "synthetic-transform-v2",
            "assay_transformations": assay_transformations,
        },
        "registration": {
            "status": ("registered_external" if registered else "unregistered_candidate"),
            "locked_payload_sha256": "",
            "registered_at": "2026-07-25T12:00:00Z" if registered else None,
            "registry_id": "registry:test:v2" if registered else None,
            "receipt_sha256": _digest("registry receipt") if registered else None,
        },
    }
    transformation["registration"]["locked_payload_sha256"] = raw_dms.transformation_locked_payload_sha256(
        transformation
    )
    if not finalize:
        return {
            "source_lock": source_lock,
            "family_map": family_map,
            "input_manifest": input_manifest,
            "raw_replicate_manifest": raw_manifest,
            "transformation_specification": transformation,
            "exclusion_ledger": {},
            "outcome_manifest": {},
        }
    exclusions = raw_dms.build_exclusion_ledger(
        source_lock,
        family_map,
        input_manifest,
        raw_manifest,
        transformation,
    )
    outcomes = raw_dms.build_outcome_manifest(
        source_lock,
        family_map,
        input_manifest,
        raw_manifest,
        transformation,
        exclusions,
    )
    return {
        "source_lock": source_lock,
        "family_map": family_map,
        "input_manifest": input_manifest,
        "raw_replicate_manifest": raw_manifest,
        "transformation_specification": transformation,
        "exclusion_ledger": exclusions,
        "outcome_manifest": outcomes,
    }


@pytest.fixture(scope="module")
def valid_release():
    return _build_release()


def _validate_outcomes(release):
    raw_dms.validate_outcome_manifest(
        release["outcome_manifest"],
        release["source_lock"],
        release["family_map"],
        release["input_manifest"],
        release["raw_replicate_manifest"],
        release["transformation_specification"],
        release["exclusion_ledger"],
    )


def test_v2_release_recomputes_outcomes_but_only_local_structure_is_ready(
    valid_release,
):
    raw_dms.validate_release(valid_release)
    outcomes = valid_release["outcome_manifest"]["records"]
    assert len(outcomes) == 10
    assert valid_release["exclusion_ledger"]["records"] == []
    assert {record["target_label"] for record in outcomes} == {0, 1}
    assert sorted({round(record["raw_contrast"], 8) for record in outcomes}) == [
        -0.1,
        0.7,
    ]
    assert {record["mutant_aggregate"] for record in outcomes} == {0.1, 0.9}
    assert {record["baseline_aggregate"] for record in outcomes} == {0.2}
    assert {record["target_label_semantics"] for record in outcomes} == {
        raw_dms.LABEL_0_SEMANTICS,
        raw_dms.LABEL_1_SEMANTICS,
    }
    assert all(record["effect_uncertainty"] is None for record in outcomes)
    assert all(
        record["uncertainty_status"] == "unavailable"
        and record["uncertainty_reason"] == raw_dms.UNCERTAINTY_UNAVAILABLE_REASON
        and record["uncertainty_matched_set_count"] == 1
        for record in outcomes
    )
    assert all(record["mutant_biological_replicate_count"] == 2 for record in outcomes)
    assert all(record["baseline_biological_replicate_count"] == 2 for record in outcomes)

    readiness = raw_dms.assess_confirmatory_readiness(valid_release)
    assert readiness["status"] == "NOT_READY_FOR_CONFIRMATORY_EXECUTION"
    assert readiness["confirmatory_eligible"] is False
    assert readiness["eligible"] is False
    assert readiness["local_structural_status"] == "READY_AWAITING_EXTERNAL_AUTHENTICATION"
    assert readiness["external_authentication_status"] == "NOT_AUTHENTICATED"
    assert readiness["observed_target_outcome_families"] == 8
    assert readiness["observed_source_outcome_families"] == 2
    assert readiness["observed_source_labels"] == [0, 1]
    assert readiness["observed_target_labels"] == [0, 1]
    assert readiness["local_missing_requirements"] == []
    assert readiness["missing_requirements"] == ["externally_authenticated_trust_index"]


def test_two_complete_matches_replay_aggregates_and_standard_error():
    release = _build_release(matched_set_count=2)
    raw_dms.validate_release(release)
    outcome = release["outcome_manifest"]["records"][0]
    assert outcome["mutant_aggregate"] == pytest.approx(1.0)
    assert outcome["baseline_aggregate"] == pytest.approx(0.2)
    assert outcome["raw_contrast"] == pytest.approx(0.8)
    assert outcome["oriented_effect"] == pytest.approx(0.8)
    assert outcome["scaled_effect"] == pytest.approx(0.8)
    assert outcome["effect_uncertainty"] == pytest.approx(0.1)
    assert outcome["uncertainty_status"] == "available"
    assert outcome["uncertainty_reason"] is None
    assert outcome["uncertainty_matched_set_count"] == 2


def test_preregistered_control_distribution_can_replace_literal_wt():
    release = _build_release(baseline_role="control")
    raw_dms.validate_release(release)
    assert (
        release["transformation_specification"]["specification"]["assay_transformations"][0]["baseline_role"]
        == "control"
    )
    assert release["outcome_manifest"]["records"][0]["target_label"] == 1


def test_shared_control_is_one_physical_pool_linked_to_two_variants():
    release = _build_release(baseline_role="control", shared_control=True)
    raw_dms.validate_release(release)
    raw_manifest = release["raw_replicate_manifest"]
    shared_pool = next(
        pool for pool in raw_manifest["baseline_pools"] if pool["reuse_policy"] == "shared_preregistered"
    )
    observations = [
        record
        for record in raw_manifest["baseline_records"]
        if record["baseline_pool_id"] == shared_pool["baseline_pool_id"]
    ]
    links = [
        link for link in raw_manifest["baseline_links"] if link["baseline_pool_id"] == shared_pool["baseline_pool_id"]
    ]
    linked_items = {link["item_id"] for link in links}
    assert len(observations) == 2
    assert len(linked_items) == 2
    assert len(links) == 4
    assert all("variant_id" not in observation for observation in observations)
    assert all("item_id" not in observation for observation in observations)

    shared_outcomes = [
        outcome
        for outcome in release["outcome_manifest"]["records"]
        if set(outcome["baseline_pool_ids"]) == {shared_pool["baseline_pool_id"]}
    ]
    assert len(shared_outcomes) == 2
    assert (
        shared_outcomes[0]["qualified_baseline_observation_ids"]
        == shared_outcomes[1]["qualified_baseline_observation_ids"]
    )
    assert shared_outcomes[0]["raw_replicate_subset_sha256"] != (shared_outcomes[1]["raw_replicate_subset_sha256"])


def test_literal_wt_mode_is_item_specific_and_never_reused(valid_release):
    raw_manifest = valid_release["raw_replicate_manifest"]
    assert all(
        pool["condition_role"] == "wild_type"
        and pool["reuse_policy"] == "item_specific"
        and pool["maximum_item_links_per_observation"] == 1
        for pool in raw_manifest["baseline_pools"]
    )
    link_counts = {}
    for link in raw_manifest["baseline_links"]:
        observation_id = link["baseline_observation_id"]
        link_counts[observation_id] = link_counts.get(observation_id, 0) + 1
    assert set(link_counts.values()) == {1}


def test_hashes_are_deterministic_and_canonical(valid_release, tmp_path):
    second = _build_release()
    assert raw_dms.canonical_sha256(valid_release) == raw_dms.canonical_sha256(second)
    path = raw_dms.write_artifact(tmp_path / "outcomes.v2.json", valid_release)
    assert path.read_bytes() == raw_dms.canonical_json_bytes(valid_release) + b"\n"


def test_caller_supplied_target_label_must_equal_recomputation(valid_release):
    bad = deepcopy(valid_release)
    first = bad["outcome_manifest"]["records"][0]
    first["target_label"] = 1 - first["target_label"]
    with pytest.raises(ValueError, match="target_label differs"):
        _validate_outcomes(bad)


def test_sign_reversed_outcome_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["outcome_manifest"]["records"][0]["oriented_effect"] *= -1
    with pytest.raises(ValueError, match="oriented_effect differs"):
        _validate_outcomes(bad)


def test_swapped_common_class_semantics_are_rejected(valid_release):
    bad = deepcopy(valid_release)
    threshold = bad["transformation_specification"]["specification"]["assay_transformations"][0]["threshold"]
    threshold["label_1_semantics"], threshold["label_0_semantics"] = (
        threshold["label_0_semantics"],
        threshold["label_1_semantics"],
    )
    with pytest.raises(ValueError, match="label 1 must mean"):
        raw_dms.validate_transformation_specification(
            bad["transformation_specification"],
            bad["source_lock"],
            bad["family_map"],
            bad["input_manifest"],
        )


def test_inverted_outcome_semantics_are_rejected(valid_release):
    bad = deepcopy(valid_release)
    outcome = bad["outcome_manifest"]["records"][0]
    outcome["target_label_semantics"] = (
        raw_dms.LABEL_0_SEMANTICS if outcome["target_label"] == 1 else raw_dms.LABEL_1_SEMANTICS
    )
    with pytest.raises(ValueError, match="common class semantics"):
        _validate_outcomes(bad)


def test_mutant_or_baseline_aggregate_tampering_is_rejected(valid_release):
    for field in ("mutant_aggregate", "baseline_aggregate"):
        bad = deepcopy(valid_release)
        bad["outcome_manifest"]["records"][0][field] += 1.0
        with pytest.raises(ValueError, match=rf"{field} differs"):
            _validate_outcomes(bad)


def test_unavailable_uncertainty_reason_tampering_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["outcome_manifest"]["records"][0]["uncertainty_reason"] = "caller_claimed_unavailable"
    with pytest.raises(ValueError, match="uncertainty_reason differs"):
        _validate_outcomes(bad)


def test_available_standard_error_tampering_is_rejected():
    bad = _build_release(matched_set_count=2)
    bad["outcome_manifest"]["records"][0]["effect_uncertainty"] = 999.0
    with pytest.raises(ValueError, match="effect_uncertainty differs"):
        _validate_outcomes(bad)


def test_uncertainty_estimator_is_frozen(valid_release):
    bad = deepcopy(valid_release)
    bad["transformation_specification"]["specification"]["assay_transformations"][0]["uncertainty"]["estimator"] = (
        "caller_selected_bootstrap"
    )
    with pytest.raises(ValueError, match="exact schema-v2 estimator"):
        raw_dms.validate_transformation_specification(
            bad["transformation_specification"],
            bad["source_lock"],
            bad["family_map"],
            bad["input_manifest"],
        )


@pytest.mark.parametrize(
    ("numeric_overflow", "matched_set_count", "message_fragment"),
    [
        ("baseline_transform", 1, "pseudocount addition"),
        ("aggregate", 1, "arithmetic mean"),
        ("ratio", 1, "ratio contrast"),
        ("effect", 1, "difference contrast"),
        ("affine", 1, "affine scaled product"),
        ("uncertainty", 2, "uncertainty estimator"),
    ],
)
def test_finite_raw_inputs_cannot_produce_nonfinite_derived_artifacts(
    numeric_overflow,
    matched_set_count,
    message_fragment,
):
    release = _build_release(
        numeric_overflow=numeric_overflow,
        matched_set_count=matched_set_count,
        finalize=False,
    )
    raw_manifest = release["raw_replicate_manifest"]
    raw_dms.validate_raw_replicate_manifest(
        raw_manifest,
        release["source_lock"],
        release["input_manifest"],
        release["family_map"],
    )
    raw_dms.validate_transformation_specification(
        release["transformation_specification"],
        release["source_lock"],
        release["family_map"],
        release["input_manifest"],
    )
    raw_values = [
        record["raw_value"] for field in ("mutant_records", "baseline_records") for record in raw_manifest[field]
    ]
    assert all(math.isfinite(value) for value in raw_values)
    assert raw_dms.canonical_json_bytes(raw_manifest)

    with pytest.raises(
        ValueError,
        match=rf"{message_fragment}.*non-finite derived value",
    ):
        raw_dms.build_exclusion_ledger(
            release["source_lock"],
            release["family_map"],
            release["input_manifest"],
            raw_manifest,
            release["transformation_specification"],
        )


@pytest.mark.parametrize(
    "aggregation_field",
    ["within_role_aggregation", "across_match_aggregation"],
)
def test_coherently_relocked_median_aggregation_is_rejected(
    valid_release,
    aggregation_field,
):
    bad = deepcopy(valid_release)
    specification = bad["transformation_specification"]
    specification["specification"]["assay_transformations"][0][aggregation_field] = "median"
    specification["registration"]["locked_payload_sha256"] = raw_dms.transformation_locked_payload_sha256(specification)
    with pytest.raises(ValueError, match="must be arithmetic mean"):
        raw_dms.validate_transformation_specification(
            specification,
            bad["source_lock"],
            bad["family_map"],
            bad["input_manifest"],
        )


def test_mismatched_wt_identity_is_rejected_before_derivation(valid_release):
    bad = deepcopy(valid_release)
    record = bad["raw_replicate_manifest"]["baseline_records"][0]
    record["protein_id"] = "PROT99"
    record["source_row_sha256"] = raw_dms.raw_record_sha256(record)
    with pytest.raises(ValueError, match="differs from frozen baseline pool"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("orientation", "operation"), "negate"),
        (("threshold", "cutoff"), 0.95),
    ],
)
def test_post_hoc_orientation_or_threshold_change_breaks_registration(
    valid_release,
    path,
    value,
):
    bad = deepcopy(valid_release)
    transform = bad["transformation_specification"]
    assay_transform = transform["specification"]["assay_transformations"][0]
    assay_transform[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="changed post hoc"):
        raw_dms.validate_transformation_specification(
            transform,
            bad["source_lock"],
            bad["family_map"],
            bad["input_manifest"],
        )


def test_missing_raw_replicate_is_rejected_against_source_lock(valid_release):
    bad = deepcopy(valid_release)
    bad["raw_replicate_manifest"]["mutant_records"].pop()
    with pytest.raises(ValueError, match="count differs"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_missing_qc_field_is_rejected_by_exact_schema(valid_release):
    bad = deepcopy(valid_release)
    bad["raw_replicate_manifest"]["mutant_records"][0].pop("qc_status")
    with pytest.raises(ValueError, match="exact schema"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_duplicate_input_identity_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["input_manifest"]["records"][1] = deepcopy(bad["input_manifest"]["records"][0])
    with pytest.raises(ValueError, match="duplicate source identities"):
        raw_dms.validate_input_manifest(
            bad["input_manifest"],
            bad["source_lock"],
            bad["family_map"],
        )


def test_duplicate_replicate_identity_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["raw_replicate_manifest"]["mutant_records"][1] = deepcopy(bad["raw_replicate_manifest"]["mutant_records"][0])
    with pytest.raises(ValueError, match="duplicate identities"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_source_target_family_overlap_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    source_family = next(
        record["family_id"] for record in bad["input_manifest"]["records"] if record["analysis_partition"] == "source"
    )
    target_input = next(
        record for record in bad["input_manifest"]["records"] if record["analysis_partition"] == "target"
    )
    target_family_record = next(
        record for record in bad["family_map"]["records"] if record["protein_id"] == target_input["protein_id"]
    )
    target_family_record["family_id"] = source_family
    target_input["family_id"] = source_family
    target_input["split_group_id"] = source_family
    bad["input_manifest"]["family_map_sha256"] = raw_dms.canonical_sha256(bad["family_map"])
    with pytest.raises(ValueError, match="family overlap"):
        raw_dms.validate_input_manifest(
            bad["input_manifest"],
            bad["source_lock"],
            bad["family_map"],
        )


def test_family_map_cannot_split_one_exact_reference_checksum_across_aliases():
    release = _build_release(reference_alias=True, finalize=False)
    family_map = release["family_map"]
    target_record = next(record for record in family_map["records"] if record["protein_id"] == "PROT02")
    target_record["family_id"] = "FAMILY99"
    with pytest.raises(
        ValueError,
        match="exact reference_sequence_sha256 cannot map to multiple families",
    ):
        raw_dms.validate_family_map(family_map, release["source_lock"])


def test_reference_checksum_alias_crossing_source_target_is_rejected():
    release = _build_release(reference_alias=True, finalize=False)
    with pytest.raises(ValueError, match="reference-sequence checksum overlap"):
        raw_dms.validate_input_manifest(
            release["input_manifest"],
            release["source_lock"],
            release["family_map"],
        )


def test_mutant_checksum_alias_crossing_source_target_is_rejected():
    release = _build_release(mutant_alias=True, finalize=False)
    with pytest.raises(ValueError, match="mutant-sequence checksum overlap"):
        raw_dms.validate_input_manifest(
            release["input_manifest"],
            release["source_lock"],
            release["family_map"],
        )


def test_label_leakage_in_input_is_rejected_before_schema_extension(valid_release):
    bad = deepcopy(valid_release)
    bad["input_manifest"]["records"][0]["target_label"] = 1
    with pytest.raises(ValueError, match="label-free"):
        raw_dms.validate_input_manifest(
            bad["input_manifest"],
            bad["source_lock"],
            bad["family_map"],
        )


def test_raw_value_digest_tampering_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["raw_replicate_manifest"]["mutant_records"][0]["raw_value"] += 100
    with pytest.raises(ValueError, match="source-row digest"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_top_level_binding_tampering_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    bad["input_manifest"]["source_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different canonical bytes"):
        raw_dms.validate_input_manifest(
            bad["input_manifest"],
            bad["source_lock"],
            bad["family_map"],
        )


def test_cloned_and_altered_baseline_physical_observation_is_rejected(
    valid_release,
):
    bad = deepcopy(valid_release)
    clone = deepcopy(bad["raw_replicate_manifest"]["baseline_records"][0])
    clone["source_row_id"] += ":forged-clone"
    clone["raw_value"] += 0.25
    clone["source_row_sha256"] = raw_dms.raw_record_sha256(clone)
    bad["raw_replicate_manifest"]["baseline_records"].append(clone)
    with pytest.raises(ValueError, match="cloned physical observation"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_cloned_physical_observation_cannot_hide_behind_renamed_pool(
    valid_release,
):
    bad = deepcopy(valid_release)
    manifest = bad["raw_replicate_manifest"]
    original_pool = manifest["baseline_pools"][0]
    original_observation = next(
        record
        for record in manifest["baseline_records"]
        if record["baseline_pool_id"] == original_pool["baseline_pool_id"]
    )
    original_link = next(
        link
        for link in manifest["baseline_links"]
        if link["baseline_observation_id"] == original_observation["baseline_observation_id"]
    )

    cloned_pool = deepcopy(original_pool)
    cloned_pool["baseline_pool_id"] += ":forged-alias"
    cloned_observation = deepcopy(original_observation)
    cloned_observation["baseline_pool_id"] = cloned_pool["baseline_pool_id"]
    cloned_observation["baseline_observation_id"] = (
        f"{cloned_pool['baseline_pool_id']}:"
        f"{cloned_observation['biological_replicate_id']}:"
        f"{cloned_observation['technical_replicate_id']}"
    )
    cloned_observation["source_row_id"] += ":forged-alias"
    cloned_observation["source_row_sha256"] = raw_dms.raw_record_sha256(cloned_observation)
    cloned_link = deepcopy(original_link)
    cloned_link["baseline_pool_id"] = cloned_pool["baseline_pool_id"]
    cloned_link["baseline_observation_id"] = cloned_observation["baseline_observation_id"]
    cloned_link["link_id"] = f"{cloned_link['item_id']}:baseline-link:{cloned_link['baseline_observation_id']}"

    manifest["baseline_pools"].append(cloned_pool)
    manifest["baseline_pools"].sort(key=lambda record: record["baseline_pool_id"])
    manifest["baseline_records"].append(cloned_observation)
    manifest["baseline_records"].sort(key=lambda record: record["baseline_observation_id"])
    manifest["baseline_links"].append(cloned_link)
    manifest["baseline_links"].sort(key=lambda record: record["link_id"])

    with pytest.raises(ValueError, match="cloned physical observation"):
        raw_dms.validate_raw_replicate_manifest(
            manifest,
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_unlinked_baseline_pool_is_rejected(valid_release):
    bad = deepcopy(valid_release)
    pool_id = bad["raw_replicate_manifest"]["baseline_pools"][0]["baseline_pool_id"]
    bad["raw_replicate_manifest"]["baseline_links"] = [
        link for link in bad["raw_replicate_manifest"]["baseline_links"] if link["baseline_pool_id"] != pool_id
    ]
    with pytest.raises(ValueError, match="baseline pool .* is unlinked"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assay_id", "ASSAY99"),
        ("assay_batch_id", "wrong-batch"),
    ],
)
def test_cross_assay_or_batch_baseline_link_is_rejected(
    valid_release,
    field,
    value,
):
    bad = deepcopy(valid_release)
    bad["raw_replicate_manifest"]["baseline_links"][0][field] = value
    with pytest.raises(ValueError, match="incompatible|crosses assay"):
        raw_dms.validate_raw_replicate_manifest(
            bad["raw_replicate_manifest"],
            bad["source_lock"],
            bad["input_manifest"],
            bad["family_map"],
        )


def test_shared_baseline_reuse_multiplicity_is_enforced():
    release = _build_release(
        baseline_role="control",
        shared_control=True,
        shared_item_count=3,
    )
    shared_pool = next(
        pool
        for pool in release["raw_replicate_manifest"]["baseline_pools"]
        if pool["reuse_policy"] == "shared_preregistered"
    )
    shared_pool["maximum_item_links_per_observation"] = 2
    with pytest.raises(ValueError, match="exceeds frozen reuse multiplicity"):
        raw_dms.validate_raw_replicate_manifest(
            release["raw_replicate_manifest"],
            release["source_lock"],
            release["input_manifest"],
            release["family_map"],
        )


def test_exclusion_ledger_is_derived_from_failed_qc_and_partitions_items():
    release = _build_release(failed_target_index=2)
    raw_dms.validate_release(release)
    exclusions = release["exclusion_ledger"]["records"]
    assert len(exclusions) == 1
    assert exclusions[0]["reason_codes"] == [
        "insufficient_baseline_qc_pass",
        "insufficient_complete_matched_sets",
    ]
    input_ids = {record["item_id"] for record in release["input_manifest"]["records"]}
    outcome_ids = {record["item_id"] for record in release["outcome_manifest"]["records"]}
    exclusion_ids = {record["item_id"] for record in exclusions}
    assert outcome_ids.isdisjoint(exclusion_ids)
    assert outcome_ids | exclusion_ids == input_ids
    readiness = raw_dms.assess_confirmatory_readiness(release)
    assert readiness["eligible"] is False
    assert "target_families_without_outcomes:FAMILY02" in readiness["missing_requirements"]


def test_source_qc_exclusion_blocks_usable_source_outcome_gate():
    release = _build_release(failed_source_index=0)
    raw_dms.validate_release(release)
    readiness = raw_dms.assess_confirmatory_readiness(release)
    assert readiness["local_structural_status"] == "NOT_READY_LOCAL_STRUCTURE"
    assert "minimum_source_outcome_items:1/2" in readiness["local_missing_requirements"]
    assert "minimum_source_outcome_families:1/2" in readiness["local_missing_requirements"]
    assert "source_binary_label_support:0" in readiness["local_missing_requirements"]


def test_target_class_support_is_required_for_local_structural_readiness():
    release = _build_release(target_all_functional=True)
    readiness = raw_dms.assess_confirmatory_readiness(release)
    assert readiness["local_structural_status"] == "NOT_READY_LOCAL_STRUCTURE"
    assert "target_binary_label_support:1" in readiness["local_missing_requirements"]


def test_exclusion_ledger_tampering_is_rejected():
    release = _build_release(failed_target_index=2)
    release["exclusion_ledger"]["records"][0]["reason_codes"] = ["post_hoc_drop"]
    with pytest.raises(ValueError, match="differs from deterministic"):
        raw_dms.validate_exclusion_ledger(
            release["exclusion_ledger"],
            release["source_lock"],
            release["family_map"],
            release["input_manifest"],
            release["raw_replicate_manifest"],
            release["transformation_specification"],
        )


@pytest.mark.parametrize(
    ("release", "expected_fragment"),
    [
        (
            lambda: _build_release(target_family_count=7),
            "minimum_evaluable_target_families:7/8",
        ),
        (
            lambda: _build_release(verified_metadata=False),
            "source_metadata:license",
        ),
        (
            lambda: _build_release(registered=False),
            "locally_declared_transformation_registration",
        ),
    ],
)
def test_confirmatory_readiness_is_fail_closed(release, expected_fragment):
    readiness = raw_dms.assess_confirmatory_readiness(release())
    assert readiness["status"] == "NOT_READY_FOR_CONFIRMATORY_EXECUTION"
    assert readiness["eligible"] is False
    assert expected_fragment in readiness["missing_requirements"]

"""Analyze preregistered causal-intervention artifacts.

The analyzer keeps three claims separate:

* fixed-direction steering/patching can support causal availability (sufficiency);
* targeted erasure plus a non-trivial rescue can support partial natural use;
* bidirectional content/routing interventions can support an activation-to-output bottleneck.

All effects are paired at the item level and cluster-bootstrapped by the declared biological split
group.  A proxy or unavailable grouping can produce pilot estimates but never a confirmatory status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .causal_intervention import (
        CAUSAL_ARTIFACT_SCHEMA,
        InterventionContract,
        activation_sha256,
        direction_from_frozen_decoder,
        execution_input_sha256,
        execution_receipt_payload,
        execution_receipt_sha256,
        validate_binary_label,
        validate_condition_record_schema,
        validate_execution_row_context,
        validate_execution_trace,
        validate_scored_record_schema,
    )
except ImportError:  # direct execution from eval/
    from causal_intervention import (
        CAUSAL_ARTIFACT_SCHEMA,
        InterventionContract,
        activation_sha256,
        direction_from_frozen_decoder,
        execution_input_sha256,
        execution_receipt_payload,
        execution_receipt_sha256,
        validate_binary_label,
        validate_condition_record_schema,
        validate_execution_row_context,
        validate_execution_trace,
        validate_scored_record_schema,
    )


NONCONFIRMATORY_GROUP_SCOPES = {
    "",
    "entity_proxy_nonconfirmatory",
    "snapshot_entity_proxy",
    "unavailable",
}
SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
NOT_ADJUDICATED = "NOT_ADJUDICATED"
DEFAULT_MIN_GROUPS = 8
DEFAULT_MIN_RANDOM_DIRECTIONS = 20
DEFAULT_N_BOOT = 2000
MIN_CONFIRMATORY_ITEMS = 30
MIN_CONFIRMATORY_GROUPS = 8
MIN_CONFIRMATORY_RANDOM_DIRECTIONS = 20
MIN_CONFIRMATORY_N_BOOT = 1000
MIN_RANDOM_CONTROL_QUANTILE = 0.95
MIN_DECODER_AUROC = 0.65
MIN_DECODER_AUROC_CI_LOWER = 0.50
MIN_DECODER_SELECTIVITY = 0.05
MIN_DECODER_SELECTIVITY_CI_LOWER = 0.0
APPROVED_CONFIRMATORY_GROUP_SCOPES = {
    "assay_batch",
    "chromosome",
    "compound_scaffold",
    "donor",
    "gene_family",
    "locus",
    "patient",
    "perturbation",
    "protein_family",
    "regulatory_element",
    "slide",
    "study",
}
CONTENT_EQUIVALENCE_MANIFEST_SCHEMA = 2
ACTIVATION_CAPTURE_MANIFEST_SCHEMA = 2
CONTENT_EQUIVALENCE_PROVENANCE_FIELDS = {
    "tokenizer_revision",
    "embedding_model_revision",
    "measurement_scope",
    "comparison_manifest_sha256",
}
CONTENT_EQUIVALENCE_RECORD_FIELDS = {
    "item_id",
    "condition",
    "direction_id",
    "source_content",
    "recipient_content",
    "source_token_ids",
    "recipient_token_ids",
    "source_embedding",
    "recipient_embedding",
    "source_content_sha256",
    "recipient_content_sha256",
    "content_token_count_absolute_difference",
    "content_embedding_cosine_distance",
}


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number >= 0.0)


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _id_token(value: Any) -> str:
    """Type-stable identity token; integer 1 and string "1" stay distinct."""

    if value is None or isinstance(value, (list, dict)):
        raise TypeError("causal item/group IDs must be non-null JSON scalars")
    return (
        f"{type(value).__name__}:"
        + json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def item_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact type-stable held-out item set."""

    tokens = sorted({_id_token(row["item_id"]) for row in records})
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def content_equivalence_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the canonical source/recipient content-comparison manifest."""

    return hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def activation_capture_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical source/recipient activation captures."""

    return hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_content_sha256(content: str) -> str:
    """Hash the exact UTF-8 content embedded in a comparison manifest."""

    if not isinstance(content, str):
        raise TypeError("canonical content must be a string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _recompute_content_comparison(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute content hashes, token-count distance, and cosine distance."""

    source_content = record.get("source_content")
    recipient_content = record.get("recipient_content")
    if not isinstance(source_content, str) or not isinstance(
        recipient_content,
        str,
    ):
        raise ValueError("content manifest must embed source/recipient strings")
    source_tokens = record.get("source_token_ids")
    recipient_tokens = record.get("recipient_token_ids")
    if (
        not isinstance(source_tokens, list)
        or not isinstance(recipient_tokens, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in [*source_tokens, *recipient_tokens]
        )
    ):
        raise ValueError("content manifest token IDs must be non-negative integers")
    source_embedding = np.asarray(record.get("source_embedding"), dtype=float)
    recipient_embedding = np.asarray(
        record.get("recipient_embedding"),
        dtype=float,
    )
    source_norm = float(np.linalg.norm(source_embedding))
    recipient_norm = float(np.linalg.norm(recipient_embedding))
    if (
        source_embedding.ndim != 1
        or recipient_embedding.shape != source_embedding.shape
        or source_embedding.size == 0
        or not np.isfinite(source_embedding).all()
        or not np.isfinite(recipient_embedding).all()
        or source_norm == 0.0
        or recipient_norm == 0.0
    ):
        raise ValueError(
            "content manifest embeddings must be aligned finite non-zero vectors"
        )
    cosine = float(
        1.0
        - np.dot(source_embedding, recipient_embedding)
        / (source_norm * recipient_norm)
    )
    if cosine < 0.0 and np.isclose(cosine, 0.0, atol=1e-15):
        cosine = 0.0
    return {
        "source_content_sha256": canonical_content_sha256(source_content),
        "recipient_content_sha256": canonical_content_sha256(
            recipient_content
        ),
        "content_token_count_absolute_difference": abs(
            len(source_tokens) - len(recipient_tokens)
        ),
        "content_embedding_cosine_distance": cosine,
    }


def _execution_receipt_gate(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Verify that post-label rows still match label-free execution receipts."""

    for row in records:
        receipt = row.get("execution_receipt")
        if not isinstance(receipt, Mapping):
            return ["label-free execution receipt is missing"]
        try:
            expected = execution_receipt_payload(row)
            expected_sha = execution_receipt_sha256(expected)
        except (TypeError, ValueError) as exc:
            return [f"label-free execution receipt is invalid: {exc}"]
        if dict(receipt) != expected:
            return [
                "causal row metadata or margins differ from its execution receipt"
            ]
        if row.get("execution_receipt_sha256") != expected_sha:
            return ["causal execution receipt checksum differs"]
    return []


def _pair_group_nesting_gate(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require each intervention pair to stay within one bootstrap dependency group."""

    groups_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in records:
        pair_id = row.get("intervention_pair_id")
        if pair_id is None:
            continue
        group_id = row.get("split_group_id")
        if group_id is None:
            return [
                "non-null intervention pairs require non-null biological groups"
            ]
        groups_by_pair[_id_token(pair_id)].add(_id_token(group_id))
    if any(len(groups) != 1 for groups in groups_by_pair.values()):
        return [
            "each intervention_pair_id must be nested within one biological bootstrap group"
        ]
    return []


def _confirmatory_design_gate(
    artifact: Mapping[str, Any],
    *,
    min_items: int,
    min_groups: int,
    min_random_directions: int,
    n_boot: int,
) -> dict[str, Any]:
    """Validate preregistration, split, control, and collateral-damage attestations.

    These fields are deliberately required for a positive confirmatory status. Missing metadata
    leaves the statistical estimates available for inspection but yields ``NOT_ADJUDICATED``.
    """

    design = artifact.get("design")
    failures: list[str] = []
    if not isinstance(design, Mapping):
        design = {}
        failures.append("design metadata are missing")

    if design.get("preregistered") is not True:
        failures.append("design.preregistered must be true")
    if design.get("selection_scope") != "outer_train_only":
        failures.append("design.selection_scope must be outer_train_only")

    analysis_lock = design.get("analysis_lock")
    requested_lock = {
        "min_items": min_items,
        "min_groups": min_groups,
        "min_random_directions": min_random_directions,
        "n_boot": n_boot,
    }
    if not isinstance(analysis_lock, Mapping):
        failures.append("design.analysis_lock is missing")
    else:
        for field, value in requested_lock.items():
            if analysis_lock.get(field) != value:
                failures.append(
                    f"analysis {field}={value} differs from the preregistered value "
                    f"{analysis_lock.get(field)!r}"
                )
    hard_minima = {
        "min_items": MIN_CONFIRMATORY_ITEMS,
        "min_groups": MIN_CONFIRMATORY_GROUPS,
        "min_random_directions": MIN_CONFIRMATORY_RANDOM_DIRECTIONS,
        "n_boot": MIN_CONFIRMATORY_N_BOOT,
    }
    for field, minimum in hard_minima.items():
        if requested_lock[field] < minimum:
            failures.append(
                f"{field}={requested_lock[field]} is below the confirmatory floor {minimum}"
            )

    firewall = design.get("train_test_firewall")
    if not isinstance(firewall, Mapping):
        failures.append("design.train_test_firewall is missing")
    else:
        if firewall.get("confirmatory") is not True:
            failures.append("train/test firewall is not confirmatory")
        if firewall.get("split_group_scope") != artifact.get("split_group_scope"):
            failures.append("firewall split-group scope does not match the artifact")

    split_scope = artifact.get("split_group_scope")
    if split_scope not in APPROVED_CONFIRMATORY_GROUP_SCOPES:
        failures.append(
            f"split_group_scope={split_scope!r} is not an approved biological scope"
        )
    group_provenance = design.get("split_group_provenance")
    if not isinstance(group_provenance, Mapping):
        failures.append("design.split_group_provenance is missing")
    else:
        if group_provenance.get("scope") != split_scope:
            failures.append("split-group provenance scope does not match the artifact")
        if (
            group_provenance.get("registry_task_id")
            != artifact.get("contract", {}).get("task_id")
        ):
            failures.append("split-group provenance task does not match the contract")
        if not str(group_provenance.get("source_field", "")).strip():
            failures.append("split-group provenance source_field is missing")
        if not _valid_sha256(group_provenance.get("source_dataset_sha256")):
            failures.append("split-group provenance dataset checksum is invalid")
        if (
            group_provenance.get("group_map_sha256")
            != group_map_sha256(artifact["records"])
        ):
            failures.append("split-group provenance group-map checksum does not match")

    failures.extend(_frozen_decoder_gate(artifact))

    controls = design.get("controls")
    if not isinstance(controls, Mapping):
        failures.append("design.controls is missing")
    else:
        for field in ("covariance_matched", "norm_matched", "variance_matched"):
            if controls.get(field) is not True:
                failures.append(f"controls.{field} must be true")
        if controls.get("construction_scope") != "outer_train_only":
            failures.append("controls.construction_scope must be outer_train_only")
        count = controls.get("random_direction_count")
        if not isinstance(count, int) or isinstance(count, bool):
            failures.append("controls.random_direction_count must be an integer")
        elif count < min_random_directions:
            failures.append(
                "controls.random_direction_count is below the locked minimum "
                f"of {min_random_directions}"
            )
        norm_tolerance = controls.get("max_norm_relative_error")
        variance_tolerance = controls.get("max_variance_relative_error")
        if not (
            _finite_nonnegative(norm_tolerance)
            and _finite_nonnegative(variance_tolerance)
        ):
            failures.append(
                "control norm/variance matching tolerances must be finite and non-negative"
            )
        else:
            random_rows = [
                row
                for row in artifact["records"]
                if row.get("direction_kind") == "random"
            ]
            if not random_rows:
                failures.append("no observed random-control records were found")
            for field, tolerance in (
                ("direction_norm_relative_error", float(norm_tolerance)),
                (
                    "projected_variance_relative_error",
                    float(variance_tolerance),
                ),
            ):
                try:
                    values = [float(row[field]) for row in random_rows]
                except (KeyError, TypeError, ValueError):
                    failures.append(
                        f"random-control records require numeric {field}"
                    )
                    continue
                if (
                    not np.isfinite(values).all()
                    or min(values, default=-1.0) < 0.0
                    or max(values, default=np.inf) > tolerance
                ):
                    failures.append(
                        f"observed random-control {field} exceeds its tolerance"
                    )

    collateral = design.get("collateral_checks")
    if not isinstance(collateral, Mapping):
        failures.append("design.collateral_checks is missing")
    else:
        if collateral.get("predeclared") is not True:
            failures.append("collateral checks were not predeclared")
        if collateral.get("status") != "PASS":
            failures.append("collateral checks did not pass")
        metric_pairs = (
            ("max_kl_observed", "max_kl_threshold"),
            (
                "max_unrelated_margin_change_observed",
                "max_unrelated_margin_change_threshold",
            ),
        )
        for observed_name, threshold_name in metric_pairs:
            observed = collateral.get(observed_name)
            threshold = collateral.get(threshold_name)
            if not (
                _finite_nonnegative(observed)
                and _finite_nonnegative(threshold)
            ):
                failures.append(
                    f"collateral {observed_name}/{threshold_name} must be finite and non-negative"
                )
            elif float(observed) > float(threshold):
                failures.append(
                    f"collateral {observed_name} exceeds {threshold_name}"
                )
        try:
            collateral_kl_values = np.asarray(
                [float(row["collateral_kl"]) for row in artifact["records"]],
                dtype=float,
            )
            unrelated_values = np.asarray(
                [
                    float(row["unrelated_margin_change"])
                    for row in artifact["records"]
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            failures.append(
                "every causal record requires collateral_kl and unrelated_margin_change"
            )
        else:
            if (
                not np.isfinite(collateral_kl_values).all()
                or not np.isfinite(unrelated_values).all()
                or np.any(collateral_kl_values < 0.0)
            ):
                failures.append(
                    "every observed collateral metric must be finite and KL must be non-negative"
                )
            else:
                observed_kl = float(np.max(collateral_kl_values))
                observed_unrelated = float(
                    np.max(np.abs(unrelated_values))
                )
                reported_kl = collateral.get("max_kl_observed")
                reported_unrelated = collateral.get(
                    "max_unrelated_margin_change_observed"
                )
                if _finite_nonnegative(reported_kl) and not np.isclose(
                    observed_kl,
                    float(reported_kl),
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    failures.append(
                        "reported max_kl_observed does not match causal records"
                    )
                if _finite_nonnegative(reported_unrelated) and not np.isclose(
                    observed_unrelated,
                    float(reported_unrelated),
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    failures.append(
                        "reported unrelated-margin maximum does not match causal records"
                    )

    failures.extend(_execution_receipt_gate(artifact["records"]))
    failures.extend(_pair_group_nesting_gate(artifact["records"]))
    return {
        "status": SUPPORTED if not failures else NOT_ADJUDICATED,
        "failures": failures,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema, intervention contract, and post-execution label arithmetic."""

    if artifact.get("schema_version") != CAUSAL_ARTIFACT_SCHEMA:
        raise ValueError(
            f"causal artifact schema must be {CAUSAL_ARTIFACT_SCHEMA}, "
            f"got {artifact.get('schema_version')!r}"
        )
    if not isinstance(artifact.get("records"), list) or not artifact["records"]:
        raise ValueError("causal artifact must contain non-empty records")
    contract = artifact.get("contract", {})
    if not isinstance(contract, Mapping):
        raise ValueError("causal artifact contract must be an object")
    try:
        contract_object = InterventionContract(**contract)
    except TypeError as exc:
        raise ValueError(f"invalid causal intervention contract: {exc}") from exc
    contract_object.validate()
    checksum_by_direction_id: dict[str, str] = {}
    random_checksums: dict[str, str] = {}
    family = str(artifact.get("intervention_family"))
    for row in artifact["records"]:
        validate_scored_record_schema(row)
        if row.get("test_label_used_for_intervention") is not False:
            raise ValueError("record does not attest test-label-free intervention")
        validate_execution_row_context(row, contract_object)
        direction_kind = str(row.get("direction_kind", ""))
        validate_condition_record_schema(family, row)
        if (
            artifact.get("intervention_family") == "steering"
            and direction_kind
            in {"target", "random", "shuffled", "surface"}
        ):
            try:
                observed_dose_scale = float(row["steering_dose_scale"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "steering record lacks the executed dose scale"
                ) from exc
            if not np.isclose(
                observed_dose_scale,
                float(contract_object.steering_dose_scale),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(
                    "steering record dose scale differs from the contract"
                )
        applied_direction = row.get("applied_direction_sha256")
        if direction_kind == "target":
            if applied_direction != contract_object.applied_direction_sha256:
                raise ValueError(
                    "target record applied-direction checksum differs from the contract"
                )
        elif direction_kind in {"random", "shuffled", "surface"} and not _valid_sha256(
            applied_direction
        ):
            raise ValueError(
                "every intervention control record requires an applied-direction checksum"
            )
        if direction_kind in {"target", "random", "shuffled", "surface"}:
            direction_id = str(row.get("direction_id", "")).strip()
            if not direction_id:
                raise ValueError("intervention record direction_id is missing")
            previous = checksum_by_direction_id.setdefault(
                direction_id,
                str(applied_direction),
            )
            if previous != applied_direction:
                raise ValueError(
                    "one direction_id maps to multiple applied-direction checksums"
                )
            if direction_kind == "random":
                random_checksums[direction_id] = str(applied_direction)
        label = validate_binary_label(
            row.get("target_label"),
            "causal record target_label",
        )
        try:
            answer_margin = float(row["answer_logit_margin"])
            correct_margin = float(row["correct_answer_margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every causal record requires finite answer and correct-answer margins"
            ) from exc
        expected = (2 * label - 1) * answer_margin
        if (
            not np.isfinite(answer_margin)
            or not np.isfinite(correct_margin)
            or not np.isclose(
                correct_margin,
                expected,
                atol=1e-12,
                rtol=1e-12,
            )
        ):
            raise ValueError(
                "correct_answer_margin does not match the post-execution target label"
            )
    if len(set(random_checksums.values())) != len(random_checksums):
        raise ValueError(
            "distinct random direction IDs must use distinct applied-direction checksums"
        )
    if len(set(checksum_by_direction_id.values())) != len(
        checksum_by_direction_id
    ):
        raise ValueError(
            "distinct intervention direction IDs must use distinct checksums"
        )


def load_artifact(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    validate_artifact(artifact)
    return artifact


def _group_map(records: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for row in records:
        item_id = _id_token(row["item_id"])
        group = row.get("split_group_id")
        group = None if group is None else _id_token(group)
        if item_id in result and result[item_id] != group:
            raise ValueError(f"item {item_id!r} changes split group across conditions")
        result[item_id] = group
    return result


def group_map_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical item-to-biological-group assignment."""

    payload = sorted(_group_map(records).items())
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _frozen_decoder_gate(artifact: Mapping[str, Any]) -> list[str]:
    """Verify the embedded train-only decoder and its causal test split."""

    embedded = artifact.get("frozen_decoder_artifact")
    if not isinstance(embedded, Mapping):
        return ["a confirmatory artifact must embed its frozen decoder artifact"]
    try:
        from .probe_common import (
            frozen_decoder_artifact_sha256,
            frozen_decoder_evaluation_artifact_sha256,
            recompute_frozen_decoder_evaluation,
            validate_frozen_decoder_artifact,
        )
    except ImportError:
        from probe_common import (
            frozen_decoder_artifact_sha256,
            frozen_decoder_evaluation_artifact_sha256,
            recompute_frozen_decoder_evaluation,
            validate_frozen_decoder_artifact,
        )

    try:
        embedded = dict(embedded)
        validate_frozen_decoder_artifact(embedded)
        direction_metadata = direction_from_frozen_decoder(embedded)
    except (TypeError, ValueError) as exc:
        return [f"embedded frozen decoder is invalid: {exc}"]

    failures = []
    contract = artifact.get("contract", {})
    if (
        frozen_decoder_artifact_sha256(embedded)
        != contract.get("direction_artifact_sha256")
    ):
        failures.append("embedded frozen decoder checksum differs from the contract")
    if embedded.get("fold_id") != contract.get("fold_id"):
        failures.append("embedded frozen decoder fold differs from the contract")
    if (
        direction_metadata["applied_direction_sha256"]
        != contract.get("applied_direction_sha256")
    ):
        failures.append(
            "embedded frozen decoder positive-class direction differs from the "
            "applied-direction checksum"
        )
    if (
        direction_metadata["direction_center_sha256"]
        != contract.get("direction_center_sha256")
    ):
        failures.append(
            "embedded frozen decoder center differs from the contract center checksum"
        )
    try:
        locked_dose_scale = float(contract.get("steering_dose_scale"))
    except (TypeError, ValueError):
        locked_dose_scale = float("nan")
    if not np.isclose(
        direction_metadata["residual_feature_rms"],
        locked_dose_scale,
        atol=1e-12,
        rtol=1e-12,
    ):
        failures.append(
            "contract steering dose scale differs from the frozen decoder residual feature RMS"
        )
    if (
        embedded.get("activation_site", {}).get("decoder_block_index")
        != contract.get("layer_index")
    ):
        failures.append(
            "embedded frozen decoder block does not match the intervention layer"
        )
    if (
        embedded.get("activation_site", {}).get("semantics")
        != "hf_hidden_states_embedding_plus_decoder_block_outputs"
    ):
        failures.append(
            "confirmatory selected decoder must use embedding-inclusive HF hidden-state semantics"
        )

    split = embedded["split"]
    embedded_test_items = [_id_token(value) for value in split["test_ids"]]
    embedded_test_groups = dict(
        zip(
            embedded_test_items,
            (_id_token(value) for value in split["test_group_ids"]),
            strict=True,
        )
    )
    causal_groups = _group_map(artifact["records"])
    if (
        len(embedded_test_items) != len(embedded_test_groups)
        or len(embedded_test_groups) != len(causal_groups)
        or embedded_test_groups != causal_groups
    ):
        failures.append(
            "embedded frozen decoder test IDs/groups do not match causal records"
        )

    provenance = embedded["training"].get("provenance", {})
    expected_provenance = {
        "task_id": contract.get("task_id"),
        "model_revision": contract.get("model_revision"),
        "source_dataset_sha256": artifact.get("design", {})
        .get("split_group_provenance", {})
        .get("source_dataset_sha256"),
        "split_group_scope": artifact.get("split_group_scope"),
        "prompt_protocol_sha256": contract.get("prompt_protocol_sha256"),
        "model_forward_code_sha256": contract.get("execution_code_sha256"),
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            failures.append(
                f"embedded frozen decoder provenance {field} differs from the causal artifact"
            )

    firewall = artifact.get("design", {}).get("train_test_firewall", {})
    expected_counts = {
        "train_entities": len(split["train_ids"]),
        "test_entities": len(split["test_ids"]),
        "train_groups": len(
            {_id_token(value) for value in split["train_group_ids"]}
        ),
        "test_groups": len(
            {_id_token(value) for value in split["test_group_ids"]}
        ),
    }
    for field, expected in expected_counts.items():
        if firewall.get(field) != expected:
            failures.append(
                f"train/test firewall {field} does not match the frozen decoder"
            )

    signal_lock = artifact.get("design", {}).get("decoder_signal_gate")
    if not isinstance(signal_lock, Mapping):
        failures.append("design.decoder_signal_gate is missing")
        return failures
    expected_signal_lock = {
        "metric": "roc_auc",
        "selection_scope": "outer_test_only_after_frozen_prediction",
    }
    for field, expected in expected_signal_lock.items():
        if signal_lock.get(field) != expected:
            failures.append(f"decoder signal gate {field} must be {expected!r}")
    try:
        n_signal_boot = int(signal_lock["n_boot"])
        signal_seed = int(signal_lock["seed"])
        confidence_level = float(signal_lock["confidence_level"])
        minimum_auroc = float(signal_lock["minimum_auroc"])
        minimum_auroc_ci_lower = float(
            signal_lock["minimum_auroc_ci_lower"]
        )
        minimum_selectivity = float(signal_lock["minimum_selectivity"])
        minimum_selectivity_ci_lower = float(
            signal_lock["minimum_selectivity_ci_lower"]
        )
    except (KeyError, TypeError, ValueError):
        failures.append("decoder signal thresholds/bootstrap lock are incomplete")
        return failures
    if (
        n_signal_boot < MIN_CONFIRMATORY_N_BOOT
        or not np.isclose(confidence_level, 0.95)
        or not np.isfinite(
            [
                minimum_auroc,
                minimum_auroc_ci_lower,
                minimum_selectivity,
                minimum_selectivity_ci_lower,
            ]
        ).all()
        or minimum_auroc < MIN_DECODER_AUROC
        or minimum_auroc_ci_lower < MIN_DECODER_AUROC_CI_LOWER
        or minimum_selectivity < MIN_DECODER_SELECTIVITY
        or minimum_selectivity_ci_lower
        < MIN_DECODER_SELECTIVITY_CI_LOWER
    ):
        failures.append(
            "decoder signal thresholds/bootstrap lock are invalid or below hard floors"
        )
        return failures

    evaluation = artifact.get("frozen_decoder_evaluation_artifact")
    if not isinstance(evaluation, Mapping):
        failures.append(
            "causal artifact lacks a separate frozen-decoder evaluation artifact"
        )
        return failures
    evaluation_lock = evaluation.get("analysis_lock", {})
    if evaluation_lock != {
        "n_boot": n_signal_boot,
        "seed": signal_seed,
        "confidence_level": confidence_level,
    }:
        failures.append(
            "decoder evaluation bootstrap lock differs from the causal design"
        )
    try:
        recomputed = recompute_frozen_decoder_evaluation(
            dict(evaluation),
            embedded,
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"held-out decoder signal evidence is invalid: {exc}")
        return failures
    prediction = evaluation.get("prediction_artifact", {})
    extraction = prediction.get("activation_extraction_provenance", {})
    expected_extraction = {
        "task_id": contract.get("task_id"),
        "model_revision": contract.get("model_revision"),
        "source_dataset_sha256": artifact.get("design", {})
        .get("split_group_provenance", {})
        .get("source_dataset_sha256"),
        "prompt_protocol_sha256": contract.get("prompt_protocol_sha256"),
        "model_forward_code_sha256": contract.get("execution_code_sha256"),
    }
    for field, expected in expected_extraction.items():
        if extraction.get(field) != expected:
            failures.append(
                f"held-out activation extraction {field} differs from the causal artifact"
            )
    if (
        frozen_decoder_evaluation_artifact_sha256(
            dict(evaluation),
            embedded,
        )
        != contract.get("decoder_evaluation_artifact_sha256")
    ):
        failures.append(
            "frozen decoder evaluation checksum differs from the contract"
        )

    evaluation_groups = {
        _id_token(row["item_id"]): _id_token(row["group_id"])
        for row in evaluation["items"]
    }
    evaluation_labels = {
        _id_token(row["item_id"]): int(row["target_label"])
        for row in evaluation["items"]
    }
    if (
        len(evaluation_groups) != len(evaluation["items"])
        or evaluation_groups != causal_groups
        or evaluation_labels != _target_label_map(artifact["records"])
    ):
        failures.append(
            "held-out decoder evaluation IDs, groups, or labels differ from causal records"
        )
    selected_ci_lower = recomputed["confidence_intervals"]["selected_auroc"][0]
    selectivity_ci_lower = recomputed["confidence_intervals"]["selectivity"][0]
    if recomputed["selected_auroc"] < minimum_auroc:
        failures.append("held-out decoder AUROC is below its preregistered threshold")
    if selected_ci_lower <= minimum_auroc_ci_lower:
        failures.append(
            "held-out decoder AUROC confidence lower bound does not clear its threshold"
        )
    if recomputed["selectivity"] < minimum_selectivity:
        failures.append(
            "held-out decoder selectivity is below its preregistered threshold"
        )
    if selectivity_ci_lower <= minimum_selectivity_ci_lower:
        failures.append(
            "held-out decoder selectivity confidence lower bound does not clear its threshold"
        )
    return failures


def _intervention_pair_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for row in records:
        item_id = _id_token(row["item_id"])
        pair_id = row.get("intervention_pair_id")
        pair_id = None if pair_id is None else _id_token(pair_id)
        if item_id in result and result[item_id] != pair_id:
            raise ValueError(
                f"item {item_id!r} changes intervention pair across conditions"
            )
        result[item_id] = pair_id
    return result


def _target_label_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in records:
        item_id = _id_token(row["item_id"])
        label = int(row["target_label"])
        if item_id in result and result[item_id] != label:
            raise ValueError(f"item {item_id!r} changes target label across conditions")
        result[item_id] = label
    return result


def _source_activation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_source_conditions: Mapping[str, str],
    locked_strength: Any,
) -> list[str]:
    """Validate donor identity, pairing, checksums, and patch strength from records."""

    failures = []
    if (
        not _finite_nonnegative(locked_strength)
        or float(locked_strength) > 1.0
    ):
        return ["locked intervention strength must be in [0, 1]"]

    checksum_by_source_key: dict[tuple[str, str, str, str], str] = {}
    source_item_by_source_key: dict[tuple[str, str, str, str], str] = {}
    checksums_by_control: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    observed_conditions = set()
    for row in records:
        condition = str(row.get("condition"))
        expected_source = expected_source_conditions.get(condition)
        if expected_source is None:
            continue
        observed_conditions.add(condition)
        try:
            observed_strength = float(row["patch_strength"])
        except (KeyError, TypeError, ValueError):
            failures.append(
                "every intervention-source record requires a numeric patch_strength"
            )
            break
        if not np.isclose(
            observed_strength,
            float(locked_strength),
            atol=1e-12,
            rtol=1e-12,
        ):
            failures.append(
                "observed intervention strength differs from the locked value"
            )
            break
        if not 0.0 <= observed_strength <= 1.0:
            failures.append("observed intervention strength must be in [0, 1]")
            break
        if row.get("source_condition") != expected_source:
            failures.append(
                "source_condition does not match the locked intervention cell"
            )
            break
        source_checksum = row.get("source_activation_sha256")
        recipient_checksum = row.get("recipient_activation_sha256")
        if not (
            _valid_sha256(source_checksum)
            and _valid_sha256(recipient_checksum)
        ):
            failures.append(
                "source and recipient activation checksums are required"
            )
            break
        if source_checksum == recipient_checksum:
            failures.append(
                "source and recipient checksums are identical; same-run identity "
                "sources are non-confirmatory"
            )
            break
        source_item_id = row.get("source_item_id")
        source_pair_id = row.get("source_intervention_pair_id")
        recipient_pair_id = row.get("intervention_pair_id")
        try:
            source_item_token = _id_token(source_item_id)
            source_pair_token = _id_token(source_pair_id)
            recipient_pair_token = _id_token(recipient_pair_id)
        except TypeError:
            failures.append(
                "source and recipient intervention-pair IDs must both be non-null "
                "JSON scalars, and source_item_id must be a non-null JSON scalar"
            )
            break
        if source_pair_token != recipient_pair_token:
            failures.append(
                "source and recipient intervention-pair IDs do not match"
            )
            break

        item_id = _id_token(row["item_id"])
        direction_id = str(row.get("direction_id") or row.get("direction_kind"))
        source_key = (
            item_id,
            _id_token(row["intervention_pair_id"]),
            expected_source,
            direction_id,
        )
        previous = checksum_by_source_key.setdefault(
            source_key,
            str(source_checksum),
        )
        previous_source_item = source_item_by_source_key.setdefault(
            source_key,
            source_item_token,
        )
        if (
            previous != source_checksum
            or previous_source_item != source_item_token
        ):
            failures.append(
                "the same locked source key resolves to different source identities "
                "or activation checksums"
            )
            break
        control_key = (expected_source, direction_id)
        checksums_by_control[control_key][item_id] = str(source_checksum)

    missing = set(expected_source_conditions) - observed_conditions
    if missing:
        failures.append(
            f"source-activation provenance is missing conditions: {sorted(missing)}"
        )
    for control_key, checksum_by_item in checksums_by_control.items():
        if len(set(checksum_by_item.values())) != len(checksum_by_item):
            failures.append(
                "one source activation checksum is reused across distinct items "
                f"for source key {control_key!r}"
            )
            break
    return failures


def _activation_capture_manifest_gate(
    artifact: Mapping[str, Any],
    *,
    expected_source_conditions: Mapping[str, str],
) -> list[str]:
    """Recompute source/recipient tensor hashes and bind capture context."""

    failures = []
    capture_fields = {
        "item_id",
        "condition",
        "direction_id",
        "source_item_id",
        "source_condition",
        "intervention_pair_id",
        "source_intervention_pair_id",
        "source_input",
        "recipient_input",
        "source_input_sha256",
        "recipient_input_sha256",
        "source_input_token_ids",
        "source_attention_mask",
        "source_execution_input_sha256",
        "source_execution_trace",
        "recipient_input_token_ids",
        "recipient_attention_mask",
        "recipient_execution_input_sha256",
        "model_revision",
        "layer_index",
        "token_position",
        "hook_site_kind",
        "prompt_protocol_sha256",
        "execution_code_sha256",
        "execution_context_sha256",
        "source_activation",
        "recipient_activation",
        "source_activation_sha256",
        "recipient_activation_sha256",
    }
    manifest = artifact.get("activation_capture_manifest")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema_version", "records"}
        or manifest.get("schema_version")
        != ACTIVATION_CAPTURE_MANIFEST_SCHEMA
        or not isinstance(manifest.get("records"), list)
    ):
        return [
            "activation-capture manifest must use the exact label-free schema"
        ]
    provenance = artifact.get("design", {}).get(
        "activation_capture_provenance"
    )
    if not isinstance(provenance, Mapping):
        failures.append("activation-capture provenance is missing")
    elif set(provenance) != {"manifest_sha256", "measurement_scope"}:
        failures.append(
            "activation-capture provenance must use the exact preregistered schema"
        )
    elif provenance.get("measurement_scope") != (
        "locked_held_out_source_recipient_pairs"
    ):
        failures.append("activation-capture measurement scope is invalid")
    elif provenance.get("manifest_sha256") != (
        activation_capture_manifest_sha256(manifest)
    ):
        failures.append("activation-capture manifest checksum differs")

    relevant = [
        row
        for row in artifact["records"]
        if row.get("condition") in expected_source_conditions
    ]

    def key(row):
        return (
            _id_token(row["item_id"]),
            str(row["condition"]),
            str(row["direction_id"]),
        )

    row_by_key = {}
    for row in relevant:
        row_key = key(row)
        if row_key in row_by_key:
            failures.append("activation-source rows contain duplicate keys")
            return failures
        row_by_key[row_key] = row
    capture_by_key = {}
    for capture in manifest["records"]:
        if not isinstance(capture, Mapping):
            failures.append("activation-capture record is invalid")
            return failures
        if set(capture) != capture_fields:
            failures.append(
                "activation-capture records must use the exact preregistered schema"
            )
            return failures
        try:
            capture_key = key(capture)
        except (KeyError, TypeError, ValueError):
            failures.append("activation-capture key is invalid")
            return failures
        if capture_key in capture_by_key:
            failures.append("activation-capture manifest contains duplicate keys")
            return failures
        capture_by_key[capture_key] = capture
    if set(row_by_key) != set(capture_by_key):
        failures.append(
            "activation-capture manifest does not cover exact source rows"
        )
        return failures

    contract = artifact["contract"]
    try:
        contract_object = InterventionContract(**contract)
        contract_object.validate()
    except (TypeError, ValueError) as exc:
        return [
            *failures,
            f"activation-capture contract is invalid: {exc}",
        ]
    item_pair_map = _intervention_pair_map(artifact["records"])
    item_group_map = _group_map(artifact["records"])
    known_item_tokens = {
        _id_token(value)
        for value in artifact.get("frozen_decoder_artifact", {})
        .get("split", {})
        .get("test_ids", [])
    }
    try:
        expected_feature_count = int(
            artifact["frozen_decoder_artifact"]["model"]["feature_count"]
        )
    except (KeyError, TypeError, ValueError):
        return [
            *failures,
            "activation-capture decoder feature width is unavailable",
        ]
    content_by_key = {}
    if artifact.get("intervention_family") == "routing_patch":
        content_manifest = artifact.get("content_equivalence_manifest")
        if not isinstance(content_manifest, Mapping) or not isinstance(
            content_manifest.get("records"),
            list,
        ):
            return [
                *failures,
                "routing activation captures require a content-equivalence manifest",
            ]
        for content_record in content_manifest["records"]:
            try:
                content_key = key(content_record)
            except (KeyError, TypeError, ValueError):
                return [
                    *failures,
                    "content-equivalence manifest key is invalid",
                ]
            if content_key in content_by_key:
                return [
                    *failures,
                    "content-equivalence manifest contains duplicate capture keys",
                ]
            content_by_key[content_key] = content_record
        if set(content_by_key) != set(row_by_key):
            return [
                *failures,
                "routing content and activation manifests must cover the same rows",
            ]
    for capture_key, row in row_by_key.items():
        capture = capture_by_key[capture_key]
        try:
            source_activation = np.asarray(
                capture["source_activation"],
                dtype=float,
            )
            recipient_activation = np.asarray(
                capture["recipient_activation"],
                dtype=float,
            )
            source_checksum = activation_sha256(
                source_activation
            )
            recipient_checksum = activation_sha256(
                recipient_activation
            )
            source_input_checksum = canonical_content_sha256(
                capture["source_input"]
            )
            recipient_input_checksum = canonical_content_sha256(
                capture["recipient_input"]
            )
            source_execution_input_checksum = execution_input_sha256(
                capture["source_input_token_ids"],
                capture["source_attention_mask"],
            )
            source_execution_trace = validate_execution_trace(
                capture["source_execution_trace"],
                contract_object,
            )
            recipient_execution_input_checksum = execution_input_sha256(
                capture["recipient_input_token_ids"],
                capture["recipient_attention_mask"],
            )
            source_item_token = _id_token(capture["source_item_id"])
            source_trace_item_token = _id_token(
                source_execution_trace["item_id"]
            )
            identity_fields_match = all(
                _id_token(capture[field]) == _id_token(row[field])
                for field in (
                    "source_item_id",
                    "intervention_pair_id",
                    "source_intervention_pair_id",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"activation capture cannot be recomputed: {exc}")
            break
        if (
            source_activation.shape != (expected_feature_count,)
            or recipient_activation.shape != (expected_feature_count,)
        ):
            failures.append(
                "activation-capture tensors must be one hidden vector with the "
                "frozen decoder feature width"
            )
            break
        if (
            source_trace_item_token != source_item_token
            or source_execution_trace["execution_input_ids"]
            != capture.get("source_input_token_ids")
            or source_execution_trace["execution_attention_mask"]
            != capture.get("source_attention_mask")
            or source_execution_trace["execution_input_sha256"]
            != capture.get("source_execution_input_sha256")
        ):
            failures.append(
                "activation donor input differs from its identity-bound source "
                "execution trace"
            )
            break
        if (
            capture.get("source_activation_sha256") != source_checksum
            or capture.get("recipient_activation_sha256")
            != recipient_checksum
            or capture.get("source_input_sha256")
            != source_input_checksum
            or capture.get("recipient_input_sha256")
            != recipient_input_checksum
            or capture.get("source_execution_input_sha256")
            != source_execution_input_checksum
            or capture.get("recipient_execution_input_sha256")
            != recipient_execution_input_checksum
        ):
            failures.append(
                "activation-capture checksums do not match embedded inputs/tensors"
            )
            break
        if (
            capture.get("recipient_input_token_ids")
            != row.get("execution_input_ids")
            or capture.get("recipient_attention_mask")
            != row.get("execution_attention_mask")
            or capture.get("recipient_execution_input_sha256")
            != row.get("execution_input_sha256")
        ):
            failures.append(
                "activation recipient input differs from the measurement execution trace"
            )
            break
        if (
            not identity_fields_match
            or capture.get("source_condition") != row.get("source_condition")
            or capture.get("source_activation_sha256")
            != row.get("source_activation_sha256")
            or capture.get("recipient_activation_sha256")
            != row.get("recipient_activation_sha256")
        ):
            failures.append(
                "activation-capture identity or checksums differ from causal rows"
            )
            break
        if source_item_token not in known_item_tokens:
            failures.append(
                "activation source item is absent from the frozen held-out item set"
            )
            break
        source_pair = item_pair_map.get(source_item_token)
        source_group = item_group_map.get(source_item_token)
        recipient_group = item_group_map.get(capture_key[0])
        try:
            captured_source_pair = _id_token(
                capture["source_intervention_pair_id"]
            )
        except TypeError:
            captured_source_pair = None
        if source_pair != captured_source_pair:
            failures.append(
                "activation source item does not belong to the claimed intervention pair"
            )
            break
        if source_group is None or source_group != recipient_group:
            failures.append(
                "activation source and recipient must share one biological dependency group"
            )
            break
        if artifact.get("intervention_family") == "routing_patch":
            content_record = content_by_key.get(capture_key)
            if content_record is None or (
                capture.get("source_input")
                != content_record.get("source_content")
                or capture.get("recipient_input")
                != content_record.get("recipient_content")
                or capture.get("source_input_token_ids")
                != content_record.get("source_token_ids")
                or capture.get("recipient_input_token_ids")
                != content_record.get("recipient_token_ids")
            ):
                failures.append(
                    "routing activation strings or token IDs differ from the locked "
                    "content manifest"
                )
                break
        if (
            capture.get("model_revision") != contract.get("model_revision")
            or capture.get("layer_index") != contract.get("layer_index")
            or capture.get("token_position") != contract.get("token_position")
            or capture.get("hook_site_kind") != contract.get("hook_site_kind")
            or capture.get("prompt_protocol_sha256")
            != contract.get("prompt_protocol_sha256")
            or capture.get("execution_code_sha256")
            != contract.get("execution_code_sha256")
            or capture.get("execution_context_sha256")
            != row.get("execution_context_sha256")
        ):
            failures.append(
                "activation capture execution context differs from the contract"
            )
            break
    return failures


def _content_equivalence_gate(
    artifact: Mapping[str, Any],
    *,
    intervention_conditions: Iterable[str],
) -> list[str]:
    """Recompute locked content-equivalence maxima from routing records."""

    failures = []
    design = artifact.get("design", {})
    lock = design.get("content_equivalence")
    if not isinstance(lock, Mapping):
        return ["design.content_equivalence is missing"]
    if lock.get("predeclared") is not True:
        failures.append("content-equivalence metrics were not predeclared")
    provenance = lock.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("content-equivalence provenance is missing")
    else:
        if set(provenance) != CONTENT_EQUIVALENCE_PROVENANCE_FIELDS:
            failures.append(
                "content-equivalence provenance must use the exact label-free schema"
            )
        if not str(provenance.get("tokenizer_revision", "")).strip():
            failures.append("content-equivalence tokenizer revision is missing")
        if not str(provenance.get("embedding_model_revision", "")).strip():
            failures.append(
                "content-equivalence embedding-model revision is missing"
            )
        if provenance.get("measurement_scope") != "locked_source_recipient_pairs":
            failures.append("content-equivalence measurement scope is not locked")
        if not _valid_sha256(provenance.get("comparison_manifest_sha256")):
            failures.append("content-equivalence manifest checksum is invalid")

    condition_set = set(intervention_conditions)
    relevant = [
        row
        for row in artifact["records"]
        if row.get("condition") in condition_set
    ]
    if not relevant:
        failures.append("no routing records were available for content equivalence")
        return failures
    manifest = artifact.get("content_equivalence_manifest")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema_version", "records"}
        or manifest.get("schema_version")
        != CONTENT_EQUIVALENCE_MANIFEST_SCHEMA
        or not isinstance(manifest.get("records"), list)
    ):
        failures.append(
            "content-equivalence comparison manifest must use the exact "
            "label-free schema"
        )
        return failures
    manifest_checksum = content_equivalence_manifest_sha256(manifest)
    if isinstance(provenance, Mapping) and (
        provenance.get("comparison_manifest_sha256") != manifest_checksum
    ):
        failures.append(
            "content-equivalence manifest checksum does not match the embedded manifest"
        )

    def manifest_key(row):
        return (
            _id_token(row["item_id"]),
            str(row["condition"]),
            str(row["direction_id"]),
        )

    relevant_by_key = {}
    for row in relevant:
        key = manifest_key(row)
        if key in relevant_by_key:
            failures.append(
                "routing rows contain duplicate content-equivalence keys"
            )
            return failures
        relevant_by_key[key] = row
    manifest_by_key = {}
    for row in manifest["records"]:
        if not isinstance(row, Mapping):
            failures.append("content-equivalence manifest record is invalid")
            return failures
        if set(row) != CONTENT_EQUIVALENCE_RECORD_FIELDS:
            failures.append(
                "content-equivalence manifest records must use the exact "
                "label-free schema"
            )
            return failures
        try:
            key = manifest_key(row)
        except (KeyError, TypeError, ValueError):
            failures.append("content-equivalence manifest key is invalid")
            return failures
        if key in manifest_by_key:
            failures.append(
                "content-equivalence manifest contains duplicate keys"
            )
            return failures
        manifest_by_key[key] = row
    if set(manifest_by_key) != set(relevant_by_key):
        failures.append(
            "content-equivalence manifest does not cover the exact routing rows"
        )
        return failures
    for key, row in relevant_by_key.items():
        manifest_row = manifest_by_key[key]
        try:
            recomputed_content = _recompute_content_comparison(manifest_row)
        except (TypeError, ValueError) as exc:
            failures.append(
                f"content-equivalence manifest cannot be recomputed: {exc}"
            )
            break
        if not (
            _valid_sha256(recomputed_content["source_content_sha256"])
            and _valid_sha256(recomputed_content["recipient_content_sha256"])
        ):
            failures.append(
                "routing rows require canonical source/recipient content checksums"
            )
            break
        exact_fields = (
            "source_content_sha256",
            "recipient_content_sha256",
            "content_token_count_absolute_difference",
        )
        if any(
            manifest_row.get(field) != recomputed_content[field]
            or row.get(field) != recomputed_content[field]
            for field in exact_fields
        ):
            failures.append(
                "routing content-equivalence hashes/counts do not match embedded content"
            )
            break
        try:
            manifest_cosine = float(
                manifest_row["content_embedding_cosine_distance"]
            )
            row_cosine = float(row["content_embedding_cosine_distance"])
        except (KeyError, TypeError, ValueError):
            failures.append(
                "routing content-equivalence cosine distance is missing"
            )
            break
        if not (
            np.isclose(
                manifest_cosine,
                recomputed_content["content_embedding_cosine_distance"],
                atol=1e-12,
                rtol=1e-12,
            )
            and np.isclose(
                row_cosine,
                recomputed_content["content_embedding_cosine_distance"],
                atol=1e-12,
                rtol=1e-12,
            )
        ):
            failures.append(
                "routing embedding distance does not match embedded vectors"
            )
            break
    metrics = lock.get("metrics")
    metric_fields = {
        "token_count_absolute_difference": (
            "content_token_count_absolute_difference",
        ),
        "embedding_cosine_distance": (
            "content_embedding_cosine_distance",
        ),
    }
    if not isinstance(metrics, Mapping):
        failures.append("content-equivalence metric locks are missing")
        return failures
    for metric_name, (record_field,) in metric_fields.items():
        metric_lock = metrics.get(metric_name)
        if not isinstance(metric_lock, Mapping):
            failures.append(
                f"content-equivalence lock for {metric_name} is missing"
            )
            continue
        try:
            values = [float(row[record_field]) for row in relevant]
            reported = float(metric_lock["max_observed"])
            threshold = float(metric_lock["max_threshold"])
        except (KeyError, TypeError, ValueError):
            failures.append(
                f"content-equivalence {metric_name} values/threshold are incomplete"
            )
            continue
        if (
            not np.isfinite(values).all()
            or min(values, default=-1.0) < 0.0
            or not _finite_nonnegative(reported)
            or not _finite_nonnegative(threshold)
        ):
            failures.append(
                f"content-equivalence {metric_name} values must be finite and non-negative"
            )
            continue
        observed = max(values)
        if not np.isclose(observed, reported, atol=1e-12, rtol=1e-12):
            failures.append(
                f"reported content-equivalence {metric_name} maximum does not match records"
            )
        if observed > threshold:
            failures.append(
                f"content-equivalence {metric_name} exceeds its locked threshold"
            )
    return failures


def _condition_values(
    records: Sequence[Mapping[str, Any]],
    condition: str,
    field: str,
) -> dict[str, float]:
    result = {}
    for row in records:
        if row.get("condition") != condition:
            continue
        item_id = _id_token(row["item_id"])
        if item_id in result:
            raise ValueError(f"duplicate {condition!r} record for item {item_id!r}")
        value = float(row[field])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {field} for item {item_id!r}")
        result[item_id] = value
    return result


def _condition_direction_matrix(
    records: Sequence[Mapping[str, Any]],
    condition: str,
    field: str,
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Collect one condition over an identical item-by-direction Cartesian grid."""

    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for row in records:
        if row.get("condition") != condition:
            continue
        item_id = _id_token(row["item_id"])
        direction_id = str(row.get("direction_id") or condition)
        if direction_id in raw[item_id]:
            raise ValueError(
                f"duplicate {condition!r}/{direction_id!r} record for item {item_id!r}"
            )
        value = float(row[field])
        if not np.isfinite(value):
            raise ValueError(f"non-finite {field} for item {item_id!r}")
        raw[item_id][direction_id] = value
    if not raw:
        return {}, set()
    direction_sets = [set(values) for values in raw.values()]
    if any(values != direction_sets[0] for values in direction_sets[1:]):
        raise ValueError(
            f"condition {condition!r} must contain the same directions for every item"
        )
    direction_ids = direction_sets[0]
    return dict(raw), direction_ids


def _condition_direction_values(
    records: Sequence[Mapping[str, Any]],
    condition: str,
    field: str,
) -> tuple[dict[str, float], set[str]]:
    """Average one condition over explicitly identified matched directions."""

    raw, direction_ids = _condition_direction_matrix(records, condition, field)
    return (
        {
            item_id: float(np.mean(list(values.values())))
            for item_id, values in raw.items()
        },
        direction_ids,
    )


def _paired_vectors(
    left: Mapping[str, float],
    right: Mapping[str, float],
    groups: Mapping[str, str | None],
    *,
    item_subset: set[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    common = set(left) & set(right)
    if item_subset is not None:
        common &= item_subset
    item_ids = sorted(common)
    if not item_ids:
        return [], np.array([]), np.array([]), None
    left_values = np.asarray([left[item] for item in item_ids], dtype=float)
    right_values = np.asarray([right[item] for item in item_ids], dtype=float)
    raw_groups = [groups.get(item) for item in item_ids]
    group_values = (
        None
        if any(group is None for group in raw_groups)
        else np.asarray(raw_groups, dtype=object)
    )
    return item_ids, left_values, right_values, group_values


def paired_mean_effect(
    left: Sequence[float],
    right: Sequence[float],
    *,
    groups: Sequence[str] | None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster-bootstrap the paired mean ``left - right``."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("paired inputs must be equal-length one-dimensional vectors")
    if len(left_values) == 0:
        raise ValueError("paired effect requires at least one item")
    differences = left_values - right_values
    rng = np.random.default_rng(seed)
    draws = []
    if groups is None:
        for _ in range(n_boot):
            index = rng.integers(0, len(differences), len(differences))
            draws.append(float(differences[index].mean()))
        bootstrap_unit = "iid_item_pilot"
        n_groups = None
    else:
        group_values = np.asarray(groups, dtype=object)
        if len(group_values) != len(differences):
            raise ValueError("groups must align with paired inputs")
        unique_groups = np.unique(group_values)
        index_by_group = {
            group: np.flatnonzero(group_values == group)
            for group in unique_groups
        }
        for _ in range(n_boot):
            sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
            # Repeated groups are intentionally repeated in a cluster bootstrap draw.
            index = np.concatenate([index_by_group[group] for group in sampled])
            draws.append(float(differences[index].mean()))
        bootstrap_unit = "split_group"
        n_groups = int(len(unique_groups))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "effect": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "n_items": int(len(differences)),
        "n_groups": n_groups,
        "bootstrap_unit": bootstrap_unit,
    }


def _status_positive(
    result: Mapping[str, Any] | None,
    *,
    min_items: int,
    min_groups: int,
    confirmatory_groups: bool,
    design_gate_passed: bool,
) -> str:
    if (
        result is None
        or result["n_items"] < min_items
        or result.get("n_groups") is None
        or result["n_groups"] < min_groups
        or not confirmatory_groups
        or not design_gate_passed
    ):
        return NOT_ADJUDICATED
    return SUPPORTED if result["ci95"][0] > 0.0 else NOT_SUPPORTED


def _combine_required_statuses(statuses: Iterable[str]) -> str:
    values = set(statuses)
    if values == {SUPPORTED}:
        return SUPPORTED
    if NOT_ADJUDICATED in values:
        return NOT_ADJUDICATED
    return NOT_SUPPORTED


def _item_direction_slopes(
    records: Sequence[Mapping[str, Any]],
    direction_kind: str,
) -> dict[tuple[str, str], float]:
    by_item_direction: dict[
        tuple[str, str],
        list[tuple[float, float]],
    ] = defaultdict(list)
    for row in records:
        if row.get("direction_kind") != direction_kind or row.get("alpha") is None:
            continue
        direction_id = str(row.get("direction_id") or direction_kind)
        by_item_direction[(_id_token(row["item_id"]), direction_id)].append(
            (float(row["alpha"]), float(row["answer_logit_margin"]))
        )
    slopes = {}
    for key, points in by_item_direction.items():
        alpha = np.asarray([point[0] for point in points], dtype=float)
        margin = np.asarray([point[1] for point in points], dtype=float)
        if not np.isfinite(alpha).all() or not np.isfinite(margin).all():
            raise ValueError("alpha sweeps contain non-finite values")
        if len(np.unique(alpha)) != len(alpha):
            raise ValueError(
                f"duplicate alpha values for item/direction {key!r}"
            )
        if len(np.unique(alpha)) < 3:
            continue
        slopes[key] = float(np.polyfit(alpha, margin, 1)[0])
    return slopes


def _same_item_sets(*values: Mapping[str, float]) -> bool:
    sets = [set(mapping) for mapping in values]
    return bool(sets) and all(item_set == sets[0] for item_set in sets[1:])


def _direction_item_slopes(
    records: Sequence[Mapping[str, Any]],
    direction_kind: str,
) -> tuple[dict[str, float], set[str]]:
    """Average slopes over locked directions of one kind for each item."""

    raw = _item_direction_slopes(records, direction_kind)
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    for (item_id, direction_id), slope in raw.items():
        by_item[item_id][direction_id] = slope
    if not by_item:
        return {}, set()
    direction_sets = [set(values) for values in by_item.values()]
    if any(values != direction_sets[0] for values in direction_sets[1:]):
        raise ValueError(
            f"direction kind {direction_kind!r} must use the identical direction-ID "
            "set for every item"
        )
    direction_ids = direction_sets[0]
    return (
        {
            item_id: float(np.mean(list(item_slopes.values())))
            for item_id, item_slopes in by_item.items()
        },
        direction_ids,
    )


def _alpha_grid_gate(
    artifact: Mapping[str, Any],
    direction_kinds: Sequence[str],
) -> dict[str, Any]:
    """Require one symmetric preregistered alpha grid for every item and direction."""

    design = artifact.get("design", {})
    locked = design.get("locked_alphas")
    failures: list[str] = []
    if not isinstance(locked, list):
        return {
            "status": NOT_ADJUDICATED,
            "failures": ["design.locked_alphas must be a list"],
        }
    try:
        grid = np.asarray([float(value) for value in locked], dtype=float)
    except (TypeError, ValueError):
        return {
            "status": NOT_ADJUDICATED,
            "failures": ["design.locked_alphas must contain numeric values"],
        }
    if (
        len(grid) < 3
        or not np.isfinite(grid).all()
        or len(np.unique(grid)) != len(grid)
    ):
        failures.append("locked alpha grid must contain at least three unique finite values")
    else:
        grid = np.sort(grid)
        if not np.any(np.isclose(grid, 0.0, atol=1e-12, rtol=0.0)):
            failures.append("locked alpha grid must include zero")
        if not np.allclose(grid, -grid[::-1], atol=1e-12, rtol=0.0):
            failures.append("locked alpha grid must be symmetric around zero")

    observed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in artifact["records"]:
        kind = row.get("direction_kind")
        if kind not in direction_kinds or row.get("alpha") is None:
            continue
        key = (
            str(kind),
            _id_token(row["item_id"]),
            str(row.get("direction_id") or kind),
        )
        observed[key].append(float(row["alpha"]))
    if not observed:
        failures.append("no alpha sweeps were observed")
    elif len(grid) >= 3 and np.isfinite(grid).all():
        for key, values in observed.items():
            candidate = np.sort(np.asarray(values, dtype=float))
            if (
                len(candidate) != len(grid)
                or not np.allclose(candidate, grid, atol=1e-12, rtol=0.0)
            ):
                failures.append(
                    "observed alpha grid differs from the locked grid for "
                    f"direction/item {key!r}"
                )
                break

    return {
        "status": SUPPORTED if not failures else NOT_ADJUDICATED,
        "locked_alphas": [] if not isinstance(locked, list) else locked,
        "failures": failures,
    }


def _direction_alpha_values(
    records: Sequence[Mapping[str, Any]],
    *,
    direction_kind: str,
    alpha: float,
) -> dict[str, float]:
    result = {}
    for row in records:
        if row.get("direction_kind") != direction_kind or row.get("alpha") is None:
            continue
        if not np.isclose(float(row["alpha"]), alpha, atol=1e-12, rtol=0.0):
            continue
        item_id = _id_token(row["item_id"])
        if item_id in result:
            raise ValueError(
                f"multiple {direction_kind!r} directions were observed at alpha={alpha} "
                f"for item {item_id!r}"
            )
        result[item_id] = float(row["answer_logit_margin"])
    return result


def analyze_sufficiency(
    artifact: Mapping[str, Any],
    *,
    control_direction_kinds: Iterable[str] = ("random", "shuffled", "surface"),
    min_items: int = 30,
    min_groups: int = DEFAULT_MIN_GROUPS,
    min_random_directions: int = DEFAULT_MIN_RANDOM_DIRECTIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare fixed target-direction slopes with per-item matched control slopes."""

    validate_artifact(artifact)
    records = artifact["records"]
    groups = _group_map(records)
    design = artifact.get("design", {})
    allowed_control_kinds = tuple(str(kind) for kind in control_direction_kinds)
    declared_control_kinds = design.get(
        "required_control_direction_kinds",
        list(allowed_control_kinds),
    )
    control_failures = []
    if (
        not isinstance(declared_control_kinds, list)
        or not declared_control_kinds
        or any(
            not isinstance(kind, str) or kind not in allowed_control_kinds
            for kind in declared_control_kinds
        )
    ):
        control_failures.append(
            "design.required_control_direction_kinds must be a non-empty subset "
            f"of {list(allowed_control_kinds)}"
        )
        declared_control_kinds = list(allowed_control_kinds)
    if not {"random", "shuffled"}.issubset(set(declared_control_kinds)):
        control_failures.append("random and shuffled controls are both required")

    target, target_direction_ids = _direction_item_slopes(records, "target")
    controls_by_kind = {
        kind: _direction_item_slopes(records, kind)
        for kind in declared_control_kinds
    }
    missing_controls = [
        kind for kind, (values, _) in controls_by_kind.items() if not values
    ]
    if not target or missing_controls:
        return {
            "status": NOT_ADJUDICATED,
            "reason": (
                "target and every preregistered matched-control alpha sweep are required; "
                f"missing={missing_controls}"
            ),
        }
    if len(target_direction_ids) != 1:
        control_failures.append("exactly one frozen target direction is required")
    random_direction_ids = controls_by_kind.get("random", ({}, set()))[1]
    if len(random_direction_ids) < min_random_directions:
        control_failures.append(
            f"observed {len(random_direction_ids)} random directions; "
            f"at least {min_random_directions} are required"
        )
    recorded_random_count = design.get("controls", {}).get(
        "random_direction_count"
    )
    if recorded_random_count != len(random_direction_ids):
        control_failures.append(
            "controls.random_direction_count does not match observed random direction IDs"
        )

    control_sets = [
        controls_by_kind[kind][0]
        for kind in declared_control_kinds
    ]
    if not _same_item_sets(target, *control_sets):
        return {
            "status": NOT_ADJUDICATED,
            "reason": "target and control alpha sweeps must contain exactly the same items",
        }

    controls = {}
    common_controls = set.intersection(*(set(values) for values in control_sets))
    for item_id in common_controls:
        controls[item_id] = float(np.mean([values[item_id] for values in control_sets]))

    items, target_values, zero, group_values = _paired_vectors(
        target,
        {item: 0.0 for item in target},
        groups,
    )
    target_effect = paired_mean_effect(
        target_values,
        zero,
        groups=group_values,
        n_boot=n_boot,
        seed=seed,
    )
    _, paired_target, paired_control, paired_groups = _paired_vectors(
        target,
        controls,
        groups,
    )
    if not len(paired_target):
        return {
            "status": NOT_ADJUDICATED,
            "reason": "target and control sweeps do not share items",
            "target_slope": target_effect,
        }
    contrast = paired_mean_effect(
        paired_target,
        paired_control,
        groups=paired_groups,
        n_boot=n_boot,
        seed=seed + 1,
    )
    contrasts_by_kind = {}
    for control_index, kind in enumerate(declared_control_kinds):
        _, kind_target, kind_control, kind_groups = _paired_vectors(
            target,
            controls_by_kind[kind][0],
            groups,
        )
        contrasts_by_kind[kind] = paired_mean_effect(
            kind_target,
            kind_control,
            groups=kind_groups,
            n_boot=n_boot,
            seed=seed + 10 + control_index,
        )
    raw_random_slopes = _item_direction_slopes(records, "random")
    random_by_direction: dict[str, dict[str, float]] = defaultdict(dict)
    for (item_id, direction_id), slope in raw_random_slopes.items():
        random_by_direction[direction_id][item_id] = slope
    if any(set(values) != set(target) for values in random_by_direction.values()):
        control_failures.append(
            "every random direction must be evaluated on the exact target item set"
        )
    random_control_quantile = design.get("random_control_quantile")
    if (
        not isinstance(random_control_quantile, (int, float))
        or isinstance(random_control_quantile, bool)
        or not np.isfinite(random_control_quantile)
        or not MIN_RANDOM_CONTROL_QUANTILE <= float(random_control_quantile) < 1.0
    ):
        control_failures.append(
            f"random_control_quantile must be in [{MIN_RANDOM_CONTROL_QUANTILE}, 1)"
        )
        random_control_quantile = MIN_RANDOM_CONTROL_QUANTILE
    random_by_item: dict[str, list[float]] = defaultdict(list)
    for (item_id, _), slope in raw_random_slopes.items():
        random_by_item[item_id].append(slope)
    random_quantile_by_item = {
        item_id: float(
            np.quantile(
                slopes,
                float(random_control_quantile),
                method="higher",
            )
        )
        for item_id, slopes in random_by_item.items()
    }
    random_quantile_slope = float(
        np.mean(list(random_quantile_by_item.values()))
    )
    _, target_for_quantile, random_quantile_values, quantile_groups = _paired_vectors(
        target,
        random_quantile_by_item,
        groups,
    )
    random_quantile_contrast = paired_mean_effect(
        target_for_quantile,
        random_quantile_values,
        groups=quantile_groups,
        n_boot=n_boot,
        seed=seed + 2,
    )
    design_gate = _confirmatory_design_gate(
        artifact,
        min_items=min_items,
        min_groups=min_groups,
        min_random_directions=min_random_directions,
        n_boot=n_boot,
    )
    alpha_grid_gate = _alpha_grid_gate(
        artifact,
        ("target", *declared_control_kinds),
    )
    baseline_zero = _condition_values(
        records,
        "baseline_unhooked",
        "answer_logit_margin",
    )
    target_zero = _direction_alpha_values(
        records,
        direction_kind="target",
        alpha=0.0,
    )
    zero_identity_max_difference = None
    if not baseline_zero or not _same_item_sets(baseline_zero, target_zero):
        control_failures.append(
            "unhooked baseline and target alpha-zero rows must contain the exact same items"
        )
    else:
        zero_identity_max_difference = float(
            max(
                abs(target_zero[item] - baseline_zero[item])
                for item in baseline_zero
            )
        )
    zero_identity = design.get("alpha_zero_identity")
    if not isinstance(zero_identity, Mapping):
        control_failures.append("design.alpha_zero_identity is missing")
    else:
        tolerance = zero_identity.get("tolerance")
        if not _finite_nonnegative(tolerance):
            control_failures.append(
                "alpha-zero identity tolerance must be finite and non-negative"
            )
        elif (
            zero_identity_max_difference is None
            or zero_identity_max_difference > float(tolerance)
        ):
            control_failures.append(
                "observed target alpha-zero output differs from the unhooked baseline"
            )
        reported = zero_identity.get("max_abs_difference_observed")
        if (
            zero_identity_max_difference is not None
            and (
                not _finite_nonnegative(reported)
                or not np.isclose(
                    float(reported),
                    zero_identity_max_difference,
                    atol=1e-12,
                    rtol=1e-12,
                )
            )
        ):
            control_failures.append(
                "reported alpha-zero identity difference does not match the records"
            )
    measurement_gate_passed = (
        not control_failures
        and design_gate["status"] == SUPPORTED
        and alpha_grid_gate["status"] == SUPPORTED
    )
    confirmatory = artifact.get("split_group_scope") not in NONCONFIRMATORY_GROUP_SCOPES
    target_status = _status_positive(
        target_effect,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    contrast_statuses = {
        _status_positive(
            result,
            min_items=min_items,
            min_groups=min_groups,
            confirmatory_groups=confirmatory,
            design_gate_passed=measurement_gate_passed,
        )
        for result in contrasts_by_kind.values()
    }
    random_quantile_status = _status_positive(
        random_quantile_contrast,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    return {
        "status": _combine_required_statuses(
            (target_status, random_quantile_status, *contrast_statuses)
        ),
        "target_slope": target_effect,
        "target_minus_control_slope": contrast,
        "target_minus_each_control_slope": contrasts_by_kind,
        "target_minus_random_quantile_slope": random_quantile_contrast,
        "random_control_quantile": float(random_control_quantile),
        "random_control_quantile_slope": random_quantile_slope,
        "alpha_zero_identity": {
            "max_abs_difference_observed": zero_identity_max_difference,
            "tolerance": (
                None
                if not isinstance(zero_identity, Mapping)
                else zero_identity.get("tolerance")
            ),
        },
        "target_items": len(items),
        "target_direction_count": len(target_direction_ids),
        "random_direction_count": len(random_direction_ids),
        "design_gate": design_gate,
        "alpha_grid_gate": alpha_grid_gate,
        "measurement_gate_failures": control_failures,
        "claim": "causal_availability_not_natural_use",
    }


def analyze_erasure_rescue(
    artifact: Mapping[str, Any],
    *,
    min_items: int = 30,
    min_groups: int = DEFAULT_MIN_GROUPS,
    min_random_directions: int = DEFAULT_MIN_RANDOM_DIRECTIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 10,
) -> dict[str, Any]:
    """Test necessity and rescue on naturally correct baseline items."""

    validate_artifact(artifact)
    records = artifact["records"]
    groups = _group_map(records)
    design = artifact.get("design", {})
    baseline = _condition_values(records, "baseline", "correct_answer_margin")
    erased = _condition_values(records, "erased_target", "correct_answer_margin")
    rescued = _condition_values(records, "rescued_target", "correct_answer_margin")
    random_erased_matrix, random_direction_ids = _condition_direction_matrix(
        records,
        "erased_random",
        "correct_answer_margin",
    )
    random_erased = {
        item_id: float(np.mean(list(values.values())))
        for item_id, values in random_erased_matrix.items()
    }
    required = (baseline, erased, rescued, random_erased)
    if any(not values for values in required):
        return {
            "status": NOT_ADJUDICATED,
            "reason": (
                "baseline, erased_target, rescued_target, and erased_random conditions "
                "are required"
            ),
        }
    if not _same_item_sets(*required):
        return {
            "status": NOT_ADJUDICATED,
            "reason": "all erasure/rescue conditions must contain exactly the same items",
        }

    naturally_correct = {item for item, value in baseline.items() if value > 0.0}
    _, base_values, erased_values, group_values = _paired_vectors(
        baseline,
        erased,
        groups,
        item_subset=naturally_correct,
    )
    if not len(base_values):
        return {
            "status": NOT_ADJUDICATED,
            "reason": "no naturally correct baseline items are shared across conditions",
        }
    necessity = paired_mean_effect(
        base_values,
        erased_values,
        groups=group_values,
        n_boot=n_boot,
        seed=seed,
    )

    target_damage = {
        item: baseline[item] - erased[item]
        for item in set(baseline) & set(erased) & naturally_correct
    }
    random_damage = {
        item: baseline[item] - random_erased[item]
        for item in set(baseline) & set(random_erased) & naturally_correct
    }
    _, target_damage_values, random_damage_values, damage_groups = _paired_vectors(
        target_damage,
        random_damage,
        groups,
    )
    specificity = (
        None
        if not len(target_damage_values)
        else paired_mean_effect(
            target_damage_values,
            random_damage_values,
            groups=damage_groups,
            n_boot=n_boot,
            seed=seed + 1,
        )
    )
    random_control_quantile = design.get("random_control_quantile")
    quantile_failure = None
    if (
        not isinstance(random_control_quantile, (int, float))
        or isinstance(random_control_quantile, bool)
        or not np.isfinite(random_control_quantile)
        or not MIN_RANDOM_CONTROL_QUANTILE <= float(random_control_quantile) < 1.0
    ):
        quantile_failure = (
            f"random_control_quantile must be in [{MIN_RANDOM_CONTROL_QUANTILE}, 1)"
        )
        random_control_quantile = MIN_RANDOM_CONTROL_QUANTILE
    random_damage_quantile = {
        item: float(
            np.quantile(
                [
                    baseline[item] - random_margin
                    for random_margin in random_erased_matrix[item].values()
                ],
                float(random_control_quantile),
                method="higher",
            )
        )
        for item in naturally_correct
    }
    _, target_for_quantile, random_quantile_values, quantile_groups = _paired_vectors(
        target_damage,
        random_damage_quantile,
        groups,
    )
    quantile_specificity = paired_mean_effect(
        target_for_quantile,
        random_quantile_values,
        groups=quantile_groups,
        n_boot=n_boot,
        seed=seed + 2,
    )

    _, rescued_values, re_erased_values, rescue_groups = _paired_vectors(
        rescued,
        erased,
        groups,
        item_subset=naturally_correct,
    )
    rescue = paired_mean_effect(
        rescued_values,
        re_erased_values,
        groups=rescue_groups,
        n_boot=n_boot,
        seed=seed + 3,
    )

    confirmatory = artifact.get("split_group_scope") not in NONCONFIRMATORY_GROUP_SCOPES
    design_gate = _confirmatory_design_gate(
        artifact,
        min_items=min_items,
        min_groups=min_groups,
        min_random_directions=min_random_directions,
        n_boot=n_boot,
    )
    measurement_failures = []
    contract = artifact["contract"]
    locked_erasure_strength = contract.get("erasure_strength")
    if design.get("locked_erasure_strength") != locked_erasure_strength:
        measurement_failures.append(
            "design.locked_erasure_strength differs from the intervention contract"
        )
    erasure_rows = [
        row
        for row in records
        if row.get("condition")
        in {"erased_target", "erased_random", "rescued_target"}
    ]
    for row in erasure_rows:
        try:
            observed_strength = float(row["erasure_strength"])
        except (KeyError, TypeError, ValueError):
            measurement_failures.append(
                "every target/random erasure row requires erasure_strength"
            )
            break
        if not np.isclose(
            observed_strength,
            float(locked_erasure_strength),
            atol=1e-12,
            rtol=1e-12,
        ):
            measurement_failures.append(
                "target and random erasures do not share the locked strength"
            )
            break
        if row.get("erasure_center_sha256") != contract.get(
            "direction_center_sha256"
        ):
            measurement_failures.append(
                "erasure center checksum differs from the train-derived contract center"
            )
            break
    if quantile_failure is not None:
        measurement_failures.append(quantile_failure)
    if len(random_direction_ids) < min_random_directions:
        measurement_failures.append(
            f"observed {len(random_direction_ids)} random erasure directions; "
            f"at least {min_random_directions} are required"
        )
    recorded_random_count = design.get("controls", {}).get(
        "random_direction_count"
    )
    if recorded_random_count != len(random_direction_ids):
        measurement_failures.append(
            "controls.random_direction_count does not match observed random erasure IDs"
        )
    nontrivial_rescue = (
        design.get("rescue_kind")
        == "held_out_counterfactual_source"
    )
    if not nontrivial_rescue:
        measurement_failures.append(
            "rescue_kind must be held_out_counterfactual_source"
        )
    if design.get("rescue_source_selection_scope") != "outer_train_only":
        measurement_failures.append(
            "rescue source selection must be fixed on the outer training fold"
        )
    measurement_failures.extend(
        _source_activation_gate(
            records,
            expected_source_conditions={
                "rescued_target": "held_out_counterfactual",
            },
            locked_strength=design.get("locked_rescue_strength"),
        )
    )
    measurement_failures.extend(
        _activation_capture_manifest_gate(
            artifact,
            expected_source_conditions={
                "rescued_target": "held_out_counterfactual",
            },
        )
    )
    measurement_gate_passed = (
        design_gate["status"] == SUPPORTED and not measurement_failures
    )
    necessity_status = _status_positive(
        necessity,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    specificity_status = _status_positive(
        specificity,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    rescue_status = _status_positive(
        rescue,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    quantile_specificity_status = _status_positive(
        quantile_specificity,
        min_items=min_items,
        min_groups=min_groups,
        confirmatory_groups=confirmatory,
        design_gate_passed=measurement_gate_passed,
    )
    statuses = {
        necessity_status,
        specificity_status,
        quantile_specificity_status,
        rescue_status,
    }
    return {
        "status": _combine_required_statuses(statuses),
        "necessity_base_minus_erased": necessity,
        "target_minus_random_erasure_damage": specificity,
        "target_minus_random_quantile_erasure_damage": quantile_specificity,
        "random_control_quantile": float(random_control_quantile),
        "rescue_minus_erased": rescue,
        "naturally_correct_items": len(naturally_correct),
        "random_direction_count": len(random_direction_ids),
        "rescue_kind": design.get("rescue_kind"),
        "design_gate": design_gate,
        "measurement_gate_failures": measurement_failures,
        "claim": "partial_natural_use_if_supported",
    }


def _expression_gap_lock_gate(
    artifact: Mapping[str, Any],
    native: Mapping[str, float],
    elicited: Mapping[str, float],
) -> tuple[list[str], list[str]]:
    """Validate the pre-intervention expression-gap cohort and observed membership."""

    design_failures = []
    empirical_failures = []
    lock = artifact.get("design", {}).get("expression_gap")
    if not isinstance(lock, Mapping):
        return ["design.expression_gap is missing"], empirical_failures
    if lock.get("preregistered") is not True:
        design_failures.append("expression-gap cohort was not preregistered")
    if lock.get("selection_scope") != "behavioral_run_before_intervention":
        design_failures.append(
            "expression-gap cohort must be locked before intervention execution"
        )
    if lock.get("subset_rule") != "native_incorrect_elicited_correct":
        design_failures.append(
            "expression-gap subset rule must be native_incorrect_elicited_correct"
        )
    if lock.get("item_set_sha256") != item_set_sha256(artifact["records"]):
        design_failures.append(
            "expression-gap item-set checksum does not match the causal records"
        )
    try:
        native_ceiling = float(lock["native_correct_margin_ceiling"])
        elicited_floor = float(lock["elicited_correct_margin_floor"])
    except (KeyError, TypeError, ValueError):
        design_failures.append("expression-gap margin thresholds are incomplete")
        return design_failures, empirical_failures
    if (
        not np.isfinite([native_ceiling, elicited_floor]).all()
        or native_ceiling > 0.0
        or elicited_floor < 0.0
    ):
        design_failures.append(
            "expression-gap thresholds must require native incorrectness and "
            "elicited correctness"
        )
        return design_failures, empirical_failures
    native_violations = [
        item for item, value in native.items() if value > native_ceiling
    ]
    elicited_violations = [
        item for item, value in elicited.items() if value <= elicited_floor
    ]
    if native_violations:
        empirical_failures.append(
            f"{len(native_violations)} locked items are not native failures"
        )
    if elicited_violations:
        empirical_failures.append(
            f"{len(elicited_violations)} locked items are not elicited successes"
        )
    return design_failures, empirical_failures


def analyze_routing(
    artifact: Mapping[str, Any],
    *,
    min_items: int = 30,
    min_groups: int = DEFAULT_MIN_GROUPS,
    min_random_directions: int = DEFAULT_MIN_RANDOM_DIRECTIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 20,
) -> dict[str, Any]:
    """Test a bidirectional content/routing bottleneck."""

    validate_artifact(artifact)
    records = artifact["records"]
    groups = _group_map(records)
    intervention_pairs = _intervention_pair_map(records)
    design = artifact.get("design", {})
    names = (
        "native",
        "elicited",
        "route_success_to_native",
        "route_native_to_success",
        "route_shuffled_to_native",
        "route_shuffled_to_success",
        "no_content_no_route",
        "route_only_no_content",
        "content_and_route",
    )
    values = {
        name: _condition_values(records, name, "correct_answer_margin")
        for name in names
    }
    random_native_matrix, random_native_ids = _condition_direction_matrix(
        records,
        "route_random_to_native",
        "correct_answer_margin",
    )
    random_success_matrix, random_success_ids = _condition_direction_matrix(
        records,
        "route_random_to_success",
        "correct_answer_margin",
    )
    random_native = {
        item_id: float(np.mean(list(direction_values.values())))
        for item_id, direction_values in random_native_matrix.items()
    }
    random_success = {
        item_id: float(np.mean(list(direction_values.values())))
        for item_id, direction_values in random_success_matrix.items()
    }
    if any(not values[name] for name in names) or not random_native or not random_success:
        return {
            "status": NOT_ADJUDICATED,
            "reason": (
                f"routing analysis requires conditions: {list(names)} plus "
                "route_random_to_native and route_random_to_success"
            ),
        }
    if not _same_item_sets(
        *(values[name] for name in names),
        random_native,
        random_success,
    ):
        return {
            "status": NOT_ADJUDICATED,
            "reason": "all routing conditions must contain exactly the same paired items",
        }

    def effect(left_name, right_name, effect_seed):
        _, left, right, group_values = _paired_vectors(
            values[left_name],
            values[right_name],
            groups,
        )
        return paired_mean_effect(
            left,
            right,
            groups=group_values,
            n_boot=n_boot,
            seed=effect_seed,
        )

    forward = effect("route_success_to_native", "native", seed)
    reverse = effect("elicited", "route_native_to_success", seed + 1)
    expression_gap = effect("elicited", "native", seed + 9)
    _, forward_target, forward_random, forward_random_groups = _paired_vectors(
        values["route_success_to_native"],
        random_native,
        groups,
    )
    forward_random_specificity = paired_mean_effect(
        forward_target,
        forward_random,
        groups=forward_random_groups,
        n_boot=n_boot,
        seed=seed + 2,
    )
    forward_shuffled_specificity = effect(
        "route_success_to_native",
        "route_shuffled_to_native",
        seed + 3,
    )
    _, reverse_random, reverse_target, reverse_random_groups = _paired_vectors(
        random_success,
        values["route_native_to_success"],
        groups,
    )
    reverse_random_specificity = paired_mean_effect(
        reverse_random,
        reverse_target,
        groups=reverse_random_groups,
        n_boot=n_boot,
        seed=seed + 4,
    )
    reverse_shuffled_specificity = effect(
        "route_shuffled_to_success",
        "route_native_to_success",
        seed + 5,
    )
    random_control_quantile = design.get("random_control_quantile")
    quantile_failure = None
    if (
        not isinstance(random_control_quantile, (int, float))
        or isinstance(random_control_quantile, bool)
        or not np.isfinite(random_control_quantile)
        or not MIN_RANDOM_CONTROL_QUANTILE <= float(random_control_quantile) < 1.0
    ):
        quantile_failure = (
            f"random_control_quantile must be in [{MIN_RANDOM_CONTROL_QUANTILE}, 1)"
        )
        random_control_quantile = MIN_RANDOM_CONTROL_QUANTILE
    forward_random_quantile = {
        item: float(
            np.quantile(
                list(direction_values.values()),
                float(random_control_quantile),
                method="higher",
            )
        )
        for item, direction_values in random_native_matrix.items()
    }
    _, forward_target_quantile, forward_null_quantile, forward_quantile_groups = (
        _paired_vectors(
            values["route_success_to_native"],
            forward_random_quantile,
            groups,
        )
    )
    forward_quantile_specificity = paired_mean_effect(
        forward_target_quantile,
        forward_null_quantile,
        groups=forward_quantile_groups,
        n_boot=n_boot,
        seed=seed + 6,
    )
    target_reverse_damage = {
        item: values["elicited"][item] - values["route_native_to_success"][item]
        for item in values["elicited"]
    }
    random_reverse_damage_quantile = {
        item: float(
            np.quantile(
                [
                    values["elicited"][item] - random_margin
                    for random_margin in direction_values.values()
                ],
                float(random_control_quantile),
                method="higher",
            )
        )
        for item, direction_values in random_success_matrix.items()
    }
    _, reverse_target_quantile, reverse_null_quantile, reverse_quantile_groups = (
        _paired_vectors(
            target_reverse_damage,
            random_reverse_damage_quantile,
            groups,
        )
    )
    reverse_quantile_specificity = paired_mean_effect(
        reverse_target_quantile,
        reverse_null_quantile,
        groups=reverse_quantile_groups,
        n_boot=n_boot,
        seed=seed + 7,
    )
    interaction_items = (
        set(values["content_and_route"])
        & set(values["native"])
        & set(values["route_only_no_content"])
        & set(values["no_content_no_route"])
    )
    interaction_values = {
        item: (
            values["content_and_route"][item]
            - values["native"][item]
            - values["route_only_no_content"][item]
            + values["no_content_no_route"][item]
        )
        for item in interaction_items
    }
    _, interaction_array, zero_array, interaction_groups = _paired_vectors(
        interaction_values,
        {item: 0.0 for item in interaction_values},
        groups,
    )
    interaction = paired_mean_effect(
        interaction_array,
        zero_array,
        groups=interaction_groups,
        n_boot=n_boot,
        seed=seed + 8,
    )
    design_gate = _confirmatory_design_gate(
        artifact,
        min_items=min_items,
        min_groups=min_groups,
        min_random_directions=min_random_directions,
        n_boot=n_boot,
    )
    measurement_failures = []
    expression_design_failures, expression_empirical_failures = (
        _expression_gap_lock_gate(
            artifact,
            values["native"],
            values["elicited"],
        )
    )
    measurement_failures.extend(expression_design_failures)
    if quantile_failure is not None:
        measurement_failures.append(quantile_failure)
    if any(pair_id is None for pair_id in intervention_pairs.values()):
        measurement_failures.append(
            "every routing item requires a non-null intervention_pair_id"
        )
    if random_native_ids != random_success_ids:
        measurement_failures.append(
            "forward and reverse routing controls use different random direction IDs"
        )
    if len(random_native_ids) < min_random_directions:
        measurement_failures.append(
            f"observed {len(random_native_ids)} routing-control directions; "
            f"at least {min_random_directions} are required"
        )
    recorded_random_count = design.get("controls", {}).get(
        "random_direction_count"
    )
    if recorded_random_count != len(random_native_ids):
        measurement_failures.append(
            "controls.random_direction_count does not match observed routing-control IDs"
        )
    pairing = design.get("intervention_pairing")
    if not isinstance(pairing, Mapping):
        measurement_failures.append("design.intervention_pairing is missing")
    else:
        if pairing.get("test_label_free") is not True:
            measurement_failures.append(
                "routing source/recipient pairing must be test-label-free"
            )
        if pairing.get("selection_scope") != "outer_train_only":
            measurement_failures.append(
                "routing source/recipient pairing must be fixed on outer training"
            )
    source_condition_lock = {
        "route_success_to_native": "elicited",
        "route_native_to_success": "native",
        "route_random_to_native": "random_control",
        "route_random_to_success": "random_control",
        "route_shuffled_to_native": "shuffled_control",
        "route_shuffled_to_success": "shuffled_control",
        "route_only_no_content": "elicited_route",
        "content_and_route": "elicited_route",
    }
    measurement_failures.extend(
        _content_equivalence_gate(
            artifact,
            intervention_conditions=source_condition_lock,
        )
    )
    if design.get("factorial_cells_locked") is not True:
        measurement_failures.append("the full 2x2 factorial cells were not locked")
    measurement_failures.extend(
        _source_activation_gate(
            records,
            expected_source_conditions=source_condition_lock,
            locked_strength=design.get("locked_patch_strength"),
        )
    )
    measurement_failures.extend(
        _activation_capture_manifest_gate(
            artifact,
            expected_source_conditions=source_condition_lock,
        )
    )
    measurement_gate_passed = (
        design_gate["status"] == SUPPORTED and not measurement_failures
    )
    confirmatory = artifact.get("split_group_scope") not in NONCONFIRMATORY_GROUP_SCOPES
    statuses = {
        _status_positive(
            result,
            min_items=min_items,
            min_groups=min_groups,
            confirmatory_groups=confirmatory,
            design_gate_passed=measurement_gate_passed,
        )
        for result in (forward, reverse, interaction)
    }
    expression_gap_status = (
        NOT_SUPPORTED
        if expression_empirical_failures
        else _status_positive(
            expression_gap,
            min_items=min_items,
            min_groups=min_groups,
            confirmatory_groups=confirmatory,
            design_gate_passed=measurement_gate_passed,
        )
    )
    statuses.add(expression_gap_status)
    for specificity_result in (
        forward_random_specificity,
        forward_shuffled_specificity,
        reverse_random_specificity,
        reverse_shuffled_specificity,
        forward_quantile_specificity,
        reverse_quantile_specificity,
    ):
        statuses.add(
            _status_positive(
                specificity_result,
                min_items=min_items,
                min_groups=min_groups,
                confirmatory_groups=confirmatory,
                design_gate_passed=measurement_gate_passed,
            )
        )
    return {
        "status": _combine_required_statuses(statuses),
        "elicited_minus_native_expression_gap": expression_gap,
        "expression_gap_empirical_failures": expression_empirical_failures,
        "successful_route_into_native": forward,
        "native_route_into_success_damage": reverse,
        "target_minus_random_route_into_native": forward_random_specificity,
        "target_minus_random_quantile_route_into_native": (
            forward_quantile_specificity
        ),
        "target_minus_shuffled_route_into_native": forward_shuffled_specificity,
        "random_minus_target_route_into_success": reverse_random_specificity,
        "target_minus_random_quantile_route_damage_into_success": (
            reverse_quantile_specificity
        ),
        "shuffled_minus_target_route_into_success": reverse_shuffled_specificity,
        "content_by_route_difference_in_differences": interaction,
        "random_control_quantile": float(random_control_quantile),
        "random_direction_count": len(random_native_ids),
        "design_gate": design_gate,
        "measurement_gate_failures": measurement_failures,
        "claim": "routing_bottleneck_if_supported",
    }


def analyze_artifact(
    artifact: Mapping[str, Any],
    *,
    min_items: int = 30,
    min_groups: int = DEFAULT_MIN_GROUPS,
    min_random_directions: int = DEFAULT_MIN_RANDOM_DIRECTIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    validate_artifact(artifact)
    family = artifact.get("intervention_family")
    result: dict[str, Any] = {
        "schema_version": CAUSAL_ARTIFACT_SCHEMA,
        "task_id": artifact.get("contract", {}).get("task_id"),
        "model_revision": artifact.get("contract", {}).get("model_revision"),
        "intervention_family": family,
        "split_group_scope": artifact.get("split_group_scope"),
        "claim_boundary": (
            "single-model/task intervention result; cross-model generality and a law-like "
            "relationship require separately preregistered replication"
        ),
        "mediator_warning": (
            "component patch effects can include interactions with other mediators and must not "
            "be summed as an additive causal decomposition"
        ),
    }
    if family == "steering":
        result["sufficiency"] = analyze_sufficiency(
            artifact,
            min_items=min_items,
            min_groups=min_groups,
            min_random_directions=min_random_directions,
            n_boot=n_boot,
            seed=seed,
        )
        result["overall_status"] = (
            "CAUSAL_AVAILABILITY_SUPPORTED"
            if result["sufficiency"]["status"] == SUPPORTED
            else result["sufficiency"]["status"]
        )
    elif family == "erasure_rescue":
        result["natural_use"] = analyze_erasure_rescue(
            artifact,
            min_items=min_items,
            min_groups=min_groups,
            min_random_directions=min_random_directions,
            n_boot=n_boot,
            seed=seed,
        )
        result["overall_status"] = (
            "PARTIAL_NATURAL_USE_SUPPORTED"
            if result["natural_use"]["status"] == SUPPORTED
            else result["natural_use"]["status"]
        )
    elif family == "routing_patch":
        result["routing"] = analyze_routing(
            artifact,
            min_items=min_items,
            min_groups=min_groups,
            min_random_directions=min_random_directions,
            n_boot=n_boot,
            seed=seed,
        )
        result["overall_status"] = (
            "ROUTING_BOTTLENECK_SUPPORTED"
            if result["routing"]["status"] == SUPPORTED
            else result["routing"]["status"]
        )
    else:
        result["overall_status"] = NOT_ADJUDICATED
        result["reason"] = (
            "content-patch measurements require a preregistered sufficiency or mediation "
            "contrast before assigning a causal status"
        )
    return result


def analyze_causal_suite(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    min_items: int = 30,
    min_groups: int = DEFAULT_MIN_GROUPS,
    min_random_directions: int = DEFAULT_MIN_RANDOM_DIRECTIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Adjudicate the combined activation-gap claim across all three evidence families."""

    required_families = ("steering", "erasure_rescue", "routing_patch")
    by_family: dict[str, Mapping[str, Any]] = {}
    failures = []
    for artifact in artifacts:
        validate_artifact(artifact)
        family = str(artifact.get("intervention_family"))
        if family in by_family:
            failures.append(f"duplicate intervention family {family!r}")
        else:
            by_family[family] = artifact
    missing = [family for family in required_families if family not in by_family]
    if missing:
        failures.append(f"missing intervention families: {missing}")

    compatibility_fields = (
        "task_id",
        "model_revision",
        "fold_id",
        "layer_index",
        "token_position",
        "direction_artifact_sha256",
        "decoder_evaluation_artifact_sha256",
        "applied_direction_sha256",
        "direction_center_sha256",
        "erasure_strength",
        "steering_dose_scale",
    )
    if not missing:
        reference = by_family[required_families[0]]
        reference_contract = reference.get("contract", {})
        reference_scope = reference.get("split_group_scope")
        reference_items = {
            _id_token(row["item_id"]) for row in reference["records"]
        }
        reference_groups = _group_map(reference["records"])
        reference_pairs = _intervention_pair_map(reference["records"])
        reference_labels = _target_label_map(reference["records"])
        reference_suite_id = reference.get("design", {}).get("causal_suite_id")
        if not isinstance(reference_suite_id, str) or not reference_suite_id.strip():
            failures.append("design.causal_suite_id is missing")
        for family in required_families[1:]:
            candidate = by_family[family]
            candidate_contract = candidate.get("contract", {})
            mismatched = [
                field
                for field in compatibility_fields
                if candidate_contract.get(field) != reference_contract.get(field)
            ]
            if mismatched:
                failures.append(
                    f"{family} contract differs on fields: {mismatched}"
                )
            if candidate.get("split_group_scope") != reference_scope:
                failures.append(f"{family} split_group_scope differs")
            candidate_items = {
                _id_token(row["item_id"]) for row in candidate["records"]
            }
            if candidate_items != reference_items:
                failures.append(
                    f"{family} does not contain the exact same held-out items"
                )
            if _group_map(candidate["records"]) != reference_groups:
                failures.append(
                    f"{family} item-to-split-group assignments differ"
                )
            if _intervention_pair_map(candidate["records"]) != reference_pairs:
                failures.append(
                    f"{family} item-to-intervention-pair assignments differ"
                )
            if _target_label_map(candidate["records"]) != reference_labels:
                failures.append(f"{family} item target labels differ")
            if (
                candidate.get("design", {}).get("causal_suite_id")
                != reference_suite_id
            ):
                failures.append(f"{family} causal_suite_id differs")

    component_analyses = {
        family: analyze_artifact(
            by_family[family],
            min_items=min_items,
            min_groups=min_groups,
            min_random_directions=min_random_directions,
            n_boot=n_boot,
            seed=seed + index * 100,
        )
        for index, family in enumerate(required_families)
        if family in by_family
    }
    expected_statuses = {
        "steering": "CAUSAL_AVAILABILITY_SUPPORTED",
        "erasure_rescue": "PARTIAL_NATURAL_USE_SUPPORTED",
        "routing_patch": "ROUTING_BOTTLENECK_SUPPORTED",
    }
    component_statuses = {
        family: analysis["overall_status"]
        for family, analysis in component_analyses.items()
    }
    if failures or missing or any(
        component_statuses.get(family) == NOT_ADJUDICATED
        for family in required_families
    ):
        overall_status = NOT_ADJUDICATED
    elif all(
        component_statuses.get(family) == expected
        for family, expected in expected_statuses.items()
    ):
        overall_status = "CAUSAL_ACTIVATION_GAP_SUPPORTED"
    else:
        overall_status = NOT_SUPPORTED
    return {
        "schema_version": CAUSAL_ARTIFACT_SCHEMA,
        "analysis_type": "causal_activation_gap_suite",
        "overall_status": overall_status,
        "component_statuses": component_statuses,
        "compatibility_failures": failures,
        "components": component_analyses,
        "claim_boundary": (
            "the combined status is model-, task-, direction-, fold-, and item-set-specific; "
            "law-like or latent-truth claims require independent transfer and replication"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="+",
        help=(
            "one causal artifact, or steering + erasure-rescue + routing artifacts "
            "for suite adjudication"
        ),
    )
    parser.add_argument("--out", help="analysis JSON path; defaults beside the artifact")
    parser.add_argument("--min-items", type=int, default=30)
    parser.add_argument("--min-groups", type=int, default=DEFAULT_MIN_GROUPS)
    parser.add_argument(
        "--min-random-directions",
        type=int,
        default=DEFAULT_MIN_RANDOM_DIRECTIONS,
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifacts = [load_artifact(path) for path in args.artifact]
    if len(artifacts) == 1:
        result = analyze_artifact(
            artifacts[0],
            min_items=args.min_items,
            min_groups=args.min_groups,
            min_random_directions=args.min_random_directions,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    else:
        result = analyze_causal_suite(
            artifacts,
            min_items=args.min_items,
            min_groups=args.min_groups,
            min_random_directions=args.min_random_directions,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    output = (
        Path(args.out)
        if args.out
        else (
            Path(args.artifact[0]).with_name(
                Path(args.artifact[0]).stem + "_analysis.json"
            )
            if len(artifacts) == 1
            else Path(args.artifact[0]).with_name("causal_suite_analysis.json")
        )
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {output}: {result['overall_status']}")


if __name__ == "__main__":
    main()

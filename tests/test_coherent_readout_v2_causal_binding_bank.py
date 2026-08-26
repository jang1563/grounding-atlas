from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTAX_DIR = ROOT / "signal" / "syntax"
sys.path.insert(0, str(SYNTAX_DIR))

import build_coherent_readout_v2_causal_binding_bank as builder  # noqa: E402

FIXTURE_PATH = SYNTAX_DIR / "coherent_readout_v2_causal_binding_bank.json"
MANIFEST_PATH = SYNTAX_DIR / "coherent_readout_v2_causal_binding_bank.manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_static_bank_is_deterministic_48_pairs_and_96_items() -> None:
    fixture = _json(FIXTURE_PATH)
    expected_pairs = [
        {
            "pair_id": pair_id,
            "cluster_id": pair_id,
            "positive_class": positive_class,
            "negative_class": negative_class,
        }
        for pair_id, positive_class, negative_class in builder.PAIR_DEFINITIONS
    ]

    assert fixture == builder.build_fixture()
    assert builder.validate_fixture(fixture) == fixture
    assert fixture["schema_version"] == builder.FIXTURE_SCHEMA
    assert fixture["analysis_id"] == builder.ANALYSIS_ID
    assert fixture["inferential_unit"] == "lexical_pair"
    assert fixture["pair_registry"] == expected_pairs
    assert len(fixture["pair_registry"]) == 48
    assert len(fixture["items"]) == 96
    assert [row["pair_id"] for row in fixture["pair_registry"]] == [
        pair_id for pair_id, _, _ in builder.PAIR_DEFINITIONS
    ]
    assert [row["item_id"] for row in fixture["items"]] == sorted(
        row["item_id"] for row in fixture["items"]
    )
    assert Counter(row["pair_id"] for row in fixture["items"]) == {
        pair_id: 2 for pair_id, _, _ in builder.PAIR_DEFINITIONS
    }
    assert Counter(row["truth_polarity"] for row in fixture["items"]) == {
        "positive": 48,
        "negative": 48,
    }
    assert Counter(
        (row["pair_id"], row["truth_polarity"]) for row in fixture["items"]
    ) == Counter(
        (pair_id, polarity)
        for pair_id, _, _ in builder.PAIR_DEFINITIONS
        for polarity in ("positive", "negative")
    )


def test_holdout_lexicon_is_disjoint_from_locked_discovery_bank() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)
    discovery = _json(builder.DISCOVERY_FIXTURE)
    discovery_reference = fixture["discovery_reference"]
    holdout_pair_ids = {row["pair_id"] for row in fixture["pair_registry"]}
    discovery_pair_ids = {row["pair_id"] for row in discovery["pair_registry"]}
    holdout_words = {
        word
        for row in fixture["pair_registry"]
        for word in (row["positive_class"], row["negative_class"])
    }
    discovery_words = {
        word
        for row in discovery["pair_registry"]
        for word in (row["positive_class"], row["negative_class"])
    }

    assert _sha256(builder.DISCOVERY_FIXTURE) == builder.DISCOVERY_FIXTURE_SHA256
    assert discovery_reference == builder._load_discovery_reference()
    assert discovery_reference["file_sha256"] == builder.DISCOVERY_FIXTURE_SHA256
    assert discovery_reference["canonical_sha256"] == builder.canonical_sha256(
        discovery
    )
    assert discovery_reference["class_words"] == sorted(discovery_words)
    assert discovery_reference["role"] == (
        "layer_localization_only_no_holdout_inference"
    )
    assert manifest["provenance"]["discovery_reference"] == discovery_reference
    assert len(holdout_words) == 96
    assert holdout_words.isdisjoint(discovery_words)
    assert holdout_pair_ids.isdisjoint(discovery_pair_ids)


def test_same_pair_counterfactual_controls_are_bidirectional() -> None:
    fixture = _json(FIXTURE_PATH)
    by_item = {row["item_id"]: row for row in fixture["items"]}

    assert len(by_item) == 96
    for item in fixture["items"]:
        counterfactual = by_item[item["same_pair_counterfactual_item_id"]]
        assert counterfactual["pair_id"] == item["pair_id"]
        assert counterfactual["cluster_id"] == item["cluster_id"]
        assert counterfactual["truth_polarity"] != item["truth_polarity"]
        assert counterfactual["declared_class"] != item["declared_class"]
        assert counterfactual["same_pair_counterfactual_item_id"] == item["item_id"]
        assert {
            counterfactual["declared_class"],
            item["declared_class"],
        } == {item["positive_class"], item["negative_class"]}


def test_unrelated_controls_are_24_reciprocal_same_polarity_dyads() -> None:
    fixture = _json(FIXTURE_PATH)
    pair_ids = [pair_id for pair_id, _, _ in builder.PAIR_DEFINITIONS]
    expected_mapping = {
        pair_id: pair_ids[index + 1] if index % 2 == 0 else pair_ids[index - 1]
        for index, pair_id in enumerate(pair_ids)
    }
    expected_clusters = {
        pair_id: "control_dyad:" + ":".join(sorted((pair_id, partner_id)))
        for pair_id, partner_id in expected_mapping.items()
    }
    observed_mapping = {
        row["source_pair_id"]: row["unrelated_pair_id"]
        for row in fixture["control_derangement"]
    }
    observed_clusters = {
        row["source_pair_id"]: row["control_cluster_id"]
        for row in fixture["control_derangement"]
    }
    cluster_counts = Counter(
        row["control_cluster_id"] for row in fixture["control_derangement"]
    )
    by_item = {row["item_id"]: row for row in fixture["items"]}
    unrelated_targets: Counter[tuple[str, str]] = Counter()

    assert len(fixture["control_derangement"]) == 48
    assert observed_mapping == expected_mapping
    assert observed_clusters == expected_clusters
    assert set(observed_mapping) == set(observed_mapping.values()) == set(pair_ids)
    assert all(source != target for source, target in observed_mapping.items())
    assert len(cluster_counts) == 24
    assert set(cluster_counts.values()) == {2}

    for source_pair_id, partner_pair_id in observed_mapping.items():
        assert observed_mapping[partner_pair_id] == source_pair_id
        assert observed_clusters[partner_pair_id] == observed_clusters[source_pair_id]

    for item in fixture["items"]:
        unrelated = by_item[item["unrelated_same_polarity_item_id"]]
        assert unrelated["pair_id"] == expected_mapping[item["pair_id"]]
        assert unrelated["pair_id"] != item["pair_id"]
        assert unrelated["item_id"] != item["item_id"]
        assert unrelated["truth_polarity"] == item["truth_polarity"]
        assert unrelated["unrelated_same_polarity_item_id"] == item["item_id"]
        assert item["unrelated_control_cluster_id"] == expected_clusters[item["pair_id"]]
        assert (
            unrelated["unrelated_control_cluster_id"]
            == item["unrelated_control_cluster_id"]
        )
        unrelated_targets[(unrelated["pair_id"], unrelated["truth_polarity"])] += 1

    assert unrelated_targets == Counter(
        (pair_id, polarity)
        for pair_id in pair_ids
        for polarity in ("positive", "negative")
    )


def test_firewall_outcome_exposure_and_zero_call_contract_are_exact() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)
    expected_exposure = {
        "prior_syntax_behavior_used_to_define_contrast": True,
        "holdout_native_behavior_observed": False,
        "intervention_outcomes_observed": False,
    }

    assert fixture["mode"] == manifest["mode"] == "development"
    assert fixture["purpose"] == manifest["purpose"] == builder.PURPOSE
    assert fixture["firewall"] == manifest["firewall"] == builder.FIREWALL
    assert all(row["firewall"] == builder.FIREWALL for row in fixture["items"])
    assert fixture["outcome_exposure"] == expected_exposure
    assert fixture["model_calls_made_by_builder"] == 0
    assert manifest["contract"] == {
        "holdout_pairs": 48,
        "holdout_items": 96,
        "items_per_pair": 2,
        "inferential_unit": "lexical_pair",
        "fixed_unrelated_control": "reciprocal_pair_dyad_same_truth_polarity",
        "unrelated_control_inferential_units": 24,
        "model_calls_made_by_builder": 0,
    }
    assert manifest["status"] == "FROZEN_NO_HOLDOUT_FORWARD"
    assert manifest["provenance"]["external_data_sources"] == []
    assert manifest["claim_scope"] == (
        "prompt_protocol_causal_transfer_only_no_biology_latent_knowledge_"
        "activation_gap_physical_law_or_model_family_claim"
    )


def test_static_manifest_binds_canonical_content_and_exact_file_bytes() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)

    assert FIXTURE_PATH.read_bytes() == _serialized(fixture)
    assert MANIFEST_PATH.read_bytes() == _serialized(manifest)
    assert manifest == builder.build_manifest(fixture, FIXTURE_PATH)
    assert manifest["schema_version"] == builder.MANIFEST_SCHEMA
    assert manifest["analysis_id"] == builder.ANALYSIS_ID
    assert manifest["artifacts"] == {
        "builder_path": (
            "signal/syntax/build_coherent_readout_v2_causal_binding_bank.py"
        ),
        "builder_sha256": _sha256(Path(builder.__file__)),
        "fixture_path": (
            "signal/syntax/coherent_readout_v2_causal_binding_bank.json"
        ),
        "fixture_sha256": _sha256(FIXTURE_PATH),
        "fixture_canonical_sha256": builder.canonical_sha256(fixture),
    }
    assert manifest["provenance"]["pair_registry_canonical_sha256"] == (
        builder.canonical_sha256(fixture["pair_registry"])
    )
    assert manifest["provenance"][
        "control_derangement_canonical_sha256"
    ] == builder.canonical_sha256(fixture["control_derangement"])


def test_builder_rebuild_is_byte_deterministic_on_temp_paths(tmp_path: Path) -> None:
    out = tmp_path / "causal_binding_bank.json"
    manifest_out = tmp_path / "causal_binding_bank.manifest.json"

    first_fixture, first_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )
    first_fixture_bytes = out.read_bytes()
    first_manifest_bytes = manifest_out.read_bytes()
    second_fixture, second_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )

    assert first_fixture == second_fixture == builder.build_fixture()
    assert first_manifest == second_manifest == builder.build_manifest(
        first_fixture, out
    )
    assert out.read_bytes() == first_fixture_bytes == FIXTURE_PATH.read_bytes()
    assert manifest_out.read_bytes() == first_manifest_bytes
    assert first_fixture_bytes == _serialized(first_fixture)
    assert first_manifest_bytes == _serialized(first_manifest)
    assert first_manifest["artifacts"]["fixture_path"] == str(out.resolve())
    assert first_manifest["artifacts"]["fixture_sha256"] == _sha256(out)


def test_builder_refuses_to_overwrite_different_frozen_artifacts(
    tmp_path: Path,
) -> None:
    out = tmp_path / "causal_binding_bank.json"
    manifest_out = tmp_path / "causal_binding_bank.manifest.json"
    different_fixture = b'{"tampered":true}\n'
    out.write_bytes(different_fixture)

    with pytest.raises(builder.CausalBindingBankError, match="refusing to overwrite"):
        builder.build_and_write(out=out, manifest_out=manifest_out)
    assert out.read_bytes() == different_fixture
    assert not manifest_out.exists()

    out.unlink()
    builder.build_and_write(out=out, manifest_out=manifest_out)
    different_manifest = b'{"tampered":true}\n'
    manifest_out.write_bytes(different_manifest)

    with pytest.raises(builder.CausalBindingBankError, match="refusing to overwrite"):
        builder.build_and_write(out=out, manifest_out=manifest_out)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()
    assert manifest_out.read_bytes() == different_manifest


def test_fixture_and_manifest_construction_reject_tampering(tmp_path: Path) -> None:
    fixture = _json(FIXTURE_PATH)

    tampered = copy.deepcopy(fixture)
    tampered["outcome_exposure"]["holdout_native_behavior_observed"] = True
    with pytest.raises(builder.CausalBindingBankError, match="outcome-exposure"):
        builder.validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["model_calls_made_by_builder"] = 1
    with pytest.raises(builder.CausalBindingBankError, match="model execution"):
        builder.validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["firewall"]["activation_gap_inference"] = "allowed"
    with pytest.raises(builder.CausalBindingBankError, match="firewall"):
        builder.validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["same_pair_counterfactual_item_id"] = tampered["items"][
        0
    ]["item_id"]
    with pytest.raises(builder.CausalBindingBankError, match="held-out items"):
        builder.validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["control_derangement"][0]["unrelated_pair_id"] = tampered[
        "control_derangement"
    ][0]["source_pair_id"]
    with pytest.raises(builder.CausalBindingBankError, match="derangement"):
        builder.validate_fixture(tampered)

    fixture_copy = tmp_path / "fixture.json"
    fixture_copy.write_bytes(_serialized(fixture))
    fixture_copy.write_bytes(fixture_copy.read_bytes() + b" ")
    byte_bound_manifest = builder.build_manifest(fixture, fixture_copy)
    assert byte_bound_manifest["artifacts"]["fixture_sha256"] == _sha256(
        fixture_copy
    )
    assert byte_bound_manifest["artifacts"]["fixture_sha256"] != _sha256(
        FIXTURE_PATH
    )
    assert byte_bound_manifest["artifacts"]["fixture_canonical_sha256"] == (
        builder.canonical_sha256(fixture)
    )

    changed_on_disk = copy.deepcopy(fixture)
    changed_on_disk["freeze_date"] = "2099-01-01"
    fixture_copy.write_bytes(_serialized(changed_on_disk))
    with pytest.raises(
        builder.CausalBindingBankError,
        match="does not contain the supplied fixture",
    ):
        builder.build_manifest(fixture, fixture_copy)

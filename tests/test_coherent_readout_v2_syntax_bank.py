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

import build_coherent_readout_v2_syntax_bank as builder  # noqa: E402

FIXTURE_PATH = SYNTAX_DIR / "coherent_readout_v2_syntax_bank.json"
MANIFEST_PATH = SYNTAX_DIR / "coherent_readout_v2_syntax_bank.manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_static_artifacts_match_builder_and_are_hash_bound() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)

    assert fixture == builder.build_fixture()
    assert builder._validate_fixture(fixture) == fixture
    assert builder._validate_manifest(manifest, fixture, FIXTURE_PATH) == manifest
    assert fixture["schema_version"] == builder.FIXTURE_SCHEMA
    assert manifest["schema_version"] == builder.MANIFEST_SCHEMA
    assert fixture["analysis_id"] == manifest["analysis_id"] == builder.ANALYSIS_ID
    assert manifest["artifacts"] == {
        "builder_path": "signal/syntax/build_coherent_readout_v2_syntax_bank.py",
        "builder_sha256": _sha256(Path(builder.__file__)),
        "fixture_canonical_sha256": builder._canonical_sha256(fixture),
        "fixture_path": "signal/syntax/coherent_readout_v2_syntax_bank.json",
        "fixture_sha256": _sha256(FIXTURE_PATH),
    }


def test_bank_has_eight_frozen_pairs_and_both_truths_once() -> None:
    fixture = _json(FIXTURE_PATH)
    items = fixture["items"]
    expected_pairs = [
        ("amber", "cobalt"),
        ("cedar", "maple"),
        ("north", "south"),
        ("circle", "square"),
        ("copper", "silver"),
        ("violin", "trumpet"),
        ("river", "mountain"),
        ("winter", "summer"),
    ]

    assert len(fixture["pair_registry"]) == 8
    assert len(items) == 16
    assert fixture["inferential_unit"] == "pair_cluster"
    assert [
        (pair["positive_class"], pair["negative_class"])
        for pair in fixture["pair_registry"]
    ] == expected_pairs
    assert [item["item_id"] for item in items] == sorted(
        item["item_id"] for item in items
    )
    assert [pair["pair_id"] for pair in fixture["pair_registry"]] == sorted(
        pair["pair_id"] for pair in fixture["pair_registry"]
    )
    assert Counter((item["pair_id"], item["truth_polarity"]) for item in items) == Counter(
        (pair_id, polarity)
        for pair_id, _, _ in builder.PAIR_DEFINITIONS
        for polarity in ("positive", "negative")
    )
    assert Counter(item["truth_polarity"] for item in items) == {
        "positive": 8,
        "negative": 8,
    }
    assert all(item["pair_id"] == item["cluster_id"] for item in items)


def test_each_item_is_a_direct_declaration_with_authenticated_text() -> None:
    fixture = _json(FIXTURE_PATH)

    for item in fixture["items"]:
        declared = item["declared_class"]
        expected_declared = item[f"{item['truth_polarity']}_class"]
        expected_text = f"The declared class is {declared}."
        assert declared == expected_declared
        assert item["declaration_text"] == expected_text
        assert item["declaration_sha256"] == hashlib.sha256(
            expected_text.encode("utf-8")
        ).hexdigest()
        assert item["item_id"] == f"syntax:{item['pair_id']}:{declared}"


def test_firewall_is_exact_and_redundant_on_every_item() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)
    expected = {
        "scope": "syntax_selection_only",
        "biology_inference": "forbidden",
        "knowledge_inference": "forbidden",
        "activation_inference": "forbidden",
    }

    assert fixture["purpose"] == manifest["purpose"] == "syntax_selection_only"
    assert fixture["firewall"] == manifest["firewall"] == expected
    assert all(item["firewall"] == expected for item in fixture["items"])
    assert manifest["claim_scope"] == (
        "syntax_selection_only_no_biology_knowledge_or_activation_inference"
    )
    assert manifest["status"] == "built_not_executed"
    assert manifest["contract"]["model_calls_made_by_builder"] == 0
    assert manifest["contract"]["inferential_unit"] == "pair_cluster"
    assert manifest["contract"]["inferential_units"] == 8
    assert fixture["model_calls_made_by_builder"] == 0


def test_json_bytes_use_the_frozen_sorted_serialization() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)

    assert FIXTURE_PATH.read_bytes() == _serialized(fixture)
    assert MANIFEST_PATH.read_bytes() == _serialized(manifest)


def test_validation_rejects_truth_text_hash_order_and_firewall_tampering() -> None:
    fixture = _json(FIXTURE_PATH)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["truth_polarity"] = "negative"
    with pytest.raises(builder.SyntaxBankError, match="truth polarity"):
        builder._validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["declaration_text"] = "The declared class is cobalt."
    with pytest.raises(builder.SyntaxBankError, match="declaration text"):
        builder._validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["declaration_sha256"] = "0" * 64
    with pytest.raises(builder.SyntaxBankError, match="digest mismatch"):
        builder._validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["firewall"]["knowledge_inference"] = "allowed"
    with pytest.raises(builder.SyntaxBankError, match="firewall"):
        builder._validate_fixture(tampered)

    tampered = copy.deepcopy(fixture)
    tampered["items"] = list(reversed(tampered["items"]))
    with pytest.raises(builder.SyntaxBankError, match="sorted"):
        builder._validate_fixture(tampered)


def test_builder_is_byte_deterministic_and_makes_zero_model_calls(
    tmp_path: Path,
) -> None:
    out = tmp_path / "syntax_bank.json"
    manifest_out = tmp_path / "syntax_bank.manifest.json"

    first_fixture, first_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )
    first_fixture_bytes = out.read_bytes()
    first_manifest_bytes = manifest_out.read_bytes()
    second_fixture, second_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )

    assert first_fixture == second_fixture == builder.build_fixture()
    assert first_manifest == second_manifest
    assert out.read_bytes() == first_fixture_bytes
    assert manifest_out.read_bytes() == first_manifest_bytes
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()
    assert first_manifest["contract"]["model_calls_made_by_builder"] == 0
    assert first_manifest["provenance"]["external_data_sources"] == []


def test_manifest_refuses_fixture_byte_or_content_mismatch(tmp_path: Path) -> None:
    fixture = builder.build_fixture()
    fixture_path = tmp_path / "syntax_bank.json"
    fixture_path.write_bytes(_serialized(fixture) + b" ")

    manifest = builder.build_manifest(fixture, fixture_path)
    assert manifest["artifacts"]["fixture_sha256"] == _sha256(fixture_path)

    tampered = copy.deepcopy(fixture)
    tampered["items"][0]["declared_class"] = "cobalt"
    with pytest.raises(builder.SyntaxBankError):
        builder.build_manifest(tampered, fixture_path)

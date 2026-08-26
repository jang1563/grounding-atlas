from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTAX_DIR = ROOT / "signal" / "syntax"
sys.path.insert(0, str(SYNTAX_DIR))
sys.path.insert(0, str(ROOT / "eval"))

import build_coherent_readout_v2_bio_projection_registry as builder  # noqa: E402
import run_coherent_readout_v2_syntax as syntax_runner  # noqa: E402

REGISTRY_PATH = SYNTAX_DIR / "coherent_readout_v2_bio_projection_registry.json"
MANIFEST_PATH = (
    SYNTAX_DIR / "coherent_readout_v2_bio_projection_registry.manifest.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_static_registry_and_manifest_are_exactly_hash_bound() -> None:
    registry = _json(REGISTRY_PATH)
    manifest = _json(MANIFEST_PATH)

    assert registry == builder.build_registry()
    assert builder._validate_registry(registry) == registry
    assert builder._validate_manifest(manifest, registry, REGISTRY_PATH) == manifest
    assert manifest["artifacts"] == {
        "builder_path": (
            "signal/syntax/build_coherent_readout_v2_bio_projection_registry.py"
        ),
        "builder_sha256": _sha256(Path(builder.__file__)),
        "registry_path": "signal/syntax/coherent_readout_v2_bio_projection_registry.json",
        "registry_sha256": _sha256(REGISTRY_PATH),
        "registry_canonical_sha256": builder._canonical_sha256(registry),
        "projection_entries_canonical_sha256": builder._canonical_sha256(
            registry["projections"]
        ),
    }


def test_all_four_source_definitions_are_bound_in_frozen_priority() -> None:
    registry = _json(REGISTRY_PATH)
    entries = registry["projections"]
    source_definitions = [entry["source_syntax_candidate_definition"] for entry in entries]

    assert registry["candidate_priority"] == list(builder.CANDIDATE_ORDER)
    assert [entry["candidate_id"] for entry in entries] == list(builder.CANDIDATE_ORDER)
    assert [entry["candidate_rank"] for entry in entries] == list(range(4))
    assert source_definitions == [dict(value) for value in builder.SOURCE_SYNTAX_CANDIDATES]
    assert source_definitions == syntax_runner.candidate_definitions()
    assert registry["source_syntax_candidate_registry_sha256"] == (
        syntax_runner.candidate_bank_sha256()
    )
    for entry in entries:
        assert entry["source_syntax_candidate_definition_sha256"] == (
            builder._canonical_sha256(entry["source_syntax_candidate_definition"])
        )


def test_biological_core_and_all_transformations_are_byte_exact() -> None:
    registry = _json(REGISTRY_PATH)
    projections = {
        entry["candidate_id"]: entry["biological_projection"]
        for entry in registry["projections"]
    }
    core = (
        "Classify the cell using exactly one label.\n"
        "{line_1}\n{line_2}\n"
        "Genes, highest expression rank first: {gene_sentence}"
    )
    instruction = (
        "Follow the label mapping. Answer with exactly one character: X or Y. "
        "Do not write any other text."
    )

    assert registry["biological_core"] == builder.BIO_CORE == core
    assert not core.endswith("\n")
    assert registry["biological_core_sha256"] == hashlib.sha256(
        core.encode("utf-8")
    ).hexdigest()
    assert projections[builder.CANDIDATE_ORDER[0]]["user_template"] == core
    assert projections[builder.CANDIDATE_ORDER[1]]["user_template"] == (
        core + "\nReturn exactly X or Y."
    )
    assert projections[builder.CANDIDATE_ORDER[2]]["user_template"] == (
        instruction + "\n" + core + "\nAnswer:"
    )
    assert projections[builder.CANDIDATE_ORDER[3]]["user_template"] == (
        core + "\nLabel:"
    )
    for entry in registry["projections"]:
        source = entry["source_syntax_candidate_definition"]
        projected = entry["biological_projection"]
        assert entry["biological_projection_sha256"] == builder._canonical_sha256(
            projected
        )
        assert {
            key: value for key, value in projected.items() if key != "user_template"
        } == {key: value for key, value in source.items() if key != "user_template"}
        assert "{declaration}" not in projected["user_template"]
        assert not projected["user_template"].endswith("\n")


def test_prefill_flags_system_assistant_and_suffixes_are_preserved() -> None:
    registry = _json(REGISTRY_PATH)
    prefill = registry["projections"][0]["biological_projection"]

    assert prefill["message_roles"] == ["system", "user", "assistant"]
    assert prefill["system_template"] == builder.SYSTEM_PREFILL
    assert prefill["assistant_template"] == "Answer:"
    assert prefill["render_mode"] == "continue_final_message"
    assert prefill["add_generation_prompt"] is False
    assert prefill["continue_final_message"] is True
    assert prefill["enable_thinking"] is False
    assert prefill["x_answer_text"] == " X"
    assert prefill["y_answer_text"] == " Y"


def test_registry_is_permanent_no_gold_outcome_blind_and_zero_call() -> None:
    registry = _json(REGISTRY_PATH)
    manifest = _json(MANIFEST_PATH)
    firewall = {
        "partition": "permanent_development_only",
        "permanent": True,
        "gold_labels": "absent",
        "syntax_outcomes_at_freeze": "unobserved",
        "confirmatory_eligibility": "prohibited",
        "promotion_to_confirmation": "forbidden",
        "claim_scope": (
            "answer_envelope_projection_only_no_biology_knowledge_or_activation_inference"
        ),
    }

    assert registry["firewall"] == manifest["firewall"] == firewall
    assert registry["model_calls_made_by_builder"] == 0
    assert manifest["contract"]["model_calls_made_by_builder"] == 0
    assert manifest["contract"]["winner_selected"] is False
    assert manifest["contract"]["syntax_outcomes_observed_by_builder"] is False
    assert manifest["contract"]["biological_items_present"] is False
    assert manifest["contract"]["gold_labels_present"] is False
    assert manifest["provenance"]["syntax_outcomes_consulted"] is False
    forbidden = {
        "selected_candidate_id",
        "E_oriented",
        "M_channel",
        "M_correct",
        "E_choice",
        "native",
        "native_correct",
        "G",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not (keys(registry) & forbidden)


def test_source_runner_provenance_is_current_and_outcome_independent() -> None:
    manifest = _json(MANIFEST_PATH)
    provenance = manifest["provenance"]

    assert provenance["source_syntax_runner_path"] == (
        "eval/run_coherent_readout_v2_syntax.py"
    )
    assert provenance["source_syntax_runner_sha256"] == _sha256(
        Path(syntax_runner.__file__)
    )
    assert provenance["external_data_sources"] == []
    assert provenance["syntax_outcomes_consulted"] is False


def test_sorted_json_and_repeated_builds_are_deterministic(tmp_path: Path) -> None:
    registry = _json(REGISTRY_PATH)
    manifest = _json(MANIFEST_PATH)
    assert REGISTRY_PATH.read_bytes() == _serialized(registry)
    assert MANIFEST_PATH.read_bytes() == _serialized(manifest)

    out = tmp_path / "registry.json"
    manifest_out = tmp_path / "registry.manifest.json"
    first_registry, first_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )
    first_bytes = (out.read_bytes(), manifest_out.read_bytes())
    second_registry, second_manifest = builder.build_and_write(
        out=out, manifest_out=manifest_out
    )

    assert first_registry == second_registry == registry
    assert first_manifest == second_manifest
    assert (out.read_bytes(), manifest_out.read_bytes()) == first_bytes
    assert out.read_bytes() == REGISTRY_PATH.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["candidate_priority"].reverse(),
        lambda value: value.__setitem__("biological_core", value["biological_core"] + "\n"),
        lambda value: value["projections"][0][
            "source_syntax_candidate_definition"
        ].__setitem__("assistant_template", "Label:"),
        lambda value: value["projections"][1]["biological_projection"].__setitem__(
            "user_template", builder.BIO_CORE
        ),
        lambda value: value["projections"][0]["biological_projection"].__setitem__(
            "continue_final_message", False
        ),
        lambda value: value["projections"][2].__setitem__(
            "biological_projection_sha256", "0" * 64
        ),
        lambda value: value["firewall"].__setitem__(
            "confirmatory_eligibility", "allowed"
        ),
        lambda value: value.__setitem__("model_calls_made_by_builder", 1),
    ),
)
def test_registry_validation_rejects_tampering(mutation) -> None:
    tampered = copy.deepcopy(_json(REGISTRY_PATH))
    mutation(tampered)
    with pytest.raises(builder.BioProjectionRegistryError):
        builder._validate_registry(tampered)


def test_manifest_rejects_registry_bytes_and_hash_tampering(tmp_path: Path) -> None:
    registry = _json(REGISTRY_PATH)
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(_serialized(registry) + b" ")
    with pytest.raises(builder.BioProjectionRegistryError, match="deterministic bytes"):
        builder.build_manifest(registry, registry_path)

    tampered_manifest = copy.deepcopy(_json(MANIFEST_PATH))
    tampered_manifest["artifacts"]["registry_canonical_sha256"] = "0" * 64
    with pytest.raises(builder.BioProjectionRegistryError, match="artifact locks"):
        builder._validate_manifest(tampered_manifest, registry, REGISTRY_PATH)

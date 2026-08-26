from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTAX_DIR = ROOT / "signal" / "syntax"
sys.path.insert(0, str(SYNTAX_DIR))

import build_coherent_readout_v3_content_routing_bank as builder  # noqa: E402

FIXTURE_PATH = SYNTAX_DIR / "coherent_readout_v3_content_routing_bank.json"
MANIFEST_PATH = (
    SYNTAX_DIR / "coherent_readout_v3_content_routing_bank.manifest.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _gram_matrix(rows: list[list[int]]) -> list[list[int]]:
    return [
        [sum(row[left] * row[right] for row in rows) for right in range(6)]
        for left in range(6)
    ]


@pytest.fixture(scope="module")
def fixture() -> dict:
    return builder.build_fixture()


def test_builder_freezes_56_uniform_symbolic_worlds_and_exact_roles(
    fixture: dict,
) -> None:
    worlds = fixture["world_registry"]
    names = [
        name for world in worlds for name in (world["entity_a"], world["entity_b"])
    ]

    assert builder.validate_fixture(fixture) == fixture
    assert fixture["schema_version"] == builder.FIXTURE_SCHEMA
    assert fixture["analysis_id"] == builder.ANALYSIS_ID
    assert fixture["inferential_unit"] == "symbolic_world"
    assert fixture["dependency_unit"] == "complete_factorial_within_symbolic_world"
    assert [world["world_index"] for world in worlds] == list(range(1, 57))
    assert Counter(world["role"] for world in worlds) == {
        "direction_fit": 16,
        "localization": 8,
        "holdout": 32,
    }
    assert [world["role"] for world in worlds[:16]] == ["direction_fit"] * 16
    assert [world["role"] for world in worlds[16:24]] == ["localization"] * 8
    assert [world["role"] for world in worlds[24:]] == ["holdout"] * 32
    assert names == [
        name
        for index in range(1, 57)
        for name in (f"symbol_a_{index:03d}", f"symbol_b_{index:03d}")
    ]
    assert len(names) == len(set(names)) == 112
    assert {len(name) for name in names} == {12}
    assert all(builder.CONTROLLED_NAME_PATTERN.fullmatch(name) for name in names)


def test_every_world_has_the_complete_two_to_fifth_factorial(fixture: dict) -> None:
    cells = fixture["cells"]
    by_world: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        by_world[cell["world_id"]].append(cell)
    expected_factor_cells = set(
        itertools.product(
            builder.FACTOR_LEVELS["queried_entity"],
            builder.FACTOR_LEVELS["queried_property"],
            builder.FACTOR_LEVELS["distractor_property"],
            builder.FACTOR_LEVELS["codebook"],
            builder.FACTOR_LEVELS["fact_line_order"],
        )
    )

    assert cells == builder._build_cells()
    assert len(cells) == 1792
    assert len({cell["cell_id"] for cell in cells}) == 1792
    assert len({cell["prompt_text"] for cell in cells}) == 1792
    assert Counter(cell["role"] for cell in cells) == {
        "direction_fit": 512,
        "localization": 256,
        "holdout": 1024,
    }
    assert len(by_world) == 56
    for world_cells in by_world.values():
        observed = {
            (
                cell["queried_entity_slot"],
                cell["queried_property"],
                cell["distractor_property"],
                cell["codebook_id"],
                cell["fact_line_order"],
            )
            for cell in world_cells
        }
        assert len(world_cells) == 32
        assert observed == expected_factor_cells

    for cell in cells:
        assert cell["native_answer"] == builder.CODEBOOKS[cell["codebook_id"]][
            cell["queried_property"]
        ]
        assert cell["prompt_sha256"] == builder.text_sha256(cell["prompt_text"])
        if cell["fact_line_order"] == "query_first":
            assert cell["queried_entity"] in cell["fact_lines"][0]
            assert cell["distractor_entity"] in cell["fact_lines"][1]
        else:
            assert cell["distractor_entity"] in cell["fact_lines"][0]
            assert cell["queried_entity"] in cell["fact_lines"][1]


def test_cross_codebook_and_counterfactual_references_are_exact_involutions(
    fixture: dict,
) -> None:
    by_id = {cell["cell_id"]: cell for cell in fixture["cells"]}

    for cell in fixture["cells"]:
        self_cell = by_id[cell["self_cell_id"]]
        anti_copy = by_id[cell["anti_copy_donor_cell_id"]]
        counterfactual = by_id[cell["text_counterfactual_cell_id"]]
        opposite_codebook = by_id[
            cell["same_content_opposite_codebook_donor_cell_id"]
        ]

        assert self_cell == cell
        assert anti_copy["world_id"] == cell["world_id"]
        assert anti_copy["queried_entity_slot"] == cell["queried_entity_slot"]
        assert anti_copy["distractor_property"] == cell["distractor_property"]
        assert anti_copy["fact_line_order"] == cell["fact_line_order"]
        assert anti_copy["queried_property"] == builder.OPPOSITE_PROPERTY[
            cell["queried_property"]
        ]
        assert anti_copy["codebook_id"] == builder.OPPOSITE_CODEBOOK[
            cell["codebook_id"]
        ]
        assert anti_copy["native_answer"] == cell["native_answer"]
        assert anti_copy["anti_copy_donor_cell_id"] == cell["cell_id"]

        assert counterfactual["queried_property"] == builder.OPPOSITE_PROPERTY[
            cell["queried_property"]
        ]
        assert counterfactual["codebook_id"] == cell["codebook_id"]
        assert counterfactual["native_answer"] != cell["native_answer"]
        assert counterfactual["text_counterfactual_cell_id"] == cell["cell_id"]

        assert opposite_codebook["queried_property"] == cell["queried_property"]
        assert opposite_codebook["codebook_id"] == builder.OPPOSITE_CODEBOOK[
            cell["codebook_id"]
        ]
        assert opposite_codebook["fact_lines"] == cell["fact_lines"]
        assert opposite_codebook["native_answer"] != cell["native_answer"]
        assert (
            opposite_codebook["same_content_opposite_codebook_donor_cell_id"]
            == cell["cell_id"]
        )
        assert (
            counterfactual["same_content_opposite_codebook_donor_cell_id"]
            == anti_copy["cell_id"]
        )
        assert (
            opposite_codebook["text_counterfactual_cell_id"]
            == anti_copy["cell_id"]
        )


def test_three_nuisance_controls_are_answer_preserving_involutions(
    fixture: dict,
) -> None:
    by_id = {cell["cell_id"]: cell for cell in fixture["cells"]}

    for cell in fixture["cells"]:
        distractor_flip = by_id[cell["distractor_flip_cell_id"]]
        entity_flip = by_id[cell["query_entity_flip_cell_id"]]
        order_flip = by_id[cell["fact_order_flip_cell_id"]]

        assert distractor_flip["distractor_property"] == builder.OPPOSITE_PROPERTY[
            cell["distractor_property"]
        ]
        assert distractor_flip["queried_entity_slot"] == cell["queried_entity_slot"]
        assert distractor_flip["queried_property"] == cell["queried_property"]
        assert distractor_flip["codebook_id"] == cell["codebook_id"]
        assert distractor_flip["fact_line_order"] == cell["fact_line_order"]
        assert distractor_flip["native_answer"] == cell["native_answer"]
        assert distractor_flip["distractor_flip_cell_id"] == cell["cell_id"]

        assert entity_flip["queried_entity_slot"] != cell["queried_entity_slot"]
        assert entity_flip["queried_entity"] == cell["distractor_entity"]
        assert entity_flip["distractor_entity"] == cell["queried_entity"]
        assert entity_flip["queried_property"] == cell["queried_property"]
        assert entity_flip["distractor_property"] == cell["distractor_property"]
        assert entity_flip["codebook_id"] == cell["codebook_id"]
        assert entity_flip["fact_line_order"] == cell["fact_line_order"]
        assert entity_flip["native_answer"] == cell["native_answer"]
        assert entity_flip["query_entity_flip_cell_id"] == cell["cell_id"]

        assert order_flip["fact_line_order"] != cell["fact_line_order"]
        assert order_flip["queried_entity_slot"] == cell["queried_entity_slot"]
        assert order_flip["queried_property"] == cell["queried_property"]
        assert order_flip["distractor_property"] == cell["distractor_property"]
        assert order_flip["codebook_id"] == cell["codebook_id"]
        assert order_flip["fact_lines"] == list(reversed(cell["fact_lines"]))
        assert order_flip["native_answer"] == cell["native_answer"]
        assert order_flip["fact_order_flip_cell_id"] == cell["cell_id"]


def test_recipient_fraction_is_exact_balanced_and_role_complete(fixture: dict) -> None:
    design = fixture["recipient_fractional_factorial"]
    by_id = {cell["cell_id"]: cell for cell in fixture["cells"]}
    declared_ids = [
        cell_id
        for world in design["selected_by_world"]
        for cell_id in world["cell_ids"]
    ]
    flagged_ids = [
        cell["cell_id"] for cell in fixture["cells"] if cell["recipient_selected"]
    ]

    assert design == builder._recipient_fractional_factorial()
    assert design["base_factors"] == ["e", "p", "m"]
    assert design["generators"] == {
        "d": "e*p",
        "order": "e*m",
        "native_answer": "p*m",
    }
    expected_gram = [
        [8 if left == right else 0 for right in range(6)] for left in range(6)
    ]
    assert design["orthogonality_contract"] == {
        "columns": [
            "query_content",
            "codebook",
            "query_entity",
            "distractor",
            "fact_order",
            "native_answer",
        ],
        "required_gram_matrix": "8I_6",
        "gram_matrix": expected_gram,
    }
    assert len(design["runs"]) == 8
    assert len(design["selected_by_world"]) == 56
    assert len(declared_ids) == len(set(declared_ids)) == 448
    assert set(declared_ids) == set(flagged_ids)
    assert Counter(by_id[cell_id]["role"] for cell_id in declared_ids) == {
        "direction_fit": 128,
        "localization": 64,
        "holdout": 256,
    }

    for world_selection in design["selected_by_world"]:
        selected = [by_id[cell_id] for cell_id in world_selection["cell_ids"]]
        assert len(selected) == 8
        assert {cell["world_id"] for cell in selected} == {
            world_selection["world_id"]
        }
        assert sorted(cell["recipient_run_index"] for cell in selected) == list(
            range(1, 9)
        )
        for factor in (
            "queried_entity_slot",
            "queried_property",
            "distractor_property",
            "codebook_id",
            "fact_line_order",
        ):
            assert sorted(Counter(cell[factor] for cell in selected).values()) == [4, 4]
        assert Counter(cell["native_answer"] for cell in selected) == {"X": 4, "Y": 4}
        contrast_rows = []
        for cell in selected:
            e = -1 if cell["queried_entity_slot"] == "a" else 1
            p = -1 if cell["queried_property"] == "P" else 1
            m = -1 if cell["codebook_id"] == "identity" else 1
            d = -1 if cell["distractor_property"] == "P" else 1
            order = -1 if cell["fact_line_order"] == "query_first" else 1
            answer = -1 if cell["native_answer"] == "Y" else 1
            assert d == e * p
            assert order == e * m
            assert answer == p * m
            contrast_rows.append([p, m, e, d, order, answer])
        assert _gram_matrix(contrast_rows) == expected_gram


def test_recipient_correction_preserves_prompt_and_reference_core(fixture: dict) -> None:
    core = [
        {
            key: value
            for key, value in cell.items()
            if key not in {"recipient_selected", "recipient_run_index"}
        }
        for cell in fixture["cells"]
    ]
    references = [
        {key: value for key, value in cell.items() if key.endswith("_cell_id")}
        for cell in fixture["cells"]
    ]
    prompts = [
        {
            "cell_id": cell["cell_id"],
            "prompt_text": cell["prompt_text"],
            "prompt_sha256": cell["prompt_sha256"],
        }
        for cell in fixture["cells"]
    ]

    assert builder.canonical_sha256(core) == (
        "55dfa6f2bc65eedb1563f3946c0202c895168e772a6d1f801ed4912571db3cdd"
    )
    assert builder.canonical_sha256(references) == (
        "3822246b94b54eeddb7928e7ca3cff5c5c338c6d6834d73e9cc0ff1ed54411bc"
    )
    assert builder.canonical_sha256(prompts) == (
        "7b18b97cd76e1699079564f1965e085f7d3270f1b039e57b68ab9766a4994223"
    )


def test_prior_banks_are_hash_locked_and_used_only_for_disjointness(
    fixture: dict,
) -> None:
    references = fixture["prior_references"]
    worlds = fixture["world_registry"]
    cells = fixture["cells"]
    new_world_ids = {world["world_id"] for world in worlds}
    new_names = {
        name for world in worlds for name in (world["entity_a"], world["entity_b"])
    }
    new_cell_ids = {cell["cell_id"] for cell in cells}

    assert _sha256(builder.DISCOVERY_FIXTURE) == builder.DISCOVERY_FIXTURE_SHA256
    assert _sha256(builder.PRIOR_HOLDOUT_FIXTURE) == (
        builder.PRIOR_HOLDOUT_FIXTURE_SHA256
    )
    assert references == builder._prior_references()
    assert {reference["reference_id"] for reference in references} == {
        "v2_syntax_discovery",
        "v2_causal_binding_holdout",
    }
    for reference in references:
        assert reference["use"] == (
            "disjointness_check_only_no_behavior_or_outcome_use"
        )
        assert new_world_ids.isdisjoint(reference["pair_ids"])
        assert new_names.isdisjoint(reference["class_words"])
        assert new_cell_ids.isdisjoint(reference["item_ids"])


def test_firewall_no_output_exposure_and_zero_call_contract_are_exact(
    fixture: dict,
) -> None:
    assert fixture["mode"] == "prospective_pre_execution"
    assert fixture["purpose"] == builder.PURPOSE
    assert fixture["firewall"] == builder.FIREWALL
    assert fixture["outcome_exposure"] == builder.OUTCOME_EXPOSURE
    assert fixture["model_calls_made_by_builder"] == 0
    assert fixture["outcome_exposure"] == {
        "prior_banks_used_for_disjointness_only": True,
        "prior_bank_model_outputs_used": False,
        "new_bank_native_model_outputs_observed": False,
        "new_bank_intervention_outputs_observed": False,
        "model_outputs_used_to_select_worlds": False,
        "model_outputs_used_to_select_recipient_cells": False,
        "prompt_outcomes_used_to_filter_cells": False,
    }
    forbidden_prompt_terms = {
        "biology",
        "latent knowledge",
        "activation gap",
        "physical law",
    }
    assert all(
        not any(term in cell["prompt_text"].lower() for term in forbidden_prompt_terms)
        for cell in fixture["cells"]
    )


def test_tampered_fixture_contracts_are_rejected(fixture: dict) -> None:
    mutations = []

    wrong_reference = copy.deepcopy(fixture)
    wrong_reference["cells"][0]["anti_copy_donor_cell_id"] = wrong_reference[
        "cells"
    ][0]["cell_id"]
    mutations.append(wrong_reference)

    wrong_answer = copy.deepcopy(fixture)
    wrong_answer["cells"][0]["native_answer"] = "Y"
    mutations.append(wrong_answer)

    wrong_recipient = copy.deepcopy(fixture)
    wrong_recipient["recipient_fractional_factorial"]["generators"]["d"] = "e*m"
    mutations.append(wrong_recipient)

    exposed = copy.deepcopy(fixture)
    exposed["outcome_exposure"]["new_bank_native_model_outputs_observed"] = True
    mutations.append(exposed)

    called = copy.deepcopy(fixture)
    called["model_calls_made_by_builder"] = 1
    mutations.append(called)

    for mutated in mutations:
        with pytest.raises(builder.ContentRoutingBankError):
            builder.validate_fixture(mutated)


def test_deterministic_temp_writes_and_differing_overwrites_are_refused(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "fixture.json"
    manifest_path = tmp_path / "manifest.json"
    first_fixture, first_manifest = builder.build_and_write(
        fixture_path, manifest_path
    )
    first_fixture_bytes = fixture_path.read_bytes()
    first_manifest_bytes = manifest_path.read_bytes()
    second_fixture, second_manifest = builder.build_and_write(
        fixture_path, manifest_path
    )

    assert second_fixture == first_fixture
    assert second_manifest == first_manifest
    assert fixture_path.read_bytes() == first_fixture_bytes
    assert manifest_path.read_bytes() == first_manifest_bytes

    fixture_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.ContentRoutingBankError, match="refusing to overwrite"):
        builder.build_and_write(fixture_path, manifest_path)

    second_dir = tmp_path / "manifest-refusal"
    second_fixture_path = second_dir / "fixture.json"
    second_manifest_path = second_dir / "manifest.json"
    builder.build_and_write(second_fixture_path, second_manifest_path)
    second_manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.ContentRoutingBankError, match="refusing to overwrite"):
        builder.build_and_write(second_fixture_path, second_manifest_path)

    with pytest.raises(builder.ContentRoutingBankError, match="paths must differ"):
        builder.build_and_write(tmp_path / "same.json", tmp_path / "same.json")


def test_frozen_artifacts_match_builder_and_manifest_hash_contract() -> None:
    fixture = _json(FIXTURE_PATH)
    manifest = _json(MANIFEST_PATH)

    assert FIXTURE_PATH.read_bytes() == _serialized(fixture)
    assert MANIFEST_PATH.read_bytes() == _serialized(manifest)
    assert fixture == builder.build_fixture()
    assert manifest == builder.build_manifest(fixture, FIXTURE_PATH)
    assert builder.validate_manifest(manifest, fixture, FIXTURE_PATH) == manifest
    assert manifest["schema_version"] == builder.MANIFEST_SCHEMA
    assert manifest["status"] == "FROZEN_PROSPECTIVE_NO_MODEL_FORWARD"
    assert manifest["contract"] == {
        "symbolic_worlds": 56,
        "symbolic_entities": 112,
        "factors": 5,
        "full_factorial_cells_per_world": 32,
        "prompt_cells": 1792,
        "prompt_cells_by_role": {
            "direction_fit": 512,
            "localization": 256,
            "holdout": 1024,
        },
        "reference_roles_per_cell": 7,
        "recipient_cells_per_world": 8,
        "recipient_cells": 448,
        "recipient_cells_by_role": {
            "direction_fit": 128,
            "localization": 64,
            "holdout": 256,
        },
        "recipient_generators": {
            "d": "e*p",
            "order": "e*m",
            "native_answer": "p*m",
        },
        "recipient_contrast_gram_matrix": "8I_6",
        "model_calls_made_by_builder": 0,
    }
    assert manifest["provenance"]["external_data_sources"] == []
    assert manifest["provenance"]["prior_references"] == fixture[
        "prior_references"
    ]
    assert manifest["provenance"]["world_registry_canonical_sha256"] == (
        builder.canonical_sha256(fixture["world_registry"])
    )
    assert manifest["provenance"]["cell_registry_canonical_sha256"] == (
        builder.canonical_sha256(fixture["cells"])
    )
    assert manifest["provenance"]["recipient_design_canonical_sha256"] == (
        builder.canonical_sha256(fixture["recipient_fractional_factorial"])
    )
    assert manifest["artifacts"] == {
        "builder_path": (
            "signal/syntax/build_coherent_readout_v3_content_routing_bank.py"
        ),
        "builder_sha256": _sha256(Path(builder.__file__)),
        "fixture_path": "signal/syntax/coherent_readout_v3_content_routing_bank.json",
        "fixture_sha256": _sha256(FIXTURE_PATH),
        "fixture_canonical_sha256": builder.canonical_sha256(fixture),
    }

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["contract"]["recipient_cells"] = 447
    with pytest.raises(builder.ContentRoutingBankError):
        builder.validate_manifest(tampered_manifest, fixture, FIXTURE_PATH)

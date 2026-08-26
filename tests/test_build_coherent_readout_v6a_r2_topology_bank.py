from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "signal" / "syntax" / "build_coherent_readout_v6a_r2_topology_bank.py"
FIXTURE = BUILDER.with_name("coherent_readout_v6a_r2_topology_bank.json")
MANIFEST = FIXTURE.with_suffix(".manifest.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("v6a_r2_builder_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_and_manifest_rebuild_exactly(builder) -> None:
    fixture = _load(FIXTURE)
    manifest = _load(MANIFEST)

    assert builder.build_fixture() == fixture
    assert builder.validate_fixture(fixture) == fixture
    assert manifest["fixture_file_sha256"] == builder.file_sha256(FIXTURE)
    assert manifest["fixture_canonical_sha256"] == builder.canonical_sha256(fixture)
    assert manifest["builder_file_sha256"] == builder.file_sha256(BUILDER)
    assert manifest["model_calls"] == manifest["tokenizer_calls"] == 0


def test_exact_counts_and_components_first_block_order(builder) -> None:
    fixture = _load(FIXTURE)
    cells = fixture["cells"]

    assert len(fixture["worlds"]) == 16
    assert len(cells) == builder.EXPECTED_CALL_COUNT == 2_304
    assert Counter(cell["family"] for cell in cells) == builder.FAMILY_COUNTS
    assert Counter(cell["execution_block"] for cell in cells) == {
        "discovery-components": 640,
        "discovery-topology": 512,
        "confirmation-components": 640,
        "confirmation-topology": 512,
    }
    rank = {block: index for index, block in enumerate(builder.EXECUTION_BLOCK_ORDER)}
    assert [rank[cell["execution_block"]] for cell in cells] == sorted(
        rank[cell["execution_block"]] for cell in cells
    )
    assert fixture["discovery_component_call_count"] == 640
    assert fixture["remaining_main_call_count"] == 1_664


def test_symbol_replacement_is_exact_fresh_and_split_disjoint(builder) -> None:
    fixture = _load(FIXTURE)
    symbols = [symbol for world in fixture["worlds"] for symbol in world["symbols"]]
    provenance = fixture["symbol_replacement_provenance"]

    assert symbols == list(builder.R2_SYMBOLS)
    assert len(symbols) == len(set(symbols)) == 64
    assert not set(symbols) & builder.V2_QUALIFICATION_SYMBOLS
    assert not set(symbols) & builder.PRIOR_ASCII_SYMBOLS
    assert builder.TOKENIZER_ELIGIBLE_CANDIDATES_HASH_ORDER[:22] == (
        builder.REPLACEMENT_SYMBOLS
    )
    assert provenance["selection_salt"] == "V6A-R2-main-replacement-v1|"
    assert provenance["unicode_database_version"] == "15.1.0"
    assert provenance["tokenizer_calls_by_builder"] == 0
    discovery = set(symbols[:32])
    confirmation = set(symbols[32:])
    assert len(discovery) == len(confirmation) == 32
    assert not discovery & confirmation


def test_world_identity_foldover_and_roles_are_fixed() -> None:
    worlds = _load(FIXTURE)["worlds"]

    assert Counter(world["role"] for world in worlds) == {
        "discovery": 8,
        "confirmation": 8,
    }
    for role in ("discovery", "confirmation"):
        members = [world for world in worlds if world["role"] == role]
        assert [world["role_index"] for world in members] == list(range(1, 9))
        assert [world["foldover_g"] for world in members] == [-1, 1] * 4
    for field in ("world_id", "instance_key", "target_entity", "other_entity"):
        assert len({world[field] for world in worlds}) == 16


def test_every_world_has_complete_registered_factorials() -> None:
    fixture = _load(FIXTURE)
    for world in fixture["worlds"]:
        cells = [cell for cell in fixture["cells"] if cell["world_id"] == world["world_id"]]
        assert Counter(cell["family"] for cell in cells) == {
            "property_retrieval": 32,
            "codebook_lookup": 16,
            "single_target_composition": 32,
            "two_fact_composition": 64,
        }
        assert {
            tuple(cell["factors"][name] for name in ("p", "v", "o", "q", "a"))
            for cell in cells
            if cell["family"] == "property_retrieval"
        } == set(product((-1, 1), repeat=5))
        assert {
            tuple(cell["factors"][name] for name in ("p", "m", "r", "v"))
            for cell in cells
            if cell["family"] == "codebook_lookup"
        } == set(product((-1, 1), repeat=4))
        assert {
            tuple(cell["factors"][name] for name in ("p", "m", "u", "q", "a"))
            for cell in cells
            if cell["family"] == "single_target_composition"
        } == set(product((-1, 1), repeat=5))
        assert {
            tuple(cell["factors"][name] for name in ("p", "m", "u", "o", "q", "a"))
            for cell in cells
            if cell["family"] == "two_fact_composition"
        } == set(product((-1, 1), repeat=6))


def test_composition_fraction_and_task_relative_coordinates() -> None:
    fixture = _load(FIXTURE)
    worlds = {world["world_id"]: world for world in fixture["worlds"]}
    for cell in fixture["cells"]:
        factors = cell["factors"]
        if cell["family"] in ("single_target_composition", "two_fact_composition"):
            g = worlds[cell["world_id"]]["foldover_g"]
            assert factors["r"] == -factors["p"] * factors["u"]
            assert factors["w"] == g * factors["p"] * factors["m"] * factors["u"]
            assert factors["v"] == factors["p"] * factors["m"] * factors["w"]
            assert factors["p"] * factors["m"] * factors["u"] * factors["w"] == g
        elif cell["family"] == "codebook_lookup":
            assert factors["u"] == -factors["p"] * factors["r"]
            assert factors["w"] == factors["p"] * factors["m"] * factors["v"]


def test_topology_moves_only_registered_blocks() -> None:
    fixture = _load(FIXTURE)
    for cell in fixture["cells"]:
        if cell["family"] == "codebook_lookup":
            continue
        text = cell["prompt_text"]
        answer = text.index("VALID OUTPUTS")
        task = text.index("TASK:")
        facts = text.index("LABELED FACTS")
        assert (task < facts) is (cell["factors"]["q"] == -1)
        assert (answer < facts) is (cell["factors"]["a"] == -1)
        if cell["family"] in ("property_retrieval", "two_fact_composition"):
            target = text.index("TARGET FACT:")
            other = text.index("OTHER FACT:")
            assert (target < other) is (cell["factors"]["o"] == -1)


def test_answer_ledger_and_natural_surfaces_reconstruct() -> None:
    fixture = _load(FIXTURE)
    worlds = {world["world_id"]: world for world in fixture["worlds"]}
    for cell in fixture["cells"]:
        world = worlds[cell["world_id"]]
        p = cell["factors"]["p"]
        prop = world["property_symbols"]["negative" if p == -1 else "positive"]
        expected = (
            prop
            if cell["family"] == "property_retrieval"
            else world["codebooks"][cell["mapping_id"]][prop]
        )
        assert cell["target_property"] == prop
        assert cell["correct_answer"] == expected
        assert set(cell["answer_options"]) == (
            set(world["property_symbols"].values())
            if cell["family"] == "property_retrieval"
            else set(world["code_symbols"].values())
        )
        assert cell["correct_answer_surface"] == " " + expected
        assert cell["distractor_answer_surface"] == " " + cell["distractor_answer"]


def test_single_target_and_two_fact_firewall() -> None:
    fixture = _load(FIXTURE)
    for cell in fixture["cells"]:
        text = cell["prompt_text"]
        if cell["family"] == "single_target_composition":
            assert cell["stage"] == "components"
            assert text.count("TARGET FACT:") == 1
            assert "OTHER FACT:" not in text
        if cell["family"] == "two_fact_composition":
            assert cell["stage"] == "topology"
            assert text.count("TARGET FACT:") == 1
            assert text.count("OTHER FACT:") == 1


def test_builder_has_zero_tokenizer_model_and_biology_calls() -> None:
    fixture = _load(FIXTURE)

    assert fixture["model_calls_made_by_builder"] == 0
    assert fixture["tokenizer_calls_made_by_builder"] == 0
    assert fixture["biological_model_calls"] == 0
    assert all(cell["model_calls_made_by_builder"] == 0 for cell in fixture["cells"])
    assert all(cell["tokenizer_calls_made_by_builder"] == 0 for cell in fixture["cells"])
    assert all(cell["biological_model_calls"] == 0 for cell in fixture["cells"])
    assert len({cell["cell_id"] for cell in fixture["cells"]}) == 2_304
    assert len({cell["prompt_text"] for cell in fixture["cells"]}) == 2_304


def test_validator_rejects_fraction_stage_and_surface_corruption(builder) -> None:
    fixture = _load(FIXTURE)

    changed = copy.deepcopy(fixture)
    target = next(
        cell for cell in changed["cells"] if cell["family"] == "two_fact_composition"
    )
    target["factors"]["w"] *= -1
    with pytest.raises(builder.V6AR2BankError, match="fraction"):
        builder.validate_fixture(changed)

    changed = copy.deepcopy(fixture)
    target = next(
        cell for cell in changed["cells"] if cell["family"] == "single_target_composition"
    )
    target["stage"] = "topology"
    with pytest.raises(builder.V6AR2BankError, match="topology stage"):
        builder.validate_fixture(changed)

    changed = copy.deepcopy(fixture)
    changed["cells"][0]["correct_answer_surface"] = changed["cells"][0]["correct_answer"]
    with pytest.raises(builder.V6AR2BankError, match="answer ledger"):
        builder.validate_fixture(changed)


def test_write_is_idempotent_and_refuses_differing_artifact(builder, tmp_path: Path) -> None:
    output = tmp_path / "r2.json"
    fixture_path, manifest_path = builder.write_fixture(output)
    first_fixture = fixture_path.read_bytes()
    first_manifest = manifest_path.read_bytes()

    builder.write_fixture(output)
    assert fixture_path.read_bytes() == first_fixture
    assert manifest_path.read_bytes() == first_manifest

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.V6AR2BankError, match="refusing to overwrite"):
        builder.write_fixture(output)

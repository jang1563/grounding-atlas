from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "signal"
    / "syntax"
    / "build_coherent_readout_v5_positional_activation_bank.py"
)
FIXTURE_PATH = BUILDER_PATH.with_name(
    "coherent_readout_v5_positional_activation_bank.json"
)
MANIFEST_PATH = BUILDER_PATH.with_name(
    "coherent_readout_v5_positional_activation_bank.manifest.json"
)


def _builder():
    spec = importlib.util.spec_from_file_location("v5_activation_bank_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def fixture(builder) -> dict:
    return builder.build_fixture()


def test_build_is_deterministic_zero_call_and_matches_frozen_artifact(
    builder, fixture: dict
) -> None:
    assert fixture == builder.build_fixture()
    assert builder.validate_fixture(fixture) is fixture
    assert fixture == _json(FIXTURE_PATH)
    assert fixture["model_calls_made_by_builder"] == 0
    assert fixture["tokenizer_calls_made_by_builder"] == 0
    assert fixture["biological_model_calls"] == 0

    manifest = _json(MANIFEST_PATH)
    assert manifest["fixture_file_sha256"] == _sha256(FIXTURE_PATH)
    assert manifest["fixture_canonical_sha256"] == builder.canonical_sha256(fixture)
    assert manifest["builder_file_sha256"] == _sha256(BUILDER_PATH)
    assert manifest["model_calls"] == manifest["tokenizer_calls"] == 0


def test_32_worlds_are_role_disjoint_and_use_every_symbol_combination_once(
    builder, fixture: dict
) -> None:
    worlds = fixture["worlds"]
    assert Counter(world["role"] for world in worlds) == {
        "fit": 8,
        "localization": 8,
        "holdout": 16,
    }
    assert [world["role"] for world in worlds[:8]] == ["fit"] * 8
    assert [world["role"] for world in worlds[8:16]] == ["localization"] * 8
    assert [world["role"] for world in worlds[16:]] == ["holdout"] * 16
    assert len({world["world_id"] for world in worlds}) == 32
    assert len({world["instance_key"] for world in worlds}) == 32
    assert len({world["target_entity"] for world in worlds}) == 32
    assert len({world["other_entity"] for world in worlds}) == 32
    assert len({world["symbol_combination_id"] for world in worlds}) == 32

    expected_combinations = {
        (tuple(properties), tuple(codes))
        for properties, codes in itertools.product(
            builder.PROPERTY_PAIRS, builder.CODE_PAIRS
        )
    }
    observed_combinations = {
        (
            tuple(world["property_symbols"].values()),
            tuple(world["code_symbols"].values()),
        )
        for world in worlds
    }
    assert observed_combinations == expected_combinations

    for role, property_repetitions, code_repetitions in (
        ("fit", 1, 2),
        ("localization", 1, 2),
        ("holdout", 2, 4),
    ):
        role_worlds = [world for world in worlds if world["role"] == role]
        property_counts = Counter(
            tuple(world["property_symbols"].values()) for world in role_worlds
        )
        code_counts = Counter(
            tuple(world["code_symbols"].values()) for world in role_worlds
        )
        assert set(property_counts.values()) == {property_repetitions}
        assert set(property_counts) == set(builder.PROPERTY_PAIRS)
        assert set(code_counts.values()) == {code_repetitions}
        assert set(code_counts) == set(builder.CODE_PAIRS)


def test_symbol_contract_is_single_character_uppercase_and_type_disjoint(
    builder, fixture: dict
) -> None:
    properties = {symbol for pair in builder.PROPERTY_PAIRS for symbol in pair}
    codes = {symbol for pair in builder.CODE_PAIRS for symbol in pair}
    assert len(properties) == 16
    assert len(codes) == 8
    assert properties.isdisjoint(codes)
    assert properties | codes == set("ABCDEFGHIJKLMNOPQRSTUVWX")
    assert fixture["tokenizer_validation_contract"] == {
        "builder_tokenizer_calls": 0,
        "symbols_are_ascii_uppercase_single_characters": True,
        "property_and_code_symbol_pools_are_disjoint": True,
        "required_runner_checks_before_any_forward_call": [
            "each answer symbol is exactly one contextual continuation token",
            "each composition target_property_span maps to exactly one token",
            "the mapped token text equals target_property",
        ],
        "candidate_symbols": list(builder.ALL_SYMBOLS),
    }
    for world in fixture["worlds"]:
        assert world["symbol_validation"] == {
            "symbols": [
                *world["property_symbols"].values(),
                *world["code_symbols"].values(),
            ],
            "all_ascii_uppercase_single_characters": True,
            "all_symbols_distinct_within_world": True,
            "contextual_single_token_status": (
                "prospectively_required_runner_validation"
            ),
        }


def test_every_world_has_exact_56_row_behavioral_baseline_and_full_factorials(
    fixture: dict,
) -> None:
    by_world: dict[str, list[dict]] = defaultdict(list)
    for cell in fixture["cells"]:
        by_world[cell["world_id"]].append(cell)

    assert len(fixture["cells"]) == 1792
    assert Counter(cell["family"] for cell in fixture["cells"]) == {
        "property_retrieval": 256,
        "codebook_lookup": 512,
        "composition": 1024,
    }
    assert len(by_world) == 32
    for rows in by_world.values():
        assert len(rows) == 56
        assert Counter(cell["family"] for cell in rows) == {
            "property_retrieval": 8,
            "codebook_lookup": 16,
            "composition": 32,
        }
        observed = defaultdict(set)
        for cell in rows:
            observed[cell["family"]].add(
                tuple(cell["factors"][factor] for factor in ("p", "m", "r", "v", "o"))
            )
        assert observed["property_retrieval"] == {
            (p, None, None, v, o)
            for p, v, o in itertools.product((-1, 1), repeat=3)
        }
        assert observed["codebook_lookup"] == {
            (p, m, r, v, None)
            for p, m, r, v in itertools.product((-1, 1), repeat=4)
        }
        assert observed["composition"] == set(
            itertools.product((-1, 1), repeat=5)
        )


def test_all_prompt_texts_are_unique_and_answer_ledgers_reconstruct(
    fixture: dict,
) -> None:
    prompts = [cell["prompt_text"] for cell in fixture["cells"]]
    assert len(prompts) == len(set(prompts)) == 1792

    for cell in fixture["cells"]:
        assert f"INSTANCE KEY: {cell['instance_key']}." in cell["prompt_text"]
        assert cell["prompt_sha256"] == hashlib.sha256(
            cell["prompt_text"].encode("utf-8")
        ).hexdigest()
        expected = (
            cell["target_property"]
            if cell["family"] == "property_retrieval"
            else cell["codebook"][cell["target_property"]]
        )
        assert cell["correct_answer"] == expected
        assert expected in cell["answer_options"]
        assert cell["correct_option_position"] == (
            "first" if cell["answer_options"][0] == expected else "last"
        )
        if cell["family"] == "property_retrieval":
            assert "Copy the one-character property" in cell["prompt_text"]
            assert cell["prompt_text"].index("TASK:") < cell["prompt_text"].index(
                "LABELED FACTS"
            )
        elif cell["family"] == "codebook_lookup":
            assert "left-side property exactly matches" in cell["prompt_text"]
        else:
            assert "Read the exact property in TARGET FACT" in cell["prompt_text"]


def test_composition_span_is_exact_and_all_causal_inputs_precede_it(
    fixture: dict,
) -> None:
    for cell in fixture["cells"]:
        if cell["family"] != "composition":
            assert cell["target_property_span"] is None
            assert cell["target_property_span_basis"] is None
            continue
        start, end = cell["target_property_span"]
        prompt = cell["prompt_text"]
        assert prompt[start:end] == cell["target_property"]
        assert cell["target_property_span_basis"] == (
            "zero_based_half_open_prompt_text_characters"
        )
        assert prompt.count("TARGET FACT:") == 1
        target_line = next(
            line for line in cell["fact_lines"] if line.startswith("TARGET FACT:")
        )
        assert start == prompt.index(target_line) + target_line.index(" has property ") + len(
            " has property "
        )
        assert (
            prompt.index("CODEBOOK RULES")
            < prompt.index("VALID OUTPUTS")
            < prompt.index("TASK:")
            < prompt.index("LABELED FACTS")
            < start
        )


def test_fact_order_mates_are_exact_answer_preserving_involutions(
    fixture: dict,
) -> None:
    by_id = {cell["cell_id"]: cell for cell in fixture["cells"]}
    for cell in fixture["cells"]:
        if cell["mate_cell_id"] is None:
            assert cell["family"] == "codebook_lookup"
            continue
        mate = by_id[cell["mate_cell_id"]]
        assert mate["mate_cell_id"] == cell["cell_id"]
        assert mate["semantic_pair_id"] == cell["semantic_pair_id"]
        assert mate["family"] == cell["family"]
        assert mate["world_id"] == cell["world_id"]
        assert mate["correct_answer"] == cell["correct_answer"]
        assert mate["answer_options"] == cell["answer_options"]
        assert mate["factors"] == {**cell["factors"], "o": -cell["factors"]["o"]}
        assert mate["fact_lines"] == list(reversed(cell["fact_lines"]))


def test_fixed_intervention_panel_is_complete_regular_fraction(
    fixture: dict,
) -> None:
    by_id = {cell["cell_id"]: cell for cell in fixture["cells"]}
    flagged = {cell["cell_id"] for cell in fixture["cells"] if cell["recipient_selected"]}
    declared: set[str] = set()

    for world in fixture["worlds"]:
        pairs = world["intervention_pairs"]
        assert len(pairs) == 8
        assert [pair["panel_index"] for pair in pairs] == list(range(1, 9))
        assert {
            (pair["factors"]["p"], pair["factors"]["m"], pair["factors"]["r"])
            for pair in pairs
        } == set(itertools.product((-1, 1), repeat=3))
        assert Counter(pair["correct_answer"] for pair in pairs) == {
            world["code_symbols"]["negative"]: 4,
            world["code_symbols"]["positive"]: 4,
        }
        for pair in pairs:
            p, m, r, v = (
                pair["factors"][name] for name in ("p", "m", "r", "v")
            )
            assert v == p * m * r
            first = by_id[pair["target_first_cell_id"]]
            second = by_id[pair["target_second_cell_id"]]
            assert first["factors"]["o"] == -1
            assert second["factors"]["o"] == 1
            assert first["recipient_selected"] and second["recipient_selected"]
            assert first["intervention_panel_index"] == pair["panel_index"]
            assert second["intervention_panel_index"] == pair["panel_index"]
            declared.update((first["cell_id"], second["cell_id"]))

    assert declared == flagged
    assert len(flagged) == 32 * 8 * 2


def test_direct_intervention_prerequisites_are_exactly_16_per_world(
    fixture: dict,
) -> None:
    by_world: dict[str, list[dict]] = defaultdict(list)
    for cell in fixture["cells"]:
        by_world[cell["world_id"]].append(cell)
    for rows in by_world.values():
        prerequisites = [cell for cell in rows if cell["intervention_prerequisite"]]
        assert Counter(cell["family"] for cell in prerequisites) == {
            "property_retrieval": 8,
            "codebook_lookup": 8,
        }
        assert all(
            cell["factors"]["v"]
            == cell["factors"]["p"]
            * cell["factors"]["m"]
            * cell["factors"]["r"]
            for cell in prerequisites
            if cell["family"] == "codebook_lookup"
        )
        assert not any(
            cell["intervention_prerequisite"]
            for cell in rows
            if cell["family"] == "composition"
        )


def test_validation_rejects_span_prompt_and_panel_mutations(
    builder, fixture: dict
) -> None:
    broken_span = copy.deepcopy(fixture)
    composition = next(
        cell for cell in broken_span["cells"] if cell["family"] == "composition"
    )
    composition["target_property_span"][0] += 1
    with pytest.raises(builder.V5BankError):
        builder.validate_fixture(broken_span)

    broken_prompt = copy.deepcopy(fixture)
    broken_prompt["cells"][1]["prompt_text"] = broken_prompt["cells"][0]["prompt_text"]
    with pytest.raises(builder.V5BankError):
        builder.validate_fixture(broken_prompt)

    broken_panel = copy.deepcopy(fixture)
    broken_panel["worlds"][0]["intervention_pairs"][0]["factors"]["v"] *= -1
    with pytest.raises(builder.V5BankError):
        builder.validate_fixture(broken_panel)

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "signal"
    / "syntax"
    / "build_coherent_readout_v4_behavioral_deconfounding_bank.py"
)


def _builder():
    spec = importlib.util.spec_from_file_location("v4_bank_builder_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_is_deterministic_and_zero_call() -> None:
    builder = _builder()
    first = builder.build_fixture()
    second = builder.build_fixture()
    assert first == second
    builder.validate_fixture(first)
    assert first["model_calls_made_by_builder"] == 0
    assert first["biological_model_calls"] == 0
    assert len(first["cells"]) == 448


def test_worlds_are_new_and_family_allocation_is_exact() -> None:
    fixture = _builder().build_fixture()
    assert [world["world_id"] for world in fixture["world_registry"]] == [
        f"behavior_world_{index:03d}" for index in range(1, 9)
    ]
    assert all(
        not world["world_id"].startswith("symbolic_world_")
        for world in fixture["world_registry"]
    )
    assert Counter(cell["family_id"] for cell in fixture["cells"]) == {
        "composition": 256,
        "property_retrieval": 64,
        "codebook_lookup": 128,
    }


def test_each_world_has_complete_56_row_factorial() -> None:
    cells = _builder().build_fixture()["cells"]
    by_world: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        by_world[cell["world_id"]].append(cell)
    assert set(map(len, by_world.values())) == {56}
    for rows in by_world.values():
        assert Counter(row["family_id"] for row in rows) == {
            "composition": 32,
            "property_retrieval": 8,
            "codebook_lookup": 16,
        }


def test_answer_ledger_options_and_positions_reconstruct() -> None:
    builder = _builder()
    for cell in builder.build_fixture()["cells"]:
        expected = (
            cell["target_property"]
            if cell["family_id"] == "property_retrieval"
            else builder.CODEBOOKS[cell["mapping_id"]][cell["target_property"]]
        )
        assert cell["correct_answer"] == expected
        assert cell["displayed_options"] == builder._displayed_options(
            cell["family_id"], cell["option_order"]
        )
        position = "first" if cell["displayed_options"][0] == expected else "last"
        assert cell["correct_option_position"] == position
        assert cell["last_option_heuristic_answer"] == cell["displayed_options"][-1]


def test_rule_fact_and_prompt_text_reconstruct_exactly() -> None:
    builder = _builder()
    for cell in builder.build_fixture()["cells"]:
        lines, rules, facts = builder._prompt_lines(
            family_id=cell["family_id"],
            target_entity=cell["target_entity"],
            other_entity=cell["other_entity"],
            target_property=cell["target_property"],
            mapping_id=cell["mapping_id"],
            target_fact_order=cell["target_fact_order"],
            rule_order=cell["rule_order"],
            displayed_options=cell["displayed_options"],
        )
        assert cell["prompt_lines"] == lines
        assert cell["rule_lines"] == rules
        assert cell["fact_lines"] == facts
        assert cell["prompt_text"] == "\n".join(lines)
        assert cell["prompt_sha256"] == builder.text_sha256(cell["prompt_text"])


def test_role_labels_and_neutral_system_remove_fixed_label_order() -> None:
    fixture = _builder().build_fixture()
    assert "X" not in fixture["neutral_system_message"]
    assert "Y" not in fixture["neutral_system_message"]
    composition = [
        cell for cell in fixture["cells"] if cell["family_id"] == "composition"
    ]
    assert all("TARGET FACT:" in cell["prompt_text"] for cell in composition)
    assert all("OTHER FACT:" in cell["prompt_text"] for cell in composition)
    assert {tuple(cell["displayed_options"]) for cell in composition} == {
        ("X", "Y"),
        ("Y", "X"),
    }


def test_registered_v3_and_first_rule_policies_reconstruct() -> None:
    builder = _builder()
    for cell in builder.build_fixture()["cells"]:
        expected_v3 = (
            builder._v3_heuristic(
                cell["target_property"],
                cell["mapping_id"],
                cell["target_fact_order"],
            )
            if cell["family_id"] == "composition"
            else None
        )
        assert cell["v3_heuristic_answer"] == expected_v3
        assert cell["first_rule_output_heuristic_answer"] == (
            builder._first_rule_output(cell["mapping_id"], cell["rule_order"])
        )


def test_semantic_bundle_permutations_are_complete() -> None:
    cells = _builder().build_fixture()["cells"]
    bundles: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        bundles[cell["semantic_bundle_id"]].append(cell)
    for bundle_id, rows in bundles.items():
        expected = 8 if bundle_id.startswith("composition:") else 4
        assert len(rows) == expected
        assert {row["permutation_index"] for row in rows} == set(range(expected))


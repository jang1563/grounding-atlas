#!/usr/bin/env python3
"""Build the prospective V4 behavior-only deconfounding bank.

This builder is deterministic and performs no tokenizer or model call.  The
bank uses eight worlds that are disjoint from the V3 symbolic-world registry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "signal"
    / "syntax"
    / "coherent_readout_v4_behavioral_deconfounding_bank.json"
)

FIXTURE_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-bank-v1"
CELL_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-cell-v1"
WORLD_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-world-v1"
MANIFEST_SCHEMA = "coherent-readout-v4-behavioral-deconfounding-bank-manifest-v1"
ANALYSIS_ID = "coherent-readout-v4-behavioral-deconfounding-v1"
FREEZE_DATE = "2026-08-02"
MODE = "prospective_development_synthetic_nonbiological"
PURPOSE = (
    "prospectively distinguish property retrieval, codebook lookup, intended "
    "composition, and registered label/order heuristics before any activation study"
)
REGISTERED_COMPOSITION_POLICIES = [
    "intended_compositional_rule",
    "frozen_v3_heuristic",
    "last_displayed_option",
    "first_displayed_codebook_rule_output",
    "constant_y",
    "constant_x",
]
NEUTRAL_SYSTEM_MESSAGE = (
    "Follow the user's labeled task. Reply with exactly the requested "
    "single-character label and nothing else."
)

WORLD_COUNT = 8
FAMILY_COUNTS = {
    "composition": 256,
    "property_retrieval": 64,
    "codebook_lookup": 128,
}
CODEBOOKS = {
    "identity": {"P": "X", "Q": "Y"},
    "swapped": {"P": "Y", "Q": "X"},
}
TARGET_PROPERTIES = ("P", "Q")
MAPPINGS = ("identity", "swapped")
FACT_ORDERS = ("target_first", "target_second")
RULE_ORDERS = ("p_rule_first", "q_rule_first")
XY_OPTION_ORDERS = ("x_then_y", "y_then_x")
PQ_OPTION_ORDERS = ("p_then_q", "q_then_p")

FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "freeze_date",
    "mode",
    "purpose",
    "neutral_system_message",
    "world_registry",
    "family_counts",
    "expected_call_count",
    "registered_composition_policies",
    "cells",
    "model_calls_made_by_builder",
    "biological_model_calls",
}
WORLD_KEYS = {
    "schema_version",
    "world_id",
    "world_index",
    "target_entity",
    "other_entity",
}
CELL_KEYS = {
    "schema_version",
    "cell_id",
    "world_id",
    "world_index",
    "family_id",
    "stratum_id",
    "target_entity",
    "other_entity",
    "target_property",
    "other_property",
    "mapping_id",
    "target_fact_order",
    "rule_order",
    "option_order",
    "answer_labels",
    "displayed_options",
    "correct_answer",
    "correct_option_position",
    "rule_lines",
    "fact_lines",
    "prompt_lines",
    "prompt_text",
    "prompt_sha256",
    "semantic_bundle_id",
    "permutation_index",
    "v3_heuristic_answer",
    "last_option_heuristic_answer",
    "first_rule_output_heuristic_answer",
    "model_calls_made_by_builder",
    "biological_model_calls",
}


class V4BankError(ValueError):
    """Raised when the deterministic bank contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_answer(target_property: str, mapping_id: str) -> str:
    return CODEBOOKS[mapping_id][target_property]


def _rule_lines(mapping_id: str, rule_order: str) -> list[str]:
    properties = (
        ("P", "Q") if rule_order == "p_rule_first" else ("Q", "P")
    )
    return [
        f"RULE: {property_name} maps to {CODEBOOKS[mapping_id][property_name]}."
        for property_name in properties
    ]


def _displayed_options(family_id: str, option_order: str) -> list[str]:
    if family_id == "property_retrieval":
        return ["P", "Q"] if option_order == "p_then_q" else ["Q", "P"]
    return ["X", "Y"] if option_order == "x_then_y" else ["Y", "X"]


def _fact_lines(
    target_entity: str,
    other_entity: str,
    target_property: str,
    target_fact_order: str,
) -> list[str]:
    other_property = "Q" if target_property == "P" else "P"
    target = f"TARGET FACT: {target_entity} has property {target_property}."
    other = f"OTHER FACT: {other_entity} has property {other_property}."
    return [target, other] if target_fact_order == "target_first" else [other, target]


def _prompt_lines(
    *,
    family_id: str,
    target_entity: str,
    other_entity: str,
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
    rule_order: str | None,
    displayed_options: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    option_text = ", ".join(displayed_options)
    if family_id == "composition":
        if mapping_id is None or rule_order is None or target_fact_order is None:
            raise V4BankError("composition prompt factors are incomplete")
        rules = _rule_lines(mapping_id, rule_order)
        facts = _fact_lines(
            target_entity,
            other_entity,
            target_property,
            target_fact_order,
        )
        lines = [
            "CODEBOOK RULES",
            *rules,
            "LABELED FACTS",
            *facts,
            "TASK: Apply the CODEBOOK RULES to the property in TARGET FACT.",
            f"Valid output codes (display order carries no meaning): {option_text}.",
        ]
        return lines, rules, facts
    if family_id == "property_retrieval":
        if target_fact_order is None:
            raise V4BankError("retrieval fact order is missing")
        facts = _fact_lines(
            target_entity,
            other_entity,
            target_property,
            target_fact_order,
        )
        lines = [
            "LABELED FACTS",
            *facts,
            "TASK: Return the property stated in TARGET FACT.",
            f"Valid output properties (display order carries no meaning): {option_text}.",
        ]
        return lines, [], facts
    if family_id == "codebook_lookup":
        if mapping_id is None or rule_order is None:
            raise V4BankError("lookup rule factors are incomplete")
        rules = _rule_lines(mapping_id, rule_order)
        lines = [
            "CODEBOOK RULES",
            *rules,
            f"GIVEN PROPERTY: {target_property}.",
            "TASK: Return the code assigned by the CODEBOOK RULES to GIVEN PROPERTY.",
            f"Valid output codes (display order carries no meaning): {option_text}.",
        ]
        return lines, rules, []
    raise V4BankError(f"unknown family: {family_id}")


def _v3_heuristic(
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
) -> str | None:
    if mapping_id is None or target_fact_order is None:
        return None
    if (
        target_property == "P"
        and mapping_id == "identity"
        and target_fact_order == "target_first"
    ):
        return "X"
    return "Y"


def _first_rule_output(mapping_id: str | None, rule_order: str | None) -> str | None:
    if mapping_id is None or rule_order is None:
        return None
    first_property = "P" if rule_order == "p_rule_first" else "Q"
    return _mapping_answer(first_property, mapping_id)


def _cell(
    *,
    world: Mapping[str, Any],
    family_id: str,
    target_property: str,
    mapping_id: str | None,
    target_fact_order: str | None,
    rule_order: str | None,
    option_order: str,
    permutation_index: int,
) -> dict[str, Any]:
    displayed = _displayed_options(family_id, option_order)
    answer_labels = ["P", "Q"] if family_id == "property_retrieval" else ["X", "Y"]
    correct_answer = (
        target_property
        if family_id == "property_retrieval"
        else _mapping_answer(target_property, str(mapping_id))
    )
    prompt_lines, rule_lines, fact_lines = _prompt_lines(
        family_id=family_id,
        target_entity=str(world["target_entity"]),
        other_entity=str(world["other_entity"]),
        target_property=target_property,
        mapping_id=mapping_id,
        target_fact_order=target_fact_order,
        rule_order=rule_order,
        displayed_options=displayed,
    )
    prompt_text = "\n".join(prompt_lines)
    factor_parts = [
        f"p-{target_property.lower()}",
        f"m-{mapping_id or 'none'}",
        f"f-{target_fact_order or 'none'}",
        f"r-{rule_order or 'none'}",
        f"o-{option_order}",
    ]
    cell_id = ":".join(
        [
            "behavior-v4",
            str(world["world_id"]),
            family_id,
            *factor_parts,
        ]
    )
    if family_id == "property_retrieval":
        semantic_bundle_id = (
            f"retrieval:{world['world_id']}:p-{target_property.lower()}"
        )
        stratum_id = (
            f"retrieval:p-{target_property.lower()}:f-{target_fact_order}:o-{option_order}"
        )
    elif family_id == "codebook_lookup":
        semantic_bundle_id = (
            f"lookup:{world['world_id']}:p-{target_property.lower()}:m-{mapping_id}"
        )
        stratum_id = (
            f"lookup:p-{target_property.lower()}:m-{mapping_id}:"
            f"r-{rule_order}:o-{option_order}"
        )
    else:
        semantic_bundle_id = (
            f"composition:{world['world_id']}:p-{target_property.lower()}:m-{mapping_id}"
        )
        stratum_id = (
            f"composition:p-{target_property.lower()}:m-{mapping_id}:"
            f"f-{target_fact_order}:r-{rule_order}:o-{option_order}"
        )
    return {
        "schema_version": CELL_SCHEMA,
        "cell_id": cell_id,
        "world_id": world["world_id"],
        "world_index": world["world_index"],
        "family_id": family_id,
        "stratum_id": stratum_id,
        "target_entity": world["target_entity"],
        "other_entity": world["other_entity"],
        "target_property": target_property,
        "other_property": "Q" if target_property == "P" else "P",
        "mapping_id": mapping_id,
        "target_fact_order": target_fact_order,
        "rule_order": rule_order,
        "option_order": option_order,
        "answer_labels": answer_labels,
        "displayed_options": displayed,
        "correct_answer": correct_answer,
        "correct_option_position": (
            "first" if displayed[0] == correct_answer else "last"
        ),
        "rule_lines": rule_lines,
        "fact_lines": fact_lines,
        "prompt_lines": prompt_lines,
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "semantic_bundle_id": semantic_bundle_id,
        "permutation_index": permutation_index,
        "v3_heuristic_answer": (
            _v3_heuristic(target_property, mapping_id, target_fact_order)
            if family_id == "composition"
            else None
        ),
        "last_option_heuristic_answer": displayed[-1],
        "first_rule_output_heuristic_answer": _first_rule_output(
            mapping_id, rule_order
        ),
        "model_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def build_fixture() -> dict[str, Any]:
    worlds = [
        {
            "schema_version": WORLD_SCHEMA,
            "world_id": f"behavior_world_{index:03d}",
            "world_index": index,
            "target_entity": f"referent_a_{index:03d}",
            "other_entity": f"referent_b_{index:03d}",
        }
        for index in range(1, WORLD_COUNT + 1)
    ]
    cells: list[dict[str, Any]] = []
    for world in worlds:
        for target_property, mapping_id in itertools.product(
            TARGET_PROPERTIES, MAPPINGS
        ):
            for permutation_index, (
                target_fact_order,
                rule_order,
                option_order,
            ) in enumerate(
                itertools.product(FACT_ORDERS, RULE_ORDERS, XY_OPTION_ORDERS)
            ):
                cells.append(
                    _cell(
                        world=world,
                        family_id="composition",
                        target_property=target_property,
                        mapping_id=mapping_id,
                        target_fact_order=target_fact_order,
                        rule_order=rule_order,
                        option_order=option_order,
                        permutation_index=permutation_index,
                    )
                )
        for target_property in TARGET_PROPERTIES:
            for permutation_index, (target_fact_order, option_order) in enumerate(
                itertools.product(FACT_ORDERS, PQ_OPTION_ORDERS)
            ):
                cells.append(
                    _cell(
                        world=world,
                        family_id="property_retrieval",
                        target_property=target_property,
                        mapping_id=None,
                        target_fact_order=target_fact_order,
                        rule_order=None,
                        option_order=option_order,
                        permutation_index=permutation_index,
                    )
                )
        for target_property, mapping_id in itertools.product(
            TARGET_PROPERTIES, MAPPINGS
        ):
            for permutation_index, (rule_order, option_order) in enumerate(
                itertools.product(RULE_ORDERS, XY_OPTION_ORDERS)
            ):
                cells.append(
                    _cell(
                        world=world,
                        family_id="codebook_lookup",
                        target_property=target_property,
                        mapping_id=mapping_id,
                        target_fact_order=None,
                        rule_order=rule_order,
                        option_order=option_order,
                        permutation_index=permutation_index,
                    )
                )
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": MODE,
        "purpose": PURPOSE,
        "neutral_system_message": NEUTRAL_SYSTEM_MESSAGE,
        "world_registry": worlds,
        "family_counts": FAMILY_COUNTS,
        "expected_call_count": sum(FAMILY_COUNTS.values()),
        "registered_composition_policies": REGISTERED_COMPOSITION_POLICIES,
        "cells": cells,
        "model_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    validate_fixture(fixture)
    return fixture


def _expected_cell_from_observed(
    cell: Mapping[str, Any], world_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    world = world_by_id.get(str(cell.get("world_id")))
    if world is None:
        raise V4BankError("cell references an unknown world")
    return _cell(
        world=world,
        family_id=str(cell.get("family_id")),
        target_property=str(cell.get("target_property")),
        mapping_id=cell.get("mapping_id"),
        target_fact_order=cell.get("target_fact_order"),
        rule_order=cell.get("rule_order"),
        option_order=str(cell.get("option_order")),
        permutation_index=cell.get("permutation_index"),
    )


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    if set(fixture) != FIXTURE_KEYS:
        raise V4BankError("fixture top-level schema changed")
    if (
        fixture["schema_version"] != FIXTURE_SCHEMA
        or fixture["analysis_id"] != ANALYSIS_ID
        or fixture["freeze_date"] != FREEZE_DATE
        or fixture["mode"] != MODE
        or fixture["purpose"] != PURPOSE
        or fixture["neutral_system_message"] != NEUTRAL_SYSTEM_MESSAGE
        or fixture["family_counts"] != FAMILY_COUNTS
        or fixture["expected_call_count"] != 448
        or fixture["registered_composition_policies"]
        != REGISTERED_COMPOSITION_POLICIES
        or fixture["model_calls_made_by_builder"] != 0
        or fixture["biological_model_calls"] != 0
    ):
        raise V4BankError("fixture header changed")
    worlds = fixture["world_registry"]
    cells = fixture["cells"]
    if not isinstance(worlds, list) or len(worlds) != WORLD_COUNT:
        raise V4BankError("world registry changed")
    if not isinstance(cells, list) or len(cells) != 448:
        raise V4BankError("cell registry changed")
    for index, world in enumerate(worlds, start=1):
        if not isinstance(world, dict) or set(world) != WORLD_KEYS:
            raise V4BankError("world schema changed")
        expected_world = {
            "schema_version": WORLD_SCHEMA,
            "world_id": f"behavior_world_{index:03d}",
            "world_index": index,
            "target_entity": f"referent_a_{index:03d}",
            "other_entity": f"referent_b_{index:03d}",
        }
        if world != expected_world:
            raise V4BankError("world identity changed")
    world_by_id = {world["world_id"]: world for world in worlds}
    if len(world_by_id) != WORLD_COUNT:
        raise V4BankError("world IDs are duplicated")
    cell_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    world_counts: Counter[str] = Counter()
    strata: dict[str, set[str]] = {
        family: set() for family in FAMILY_COUNTS
    }
    bundle_members: Counter[str] = Counter()
    bundle_permutations: dict[str, set[int]] = {}
    for raw_cell in cells:
        if not isinstance(raw_cell, dict) or set(raw_cell) != CELL_KEYS:
            raise V4BankError("cell schema changed")
        if (
            not isinstance(raw_cell.get("world_index"), int)
            or isinstance(raw_cell.get("world_index"), bool)
            or not isinstance(raw_cell.get("permutation_index"), int)
            or isinstance(raw_cell.get("permutation_index"), bool)
            or raw_cell["permutation_index"] < 0
        ):
            raise V4BankError("cell integer identity changed")
        expected = _expected_cell_from_observed(raw_cell, world_by_id)
        if raw_cell != expected:
            raise V4BankError(f"cell does not reconstruct: {raw_cell.get('cell_id')}")
        cell_id = raw_cell["cell_id"]
        if cell_id in cell_ids:
            raise V4BankError("cell IDs are duplicated")
        cell_ids.add(cell_id)
        family = raw_cell["family_id"]
        family_counts[family] += 1
        world_counts[raw_cell["world_id"]] += 1
        strata[family].add(raw_cell["stratum_id"])
        bundle = raw_cell["semantic_bundle_id"]
        bundle_members[bundle] += 1
        bundle_permutations.setdefault(bundle, set()).add(
            raw_cell["permutation_index"]
        )
    if dict(family_counts) != FAMILY_COUNTS:
        raise V4BankError("family allocation changed")
    if set(world_counts.values()) != {56} or set(world_counts) != set(world_by_id):
        raise V4BankError("per-world allocation changed")
    if {family: len(values) for family, values in strata.items()} != {
        "composition": 32,
        "property_retrieval": 8,
        "codebook_lookup": 16,
    }:
        raise V4BankError("full factorial stratum coverage changed")
    expected_bundle_sizes = {
        "composition": 8,
        "retrieval": 4,
        "lookup": 4,
    }
    for bundle, count in bundle_members.items():
        prefix = bundle.split(":", 1)[0]
        expected_size = expected_bundle_sizes[prefix]
        if count != expected_size or bundle_permutations[bundle] != set(
            range(expected_size)
        ):
            raise V4BankError("semantic-bundle permutation coverage changed")


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    if path.exists():
        raise V4BankError(f"refusing to overwrite differing frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V4BankError(f"stale atomic temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_fixture(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    fixture = build_fixture()
    _atomic_write(output, _artifact_bytes(fixture))
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "fixture_path": str(output),
        "fixture_file_sha256": file_sha256(output),
        "fixture_canonical_sha256": canonical_sha256(fixture),
        "builder_path": str(Path(__file__).resolve()),
        "builder_file_sha256": file_sha256(Path(__file__)),
        "world_count": WORLD_COUNT,
        "family_counts": FAMILY_COUNTS,
        "cell_count": len(fixture["cells"]),
        "model_calls": 0,
        "biological_model_calls": 0,
    }
    _atomic_write(manifest_path, _artifact_bytes(manifest))
    return output, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, manifest = write_fixture(args.output)
    print(
        json.dumps(
            {
                "fixture": str(output),
                "manifest": str(manifest),
                "cells": 448,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

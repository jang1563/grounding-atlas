#!/usr/bin/env python3
"""Build the zero-forward V6A direct-component qualification bank.

The bank contains no composition prompts and therefore cannot expose the V6A
TARGET-order or topology effects during model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name(
    "coherent_readout_v6a_component_qualification_bank.json"
)

ANALYSIS_ID = "coherent-readout-v6a-component-qualification-v1"
FIXTURE_SCHEMA = "coherent-readout-v6a-component-qualification-bank-v1"
WORLD_SCHEMA = "coherent-readout-v6a-component-qualification-world-v1"
CELL_SCHEMA = "coherent-readout-v6a-component-qualification-cell-v1"
MANIFEST_SCHEMA = "coherent-readout-v6a-component-qualification-manifest-v1"
DESIGN_DATE = "2026-08-03"
WORLD_COUNT = 8
PER_WORLD_COUNTS = {"property_retrieval": 32, "codebook_lookup": 16}
FAMILY_COUNTS = {
    family: WORLD_COUNT * count for family, count in PER_WORLD_COUNTS.items()
}
EXPECTED_CALL_COUNT = sum(FAMILY_COUNTS.values())

SYSTEM_MESSAGE = (
    "Follow the user's instructions. Your entire response must be exactly one "
    "registered character from the valid output choices. Do not write any other text."
)
ASSISTANT_PREFILL = "ANSWER:"

# These 32 characters are reserved for disposable qualification only. They are
# disjoint from the ASCII uppercase V2--V5 pools and from the frozen V6A pool.
# The runner, not this static builder, must prove contextual single-token use.
QUALIFICATION_SYMBOLS = tuple("αβγδεκλμνπστφΓΔабвгдежзиклмнопрс")
V6A_RESERVED_SYMBOLS = frozenset(
    "àáâäåæçèéêíîóôöøúüýþăąćčđęıłńőœśşšżžơưǎǐǒǔǝǥǧǫǯǵǹǻǽ"
    "туфхцчшэяµÀÁÂ"
)
PRIOR_ASCII_SYMBOLS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

FACTOR_CODING = {
    "p": {"negative": "property_0", "positive": "property_1"},
    "m": {"negative": "identity", "positive": "swapped"},
    "r": {"negative": "property_0_first", "positive": "property_1_first"},
    "v": {"negative": "symbol_0_first", "positive": "symbol_1_first"},
    "o": {"negative": "target_first", "positive": "target_second"},
    "q": {"negative": "task_before_facts", "positive": "task_after_facts"},
    "a": {
        "negative": "valid_answer_set_before_facts",
        "positive": "valid_answer_set_after_facts",
    },
}


class V6AQualificationBankError(ValueError):
    """Raised when the qualification fixture violates its static contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _sign_tag(name: str, value: int | None) -> str:
    return f"{name}{'n' if value == -1 else 'p' if value == 1 else 'x'}"


def _level(name: str, value: int | None) -> str | None:
    if value is None:
        return None
    if value not in (-1, 1):
        raise V6AQualificationBankError(f"factor {name} is not signed")
    return FACTOR_CODING[name]["negative" if value == -1 else "positive"]


def _worlds() -> list[dict[str, Any]]:
    if (
        len(QUALIFICATION_SYMBOLS) != WORLD_COUNT * 4
        or len(set(QUALIFICATION_SYMBOLS)) != len(QUALIFICATION_SYMBOLS)
        or set(QUALIFICATION_SYMBOLS) & V6A_RESERVED_SYMBOLS
        or set(QUALIFICATION_SYMBOLS) & PRIOR_ASCII_SYMBOLS
    ):
        raise V6AQualificationBankError("qualification symbol firewall changed")
    worlds: list[dict[str, Any]] = []
    for index in range(1, WORLD_COUNT + 1):
        offset = (index - 1) * 4
        property_0, property_1, code_0, code_1 = QUALIFICATION_SYMBOLS[offset : offset + 4]
        world_id = f"v6a_qualification_world_{index:03d}"
        worlds.append(
            {
                "schema_version": WORLD_SCHEMA,
                "world_id": world_id,
                "world_index": index,
                "role": "qualification",
                "instance_key": f"V6AQ-W{index:03d}-N{index * 7919 % 10000:04d}",
                "target_entity": f"qualification_entity_t_{index:03d}",
                "other_entity": f"qualification_entity_o_{index:03d}",
                "property_symbols": {"negative": property_0, "positive": property_1},
                "code_symbols": {"negative": code_0, "positive": code_1},
                "codebooks": {
                    "identity": {property_0: code_0, property_1: code_1},
                    "swapped": {property_0: code_1, property_1: code_0},
                },
                "symbols": [property_0, property_1, code_0, code_1],
            }
        )
    return worlds


def _symbol(source: Mapping[str, str], sign: int) -> str:
    return str(source["negative" if sign == -1 else "positive"])


def _mapping_id(m: int) -> str:
    return "identity" if m == -1 else "swapped"


def _ordered_pair(source: Mapping[str, str], order: int) -> list[str]:
    values = [str(source["negative"]), str(source["positive"])]
    return values if order == -1 else list(reversed(values))


def _fact_lines(world: Mapping[str, Any], p: int, o: int) -> list[str]:
    target_property = _symbol(world["property_symbols"], p)
    other_property = _symbol(world["property_symbols"], -p)
    target = f"TARGET FACT: {world['target_entity']} has property {target_property}."
    other = f"OTHER FACT: {world['other_entity']} has property {other_property}."
    return [target, other] if o == -1 else [other, target]


def _topology_lines(
    header: list[str], answer_block: list[str], task_block: list[str], fact_block: list[str], q: int, a: int
) -> list[str]:
    if (q, a) == (-1, -1):
        blocks = (header, answer_block, task_block, fact_block)
    elif (q, a) == (-1, 1):
        blocks = (header, task_block, fact_block, answer_block)
    elif (q, a) == (1, -1):
        blocks = (header, answer_block, fact_block, task_block)
    elif (q, a) == (1, 1):
        blocks = (header, fact_block, task_block, answer_block)
    else:
        raise V6AQualificationBankError("topology factors are not signed")
    return [line for block in blocks for line in block]


def _retrieval_cell(
    world: Mapping[str, Any], *, p: int, v: int, o: int, q: int, a: int
) -> dict[str, Any]:
    options = _ordered_pair(world["property_symbols"], v)
    facts = _fact_lines(world, p, o)
    header = [f"INSTANCE KEY: {world['instance_key']}."]
    answer_block = [
        f"VALID OUTPUTS (display order carries no meaning): {options[0]}, {options[1]}."
    ]
    task_block = [
        "TASK: Copy the one-character property stated in TARGET FACT; return only that character."
    ]
    fact_block = ["LABELED FACTS", *facts]
    lines = _topology_lines(header, answer_block, task_block, fact_block, q, a)
    factors = {"p": p, "m": None, "r": None, "v": v, "o": o, "q": q, "a": a}
    tags = ":".join(_sign_tag(name, factors[name]) for name in factors)
    correct = _symbol(world["property_symbols"], p)
    prompt_text = "\n".join(lines)
    return {
        "schema_version": CELL_SCHEMA,
        "cell_id": f"v6aq:{world['world_id']}:property_retrieval:{tags}",
        "world_id": world["world_id"],
        "world_index": world["world_index"],
        "role": "qualification",
        "family": "property_retrieval",
        "factors": factors,
        "factor_levels": {name: _level(name, value) for name, value in factors.items()},
        "target_property": correct,
        "mapping_id": None,
        "answer_options": options,
        "correct_answer": correct,
        "distractor_answer": next(value for value in options if value != correct),
        "correct_option_position": "first" if options[0] == correct else "last",
        "rule_lines": [],
        "fact_lines": facts,
        "prompt_lines": lines,
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "assistant_prefill": ASSISTANT_PREFILL,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def _lookup_cell(
    world: Mapping[str, Any], *, p: int, m: int, r: int, v: int
) -> dict[str, Any]:
    mapping_id = _mapping_id(m)
    codebook = dict(world["codebooks"][mapping_id])
    properties = _ordered_pair(world["property_symbols"], r)
    rules = [f"RULE: {prop} maps to {codebook[prop]}." for prop in properties]
    options = _ordered_pair(world["code_symbols"], v)
    target_property = _symbol(world["property_symbols"], p)
    correct = str(codebook[target_property])
    lines = [
        f"INSTANCE KEY: {world['instance_key']}.",
        "CODEBOOK RULES",
        *rules,
        f"VALID OUTPUTS (display order carries no meaning): {options[0]}, {options[1]}.",
        f"GIVEN PROPERTY: {target_property}.",
        (
            "TASK: Find the rule whose left-side property exactly matches GIVEN PROPERTY; "
            "return only that rule's right-side character."
        ),
    ]
    factors = {"p": p, "m": m, "r": r, "v": v, "o": None, "q": None, "a": None}
    tags = ":".join(_sign_tag(name, factors[name]) for name in factors)
    prompt_text = "\n".join(lines)
    return {
        "schema_version": CELL_SCHEMA,
        "cell_id": f"v6aq:{world['world_id']}:codebook_lookup:{tags}",
        "world_id": world["world_id"],
        "world_index": world["world_index"],
        "role": "qualification",
        "family": "codebook_lookup",
        "factors": factors,
        "factor_levels": {name: _level(name, value) for name, value in factors.items()},
        "target_property": target_property,
        "mapping_id": mapping_id,
        "answer_options": options,
        "correct_answer": correct,
        "distractor_answer": next(value for value in options if value != correct),
        "correct_option_position": "first" if options[0] == correct else "last",
        "rule_lines": rules,
        "fact_lines": [],
        "prompt_lines": lines,
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "assistant_prefill": ASSISTANT_PREFILL,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def build_fixture() -> dict[str, Any]:
    worlds = _worlds()
    cells: list[dict[str, Any]] = []
    for world in worlds:
        for p, v, o, q, a in itertools.product((-1, 1), repeat=5):
            cells.append(_retrieval_cell(world, p=p, v=v, o=o, q=q, a=a))
        for p, m, r, v in itertools.product((-1, 1), repeat=4):
            cells.append(_lookup_cell(world, p=p, m=m, r=r, v=v))
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "design_date": DESIGN_DATE,
        "mode": "disposable_component_only_no_composition",
        "purpose": (
            "qualify one fixed model and prompt wording using direct components without "
            "observing any V6A topology or composition effect"
        ),
        "system_message": SYSTEM_MESSAGE,
        "assistant_prefill": ASSISTANT_PREFILL,
        "world_count": WORLD_COUNT,
        "per_world_family_counts": PER_WORLD_COUNTS,
        "family_counts": FAMILY_COUNTS,
        "expected_call_count": EXPECTED_CALL_COUNT,
        "factor_coding": FACTOR_CODING,
        "qualification_symbols": list(QUALIFICATION_SYMBOLS),
        "v6a_reserved_symbols_sha256": canonical_sha256(sorted(V6A_RESERVED_SYMBOLS)),
        "calibration_firewall": {
            "composition_calls": 0,
            "composition_gap_or_topology_effect_available_to_selection": False,
            "direct_retrieval_topology_accuracy_available": True,
            "symbol_overlap_with_v2_v5": False,
            "symbol_overlap_with_v6a": False,
            "model_selection_criterion": "direct_component_accuracy_only",
        },
        "tokenizer_validation_contract": {
            "builder_tokenizer_calls": 0,
            "all_symbols_one_contextual_prompt_token": "runner_required",
            "all_symbols_one_contextual_continuation_token": "runner_required",
            "assistant_prefill_is_final_attended_site": "runner_required",
            "topology_mates_same_response_site_index_and_token": "runner_required",
        },
        "worlds": worlds,
        "cells": cells,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    return validate_fixture(fixture)


def validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    worlds = fixture.get("worlds")
    cells = fixture.get("cells")
    if not isinstance(worlds, list) or len(worlds) != WORLD_COUNT:
        raise V6AQualificationBankError("qualification world count changed")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALL_COUNT:
        raise V6AQualificationBankError("qualification cell count changed")
    if fixture.get("calibration_firewall", {}).get("composition_calls") != 0:
        raise V6AQualificationBankError("qualification contains composition calls")
    all_symbols = [symbol for world in worlds for symbol in world["symbols"]]
    if (
        len(all_symbols) != 32
        or len(set(all_symbols)) != 32
        or set(all_symbols) != set(QUALIFICATION_SYMBOLS)
        or set(all_symbols) & V6A_RESERVED_SYMBOLS
        or set(all_symbols) & PRIOR_ASCII_SYMBOLS
    ):
        raise V6AQualificationBankError("qualification symbol allocation changed")

    ids = [cell.get("cell_id") for cell in cells]
    prompts = [cell.get("prompt_text") for cell in cells]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise V6AQualificationBankError("qualification IDs or prompts are duplicated")
    if Counter(cell.get("family") for cell in cells) != Counter(FAMILY_COUNTS):
        raise V6AQualificationBankError("qualification family counts changed")
    world_counts = Counter(cell.get("world_id") for cell in cells)
    if set(world_counts.values()) != {sum(PER_WORLD_COUNTS.values())}:
        raise V6AQualificationBankError("qualification per-world counts changed")
    if any(cell.get("family") == "composition" for cell in cells):
        raise V6AQualificationBankError("qualification leaked a composition cell")

    expected_retrieval = set(itertools.product((-1, 1), repeat=5))
    expected_lookup = set(itertools.product((-1, 1), repeat=4))
    for world in worlds:
        members = [cell for cell in cells if cell["world_id"] == world["world_id"]]
        retrieval = {
            tuple(cell["factors"][name] for name in ("p", "v", "o", "q", "a"))
            for cell in members
            if cell["family"] == "property_retrieval"
        }
        lookup = {
            tuple(cell["factors"][name] for name in ("p", "m", "r", "v"))
            for cell in members
            if cell["family"] == "codebook_lookup"
        }
        if retrieval != expected_retrieval or lookup != expected_lookup:
            raise V6AQualificationBankError("qualification factorial coverage changed")
        for cell in members:
            if cell["correct_answer"] not in cell["answer_options"]:
                raise V6AQualificationBankError("correct answer is not a valid option")
            if cell["distractor_answer"] == cell["correct_answer"]:
                raise V6AQualificationBankError("answer pair collapsed")
            if cell["prompt_sha256"] != text_sha256(cell["prompt_text"]):
                raise V6AQualificationBankError("prompt hash changed")
            if cell["assistant_prefill"] != ASSISTANT_PREFILL:
                raise V6AQualificationBankError("assistant prefill changed")
            if f"INSTANCE KEY: {world['instance_key']}." not in cell["prompt_text"]:
                raise V6AQualificationBankError("prompt omits its instance key")
    return dict(fixture)


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
        raise V6AQualificationBankError(f"refusing to overwrite differing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AQualificationBankError(f"stale temporary artifact exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_fixture(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    fixture = build_fixture()
    _atomic_write(output, _artifact_bytes(fixture))
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "design_date": DESIGN_DATE,
        "fixture_path": relative_path(output),
        "fixture_file_sha256": file_sha256(output),
        "fixture_canonical_sha256": canonical_sha256(fixture),
        "builder_path": relative_path(Path(__file__)),
        "builder_file_sha256": file_sha256(Path(__file__)),
        "world_count": WORLD_COUNT,
        "family_counts": FAMILY_COUNTS,
        "cell_count": EXPECTED_CALL_COUNT,
        "composition_cell_count": 0,
        "model_calls": 0,
        "tokenizer_calls": 0,
        "biological_model_calls": 0,
    }
    _atomic_write(manifest_path, _artifact_bytes(manifest))
    return output, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture, manifest = write_fixture(args.output)
    print(
        json.dumps(
            {
                "fixture": relative_path(fixture),
                "manifest": relative_path(manifest),
                "worlds": WORLD_COUNT,
                "cells": EXPECTED_CALL_COUNT,
                "composition_calls": 0,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the zero-forward V6A-R2 natural-token topology bank."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name("coherent_readout_v6a_r2_topology_bank.json")

ANALYSIS_ID = "coherent-readout-v6a-r2-natural-token-topology-v1"
FIXTURE_SCHEMA = "coherent-readout-v6a-r2-topology-bank-v1"
WORLD_SCHEMA = "coherent-readout-v6a-r2-topology-world-v1"
CELL_SCHEMA = "coherent-readout-v6a-r2-topology-cell-v1"
MANIFEST_SCHEMA = "coherent-readout-v6a-r2-topology-manifest-v1"
DESIGN_DATE = "2026-08-03"
REGISTRATION_STATE = "FROZEN_BEFORE_ANY_R2_MODEL_FORWARD"

WORLD_COUNT = 16
ROLE_WORLD_COUNT = 8
PER_WORLD_FAMILY_COUNTS = {
    "property_retrieval": 32,
    "codebook_lookup": 16,
    "single_target_composition": 32,
    "two_fact_composition": 64,
}
FAMILY_COUNTS = {
    family: WORLD_COUNT * count for family, count in PER_WORLD_FAMILY_COUNTS.items()
}
PER_ROLE_FAMILY_COUNTS = {
    family: ROLE_WORLD_COUNT * count
    for family, count in PER_WORLD_FAMILY_COUNTS.items()
}
EXPECTED_CALL_COUNT = sum(FAMILY_COUNTS.values())
DISCOVERY_COMPONENT_CALL_COUNT = ROLE_WORLD_COUNT * (
    PER_WORLD_FAMILY_COUNTS["property_retrieval"]
    + PER_WORLD_FAMILY_COUNTS["codebook_lookup"]
    + PER_WORLD_FAMILY_COUNTS["single_target_composition"]
)
REMAINING_MAIN_CALL_COUNT = EXPECTED_CALL_COUNT - DISCOVERY_COMPONENT_CALL_COUNT
EXECUTION_BLOCK_COUNTS = {
    "discovery-components": 640,
    "discovery-topology": 512,
    "confirmation-components": 640,
    "confirmation-topology": 512,
}
EXECUTION_BLOCK_ORDER = tuple(EXECUTION_BLOCK_COUNTS)

SYSTEM_MESSAGE = (
    "Follow the user's instructions. The assistant message is prefilled with ANSWER:. "
    "Continue it with exactly one ASCII space followed by exactly one registered character "
    "from the valid output choices, and nothing else."
)
ASSISTANT_PREFILL = "ANSWER:"
NATURAL_SURFACE_PREFIX = " "

ORIGINAL_V6A_SYMBOLS = tuple(
    "àáâäåæçèéêíîóôöøúüýþăąćčđęıłńőœśşšżžơưǎǐǒǔǝǥǧǫǯǵǹǻǽ"
    "туфхцчшэяµÀÁÂ"
)
NATURAL_TOKEN_INVALID_SYMBOLS = tuple("ăąćęıńőơưǎǐǒǔǝǥǧǫǯǵǹǻǽ")
REPLACEMENT_SYMBOLS = tuple("ŻЧАŚЯКŁŞÜУОÅНЕШЮіЦÎÈĐГ")
R2_SYMBOLS = tuple(
    "àáâäåæçèéêíîóôöøúüýþŻЧАčđŚЯłКŁœśşšżžŞÜУОÅНЕШЮіЦÎÈĐГ"
    "туфхцчшэяµÀÁÂ"
)
V2_QUALIFICATION_SYMBOLS = frozenset("αβγδεκλμνπστφΓΔабвгдежзиклмнопрс")
PRIOR_ASCII_SYMBOLS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SELECTION_SALT = "V6A-R2-main-replacement-v1|"
SELECTION_UNICODE_VERSION = "15.1.0"
SELECTION_CODEPOINT_RANGES = ((0x00C0, 0x02AF), (0x0370, 0x052F))
TOKENIZER_ELIGIBLE_CANDIDATES_HASH_ORDER = tuple(
    "ŻЧАŚЯКŁŞÜУОÅНЕШЮіЦÎÈĐГПÃÄÖØИЗЛÇДФÔÉСЖБХЭМТРİВ"
)

FACTOR_NAMES = ("p", "m", "r", "v", "u", "w", "o", "q", "a")
FACTOR_CODING = {
    "p": {"negative": "property_0", "positive": "property_1"},
    "m": {"negative": "identity", "positive": "swapped"},
    "r": {"negative": "property_0_first", "positive": "property_1_first"},
    "v": {"negative": "symbol_0_first", "positive": "symbol_1_first"},
    "u": {"negative": "target_rule_first", "positive": "target_rule_last"},
    "w": {"negative": "correct_option_first", "positive": "correct_option_last"},
    "o": {"negative": "target_first", "positive": "target_second"},
    "q": {"negative": "task_before_facts", "positive": "task_after_facts"},
    "a": {
        "negative": "valid_answer_set_before_facts",
        "positive": "valid_answer_set_after_facts",
    },
}


class V6AR2BankError(ValueError):
    """Raised when the R2 fixture violates its zero-forward static contract."""


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
    suffix = "n" if value == -1 else "p" if value == 1 else "x"
    return f"{name}{suffix}"


def _level(name: str, value: int | None) -> str | None:
    if value is None:
        return None
    if value not in (-1, 1):
        raise V6AR2BankError(f"factor {name} is not signed")
    return FACTOR_CODING[name]["negative" if value == -1 else "positive"]


def _factor_payload(**values: int | None) -> tuple[dict[str, int | None], dict[str, str | None]]:
    factors = {name: values.get(name) for name in FACTOR_NAMES}
    return factors, {name: _level(name, factors[name]) for name in FACTOR_NAMES}


def _symbol(source: Mapping[str, str], sign: int) -> str:
    if sign not in (-1, 1):
        raise V6AR2BankError("symbol sign is not coded as -1/+1")
    return str(source["negative" if sign == -1 else "positive"])


def _ordered_pair(source: Mapping[str, str], order: int) -> list[str]:
    if order not in (-1, 1):
        raise V6AR2BankError("display order is not coded as -1/+1")
    values = [str(source["negative"]), str(source["positive"])]
    return values if order == -1 else list(reversed(values))


def _mapping_id(m: int) -> str:
    if m not in (-1, 1):
        raise V6AR2BankError("mapping is not coded as -1/+1")
    return "identity" if m == -1 else "swapped"


def _execution_block(role: str, stage: str) -> str:
    if role not in ("discovery", "confirmation") or stage not in ("components", "topology"):
        raise V6AR2BankError("unknown role/stage pair")
    return f"{role}-{stage}"


def _worlds() -> list[dict[str, Any]]:
    if (
        len(ORIGINAL_V6A_SYMBOLS) != 64
        or len(R2_SYMBOLS) != 64
        or len(set(R2_SYMBOLS)) != 64
        or len(NATURAL_TOKEN_INVALID_SYMBOLS) != 22
        or len(REPLACEMENT_SYMBOLS) != 22
        or TOKENIZER_ELIGIBLE_CANDIDATES_HASH_ORDER[:22] != REPLACEMENT_SYMBOLS
        or len(TOKENIZER_ELIGIBLE_CANDIDATES_HASH_ORDER) != 45
        or set(R2_SYMBOLS) & V2_QUALIFICATION_SYMBOLS
        or set(R2_SYMBOLS) & PRIOR_ASCII_SYMBOLS
    ):
        raise V6AR2BankError("R2 symbol registry or replacement provenance changed")
    resolved = iter(REPLACEMENT_SYMBOLS)
    reconstructed = tuple(
        next(resolved) if symbol in set(NATURAL_TOKEN_INVALID_SYMBOLS) else symbol
        for symbol in ORIGINAL_V6A_SYMBOLS
    )
    if reconstructed != R2_SYMBOLS:
        raise V6AR2BankError("R2 replacement slots do not reconstruct the registry")

    worlds: list[dict[str, Any]] = []
    for global_index in range(1, WORLD_COUNT + 1):
        role = "discovery" if global_index <= ROLE_WORLD_COUNT else "confirmation"
        role_index = global_index if role == "discovery" else global_index - ROLE_WORLD_COUNT
        offset = (global_index - 1) * 4
        property_0, property_1, code_0, code_1 = R2_SYMBOLS[offset : offset + 4]
        role_tag = "D" if role == "discovery" else "C"
        world_id = f"v6a_r2_{role}_world_{role_index:03d}"
        worlds.append(
            {
                "schema_version": WORLD_SCHEMA,
                "world_id": world_id,
                "global_index": global_index,
                "role": role,
                "role_index": role_index,
                "foldover_g": -1 if role_index % 2 else 1,
                "instance_key": (
                    f"V6A-R2-{role_tag}-W{role_index:03d}-"
                    f"N{global_index * 6151 % 10000:04d}"
                ),
                "target_entity": f"v6a_r2_{role}_entity_t_{role_index:03d}",
                "other_entity": f"v6a_r2_{role}_entity_o_{role_index:03d}",
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


def _topology_lines(
    header: Sequence[str],
    answer_block: Sequence[str],
    task_block: Sequence[str],
    fact_block: Sequence[str],
    q: int,
    a: int,
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
        raise V6AR2BankError("topology factors are not signed")
    return [line for block in blocks for line in block]


def _fact_lines(world: Mapping[str, Any], p: int, o: int) -> list[str]:
    target_property = _symbol(world["property_symbols"], p)
    other_property = _symbol(world["property_symbols"], -p)
    target = f"TARGET FACT: {world['target_entity']} has property {target_property}."
    other = f"OTHER FACT: {world['other_entity']} has property {other_property}."
    return [target, other] if o == -1 else [other, target]


def _cell(
    world: Mapping[str, Any],
    *,
    family: str,
    stage: str,
    factors: Mapping[str, int | None],
    target_property: str,
    mapping_id: str | None,
    answer_options: Sequence[str],
    correct_answer: str,
    rule_lines: Sequence[str],
    fact_lines: Sequence[str],
    prompt_lines: Sequence[str],
) -> dict[str, Any]:
    if len(answer_options) != 2 or correct_answer not in answer_options:
        raise V6AR2BankError("answer ledger is malformed")
    distractor = next(answer for answer in answer_options if answer != correct_answer)
    ordered_factors = {name: factors.get(name) for name in FACTOR_NAMES}
    tags = ":".join(_sign_tag(name, ordered_factors[name]) for name in FACTOR_NAMES)
    prompt_text = "\n".join(prompt_lines)
    return {
        "schema_version": CELL_SCHEMA,
        "cell_id": f"v6a-r2:{world['world_id']}:{family}:{tags}",
        "world_id": world["world_id"],
        "global_index": world["global_index"],
        "role": world["role"],
        "role_index": world["role_index"],
        "foldover_g": world["foldover_g"],
        "stage": stage,
        "execution_block": _execution_block(str(world["role"]), stage),
        "family": family,
        "factors": ordered_factors,
        "factor_levels": {
            name: _level(name, ordered_factors[name]) for name in FACTOR_NAMES
        },
        "target_property": target_property,
        "mapping_id": mapping_id,
        "answer_options": list(answer_options),
        "correct_answer": correct_answer,
        "distractor_answer": distractor,
        "correct_answer_surface": NATURAL_SURFACE_PREFIX + correct_answer,
        "distractor_answer_surface": NATURAL_SURFACE_PREFIX + distractor,
        "correct_option_position": "first" if answer_options[0] == correct_answer else "last",
        "rule_lines": list(rule_lines),
        "fact_lines": list(fact_lines),
        "prompt_lines": list(prompt_lines),
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "assistant_prefill": ASSISTANT_PREFILL,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def _retrieval_cell(
    world: Mapping[str, Any], *, p: int, v: int, o: int, q: int, a: int
) -> dict[str, Any]:
    options = _ordered_pair(world["property_symbols"], v)
    facts = _fact_lines(world, p, o)
    header = [f"INSTANCE KEY: {world['instance_key']}."]
    answer = [
        f"VALID OUTPUTS (display order carries no meaning): {options[0]}, {options[1]}."
    ]
    task = [
        "TASK: Copy the one-character property stated in TARGET FACT; return only that character."
    ]
    fact_block = ["LABELED FACTS", *facts]
    factors, _ = _factor_payload(p=p, v=v, o=o, q=q, a=a)
    correct = _symbol(world["property_symbols"], p)
    return _cell(
        world,
        family="property_retrieval",
        stage="components",
        factors=factors,
        target_property=correct,
        mapping_id=None,
        answer_options=options,
        correct_answer=correct,
        rule_lines=[],
        fact_lines=facts,
        prompt_lines=_topology_lines(header, answer, task, fact_block, q, a),
    )


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
    u = -p * r
    w = p * m * v
    factors, _ = _factor_payload(p=p, m=m, r=r, v=v, u=u, w=w)
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
    return _cell(
        world,
        family="codebook_lookup",
        stage="components",
        factors=factors,
        target_property=target_property,
        mapping_id=mapping_id,
        answer_options=options,
        correct_answer=correct,
        rule_lines=rules,
        fact_lines=[],
        prompt_lines=lines,
    )


def _composition_cell(
    world: Mapping[str, Any],
    *,
    p: int,
    m: int,
    u: int,
    q: int,
    a: int,
    o: int | None,
) -> dict[str, Any]:
    g = int(world["foldover_g"])
    w = g * p * m * u
    r = -p * u
    v = p * m * w
    mapping_id = _mapping_id(m)
    codebook = dict(world["codebooks"][mapping_id])
    properties = _ordered_pair(world["property_symbols"], r)
    rules = [f"RULE: {prop} maps to {codebook[prop]}." for prop in properties]
    options = _ordered_pair(world["code_symbols"], v)
    target_property = _symbol(world["property_symbols"], p)
    correct = str(codebook[target_property])
    header = [f"INSTANCE KEY: {world['instance_key']}.", "CODEBOOK RULES", *rules]
    answer = [
        f"VALID OUTPUTS (display order carries no meaning): {options[0]}, {options[1]}."
    ]
    task = [
        (
            "TASK: Read the one-character property in TARGET FACT, find its exact CODEBOOK "
            "RULE, and return only that rule's right-side character."
        )
    ]
    if o is None:
        facts = [f"TARGET FACT: {world['target_entity']} has property {target_property}."]
        family = "single_target_composition"
        stage = "components"
    else:
        facts = _fact_lines(world, p, o)
        family = "two_fact_composition"
        stage = "topology"
    fact_block = ["LABELED FACTS", *facts]
    factors, _ = _factor_payload(p=p, m=m, r=r, v=v, u=u, w=w, o=o, q=q, a=a)
    return _cell(
        world,
        family=family,
        stage=stage,
        factors=factors,
        target_property=target_property,
        mapping_id=mapping_id,
        answer_options=options,
        correct_answer=correct,
        rule_lines=rules,
        fact_lines=facts,
        prompt_lines=_topology_lines(header, answer, task, fact_block, q, a),
    )


def _component_cells(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for p, v, o, q, a in itertools.product((-1, 1), repeat=5):
        cells.append(_retrieval_cell(world, p=p, v=v, o=o, q=q, a=a))
    for p, m, r, v in itertools.product((-1, 1), repeat=4):
        cells.append(_lookup_cell(world, p=p, m=m, r=r, v=v))
    for p, m, u in itertools.product((-1, 1), repeat=3):
        for q, a in itertools.product((-1, 1), repeat=2):
            cells.append(_composition_cell(world, p=p, m=m, u=u, q=q, a=a, o=None))
    return cells


def _topology_cells(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _composition_cell(world, p=p, m=m, u=u, o=o, q=q, a=a)
        for p, m, u in itertools.product((-1, 1), repeat=3)
        for o, q, a in itertools.product((-1, 1), repeat=3)
    ]


def build_fixture() -> dict[str, Any]:
    worlds = _worlds()
    discovery = [world for world in worlds if world["role"] == "discovery"]
    confirmation = [world for world in worlds if world["role"] == "confirmation"]
    cells = [cell for world in discovery for cell in _component_cells(world)]
    cells.extend(cell for world in discovery for cell in _topology_cells(world))
    cells.extend(cell for world in confirmation for cell in _component_cells(world))
    cells.extend(cell for world in confirmation for cell in _topology_cells(world))
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "design_date": DESIGN_DATE,
        "registration_state": REGISTRATION_STATE,
        "mode": "fresh_behavioral_topology_components_first",
        "purpose": (
            "identify replicated response-topology effects before any activation intervention"
        ),
        "system_message": SYSTEM_MESSAGE,
        "assistant_prefill": ASSISTANT_PREFILL,
        "natural_answer_surface_prefix": NATURAL_SURFACE_PREFIX,
        "world_count": WORLD_COUNT,
        "role_world_count": ROLE_WORLD_COUNT,
        "per_world_family_counts": PER_WORLD_FAMILY_COUNTS,
        "family_counts": FAMILY_COUNTS,
        "per_role_family_counts": PER_ROLE_FAMILY_COUNTS,
        "expected_call_count": EXPECTED_CALL_COUNT,
        "discovery_component_call_count": DISCOVERY_COMPONENT_CALL_COUNT,
        "remaining_main_call_count": REMAINING_MAIN_CALL_COUNT,
        "execution_block_order": list(EXECUTION_BLOCK_ORDER),
        "execution_block_counts": EXECUTION_BLOCK_COUNTS,
        "factor_coding": FACTOR_CODING,
        "r2_symbols": list(R2_SYMBOLS),
        "symbol_replacement_provenance": {
            "original_v6a_symbols": list(ORIGINAL_V6A_SYMBOLS),
            "natural_token_invalid_symbols_in_slot_order": list(
                NATURAL_TOKEN_INVALID_SYMBOLS
            ),
            "replacement_symbols_in_slot_order": list(REPLACEMENT_SYMBOLS),
            "selection_salt": SELECTION_SALT,
            "unicode_database_version": SELECTION_UNICODE_VERSION,
            "candidate_codepoint_ranges_inclusive": [
                [lower, upper] for lower, upper in SELECTION_CODEPOINT_RANGES
            ],
            "eligible_candidates_in_hash_order": list(
                TOKENIZER_ELIGIBLE_CANDIDATES_HASH_ORDER
            ),
            "eligibility": (
                "single NFC letter scalar; fresh; exact one-token decode of ASCII-space+glyph"
            ),
            "model_calls": 0,
            "tokenizer_calls_by_builder": 0,
        },
        "firewall": {
            "v2_qualification_symbol_overlap": False,
            "prior_ascii_symbol_overlap": False,
            "discovery_confirmation_symbol_overlap": False,
            "v2_two_fact_composition_calls": 0,
            "v2_single_target_composition_calls": 0,
            "selection_by_model_output": False,
        },
        "tokenizer_validation_contract": {
            "builder_tokenizer_calls": 0,
            "natural_surface_exact_decode": "runner_required",
            "one_token_prefix_extension": "runner_required",
            "prompt_occurrence_id_equals_natural_continuation_id": "runner_required",
            "bare_token_variants_prohibited": "runner_required",
            "colon_response_site_token_id_25": "runner_required",
            "topology_mates_same_response_shape": "runner_required",
        },
        "worlds": worlds,
        "cells": cells,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    return validate_fixture(fixture)


def _world_family_factorials(
    world: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> None:
    members = [cell for cell in cells if cell["world_id"] == world["world_id"]]
    expected = {
        "property_retrieval": set(itertools.product((-1, 1), repeat=5)),
        "codebook_lookup": set(itertools.product((-1, 1), repeat=4)),
        "single_target_composition": set(itertools.product((-1, 1), repeat=5)),
        "two_fact_composition": set(itertools.product((-1, 1), repeat=6)),
    }
    observed = {
        "property_retrieval": {
            tuple(cell["factors"][name] for name in ("p", "v", "o", "q", "a"))
            for cell in members
            if cell["family"] == "property_retrieval"
        },
        "codebook_lookup": {
            tuple(cell["factors"][name] for name in ("p", "m", "r", "v"))
            for cell in members
            if cell["family"] == "codebook_lookup"
        },
        "single_target_composition": {
            tuple(cell["factors"][name] for name in ("p", "m", "u", "q", "a"))
            for cell in members
            if cell["family"] == "single_target_composition"
        },
        "two_fact_composition": {
            tuple(cell["factors"][name] for name in ("p", "m", "u", "o", "q", "a"))
            for cell in members
            if cell["family"] == "two_fact_composition"
        },
    }
    if observed != expected:
        raise V6AR2BankError("R2 within-world factorial coverage changed")


def _validate_answer_ledger(
    world: Mapping[str, Any], cell: Mapping[str, Any]
) -> None:
    factors = cell["factors"]
    p = factors["p"]
    if cell["target_property"] != _symbol(world["property_symbols"], p):
        raise V6AR2BankError("target-property ledger changed")
    family = cell["family"]
    if family == "property_retrieval":
        expected = cell["target_property"]
        allowed = set(world["property_symbols"].values())
    else:
        mapping_id = _mapping_id(factors["m"])
        expected = world["codebooks"][mapping_id][cell["target_property"]]
        allowed = set(world["code_symbols"].values())
        if cell["mapping_id"] != mapping_id:
            raise V6AR2BankError("mapping ledger changed")
    if (
        cell["correct_answer"] != expected
        or set(cell["answer_options"]) != allowed
        or cell["distractor_answer"] == expected
        or cell["correct_answer_surface"] != NATURAL_SURFACE_PREFIX + expected
        or cell["distractor_answer_surface"]
        != NATURAL_SURFACE_PREFIX + cell["distractor_answer"]
    ):
        raise V6AR2BankError("answer ledger changed")


def validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    worlds = fixture.get("worlds")
    cells = fixture.get("cells")
    if not isinstance(worlds, list) or len(worlds) != WORLD_COUNT:
        raise V6AR2BankError("R2 world count changed")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALL_COUNT:
        raise V6AR2BankError("R2 cell count changed")
    if (
        fixture.get("schema_version") != FIXTURE_SCHEMA
        or fixture.get("registration_state") != REGISTRATION_STATE
        or fixture.get("system_message") != SYSTEM_MESSAGE
        or fixture.get("assistant_prefill") != ASSISTANT_PREFILL
        or fixture.get("natural_answer_surface_prefix") != NATURAL_SURFACE_PREFIX
        or fixture.get("r2_symbols") != list(R2_SYMBOLS)
        or fixture.get("execution_block_order") != list(EXECUTION_BLOCK_ORDER)
        or fixture.get("execution_block_counts") != EXECUTION_BLOCK_COUNTS
        or fixture.get("expected_call_count") != EXPECTED_CALL_COUNT
        or fixture.get("discovery_component_call_count") != DISCOVERY_COMPONENT_CALL_COUNT
        or fixture.get("remaining_main_call_count") != REMAINING_MAIN_CALL_COUNT
        or fixture.get("model_calls_made_by_builder") != 0
        or fixture.get("tokenizer_calls_made_by_builder") != 0
    ):
        raise V6AR2BankError("R2 top-level contract changed")

    all_symbols = [symbol for world in worlds for symbol in world["symbols"]]
    if (
        all_symbols != list(R2_SYMBOLS)
        or len(set(all_symbols)) != 64
        or set(all_symbols) & V2_QUALIFICATION_SYMBOLS
        or set(all_symbols) & PRIOR_ASCII_SYMBOLS
    ):
        raise V6AR2BankError("R2 symbol firewall changed")
    discovery_symbols = {
        symbol for world in worlds if world["role"] == "discovery" for symbol in world["symbols"]
    }
    confirmation_symbols = set(all_symbols) - discovery_symbols
    if len(discovery_symbols) != 32 or len(confirmation_symbols) != 32 or discovery_symbols & confirmation_symbols:
        raise V6AR2BankError("R2 split symbol firewall changed")
    if len({world["world_id"] for world in worlds}) != WORLD_COUNT:
        raise V6AR2BankError("R2 world IDs are not unique")
    for field in ("instance_key", "target_entity", "other_entity"):
        if len({world[field] for world in worlds}) != WORLD_COUNT:
            raise V6AR2BankError(f"R2 world field is not unique: {field}")
    if Counter(world["role"] for world in worlds) != {"discovery": 8, "confirmation": 8}:
        raise V6AR2BankError("R2 role allocation changed")
    for role in ("discovery", "confirmation"):
        role_worlds = [world for world in worlds if world["role"] == role]
        if Counter(world["foldover_g"] for world in role_worlds) != {-1: 4, 1: 4}:
            raise V6AR2BankError("R2 foldover allocation changed")

    ids = [cell.get("cell_id") for cell in cells]
    prompts = [cell.get("prompt_text") for cell in cells]
    if len(set(ids)) != EXPECTED_CALL_COUNT or len(set(prompts)) != EXPECTED_CALL_COUNT:
        raise V6AR2BankError("R2 cell IDs or prompts are duplicated")
    if Counter(cell.get("family") for cell in cells) != Counter(FAMILY_COUNTS):
        raise V6AR2BankError("R2 family counts changed")
    if Counter(cell.get("execution_block") for cell in cells) != Counter(EXECUTION_BLOCK_COUNTS):
        raise V6AR2BankError("R2 execution-block counts changed")
    block_rank = {block: index for index, block in enumerate(EXECUTION_BLOCK_ORDER)}
    ranks = [block_rank.get(cell.get("execution_block"), -1) for cell in cells]
    if ranks != sorted(ranks):
        raise V6AR2BankError("R2 execution-block order changed")

    world_lookup = {world["world_id"]: world for world in worlds}
    for world in worlds:
        members = [cell for cell in cells if cell["world_id"] == world["world_id"]]
        if Counter(cell["family"] for cell in members) != Counter(PER_WORLD_FAMILY_COUNTS):
            raise V6AR2BankError("R2 per-world family counts changed")
        _world_family_factorials(world, cells)
    for cell in cells:
        world = world_lookup.get(cell.get("world_id"))
        if world is None or cell.get("role") != world["role"]:
            raise V6AR2BankError("R2 cell does not resolve to its world")
        if cell.get("prompt_sha256") != text_sha256(cell.get("prompt_text", "")):
            raise V6AR2BankError("R2 prompt hash changed")
        if cell.get("prompt_lines") != cell.get("prompt_text", "").split("\n"):
            raise V6AR2BankError("R2 prompt line registry changed")
        if cell.get("assistant_prefill") != ASSISTANT_PREFILL:
            raise V6AR2BankError("R2 assistant prefill changed")
        if f"INSTANCE KEY: {world['instance_key']}." not in cell["prompt_text"]:
            raise V6AR2BankError("R2 prompt omits its instance key")
        if cell["stage"] == "topology" and cell["family"] != "two_fact_composition":
            raise V6AR2BankError("R2 topology stage contains a non-topology family")
        if cell["stage"] == "components" and cell["family"] == "two_fact_composition":
            raise V6AR2BankError("R2 component stage contains two-fact topology")
        _validate_answer_ledger(world, cell)
        factors = cell["factors"]
        if cell["family"] in ("single_target_composition", "two_fact_composition"):
            if (
                factors["r"] != -factors["p"] * factors["u"]
                or factors["w"] != world["foldover_g"] * factors["p"] * factors["m"] * factors["u"]
                or factors["v"] != factors["p"] * factors["m"] * factors["w"]
            ):
                raise V6AR2BankError("R2 composition fraction changed")
        if cell["family"] == "codebook_lookup" and (
            factors["u"] != -factors["p"] * factors["r"]
            or factors["w"] != factors["p"] * factors["m"] * factors["v"]
        ):
            raise V6AR2BankError("R2 lookup task-relative ledger changed")
    return dict(fixture)


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    if path.exists():
        raise V6AR2BankError(f"refusing to overwrite differing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V6AR2BankError(f"stale temporary artifact exists: {temporary}")
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
        "registration_state": REGISTRATION_STATE,
        "fixture_path": relative_path(output),
        "fixture_file_sha256": file_sha256(output),
        "fixture_canonical_sha256": canonical_sha256(fixture),
        "builder_path": relative_path(Path(__file__)),
        "builder_file_sha256": file_sha256(Path(__file__)),
        "world_count": WORLD_COUNT,
        "family_counts": FAMILY_COUNTS,
        "cell_count": EXPECTED_CALL_COUNT,
        "execution_block_counts": EXECUTION_BLOCK_COUNTS,
        "r2_symbols_sha256": canonical_sha256(list(R2_SYMBOLS)),
        "replacement_provenance_sha256": canonical_sha256(
            fixture["symbol_replacement_provenance"]
        ),
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
                "discovery_components": DISCOVERY_COMPONENT_CALL_COUNT,
                "remaining_main": REMAINING_MAIN_CALL_COUNT,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

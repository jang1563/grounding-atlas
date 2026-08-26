#!/usr/bin/env python3
"""Build the prospective V5 positional-activation bank without model calls.

The bank contains fresh, prompt-unique synthetic worlds and three disjoint
roles.  Every world has the same 56-row behavioral baseline: eight property
retrieval prompts, sixteen codebook lookup prompts, and the complete 2^5
composition factorial.  A fixed regular half-fraction selects eight matched
fact-order pairs per world for later activation intervention.

This module freezes prompts and identifiers only.  The experiment runner must
independently verify tokenizer-dependent single-token and offset contracts
before making any forward call.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name(
    "coherent_readout_v5_positional_activation_bank.json"
)

ANALYSIS_ID = "coherent-readout-v5-positional-activation-v1"
FIXTURE_SCHEMA = "coherent-readout-v5-positional-activation-bank-v1"
WORLD_SCHEMA = "coherent-readout-v5-positional-activation-world-v1"
CELL_SCHEMA = "coherent-readout-v5-positional-activation-cell-v1"
PAIR_SCHEMA = "coherent-readout-v5-positional-activation-pair-v1"
MANIFEST_SCHEMA = "coherent-readout-v5-positional-activation-manifest-v1"
FREEZE_DATE = "2026-08-03"
MODE = "prospective_synthetic_nonbiological_causal_positional_binding"
PURPOSE = (
    "test model-prompt-token-layer-specific causal mediation of a synthetic "
    "fact-order gap using a predeclared target-property activation site"
)
NEUTRAL_SYSTEM_MESSAGE = (
    "Follow the user's instructions. Your entire response must be exactly one "
    "uppercase character from the valid output choices. Do not write any other text."
)

ROLE_WORLD_COUNTS = {"fit": 8, "localization": 8, "holdout": 16}
ROLE_RANGES = {"fit": (1, 8), "localization": (9, 16), "holdout": (17, 32)}
WORLD_COUNT = sum(ROLE_WORLD_COUNTS.values())
PER_WORLD_FAMILY_COUNTS = {
    "property_retrieval": 8,
    "codebook_lookup": 16,
    "composition": 32,
}
FAMILY_COUNTS = {
    family: count * WORLD_COUNT
    for family, count in PER_WORLD_FAMILY_COUNTS.items()
}
EXPECTED_CALL_COUNT = sum(FAMILY_COUNTS.values())

# Property and code alphabets are type-disjoint.  The runner must still verify
# their contextual tokenization under the exact locked chat template.
PROPERTY_PAIRS = (
    ("A", "B"),
    ("C", "D"),
    ("E", "F"),
    ("G", "H"),
    ("I", "J"),
    ("K", "L"),
    ("M", "N"),
    ("O", "P"),
)
CODE_PAIRS = (("Q", "R"), ("S", "T"), ("U", "V"), ("W", "X"))
ALL_SYMBOLS = tuple(
    symbol for pair in (*PROPERTY_PAIRS, *CODE_PAIRS) for symbol in pair
)

FACTOR_NAMES = ("p", "m", "r", "v", "o")
FACTOR_CODING = {
    "p": {"negative": "property_0", "positive": "property_1"},
    "m": {"negative": "identity", "positive": "swapped"},
    "r": {"negative": "property_0_first", "positive": "property_1_first"},
    "v": {"negative": "code_0_first", "positive": "code_1_first"},
    "o": {"negative": "target_first", "positive": "target_second"},
}
FAMILY_FACTOR_NAMES = {
    "property_retrieval": ("p", "v", "o"),
    "codebook_lookup": ("p", "m", "r", "v"),
    "composition": FACTOR_NAMES,
}

FIREWALL = {
    "data_domain": "synthetic_symbolic_only",
    "biology_data_or_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "real_world_semantic_inference": "forbidden",
    "model_family_generalization": "forbidden",
    "maximum_claim": (
        "model-prompt-token-layer-specific selective causal mediation of the "
        "preregistered synthetic positional gap"
    ),
}

TOKENIZER_VALIDATION_CONTRACT = {
    "builder_tokenizer_calls": 0,
    "symbols_are_ascii_uppercase_single_characters": True,
    "property_and_code_symbol_pools_are_disjoint": True,
    "required_runner_checks_before_any_forward_call": [
        "each answer symbol is exactly one contextual continuation token",
        "each composition target_property_span maps to exactly one token",
        "the mapped token text equals target_property",
    ],
    "candidate_symbols": list(ALL_SYMBOLS),
}

FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "freeze_date",
    "mode",
    "purpose",
    "system_message",
    "neutral_system_message",
    "inferential_unit",
    "dependency_unit",
    "role_world_counts",
    "per_world_family_counts",
    "family_counts",
    "expected_call_count",
    "factor_coding",
    "intervention_panel_contract",
    "tokenizer_validation_contract",
    "firewall",
    "worlds",
    "cells",
    "model_calls_made_by_builder",
    "tokenizer_calls_made_by_builder",
    "biological_model_calls",
}
WORLD_KEYS = {
    "schema_version",
    "world_id",
    "world_index",
    "role",
    "role_world_index",
    "instance_key",
    "target_entity",
    "other_entity",
    "symbol_combination_id",
    "property_symbols",
    "code_symbols",
    "codebooks",
    "symbol_validation",
    "intervention_pairs",
}
PAIR_KEYS = {
    "schema_version",
    "pair_id",
    "panel_index",
    "world_id",
    "role",
    "factors",
    "target_property",
    "correct_answer",
    "target_first_cell_id",
    "target_second_cell_id",
}
CELL_KEYS = {
    "schema_version",
    "cell_id",
    "world_id",
    "world_index",
    "role",
    "family",
    "stratum_id",
    "instance_key",
    "target_entity",
    "other_entity",
    "target_property",
    "other_property",
    "mapping_id",
    "rule_order",
    "option_order",
    "fact_order",
    "factors",
    "factor_levels",
    "answer_options",
    "correct_answer",
    "correct_option_position",
    "codebook",
    "rule_lines",
    "fact_lines",
    "prompt_lines",
    "prompt_text",
    "prompt_sha256",
    "target_property_span",
    "target_property_span_basis",
    "semantic_pair_id",
    "mate_cell_id",
    "recipient_selected",
    "intervention_prerequisite",
    "intervention_panel_index",
    "model_calls_made_by_builder",
    "tokenizer_calls_made_by_builder",
    "biological_model_calls",
}

INSTANCE_KEY_PATTERN = re.compile(r"^V5-W[0-9]{3}-K[A-Z2-9]{3}$")
NONCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class V5BankError(ValueError):
    """Raised when the prospective V5 bank contract is violated."""


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


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _role_for_world(world_index: int) -> str:
    for role, (first, last) in ROLE_RANGES.items():
        if first <= world_index <= last:
            return role
    raise V5BankError(f"world index outside frozen ranges: {world_index}")


def _symbol_combinations_for_role(role: str) -> list[tuple[int, int]]:
    all_combinations = list(
        itertools.product(range(len(PROPERTY_PAIRS)), range(len(CODE_PAIRS)))
    )
    if role == "fit":
        return [
            (property_index, property_index % len(CODE_PAIRS))
            for property_index in range(len(PROPERTY_PAIRS))
        ]
    if role == "localization":
        return [
            (property_index, (property_index + 1) % len(CODE_PAIRS))
            for property_index in range(len(PROPERTY_PAIRS))
        ]
    if role == "holdout":
        reserved = set(_symbol_combinations_for_role("fit")) | set(
            _symbol_combinations_for_role("localization")
        )
        return [pair for pair in all_combinations if pair not in reserved]
    raise V5BankError(f"unknown role: {role}")


def _codebook(
    property_symbols: Sequence[str], code_symbols: Sequence[str], mapping_id: str
) -> dict[str, str]:
    if mapping_id == "identity":
        return {
            property_symbols[0]: code_symbols[0],
            property_symbols[1]: code_symbols[1],
        }
    if mapping_id == "swapped":
        return {
            property_symbols[0]: code_symbols[1],
            property_symbols[1]: code_symbols[0],
        }
    raise V5BankError(f"unknown mapping: {mapping_id}")


def _instance_key(world_index: int) -> str:
    digest = hashlib.sha256(
        f"{ANALYSIS_ID}:role-neutral-instance:{world_index}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    nonce = "".join(
        NONCE_ALPHABET[(value // (len(NONCE_ALPHABET) ** power)) % len(NONCE_ALPHABET)]
        for power in (2, 1, 0)
    )
    return f"V5-W{world_index:03d}-K{nonce}"


def _worlds_without_pairs() -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    world_index = 0
    for role in ("fit", "localization", "holdout"):
        combinations = _symbol_combinations_for_role(role)
        if len(combinations) != ROLE_WORLD_COUNTS[role]:
            raise V5BankError("role symbol-combination allocation changed")
        for role_world_index, (property_index, code_index) in enumerate(
            combinations, start=1
        ):
            world_index += 1
            properties = PROPERTY_PAIRS[property_index]
            codes = CODE_PAIRS[code_index]
            world_id = f"activation_world_{world_index:03d}"
            all_world_symbols = (*properties, *codes)
            worlds.append(
                {
                    "schema_version": WORLD_SCHEMA,
                    "world_id": world_id,
                    "world_index": world_index,
                    "role": role,
                    "role_world_index": role_world_index,
                    "instance_key": _instance_key(world_index),
                    "target_entity": f"target_referent_{world_index:03d}",
                    "other_entity": f"other_referent_{world_index:03d}",
                    "symbol_combination_id": (
                        f"property_pair_{property_index + 1}:"
                        f"code_pair_{code_index + 1}"
                    ),
                    "property_symbols": {
                        "negative": properties[0],
                        "positive": properties[1],
                    },
                    "code_symbols": {
                        "negative": codes[0],
                        "positive": codes[1],
                    },
                    "codebooks": {
                        mapping: _codebook(properties, codes, mapping)
                        for mapping in ("identity", "swapped")
                    },
                    "symbol_validation": {
                        "symbols": list(all_world_symbols),
                        "all_ascii_uppercase_single_characters": all(
                            len(symbol) == 1
                            and symbol.isascii()
                            and symbol.isupper()
                            for symbol in all_world_symbols
                        ),
                        "all_symbols_distinct_within_world": (
                            len(set(all_world_symbols)) == 4
                        ),
                        "contextual_single_token_status": (
                            "prospectively_required_runner_validation"
                        ),
                    },
                    "intervention_pairs": [],
                }
            )
    return worlds


def _level_from_sign(factor: str, sign: int | None) -> str | None:
    if sign is None:
        return None
    if sign not in (-1, 1):
        raise V5BankError(f"factor {factor} must be coded -1 or +1")
    polarity = "negative" if sign == -1 else "positive"
    return FACTOR_CODING[factor][polarity]


def _symbol_from_p(world: Mapping[str, Any], p: int) -> str:
    polarity = "negative" if p == -1 else "positive"
    return str(world["property_symbols"][polarity])


def _other_property(world: Mapping[str, Any], p: int) -> str:
    return _symbol_from_p(world, -p)


def _mapping_from_m(m: int) -> str:
    return str(_level_from_sign("m", m))


def _rule_lines(world: Mapping[str, Any], m: int, r: int) -> list[str]:
    properties = [
        str(world["property_symbols"]["negative"]),
        str(world["property_symbols"]["positive"]),
    ]
    if r == 1:
        properties.reverse()
    codebook = world["codebooks"][_mapping_from_m(m)]
    return [f"RULE: {symbol} maps to {codebook[symbol]}." for symbol in properties]


def _answer_options(
    world: Mapping[str, Any], family: str, v: int
) -> list[str]:
    source = (
        world["property_symbols"]
        if family == "property_retrieval"
        else world["code_symbols"]
    )
    options = [str(source["negative"]), str(source["positive"])]
    if v == 1:
        options.reverse()
    return options


def _fact_lines(world: Mapping[str, Any], p: int, o: int) -> list[str]:
    target = (
        f"TARGET FACT: {world['target_entity']} has property "
        f"{_symbol_from_p(world, p)}."
    )
    other = (
        f"OTHER FACT: {world['other_entity']} has property "
        f"{_other_property(world, p)}."
    )
    return [target, other] if o == -1 else [other, target]


def _prompt(
    *,
    world: Mapping[str, Any],
    family: str,
    p: int,
    m: int | None,
    r: int | None,
    v: int,
    o: int | None,
) -> tuple[list[str], list[str], list[str], list[str], list[int] | None]:
    options = _answer_options(world, family, v)
    option_text = ", ".join(options)
    instance_line = f"INSTANCE KEY: {world['instance_key']}."
    rules: list[str] = []
    facts: list[str] = []

    if family == "composition":
        if m is None or r is None or o is None:
            raise V5BankError("composition factors are incomplete")
        rules = _rule_lines(world, m, r)
        facts = _fact_lines(world, p, o)
        lines = [
            instance_line,
            "CODEBOOK RULES",
            *rules,
            f"VALID OUTPUTS (display order carries no meaning): {option_text}.",
            (
                "TASK: Read the exact property in TARGET FACT; find the rule "
                "whose left-side property matches it; return only that rule's "
                "right-side code."
            ),
            "LABELED FACTS",
            *facts,
        ]
        prompt_text = "\n".join(lines)
        target_line = next(line for line in facts if line.startswith("TARGET FACT:"))
        line_start = prompt_text.index(target_line)
        property_text = _symbol_from_p(world, p)
        marker = " has property "
        property_start = line_start + target_line.index(marker) + len(marker)
        span = [property_start, property_start + len(property_text)]
        return lines, rules, facts, options, span

    if family == "property_retrieval":
        if o is None or m is not None or r is not None:
            raise V5BankError("retrieval factors changed")
        facts = _fact_lines(world, p, o)
        lines = [
            instance_line,
            f"VALID OUTPUT PROPERTIES (display order carries no meaning): {option_text}.",
            "TASK: Copy the one-character property stated in TARGET FACT.",
            "LABELED FACTS",
            *facts,
        ]
        return lines, [], facts, options, None

    if family == "codebook_lookup":
        if m is None or r is None or o is not None:
            raise V5BankError("lookup factors changed")
        rules = _rule_lines(world, m, r)
        lines = [
            instance_line,
            "CODEBOOK RULES",
            *rules,
            f"VALID OUTPUTS (display order carries no meaning): {option_text}.",
            f"GIVEN PROPERTY: {_symbol_from_p(world, p)}.",
            (
                "TASK: Find the rule whose left-side property exactly matches "
                "GIVEN PROPERTY; return only that rule's right-side code."
            ),
        ]
        return lines, rules, [], options, None

    raise V5BankError(f"unknown family: {family}")


def _factor_dict(
    *, p: int, m: int | None, r: int | None, v: int, o: int | None
) -> dict[str, int | None]:
    return {"p": p, "m": m, "r": r, "v": v, "o": o}


def _factor_level_dict(
    factors: Mapping[str, int | None]
) -> dict[str, str | None]:
    return {factor: _level_from_sign(factor, factors[factor]) for factor in FACTOR_NAMES}


def _factor_tag(name: str, value: int | None) -> str:
    return f"{name}{'n' if value == -1 else 'p' if value == 1 else 'x'}"


def _cell_id(
    world_id: str,
    family: str,
    *,
    p: int,
    m: int | None,
    r: int | None,
    v: int,
    o: int | None,
) -> str:
    factors = _factor_dict(p=p, m=m, r=r, v=v, o=o)
    tags = ":".join(_factor_tag(name, factors[name]) for name in FACTOR_NAMES)
    return f"activation-v5:{world_id}:{family}:{tags}"


def _semantic_pair_id(
    world_id: str,
    family: str,
    *,
    p: int,
    m: int | None,
    r: int | None,
    v: int,
) -> str:
    factors = {"p": p, "m": m, "r": r, "v": v}
    tags = ":".join(_factor_tag(name, factors[name]) for name in ("p", "m", "r", "v"))
    return f"activation-v5-pair:{world_id}:{family}:{tags}"


def _cell(
    *,
    world: Mapping[str, Any],
    family: str,
    p: int,
    m: int | None,
    r: int | None,
    v: int,
    o: int | None,
) -> dict[str, Any]:
    factors = _factor_dict(p=p, m=m, r=r, v=v, o=o)
    levels = _factor_level_dict(factors)
    lines, rules, facts, options, span = _prompt(
        world=world, family=family, p=p, m=m, r=r, v=v, o=o
    )
    prompt_text = "\n".join(lines)
    target_property = _symbol_from_p(world, p)
    mapping_id = _mapping_from_m(m) if m is not None else None
    codebook = dict(world["codebooks"][mapping_id]) if mapping_id else None
    correct_answer = (
        target_property if family == "property_retrieval" else codebook[target_property]
    )
    fact_order = _level_from_sign("o", o)
    cell_id = _cell_id(
        str(world["world_id"]), family, p=p, m=m, r=r, v=v, o=o
    )
    pair_id = _semantic_pair_id(
        str(world["world_id"]), family, p=p, m=m, r=r, v=v
    )
    mate_cell_id = (
        _cell_id(
            str(world["world_id"]), family, p=p, m=m, r=r, v=v, o=-o
        )
        if o is not None
        else None
    )
    recipient_selected = bool(
        family == "composition" and v == p * int(m) * int(r)
    )
    intervention_prerequisite = bool(
        family == "property_retrieval"
        or (
            family == "codebook_lookup"
            and v == p * int(m) * int(r)
        )
    )
    panel_index = None
    if recipient_selected:
        panel_signs = list(itertools.product((-1, 1), repeat=3))
        panel_index = panel_signs.index((p, int(m), int(r))) + 1
    stratum = ":".join(
        [family, *(_factor_tag(name, factors[name]) for name in FACTOR_NAMES)]
    )
    return {
        "schema_version": CELL_SCHEMA,
        "cell_id": cell_id,
        "world_id": world["world_id"],
        "world_index": world["world_index"],
        "role": world["role"],
        "family": family,
        "stratum_id": stratum,
        "instance_key": world["instance_key"],
        "target_entity": world["target_entity"] if family != "codebook_lookup" else None,
        "other_entity": world["other_entity"] if family != "codebook_lookup" else None,
        "target_property": target_property,
        "other_property": _other_property(world, p) if family != "codebook_lookup" else None,
        "mapping_id": mapping_id,
        "rule_order": levels["r"],
        "option_order": levels["v"],
        "fact_order": fact_order,
        "factors": factors,
        "factor_levels": levels,
        "answer_options": options,
        "correct_answer": correct_answer,
        "correct_option_position": "first" if options[0] == correct_answer else "last",
        "codebook": codebook,
        "rule_lines": rules,
        "fact_lines": facts,
        "prompt_lines": lines,
        "prompt_text": prompt_text,
        "prompt_sha256": text_sha256(prompt_text),
        "target_property_span": span,
        "target_property_span_basis": (
            "zero_based_half_open_prompt_text_characters" if span is not None else None
        ),
        "semantic_pair_id": pair_id,
        "mate_cell_id": mate_cell_id,
        "recipient_selected": recipient_selected,
        "intervention_prerequisite": intervention_prerequisite,
        "intervention_panel_index": panel_index,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }


def _build_cells(worlds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for world in worlds:
        for p, v, o in itertools.product((-1, 1), repeat=3):
            cells.append(
                _cell(
                    world=world,
                    family="property_retrieval",
                    p=p,
                    m=None,
                    r=None,
                    v=v,
                    o=o,
                )
            )
        for p, m, r, v in itertools.product((-1, 1), repeat=4):
            cells.append(
                _cell(
                    world=world,
                    family="codebook_lookup",
                    p=p,
                    m=m,
                    r=r,
                    v=v,
                    o=None,
                )
            )
        for p, m, r, v, o in itertools.product((-1, 1), repeat=5):
            cells.append(
                _cell(
                    world=world,
                    family="composition",
                    p=p,
                    m=m,
                    r=r,
                    v=v,
                    o=o,
                )
            )
    return cells


def _intervention_pairs(
    world: Mapping[str, Any], cells_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for panel_index, (p, m, r) in enumerate(
        itertools.product((-1, 1), repeat=3), start=1
    ):
        v = p * m * r
        first_id = _cell_id(
            str(world["world_id"]),
            "composition",
            p=p,
            m=m,
            r=r,
            v=v,
            o=-1,
        )
        second_id = _cell_id(
            str(world["world_id"]),
            "composition",
            p=p,
            m=m,
            r=r,
            v=v,
            o=1,
        )
        first = cells_by_id[first_id]
        second = cells_by_id[second_id]
        if first["correct_answer"] != second["correct_answer"]:
            raise V5BankError("intervention order pair changes native answer")
        pairs.append(
            {
                "schema_version": PAIR_SCHEMA,
                "pair_id": first["semantic_pair_id"],
                "panel_index": panel_index,
                "world_id": world["world_id"],
                "role": world["role"],
                "factors": {"p": p, "m": m, "r": r, "v": v},
                "target_property": first["target_property"],
                "correct_answer": first["correct_answer"],
                "target_first_cell_id": first_id,
                "target_second_cell_id": second_id,
            }
        )
    return pairs


def build_fixture() -> dict[str, Any]:
    worlds = _worlds_without_pairs()
    cells = _build_cells(worlds)
    cells_by_id = {cell["cell_id"]: cell for cell in cells}
    if len(cells_by_id) != len(cells):
        raise V5BankError("cell IDs are duplicated before pair construction")
    for world in worlds:
        world["intervention_pairs"] = _intervention_pairs(world, cells_by_id)
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": MODE,
        "purpose": PURPOSE,
        "system_message": NEUTRAL_SYSTEM_MESSAGE,
        "neutral_system_message": NEUTRAL_SYSTEM_MESSAGE,
        "inferential_unit": "synthetic_world",
        "dependency_unit": "complete_56_prompt_factorial_within_world",
        "role_world_counts": ROLE_WORLD_COUNTS,
        "per_world_family_counts": PER_WORLD_FAMILY_COUNTS,
        "family_counts": FAMILY_COUNTS,
        "expected_call_count": EXPECTED_CALL_COUNT,
        "factor_coding": FACTOR_CODING,
        "intervention_panel_contract": {
            "semantic_pairs_per_world": 8,
            "composition_rows_per_world": 16,
            "base_factors": ["p", "m", "r"],
            "generator": "v=p*m*r",
            "paired_factor": "o",
            "fact_order_levels": {"target_first": -1, "target_second": 1},
            "outcome_filtering_allowed": False,
        },
        "tokenizer_validation_contract": TOKENIZER_VALIDATION_CONTRACT,
        "firewall": FIREWALL,
        "worlds": worlds,
        "cells": cells,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    validate_fixture(fixture)
    return fixture


def _expected_cell_from_observed(
    cell: Mapping[str, Any], worlds_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    world = worlds_by_id.get(str(cell.get("world_id")))
    if world is None:
        raise V5BankError("cell references unknown world")
    factors = cell.get("factors")
    if not isinstance(factors, dict) or set(factors) != set(FACTOR_NAMES):
        raise V5BankError("cell factor schema changed")
    return _cell(
        world=world,
        family=str(cell.get("family")),
        p=factors["p"],
        m=factors["m"],
        r=factors["r"],
        v=factors["v"],
        o=factors["o"],
    )


def validate_fixture(fixture: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(fixture) != FIXTURE_KEYS:
        raise V5BankError("fixture top-level schema changed")
    expected_header = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": MODE,
        "purpose": PURPOSE,
        "system_message": NEUTRAL_SYSTEM_MESSAGE,
        "neutral_system_message": NEUTRAL_SYSTEM_MESSAGE,
        "inferential_unit": "synthetic_world",
        "dependency_unit": "complete_56_prompt_factorial_within_world",
        "role_world_counts": ROLE_WORLD_COUNTS,
        "per_world_family_counts": PER_WORLD_FAMILY_COUNTS,
        "family_counts": FAMILY_COUNTS,
        "expected_call_count": EXPECTED_CALL_COUNT,
        "factor_coding": FACTOR_CODING,
        "intervention_panel_contract": {
            "semantic_pairs_per_world": 8,
            "composition_rows_per_world": 16,
            "base_factors": ["p", "m", "r"],
            "generator": "v=p*m*r",
            "paired_factor": "o",
            "fact_order_levels": {"target_first": -1, "target_second": 1},
            "outcome_filtering_allowed": False,
        },
        "tokenizer_validation_contract": TOKENIZER_VALIDATION_CONTRACT,
        "firewall": FIREWALL,
        "model_calls_made_by_builder": 0,
        "tokenizer_calls_made_by_builder": 0,
        "biological_model_calls": 0,
    }
    for key, expected in expected_header.items():
        if fixture[key] != expected:
            raise V5BankError(f"fixture header changed: {key}")

    worlds = fixture["worlds"]
    cells = fixture["cells"]
    if not isinstance(worlds, list) or len(worlds) != WORLD_COUNT:
        raise V5BankError("world registry changed")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALL_COUNT:
        raise V5BankError("cell registry changed")

    expected_worlds = _worlds_without_pairs()
    worlds_by_id: dict[str, Mapping[str, Any]] = {}
    instance_keys: set[str] = set()
    for observed, expected_base in zip(worlds, expected_worlds, strict=True):
        if not isinstance(observed, dict) or set(observed) != WORLD_KEYS:
            raise V5BankError("world schema changed")
        if not isinstance(observed["intervention_pairs"], list):
            raise V5BankError("world intervention pairs changed")
        expected_base["intervention_pairs"] = observed["intervention_pairs"]
        if observed != expected_base:
            raise V5BankError(f"world identity changed: {observed.get('world_id')}")
        if not INSTANCE_KEY_PATTERN.fullmatch(observed["instance_key"]):
            raise V5BankError("instance-key syntax changed")
        if observed["world_id"] in worlds_by_id:
            raise V5BankError("world IDs are duplicated")
        if observed["instance_key"] in instance_keys:
            raise V5BankError("world instance keys are duplicated")
        worlds_by_id[observed["world_id"]] = observed
        instance_keys.add(observed["instance_key"])

    if Counter(world["role"] for world in worlds) != ROLE_WORLD_COUNTS:
        raise V5BankError("world-role allocation changed")

    cell_ids: set[str] = set()
    prompts: set[str] = set()
    family_counts: Counter[str] = Counter()
    world_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stratum_sets: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_KEYS:
            raise V5BankError("cell schema changed")
        if cell != _expected_cell_from_observed(cell, worlds_by_id):
            raise V5BankError(f"cell does not reconstruct: {cell.get('cell_id')}")
        if cell["cell_id"] in cell_ids:
            raise V5BankError("cell IDs are duplicated")
        if cell["prompt_text"] in prompts:
            raise V5BankError("prompt texts are duplicated")
        cell_ids.add(cell["cell_id"])
        prompts.add(cell["prompt_text"])
        family_counts[cell["family"]] += 1
        world_family_counts[cell["world_id"]][cell["family"]] += 1
        stratum_sets[cell["family"]].add(cell["stratum_id"])
        if f"INSTANCE KEY: {cell['instance_key']}." not in cell["prompt_text"]:
            raise V5BankError("prompt omits its world instance key")
        span = cell["target_property_span"]
        if cell["family"] == "composition":
            if (
                not isinstance(span, list)
                or len(span) != 2
                or not all(isinstance(value, int) for value in span)
                or cell["prompt_text"][span[0] : span[1]] != cell["target_property"]
            ):
                raise V5BankError("composition target-property span changed")
            codebook_position = cell["prompt_text"].index("CODEBOOK RULES")
            options_position = cell["prompt_text"].index("VALID OUTPUTS")
            task_position = cell["prompt_text"].index("TASK:")
            facts_position = cell["prompt_text"].index("LABELED FACTS")
            target_position = span[0]
            if not (
                codebook_position < options_position < task_position
                < facts_position < target_position
            ):
                raise V5BankError("causal-prefix prompt order changed")
        else:
            if span is not None or cell["target_property_span_basis"] is not None:
                raise V5BankError("non-composition activation span must be null")
            if cell["family"] == "property_retrieval" and not (
                cell["prompt_text"].index("TASK:")
                < cell["prompt_text"].index("LABELED FACTS")
            ):
                raise V5BankError("retrieval causal-prefix prompt order changed")

    if dict(family_counts) != FAMILY_COUNTS:
        raise V5BankError("family allocation changed")
    if any(dict(counts) != PER_WORLD_FAMILY_COUNTS for counts in world_family_counts.values()):
        raise V5BankError("per-world family allocation changed")
    if len(world_family_counts) != WORLD_COUNT:
        raise V5BankError("a world has no cells")
    if {family: len(values) for family, values in stratum_sets.items()} != {
        "property_retrieval": 8,
        "codebook_lookup": 16,
        "composition": 32,
    }:
        raise V5BankError("factorial stratum coverage changed")

    cells_by_id = {cell["cell_id"]: cell for cell in cells}
    selected_ids = {
        cell["cell_id"] for cell in cells if cell["recipient_selected"]
    }
    declared_ids: set[str] = set()
    for world in worlds:
        expected_pairs = _intervention_pairs(world, cells_by_id)
        if world["intervention_pairs"] != expected_pairs:
            raise V5BankError("world intervention-pair registry changed")
        if len(expected_pairs) != 8:
            raise V5BankError("intervention pair count changed")
        for pair in expected_pairs:
            if set(pair) != PAIR_KEYS:
                raise V5BankError("intervention pair schema changed")
            first = cells_by_id[pair["target_first_cell_id"]]
            second = cells_by_id[pair["target_second_cell_id"]]
            if (
                first["mate_cell_id"] != second["cell_id"]
                or second["mate_cell_id"] != first["cell_id"]
                or first["semantic_pair_id"] != pair["pair_id"]
                or second["semantic_pair_id"] != pair["pair_id"]
                or first["correct_answer"] != second["correct_answer"]
                or first["factors"]["o"] != -1
                or second["factors"]["o"] != 1
            ):
                raise V5BankError("intervention pair relation changed")
            declared_ids.update((first["cell_id"], second["cell_id"]))
    if selected_ids != declared_ids or len(selected_ids) != WORLD_COUNT * 16:
        raise V5BankError("recipient-panel declaration changed")
    return fixture


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
        raise V5BankError(f"refusing to overwrite differing frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise V5BankError(f"stale atomic temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_fixture(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    fixture = build_fixture()
    _atomic_write(output, _artifact_bytes(fixture))
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "fixture_path": relative_path(output),
        "fixture_file_sha256": file_sha256(output),
        "fixture_canonical_sha256": canonical_sha256(fixture),
        "builder_path": relative_path(Path(__file__)),
        "builder_file_sha256": file_sha256(Path(__file__)),
        "world_count": WORLD_COUNT,
        "role_world_counts": ROLE_WORLD_COUNTS,
        "family_counts": FAMILY_COUNTS,
        "cell_count": len(fixture["cells"]),
        "intervention_pair_count": WORLD_COUNT * 8,
        "model_calls": 0,
        "tokenizer_calls": 0,
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
                "fixture": relative_path(output),
                "manifest": relative_path(manifest),
                "worlds": WORLD_COUNT,
                "cells": EXPECTED_CALL_COUNT,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Build the frozen v3 cross-codebook anti-copy syntax bank.

The fixture is entirely synthetic and prospective.  It enumerates a complete
five-factor prompt design in 56 uniformly named symbolic worlds and freezes a
balanced eight-cell recipient fraction in every world.  Building the bank makes
zero model calls and uses earlier banks only to prove identifier disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ID = "coherent-readout-v3-content-routing-bank-v1"
FIXTURE_SCHEMA = "coherent-readout-v3-content-routing-bank-v1"
CELL_SCHEMA = "coherent-readout-v3-content-routing-cell-v1"
MANIFEST_SCHEMA = "coherent-readout-v3-content-routing-bank-manifest-v1"
RECIPIENT_SCHEMA = "coherent-readout-v3-recipient-fraction-v1"
FREEZE_DATE = "2026-08-02"
PURPOSE = "prospective_nonbiological_cross_codebook_anti_copy_factorial"
CONTROLLED_NAME_PATTERN = re.compile(r"^symbol_[ab]_[0-9]{3}$")

DISCOVERY_FIXTURE = Path(__file__).with_name("coherent_readout_v2_syntax_bank.json")
DISCOVERY_FIXTURE_SHA256 = (
    "d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8"
)
PRIOR_HOLDOUT_FIXTURE = Path(__file__).with_name(
    "coherent_readout_v2_causal_binding_bank.json"
)
PRIOR_HOLDOUT_FIXTURE_SHA256 = (
    "2c40ba0c796202059056aec4535fd7656eab2b446d8895816bbae2034ebcbcdb"
)

DEFAULT_OUT = Path(__file__).with_name("coherent_readout_v3_content_routing_bank.json")
DEFAULT_MANIFEST = Path(__file__).with_name(
    "coherent_readout_v3_content_routing_bank.manifest.json"
)

ROLE_RANGES = {
    "direction_fit": [1, 16],
    "localization": [17, 24],
    "holdout": [25, 56],
}
ROLE_WORLD_COUNTS = {"direction_fit": 16, "localization": 8, "holdout": 32}
ROLE_PROMPT_COUNTS = {"direction_fit": 512, "localization": 256, "holdout": 1024}
ROLE_RECIPIENT_COUNTS = {"direction_fit": 128, "localization": 64, "holdout": 256}

FACTOR_LEVELS = {
    "queried_entity": ["a", "b"],
    "queried_property": ["P", "Q"],
    "distractor_property": ["P", "Q"],
    "codebook": ["identity", "swapped"],
    "fact_line_order": ["query_first", "query_second"],
}
CODEBOOKS = {
    "identity": {"P": "X", "Q": "Y"},
    "swapped": {"P": "Y", "Q": "X"},
}
OPPOSITE_PROPERTY = {"P": "Q", "Q": "P"}
OPPOSITE_CODEBOOK = {"identity": "swapped", "swapped": "identity"}
RECIPIENT_CONTRAST_COLUMNS = [
    "query_content",
    "codebook",
    "query_entity",
    "distractor",
    "fact_order",
    "native_answer",
]

FIREWALL = {
    "scope": PURPOSE,
    "data_domain": "synthetic_symbolic_only",
    "biology_data_or_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "activation_gap_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "real_world_semantic_inference": "forbidden",
    "model_family_generalization": "forbidden",
}

OUTCOME_EXPOSURE = {
    "prior_banks_used_for_disjointness_only": True,
    "prior_bank_model_outputs_used": False,
    "new_bank_native_model_outputs_observed": False,
    "new_bank_intervention_outputs_observed": False,
    "model_outputs_used_to_select_worlds": False,
    "model_outputs_used_to_select_recipient_cells": False,
    "prompt_outcomes_used_to_filter_cells": False,
}

REFERENCE_CONTRACT = {
    "self_cell_id": {
        "transformation": "identity",
        "native_answer_relation": "same",
    },
    "anti_copy_donor_cell_id": {
        "transformation": "flip_queried_property_and_codebook",
        "fixed_factors": [
            "world",
            "queried_entity",
            "distractor_property",
            "fact_line_order",
        ],
        "native_answer_relation": "same",
    },
    "text_counterfactual_cell_id": {
        "transformation": "flip_queried_property_only",
        "native_answer_relation": "opposite",
    },
    "same_content_opposite_codebook_donor_cell_id": {
        "transformation": "flip_codebook_only",
        "native_answer_relation": "opposite",
    },
    "distractor_flip_cell_id": {
        "transformation": "flip_distractor_property_only",
        "native_answer_relation": "same",
    },
    "query_entity_flip_cell_id": {
        "transformation": "flip_queried_entity_only",
        "native_answer_relation": "same",
    },
    "fact_order_flip_cell_id": {
        "transformation": "flip_fact_line_order_only",
        "native_answer_relation": "same",
    },
    "all_transformations_are_involutions": True,
    "itemwise_outcome_filtering_allowed": False,
}


class ContentRoutingBankError(ValueError):
    """Raised when the frozen cross-codebook bank contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_reference(
    path: Path,
    expected_sha256: str,
    *,
    reference_id: str,
    expected_pairs: int,
    expected_items: int,
) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ContentRoutingBankError(
            f"prior reference differs from frozen hash: {reference_id}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContentRoutingBankError(
            f"prior reference is not valid JSON: {reference_id}"
        ) from error
    pairs = value.get("pair_registry")
    items = value.get("items")
    if not isinstance(pairs, list) or len(pairs) != expected_pairs:
        raise ContentRoutingBankError(f"prior pair count changed: {reference_id}")
    if not isinstance(items, list) or len(items) != expected_items:
        raise ContentRoutingBankError(f"prior item count changed: {reference_id}")
    try:
        pair_ids = [row["pair_id"] for row in pairs]
        class_words = sorted(
            {
                word
                for row in pairs
                for word in (row["positive_class"], row["negative_class"])
            }
        )
        item_ids = sorted(row["item_id"] for row in items)
    except (KeyError, TypeError) as error:
        raise ContentRoutingBankError(
            f"prior reference identifiers changed: {reference_id}"
        ) from error
    if not all(
        isinstance(identifier, str)
        for identifier in (*pair_ids, *class_words, *item_ids)
    ):
        raise ContentRoutingBankError(
            f"prior reference identifiers changed: {reference_id}"
        )
    return {
        "reference_id": reference_id,
        "path": relative_path(path),
        "file_sha256": expected_sha256,
        "canonical_sha256": canonical_sha256(value),
        "pair_ids": pair_ids,
        "class_words": class_words,
        "item_ids": item_ids,
        "pair_count": len(pairs),
        "item_count": len(items),
        "use": "disjointness_check_only_no_behavior_or_outcome_use",
    }


def _prior_references() -> list[dict[str, Any]]:
    return [
        _load_reference(
            DISCOVERY_FIXTURE,
            DISCOVERY_FIXTURE_SHA256,
            reference_id="v2_syntax_discovery",
            expected_pairs=8,
            expected_items=16,
        ),
        _load_reference(
            PRIOR_HOLDOUT_FIXTURE,
            PRIOR_HOLDOUT_FIXTURE_SHA256,
            reference_id="v2_causal_binding_holdout",
            expected_pairs=48,
            expected_items=96,
        ),
    ]


def _role_for_world(world_index: int) -> str:
    for role, (first, last) in ROLE_RANGES.items():
        if first <= world_index <= last:
            return role
    raise ContentRoutingBankError(f"world index outside frozen ranges: {world_index}")


def _world_registry() -> list[dict[str, Any]]:
    return [
        {
            "world_index": index,
            "world_id": f"symbolic_world_{index:03d}",
            "role": _role_for_world(index),
            "entity_a": f"symbol_a_{index:03d}",
            "entity_b": f"symbol_b_{index:03d}",
        }
        for index in range(1, 57)
    ]


def _factor_sign(level: str, negative_level: str, positive_level: str) -> int:
    if level == negative_level:
        return -1
    if level == positive_level:
        return 1
    raise ContentRoutingBankError(f"unknown factor level: {level}")


def _recipient_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_index, (e_sign, p_sign, m_sign) in enumerate(
        itertools.product((-1, 1), repeat=3), start=1
    ):
        d_sign = e_sign * p_sign
        order_sign = e_sign * m_sign
        native_answer_sign = p_sign * m_sign
        levels = {
            "queried_entity": "a" if e_sign == -1 else "b",
            "queried_property": "P" if p_sign == -1 else "Q",
            "distractor_property": "P" if d_sign == -1 else "Q",
            "codebook": "identity" if m_sign == -1 else "swapped",
            "fact_line_order": (
                "query_first" if order_sign == -1 else "query_second"
            ),
        }
        runs.append(
            {
                "run_index": run_index,
                "base_signs": {"e": e_sign, "p": p_sign, "m": m_sign},
                "derived_signs": {
                    "d": d_sign,
                    "order": order_sign,
                    "native_answer": native_answer_sign,
                },
                "levels": levels,
                "native_answer": CODEBOOKS[levels["codebook"]][
                    levels["queried_property"]
                ],
            }
        )
    return runs


def _gram_matrix(rows: list[list[int]]) -> list[list[int]]:
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ContentRoutingBankError("contrast matrix is empty or ragged")
    width = len(rows[0])
    return [
        [sum(row[left] * row[right] for row in rows) for right in range(width)]
        for left in range(width)
    ]


def _recipient_contrast_rows(runs: list[dict[str, Any]]) -> list[list[int]]:
    return [
        [
            run["base_signs"]["p"],
            run["base_signs"]["m"],
            run["base_signs"]["e"],
            run["derived_signs"]["d"],
            run["derived_signs"]["order"],
            run["derived_signs"]["native_answer"],
        ]
        for run in runs
    ]


def _recipient_run_lookup() -> dict[tuple[str, str, str, str, str], int]:
    return {
        (
            run["levels"]["queried_entity"],
            run["levels"]["queried_property"],
            run["levels"]["distractor_property"],
            run["levels"]["codebook"],
            run["levels"]["fact_line_order"],
        ): run["run_index"]
        for run in _recipient_runs()
    }


def _cell_id(
    world_index: int,
    queried_entity: str,
    queried_property: str,
    distractor_property: str,
    codebook: str,
    fact_line_order: str,
) -> str:
    return (
        f"content-routing-v3:w{world_index:03d}:e-{queried_entity}:"
        f"q-{queried_property.lower()}:d-{distractor_property.lower()}:"
        f"c-{codebook}:o-{fact_line_order.replace('_', '-')}"
    )


def _prompt_parts(
    *,
    queried_entity: str,
    distractor_entity: str,
    queried_property: str,
    distractor_property: str,
    codebook: str,
    fact_line_order: str,
) -> tuple[list[str], str]:
    mapping = CODEBOOKS[codebook]
    codebook_line = f"Codebook: P maps to {mapping['P']}; Q maps to {mapping['Q']}."
    query_fact = f"Fact: {queried_entity} has property {queried_property}."
    distractor_fact = f"Fact: {distractor_entity} has property {distractor_property}."
    fact_lines = (
        [query_fact, distractor_fact]
        if fact_line_order == "query_first"
        else [distractor_fact, query_fact]
    )
    question = (
        "Question: Which code does the codebook assign to the property of "
        f"{queried_entity}? Answer with X or Y."
    )
    return fact_lines, "\n".join([codebook_line, *fact_lines, question])


def _build_cells() -> list[dict[str, Any]]:
    recipient_lookup = _recipient_run_lookup()
    cells: list[dict[str, Any]] = []
    for world in _world_registry():
        world_index = world["world_index"]
        entities = {"a": world["entity_a"], "b": world["entity_b"]}
        for (
            queried_entity_slot,
            queried_property,
            distractor_property,
            codebook,
            fact_line_order,
        ) in itertools.product(
            FACTOR_LEVELS["queried_entity"],
            FACTOR_LEVELS["queried_property"],
            FACTOR_LEVELS["distractor_property"],
            FACTOR_LEVELS["codebook"],
            FACTOR_LEVELS["fact_line_order"],
        ):
            distractor_entity_slot = "b" if queried_entity_slot == "a" else "a"
            queried_entity = entities[queried_entity_slot]
            distractor_entity = entities[distractor_entity_slot]
            factor_tuple = (
                queried_entity_slot,
                queried_property,
                distractor_property,
                codebook,
                fact_line_order,
            )
            recipient_run_index = recipient_lookup.get(factor_tuple)
            fact_lines, prompt_text = _prompt_parts(
                queried_entity=queried_entity,
                distractor_entity=distractor_entity,
                queried_property=queried_property,
                distractor_property=distractor_property,
                codebook=codebook,
                fact_line_order=fact_line_order,
            )
            cell_id = _cell_id(world_index, *factor_tuple)
            cells.append(
                {
                    "schema_version": CELL_SCHEMA,
                    "cell_id": cell_id,
                    "world_index": world_index,
                    "world_id": world["world_id"],
                    "role": world["role"],
                    "queried_entity_slot": queried_entity_slot,
                    "queried_entity": queried_entity,
                    "distractor_entity_slot": distractor_entity_slot,
                    "distractor_entity": distractor_entity,
                    "queried_property": queried_property,
                    "distractor_property": distractor_property,
                    "codebook_id": codebook,
                    "fact_line_order": fact_line_order,
                    "fact_lines": fact_lines,
                    "prompt_text": prompt_text,
                    "prompt_sha256": text_sha256(prompt_text),
                    "native_answer": CODEBOOKS[codebook][queried_property],
                    "recipient_selected": recipient_run_index is not None,
                    "recipient_run_index": recipient_run_index,
                    "self_cell_id": cell_id,
                    "anti_copy_donor_cell_id": _cell_id(
                        world_index,
                        queried_entity_slot,
                        OPPOSITE_PROPERTY[queried_property],
                        distractor_property,
                        OPPOSITE_CODEBOOK[codebook],
                        fact_line_order,
                    ),
                    "text_counterfactual_cell_id": _cell_id(
                        world_index,
                        queried_entity_slot,
                        OPPOSITE_PROPERTY[queried_property],
                        distractor_property,
                        codebook,
                        fact_line_order,
                    ),
                    "same_content_opposite_codebook_donor_cell_id": _cell_id(
                        world_index,
                        queried_entity_slot,
                        queried_property,
                        distractor_property,
                        OPPOSITE_CODEBOOK[codebook],
                        fact_line_order,
                    ),
                    "distractor_flip_cell_id": _cell_id(
                        world_index,
                        queried_entity_slot,
                        queried_property,
                        OPPOSITE_PROPERTY[distractor_property],
                        codebook,
                        fact_line_order,
                    ),
                    "query_entity_flip_cell_id": _cell_id(
                        world_index,
                        distractor_entity_slot,
                        queried_property,
                        distractor_property,
                        codebook,
                        fact_line_order,
                    ),
                    "fact_order_flip_cell_id": _cell_id(
                        world_index,
                        queried_entity_slot,
                        queried_property,
                        distractor_property,
                        codebook,
                        (
                            "query_second"
                            if fact_line_order == "query_first"
                            else "query_first"
                        ),
                    ),
                }
            )
    return cells


def _recipient_fractional_factorial() -> dict[str, Any]:
    runs = _recipient_runs()
    selected_by_world = []
    for world in _world_registry():
        selected_by_world.append(
            {
                "world_id": world["world_id"],
                "role": world["role"],
                "cell_ids": [
                    _cell_id(
                        world["world_index"],
                        run["levels"]["queried_entity"],
                        run["levels"]["queried_property"],
                        run["levels"]["distractor_property"],
                        run["levels"]["codebook"],
                        run["levels"]["fact_line_order"],
                    )
                    for run in runs
                ],
            }
        )
    return {
        "schema_version": RECIPIENT_SCHEMA,
        "base_factors": ["e", "p", "m"],
        "factor_symbols": {
            "e": "queried_entity",
            "p": "queried_property",
            "m": "codebook",
            "d": "distractor_property",
            "order": "fact_line_order",
            "native_answer": "codebooks[codebook][queried_property]",
        },
        "sign_convention": {
            "e": {"-1": "a", "+1": "b"},
            "p": {"-1": "P", "+1": "Q"},
            "m": {"-1": "identity", "+1": "swapped"},
            "d": {"-1": "P", "+1": "Q"},
            "order": {"-1": "query_first", "+1": "query_second"},
            "native_answer": {"-1": "Y", "+1": "X"},
        },
        "generators": {
            "d": "e*p",
            "order": "e*m",
            "native_answer": "p*m",
        },
        "orthogonality_contract": {
            "columns": RECIPIENT_CONTRAST_COLUMNS,
            "required_gram_matrix": "8I_6",
            "gram_matrix": _gram_matrix(_recipient_contrast_rows(runs)),
        },
        "runs_per_world": 8,
        "runs": runs,
        "selected_by_world": selected_by_world,
    }


def _validate_disjointness(
    worlds: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    world_ids = {world["world_id"] for world in worlds}
    symbolic_names = {
        name for world in worlds for name in (world["entity_a"], world["entity_b"])
    }
    cell_ids = {cell["cell_id"] for cell in cells}
    for reference in references:
        if world_ids & set(reference["pair_ids"]):
            raise ContentRoutingBankError("new world IDs overlap a prior bank")
        if symbolic_names & set(reference["class_words"]):
            raise ContentRoutingBankError("new symbolic names overlap a prior bank")
        if cell_ids & set(reference["item_ids"]):
            raise ContentRoutingBankError("new cell IDs overlap a prior bank")


def _validate_reference_graph(cells: list[dict[str, Any]]) -> None:
    by_id = {cell["cell_id"]: cell for cell in cells}
    if len(by_id) != len(cells):
        raise ContentRoutingBankError("cell IDs are not unique")
    reference_fields = (
        "self_cell_id",
        "anti_copy_donor_cell_id",
        "text_counterfactual_cell_id",
        "same_content_opposite_codebook_donor_cell_id",
        "distractor_flip_cell_id",
        "query_entity_flip_cell_id",
        "fact_order_flip_cell_id",
    )
    for cell in cells:
        if any(cell[field] not in by_id for field in reference_fields):
            raise ContentRoutingBankError("a frozen cell reference does not resolve")
        self_cell = by_id[cell["self_cell_id"]]
        anti_copy = by_id[cell["anti_copy_donor_cell_id"]]
        counterfactual = by_id[cell["text_counterfactual_cell_id"]]
        opposite_codebook = by_id[
            cell["same_content_opposite_codebook_donor_cell_id"]
        ]
        distractor_flip = by_id[cell["distractor_flip_cell_id"]]
        query_entity_flip = by_id[cell["query_entity_flip_cell_id"]]
        fact_order_flip = by_id[cell["fact_order_flip_cell_id"]]
        if self_cell is not cell:
            raise ContentRoutingBankError("self reference is not identity")
        invariant_fields = (
            "world_id",
            "queried_entity_slot",
            "distractor_property",
            "fact_line_order",
        )
        if any(anti_copy[field] != cell[field] for field in invariant_fields):
            raise ContentRoutingBankError("anti-copy donor changed a fixed factor")
        if (
            anti_copy["queried_property"]
            != OPPOSITE_PROPERTY[cell["queried_property"]]
            or anti_copy["codebook_id"] != OPPOSITE_CODEBOOK[cell["codebook_id"]]
            or anti_copy["native_answer"] != cell["native_answer"]
            or anti_copy["anti_copy_donor_cell_id"] != cell["cell_id"]
        ):
            raise ContentRoutingBankError("anti-copy donor relation is invalid")
        counterfactual_invariants = (
            "world_id",
            "queried_entity_slot",
            "distractor_property",
            "codebook_id",
            "fact_line_order",
        )
        if any(
            counterfactual[field] != cell[field]
            for field in counterfactual_invariants
        ) or (
            counterfactual["queried_property"]
            != OPPOSITE_PROPERTY[cell["queried_property"]]
            or counterfactual["native_answer"] == cell["native_answer"]
            or counterfactual["text_counterfactual_cell_id"] != cell["cell_id"]
        ):
            raise ContentRoutingBankError("text counterfactual relation is invalid")
        codebook_invariants = (
            "world_id",
            "queried_entity_slot",
            "queried_property",
            "distractor_property",
            "fact_line_order",
            "fact_lines",
        )
        if any(
            opposite_codebook[field] != cell[field] for field in codebook_invariants
        ) or (
            opposite_codebook["codebook_id"]
            != OPPOSITE_CODEBOOK[cell["codebook_id"]]
            or opposite_codebook["native_answer"] == cell["native_answer"]
            or opposite_codebook[
                "same_content_opposite_codebook_donor_cell_id"
            ]
            != cell["cell_id"]
        ):
            raise ContentRoutingBankError("opposite-codebook relation is invalid")
        if (
            counterfactual["same_content_opposite_codebook_donor_cell_id"]
            != anti_copy["cell_id"]
            or opposite_codebook["text_counterfactual_cell_id"]
            != anti_copy["cell_id"]
        ):
            raise ContentRoutingBankError("reference transformations do not commute")
        nuisance_fixed_fields = (
            "world_id",
            "queried_entity_slot",
            "queried_property",
            "codebook_id",
            "fact_line_order",
        )
        if any(
            distractor_flip[field] != cell[field] for field in nuisance_fixed_fields
        ) or (
            distractor_flip["distractor_property"]
            != OPPOSITE_PROPERTY[cell["distractor_property"]]
            or distractor_flip["native_answer"] != cell["native_answer"]
            or distractor_flip["distractor_flip_cell_id"] != cell["cell_id"]
        ):
            raise ContentRoutingBankError("distractor-only nuisance relation is invalid")
        entity_fixed_fields = (
            "world_id",
            "queried_property",
            "distractor_property",
            "codebook_id",
            "fact_line_order",
        )
        if any(
            query_entity_flip[field] != cell[field] for field in entity_fixed_fields
        ) or (
            query_entity_flip["queried_entity_slot"]
            == cell["queried_entity_slot"]
            or query_entity_flip["native_answer"] != cell["native_answer"]
            or query_entity_flip["query_entity_flip_cell_id"] != cell["cell_id"]
        ):
            raise ContentRoutingBankError("query-entity nuisance relation is invalid")
        order_fixed_fields = (
            "world_id",
            "queried_entity_slot",
            "queried_property",
            "distractor_property",
            "codebook_id",
        )
        if any(
            fact_order_flip[field] != cell[field] for field in order_fixed_fields
        ) or (
            fact_order_flip["fact_line_order"] == cell["fact_line_order"]
            or fact_order_flip["native_answer"] != cell["native_answer"]
            or fact_order_flip["fact_order_flip_cell_id"] != cell["cell_id"]
        ):
            raise ContentRoutingBankError("fact-order nuisance relation is invalid")


def _validate_recipient_design(
    recipient: Mapping[str, Any], cells: list[dict[str, Any]]
) -> None:
    expected = _recipient_fractional_factorial()
    if dict(recipient) != expected:
        raise ContentRoutingBankError("recipient fractional factorial changed")
    by_id = {cell["cell_id"]: cell for cell in cells}
    declared_ids = [
        cell_id
        for world_selection in recipient["selected_by_world"]
        for cell_id in world_selection["cell_ids"]
    ]
    flagged_ids = [cell["cell_id"] for cell in cells if cell["recipient_selected"]]
    if len(declared_ids) != 448 or len(set(declared_ids)) != 448:
        raise ContentRoutingBankError("recipient IDs are not 56 distinct eight-cell sets")
    if set(declared_ids) != set(flagged_ids):
        raise ContentRoutingBankError("recipient registry and cell flags disagree")
    role_counts: Counter[str] = Counter()
    for world_selection in recipient["selected_by_world"]:
        selected = [by_id[cell_id] for cell_id in world_selection["cell_ids"]]
        if len(selected) != 8 or {cell["world_id"] for cell in selected} != {
            world_selection["world_id"]
        }:
            raise ContentRoutingBankError("recipient world allocation changed")
        if sorted(cell["recipient_run_index"] for cell in selected) != list(
            range(1, 9)
        ):
            raise ContentRoutingBankError("recipient run indices changed")
        for factor, field in (
            ("m", "queried_entity_slot"),
            ("q", "queried_property"),
            ("d", "distractor_property"),
            ("c", "codebook_id"),
            ("order", "fact_line_order"),
        ):
            counts = Counter(cell[field] for cell in selected)
            if sorted(counts.values()) != [4, 4]:
                raise ContentRoutingBankError(f"recipient factor {factor} is unbalanced")
        if Counter(cell["native_answer"] for cell in selected) != {"X": 4, "Y": 4}:
            raise ContentRoutingBankError("recipient native answers are unbalanced")
        contrast_rows: list[list[int]] = []
        for cell in selected:
            e_sign = _factor_sign(cell["queried_entity_slot"], "a", "b")
            p_sign = _factor_sign(cell["queried_property"], "P", "Q")
            m_sign = _factor_sign(cell["codebook_id"], "identity", "swapped")
            d_sign = _factor_sign(cell["distractor_property"], "P", "Q")
            order_sign = _factor_sign(
                cell["fact_line_order"], "query_first", "query_second"
            )
            answer_sign = _factor_sign(cell["native_answer"], "Y", "X")
            if (
                d_sign != e_sign * p_sign
                or order_sign != e_sign * m_sign
                or answer_sign != p_sign * m_sign
            ):
                raise ContentRoutingBankError("recipient generator relation changed")
            contrast_rows.append(
                [
                    p_sign,
                    m_sign,
                    e_sign,
                    d_sign,
                    order_sign,
                    answer_sign,
                ]
            )
        expected_gram = [
            [8 if left == right else 0 for right in range(6)]
            for left in range(6)
        ]
        if _gram_matrix(contrast_rows) != expected_gram:
            raise ContentRoutingBankError(
                "recipient contrasts are not pairwise orthogonal"
            )
        role_counts.update([world_selection["role"]] * len(selected))
    if dict(role_counts) != ROLE_RECIPIENT_COUNTS:
        raise ContentRoutingBankError("recipient role counts changed")


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "analysis_id",
        "freeze_date",
        "mode",
        "purpose",
        "firewall",
        "outcome_exposure",
        "model_calls_made_by_builder",
        "inferential_unit",
        "dependency_unit",
        "prior_references",
        "naming_contract",
        "factor_contract",
        "codebooks",
        "reference_contract",
        "world_registry",
        "cells",
        "recipient_fractional_factorial",
    }
    if set(value) != expected_keys:
        raise ContentRoutingBankError("fixture schema changed")
    if value["schema_version"] != FIXTURE_SCHEMA or value["analysis_id"] != ANALYSIS_ID:
        raise ContentRoutingBankError("fixture identity changed")
    if value["freeze_date"] != FREEZE_DATE:
        raise ContentRoutingBankError("freeze date changed")
    if value["mode"] != "prospective_pre_execution" or value["purpose"] != PURPOSE:
        raise ContentRoutingBankError("fixture mode or purpose changed")
    if value["firewall"] != FIREWALL or value["outcome_exposure"] != OUTCOME_EXPOSURE:
        raise ContentRoutingBankError("firewall or output-exposure contract changed")
    if value["model_calls_made_by_builder"] != 0:
        raise ContentRoutingBankError("builder claims model execution")
    if value["inferential_unit"] != "symbolic_world":
        raise ContentRoutingBankError("inferential unit changed")
    if value["dependency_unit"] != "complete_factorial_within_symbolic_world":
        raise ContentRoutingBankError("dependency unit changed")
    if value["codebooks"] != CODEBOOKS or value["reference_contract"] != REFERENCE_CONTRACT:
        raise ContentRoutingBankError("codebook or reference contract changed")

    references = _prior_references()
    if value["prior_references"] != references:
        raise ContentRoutingBankError("prior-reference registry changed")
    expected_naming = {
        "construction": "symbol_{a_or_b}_{world_index_3_digits}",
        "world_indices": {"first": 1, "last": 56, "width": 3},
        "globally_unique": True,
        "semantic_content": "controlled_symbolic_identifier_only",
    }
    if value["naming_contract"] != expected_naming:
        raise ContentRoutingBankError("controlled naming contract changed")
    expected_factor_contract = {
        "factor_order": list(FACTOR_LEVELS),
        "factor_levels": FACTOR_LEVELS,
        "full_factorial": "2^5",
        "cells_per_world": 32,
        "native_answer_rule": "codebooks[codebook][queried_property]",
        "world_roles": ROLE_RANGES,
        "prompt_cells_by_role": ROLE_PROMPT_COUNTS,
    }
    if value["factor_contract"] != expected_factor_contract:
        raise ContentRoutingBankError("factor contract changed")

    worlds = _world_registry()
    if value["world_registry"] != worlds or len(worlds) != 56:
        raise ContentRoutingBankError("world registry changed")
    if [world["world_index"] for world in worlds] != list(range(1, 57)):
        raise ContentRoutingBankError("world order changed")
    if Counter(world["role"] for world in worlds) != ROLE_WORLD_COUNTS:
        raise ContentRoutingBankError("world role allocation changed")
    symbolic_names = [
        name for world in worlds for name in (world["entity_a"], world["entity_b"])
    ]
    expected_names = [
        name
        for index in range(1, 57)
        for name in (f"symbol_a_{index:03d}", f"symbol_b_{index:03d}")
    ]
    if symbolic_names != expected_names or len(set(symbolic_names)) != 112:
        raise ContentRoutingBankError("symbolic names changed or are not unique")
    if any(CONTROLLED_NAME_PATTERN.fullmatch(name) is None for name in symbolic_names):
        raise ContentRoutingBankError("symbolic name format changed")

    cells = value["cells"]
    expected_cells = _build_cells()
    if cells != expected_cells or len(cells) != 1792:
        raise ContentRoutingBankError("complete factorial cells changed")
    if Counter(cell["role"] for cell in cells) != ROLE_PROMPT_COUNTS:
        raise ContentRoutingBankError("prompt role counts changed")
    if len({cell["prompt_text"] for cell in cells}) != 1792:
        raise ContentRoutingBankError("prompt texts are not unique")
    world_cell_counts = Counter(cell["world_id"] for cell in cells)
    if set(world_cell_counts.values()) != {32} or len(world_cell_counts) != 56:
        raise ContentRoutingBankError("world factorial coverage changed")
    for cell in cells:
        if cell["schema_version"] != CELL_SCHEMA:
            raise ContentRoutingBankError("cell schema changed")
        if cell["native_answer"] != CODEBOOKS[cell["codebook_id"]][
            cell["queried_property"]
        ]:
            raise ContentRoutingBankError("native answer is not codebook(query_property)")
        if cell["prompt_sha256"] != text_sha256(cell["prompt_text"]):
            raise ContentRoutingBankError("prompt hash changed")
        if cell["fact_line_order"] == "query_first":
            expected_first_entity = cell["queried_entity"]
        else:
            expected_first_entity = cell["distractor_entity"]
        if expected_first_entity not in cell["fact_lines"][0]:
            raise ContentRoutingBankError("fact-line order changed")
    _validate_reference_graph(cells)
    _validate_recipient_design(value["recipient_fractional_factorial"], cells)
    _validate_disjointness(worlds, cells, references)
    return dict(value)


def build_fixture() -> dict[str, Any]:
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": "prospective_pre_execution",
        "purpose": PURPOSE,
        "firewall": FIREWALL,
        "outcome_exposure": OUTCOME_EXPOSURE,
        "model_calls_made_by_builder": 0,
        "inferential_unit": "symbolic_world",
        "dependency_unit": "complete_factorial_within_symbolic_world",
        "prior_references": _prior_references(),
        "naming_contract": {
            "construction": "symbol_{a_or_b}_{world_index_3_digits}",
            "world_indices": {"first": 1, "last": 56, "width": 3},
            "globally_unique": True,
            "semantic_content": "controlled_symbolic_identifier_only",
        },
        "factor_contract": {
            "factor_order": list(FACTOR_LEVELS),
            "factor_levels": FACTOR_LEVELS,
            "full_factorial": "2^5",
            "cells_per_world": 32,
            "native_answer_rule": "codebooks[codebook][queried_property]",
            "world_roles": ROLE_RANGES,
            "prompt_cells_by_role": ROLE_PROMPT_COUNTS,
        },
        "codebooks": CODEBOOKS,
        "reference_contract": REFERENCE_CONTRACT,
        "world_registry": _world_registry(),
        "cells": _build_cells(),
        "recipient_fractional_factorial": _recipient_fractional_factorial(),
    }
    return validate_fixture(fixture)


def _manifest_payload(fixture: Mapping[str, Any], fixture_path: Path) -> dict[str, Any]:
    locked = validate_fixture(fixture)
    if not fixture_path.is_file():
        raise ContentRoutingBankError("fixture must be written before its manifest")
    try:
        on_disk = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContentRoutingBankError("fixture path is not valid JSON") from error
    if on_disk != locked:
        raise ContentRoutingBankError("fixture path does not contain the supplied fixture")
    recipient = locked["recipient_fractional_factorial"]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "FROZEN_PROSPECTIVE_NO_MODEL_FORWARD",
        "freeze_date": FREEZE_DATE,
        "mode": "prospective_pre_execution",
        "purpose": PURPOSE,
        "claim_scope": (
            "synthetic_cross_codebook_content_routing_only_no_biology_latent_"
            "knowledge_activation_gap_physical_law_real_world_or_model_family_claim"
        ),
        "firewall": FIREWALL,
        "outcome_exposure": OUTCOME_EXPOSURE,
        "contract": {
            "symbolic_worlds": 56,
            "symbolic_entities": 112,
            "factors": 5,
            "full_factorial_cells_per_world": 32,
            "prompt_cells": 1792,
            "prompt_cells_by_role": ROLE_PROMPT_COUNTS,
            "reference_roles_per_cell": 7,
            "recipient_cells_per_world": 8,
            "recipient_cells": 448,
            "recipient_cells_by_role": ROLE_RECIPIENT_COUNTS,
            "recipient_generators": {
                "d": "e*p",
                "order": "e*m",
                "native_answer": "p*m",
            },
            "recipient_contrast_gram_matrix": "8I_6",
            "model_calls_made_by_builder": 0,
        },
        "provenance": {
            "source_type": "deterministic_uniform_synthetic_symbolic_factorial",
            "external_data_sources": [],
            "prior_references": locked["prior_references"],
            "world_registry_canonical_sha256": canonical_sha256(
                locked["world_registry"]
            ),
            "cell_registry_canonical_sha256": canonical_sha256(locked["cells"]),
            "recipient_design_canonical_sha256": canonical_sha256(recipient),
            "reference_contract_canonical_sha256": canonical_sha256(
                locked["reference_contract"]
            ),
        },
        "artifacts": {
            "builder_path": relative_path(Path(__file__)),
            "builder_sha256": file_sha256(Path(__file__)),
            "fixture_path": relative_path(fixture_path),
            "fixture_sha256": file_sha256(fixture_path),
            "fixture_canonical_sha256": canonical_sha256(locked),
        },
    }


def build_manifest(fixture: Mapping[str, Any], fixture_path: Path) -> dict[str, Any]:
    return _manifest_payload(fixture, fixture_path)


def validate_manifest(
    value: Mapping[str, Any], fixture: Mapping[str, Any], fixture_path: Path
) -> dict[str, Any]:
    expected = _manifest_payload(fixture, fixture_path)
    if dict(value) != expected:
        raise ContentRoutingBankError("manifest schema or binding changed")
    return dict(value)


def _write_frozen_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ContentRoutingBankError(f"refusing to overwrite frozen artifact: {path}")
    path.write_bytes(payload)


def build_and_write(
    out: Path = DEFAULT_OUT, manifest_out: Path = DEFAULT_MANIFEST
) -> tuple[dict[str, Any], dict[str, Any]]:
    if out.resolve() == manifest_out.resolve():
        raise ContentRoutingBankError("fixture and manifest paths must differ")
    fixture = build_fixture()
    _write_frozen_json(out, fixture)
    manifest = build_manifest(fixture, out)
    _write_frozen_json(manifest_out, manifest)
    return fixture, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    fixture, manifest = build_and_write(args.out, args.manifest_out)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "worlds": len(fixture["world_registry"]),
                "prompt_cells": len(fixture["cells"]),
                "recipient_cells": sum(
                    cell["recipient_selected"] for cell in fixture["cells"]
                ),
                "fixture_canonical_sha256": canonical_sha256(fixture),
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

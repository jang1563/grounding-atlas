"""Run the preregistered V5 causal TARGET-order/context experiment.

The experiment is deliberately phase gated.  ``plan`` performs fixture,
tokenizer, and dependency checks but makes zero model forwards.  Each later
phase is authorized by an immutable artifact issued by the frozen analyzer.
All outcomes are teacher-forced next-token logits; generation is never used.

This runner is specific to a synthetic prompt task.  It does not support a
claim about biology, latent biological knowledge, or a physical law.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import model_hooks
except ImportError:  # direct execution from eval/
    import model_hooks


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v5_positional_activation"
    / "qwen2.5-1.5b-instruct"
)

FROZEN_PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V5_POSITIONAL_ACTIVATION_GAP_PREREG.md"
FROZEN_PREREGISTRATION_SHA256 = "0a2b5fef4329bf5fad3a004a8a03da2b4f00f0ebc344359e5f2857073e439f1e"
FIXTURE_BUILDER = ROOT / "signal" / "syntax" / "build_coherent_readout_v5_positional_activation_bank.py"
FIXTURE_BUILDER_SHA256 = "9356b383daf0eb4bdf501eb201a2834b4acb41c05638fa76c4cbece6d9948150"
FIXTURE = ROOT / "signal" / "syntax" / "coherent_readout_v5_positional_activation_bank.json"
FIXTURE_SHA256 = "defa5ed2c0ab1f0f6c7ac7cf5eaa4abe453daac682676b9d783770ffae6da903"
FIXTURE_CANONICAL_SHA256 = "0ac414c77ea6a84f4003b42ca9ca2a356d3a61f3c0e92abfb5a0aa4868abb75c"
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
FIXTURE_MANIFEST_SHA256 = "01bd21c52e307b62a073f57f0a3e9086096894e7bb975d6ceda3d0f8930729cb"
MODEL_HOOKS_SHA256 = "62495bd77adc40d7fd5e5643df334eb98aba363f5b81b4b7925314e877bad0c4"
DEFAULT_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v5_positional_activation.py"

DEFAULT_DESIGN = RESULT_ROOT / "design.json"
DEFAULT_PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
DEFAULT_TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"
DEFAULT_DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
DEFAULT_BASIS_LOCK = RESULT_ROOT / "fit_basis_lock.json"
DEFAULT_BASIS_SIDECAR = RESULT_ROOT / "fit_basis.npy"
DEFAULT_LOCALIZATION_BASELINE_ENTRY = RESULT_ROOT / "localization_baseline_entry.json"
DEFAULT_LOCALIZATION_PATCH_ENTRY = RESULT_ROOT / "localization_patch_entry.json"
DEFAULT_LAYER_LOCK = RESULT_ROOT / "layer_lock.json"
DEFAULT_HOLDOUT_BASELINE_ENTRY = RESULT_ROOT / "holdout_baseline_entry.json"
DEFAULT_HOLDOUT_PATCH_ENTRY = RESULT_ROOT / "holdout_patch_entry.json"

PLAN_SCHEMA = "coherent-readout-v5-positional-activation-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v5-positional-activation-design-v1"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v5-positional-activation-plan-manifest-v1"
TOKENIZATION_RECEIPT_SCHEMA = "coherent-readout-v5-positional-activation-tokenization-v1"
DEPENDENCY_LOCK_SCHEMA = "coherent-readout-v5-positional-activation-dependencies-v1"
PROMPT_SCHEMA = "coherent-readout-v5-positional-activation-prompt-v1"
BASELINE_TEMPLATE_SCHEMA = "coherent-readout-v5-positional-activation-baseline-template-v1"
PATCH_TEMPLATE_SCHEMA = "coherent-readout-v5-positional-activation-patch-template-v1"
BASELINE_SCHEMA = "coherent-readout-v5-positional-activation-baseline-v1"
PATCH_SCHEMA = "coherent-readout-v5-positional-activation-patch-v1"
ATTEMPT_SCHEMA = "coherent-readout-v5-positional-activation-attempt-v1"
EXECUTION_MANIFEST_SCHEMA = "coherent-readout-v5-positional-activation-execution-v1"
LOCALIZATION_BASELINE_ENTRY_SCHEMA = "coherent-readout-v5-positional-localization-baseline-entry-v1"
LOCALIZATION_PATCH_ENTRY_SCHEMA = "coherent-readout-v5-positional-localization-patch-entry-v1"
LAYER_LOCK_SCHEMA = "coherent-readout-v5-positional-layer-lock-v1"
HOLDOUT_BASELINE_ENTRY_SCHEMA = "coherent-readout-v5-positional-holdout-baseline-entry-v1"
HOLDOUT_PATCH_ENTRY_SCHEMA = "coherent-readout-v5-positional-holdout-patch-entry-v1"

CLAIM_BOUNDARIES = {
    "supported_scope": (
        "synthetic_model_prompt_target_property_token_and_selected_layer_specific_"
        "causal_target_order_context_state_mediation"
    ),
    "biology_inference": "forbidden",
    "latent_biological_knowledge_inference": "forbidden",
    "general_activation_gap_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
    "absolute_position_vs_preceding_other_interference": "not_separated",
    "natural_gap_is_behaviorally_preexisting": True,
    "intervention_is_selective_rank_one_coordinate_replacement": True,
}

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_WEIGHTS_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
MODEL_CONFIG_SHA256 = "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
TOKENIZER_CONFIG_SHA256 = "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
TOKENIZER_JSON_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
MODEL_LAYERS = 28
MODEL_WIDTH = 1536
MODEL_VOCAB_SIZE = 151_936
TOKENIZER_VOCAB_SIZE = 151_665
DEVICE = "mps"
DTYPE = "float32"
ATTENTION_IMPLEMENTATION = "sdpa"

SYSTEM_EXACT = (
    "Follow the user's instructions. Your entire response must be exactly one "
    "uppercase character from the valid output choices. Do not write any other text."
)
CHAT_FLAGS = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "enable_thinking": False,
}
LAYER_GRID = (12, 16, 20, 24)
AXIS_REGISTRY = ("center", "positional", "answer", "null")
ROLE_ORDER = ("fit", "localization", "holdout")
ROLE_WORLD_COUNTS = {"fit": 8, "localization": 8, "holdout": 16}
BASELINES_PER_WORLD = 56
COMPOSITIONS_PER_WORLD = 32
INTERVENTION_PAIRS_PER_WORLD = 8
RAW_SHARD_ROWS = 64
NUMERICAL_TOLERANCE = 1e-6
DOSE_MATCH_TOLERANCE = 1e-5

PATCH_CONDITIONS = (
    "positional_rescue",
    "positional_damage",
    "answer_rescue_sham",
    "answer_damage_sham",
    "null_rescue_sham",
    "null_damage_sham",
)
ALL_PATCH_CONDITIONS = PATCH_CONDITIONS + ("identity",)
EXPECTED_COUNTS = {
    "fit-baseline": 448,
    "localization-baseline": 448,
    "localization-patch": 1568,
    "holdout-baseline": 896,
    "holdout-patch": 784,
}
EXPECTED_ACTIVATION_ROWS = {
    "fit-baseline": 256,
    "localization-baseline": 256,
    "holdout-baseline": 512,
}
EXPECTED_CUMULATIVE_CALLS = {
    "fit-baseline": 448,
    "localization-baseline": 896,
    "localization-patch": 2464,
    "holdout-baseline": 3360,
    "holdout-patch": 4144,
}


def _phase_paths(phase: str) -> dict[str, Path]:
    stem = phase.replace("-", "_")
    result = {
        "attempt": RESULT_ROOT / f"{stem}_attempt.json",
        "records": RESULT_ROOT / f"{stem}_records.jsonl",
        "manifest": RESULT_ROOT / f"{stem}_execution_manifest.json",
    }
    if phase.endswith("baseline"):
        result["activations"] = RESULT_ROOT / f"{stem}_activations.npy"
    else:
        result["patched_activations"] = RESULT_ROOT / f"{stem}_patched_activations.npy"
    return result


PHASE_PATHS = {phase: _phase_paths(phase) for phase in EXPECTED_COUNTS}


class PositionalActivationRunnerError(ValueError):
    """Raised when an execution violates the frozen V5 contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def f32_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise PositionalActivationRunnerError("float32 artifact is not finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PositionalActivationRunnerError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PositionalActivationRunnerError(f"JSON artifact must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise PositionalActivationRunnerError(f"cannot read JSONL artifact: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise PositionalActivationRunnerError("JSONL records must be objects")
    return values


def _atomic_frozen_write(path: Path, payload: bytes) -> None:
    """Atomically write once; an exact replay is a no-op."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PositionalActivationRunnerError(f"refusing to overwrite differing frozen artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise PositionalActivationRunnerError(f"stale atomic temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
    _atomic_frozen_write(path, payload)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(value)) + "\n" for value in values).encode()
    _atomic_frozen_write(path, payload)


def write_array(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise PositionalActivationRunnerError("array sidecar is not finite")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_frozen_write(path, buffer.getvalue())


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise PositionalActivationRunnerError(f"required package is missing: {name}") from error


def _load_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("v5_positional_fixture_builder", FIXTURE_BUILDER)
    if spec is None or spec.loader is None:
        raise PositionalActivationRunnerError("cannot import V5 fixture builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_and_rebuild_fixture() -> dict[str, Any]:
    """Independently rebuild and compare the prospective fixture."""

    for path in (FROZEN_PREREGISTRATION, FIXTURE_BUILDER, FIXTURE, FIXTURE_MANIFEST):
        if not path.is_file():
            raise PositionalActivationRunnerError(f"required static artifact is missing: {path}")
    frozen_hashes = {
        FROZEN_PREREGISTRATION: FROZEN_PREREGISTRATION_SHA256,
        FIXTURE_BUILDER: FIXTURE_BUILDER_SHA256,
        FIXTURE: FIXTURE_SHA256,
        FIXTURE_MANIFEST: FIXTURE_MANIFEST_SHA256,
        ROOT / "eval" / "model_hooks.py": MODEL_HOOKS_SHA256,
    }
    for path, expected in frozen_hashes.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise PositionalActivationRunnerError(f"static freeze hash changed: {path}")
    fixture = _load_json(FIXTURE)
    if canonical_sha256(fixture) != FIXTURE_CANONICAL_SHA256:
        raise PositionalActivationRunnerError("fixture canonical freeze hash changed")
    builder = _load_builder_module()
    try:
        rebuilt = builder.build_fixture()
        builder.validate_fixture(rebuilt)
    except Exception as error:
        raise PositionalActivationRunnerError("fixture semantic rebuild failed") from error
    if rebuilt != fixture:
        raise PositionalActivationRunnerError("fixture differs from deterministic rebuild")
    _validate_fixture(fixture)
    return fixture


def _worlds(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = fixture.get("worlds", fixture.get("world_registry"))
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise PositionalActivationRunnerError("fixture worlds registry is invalid")
    return [dict(item) for item in raw]


def _normal_role(value: Any) -> str:
    aliases = {"fit": "fit", "direction_fit": "fit", "localization": "localization", "holdout": "holdout"}
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise PositionalActivationRunnerError(f"unknown fixture role: {value}") from error


def _family(cell: Mapping[str, Any]) -> str:
    value = str(cell.get("family", cell.get("family_id", cell.get("task_type", ""))))
    aliases = {
        "retrieval": "retrieval",
        "property_retrieval": "retrieval",
        "lookup": "lookup",
        "codebook_lookup": "lookup",
        "composition": "composition",
        "joint": "composition",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise PositionalActivationRunnerError(f"unknown cell family: {value}") from error


def _order(cell: Mapping[str, Any]) -> str | None:
    value = cell.get(
        "order",
        cell.get("fact_order", cell.get("target_fact_order", cell.get("fact_line_order"))),
    )
    if value is None:
        return None
    aliases = {
        "first": "first",
        "second": "second",
        "target_first": "first",
        "target_second": "second",
        "query_first": "first",
        "query_second": "second",
    }
    try:
        return aliases[str(value)]
    except KeyError as error:
        raise PositionalActivationRunnerError(f"unknown composition order: {value}") from error


def _answer_choices(cell: Mapping[str, Any]) -> tuple[str, str]:
    expected = cell.get("correct_answer", cell.get("native_answer"))
    raw = cell.get(
        "valid_answers",
        cell.get(
            "answer_options",
            cell.get("answer_labels", cell.get("displayed_options", cell.get("valid_output_codes"))),
        ),
    )
    if not isinstance(expected, str) or len(expected) != 1 or not expected.isupper():
        raise PositionalActivationRunnerError("correct answer must be one uppercase character")
    if not isinstance(raw, list) or len(raw) != 2 or len(set(raw)) != 2:
        raise PositionalActivationRunnerError("each cell needs exactly two valid answers")
    if any(not isinstance(item, str) or len(item) != 1 or not item.isupper() for item in raw):
        raise PositionalActivationRunnerError("valid answers must be uppercase characters")
    if expected not in raw:
        raise PositionalActivationRunnerError("correct answer is outside valid answers")
    other = raw[1] if raw[0] == expected else raw[0]
    return expected, other


def _target_span(cell: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = cell.get("target_property_span", cell.get("target_property_char_span"))
    if raw is None:
        return None
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) for x in raw)
        or not 0 <= raw[0] < raw[1] <= len(str(cell.get("prompt_text", "")))
    ):
        raise PositionalActivationRunnerError("target-property character span is invalid")
    return int(raw[0]), int(raw[1])


def _intervention_pairs(world: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    raw = world.get("intervention_pairs")
    if isinstance(raw, list):
        result = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise PositionalActivationRunnerError("world intervention pair is invalid")
            first = item.get("first_cell_id", item.get("target_first_cell_id", item.get("first")))
            second = item.get("second_cell_id", item.get("target_second_cell_id", item.get("second")))
            pair_id = str(item.get("pair_id", f"{world['world_id']}:pair:{index:02d}"))
            if not isinstance(first, str) or not isinstance(second, str):
                raise PositionalActivationRunnerError("intervention pair lacks first/second IDs")
            result.append({"pair_id": pair_id, "first_cell_id": first, "second_cell_id": second})
        return result

    # Compatibility with a cell-level explicit pair contract.
    selected = [cell for cell in cells if _family(cell) == "composition" and cell.get("recipient_selected") is True]
    grouped: dict[str, dict[str, str]] = {}
    for cell in selected:
        pair_id = str(cell.get("pair_id", cell.get("semantic_pair_id", "")))
        order = _order(cell)
        if not pair_id or order not in {"first", "second"}:
            raise PositionalActivationRunnerError("selected cell lacks pair/order metadata")
        grouped.setdefault(pair_id, {})[f"{order}_cell_id"] = str(cell["cell_id"])
    return [{"pair_id": pair_id, **value} for pair_id, value in sorted(grouped.items())]


def _validate_fixture(fixture: Mapping[str, Any]) -> None:
    cells = fixture.get("cells")
    worlds = _worlds(fixture)
    if not isinstance(cells, list) or any(not isinstance(cell, dict) for cell in cells):
        raise PositionalActivationRunnerError("fixture cells registry is invalid")
    if len(worlds) != 32 or len(cells) != 32 * BASELINES_PER_WORLD:
        raise PositionalActivationRunnerError("fixture world/cell count changed")
    fixture_system = fixture.get("system_message", fixture.get("system_exact"))
    if fixture_system != SYSTEM_EXACT:
        raise PositionalActivationRunnerError("fixture and runner system messages differ")
    if len({cell.get("cell_id") for cell in cells}) != len(cells):
        raise PositionalActivationRunnerError("fixture cell IDs are duplicated")
    if len({world.get("world_id") for world in worlds}) != len(worlds):
        raise PositionalActivationRunnerError("fixture world IDs are duplicated")
    role_counts = {role: 0 for role in ROLE_ORDER}
    for world in worlds:
        role_counts[_normal_role(world.get("role"))] += 1
    if role_counts != ROLE_WORLD_COUNTS:
        raise PositionalActivationRunnerError("fixture split counts changed")
    world_ids = {str(world["world_id"]) for world in worlds}
    by_world: dict[str, list[dict[str, Any]]] = {world_id: [] for world_id in world_ids}
    prompts: set[str] = set()
    for cell in cells:
        required = ("cell_id", "world_id", "prompt_text")
        if any(not isinstance(cell.get(key), str) or not cell.get(key) for key in required):
            raise PositionalActivationRunnerError("fixture cell identity/text is invalid")
        if str(cell["world_id"]) not in world_ids:
            raise PositionalActivationRunnerError("cell refers to an unknown world")
        if cell["prompt_text"] in prompts:
            raise PositionalActivationRunnerError("fixture prompts must be globally unique")
        prompts.add(cell["prompt_text"])
        _answer_choices(cell)
        family = _family(cell)
        if family == "composition":
            span = _target_span(cell)
            target = cell.get("target_property")
            if span is None or not isinstance(target, str) or cell["prompt_text"][slice(*span)] != target:
                raise PositionalActivationRunnerError("composition target span does not select target property")
            if _order(cell) not in {"first", "second"}:
                raise PositionalActivationRunnerError("composition cell lacks first/second order")
            factors = cell.get("factors")
            if not isinstance(factors, Mapping) or set(factors) != {"p", "m", "r", "v", "o"}:
                raise PositionalActivationRunnerError("composition factors must be exactly p,m,r,v,o")
            if any(isinstance(value, bool) or value not in (-1, 1) for value in factors.values()):
                raise PositionalActivationRunnerError("composition factors must be +/-1")
            if int(factors["o"]) != (-1 if _order(cell) == "first" else 1):
                raise PositionalActivationRunnerError("composition order sign convention changed")
        by_world[str(cell["world_id"])].append(cell)
    for world in worlds:
        rows = by_world[str(world["world_id"])]
        role = _normal_role(world.get("role"))
        if any(_normal_role(row.get("role")) != role for row in rows):
            raise PositionalActivationRunnerError("cell role differs from its world role")
        family_counts = {
            name: sum(_family(row) == name for row in rows) for name in ("retrieval", "lookup", "composition")
        }
        if family_counts != {"retrieval": 8, "lookup": 16, "composition": 32}:
            raise PositionalActivationRunnerError("per-world baseline factorial changed")
        if sum(row.get("intervention_prerequisite") is True for row in rows) != 16:
            raise PositionalActivationRunnerError("per-world intervention prerequisite count changed")
        pairs = _intervention_pairs(world, rows)
        if len(pairs) != INTERVENTION_PAIRS_PER_WORLD:
            raise PositionalActivationRunnerError("per-world intervention-pair count changed")
        row_map = {str(row["cell_id"]): row for row in rows}
        used: set[str] = set()
        for pair in pairs:
            first = row_map.get(pair["first_cell_id"])
            second = row_map.get(pair["second_cell_id"])
            if first is None or second is None or _family(first) != "composition" or _family(second) != "composition":
                raise PositionalActivationRunnerError("intervention pair does not resolve within world")
            if _order(first) != "first" or _order(second) != "second":
                raise PositionalActivationRunnerError("intervention pair order is reversed")
            if _answer_choices(first) != _answer_choices(second):
                raise PositionalActivationRunnerError("paired cells do not have the same answer")
            if pair["first_cell_id"] in used or pair["second_cell_id"] in used:
                raise PositionalActivationRunnerError("intervention pair reuses a cell")
            used.update((pair["first_cell_id"], pair["second_cell_id"]))


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise PositionalActivationRunnerError(f"{label} must contain one row")
        raw = raw[0]
    if not raw or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise PositionalActivationRunnerError(f"{label} must be a nonempty integer vector")
    return [int(item) for item in raw]


def _contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(tokenizer.encode(rendered, add_special_tokens=False), "prompt")
    combined = _as_int_vector(tokenizer.encode(rendered + answer, add_special_tokens=False), "prompt plus answer")
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise PositionalActivationRunnerError(f"answer {answer} is not one contextual token")
    return combined[-1]


def locate_exact_token(offsets: Sequence[Sequence[int]], span: tuple[int, int], rendered: str | None = None) -> int:
    """Resolve a character span to exactly one lexical token.

    Byte-level tokenizers can include the preceding space in a token's reported
    offset.  The selected token may therefore strictly contain ``span``, but no
    second token may overlap it and the extra characters must be whitespace.
    """

    matches = [index for index, pair in enumerate(offsets) if int(pair[0]) < span[1] and int(pair[1]) > span[0]]
    if len(matches) != 1:
        raise PositionalActivationRunnerError("TARGET property span is not exactly one token")
    index = matches[0]
    start, end = map(int, offsets[index])
    if start > span[0] or end < span[1]:
        raise PositionalActivationRunnerError("TARGET property token does not cover its span")
    if rendered is not None and rendered[start:end].strip() != rendered[slice(*span)]:
        raise PositionalActivationRunnerError("TARGET property shares a lexical token")
    return index


def render_prompt(tokenizer: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    prompt_text = str(cell["prompt_text"])
    if cell.get("prompt_sha256") not in (None, text_sha256(prompt_text)):
        raise PositionalActivationRunnerError("fixture prompt digest changed")
    messages = [
        {"role": "system", "content": SYSTEM_EXACT},
        {"role": "user", "content": prompt_text},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    if not isinstance(rendered, str) or not rendered:
        raise PositionalActivationRunnerError("chat template returned no rendered prompt")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = _as_int_vector(encoded["input_ids"], "rendered input IDs")
    offsets = [tuple(int(x) for x in pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise PositionalActivationRunnerError("token offsets and IDs differ in length")
    prompt_occurrences = [index for index in range(len(rendered)) if rendered.startswith(prompt_text, index)]
    if len(prompt_occurrences) != 1:
        raise PositionalActivationRunnerError("user prompt does not occur exactly once in rendered chat")
    family = _family(cell)
    target_token_index = None
    rendered_span = None
    if family == "composition":
        local_span = _target_span(cell)
        if local_span is None:
            raise PositionalActivationRunnerError("composition target span is missing")
        rendered_span = (prompt_occurrences[0] + local_span[0], prompt_occurrences[0] + local_span[1])
        target_token_index = locate_exact_token(offsets, rendered_span, rendered)
        target_property = str(cell["target_property"])
        if rendered[slice(*rendered_span)] != target_property:
            raise PositionalActivationRunnerError("rendered target span changed")
    expected, distractor = _answer_choices(cell)
    expected_id = _contextual_token_id(tokenizer, rendered, expected)
    distractor_id = _contextual_token_id(tokenizer, rendered, distractor)
    if expected_id == distractor_id:
        raise PositionalActivationRunnerError("answer choices share a token ID")
    identity = {
        "schema_version": PROMPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "world_id": cell["world_id"],
        "role": _normal_role(cell.get("role")),
        "family": family,
        "system_text_sha256": text_sha256(SYSTEM_EXACT),
        "user_text_sha256": text_sha256(prompt_text),
        "rendered_text_sha256": text_sha256(rendered),
        "input_ids_sha256": canonical_sha256(ids),
    }
    return {
        **identity,
        "prompt_id": canonical_sha256(identity),
        "system_text": SYSTEM_EXACT,
        "user_text": prompt_text,
        "rendered_text": rendered,
        "execution_input_ids": ids,
        "execution_attention_mask": [1] * len(ids),
        "input_token_count": len(ids),
        "final_attended_token_index": len(ids) - 1,
        "expected_answer": expected,
        "expected_token_id": expected_id,
        "distractor_answer": distractor,
        "distractor_token_id": distractor_id,
        "target_property_rendered_span": None if rendered_span is None else list(rendered_span),
        "target_property_token_index": target_token_index,
        "target_property_token_id": None if target_token_index is None else ids[target_token_index],
    }


def _condition_spec(condition: str) -> tuple[str, str, str]:
    specs = {
        "positional_rescue": ("second", "first", "positional"),
        "positional_damage": ("first", "second", "positional"),
        "answer_rescue_sham": ("second", "first", "answer"),
        "answer_damage_sham": ("first", "second", "answer"),
        "null_rescue_sham": ("second", "first", "null"),
        "null_damage_sham": ("first", "second", "null"),
    }
    try:
        return specs[condition]
    except KeyError as error:
        raise PositionalActivationRunnerError(f"unknown patch condition: {condition}") from error


def _baseline_template(cell: Mapping[str, Any], prompt: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": BASELINE_TEMPLATE_SCHEMA,
        "role": _normal_role(cell.get("role")),
        "family": _family(cell),
        "world_id": cell["world_id"],
        "cell_id": cell["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "capture_target_property": _family(cell) == "composition",
        "intervention_prerequisite": cell.get("intervention_prerequisite") is True,
    }
    return {**identity, "template_id": canonical_sha256(identity)}


def _patch_template(
    *,
    role: str,
    world_id: str,
    pair_id: str,
    first_cell_id: str,
    second_cell_id: str,
    condition: str,
    layer: int | None,
) -> dict[str, Any]:
    recipient_order, source_order, direction = _condition_spec(condition)
    ids = {"first": first_cell_id, "second": second_cell_id}
    operation = "positional_coordinate_patch" if direction == "positional" else "dose_matched_additive_sham"
    identity = {
        "schema_version": PATCH_TEMPLATE_SCHEMA,
        "role": role,
        "world_id": world_id,
        "pair_id": pair_id,
        "condition": condition,
        "operation": operation,
        "direction_name": direction,
        "recipient_order": recipient_order,
        "source_order": source_order,
        "recipient_cell_id": ids[recipient_order],
        "source_cell_id": ids[source_order],
        "layer": layer,
        "layer_from_lock": layer is None,
        "strength": 1.0,
    }
    return {**identity, "template_id": canonical_sha256(identity)}


def _identity_template(
    *, role: str, world_id: str, recipient_cell_id: str, pair_id: str, layer: int | None
) -> dict[str, Any]:
    identity = {
        "schema_version": PATCH_TEMPLATE_SCHEMA,
        "role": role,
        "world_id": world_id,
        "pair_id": pair_id,
        "condition": "identity",
        "operation": "identity_zero_displacement",
        "direction_name": "positional",
        "recipient_order": "first",
        "source_order": "first",
        "recipient_cell_id": recipient_cell_id,
        "source_cell_id": recipient_cell_id,
        "layer": layer,
        "layer_from_lock": layer is None,
        "strength": 1.0,
    }
    return {**identity, "template_id": canonical_sha256(identity)}


def raw_shard_specs(phase: str, count: int | None = None) -> list[dict[str, Any]]:
    expected = EXPECTED_COUNTS[phase] if count is None else int(count)
    specs = []
    for start in range(0, expected, RAW_SHARD_ROWS):
        stop = min(start + RAW_SHARD_ROWS, expected)
        index = len(specs)
        specs.append(
            {
                "index": index,
                "start_row": start,
                "stop_row": stop,
                "rows": stop - start,
                "path": str(RESULT_ROOT / "raw_logits" / phase / f"shard_{index:03d}.npy"),
                "shape": [stop - start, MODEL_VOCAB_SIZE],
                "dtype": "<f4",
            }
        )
    return specs


def matched_pair_shape_receipts(
    worlds: Sequence[Mapping[str, Any]],
    prompts_by_cell: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require every planned intervention pair to have identical input shape."""

    receipts: list[dict[str, Any]] = []
    for world in sorted(worlds, key=lambda item: str(item["world_id"])):
        pairs = world.get("intervention_pairs")
        if not isinstance(pairs, list) or len(pairs) != INTERVENTION_PAIRS_PER_WORLD:
            raise PositionalActivationRunnerError("world intervention-pair registry changed")
        for pair in sorted(pairs, key=lambda item: str(item["pair_id"])):
            first_id = str(pair["target_first_cell_id"])
            second_id = str(pair["target_second_cell_id"])
            first = prompts_by_cell.get(first_id)
            second = prompts_by_cell.get(second_id)
            if first is None or second is None:
                raise PositionalActivationRunnerError("matched intervention prompt does not resolve")
            first_shape = (
                int(first["input_token_count"]),
                len(first["execution_attention_mask"]),
            )
            second_shape = (
                int(second["input_token_count"]),
                len(second["execution_attention_mask"]),
            )
            if (
                first_shape != second_shape
                or first["execution_attention_mask"] != [1] * first_shape[0]
                or second["execution_attention_mask"] != [1] * second_shape[0]
            ):
                raise PositionalActivationRunnerError("matched intervention source/recipient token shapes differ")
            receipts.append(
                {
                    "world_id": world["world_id"],
                    "pair_id": pair["pair_id"],
                    "target_first_cell_id": first_id,
                    "target_second_cell_id": second_id,
                    "input_token_count": first_shape[0],
                    "attention_mask_length": first_shape[1],
                    "target_first_token_index": first["target_property_token_index"],
                    "target_second_token_index": second["target_property_token_index"],
                }
            )
    expected = sum(ROLE_WORLD_COUNTS.values()) * INTERVENTION_PAIRS_PER_WORLD
    if len(receipts) != expected:
        raise PositionalActivationRunnerError("matched intervention-pair receipt count changed")
    return receipts


def _dependency_lock(analyzer_path: Path) -> dict[str, Any]:
    test_paths = {
        "fixture_tests": ROOT / "tests" / "test_build_coherent_readout_v5_positional_activation_bank.py",
        "runner_tests": ROOT / "tests" / "test_run_coherent_readout_v5_positional_activation.py",
        "analyzer_tests": ROOT / "tests" / "test_analyze_coherent_readout_v5_positional_activation.py",
    }
    paths = {
        "runner": Path(__file__),
        "analyzer": analyzer_path,
        "model_hooks": ROOT / "eval" / "model_hooks.py",
        "fixture_builder": FIXTURE_BUILDER,
        "fixture": FIXTURE,
        "fixture_manifest": FIXTURE_MANIFEST,
        "preregistration": FROZEN_PREREGISTRATION,
        **test_paths,
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise PositionalActivationRunnerError(f"dependency is missing: {missing[0]}")
    try:
        import torch
    except ImportError as error:
        raise PositionalActivationRunnerError("torch is required for dependency lock") from error
    core = {
        "schema_version": DEPENDENCY_LOCK_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in ("huggingface-hub", "numpy", "safetensors", "tokenizers", "torch", "transformers")
        },
        "implementation_files": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
        },
        "runtime": {
            "device": DEVICE,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def _require_current_dependency_lock(stored: Mapping[str, Any]) -> None:
    """Replay the dependency census so a frozen receipt cannot mask runtime drift."""

    current = _dependency_lock(DEFAULT_ANALYZER)
    if current != dict(stored):
        raise PositionalActivationRunnerError(
            "current package, runtime, platform, or implementation dependency lock drifted"
        )


def _load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise PositionalActivationRunnerError("transformers is required for plan tokenization") from error
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
    )
    if len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise PositionalActivationRunnerError("tokenizer vocabulary size changed")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or text_sha256(template) != CHAT_TEMPLATE_SHA256:
        raise PositionalActivationRunnerError("effective chat template changed")
    return tokenizer


def build_plan(
    tokenizer: Any, analyzer_path: Path = DEFAULT_ANALYZER
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = load_and_rebuild_fixture()
    dependency = _dependency_lock(analyzer_path)
    worlds = sorted(_worlds(fixture), key=lambda item: str(item["world_id"]))
    cells = sorted((dict(cell) for cell in fixture["cells"]), key=lambda item: str(item["cell_id"]))
    prompts = [render_prompt(tokenizer, cell) for cell in cells]
    prompt_by_cell = {str(prompt["cell_id"]): prompt for prompt in prompts}
    pair_shape_receipts = matched_pair_shape_receipts(worlds, prompt_by_cell)
    baseline_templates = [_baseline_template(cell, prompt_by_cell[str(cell["cell_id"])]) for cell in cells]
    cells_by_world: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        cells_by_world.setdefault(str(cell["world_id"]), []).append(cell)
    patch_templates: list[dict[str, Any]] = []
    for world in worlds:
        role = _normal_role(world.get("role"))
        if role == "fit":
            continue
        pairs = sorted(
            _intervention_pairs(world, cells_by_world[str(world["world_id"])]), key=lambda item: item["pair_id"]
        )
        layers: Sequence[int | None] = LAYER_GRID if role == "localization" else (None,)
        for layer in layers:
            for pair in pairs:
                for condition in PATCH_CONDITIONS:
                    patch_templates.append(
                        _patch_template(
                            role=role,
                            world_id=str(world["world_id"]),
                            pair_id=pair["pair_id"],
                            first_cell_id=pair["first_cell_id"],
                            second_cell_id=pair["second_cell_id"],
                            condition=condition,
                            layer=layer,
                        )
                    )
            sentinel = pairs[0]
            patch_templates.append(
                _identity_template(
                    role=role,
                    world_id=str(world["world_id"]),
                    recipient_cell_id=sentinel["first_cell_id"],
                    pair_id=sentinel["pair_id"],
                    layer=layer,
                )
            )
    patch_templates.sort(
        key=lambda row: (
            ROLE_ORDER.index(row["role"]),
            row["world_id"],
            -1 if row["layer"] is None else row["layer"],
            row["pair_id"],
            ALL_PATCH_CONDITIONS.index(row["condition"]),
        )
    )
    if len(patch_templates) != 1568 + 784:
        raise PositionalActivationRunnerError("patch-template count changed")
    if len({row["template_id"] for row in patch_templates}) != len(patch_templates):
        raise PositionalActivationRunnerError("patch-template IDs are duplicated")
    static = dependency["implementation_files"]
    receipt_core = {
        "schema_version": TOKENIZATION_RECEIPT_SCHEMA,
        "model_calls": 0,
        "generation_used": False,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "chat_flags": CHAT_FLAGS,
        "prompt_count": len(prompts),
        "composition_target_token_count": sum(prompt["family"] == "composition" for prompt in prompts),
        "matched_intervention_pair_shape_count": len(pair_shape_receipts),
        "matched_intervention_pair_shapes_sha256": canonical_sha256(pair_shape_receipts),
        "prompt_receipts": [
            {
                "cell_id": prompt["cell_id"],
                "prompt_id": prompt["prompt_id"],
                "rendered_text_sha256": prompt["rendered_text_sha256"],
                "input_ids_sha256": prompt["input_ids_sha256"],
                "input_token_count": prompt["input_token_count"],
                "target_property_token_index": prompt["target_property_token_index"],
                "expected_token_id": prompt["expected_token_id"],
                "distractor_token_id": prompt["distractor_token_id"],
            }
            for prompt in prompts
        ],
    }
    receipt = {**receipt_core, "canonical_sha256": canonical_sha256(receipt_core)}
    plan_core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": "coherent-readout-v5-causal-target-order-context-gap-v1",
        "mode": "prospective_synthetic_nonbiological_causal_mediation",
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "config_sha256": MODEL_CONFIG_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "decoder_layers": MODEL_LAYERS,
            "hidden_width": MODEL_WIDTH,
            "vocab_size": MODEL_VOCAB_SIZE,
            "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
            "device": DEVICE,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "locks": {name: dict(value) for name, value in static.items()},
        "layer_grid": list(LAYER_GRID),
        "axis_registry": list(AXIS_REGISTRY),
        "role_world_counts": ROLE_WORLD_COUNTS,
        "baseline_templates": baseline_templates,
        "patch_templates": patch_templates,
        "prompts": prompts,
        "cell_registry": cells,
        "world_registry": worlds,
        "raw_logits_shards": {phase: raw_shard_specs(phase) for phase in EXPECTED_COUNTS},
        "expected_counts": dict(EXPECTED_COUNTS),
        "expected_activation_rows": dict(EXPECTED_ACTIVATION_ROWS),
        "expected_cumulative_calls": dict(EXPECTED_CUMULATIVE_CALLS),
        "total_model_calls": 4144,
        "model_calls_before_plan_freeze": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }
    plan = {**plan_core, "call_plan_sha256": canonical_sha256(plan_core)}
    return plan, receipt, dependency


def design_from_plan(
    plan: Mapping[str, Any],
    *,
    receipt_path: Path = DEFAULT_TOKENIZATION_RECEIPT,
    dependency_path: Path = DEFAULT_DEPENDENCY_LOCK,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    dependency = _load_json(dependency_path)
    return {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "mode": plan["mode"],
        "claim_scope": "model_prompt_token_layer_specific_target_order_context_interference_mediation",
        "prohibited_claims": [
            "biology",
            "latent_biological_knowledge",
            "general_activation_gap",
            "physical_law",
            "model_family_generality",
        ],
        "call_plan_sha256": plan["call_plan_sha256"],
        "model": plan["model"],
        "layer_grid": list(LAYER_GRID),
        "axis_registry": list(AXIS_REGISTRY),
        "conditions": list(ALL_PATCH_CONDITIONS),
        "expected_counts": dict(EXPECTED_COUNTS),
        "expected_cumulative_calls": dict(EXPECTED_CUMULATIVE_CALLS),
        "locks": {
            **plan["locks"],
            "dependency_lock": {
                "path": str(dependency_path),
                "file_sha256": file_sha256(dependency_path),
                "canonical_sha256": dependency["canonical_sha256"],
            },
            "tokenization_receipt": {
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "canonical_sha256": receipt["canonical_sha256"],
            },
        },
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }


def validate_plan(plan: Mapping[str, Any], design: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA or design.get("schema_version") != DESIGN_SCHEMA:
        raise PositionalActivationRunnerError("plan/design schema changed")
    core = {key: value for key, value in plan.items() if key != "call_plan_sha256"}
    if canonical_sha256(core) != plan.get("call_plan_sha256"):
        raise PositionalActivationRunnerError("call-plan canonical hash changed")
    if design.get("call_plan_sha256") != plan.get("call_plan_sha256"):
        raise PositionalActivationRunnerError("plan/design hash binding changed")
    if plan.get("expected_counts") != EXPECTED_COUNTS or plan.get("total_model_calls") != 4144:
        raise PositionalActivationRunnerError("predeclared call count changed")
    if len(plan.get("baseline_templates", [])) != 1792 or len(plan.get("patch_templates", [])) != 2352:
        raise PositionalActivationRunnerError("plan template counts changed")
    if plan.get("model_calls_before_plan_freeze") != 0 or design.get("model_calls") != 0:
        raise PositionalActivationRunnerError("pre-forward artifact reports model calls")
    if plan.get("generation_used") is not False or design.get("generation_used") is not False:
        raise PositionalActivationRunnerError("plan permits generation")


def _all_downstream_paths() -> list[Path]:
    paths = [path for phase_paths in PHASE_PATHS.values() for path in phase_paths.values()]
    paths.extend(
        [
            DEFAULT_BASIS_LOCK,
            DEFAULT_BASIS_SIDECAR,
            DEFAULT_LOCALIZATION_BASELINE_ENTRY,
            DEFAULT_LOCALIZATION_PATCH_ENTRY,
            DEFAULT_LAYER_LOCK,
            DEFAULT_HOLDOUT_BASELINE_ENTRY,
            DEFAULT_HOLDOUT_PATCH_ENTRY,
            RESULT_ROOT / "fit_analysis.json",
            RESULT_ROOT / "localization_baseline_analysis.json",
            RESULT_ROOT / "localization_analysis.json",
            RESULT_ROOT / "holdout_baseline_analysis.json",
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ]
    )
    return paths


def run_plan(analyzer_path: Path = DEFAULT_ANALYZER) -> None:
    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise PositionalActivationRunnerError("plan requires the default V5 analyzer path")
    existing = [path for path in _all_downstream_paths() if path.exists()]
    raw_existing = list((RESULT_ROOT / "raw_logits").glob("**/*.npy"))
    if existing or raw_existing:
        raise PositionalActivationRunnerError(
            f"refusing to freeze plan after execution artifact: {(existing or raw_existing)[0]}"
        )
    _verify_cached_model_assets()
    tokenizer = _load_tokenizer()
    plan, receipt, dependency = build_plan(tokenizer, analyzer_path)
    write_json(DEFAULT_TOKENIZATION_RECEIPT, receipt)
    write_json(DEFAULT_DEPENDENCY_LOCK, dependency)
    design = design_from_plan(plan)
    validate_plan(plan, design)
    write_json(DEFAULT_DESIGN, design)
    manifest = {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": "PLAN_AND_DESIGN_FROZEN_NO_FORWARD",
        "plan": plan,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_path": str(DEFAULT_DESIGN),
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "tokenization_receipt_file_sha256": file_sha256(DEFAULT_TOKENIZATION_RECEIPT),
        "dependency_lock_file_sha256": file_sha256(DEFAULT_DEPENDENCY_LOCK),
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }
    write_json(DEFAULT_PLAN_MANIFEST, manifest)


def _load_frozen_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(DEFAULT_PLAN_MANIFEST)
    design = _load_json(DEFAULT_DESIGN)
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise PositionalActivationRunnerError("plan manifest has no plan object")
    validate_plan(plan, design)
    if (
        manifest.get("schema_version") != PLAN_MANIFEST_SCHEMA
        or manifest.get("status") != "PLAN_AND_DESIGN_FROZEN_NO_FORWARD"
        or manifest.get("call_plan_sha256") != plan["call_plan_sha256"]
        or manifest.get("design_file_sha256") != file_sha256(DEFAULT_DESIGN)
        or manifest.get("tokenization_receipt_file_sha256") != file_sha256(DEFAULT_TOKENIZATION_RECEIPT)
        or manifest.get("dependency_lock_file_sha256") != file_sha256(DEFAULT_DEPENDENCY_LOCK)
        or manifest.get("model_calls") != 0
        or manifest.get("generation_used") is not False
    ):
        raise PositionalActivationRunnerError("frozen plan-manifest binding changed")
    locks = design.get("locks")
    if not isinstance(locks, Mapping):
        raise PositionalActivationRunnerError("design locks are missing")
    dependency_receipt = locks.get("dependency_lock")
    tokenization_receipt = locks.get("tokenization_receipt")
    receipt_values: dict[str, dict[str, Any]] = {}
    for label, item in (
        ("dependency lock", dependency_receipt),
        ("tokenization receipt", tokenization_receipt),
    ):
        if not isinstance(item, Mapping):
            raise PositionalActivationRunnerError(f"{label} receipt is missing")
        path = Path(str(item.get("path", "")))
        value = _load_json(path)
        if file_sha256(path) != item.get("file_sha256") or value.get("canonical_sha256") != item.get(
            "canonical_sha256"
        ):
            raise PositionalActivationRunnerError(f"{label} changed")
        core = {key: candidate for key, candidate in value.items() if key != "canonical_sha256"}
        if canonical_sha256(core) != value.get("canonical_sha256"):
            raise PositionalActivationRunnerError(f"{label} canonical hash changed")
        receipt_values[label] = value
    implementation_files = receipt_values["dependency lock"].get("implementation_files")
    _require_current_dependency_lock(receipt_values["dependency lock"])
    if not isinstance(implementation_files, Mapping) or not implementation_files:
        raise PositionalActivationRunnerError("dependency implementation registry is missing")
    for name, expected in implementation_files.items():
        item = locks.get(name)
        if not isinstance(expected, Mapping) or item != expected:
            raise PositionalActivationRunnerError(f"design lock differs from dependency lock: {name}")
        path = Path(str(expected.get("path", "")))
        if not path.is_file() or file_sha256(path) != expected.get("sha256"):
            raise PositionalActivationRunnerError(f"frozen implementation changed: {name}")
    return plan, design


def _verify_cached_model_assets() -> None:
    """Reuse the audited V2 asset verifier without importing its runner state."""

    try:
        from . import run_coherent_readout_v2_causal_binding as v2
        from . import run_coherent_readout_v2_syntax as syntax
    except ImportError:
        import run_coherent_readout_v2_causal_binding as v2
        import run_coherent_readout_v2_syntax as syntax
    syntax.verify_cached_model_weights(MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256)
    for filename, digest in (
        ("config.json", MODEL_CONFIG_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("tokenizer.json", TOKENIZER_JSON_SHA256),
    ):
        path = v2._verify_cached_model_asset(filename, digest)
        if not path.is_file():
            raise PositionalActivationRunnerError(f"cached model asset is missing: {filename}")


def validate_loaded_model(model: Any) -> None:
    try:
        import torch
    except ImportError as error:
        raise PositionalActivationRunnerError("torch is required for model validation") from error
    layers = model_hooks.resolve_decoder_layers(model)
    config = getattr(model, "config", None)
    if len(layers) != MODEL_LAYERS:
        raise PositionalActivationRunnerError("loaded decoder-layer count changed")
    if getattr(config, "hidden_size", None) != MODEL_WIDTH:
        raise PositionalActivationRunnerError("loaded hidden width changed")
    if getattr(config, "vocab_size", None) != MODEL_VOCAB_SIZE:
        raise PositionalActivationRunnerError("loaded vocabulary size changed")
    if getattr(config, "_attn_implementation", None) != ATTENTION_IMPLEMENTATION:
        raise PositionalActivationRunnerError("loaded attention implementation changed")
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or {tensor.device.type for tensor in tensors} != {DEVICE}:
        raise PositionalActivationRunnerError("loaded tensors are not all on MPS")
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating or {tensor.dtype for tensor in floating} != {torch.float32}:
        raise PositionalActivationRunnerError("loaded tensors are not all float32")


def _load_model() -> Any:
    _verify_cached_model_assets()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise PositionalActivationRunnerError("torch and transformers are required") from error
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        .to(torch.device(DEVICE))
        .eval()
    )
    validate_loaded_model(model)
    return model


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise PositionalActivationRunnerError("model has no parameter device") from error


def _inputs(prompt: Mapping[str, Any], model: Any) -> dict[str, Any]:
    import torch

    ids = prompt.get("execution_input_ids")
    mask = prompt.get("execution_attention_mask")
    if not isinstance(ids, list) or not ids or mask != [1] * len(ids) or prompt.get("input_token_count") != len(ids):
        raise PositionalActivationRunnerError("stored execution prompt is invalid")
    device = _model_device(model)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
    }


def execution_input_sha256(prompt: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "coherent-readout-v5-execution-input-v1",
            "input_ids": prompt["execution_input_ids"],
            "attention_mask": prompt["execution_attention_mask"],
        }
    )


def full_vocab_diagnostics(
    row: np.ndarray,
    *,
    expected_answer: str,
    expected_token_id: int,
    distractor_answer: str,
    distractor_token_id: int,
) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise PositionalActivationRunnerError("model returned an invalid full-vocabulary row")
    if not 0 <= expected_token_id < len(value) or not 0 <= distractor_token_id < len(value):
        raise PositionalActivationRunnerError("answer token ID is outside vocabulary")
    expected_logit = float(value[expected_token_id])
    distractor_logit = float(value[distractor_token_id])
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    peak = float(value.astype(np.float64).max())
    logsumexp = peak + math.log(float(np.exp(value.astype(np.float64) - peak).sum()))
    label_logsumexp = float(np.logaddexp(expected_logit, distractor_logit))
    if expected_logit > distractor_logit:
        predicted = expected_answer
        predicted_id = expected_token_id
    elif distractor_logit > expected_logit:
        predicted = distractor_answer
        predicted_id = distractor_token_id
    else:
        predicted = None
        predicted_id = None
    return {
        "expected_logit": expected_logit,
        "distractor_logit": distractor_logit,
        "expected_minus_distractor_margin": expected_logit - distractor_logit,
        "predicted_answer": predicted,
        "predicted_token_id": predicted_id,
        "answer_correct": expected_logit > distractor_logit,
        "answer_tie": expected_logit == distractor_logit,
        "greedy_token_id": maximum_ids[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "full_vocab_logits_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _baseline_forward(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    layers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    import torch

    target_index = prompt.get("target_property_token_index")
    if target_index is None:
        if layers:
            raise PositionalActivationRunnerError("non-composition baseline cannot capture layers")
        captures: list[Any] = []
    else:
        if tuple(layers) not in (tuple(LAYER_GRID),) and len(layers) != 1:
            raise PositionalActivationRunnerError("baseline capture layer registry changed")
        if any(layer not in LAYER_GRID for layer in layers):
            raise PositionalActivationRunnerError("baseline capture layer is outside grid")
        captures = [model_hooks.ResidualStreamCapture(model, layer, token_index=int(target_index)) for layer in layers]
    with torch.inference_mode(), ExitStack() as stack:
        for capture in captures:
            stack.enter_context(capture)
        output = model(**_inputs(prompt, model), use_cache=False, return_dict=True)
    counts = [len(capture.values) for capture in captures]
    if counts != [1] * len(captures) or any(capture.active for capture in captures):
        raise PositionalActivationRunnerError("baseline capture count or cleanup failed")
    activations = None
    if captures:
        activations = np.ascontiguousarray(
            np.stack([capture.values[0][0].detach().float().cpu().numpy() for capture in captures]),
            dtype="<f4",
        )
        if activations.shape != (len(layers), MODEL_WIDTH) or not np.isfinite(activations).all():
            raise PositionalActivationRunnerError("captured activation shape changed")
    logits = np.ascontiguousarray(output.logits[0, -1, :].detach().float().cpu().numpy(), dtype="<f4")
    if logits.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(logits).all():
        raise PositionalActivationRunnerError("baseline logits shape changed")
    trace = {
        "use_cache": False,
        "return_dict": True,
        "generation_used": False,
        "teacher_forced_prompt_forward": True,
        "capture_layers": [int(layer) for layer in layers],
        "capture_counts": counts,
        "captures_removed": not any(capture.active for capture in captures),
        "hook_site": "resid_post",
        "token_site": "target_property_token" if target_index is not None else None,
        "token_index": target_index,
        "model_calls": 1,
    }
    return logits, activations, trace


def expected_projected_patch(
    recipient: np.ndarray,
    source: np.ndarray,
    direction: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    recipient64 = np.asarray(recipient, dtype=np.float64)
    source64 = np.asarray(source, dtype=np.float64)
    direction64 = np.asarray(direction, dtype=np.float64)
    if recipient64.shape != source64.shape or recipient64.shape != direction64.shape:
        raise PositionalActivationRunnerError("projected-patch vectors differ in shape")
    norm = float(np.linalg.norm(direction64))
    if not np.isfinite(norm) or norm == 0.0 or not 0.0 <= float(strength) <= 1.0:
        raise PositionalActivationRunnerError("projected-patch direction/strength is invalid")
    unit = direction64 / norm
    delta = float(unit @ (source64 - recipient64))
    return np.ascontiguousarray(recipient64 + float(strength) * delta * unit, dtype="<f4")


def expected_directional_displacement(recipient: np.ndarray, direction: np.ndarray, scalar: float) -> np.ndarray:
    """Apply a signed, norm-matched displacement along a unit-normalized axis."""

    recipient64 = np.asarray(recipient, dtype=np.float64)
    direction64 = np.asarray(direction, dtype=np.float64)
    scalar = float(scalar)
    if recipient64.shape != direction64.shape or not math.isfinite(scalar):
        raise PositionalActivationRunnerError("directional displacement inputs are invalid")
    norm = float(np.linalg.norm(direction64))
    if not np.isfinite(norm) or norm == 0.0:
        raise PositionalActivationRunnerError("directional displacement axis is invalid")
    return np.ascontiguousarray(recipient64 + scalar * direction64 / norm, dtype="<f4")


def _patch_forward(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    layer: int,
    recipient_activation: np.ndarray,
    source_activation: np.ndarray,
    direction: np.ndarray,
    signed_scalar: float,
    positional_scalar_d: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    target_index = prompt.get("target_property_token_index")
    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise PositionalActivationRunnerError("patch recipient lacks TARGET token index")
    recipient = np.ascontiguousarray(np.asarray(recipient_activation, dtype="<f4"))
    source = np.ascontiguousarray(np.asarray(source_activation, dtype="<f4"))
    axis = np.ascontiguousarray(np.asarray(direction, dtype="<f4"))
    if any(value.shape != (MODEL_WIDTH,) or not np.isfinite(value).all() for value in (recipient, source, axis)):
        raise PositionalActivationRunnerError("patch vector shape or finiteness changed")
    signed_scalar = float(signed_scalar)
    positional_scalar_d = float(positional_scalar_d)
    if not math.isfinite(signed_scalar) or not math.isfinite(positional_scalar_d):
        raise PositionalActivationRunnerError("patch displacement scalar is not finite")
    expected = expected_directional_displacement(recipient, axis, signed_scalar)
    base_transform = model_hooks.steering_transform(
        axis,
        signed_scalar,
        token_index=target_index,
        dose_scale=1.0,
    )
    trace: dict[str, Any] = {
        "hook_calls": 0,
        "hook_removed": False,
        "non_target_tokens_unchanged": True,
        "pre_activation_matches_registered_recipient": True,
        "post_activation_matches_expected": True,
    }
    post_holder: dict[str, np.ndarray] = {}

    def traced_transform(hidden: Any) -> Any:
        trace["hook_calls"] += 1
        pre = hidden[:, target_index, :].detach().float().cpu().numpy()[0].astype("<f4", copy=False)
        changed = base_transform(hidden)
        post = changed[:, target_index, :].detach().float().cpu().numpy()[0].astype("<f4", copy=False)
        post_holder["value"] = np.ascontiguousarray(post.copy(), dtype="<f4")
        before_equal = torch.equal(changed[:, :target_index, :], hidden[:, :target_index, :])
        after_equal = torch.equal(changed[:, target_index + 1 :, :], hidden[:, target_index + 1 :, :])
        trace["non_target_tokens_unchanged"] = bool(
            trace["non_target_tokens_unchanged"] and before_equal and after_equal
        )
        trace["pre_activation_matches_registered_recipient"] = bool(
            trace["pre_activation_matches_registered_recipient"] and f32_sha256(pre) == f32_sha256(recipient)
        )
        error = float(np.linalg.norm(post.astype(np.float64) - expected.astype(np.float64)))
        tolerance = NUMERICAL_TOLERANCE * max(1.0, float(np.linalg.norm(expected.astype(np.float64))))
        trace["post_activation_matches_expected"] = bool(
            trace["post_activation_matches_expected"] and error <= tolerance
        )
        trace["pre_activation_sha256"] = f32_sha256(pre)
        trace["post_activation_sha256"] = f32_sha256(post)
        trace["expected_activation_sha256"] = f32_sha256(expected)
        trace["post_expected_l2_error"] = error
        trace["post_expected_l2_tolerance"] = tolerance
        unit = axis.astype(np.float64) / np.linalg.norm(axis.astype(np.float64))
        displacement = post.astype(np.float64) - pre.astype(np.float64)
        orthogonal = displacement - unit * float(unit @ displacement)
        trace["orthogonal_displacement_l2"] = float(np.linalg.norm(orthogonal))
        trace["displacement_l2"] = float(np.linalg.norm(displacement))
        trace["positional_scalar_d"] = positional_scalar_d
        trace["applied_signed_scalar"] = signed_scalar
        trace["registered_positional_dose"] = signed_scalar
        trace["expected_displacement_l2"] = abs(signed_scalar)
        trace["observed_displacement_l2"] = trace["displacement_l2"]
        trace["applied_displacement_l2"] = trace["displacement_l2"]
        trace["displacement_abs_scalar_error"] = abs(trace["displacement_l2"] - abs(signed_scalar))
        trace["displacement_abs_scalar_tolerance"] = DOSE_MATCH_TOLERANCE * max(1.0, abs(signed_scalar))
        trace["pre_axis_coefficient"] = float(unit @ pre.astype(np.float64))
        trace["source_axis_coefficient"] = float(unit @ source.astype(np.float64))
        trace["post_axis_coefficient"] = float(unit @ post.astype(np.float64))
        return changed

    intervention = model_hooks.ResidualStreamIntervention(model, layer, traced_transform)
    with torch.inference_mode(), intervention:
        output = model(**_inputs(prompt, model), use_cache=False, return_dict=True)
    trace.update(
        {
            "hook_removed": not intervention.active,
            "operation": "projected_patch",
            "layer": layer,
            "token_index": target_index,
            "strength": 1.0,
            "model_calls": 1,
            "generation_used": False,
        }
    )
    required = (
        trace["hook_calls"] == 1,
        trace["hook_removed"],
        trace["non_target_tokens_unchanged"],
        trace["pre_activation_matches_registered_recipient"],
        trace["post_activation_matches_expected"],
        trace["orthogonal_displacement_l2"] <= NUMERICAL_TOLERANCE * max(1.0, trace["displacement_l2"]),
        trace["displacement_abs_scalar_error"] <= trace["displacement_abs_scalar_tolerance"],
    )
    if not all(required):
        raise PositionalActivationRunnerError("intervention hook or numerical gate failed")
    logits = np.ascontiguousarray(output.logits[0, -1, :].detach().float().cpu().numpy(), dtype="<f4")
    post = post_holder.get("value")
    if logits.shape != (MODEL_VOCAB_SIZE,) or post is None or post.shape != (MODEL_WIDTH,):
        raise PositionalActivationRunnerError("patched output shape changed")
    return logits, post, trace


class RawLogitShardWriter:
    """Write the predeclared raw full-vocabulary shards without holding a phase in RAM."""

    def __init__(self, phase: str, specs: Sequence[Mapping[str, Any]]):
        self.phase = phase
        self.specs = [dict(spec) for spec in specs]
        self.next_row = 0
        self._buffer: list[np.ndarray] = []
        self._receipts: list[dict[str, Any]] = []
        if self.specs != raw_shard_specs(phase):
            raise PositionalActivationRunnerError("raw-logit shard plan changed")
        existing = [Path(spec["path"]) for spec in self.specs if Path(spec["path"]).exists()]
        if existing:
            raise PositionalActivationRunnerError(f"raw-logit shard already exists: {existing[0]}")

    def append(self, row: np.ndarray) -> dict[str, Any]:
        if self.next_row >= EXPECTED_COUNTS[self.phase]:
            raise PositionalActivationRunnerError("too many raw-logit rows")
        value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
        if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
            raise PositionalActivationRunnerError("invalid raw-logit row")
        shard_index = self.next_row // RAW_SHARD_ROWS
        spec = self.specs[shard_index]
        row_in_shard = self.next_row - int(spec["start_row"])
        binding = {
            "raw_logits_global_row": self.next_row,
            "raw_logits_shard_index": shard_index,
            "raw_logits_row_in_shard": row_in_shard,
            "raw_logits_row_sha256": f32_sha256(value),
        }
        self._buffer.append(value.copy())
        self.next_row += 1
        if self.next_row == int(spec["stop_row"]):
            matrix = np.ascontiguousarray(np.stack(self._buffer), dtype="<f4")
            if list(matrix.shape) != spec["shape"]:
                raise PositionalActivationRunnerError("raw-logit shard shape changed")
            path = Path(spec["path"])
            write_array(path, matrix)
            self._receipts.append(
                {
                    **spec,
                    "file_sha256": file_sha256(path),
                    "logical_sha256": f32_sha256(matrix),
                    "size_bytes": path.stat().st_size,
                }
            )
            self._buffer = []
        return binding

    def finalize(self) -> list[dict[str, Any]]:
        if self.next_row != EXPECTED_COUNTS[self.phase] or self._buffer:
            raise PositionalActivationRunnerError("raw-logit phase ended at wrong row count")
        if len(self._receipts) != len(self.specs):
            raise PositionalActivationRunnerError("raw-logit shard receipt count changed")
        return [dict(receipt) for receipt in self._receipts]


def _cell_stratum(cell: Mapping[str, Any]) -> str:
    explicit = cell.get("stratum_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    factors = cell.get("factors")
    if isinstance(factors, Mapping) and all(factors.get(name) in (-1, 1) for name in ("p", "m", "r", "v", "o")):
        return ":".join(f"{name}{int(factors[name]):+d}" for name in ("p", "m", "r", "v", "o"))
    return _family(cell)


def _baseline_record(
    plan: Mapping[str, Any],
    template: Mapping[str, Any],
    prompt: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    phase: str,
    call_index: int,
    raw_binding: Mapping[str, Any],
    activation_row: int | None,
    layers: Sequence[int],
    activations: np.ndarray | None,
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": BASELINE_SCHEMA,
        "phase": phase,
        "role": _normal_role(cell.get("role")),
        "call_index": call_index,
        "world_id": cell["world_id"],
        "cell_id": cell["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "call_plan_sha256": plan["call_plan_sha256"],
    }
    activation_sha = None
    layer_hashes: dict[str, str] = {}
    if activations is not None:
        activation_sha = f32_sha256(activations)
        layer_hashes = {str(layer): f32_sha256(activations[index]) for index, layer in enumerate(layers)}
    factors = cell.get("factors")
    factor_value = dict(factors) if isinstance(factors, Mapping) else None
    pair_id = cell.get("semantic_pair_id", cell.get("pair_id"))
    record_id = canonical_sha256(identity)
    return {
        **identity,
        "record_id": record_id,
        "baseline_id": record_id,
        "template_id": template["template_id"],
        "family": _family(cell),
        "stratum_id": _cell_stratum(cell),
        "intervention_prerequisite": bool(
            cell.get("intervention_prerequisite", _family(cell) in {"retrieval", "lookup"})
        ),
        "expected_answer": prompt["expected_answer"],
        "expected_token_id": prompt["expected_token_id"],
        "distractor_answer": prompt["distractor_answer"],
        "distractor_token_id": prompt["distractor_token_id"],
        "factors": factor_value,
        "factor_levels": cell.get("factor_levels"),
        "order": _order(cell),
        "pair_id": pair_id,
        "mate_cell_id": cell.get("mate_cell_id"),
        "target_property": cell.get("target_property"),
        "target_token_index": prompt["target_property_token_index"],
        "execution_input_sha256": execution_input_sha256(prompt),
        **dict(raw_binding),
        "activation_row": activation_row,
        "captured_layers": [int(layer) for layer in layers],
        "activation_sha256": activation_sha,
        "activation_layer_sha256": layer_hashes,
        "diagnostics": dict(diagnostics),
        "trace": dict(trace),
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": file_sha256(FROZEN_PREREGISTRATION),
        "generation_used": False,
        "biological_model_calls": 0,
    }


def _patch_record(
    plan: Mapping[str, Any],
    template: Mapping[str, Any],
    recipient_cell: Mapping[str, Any],
    recipient_prompt: Mapping[str, Any],
    *,
    phase: str,
    call_index: int,
    layer: int,
    raw_binding: Mapping[str, Any],
    patched_activation_row: int,
    recipient_baseline: Mapping[str, Any],
    source_baseline: Mapping[str, Any],
    recipient_activation: np.ndarray,
    source_activation: np.ndarray,
    direction: np.ndarray,
    patched_activation: np.ndarray,
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": PATCH_SCHEMA,
        "phase": phase,
        "role": template["role"],
        "call_index": call_index,
        "world_id": template["world_id"],
        "pair_id": template["pair_id"],
        "template_id": template["template_id"],
        "condition": template["condition"],
        "layer": layer,
        "recipient_cell_id": template["recipient_cell_id"],
        "source_cell_id": template["source_cell_id"],
        "call_plan_sha256": plan["call_plan_sha256"],
    }
    result_trace = dict(trace)
    if template["condition"] == "identity":
        before = recipient_baseline["diagnostics"]
        result_trace.update(
            {
                "identity_expected_logit_abs_error": abs(
                    float(diagnostics["expected_logit"]) - float(before["expected_logit"])
                ),
                "identity_distractor_logit_abs_error": abs(
                    float(diagnostics["distractor_logit"]) - float(before["distractor_logit"])
                ),
                "identity_global_argmax_preserved": (diagnostics["maximum_token_ids"] == before["maximum_token_ids"]),
            }
        )
    else:
        result_trace.update(
            {
                "identity_expected_logit_abs_error": None,
                "identity_distractor_logit_abs_error": None,
                "identity_global_argmax_preserved": None,
            }
        )
    record_id = canonical_sha256(identity)
    return {
        **identity,
        "record_id": record_id,
        "patch_id": record_id,
        "operation": template["operation"],
        "direction_name": template["direction_name"],
        "recipient_order": template["recipient_order"],
        "source_order": template["source_order"],
        "cell_id": template["recipient_cell_id"],
        "prompt_id": recipient_prompt["prompt_id"],
        "family": "composition",
        "stratum_id": _cell_stratum(recipient_cell),
        "expected_answer": recipient_prompt["expected_answer"],
        "expected_token_id": recipient_prompt["expected_token_id"],
        "distractor_answer": recipient_prompt["distractor_answer"],
        "distractor_token_id": recipient_prompt["distractor_token_id"],
        "factors": dict(recipient_cell["factors"]),
        "order": _order(recipient_cell),
        "target_token_index": recipient_prompt["target_property_token_index"],
        "execution_input_sha256": execution_input_sha256(recipient_prompt),
        **dict(raw_binding),
        "patched_activation_row": patched_activation_row,
        "recipient_baseline_id": recipient_baseline["baseline_id"],
        "recipient_activation_row": recipient_baseline["activation_row"],
        "source_baseline_id": source_baseline["baseline_id"],
        "source_activation_row": source_baseline["activation_row"],
        "recipient_activation_sha256": f32_sha256(recipient_activation),
        "source_activation_sha256": f32_sha256(source_activation),
        "applied_direction_sha256": f32_sha256(direction),
        "patched_activation_sha256": f32_sha256(patched_activation),
        "positional_scalar_d": float(result_trace["positional_scalar_d"]),
        "applied_signed_scalar": float(result_trace["applied_signed_scalar"]),
        "registered_positional_dose": float(result_trace["registered_positional_dose"]),
        "expected_displacement_l2": float(result_trace["expected_displacement_l2"]),
        "observed_displacement_l2": float(result_trace["observed_displacement_l2"]),
        "applied_displacement_l2": float(result_trace["applied_displacement_l2"]),
        "displacement_abs_scalar_error": float(result_trace["displacement_abs_scalar_error"]),
        "baseline_margin": float(recipient_baseline["diagnostics"]["expected_minus_distractor_margin"]),
        "diagnostics": dict(diagnostics),
        "trace": result_trace,
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": file_sha256(FROZEN_PREREGISTRATION),
        "generation_used": False,
        "biological_model_calls": 0,
    }


def _phase_baseline_templates(plan: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in plan["baseline_templates"] if row["role"] == role]
    expected = ROLE_WORLD_COUNTS[role] * BASELINES_PER_WORLD
    if len(rows) != expected:
        raise PositionalActivationRunnerError("phase baseline-template count changed")
    return sorted(rows, key=lambda row: str(row["cell_id"]))


def _phase_patch_templates(plan: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in plan["patch_templates"] if row["role"] == role]
    expected = EXPECTED_COUNTS[f"{role}-patch"]
    if len(rows) != expected:
        raise PositionalActivationRunnerError("phase patch-template count changed")
    return rows


def _execute_baselines(
    model: Any,
    plan: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    templates = _phase_baseline_templates(plan, role)
    prompts = {str(row["cell_id"]): row for row in plan["prompts"]}
    cells = {str(row["cell_id"]): row for row in plan["cell_registry"]}
    writer = RawLogitShardWriter(phase, plan["raw_logits_shards"][phase])
    records: list[dict[str, Any]] = []
    activation_rows: list[np.ndarray] = []
    for call_index, template in enumerate(templates):
        cell_id = str(template["cell_id"])
        cell = cells.get(cell_id)
        prompt = prompts.get(cell_id)
        if cell is None or prompt is None or _baseline_template(cell, prompt) != template:
            raise PositionalActivationRunnerError("baseline template does not resolve exactly")
        capture_layers = tuple(layers) if template["family"] == "composition" else ()
        logits, activation, trace = _baseline_forward(model, prompt, layers=capture_layers)
        diagnostics = full_vocab_diagnostics(
            logits,
            expected_answer=prompt["expected_answer"],
            expected_token_id=int(prompt["expected_token_id"]),
            distractor_answer=prompt["distractor_answer"],
            distractor_token_id=int(prompt["distractor_token_id"]),
        )
        raw_binding = writer.append(logits)
        activation_row = None
        if activation is not None:
            activation_row = len(activation_rows)
            activation_rows.append(activation)
        records.append(
            _baseline_record(
                plan,
                template,
                prompt,
                cell,
                phase=phase,
                call_index=call_index,
                raw_binding=raw_binding,
                activation_row=activation_row,
                layers=capture_layers,
                activations=activation,
                diagnostics=diagnostics,
                trace=trace,
            )
        )
        if (call_index + 1) % 32 == 0 or call_index + 1 == len(templates):
            print(canonical_json({"phase": phase, "completed": call_index + 1, "total": len(templates)}), flush=True)
    raw_receipts = writer.finalize()
    matrix = np.ascontiguousarray(np.stack(activation_rows), dtype="<f4")
    expected_rows = EXPECTED_ACTIVATION_ROWS[phase]
    expected_shape = (expected_rows, len(layers), MODEL_WIDTH)
    if matrix.shape != expected_shape:
        raise PositionalActivationRunnerError("baseline activation sidecar shape changed")
    if role == "holdout":
        matrix = np.ascontiguousarray(matrix[:, 0, :], dtype="<f4")
    if any(
        record["activation_row"] != index
        for index, record in enumerate(row for row in records if row["family"] == "composition")
    ):
        raise PositionalActivationRunnerError("baseline activation row registry changed")
    return records, matrix, raw_receipts


def _activation_for_record(record: Mapping[str, Any], activations: np.ndarray, *, layer: int, role: str) -> np.ndarray:
    row = record.get("activation_row")
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(activations):
        raise PositionalActivationRunnerError("baseline activation row is invalid")
    if role == "localization":
        if activations.ndim != 3 or layer not in LAYER_GRID:
            raise PositionalActivationRunnerError("localization activation sidecar changed")
        value = activations[row, LAYER_GRID.index(layer)]
    elif role == "holdout":
        if activations.ndim != 2:
            raise PositionalActivationRunnerError("holdout activation sidecar changed")
        value = activations[row]
    else:
        raise PositionalActivationRunnerError("invalid patch role")
    value = np.ascontiguousarray(value, dtype="<f4")
    if record.get("activation_layer_sha256", {}).get(str(layer)) != f32_sha256(value):
        raise PositionalActivationRunnerError("activation vector differs from record hash")
    return value


def _load_basis(plan: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    lock = _load_json(DEFAULT_BASIS_LOCK)
    if (
        lock.get("status") != "FIT_BASIS_ADMITTED"
        or lock.get("admitted") is not True
        or lock.get("call_plan_sha256") != plan["call_plan_sha256"]
        or lock.get("layers") != list(LAYER_GRID)
        or lock.get("axis_registry") != list(AXIS_REGISTRY)
    ):
        raise PositionalActivationRunnerError("fit basis was not admitted")
    sidecar = lock.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise PositionalActivationRunnerError("fit basis lock lacks sidecar receipt")
    path = Path(str(sidecar.get("path", "")))
    if path.resolve() != DEFAULT_BASIS_SIDECAR.resolve() or not path.is_file():
        raise PositionalActivationRunnerError("fit basis sidecar path changed")
    if file_sha256(path) != sidecar.get("file_sha256"):
        raise PositionalActivationRunnerError("fit basis sidecar file hash changed")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PositionalActivationRunnerError("cannot load fit basis sidecar") from error
    basis = np.ascontiguousarray(value, dtype="<f4")
    if (
        basis.shape != (len(LAYER_GRID), len(AXIS_REGISTRY), MODEL_WIDTH)
        or not np.isfinite(basis).all()
        or list(basis.shape) != sidecar.get("shape")
        or sidecar.get("dtype") != "<f4"
        or f32_sha256(basis) != sidecar.get("logical_sha256")
    ):
        raise PositionalActivationRunnerError("fit basis logical receipt changed")
    for layer_position in range(len(LAYER_GRID)):
        for axis_position in range(1, len(AXIS_REGISTRY)):
            norm = float(np.linalg.norm(basis[layer_position, axis_position].astype(np.float64)))
            if not np.isclose(norm, 1.0, atol=1e-5, rtol=1e-5):
                raise PositionalActivationRunnerError("fit direction is not unit norm")
    return basis, lock


def _basis_direction(basis: np.ndarray, *, layer: int, direction_name: str) -> np.ndarray:
    if layer not in LAYER_GRID or direction_name not in AXIS_REGISTRY[1:]:
        raise PositionalActivationRunnerError("basis vector lookup is invalid")
    return np.ascontiguousarray(basis[LAYER_GRID.index(layer), AXIS_REGISTRY.index(direction_name)], dtype="<f4")


def _execute_patches(
    model: Any,
    plan: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    templates: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    basis: np.ndarray,
    selected_layer: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    cells = {str(row["cell_id"]): row for row in plan["cell_registry"]}
    prompts = {str(row["cell_id"]): row for row in plan["prompts"]}
    baselines = {str(row["cell_id"]): row for row in baseline_records}
    if len(baselines) != len(baseline_records):
        raise PositionalActivationRunnerError("baseline record IDs are duplicated")
    writer = RawLogitShardWriter(phase, plan["raw_logits_shards"][phase])
    records: list[dict[str, Any]] = []
    patched_rows: list[np.ndarray] = []
    for call_index, template_value in enumerate(templates):
        template = dict(template_value)
        layer = template["layer"] if role == "localization" else selected_layer
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in LAYER_GRID:
            raise PositionalActivationRunnerError("patch layer is invalid")
        recipient_id = str(template["recipient_cell_id"])
        source_id = str(template["source_cell_id"])
        recipient_cell = cells.get(recipient_id)
        recipient_prompt = prompts.get(recipient_id)
        recipient_baseline = baselines.get(recipient_id)
        source_baseline = baselines.get(source_id)
        if any(value is None for value in (recipient_cell, recipient_prompt, recipient_baseline, source_baseline)):
            raise PositionalActivationRunnerError("patch template does not resolve")
        recipient_activation = _activation_for_record(recipient_baseline, activations, layer=layer, role=role)
        source_activation = _activation_for_record(source_baseline, activations, layer=layer, role=role)
        direction = _basis_direction(basis, layer=layer, direction_name=str(template["direction_name"]))
        positional_direction = _basis_direction(basis, layer=layer, direction_name="positional").astype(np.float64)
        positional_direction /= np.linalg.norm(positional_direction)
        if template["recipient_order"] == "second":
            first_activation = source_activation
            second_activation = recipient_activation
        elif template["recipient_order"] == "first":
            first_activation = recipient_activation
            second_activation = source_activation
        else:
            raise PositionalActivationRunnerError("patch template recipient order changed")
        positional_scalar_d = float(
            (first_activation.astype(np.float64) - second_activation.astype(np.float64)) @ positional_direction
        )
        if template["condition"] == "identity":
            signed_scalar = 0.0
        elif template["condition"].endswith("rescue") or "rescue" in template["condition"]:
            signed_scalar = positional_scalar_d
        elif template["condition"].endswith("damage") or "damage" in template["condition"]:
            signed_scalar = -positional_scalar_d
        else:
            raise PositionalActivationRunnerError("patch condition has no signed-dose rule")
        logits, patched, trace = _patch_forward(
            model,
            recipient_prompt,
            layer=layer,
            recipient_activation=recipient_activation,
            source_activation=source_activation,
            direction=direction,
            signed_scalar=signed_scalar,
            positional_scalar_d=positional_scalar_d,
        )
        trace["direction_name"] = template["direction_name"]
        trace["operation"] = template["operation"]
        diagnostics = full_vocab_diagnostics(
            logits,
            expected_answer=recipient_prompt["expected_answer"],
            expected_token_id=int(recipient_prompt["expected_token_id"]),
            distractor_answer=recipient_prompt["distractor_answer"],
            distractor_token_id=int(recipient_prompt["distractor_token_id"]),
        )
        raw_binding = writer.append(logits)
        patched_activation_row = len(patched_rows)
        patched_rows.append(patched)
        records.append(
            _patch_record(
                plan,
                template,
                recipient_cell,
                recipient_prompt,
                phase=phase,
                call_index=call_index,
                layer=layer,
                raw_binding=raw_binding,
                patched_activation_row=patched_activation_row,
                recipient_baseline=recipient_baseline,
                source_baseline=source_baseline,
                recipient_activation=recipient_activation,
                source_activation=source_activation,
                direction=direction,
                patched_activation=patched,
                diagnostics=diagnostics,
                trace=trace,
            )
        )
        if (call_index + 1) % 32 == 0 or call_index + 1 == len(templates):
            print(canonical_json({"phase": phase, "completed": call_index + 1, "total": len(templates)}), flush=True)
    raw_receipts = writer.finalize()
    matrix = np.ascontiguousarray(np.stack(patched_rows), dtype="<f4")
    if matrix.shape != (EXPECTED_COUNTS[phase], MODEL_WIDTH):
        raise PositionalActivationRunnerError("patched-activation sidecar shape changed")
    return records, matrix, raw_receipts


def _prior_cumulative_calls(phase: str) -> int:
    index = tuple(EXPECTED_COUNTS).index(phase)
    return 0 if index == 0 else EXPECTED_CUMULATIVE_CALLS[tuple(EXPECTED_COUNTS)[index - 1]]


def _require_absent(paths: Iterable[Path], phase: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise PositionalActivationRunnerError(f"refusing to overwrite or resume {phase}: {existing[0]}")


def _require_no_downstream_artifacts(phase: str) -> None:
    phases = tuple(EXPECTED_COUNTS)
    index = phases.index(phase)
    later = [path for later_phase in phases[index + 1 :] for path in PHASE_PATHS[later_phase].values()]
    analysis_paths = {
        "fit-baseline": [
            DEFAULT_BASIS_LOCK,
            DEFAULT_BASIS_SIDECAR,
            DEFAULT_LOCALIZATION_BASELINE_ENTRY,
            DEFAULT_LOCALIZATION_PATCH_ENTRY,
            DEFAULT_LAYER_LOCK,
            DEFAULT_HOLDOUT_BASELINE_ENTRY,
            DEFAULT_HOLDOUT_PATCH_ENTRY,
        ],
        "localization-baseline": [
            DEFAULT_LOCALIZATION_PATCH_ENTRY,
            DEFAULT_LAYER_LOCK,
            DEFAULT_HOLDOUT_BASELINE_ENTRY,
            DEFAULT_HOLDOUT_PATCH_ENTRY,
        ],
        "localization-patch": [
            DEFAULT_LAYER_LOCK,
            DEFAULT_HOLDOUT_BASELINE_ENTRY,
            DEFAULT_HOLDOUT_PATCH_ENTRY,
        ],
        "holdout-baseline": [DEFAULT_HOLDOUT_PATCH_ENTRY],
        "holdout-patch": [],
    }
    later.extend(analysis_paths[phase])
    later.extend(
        [
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ]
    )
    later.extend(
        path for later_phase in phases[index + 1 :] for path in (RESULT_ROOT / "raw_logits" / later_phase).glob("*.npy")
    )
    existing = [path for path in later if path.exists()]
    if existing:
        raise PositionalActivationRunnerError(
            f"refusing non-monotone {phase}; downstream artifact exists: {existing[0]}"
        )


def _write_execution_attempt(
    path: Path,
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    prerequisites: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(DEFAULT_PLAN_MANIFEST),
        "runner_sha256": file_sha256(Path(__file__)),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "preregistration_sha256": file_sha256(FROZEN_PREREGISTRATION),
        "prerequisite_bindings": dict(prerequisites or {}),
        "predeclared_phase_calls": EXPECTED_COUNTS[phase],
        "model_calls_before_attempt": _prior_cumulative_calls(phase),
        "generation_used": False,
        "biological_model_calls": 0,
    }
    write_json(path, value)
    return value


def _array_receipt(
    path: Path,
    array: np.ndarray,
    *,
    logical_map: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise PositionalActivationRunnerError("sidecar is missing at manifest time")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "logical_sha256": f32_sha256(array),
        "shape": list(array.shape),
        "dtype": "<f4",
        "size_bytes": path.stat().st_size,
        "logical_id_map": dict(logical_map),
    }


def _execution_manifest(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    raw_receipts: Sequence[Mapping[str, Any]],
    sidecar_path: Path,
    sidecar: np.ndarray,
    prerequisites: Mapping[str, str] | None,
    selected_layer: int | None,
) -> dict[str, Any]:
    paths = PHASE_PATHS[phase]
    if phase.endswith("baseline"):
        logical_map = {
            str(record["cell_id"]): {
                "activation_row": record["activation_row"],
                "activation_sha256": record["activation_sha256"],
                "activation_layer_sha256": record["activation_layer_sha256"],
            }
            for record in records
            if record["activation_row"] is not None
        }
        activations = _array_receipt(sidecar_path, sidecar, logical_map=logical_map)
        patched_activations = None
    else:
        logical_map = {
            str(record["record_id"]): {
                "patched_activation_row": record["patched_activation_row"],
                "patched_activation_sha256": record["patched_activation_sha256"],
            }
            for record in records
        }
        activations = None
        patched_activations = _array_receipt(sidecar_path, sidecar, logical_map=logical_map)
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "status": "EXECUTION_COMPLETE_NOT_ANALYZED",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(DEFAULT_PLAN_MANIFEST),
        "runner_sha256": file_sha256(Path(__file__)),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "preregistration_sha256": file_sha256(FROZEN_PREREGISTRATION),
        "attempt": {
            "path": str(paths["attempt"]),
            "file_sha256": file_sha256(paths["attempt"]),
        },
        "records": {
            "path": str(paths["records"]),
            "file_sha256": file_sha256(paths["records"]),
            "canonical_sha256": canonical_sha256([dict(record) for record in records]),
            "count": len(records),
            "size_bytes": paths["records"].stat().st_size,
        },
        "raw_logits_shards": [dict(receipt) for receipt in raw_receipts],
        "activations": activations,
        "patched_activations": patched_activations,
        "prerequisite_bindings": dict(prerequisites or {}),
        "selected_layer": selected_layer,
        "phase_model_calls": len(records),
        "cumulative_model_calls": EXPECTED_CUMULATIVE_CALLS[phase],
        "generation_used": False,
        "biological_model_calls": 0,
        "partial_resume_allowed": False,
    }


def _write_completed_phase(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    raw_receipts: Sequence[Mapping[str, Any]],
    sidecar: np.ndarray,
    prerequisites: Mapping[str, str] | None = None,
    selected_layer: int | None = None,
) -> None:
    paths = PHASE_PATHS[phase]
    sidecar_key = "activations" if phase.endswith("baseline") else "patched_activations"
    write_jsonl(paths["records"], records)
    write_array(paths[sidecar_key], sidecar)
    manifest = _execution_manifest(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar_path=paths[sidecar_key],
        sidecar=sidecar,
        prerequisites=prerequisites,
        selected_layer=selected_layer,
    )
    write_json(paths["manifest"], manifest)


def _load_array_from_receipt(receipt: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(receipt.get("path", "")))
    if not path.is_file() or file_sha256(path) != receipt.get("file_sha256"):
        raise PositionalActivationRunnerError("sidecar file receipt changed")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PositionalActivationRunnerError("cannot load sidecar") from error
    array = np.ascontiguousarray(value, dtype="<f4")
    if (
        list(array.shape) != receipt.get("shape")
        or receipt.get("dtype") != "<f4"
        or f32_sha256(array) != receipt.get("logical_sha256")
        or path.stat().st_size != receipt.get("size_bytes")
    ):
        raise PositionalActivationRunnerError("sidecar logical receipt changed")
    return array


def _validate_execution_manifest(
    phase: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    plan, design = _load_frozen_plan()
    paths = PHASE_PATHS[phase]
    manifest = _load_json(paths["manifest"])
    expected_count = EXPECTED_COUNTS[phase]
    if (
        manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA
        or manifest.get("status") != "EXECUTION_COMPLETE_NOT_ANALYZED"
        or manifest.get("phase") != phase
        or manifest.get("call_plan_sha256") != plan["call_plan_sha256"]
        or manifest.get("design_file_sha256") != file_sha256(DEFAULT_DESIGN)
        or manifest.get("plan_manifest_file_sha256") != file_sha256(DEFAULT_PLAN_MANIFEST)
        or manifest.get("runner_sha256") != file_sha256(Path(__file__))
        or manifest.get("analyzer_sha256") != design["locks"]["analyzer"]["sha256"]
        or manifest.get("preregistration_sha256") != file_sha256(FROZEN_PREREGISTRATION)
        or manifest.get("phase_model_calls") != expected_count
        or manifest.get("cumulative_model_calls") != EXPECTED_CUMULATIVE_CALLS[phase]
        or manifest.get("generation_used") is not False
        or manifest.get("partial_resume_allowed") is not False
    ):
        raise PositionalActivationRunnerError(f"{phase} execution manifest changed")
    attempt = _load_json(paths["attempt"])
    if (
        manifest.get("attempt", {}).get("file_sha256") != file_sha256(paths["attempt"])
        or attempt.get("phase") != phase
        or attempt.get("call_plan_sha256") != plan["call_plan_sha256"]
        or attempt.get("predeclared_phase_calls") != expected_count
        or attempt.get("model_calls_before_attempt") != _prior_cumulative_calls(phase)
    ):
        raise PositionalActivationRunnerError(f"{phase} attempt binding changed")
    records = load_jsonl(paths["records"])
    record_receipt = manifest.get("records", {})
    if (
        len(records) != expected_count
        or record_receipt.get("count") != expected_count
        or record_receipt.get("file_sha256") != file_sha256(paths["records"])
        or record_receipt.get("canonical_sha256") != canonical_sha256(records)
        or [record.get("call_index") for record in records] != list(range(expected_count))
        or any(record.get("call_plan_sha256") != plan["call_plan_sha256"] for record in records)
    ):
        raise PositionalActivationRunnerError(f"{phase} record registry changed")
    raw_receipts = manifest.get("raw_logits_shards")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(raw_shard_specs(phase)):
        raise PositionalActivationRunnerError(f"{phase} raw shard registry changed")
    for expected, receipt in zip(raw_shard_specs(phase), raw_receipts, strict=True):
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise PositionalActivationRunnerError(f"{phase} raw shard plan changed")
        path = Path(receipt["path"])
        if not path.is_file() or file_sha256(path) != receipt.get("file_sha256"):
            raise PositionalActivationRunnerError(f"{phase} raw shard file changed")
        shard = _load_array_from_receipt(receipt)
        if shard.shape != tuple(expected["shape"]):
            raise PositionalActivationRunnerError(f"{phase} raw shard shape changed")
    receipt = manifest.get("activations" if phase.endswith("baseline") else "patched_activations")
    if not isinstance(receipt, Mapping):
        raise PositionalActivationRunnerError(f"{phase} sidecar receipt is missing")
    sidecar = _load_array_from_receipt(receipt)
    return manifest, records, sidecar


def _require_authority(
    path: Path,
    *,
    plan: Mapping[str, Any],
    schema: str,
    status: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    authority = _load_json(path)
    expected = {
        "schema_version": schema,
        "status": status,
        "call_plan_sha256": plan["call_plan_sha256"],
        **dict(bindings),
        "model_calls_issued_by_analyzer": 0,
        "generation_used": False,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    if authority != expected:
        raise PositionalActivationRunnerError(
            f"analyzer authority content changed or did not authorize stage: {path.name}"
        )
    return authority


def _selected_layer(plan: Mapping[str, Any]) -> int:
    lock = _load_json(DEFAULT_LAYER_LOCK)
    selected = lock.get("selected_layer")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected not in LAYER_GRID:
        raise PositionalActivationRunnerError("selected layer is invalid")
    localization_analysis = RESULT_ROOT / "localization_analysis.json"
    if not localization_analysis.is_file():
        raise PositionalActivationRunnerError("localization analysis is missing")
    expected = {
        "schema_version": LAYER_LOCK_SCHEMA,
        "status": "LOCALIZATION_LAYER_SELECTED",
        "call_plan_sha256": plan["call_plan_sha256"],
        "selected_layer": selected,
        "selection_rule": "shallowest_layer_passing_all_preregistered_gates",
        "localization_patch_execution_manifest_file_sha256": file_sha256(PHASE_PATHS["localization-patch"]["manifest"]),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "localization_analysis_file_sha256": file_sha256(localization_analysis),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    if lock != expected:
        raise PositionalActivationRunnerError("selected-layer lock content changed")
    return selected


def _phase_output_paths(phase: str, plan: Mapping[str, Any]) -> list[Path]:
    return [
        *PHASE_PATHS[phase].values(),
        *(Path(spec["path"]) for spec in plan["raw_logits_shards"][phase]),
    ]


def run_fit_baseline() -> None:
    phase = "fit-baseline"
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts(phase)
    _require_absent(_phase_output_paths(phase, plan), phase)
    paths = PHASE_PATHS[phase]
    _write_execution_attempt(paths["attempt"], phase=phase, plan=plan, design=design)
    model = _load_model()
    records, activations, raw_receipts = _execute_baselines(model, plan, role="fit", phase=phase, layers=LAYER_GRID)
    _write_completed_phase(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar=activations,
    )


def run_localization_baseline() -> None:
    phase = "localization-baseline"
    plan, design = _load_frozen_plan()
    _validate_execution_manifest("fit-baseline")
    _, basis_lock = _load_basis(plan)
    if basis_lock.get("fit_baseline_execution_manifest_file_sha256") != file_sha256(
        PHASE_PATHS["fit-baseline"]["manifest"]
    ):
        raise PositionalActivationRunnerError("fit basis is not bound to fit execution")
    _require_no_downstream_artifacts(phase)
    _require_absent(_phase_output_paths(phase, plan), phase)
    prerequisites = {
        "fit_baseline_execution_manifest_file_sha256": file_sha256(PHASE_PATHS["fit-baseline"]["manifest"]),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
    }
    _require_authority(
        DEFAULT_LOCALIZATION_BASELINE_ENTRY,
        plan=plan,
        schema=LOCALIZATION_BASELINE_ENTRY_SCHEMA,
        status="LOCALIZATION_BASELINE_AUTHORIZED",
        bindings=prerequisites,
    )
    prerequisites["localization_baseline_entry_file_sha256"] = file_sha256(DEFAULT_LOCALIZATION_BASELINE_ENTRY)
    paths = PHASE_PATHS[phase]
    _write_execution_attempt(paths["attempt"], phase=phase, plan=plan, design=design, prerequisites=prerequisites)
    model = _load_model()
    records, activations, raw_receipts = _execute_baselines(
        model, plan, role="localization", phase=phase, layers=LAYER_GRID
    )
    _write_completed_phase(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar=activations,
        prerequisites=prerequisites,
    )


def run_localization_patch() -> None:
    phase = "localization-patch"
    plan, design = _load_frozen_plan()
    _, baseline_records, activations = _validate_execution_manifest("localization-baseline")
    basis, _ = _load_basis(plan)
    prerequisites = {
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
    }
    _require_authority(
        DEFAULT_LOCALIZATION_PATCH_ENTRY,
        plan=plan,
        schema=LOCALIZATION_PATCH_ENTRY_SCHEMA,
        status="LOCALIZATION_PATCH_AUTHORIZED",
        bindings=prerequisites,
    )
    prerequisites["localization_patch_entry_file_sha256"] = file_sha256(DEFAULT_LOCALIZATION_PATCH_ENTRY)
    _require_no_downstream_artifacts(phase)
    _require_absent(_phase_output_paths(phase, plan), phase)
    paths = PHASE_PATHS[phase]
    _write_execution_attempt(paths["attempt"], phase=phase, plan=plan, design=design, prerequisites=prerequisites)
    model = _load_model()
    records, patched, raw_receipts = _execute_patches(
        model,
        plan,
        role="localization",
        phase=phase,
        templates=_phase_patch_templates(plan, "localization"),
        baseline_records=baseline_records,
        activations=activations,
        basis=basis,
        selected_layer=None,
    )
    _write_completed_phase(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar=patched,
        prerequisites=prerequisites,
    )


def run_holdout_baseline() -> None:
    phase = "holdout-baseline"
    plan, design = _load_frozen_plan()
    _validate_execution_manifest("localization-patch")
    _load_basis(plan)
    selected = _selected_layer(plan)
    holdout_entry_bindings = {
        "localization_patch_execution_manifest_file_sha256": file_sha256(PHASE_PATHS["localization-patch"]["manifest"]),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(DEFAULT_LAYER_LOCK),
    }
    _require_authority(
        DEFAULT_HOLDOUT_BASELINE_ENTRY,
        plan=plan,
        schema=HOLDOUT_BASELINE_ENTRY_SCHEMA,
        status="HOLDOUT_BASELINE_AUTHORIZED",
        bindings=holdout_entry_bindings,
    )
    prerequisites = {
        **holdout_entry_bindings,
        "holdout_baseline_entry_file_sha256": file_sha256(DEFAULT_HOLDOUT_BASELINE_ENTRY),
    }
    _require_no_downstream_artifacts(phase)
    _require_absent(_phase_output_paths(phase, plan), phase)
    paths = PHASE_PATHS[phase]
    _write_execution_attempt(paths["attempt"], phase=phase, plan=plan, design=design, prerequisites=prerequisites)
    model = _load_model()
    records, activations, raw_receipts = _execute_baselines(
        model, plan, role="holdout", phase=phase, layers=(selected,)
    )
    _write_completed_phase(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar=activations,
        prerequisites=prerequisites,
        selected_layer=selected,
    )


def run_holdout_patch() -> None:
    phase = "holdout-patch"
    plan, design = _load_frozen_plan()
    _, baseline_records, activations = _validate_execution_manifest("holdout-baseline")
    basis, _ = _load_basis(plan)
    selected = _selected_layer(plan)
    prerequisites = {
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(PHASE_PATHS["holdout-baseline"]["manifest"]),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(DEFAULT_LAYER_LOCK),
    }
    _require_authority(
        DEFAULT_HOLDOUT_PATCH_ENTRY,
        plan=plan,
        schema=HOLDOUT_PATCH_ENTRY_SCHEMA,
        status="HOLDOUT_PATCH_AUTHORIZED",
        bindings=prerequisites,
    )
    prerequisites["holdout_patch_entry_file_sha256"] = file_sha256(DEFAULT_HOLDOUT_PATCH_ENTRY)
    _require_no_downstream_artifacts(phase)
    _require_absent(_phase_output_paths(phase, plan), phase)
    paths = PHASE_PATHS[phase]
    _write_execution_attempt(paths["attempt"], phase=phase, plan=plan, design=design, prerequisites=prerequisites)
    model = _load_model()
    records, patched, raw_receipts = _execute_patches(
        model,
        plan,
        role="holdout",
        phase=phase,
        templates=_phase_patch_templates(plan, "holdout"),
        baseline_records=baseline_records,
        activations=activations,
        basis=basis,
        selected_layer=selected,
    )
    _write_completed_phase(
        phase=phase,
        plan=plan,
        design=design,
        records=records,
        raw_receipts=raw_receipts,
        sidecar=patched,
        prerequisites=prerequisites,
        selected_layer=selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan",
            "fit-baseline",
            "localization-baseline",
            "localization-patch",
            "holdout-baseline",
            "holdout-patch",
        ),
    )
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    args = parser.parse_args()
    actions = {
        "plan": lambda: run_plan(args.analyzer),
        "fit-baseline": run_fit_baseline,
        "localization-baseline": run_localization_baseline,
        "localization-patch": run_localization_patch,
        "holdout-baseline": run_holdout_baseline,
        "holdout-patch": run_holdout_patch,
    }
    actions[args.phase]()


if __name__ == "__main__":
    main()

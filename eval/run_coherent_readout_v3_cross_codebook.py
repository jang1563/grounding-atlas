"""Run the frozen v3 cross-codebook anti-copy experiment.

The runner is deliberately staged.  ``plan`` performs tokenizer-only checks and
writes the complete zero-forward call plan.  Later phases refuse to run unless
all earlier artifacts and authorizations are byte-for-byte intact.  The model is
never asked to generate; every outcome is a next-token logit measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import model_hooks
    from . import run_coherent_binary_readout as v1_runner
    from . import run_coherent_readout_v2_causal_binding as v2_runner
    from . import run_coherent_readout_v2_syntax as syntax_runner
except ImportError:  # direct execution from eval/
    import model_hooks
    import run_coherent_binary_readout as v1_runner
    import run_coherent_readout_v2_causal_binding as v2_runner
    import run_coherent_readout_v2_syntax as syntax_runner


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v3_cross_codebook"
    / "qwen2.5-1.5b-instruct"
)

PLAN_SCHEMA = "coherent-readout-v3-cross-codebook-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v3-cross-codebook-design-v1"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v3-cross-codebook-plan-manifest-v1"
TOKENIZATION_RECEIPT_SCHEMA = (
    "coherent-readout-v3-cross-codebook-tokenization-receipt-v1"
)
DEPENDENCY_LOCK_SCHEMA = "coherent-readout-v3-cross-codebook-dependency-lock-v1"
PROMPT_SCHEMA = "coherent-readout-v3-cross-codebook-prompt-v1"
BASELINE_TEMPLATE_SCHEMA = (
    "coherent-readout-v3-cross-codebook-baseline-template-v1"
)
PATCH_TEMPLATE_SCHEMA = "coherent-readout-v3-cross-codebook-patch-template-v1"
BASELINE_SCHEMA = "coherent-readout-v3-cross-codebook-baseline-v1"
PATCH_SCHEMA = "coherent-readout-v3-cross-codebook-patch-v1"
EXECUTION_MANIFEST_SCHEMA = "coherent-readout-v3-cross-codebook-execution-v1"
BASIS_SCHEMA = "coherent-readout-v3-cross-codebook-fit-basis-v1"
ATTEMPT_SCHEMA = "coherent-readout-v3-cross-codebook-attempt-v1"

FROZEN_PREREGISTRATION = (
    ROOT / "docs" / "COHERENT_READOUT_V3_CROSS_CODEBOOK_ANTICOPY_PREREG.md"
)
FROZEN_PREREGISTRATION_SHA256 = (
    "98a31bd903744fa054e696ecb421c07da78557d8f8a51ab00fb471514776f949"
)
FIXTURE_BUILDER = (
    ROOT / "signal" / "syntax" / "build_coherent_readout_v3_content_routing_bank.py"
)
FIXTURE_BUILDER_SHA256 = (
    "74192ecda496667b094b6ad0420fd56efcc41644de41aae06cf42758751263f1"
)
FIXTURE = ROOT / "signal" / "syntax" / "coherent_readout_v3_content_routing_bank.json"
FIXTURE_SHA256 = (
    "a63dced290410ef6d463a0f2c04431dcea871ea564f2a6d0b2e0a05b4bb0d78f"
)
FIXTURE_CANONICAL_SHA256 = (
    "65a4f768fcebd648f51667dce84e594049804cb412248e7abe37351e3ac4e5b4"
)
FIXTURE_MANIFEST = FIXTURE.with_suffix(".manifest.json")
FIXTURE_MANIFEST_SHA256 = (
    "bf8559d7922a01403d7d77aa914f459cc88ecec94311da815bd37c28a4340425"
)
MODEL_HOOKS_SHA256 = (
    "62495bd77adc40d7fd5e5643df334eb98aba363f5b81b4b7925314e877bad0c4"
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_WEIGHTS_SHA256 = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
MODEL_CONFIG_SHA256 = (
    "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
MODEL_LAYERS = 28
MODEL_WIDTH = 1536
MODEL_VOCAB_SIZE = 151_936
TOKENIZER_VOCAB_SIZE = 151_665
X_TOKEN_ID = 55
Y_TOKEN_ID = 56
DEVICE = "mps"
DTYPE = "float32"
ATTENTION_IMPLEMENTATION = "sdpa"

SYSTEM_EXACT = (
    "Follow the user's codebook. Your entire response must be exactly one "
    "character: X or Y. Do not write any other text."
)
CHAT_FLAGS = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "enable_thinking": False,
}
EXPECTED_TOKEN_COUNT = 103
LAYER_GRID = (8, 12, 16, 20, 24)
FACTOR_ORDER = ("p", "m", "d", "e", "o")
FACTOR_MASKS = tuple(range(32))
SVD_RELATIVE_TOLERANCE = 1e-6
NUMERICAL_TOLERANCE = 1e-6
LOGIT_IDENTITY_TOLERANCE = 1e-4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260_804

LOCALIZATION_CONDITIONS = (
    "content_anticopy",
    "content_same",
    "answer_anticopy",
    "codebook_anticopy",
    "distractor_anticopy",
    "query_anticopy",
    "order_anticopy",
    "null_0_anticopy",
    "null_1_anticopy",
    "null_2_anticopy",
    "null_3_anticopy",
    "full_anticopy",
    "identity",
    "full_text_counterfactual",
)
HOLDOUT_CONDITIONS = LOCALIZATION_CONDITIONS + (
    "content_erase",
    "content_rescue_same",
    "content_rescue_opposite",
    "null_0_erase",
    "null_1_erase",
    "null_2_erase",
    "null_3_erase",
)
DIRECTION_NAMES = (
    "content",
    "answer",
    "codebook",
    "distractor",
    "query",
    "order",
    "null_0",
    "null_1",
    "null_2",
    "null_3",
)
GENUINE_TARGET_MASKS = {
    "content": 1 << FACTOR_ORDER.index("p"),
    "answer": (1 << FACTOR_ORDER.index("p")) | (1 << FACTOR_ORDER.index("m")),
    "codebook": 1 << FACTOR_ORDER.index("m"),
    "distractor": 1 << FACTOR_ORDER.index("d"),
    "query": 1 << FACTOR_ORDER.index("e"),
    "order": 1 << FACTOR_ORDER.index("o"),
}

DEFAULT_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v3_cross_codebook.py"
DEFAULT_DESIGN = RESULT_ROOT / "design.json"
DEFAULT_PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
DEFAULT_TOKENIZATION_RECEIPT = RESULT_ROOT / "tokenization_receipt.json"
DEFAULT_DEPENDENCY_LOCK = RESULT_ROOT / "dependency_lock.json"
DEFAULT_BASIS_LOCK = RESULT_ROOT / "fit_basis_lock.json"
DEFAULT_BASIS_SIDECAR = RESULT_ROOT / "fit_basis.npy"
DEFAULT_BASIS_DETAILS = RESULT_ROOT / "fit_basis_details.json"
DEFAULT_BASIS_CALCULATIONS = RESULT_ROOT / "fit_basis_calculations.npy"
DEFAULT_LOCALIZATION_ENTRY = RESULT_ROOT / "localization_entry.json"
DEFAULT_LAYER_LOCK = RESULT_ROOT / "layer_lock.json"
DEFAULT_HOLDOUT_ENTRY = RESULT_ROOT / "holdout_entry.json"

PHASE_PATHS = {
    "fit-baseline": {
        "attempt": RESULT_ROOT / "fit_baseline_attempt.json",
        "records": RESULT_ROOT / "fit_baseline_records.jsonl",
        "activations": RESULT_ROOT / "fit_activations.npy",
        "manifest": RESULT_ROOT / "fit_baseline_execution_manifest.json",
    },
    "localization-baseline": {
        "attempt": RESULT_ROOT / "localization_baseline_attempt.json",
        "records": RESULT_ROOT / "localization_baseline_records.jsonl",
        "activations": RESULT_ROOT / "localization_activations.npy",
        "manifest": RESULT_ROOT / "localization_baseline_execution_manifest.json",
    },
    "localization-patch": {
        "attempt": RESULT_ROOT / "localization_patch_attempt.json",
        "records": RESULT_ROOT / "localization_patch_records.jsonl",
        "patched_activations": RESULT_ROOT / "localization_patched_activations.npy",
        "manifest": RESULT_ROOT / "localization_patch_execution_manifest.json",
    },
    "holdout-baseline": {
        "attempt": RESULT_ROOT / "holdout_baseline_attempt.json",
        "records": RESULT_ROOT / "holdout_baseline_records.jsonl",
        "activations": RESULT_ROOT / "holdout_activations.npy",
        "manifest": RESULT_ROOT / "holdout_baseline_execution_manifest.json",
    },
    "holdout-patch": {
        "attempt": RESULT_ROOT / "holdout_patch_attempt.json",
        "records": RESULT_ROOT / "holdout_patch_records.jsonl",
        "patched_activations": RESULT_ROOT / "holdout_patched_activations.npy",
        "manifest": RESULT_ROOT / "holdout_patch_execution_manifest.json",
    },
}


class CrossCodebookRunnerError(ValueError):
    """Raised when an execution would violate the frozen V3 contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def f32_sha256(value: Any) -> str:
    """Hash canonical little-endian float32 bytes (shape is bound separately)."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise CrossCodebookRunnerError("float32 artifact contains non-finite values")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def f64_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.isfinite(array).all():
        raise CrossCodebookRunnerError("float64 artifact contains non-finite values")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossCodebookRunnerError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise CrossCodebookRunnerError(f"JSON artifact must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise CrossCodebookRunnerError(f"cannot read JSONL artifact: {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise CrossCodebookRunnerError("JSONL records must be objects")
    return values


def _atomic_frozen_write(path: Path, payload: bytes) -> None:
    """Write once atomically; an identical replay is a no-op."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CrossCodebookRunnerError(
                f"refusing to overwrite differing frozen artifact: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CrossCodebookRunnerError(f"stale atomic temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_frozen_write(path, payload)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(value)) + "\n" for value in values).encode(
        "utf-8"
    )
    _atomic_frozen_write(path, payload)


def write_array(path: Path, value: np.ndarray) -> None:
    import io

    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise CrossCodebookRunnerError("activation sidecar is not finite")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_frozen_write(path, buffer.getvalue())


def write_f64_array(path: Path, value: np.ndarray) -> None:
    import io

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.isfinite(array).all():
        raise CrossCodebookRunnerError("float64 calculation sidecar is not finite")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_frozen_write(path, buffer.getvalue())


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise CrossCodebookRunnerError(
            f"required distribution is unavailable: {distribution}"
        ) from error


def _require_frozen_static_inputs() -> None:
    expected = (
        (FROZEN_PREREGISTRATION, FROZEN_PREREGISTRATION_SHA256, "preregistration"),
        (FIXTURE_BUILDER, FIXTURE_BUILDER_SHA256, "fixture builder"),
        (FIXTURE, FIXTURE_SHA256, "fixture"),
        (FIXTURE_MANIFEST, FIXTURE_MANIFEST_SHA256, "fixture manifest"),
        (ROOT / "eval" / "model_hooks.py", MODEL_HOOKS_SHA256, "model hooks"),
    )
    for path, digest, label in expected:
        if not path.is_file() or file_sha256(path) != digest:
            raise CrossCodebookRunnerError(f"{label} differs from its frozen hash")


def _load_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location("v3_content_routing_builder", FIXTURE_BUILDER)
    if spec is None or spec.loader is None:
        raise CrossCodebookRunnerError("cannot import the frozen fixture builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_and_rebuild_fixture() -> dict[str, Any]:
    """Require byte identity and an independent deterministic semantic rebuild."""

    _require_frozen_static_inputs()
    fixture = _load_json(FIXTURE)
    if canonical_sha256(fixture) != FIXTURE_CANONICAL_SHA256:
        raise CrossCodebookRunnerError("fixture canonical hash changed")
    builder = _load_builder_module()
    rebuilt = builder.build_fixture()
    builder.validate_fixture(rebuilt)
    if rebuilt != fixture or canonical_sha256(rebuilt) != FIXTURE_CANONICAL_SHA256:
        raise CrossCodebookRunnerError("fixture semantic rebuild differs from disk")
    _validate_fixture_semantics(fixture)
    return fixture


def factor_signs(cell: Mapping[str, Any]) -> dict[str, int]:
    mappings = {
        "p": {"P": -1, "Q": 1},
        "m": {"identity": -1, "swapped": 1},
        "d": {"P": -1, "Q": 1},
        "e": {"a": -1, "b": 1},
        "o": {"query_first": -1, "query_second": 1},
    }
    fields = {
        "p": "queried_property",
        "m": "codebook_id",
        "d": "distractor_property",
        "e": "queried_entity_slot",
        "o": "fact_line_order",
    }
    try:
        return {name: mappings[name][cell[field]] for name, field in fields.items()}
    except (KeyError, TypeError) as error:
        raise CrossCodebookRunnerError("cell factor level is invalid") from error


def walsh_sign(signs: Mapping[str, int], mask: int) -> int:
    if mask < 0 or mask >= 32:
        raise CrossCodebookRunnerError("Walsh mask is outside [0,31]")
    value = 1
    for bit, name in enumerate(FACTOR_ORDER):
        if mask & (1 << bit):
            value *= int(signs[name])
    return value


def _validate_fixture_semantics(fixture: Mapping[str, Any]) -> None:
    cells = fixture.get("cells")
    worlds = fixture.get("world_registry")
    if not isinstance(cells, list) or len(cells) != 1792:
        raise CrossCodebookRunnerError("fixture must contain 1,792 cells")
    if not isinstance(worlds, list) or len(worlds) != 56:
        raise CrossCodebookRunnerError("fixture must contain 56 worlds")
    by_id = {cell.get("cell_id"): cell for cell in cells if isinstance(cell, dict)}
    if len(by_id) != len(cells) or None in by_id:
        raise CrossCodebookRunnerError("fixture cell IDs are invalid or duplicated")
    role_counts = {role: 0 for role in ("direction_fit", "localization", "holdout")}
    selected_counts = {role: 0 for role in role_counts}
    world_counts: dict[str, int] = {}
    for cell in cells:
        role = cell.get("role")
        if role not in role_counts:
            raise CrossCodebookRunnerError("fixture role is invalid")
        role_counts[role] += 1
        selected_counts[role] += int(cell.get("recipient_selected") is True)
        world_counts[cell["world_id"]] = world_counts.get(cell["world_id"], 0) + 1
        signs = factor_signs(cell)
        answer_sign = 1 if cell.get("native_answer") == "X" else -1
        if answer_sign != signs["p"] * signs["m"]:
            raise CrossCodebookRunnerError("native answer does not equal p*m")
        for field in (
            "self_cell_id",
            "anti_copy_donor_cell_id",
            "text_counterfactual_cell_id",
            "same_content_opposite_codebook_donor_cell_id",
            "distractor_flip_cell_id",
            "query_entity_flip_cell_id",
            "fact_order_flip_cell_id",
        ):
            other = by_id.get(cell.get(field))
            if other is None or other.get("world_id") != cell.get("world_id"):
                raise CrossCodebookRunnerError("fixture source reference is invalid")
            if by_id.get(other.get(field)) is not cell:
                raise CrossCodebookRunnerError("fixture source relation is not involutive")
    if role_counts != {"direction_fit": 512, "localization": 256, "holdout": 1024}:
        raise CrossCodebookRunnerError("fixture role cell counts changed")
    if selected_counts != {"direction_fit": 128, "localization": 64, "holdout": 256}:
        raise CrossCodebookRunnerError("fixture selected-recipient counts changed")
    if len(world_counts) != 56 or set(world_counts.values()) != {32}:
        raise CrossCodebookRunnerError("fixture world factorial coverage changed")
    for world_id in sorted(world_counts):
        selected = [
            cell for cell in cells if cell["world_id"] == world_id and cell["recipient_selected"]
        ]
        columns = []
        for cell in selected:
            signs = factor_signs(cell)
            columns.append(
                [
                    signs["e"],
                    signs["p"],
                    signs["m"],
                    signs["d"],
                    signs["o"],
                    signs["p"] * signs["m"],
                ]
            )
        matrix = np.asarray(columns, dtype=int)
        if matrix.shape != (8, 6) or not np.array_equal(matrix.T @ matrix, 8 * np.eye(6)):
            raise CrossCodebookRunnerError("recipient fraction Gram matrix changed")


def sylvester_hadamard(order: int = 16) -> np.ndarray:
    if order < 1 or order & (order - 1):
        raise CrossCodebookRunnerError("Hadamard order must be a positive power of two")
    result = np.asarray([[1]], dtype=np.int8)
    while result.shape[0] < order:
        result = np.block([[result, result], [result, -result]])
    return result


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise CrossCodebookRunnerError(f"{label} must contain one row")
        raw = raw[0]
    if not raw or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise CrossCodebookRunnerError(f"{label} must be a nonempty integer vector")
    return [int(item) for item in raw]


def _contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(tokenizer.encode(rendered, add_special_tokens=False), "prompt")
    combined = _as_int_vector(
        tokenizer.encode(rendered + answer, add_special_tokens=False),
        f"prompt plus {answer}",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise CrossCodebookRunnerError(f"answer {answer} is not one contextual token")
    return combined[-1]


def _literal_token_checks(tokenizer: Any, rendered: str) -> list[dict[str, Any]]:
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = _as_int_vector(encoded["input_ids"], "offset input IDs")
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    checks = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])[PQXY](?![A-Za-z0-9_])", rendered):
        overlaps = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < match.end() and end > match.start()
        ]
        if len(overlaps) != 1:
            raise CrossCodebookRunnerError("a literal P/Q/X/Y occurrence split into tokens")
        index = overlaps[0]
        start, end = offsets[index]
        if rendered[start:end].strip() != match.group(0):
            raise CrossCodebookRunnerError("a literal P/Q/X/Y shares a lexical token")
        checks.append(
            {
                "symbol": match.group(0),
                "character_start": match.start(),
                "character_end": match.end(),
                "token_index": index,
                "token_id": ids[index],
            }
        )
    if not checks:
        raise CrossCodebookRunnerError("prompt contains no literal token checks")
    return checks


def render_prompt(tokenizer: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    if text_sha256(str(cell["prompt_text"])) != cell.get("prompt_sha256"):
        raise CrossCodebookRunnerError("fixture prompt text digest changed")
    messages = [
        {"role": "system", "content": SYSTEM_EXACT},
        {"role": "user", "content": cell["prompt_text"]},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    if not isinstance(rendered, str) or not rendered:
        raise CrossCodebookRunnerError(
            "chat template did not return nonempty rendered text"
        )
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **CHAT_FLAGS,
    )
    if isinstance(tokenized_chat, Mapping):
        if "input_ids" not in tokenized_chat:
            raise CrossCodebookRunnerError(
                "tokenized chat template did not return input_ids"
            )
        input_ids = _as_int_vector(tokenized_chat["input_ids"], "chat-template IDs")
        attention_mask = _as_int_vector(
            tokenized_chat.get("attention_mask", [1] * len(input_ids)),
            "chat-template attention mask",
        )
    else:
        input_ids = _as_int_vector(tokenized_chat, "chat-template IDs")
        attention_mask = [1] * len(input_ids)
    if len(attention_mask) != len(input_ids) or any(
        value != 1 for value in attention_mask
    ):
        raise CrossCodebookRunnerError("chat-template attention mask is not all attended")
    retokenized = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "retokenized IDs"
    )
    if input_ids != retokenized:
        raise CrossCodebookRunnerError("rendered prompt does not retokenize identically")
    if len(input_ids) != EXPECTED_TOKEN_COUNT:
        raise CrossCodebookRunnerError("rendered prompt token count changed")
    x_id = _contextual_token_id(tokenizer, rendered, "X")
    y_id = _contextual_token_id(tokenizer, rendered, "Y")
    if (x_id, y_id) != (X_TOKEN_ID, Y_TOKEN_ID):
        raise CrossCodebookRunnerError("contextual next-answer IDs changed")
    literal_checks = _literal_token_checks(tokenizer, rendered)
    identity = {
        "schema_version": PROMPT_SCHEMA,
        "cell_id": cell["cell_id"],
        "world_id": cell["world_id"],
        "role": cell["role"],
        "system_text_sha256": text_sha256(SYSTEM_EXACT),
        "user_text_sha256": text_sha256(cell["prompt_text"]),
        "rendered_text_sha256": text_sha256(rendered),
        "input_ids_sha256": canonical_sha256(input_ids),
    }
    return {
        **identity,
        "prompt_id": canonical_sha256(identity),
        "system_text": SYSTEM_EXACT,
        "user_text": cell["prompt_text"],
        "rendered_text": rendered,
        "execution_input_ids": input_ids,
        "execution_attention_mask": attention_mask,
        "input_token_count": len(input_ids),
        "final_attended_token_index": len(input_ids) - 1,
        "x_token_id": x_id,
        "y_token_id": y_id,
        "literal_token_checks": literal_checks,
    }


def _condition_spec(condition: str) -> tuple[str, str | None, str | None]:
    projected = {
        "content_anticopy": ("projected_patch", "anti_copy_donor_cell_id", "content"),
        "content_same": (
            "projected_patch",
            "same_content_opposite_codebook_donor_cell_id",
            "content",
        ),
        "answer_anticopy": ("projected_patch", "anti_copy_donor_cell_id", "answer"),
        "codebook_anticopy": (
            "projected_patch",
            "anti_copy_donor_cell_id",
            "codebook",
        ),
        "distractor_anticopy": (
            "projected_patch",
            "anti_copy_donor_cell_id",
            "distractor",
        ),
        "query_anticopy": ("projected_patch", "anti_copy_donor_cell_id", "query"),
        "order_anticopy": ("projected_patch", "anti_copy_donor_cell_id", "order"),
        **{
            f"null_{index}_anticopy": (
                "projected_patch",
                "anti_copy_donor_cell_id",
                f"null_{index}",
            )
            for index in range(4)
        },
        "full_anticopy": ("full_patch", "anti_copy_donor_cell_id", None),
        "identity": ("full_patch", "self_cell_id", None),
        "full_text_counterfactual": (
            "full_patch",
            "text_counterfactual_cell_id",
            None,
        ),
        "content_erase": ("erasure", None, "content"),
        "content_rescue_same": (
            "rescue",
            "same_content_opposite_codebook_donor_cell_id",
            "content",
        ),
        "content_rescue_opposite": (
            "rescue",
            "anti_copy_donor_cell_id",
            "content",
        ),
        **{
            f"null_{index}_erase": ("erasure", None, f"null_{index}")
            for index in range(4)
        },
    }
    try:
        return projected[condition]
    except KeyError as error:
        raise CrossCodebookRunnerError(f"unknown intervention condition: {condition}") from error


def _condition_index(role: str, condition: str) -> int:
    registry = (
        LOCALIZATION_CONDITIONS if role == "localization" else HOLDOUT_CONDITIONS
    )
    try:
        return registry.index(condition)
    except ValueError as error:
        raise CrossCodebookRunnerError("condition is outside its role registry") from error


def _baseline_template(cell: Mapping[str, Any], prompt: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": BASELINE_TEMPLATE_SCHEMA,
        "role": cell["role"],
        "cell_id": cell["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "capture_layers": list(LAYER_GRID) if cell["role"] != "holdout" else None,
        "capture_layer_from_lock": cell["role"] == "holdout",
    }
    return {**identity, "template_id": canonical_sha256(identity)}


def _patch_template(
    recipient: Mapping[str, Any],
    *,
    condition: str,
    layer: int | None,
) -> dict[str, Any]:
    operation, source_field, direction = _condition_spec(condition)
    source_cell_id = None if source_field is None else recipient[source_field]
    identity = {
        "schema_version": PATCH_TEMPLATE_SCHEMA,
        "role": recipient["role"],
        "recipient_cell_id": recipient["cell_id"],
        "source_cell_id": source_cell_id,
        "source_relation": source_field,
        "condition": condition,
        "operation": operation,
        "direction_name": direction,
        "layer": layer,
        "layer_from_lock": layer is None,
        "token_index": -1,
        "strength": 1.0,
        "center": "fit_only_mu" if operation in {"erasure", "rescue"} else None,
    }
    return {**identity, "template_id": canonical_sha256(identity)}


def _config_and_tokenizer() -> tuple[Any, int]:
    v2_runner._verify_cached_plan_assets()
    tokenizer = v1_runner._load_hf_tokenizer(
        MODEL_ID, MODEL_ID, MODEL_REVISION, local_files_only=True
    )
    vocab_size = v1_runner._load_hf_config_vocab_size(
        MODEL_ID, MODEL_REVISION, local_files_only=True
    )
    if vocab_size != MODEL_VOCAB_SIZE or len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise CrossCodebookRunnerError("model/tokenizer vocabulary lock changed")
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise CrossCodebookRunnerError("effective chat template changed")
    return tokenizer, vocab_size


def _verify_cached_model_assets() -> None:
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    for filename, digest in (
        ("config.json", MODEL_CONFIG_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("tokenizer.json", TOKENIZER_JSON_SHA256),
    ):
        path = v2_runner._verify_cached_model_asset(filename, digest)
        if not path.is_file():
            raise CrossCodebookRunnerError(f"cached model asset is missing: {filename}")


def validate_loaded_model(model: Any) -> None:
    try:
        import torch
    except ImportError as error:
        raise CrossCodebookRunnerError("torch is required for model validation") from error
    layers = model_hooks.resolve_decoder_layers(model)
    config = getattr(model, "config", None)
    if len(layers) != MODEL_LAYERS:
        raise CrossCodebookRunnerError("loaded decoder layer count changed")
    if getattr(config, "hidden_size", None) != MODEL_WIDTH:
        raise CrossCodebookRunnerError("loaded hidden width changed")
    if getattr(config, "vocab_size", None) != MODEL_VOCAB_SIZE:
        raise CrossCodebookRunnerError("loaded vocabulary size changed")
    if getattr(config, "_attn_implementation", None) != ATTENTION_IMPLEMENTATION:
        raise CrossCodebookRunnerError("loaded attention implementation changed")
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or {tensor.device.type for tensor in tensors} != {DEVICE}:
        raise CrossCodebookRunnerError("loaded tensors are not all on MPS")
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating or {tensor.dtype for tensor in floating} != {torch.float32}:
        raise CrossCodebookRunnerError("loaded floating tensors are not all float32")


def _load_model() -> Any:
    _verify_cached_model_assets()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise CrossCodebookRunnerError(
            "model execution requires torch and transformers"
        ) from error
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(torch.device(DEVICE)).eval()
    validate_loaded_model(model)
    return model


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise CrossCodebookRunnerError("model has no parameter device") from error


def _inputs(prompt: Mapping[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = _model_device(model)
    ids = prompt["execution_input_ids"]
    mask = prompt["execution_attention_mask"]
    if len(ids) != EXPECTED_TOKEN_COUNT or mask != [1] * EXPECTED_TOKEN_COUNT:
        raise CrossCodebookRunnerError("execution prompt token shape changed")
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([mask], dtype=torch.long, device=device),
    }


def execution_input_sha256(prompt: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "coherent-readout-v3-execution-input-v1",
            "input_ids": prompt["execution_input_ids"],
            "attention_mask": prompt["execution_attention_mask"],
        }
    )


def _full_vocab_diagnostics(row: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise CrossCodebookRunnerError("model returned an invalid vocabulary row")
    x_logit = float(value[X_TOKEN_ID])
    y_logit = float(value[Y_TOKEN_ID])
    maximum = float(value.max())
    maximum_ids = [int(index) for index in np.flatnonzero(value == maximum)]
    peak = float(value.astype(np.float64).max())
    logsumexp = peak + math.log(float(np.exp(value.astype(np.float64) - peak).sum()))
    label_mass = math.exp(float(np.logaddexp(x_logit, y_logit)) - logsumexp)
    return {
        "x_logit": x_logit,
        "y_logit": y_logit,
        "x_minus_y_margin": x_logit - y_logit,
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": label_mass,
        "greedy_token_id": maximum_ids[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": hashlib.sha256(
            value.tobytes(order="C")
        ).hexdigest(),
    }


def target_oriented_margin_from_diagnostics(
    native_answer: str, diagnostics: Mapping[str, Any]
) -> float:
    x = float(diagnostics["x_logit"])
    y = float(diagnostics["y_logit"])
    if native_answer == "X":
        return y - x
    if native_answer == "Y":
        return x - y
    raise CrossCodebookRunnerError("native answer must be X or Y")


def _baseline_forward(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    layers: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import torch

    if not layers or len(set(layers)) != len(layers) or any(
        layer < 0 or layer >= MODEL_LAYERS for layer in layers
    ):
        raise CrossCodebookRunnerError("baseline capture layer registry is invalid")
    captures = [
        model_hooks.ResidualStreamCapture(model, layer, token_index=-1)
        for layer in layers
    ]
    with torch.inference_mode(), ExitStack() as stack:
        for capture in captures:
            stack.enter_context(capture)
        output = model(
            **_inputs(prompt, model),
            use_cache=False,
            return_dict=True,
        )
    capture_counts = [len(capture.values) for capture in captures]
    active_after = [capture.active for capture in captures]
    if capture_counts != [1] * len(captures) or any(active_after):
        raise CrossCodebookRunnerError("baseline capture count or cleanup failed")
    activations = np.stack(
        [capture.values[0][0].detach().float().cpu().numpy() for capture in captures]
    ).astype("<f4", copy=False)
    if activations.shape != (len(layers), MODEL_WIDTH) or not np.isfinite(
        activations
    ).all():
        raise CrossCodebookRunnerError("captured baseline activation shape changed")
    logits = output.logits[0, -1, :].detach().float().cpu().numpy()
    diagnostics = _full_vocab_diagnostics(logits)
    trace = {
        "use_cache": False,
        "return_dict": True,
        "generation_used": False,
        "teacher_forced_prompt_forward": True,
        "capture_layers": [int(layer) for layer in layers],
        "capture_counts": capture_counts,
        "captures_removed": not any(active_after),
        "final_attended_token_index": -1,
        "model_calls": 1,
    }
    return diagnostics, activations, trace


def _baseline_record(
    plan: Mapping[str, Any],
    prompt: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    phase: str,
    activation_row: int,
    layers: Sequence[int],
    activations: np.ndarray,
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    factor_values = factor_signs(cell)
    identity = {
        "schema_version": BASELINE_SCHEMA,
        "phase": phase,
        "role": cell["role"],
        "world_id": cell["world_id"],
        "cell_id": cell["cell_id"],
        "prompt_id": prompt["prompt_id"],
        "activation_row": activation_row,
        "captured_layers": list(layers),
        "call_plan_sha256": plan["call_plan_sha256"],
    }
    baseline_id = canonical_sha256(identity)
    template = _baseline_template(cell, prompt)
    native = cell["native_answer"]
    target_margin = target_oriented_margin_from_diagnostics(native, diagnostics)
    return {
        **identity,
        "record_id": baseline_id,
        "baseline_id": baseline_id,
        "template_id": template["template_id"],
        "condition": "unpatched_baseline",
        "recipient_cell_id": cell["cell_id"],
        "source_cell_id": None,
        "hook_site": "resid_post",
        "token_site": "final_attended_context_token",
        "token_index": -1,
        "queried_property": cell["queried_property"],
        "codebook_id": cell["codebook_id"],
        "distractor_property": cell["distractor_property"],
        "queried_entity_slot": cell["queried_entity_slot"],
        "fact_line_order": cell["fact_line_order"],
        "factor_signs": factor_values,
        "native_answer": native,
        "counterfactual_target": "Y" if native == "X" else "X",
        "target_oriented_margin": target_margin,
        "native_oriented_margin": -target_margin,
        "execution_input_sha256": execution_input_sha256(prompt),
        "input_token_count": prompt["input_token_count"],
        "activation_shape": list(activations.shape),
        "activation_sha256": f32_sha256(activations),
        "activation_layer_sha256": {
            str(layer): f32_sha256(activations[index])
            for index, layer in enumerate(layers)
        },
        "diagnostics": dict(diagnostics),
        "forward_trace": dict(trace),
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "biological_model_calls": 0,
    }


def expected_intervention_activation(
    recipient: np.ndarray,
    *,
    operation: str,
    source: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the frozen intervention in CPU float64, returned as f32."""

    recipient64 = np.asarray(recipient, dtype=np.float64)
    if recipient64.ndim != 1 or not np.isfinite(recipient64).all():
        raise CrossCodebookRunnerError("recipient activation is invalid")
    source64 = None if source is None else np.asarray(source, dtype=np.float64)
    if source64 is not None and (
        source64.shape != recipient64.shape or not np.isfinite(source64).all()
    ):
        raise CrossCodebookRunnerError("source activation is invalid")
    if operation == "full_patch":
        if source64 is None:
            raise CrossCodebookRunnerError("full patch requires a source")
        return np.ascontiguousarray(source64, dtype="<f4")
    if direction is None:
        raise CrossCodebookRunnerError("selective intervention requires a direction")
    unit = np.asarray(direction, dtype=np.float64)
    if unit.shape != recipient64.shape or not np.isfinite(unit).all():
        raise CrossCodebookRunnerError("intervention direction is invalid")
    norm = float(np.linalg.norm(unit))
    if not math.isfinite(norm) or norm == 0.0:
        raise CrossCodebookRunnerError("intervention direction has zero norm")
    unit = unit / norm
    if operation in {"projected_patch", "rescue"}:
        if source64 is None:
            raise CrossCodebookRunnerError("projected intervention requires a source")
        result = recipient64 + unit * float(unit @ (source64 - recipient64))
    elif operation == "erasure":
        if center is None:
            raise CrossCodebookRunnerError("erasure requires the fit-only center")
        center64 = np.asarray(center, dtype=np.float64)
        if center64.shape != recipient64.shape or not np.isfinite(center64).all():
            raise CrossCodebookRunnerError("erasure center is invalid")
        result = recipient64 - unit * float(unit @ (recipient64 - center64))
    else:
        raise CrossCodebookRunnerError(f"unsupported intervention operation: {operation}")
    return np.ascontiguousarray(result, dtype="<f4")


def _projector_diagnostics(direction: np.ndarray) -> dict[str, float]:
    applied = np.ascontiguousarray(np.asarray(direction, dtype="<f4"))
    norm_f32 = np.float32(
        np.sqrt(np.sum(applied * applied, dtype=np.float32), dtype=np.float32)
    )
    norm = float(norm_f32)
    if applied.ndim != 1 or not math.isfinite(norm) or norm == 0.0:
        raise CrossCodebookRunnerError("projector direction is invalid")
    unit = np.ascontiguousarray(applied / norm_f32, dtype="<f4").astype(np.float64)
    norm_squared = float(unit @ unit)
    return {
        "direction_input_norm": norm,
        "projector_symmetry_error": 0.0,
        "projector_idempotence_error": abs(norm_squared * (norm_squared - 1.0)),
        "unit_norm_error": abs(math.sqrt(norm_squared) - 1.0),
    }


def _intervention_transform(
    *,
    operation: str,
    source: np.ndarray | None,
    direction: np.ndarray | None,
    center: np.ndarray | None,
) -> Any:
    if operation == "full_patch":
        if source is None:
            raise CrossCodebookRunnerError("full patch source is missing")
        return model_hooks.patch_transform(source, token_index=-1, strength=1.0)
    if direction is None:
        raise CrossCodebookRunnerError("selective direction is missing")
    if operation == "projected_patch":
        if source is None:
            raise CrossCodebookRunnerError("projected source is missing")
        return model_hooks.patch_transform(
            source,
            token_index=-1,
            direction=direction,
            center=center,
            strength=1.0,
        )
    if operation == "erasure":
        if center is None:
            raise CrossCodebookRunnerError("erasure center is missing")
        return model_hooks.directional_erasure_transform(
            direction,
            token_index=-1,
            center=center,
            strength=1.0,
        )
    if operation == "rescue":
        if source is None or center is None:
            raise CrossCodebookRunnerError("rescue source or center is missing")
        return model_hooks.compose_transforms(
            model_hooks.directional_erasure_transform(
                direction,
                token_index=-1,
                center=center,
                strength=1.0,
            ),
            model_hooks.patch_transform(
                source,
                token_index=-1,
                direction=direction,
                center=center,
                strength=1.0,
            ),
        )
    raise CrossCodebookRunnerError(f"unknown intervention operation: {operation}")


def _patch_forward(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    layer: int,
    operation: str,
    recipient_activation: np.ndarray,
    source_activation: np.ndarray | None,
    direction: np.ndarray | None,
    center: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    import torch

    recipient = np.ascontiguousarray(recipient_activation, dtype="<f4")
    source = (
        None
        if source_activation is None
        else np.ascontiguousarray(source_activation, dtype="<f4")
    )
    applied_direction = (
        None if direction is None else np.ascontiguousarray(direction, dtype="<f4")
    )
    fit_center = None if center is None else np.ascontiguousarray(center, dtype="<f4")
    if recipient.shape != (MODEL_WIDTH,) or not np.isfinite(recipient).all():
        raise CrossCodebookRunnerError("recipient patch activation shape changed")
    if source is not None and (
        source.shape != (MODEL_WIDTH,) or not np.isfinite(source).all()
    ):
        raise CrossCodebookRunnerError("source patch activation shape changed")
    if applied_direction is not None and applied_direction.shape != (MODEL_WIDTH,):
        raise CrossCodebookRunnerError("applied direction shape changed")
    if fit_center is not None and fit_center.shape != (MODEL_WIDTH,):
        raise CrossCodebookRunnerError("fit center shape changed")
    base_transform = _intervention_transform(
        operation=operation,
        source=source,
        direction=applied_direction,
        center=fit_center,
    )
    expected_cpu = expected_intervention_activation(
        recipient,
        operation=operation,
        source=source,
        direction=applied_direction,
        center=fit_center,
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
        pre = hidden[:, -1, :].detach().float().cpu().numpy()[0].astype("<f4", copy=False)
        changed = base_transform(hidden)
        post = (
            changed[:, -1, :].detach().float().cpu().numpy()[0].astype("<f4", copy=False)
        )
        post_holder["value"] = np.ascontiguousarray(post.copy(), dtype="<f4")
        trace["non_target_tokens_unchanged"] = bool(
            trace["non_target_tokens_unchanged"]
            and torch.equal(changed[:, :-1, :], hidden[:, :-1, :])
        )
        trace["pre_activation_matches_registered_recipient"] = bool(
            trace["pre_activation_matches_registered_recipient"]
            and f32_sha256(pre) == f32_sha256(recipient)
        )
        expected_error = float(np.linalg.norm(post.astype(np.float64) - expected_cpu))
        expected_tolerance = NUMERICAL_TOLERANCE * max(
            1.0, float(np.linalg.norm(expected_cpu.astype(np.float64)))
        )
        trace["post_activation_matches_expected"] = bool(
            trace["post_activation_matches_expected"]
            and expected_error <= expected_tolerance
        )
        trace["pre_activation_sha256"] = f32_sha256(pre)
        trace["post_activation_sha256"] = f32_sha256(post)
        trace["expected_activation_sha256"] = f32_sha256(expected_cpu)
        trace["post_expected_l2_error"] = expected_error
        trace["post_expected_l2_tolerance"] = expected_tolerance
        delta = post.astype(np.float64) - pre.astype(np.float64)
        trace["displacement_l2"] = float(np.linalg.norm(delta))
        if applied_direction is not None:
            unit = applied_direction.astype(np.float64)
            unit /= np.linalg.norm(unit)
            origin = (
                np.zeros(MODEL_WIDTH, dtype=np.float64)
                if fit_center is None
                else fit_center.astype(np.float64)
            )
            trace["pre_axis_coefficient"] = float(
                unit @ (pre.astype(np.float64) - origin)
            )
            trace["source_axis_coefficient"] = (
                None
                if source is None
                else float(unit @ (source.astype(np.float64) - origin))
            )
            trace["post_axis_coefficient"] = float(
                unit @ (post.astype(np.float64) - origin)
            )
            trace["expected_axis_coefficient"] = float(
                unit @ (expected_cpu.astype(np.float64) - origin)
            )
            trace["post_expected_axis_coefficient_error"] = abs(
                trace["post_axis_coefficient"]
                - trace["expected_axis_coefficient"]
            )
            orthogonal = delta - unit * float(unit @ delta)
            orthogonal_norm = float(np.linalg.norm(orthogonal))
            orthogonal_tolerance = NUMERICAL_TOLERANCE * max(
                1.0, float(np.linalg.norm(pre.astype(np.float64)))
            )
            trace.update(_projector_diagnostics(applied_direction))
            trace["orthogonal_displacement_l2"] = orthogonal_norm
            trace["orthogonal_displacement_tolerance"] = orthogonal_tolerance
            if source is not None:
                full_displacement = float(
                    np.linalg.norm(source.astype(np.float64) - pre.astype(np.float64))
                )
            else:
                if fit_center is None:
                    raise CrossCodebookRunnerError("selective reference center is missing")
                full_displacement = float(
                    np.linalg.norm(
                        fit_center.astype(np.float64) - pre.astype(np.float64)
                    )
                )
            displacement_tolerance = NUMERICAL_TOLERANCE * max(
                1.0, full_displacement
            )
            trace["corresponding_full_displacement_l2"] = full_displacement
            trace["selective_displacement_tolerance"] = displacement_tolerance
            trace["orthogonal_displacement_pass"] = (
                orthogonal_norm <= orthogonal_tolerance
            )
            trace["selective_not_larger_than_full_pass"] = (
                trace["displacement_l2"] <= full_displacement + displacement_tolerance
            )
        else:
            trace.update(
                {
                    "direction_input_norm": None,
                    "projector_symmetry_error": None,
                    "projector_idempotence_error": None,
                    "unit_norm_error": None,
                    "orthogonal_displacement_l2": None,
                    "orthogonal_displacement_tolerance": None,
                    "corresponding_full_displacement_l2": None,
                    "selective_displacement_tolerance": None,
                    "orthogonal_displacement_pass": True,
                    "selective_not_larger_than_full_pass": True,
                    "pre_axis_coefficient": None,
                    "source_axis_coefficient": None,
                    "post_axis_coefficient": None,
                    "expected_axis_coefficient": None,
                    "post_expected_axis_coefficient_error": None,
                }
            )
        return changed

    intervention = model_hooks.ResidualStreamIntervention(model, layer, traced_transform)
    with torch.inference_mode(), intervention:
        output = model(
            **_inputs(prompt, model),
            use_cache=False,
            return_dict=True,
        )
    trace["hook_removed"] = not intervention.active
    trace["finite_activations"] = bool(
        np.isfinite(recipient).all()
        and (source is None or np.isfinite(source).all())
        and np.isfinite(expected_cpu).all()
    )
    trace["operation"] = operation
    trace["intervention_kind"] = operation
    trace["layer"] = layer
    trace["token_index"] = -1
    trace["strength"] = 1.0
    trace["model_calls"] = 1
    trace["generation_used"] = False
    trace["patched_activation_hash_pass"] = (
        trace["post_activation_sha256"] == trace["expected_activation_sha256"]
        or trace["post_expected_l2_error"] <= trace["post_expected_l2_tolerance"]
    )
    required = (
        trace["hook_calls"] == 1,
        trace["hook_removed"],
        trace["non_target_tokens_unchanged"],
        trace["pre_activation_matches_registered_recipient"],
        trace["post_activation_matches_expected"],
        trace["orthogonal_displacement_pass"],
        trace["selective_not_larger_than_full_pass"],
        trace["finite_activations"],
        trace["patched_activation_hash_pass"],
        trace["projector_symmetry_error"] is None
        or trace["projector_symmetry_error"] <= NUMERICAL_TOLERANCE,
        trace["projector_idempotence_error"] is None
        or trace["projector_idempotence_error"] <= NUMERICAL_TOLERANCE,
    )
    if not all(required):
        raise CrossCodebookRunnerError("intervention hook or numerical gate failed")
    logits = output.logits[0, -1, :].detach().float().cpu().numpy()
    diagnostics = _full_vocab_diagnostics(logits)
    trace["finite_logits"] = True
    post_activation = post_holder.get("value")
    if post_activation is None or post_activation.shape != (MODEL_WIDTH,):
        raise CrossCodebookRunnerError("patched activation was not captured")
    return diagnostics, trace, post_activation


def _patch_record(
    plan: Mapping[str, Any],
    template: Mapping[str, Any],
    recipient_cell: Mapping[str, Any],
    recipient_prompt: Mapping[str, Any],
    *,
    phase: str,
    layer: int,
    patched_activation_row: int,
    recipient_baseline: Mapping[str, Any],
    source_baseline: Mapping[str, Any] | None,
    recipient_activation: np.ndarray,
    source_activation: np.ndarray | None,
    direction: np.ndarray | None,
    center: np.ndarray | None,
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": PATCH_SCHEMA,
        "phase": phase,
        "role": recipient_cell["role"],
        "world_id": recipient_cell["world_id"],
        "recipient_cell_id": recipient_cell["cell_id"],
        "source_cell_id": template["source_cell_id"],
        "template_id": template["template_id"],
        "condition": template["condition"],
        "operation": template["operation"],
        "direction_name": template["direction_name"],
        "layer": layer,
        "call_plan_sha256": plan["call_plan_sha256"],
    }
    native = recipient_cell["native_answer"]
    target_margin = target_oriented_margin_from_diagnostics(native, diagnostics)
    patch_id = canonical_sha256(identity)
    return {
        **identity,
        "record_id": patch_id,
        "patch_id": patch_id,
        "source_relation": template["source_relation"],
        "cell_id": recipient_cell["cell_id"],
        "prompt_id": recipient_prompt["prompt_id"],
        "execution_input_sha256": execution_input_sha256(recipient_prompt),
        "hook_site": "resid_post",
        "token_site": "final_attended_context_token",
        "token_index": -1,
        "strength": 1.0,
        "center": template["center"],
        "queried_property": recipient_cell["queried_property"],
        "codebook_id": recipient_cell["codebook_id"],
        "distractor_property": recipient_cell["distractor_property"],
        "queried_entity_slot": recipient_cell["queried_entity_slot"],
        "fact_line_order": recipient_cell["fact_line_order"],
        "factor_signs": factor_signs(recipient_cell),
        "native_answer": native,
        "counterfactual_target": "Y" if native == "X" else "X",
        "target_oriented_margin": target_margin,
        "native_oriented_margin": -target_margin,
        "patched_activation_row": patched_activation_row,
        "recipient_baseline_id": recipient_baseline["baseline_id"],
        "recipient_activation_row": recipient_baseline["activation_row"],
        "source_baseline_id": (
            None if source_baseline is None else source_baseline["baseline_id"]
        ),
        "source_activation_row": (
            None if source_baseline is None else source_baseline["activation_row"]
        ),
        "recipient_activation_sha256": f32_sha256(recipient_activation),
        "source_activation_sha256": (
            None if source_activation is None else f32_sha256(source_activation)
        ),
        "applied_direction_sha256": (
            None if direction is None else f32_sha256(direction)
        ),
        "fit_center_sha256": None if center is None else f32_sha256(center),
        "patched_activation_sha256": trace["post_activation_sha256"],
        "expected_activation_sha256": trace["expected_activation_sha256"],
        "diagnostics": dict(diagnostics),
        "hook_trace": dict(trace),
        "runner_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "biological_model_calls": 0,
    }


def _dependency_lock(analyzer_path: Path) -> dict[str, Any]:
    if not analyzer_path.is_file():
        raise CrossCodebookRunnerError(f"analyzer is missing: {analyzer_path}")
    test_paths = sorted(ROOT.glob("tests/test_*v3*"))
    if not test_paths or any(not path.is_file() for path in test_paths):
        raise CrossCodebookRunnerError("V3 test-file registry is empty")
    try:
        import torch
    except ImportError as error:
        raise CrossCodebookRunnerError("torch is required for dependency lock") from error
    core = {
        "schema_version": DEPENDENCY_LOCK_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in (
                "huggingface-hub",
                "numpy",
                "safetensors",
                "tokenizers",
                "torch",
                "transformers",
            )
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
        "implementation_files": {
            "runner": {"path": str(Path(__file__)), "sha256": file_sha256(Path(__file__))},
            "analyzer": {"path": str(analyzer_path), "sha256": file_sha256(analyzer_path)},
            "model_hooks": {
                "path": str(ROOT / "eval" / "model_hooks.py"),
                "sha256": MODEL_HOOKS_SHA256,
            },
            "run_coherent_binary_readout": {
                "path": str(ROOT / "eval" / "run_coherent_binary_readout.py"),
                "sha256": file_sha256(
                    ROOT / "eval" / "run_coherent_binary_readout.py"
                ),
            },
            "run_coherent_readout_v2_causal_binding": {
                "path": str(
                    ROOT / "eval" / "run_coherent_readout_v2_causal_binding.py"
                ),
                "sha256": file_sha256(
                    ROOT / "eval" / "run_coherent_readout_v2_causal_binding.py"
                ),
            },
            "run_coherent_readout_v2_syntax": {
                "path": str(ROOT / "eval" / "run_coherent_readout_v2_syntax.py"),
                "sha256": file_sha256(
                    ROOT / "eval" / "run_coherent_readout_v2_syntax.py"
                ),
            },
            "tests": [
                {"path": str(path), "sha256": file_sha256(path)} for path in test_paths
            ],
        },
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def build_plan(
    tokenizer: Any,
    analyzer_path: Path = DEFAULT_ANALYZER,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the complete zero-forward plan, token receipt, and dependency lock."""

    fixture = load_and_rebuild_fixture()
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise CrossCodebookRunnerError("effective chat template changed")
    dependency = _dependency_lock(analyzer_path)
    cells = sorted(fixture["cells"], key=lambda value: value["cell_id"])
    prompts = [render_prompt(tokenizer, cell) for cell in cells]
    prompt_by_cell = {prompt["cell_id"]: prompt for prompt in prompts}
    if len(prompt_by_cell) != 1792:
        raise CrossCodebookRunnerError("prompt registry changed")
    baseline_templates = [
        _baseline_template(cell, prompt_by_cell[cell["cell_id"]]) for cell in cells
    ]
    selected = [cell for cell in cells if cell["recipient_selected"]]
    patch_templates: list[dict[str, Any]] = []
    for recipient in selected:
        if recipient["role"] == "localization":
            for layer in LAYER_GRID:
                for condition in LOCALIZATION_CONDITIONS:
                    patch_templates.append(
                        _patch_template(recipient, condition=condition, layer=layer)
                    )
        elif recipient["role"] == "holdout":
            for condition in HOLDOUT_CONDITIONS:
                patch_templates.append(
                    _patch_template(recipient, condition=condition, layer=None)
                )
    if len(patch_templates) != 4480 + 5376:
        raise CrossCodebookRunnerError("intervention template count changed")
    if len({item["template_id"] for item in patch_templates}) != len(patch_templates):
        raise CrossCodebookRunnerError("intervention template IDs are duplicated")
    hadamard = sylvester_hadamard(16)
    null_signs = hadamard[:, 1:5].astype(int).tolist()
    receipt_core = {
        "schema_version": TOKENIZATION_RECEIPT_SCHEMA,
        "model_calls": 0,
        "generation_used": False,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "chat_flags": CHAT_FLAGS,
        "expected_token_count": EXPECTED_TOKEN_COUNT,
        "x_next_token_id": X_TOKEN_ID,
        "y_next_token_id": Y_TOKEN_ID,
        "prompt_count": len(prompts),
        "prompt_receipts": [
            {
                "cell_id": prompt["cell_id"],
                "prompt_id": prompt["prompt_id"],
                "rendered_text_sha256": prompt["rendered_text_sha256"],
                "input_ids_sha256": prompt["input_ids_sha256"],
                "input_token_count": prompt["input_token_count"],
                "literal_token_checks_sha256": canonical_sha256(
                    prompt["literal_token_checks"]
                ),
            }
            for prompt in prompts
        ],
    }
    receipt = {**receipt_core, "canonical_sha256": canonical_sha256(receipt_core)}
    plan_core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": "coherent-readout-v3-cross-codebook-anticopy-v1",
        "freeze_date": "2026-08-02",
        "mode": "prospective_development_synthetic_nonbiological",
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
        "locks": {
            "preregistration": {"path": str(FROZEN_PREREGISTRATION), "sha256": FROZEN_PREREGISTRATION_SHA256},
            "fixture_builder": {"path": str(FIXTURE_BUILDER), "sha256": FIXTURE_BUILDER_SHA256},
            "fixture": {"path": str(FIXTURE), "sha256": FIXTURE_SHA256, "canonical_sha256": FIXTURE_CANONICAL_SHA256},
            "fixture_manifest": {"path": str(FIXTURE_MANIFEST), "sha256": FIXTURE_MANIFEST_SHA256},
            "model_hooks": {"path": str(ROOT / "eval" / "model_hooks.py"), "sha256": MODEL_HOOKS_SHA256},
            "dependency_lock_canonical_sha256": dependency["canonical_sha256"],
            "tokenization_receipt_canonical_sha256": receipt["canonical_sha256"],
        },
        "roles": {
            "direction_fit": {"worlds": 16, "cells": 512, "recipients": 128},
            "localization": {"worlds": 8, "cells": 256, "recipients": 64},
            "holdout": {"worlds": 32, "cells": 1024, "recipients": 256},
        },
        "layer_grid": list(LAYER_GRID),
        "factor_order": list(FACTOR_ORDER),
        "walsh_masks": list(FACTOR_MASKS),
        "svd_relative_tolerance": SVD_RELATIVE_TOLERANCE,
        "hadamard_h16": hadamard.astype(int).tolist(),
        "null_signs_h16_columns_1_to_4": null_signs,
        "fit_world_ids": sorted(
            world["world_id"] for world in fixture["world_registry"] if world["role"] == "direction_fit"
        ),
        "localization_conditions": list(LOCALIZATION_CONDITIONS),
        "holdout_conditions": list(HOLDOUT_CONDITIONS),
        "prompts": prompts,
        "baseline_templates": baseline_templates,
        "patch_templates": sorted(
            patch_templates,
            key=lambda value: (
                value["role"],
                value["recipient_cell_id"],
                -1 if value["layer"] is None else value["layer"],
                _condition_index(value["role"], value["condition"]),
            ),
        ),
        "cell_registry": cells,
        "expected_counts": {
            "all_prompts": 1792,
            "fit_baselines": 512,
            "localization_baselines": 256,
            "localization_patch_rows": 4480,
            "holdout_baselines": 1024,
            "holdout_patch_rows": 5376,
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "symbolic_world"},
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
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise CrossCodebookRunnerError("plan schema changed")
    receipt = _load_json(receipt_path)
    dependency = _load_json(dependency_path)
    return {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "mode": plan["mode"],
        "claim_scope": "single_model_synthetic_symbolic_projected_content_recomposition_only",
        "prohibited_claims": [
            "biology",
            "latent_knowledge",
            "activation_gap",
            "physical_law",
            "general_variable_binding",
            "model_family_generality",
        ],
        "model": plan["model"],
        "locks": {
            **plan["locks"],
            "runner": {"path": str(Path(__file__)), "sha256": file_sha256(Path(__file__))},
            "analyzer": dependency["implementation_files"]["analyzer"],
            "v3_tests": dependency["implementation_files"]["tests"],
            "dependency_lock": {"path": str(dependency_path), "file_sha256": file_sha256(dependency_path), "canonical_sha256": dependency["canonical_sha256"]},
            "tokenization_receipt": {"path": str(receipt_path), "file_sha256": file_sha256(receipt_path), "canonical_sha256": receipt["canonical_sha256"]},
        },
        "call_plan_sha256": plan["call_plan_sha256"],
        "layer_grid": plan["layer_grid"],
        "conditions": {"localization": plan["localization_conditions"], "holdout": plan["holdout_conditions"]},
        "expected_counts": plan["expected_counts"],
        "basis_rule": {
            "factor_order": list(FACTOR_ORDER),
            "all_nonempty_walsh_coefficients": 31,
            "svd_relative_tolerance": SVD_RELATIVE_TOLERANCE,
            "null_signs": plan["null_signs_h16_columns_1_to_4"],
            "erasure_center": "direction_fit_intercept_only",
        },
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }


def validate_plan(plan: Mapping[str, Any], design: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA or design.get("schema_version") != DESIGN_SCHEMA:
        raise CrossCodebookRunnerError("plan/design schema changed")
    core = {key: value for key, value in plan.items() if key != "call_plan_sha256"}
    if canonical_sha256(core) != plan.get("call_plan_sha256"):
        raise CrossCodebookRunnerError("call-plan canonical hash changed")
    if design.get("call_plan_sha256") != plan.get("call_plan_sha256"):
        raise CrossCodebookRunnerError("design and plan hashes differ")
    if plan.get("model_calls_before_plan_freeze") != 0 or design.get("model_calls") != 0:
        raise CrossCodebookRunnerError("pre-forward artifact reports model calls")
    if plan.get("generation_used") is not False or design.get("generation_used") is not False:
        raise CrossCodebookRunnerError("pre-forward artifact permits generation")
    if len(plan.get("prompts", [])) != 1792 or len(plan.get("patch_templates", [])) != 9856:
        raise CrossCodebookRunnerError("plan row templates changed")


def run_plan(analyzer_path: Path = DEFAULT_ANALYZER) -> None:
    if analyzer_path.resolve() != DEFAULT_ANALYZER.resolve():
        raise CrossCodebookRunnerError(
            "V3 plan requires the frozen default analyzer path"
        )
    downstream = [path for paths in PHASE_PATHS.values() for path in paths.values()]
    downstream.extend(
        [
            RESULT_ROOT / "fit_basis_analysis.json",
            DEFAULT_BASIS_SIDECAR,
            DEFAULT_BASIS_CALCULATIONS,
            DEFAULT_BASIS_DETAILS,
            DEFAULT_BASIS_LOCK,
            DEFAULT_LOCALIZATION_ENTRY,
            RESULT_ROOT / "localization_baseline_analysis.json",
            RESULT_ROOT / "localization_analysis.json",
            DEFAULT_LAYER_LOCK,
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ]
    )
    existing_downstream = [path for path in downstream if path.exists()]
    if existing_downstream:
        raise CrossCodebookRunnerError(
            "refusing to freeze a zero-forward plan after execution artifact: "
            f"{existing_downstream[0]}"
        )
    _require_frozen_static_inputs()
    tokenizer, _ = _config_and_tokenizer()
    plan, receipt, dependency = build_plan(tokenizer, analyzer_path)
    write_json(DEFAULT_TOKENIZATION_RECEIPT, receipt)
    write_json(DEFAULT_DEPENDENCY_LOCK, dependency)
    design = design_from_plan(plan)
    validate_plan(plan, design)
    manifest = {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": "PLAN_AND_DESIGN_FROZEN_NO_FORWARD",
        "mode": plan["mode"],
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
        "design_path": str(DEFAULT_DESIGN),
        "plan": plan,
        "call_plan_sha256": plan["call_plan_sha256"],
        "tokenization_receipt_file_sha256": file_sha256(DEFAULT_TOKENIZATION_RECEIPT),
        "dependency_lock_file_sha256": file_sha256(DEFAULT_DEPENDENCY_LOCK),
    }
    write_json(DEFAULT_DESIGN, design)
    manifest["design_file_sha256"] = file_sha256(DEFAULT_DESIGN)
    write_json(DEFAULT_PLAN_MANIFEST, manifest)


def _residualize(target: np.ndarray, nuisance: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    target64 = np.asarray(target, dtype=np.float64)
    nuisance64 = np.asarray(nuisance, dtype=np.float64)
    if target64.ndim != 1 or nuisance64.ndim != 2 or nuisance64.shape[1] != target64.shape[0]:
        raise CrossCodebookRunnerError("residualization dimensions are invalid")
    _, singular_values, vh = np.linalg.svd(nuisance64, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    rank = int(np.sum(singular_values > SVD_RELATIVE_TOLERANCE * sigma_max)) if sigma_max > 0.0 else 0
    basis = vh[:rank]
    projection = basis.T @ (basis @ target64) if rank else np.zeros_like(target64)
    return target64 - projection, singular_values, rank


def _compute_fit_basis_full(
    cells: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    *,
    layer_grid: Sequence[int] = LAYER_GRID,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Compute all fit-only Walsh coefficients and frozen rank-one directions."""

    array = np.asarray(activations, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != len(cells) or array.shape[1] != len(layer_grid):
        raise CrossCodebookRunnerError("fit activation array dimensions changed")
    if not len(cells) or not np.isfinite(array).all():
        raise CrossCodebookRunnerError("fit activation array is empty or non-finite")
    widths = array.shape[2]
    signs = [factor_signs(cell) for cell in cells]
    world_ids = sorted({str(cell["world_id"]) for cell in cells})
    if len(world_ids) != 16 or any(sum(cell["world_id"] == world for cell in cells) != 32 for world in world_ids):
        raise CrossCodebookRunnerError("basis requires 16 complete 32-cell fit worlds")
    sign_matrix = np.asarray(
        [[walsh_sign(cell_signs, mask) for mask in FACTOR_MASKS] for cell_signs in signs],
        dtype=np.float64,
    )
    if not np.array_equal(sign_matrix.T @ sign_matrix, len(cells) * np.eye(32)):
        raise CrossCodebookRunnerError("fit Walsh design is not orthogonal")
    registry = ["walsh_00_intercept", "fit_intercept"]
    registry.extend(f"walsh_{mask:02d}" for mask in range(1, 32))
    registry.extend(f"direction_{name}" for name in DIRECTION_NAMES)
    if len(registry) != 43:
        raise AssertionError("basis registry size changed")
    sidecar = np.empty((len(layer_grid), len(registry), widths), dtype="<f4")
    calculation_registry = [f"walsh_{mask:02d}" for mask in range(32)]
    calculation_registry.extend(f"residual_{name}" for name in DIRECTION_NAMES)
    calculation_registry.extend(f"within_world_content_{index:02d}" for index in range(16))
    calculation_registry.extend(f"null_raw_{index}" for index in range(4))
    if len(calculation_registry) != 62:
        raise AssertionError("fit calculation registry size changed")
    calculations = np.empty(
        (len(layer_grid), len(calculation_registry), widths), dtype="<f8"
    )
    details: dict[str, Any] = {
        "schema_version": BASIS_SCHEMA,
        "layer_grid": list(layer_grid),
        "factor_order": list(FACTOR_ORDER),
        "registry": registry,
        "fit_world_ids": world_ids,
        "fit_cell_ids": [str(cell["cell_id"]) for cell in cells],
        "svd_relative_tolerance": SVD_RELATIVE_TOLERANCE,
        "calculation_registry": calculation_registry,
        "layers": {},
    }
    hadamard_null = sylvester_hadamard(16)[:, 1:5].astype(np.float64)
    world_indices = {world: [index for index, cell in enumerate(cells) if cell["world_id"] == world] for world in world_ids}
    p_mask = GENUINE_TARGET_MASKS["content"]
    for layer_position, layer in enumerate(layer_grid):
        hidden = array[:, layer_position, :]
        coefficients = sign_matrix.T @ hidden / len(cells)
        intercept = hidden.mean(axis=0)
        if not np.allclose(coefficients[0], intercept, rtol=0.0, atol=1e-12):
            raise CrossCodebookRunnerError("Walsh intercept and fit mean differ")
        genuine = coefficients[1:]
        directions: dict[str, np.ndarray] = {}
        residuals: dict[str, np.ndarray] = {}
        layer_details: dict[str, Any] = {
            "intercept_f64_sha256": f64_sha256(intercept),
            "walsh_coefficients_f64_sha256": f64_sha256(coefficients),
            "directions": {},
            "within_world_content_coefficient_f64_sha256": None,
            "null_raw_f64_sha256": None,
        }
        for name, mask in GENUINE_TARGET_MASKS.items():
            target_index = mask - 1
            nuisance = np.delete(genuine, target_index, axis=0)
            residual, singular_values, rank = _residualize(genuine[target_index], nuisance)
            raw_norm = float(np.linalg.norm(genuine[target_index]))
            residual_norm = float(np.linalg.norm(residual))
            ratio = residual_norm / max(raw_norm, 1e-12)
            if not math.isfinite(ratio) or ratio <= SVD_RELATIVE_TOLERANCE or residual_norm == 0.0:
                raise CrossCodebookRunnerError(f"ineligible residual for {name} at layer {layer}")
            direction = residual / residual_norm
            if abs(float(np.linalg.norm(direction)) - 1.0) > NUMERICAL_TOLERANCE:
                raise CrossCodebookRunnerError("direction is not unit norm")
            directions[name] = direction
            residuals[name] = residual
            layer_details["directions"][name] = {
                "target_mask": mask,
                "nuisance_masks": [value for value in range(1, 32) if value != mask],
                "singular_values": singular_values.tolist(),
                "retained_rank": rank,
                "raw_f64_sha256": f64_sha256(genuine[target_index]),
                "residual_f64_sha256": f64_sha256(residual),
                "residual_norm": residual_norm,
                "raw_norm": raw_norm,
                "residual_ratio": ratio,
                "direction_f32_sha256": f32_sha256(direction),
            }
        within_world_p = []
        for world in world_ids:
            indices = world_indices[world]
            world_signs = sign_matrix[indices, p_mask]
            within_world_p.append(world_signs @ hidden[indices] / 32.0)
        within_world_p_array = np.asarray(within_world_p, dtype=np.float64)
        null_raw = hadamard_null.T @ within_world_p_array / 16.0
        layer_details["within_world_content_coefficient_f64_sha256"] = f64_sha256(within_world_p_array)
        layer_details["null_raw_f64_sha256"] = f64_sha256(null_raw)
        for index in range(4):
            name = f"null_{index}"
            residual, singular_values, rank = _residualize(null_raw[index], genuine)
            raw_norm = float(np.linalg.norm(null_raw[index]))
            residual_norm = float(np.linalg.norm(residual))
            ratio = residual_norm / max(raw_norm, 1e-12)
            if not math.isfinite(ratio) or ratio <= SVD_RELATIVE_TOLERANCE or residual_norm == 0.0:
                raise CrossCodebookRunnerError(f"ineligible null residual {index} at layer {layer}")
            direction = residual / residual_norm
            if abs(float(np.linalg.norm(direction)) - 1.0) > NUMERICAL_TOLERANCE:
                raise CrossCodebookRunnerError("null direction is not unit norm")
            directions[name] = direction
            residuals[name] = residual
            layer_details["directions"][name] = {
                "hadamard_column": index + 1,
                "singular_values": singular_values.tolist(),
                "retained_rank": rank,
                "raw_f64_sha256": f64_sha256(null_raw[index]),
                "residual_f64_sha256": f64_sha256(residual),
                "residual_norm": residual_norm,
                "raw_norm": raw_norm,
                "residual_ratio": ratio,
                "direction_f32_sha256": f32_sha256(direction),
            }
        vectors = [coefficients[0], intercept]
        vectors.extend(coefficients[mask] for mask in range(1, 32))
        vectors.extend(directions[name] for name in DIRECTION_NAMES)
        layer_sidecar = np.asarray(vectors, dtype="<f4")
        if layer_sidecar.shape != (43, widths):
            raise CrossCodebookRunnerError("basis sidecar registry changed")
        sidecar[layer_position] = layer_sidecar
        calculation_vectors = [coefficients[mask] for mask in range(32)]
        calculation_vectors.extend(residuals[name] for name in DIRECTION_NAMES)
        calculation_vectors.extend(within_world_p_array[index] for index in range(16))
        calculation_vectors.extend(null_raw[index] for index in range(4))
        layer_calculations = np.asarray(calculation_vectors, dtype="<f8")
        if layer_calculations.shape != (62, widths):
            raise CrossCodebookRunnerError("fit calculation sidecar registry changed")
        calculations[layer_position] = layer_calculations
        layer_details["applied_layer_f32_sha256"] = f32_sha256(layer_sidecar)
        layer_details["calculation_layer_f64_sha256"] = f64_sha256(
            layer_calculations
        )
        details["layers"][str(layer)] = layer_details
    details["sidecar_shape"] = list(sidecar.shape)
    details["sidecar_logical_sha256"] = f32_sha256(sidecar)
    details["calculation_sidecar_shape"] = list(calculations.shape)
    details["calculation_sidecar_logical_sha256"] = f64_sha256(calculations)
    return sidecar, details, calculations


def compute_fit_basis(
    cells: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    *,
    layer_grid: Sequence[int] = LAYER_GRID,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper returning the applied f32 basis and JSON details."""

    sidecar, details, _ = _compute_fit_basis_full(
        cells, activations, layer_grid=layer_grid
    )
    return sidecar, details


def _fit_cells_in_activation_order(
    plan: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    cells_by_id = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    fit_cells = [cell for cell in plan["cell_registry"] if cell["role"] == "direction_fit"]
    if len(baseline_records) != 512 or len(fit_cells) != 512:
        raise CrossCodebookRunnerError("fit basis row count changed")
    by_row: list[Mapping[str, Any] | None] = [None] * 512
    for record in baseline_records:
        cell_id = record.get("cell_id")
        row = record.get("activation_row")
        if cell_id not in cells_by_id or cells_by_id[cell_id]["role"] != "direction_fit":
            raise CrossCodebookRunnerError("fit baseline cell is outside the fit split")
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < 512 or by_row[row] is not None:
            raise CrossCodebookRunnerError("fit activation row registry is invalid")
        by_row[row] = cells_by_id[cell_id]
    if any(cell is None for cell in by_row):
        raise CrossCodebookRunnerError("fit activation row registry is incomplete")
    return [cell for cell in by_row if cell is not None]


def _fit_basis(
    plan: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    cells = _fit_cells_in_activation_order(plan, baseline_records)
    return compute_fit_basis(cells, activations)


def _fit_basis_with_calculations(
    plan: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    cells = _fit_cells_in_activation_order(plan, baseline_records)
    return _compute_fit_basis_full(cells, activations)


def _load_activation_sidecar(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise CrossCodebookRunnerError(f"activation sidecar is missing: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CrossCodebookRunnerError(f"cannot load activation sidecar: {path}") from error
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("<f4"):
        raise CrossCodebookRunnerError("activation sidecar dtype changed")
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise CrossCodebookRunnerError("activation sidecar shape or values changed")
    return array


def _load_f64_sidecar(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise CrossCodebookRunnerError(f"float64 sidecar is missing: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise CrossCodebookRunnerError(f"cannot load float64 sidecar: {path}") from error
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("<f8"):
        raise CrossCodebookRunnerError("float64 sidecar dtype changed")
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise CrossCodebookRunnerError("float64 sidecar shape or values changed")
    return array


def _validate_stored_diagnostics(value: Mapping[str, Any]) -> None:
    expected = {
        "x_logit",
        "y_logit",
        "x_minus_y_margin",
        "full_vocab_logsumexp",
        "label_probability_mass",
        "greedy_token_id",
        "greedy_logit",
        "maximum_token_ids",
        "maximum_tie_count",
        "full_vocab_logits_sha256",
    }
    if set(value) != expected:
        raise CrossCodebookRunnerError("stored diagnostic schema changed")
    numeric = (
        "x_logit",
        "y_logit",
        "x_minus_y_margin",
        "full_vocab_logsumexp",
        "label_probability_mass",
        "greedy_logit",
    )
    if any(
        isinstance(value[key], bool)
        or not isinstance(value[key], (int, float))
        or not math.isfinite(float(value[key]))
        for key in numeric
    ):
        raise CrossCodebookRunnerError("stored diagnostics contain invalid numbers")
    x = float(value["x_logit"])
    y = float(value["y_logit"])
    if abs(float(value["x_minus_y_margin"]) - (x - y)) > 1e-7:
        raise CrossCodebookRunnerError("stored diagnostic margin changed")
    expected_mass = math.exp(
        float(np.logaddexp(x, y)) - float(value["full_vocab_logsumexp"])
    )
    if abs(float(value["label_probability_mass"]) - expected_mass) > 1e-12:
        raise CrossCodebookRunnerError("stored diagnostic channel mass changed")
    maximum_ids = value["maximum_token_ids"]
    if (
        not isinstance(maximum_ids, list)
        or not maximum_ids
        or maximum_ids != sorted(set(maximum_ids))
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < MODEL_VOCAB_SIZE
            for token in maximum_ids
        )
        or value["maximum_tie_count"] != len(maximum_ids)
        or value["greedy_token_id"] != maximum_ids[0]
    ):
        raise CrossCodebookRunnerError("stored diagnostic maximum registry changed")
    greedy = float(value["greedy_logit"])
    if greedy < max(x, y) or float(value["full_vocab_logsumexp"]) < greedy:
        raise CrossCodebookRunnerError("stored diagnostic maximum ordering changed")
    if (X_TOKEN_ID in maximum_ids) != (abs(greedy - x) <= 1e-7):
        raise CrossCodebookRunnerError("stored X maximum membership changed")
    if (Y_TOKEN_ID in maximum_ids) != (abs(greedy - y) <= 1e-7):
        raise CrossCodebookRunnerError("stored Y maximum membership changed")
    digest = value["full_vocab_logits_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CrossCodebookRunnerError("stored logit digest changed")


def _load_frozen_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reload and independently rebuild every pre-forward artifact."""

    _require_frozen_static_inputs()
    for path in (
        DEFAULT_PLAN_MANIFEST,
        DEFAULT_DESIGN,
        DEFAULT_TOKENIZATION_RECEIPT,
        DEFAULT_DEPENDENCY_LOCK,
    ):
        if not path.is_file():
            raise CrossCodebookRunnerError(f"pre-forward artifact is missing: {path}")
    manifest = _load_json(DEFAULT_PLAN_MANIFEST)
    design = _load_json(DEFAULT_DESIGN)
    receipt = _load_json(DEFAULT_TOKENIZATION_RECEIPT)
    dependency = _load_json(DEFAULT_DEPENDENCY_LOCK)
    required_manifest = {
        "schema_version",
        "status",
        "mode",
        "model_calls",
        "generation_used",
        "biological_model_calls",
        "design_path",
        "design_file_sha256",
        "plan",
        "call_plan_sha256",
        "tokenization_receipt_file_sha256",
        "dependency_lock_file_sha256",
    }
    if set(manifest) != required_manifest:
        raise CrossCodebookRunnerError("plan manifest schema changed")
    if (
        manifest["schema_version"] != PLAN_MANIFEST_SCHEMA
        or manifest["status"] != "PLAN_AND_DESIGN_FROZEN_NO_FORWARD"
        or manifest["model_calls"] != 0
        or manifest["generation_used"] is not False
        or manifest["biological_model_calls"] != 0
    ):
        raise CrossCodebookRunnerError("plan manifest is not a zero-forward freeze")
    if (
        manifest["design_path"] != str(DEFAULT_DESIGN)
        or manifest["design_file_sha256"] != file_sha256(DEFAULT_DESIGN)
        or manifest["tokenization_receipt_file_sha256"]
        != file_sha256(DEFAULT_TOKENIZATION_RECEIPT)
        or manifest["dependency_lock_file_sha256"]
        != file_sha256(DEFAULT_DEPENDENCY_LOCK)
    ):
        raise CrossCodebookRunnerError("pre-forward file binding changed")
    plan = manifest["plan"]
    if not isinstance(plan, dict):
        raise CrossCodebookRunnerError("plan manifest does not embed an object plan")
    validate_plan(plan, design)
    if manifest["call_plan_sha256"] != plan["call_plan_sha256"]:
        raise CrossCodebookRunnerError("manifest call-plan hash changed")
    locks = design.get("locks")
    if not isinstance(locks, dict):
        raise CrossCodebookRunnerError("design locks are invalid")
    if (
        locks.get("runner", {}).get("path") != str(Path(__file__))
        or locks.get("runner", {}).get("sha256") != file_sha256(Path(__file__))
        or locks.get("analyzer", {}).get("path") != str(DEFAULT_ANALYZER)
        or locks.get("analyzer", {}).get("sha256") != file_sha256(DEFAULT_ANALYZER)
    ):
        raise CrossCodebookRunnerError("current runner or analyzer differs from design")
    if (
        locks.get("tokenization_receipt", {}).get("file_sha256")
        != file_sha256(DEFAULT_TOKENIZATION_RECEIPT)
        or locks.get("tokenization_receipt", {}).get("canonical_sha256")
        != receipt.get("canonical_sha256")
        or locks.get("dependency_lock", {}).get("file_sha256")
        != file_sha256(DEFAULT_DEPENDENCY_LOCK)
        or locks.get("dependency_lock", {}).get("canonical_sha256")
        != dependency.get("canonical_sha256")
    ):
        raise CrossCodebookRunnerError("receipt or dependency lock changed")
    tokenizer, _ = _config_and_tokenizer()
    rebuilt_plan, rebuilt_receipt, rebuilt_dependency = build_plan(
        tokenizer, DEFAULT_ANALYZER
    )
    if rebuilt_plan != plan or rebuilt_receipt != receipt or rebuilt_dependency != dependency:
        raise CrossCodebookRunnerError("independent pre-forward plan rebuild differs")
    rebuilt_design = design_from_plan(rebuilt_plan)
    if rebuilt_design != design:
        raise CrossCodebookRunnerError("independent design rebuild differs")
    return plan, design


def _require_absent(paths: Sequence[Path], phase: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise CrossCodebookRunnerError(
            f"refusing to re-enter {phase}; phase artifact already exists: {existing[0]}"
        )


def _require_no_downstream_artifacts(phase: str) -> None:
    ordered = (
        "fit-baseline",
        "localization-baseline",
        "localization-patch",
        "holdout-baseline",
        "holdout-patch",
    )
    if phase not in ordered:
        raise CrossCodebookRunnerError("unknown phase for monotone-state check")
    later_paths: list[Path] = []
    for later in ordered[ordered.index(phase) + 1 :]:
        later_paths.extend(PHASE_PATHS[later].values())
    analysis_and_locks = {
        "fit-baseline": [
            RESULT_ROOT / "fit_basis_analysis.json",
            DEFAULT_BASIS_SIDECAR,
            DEFAULT_BASIS_CALCULATIONS,
            DEFAULT_BASIS_DETAILS,
            DEFAULT_BASIS_LOCK,
            DEFAULT_LOCALIZATION_ENTRY,
            RESULT_ROOT / "localization_baseline_analysis.json",
            RESULT_ROOT / "localization_analysis.json",
            DEFAULT_LAYER_LOCK,
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ],
        "localization-baseline": [
            DEFAULT_LOCALIZATION_ENTRY,
            RESULT_ROOT / "localization_baseline_analysis.json",
            RESULT_ROOT / "localization_analysis.json",
            DEFAULT_LAYER_LOCK,
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ],
        "localization-patch": [
            RESULT_ROOT / "localization_analysis.json",
            DEFAULT_LAYER_LOCK,
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ],
        "holdout-baseline": [
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ],
        "holdout-patch": [
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis.md",
            RESULT_ROOT / "analysis_manifest.json",
        ],
    }
    later_paths.extend(analysis_and_locks[phase])
    existing = [path for path in later_paths if path.exists()]
    if existing:
        raise CrossCodebookRunnerError(
            f"refusing non-monotone {phase}; downstream artifact exists: {existing[0]}"
        )


def _write_execution_attempt(
    path: Path,
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    prerequisite_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(DEFAULT_PLAN_MANIFEST),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "runner_sha256": file_sha256(Path(__file__)),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "prerequisite_bindings": dict(prerequisite_bindings or {}),
        "model_calls_before_attempt": 0,
        "generation_used": False,
        "biological_model_calls": 0,
    }
    write_json(path, value)
    return value


def _execution_manifest(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    attempt_path: Path,
    records_path: Path,
    records: Sequence[Mapping[str, Any]],
    activations_path: Path | None,
    activations: np.ndarray | None,
    patched_activations_path: Path | None = None,
    patched_activations: np.ndarray | None = None,
    prerequisite_bindings: Mapping[str, str] | None = None,
    selected_layer: int | None = None,
) -> dict[str, Any]:
    if not attempt_path.is_file() or not records_path.is_file():
        raise CrossCodebookRunnerError("execution manifest input artifact is missing")
    if (activations_path is None) != (activations is None):
        raise CrossCodebookRunnerError("activation manifest arguments are inconsistent")
    if (patched_activations_path is None) != (patched_activations is None):
        raise CrossCodebookRunnerError(
            "patched-activation manifest arguments are inconsistent"
        )
    activation_receipt = None
    if activations_path is not None and activations is not None:
        if not activations_path.is_file():
            raise CrossCodebookRunnerError("activation sidecar is missing at manifest time")
        activation_receipt = {
            "path": str(activations_path),
            "file_sha256": file_sha256(activations_path),
            "logical_sha256": f32_sha256(activations),
            "shape": list(activations.shape),
            "dtype": "<f4",
            "logical_id_map": {
                str(record["cell_id"]): {
                    "activation_row": record["activation_row"],
                    "activation_sha256": record["activation_sha256"],
                    "activation_layer_sha256": record["activation_layer_sha256"],
                }
                for record in records
            },
        }
    patched_activation_receipt = None
    if patched_activations_path is not None and patched_activations is not None:
        if not patched_activations_path.is_file():
            raise CrossCodebookRunnerError(
                "patched activation sidecar is missing at manifest time"
            )
        patched_activation_receipt = {
            "path": str(patched_activations_path),
            "file_sha256": file_sha256(patched_activations_path),
            "logical_sha256": f32_sha256(patched_activations),
            "shape": list(patched_activations.shape),
            "dtype": "<f4",
            "logical_id_map": {
                str(record["record_id"]): {
                    "patched_activation_row": record["patched_activation_row"],
                    "patched_activation_sha256": record[
                        "patched_activation_sha256"
                    ],
                }
                for record in records
            },
        }
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "status": "EXECUTION_COMPLETE_NOT_ANALYZED",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(DEFAULT_PLAN_MANIFEST),
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "runner_sha256": file_sha256(Path(__file__)),
        "analyzer_sha256": design["locks"]["analyzer"]["sha256"],
        "attempt": {
            "path": str(attempt_path),
            "file_sha256": file_sha256(attempt_path),
        },
        "records": {
            "path": str(records_path),
            "file_sha256": file_sha256(records_path),
            "canonical_sha256": canonical_sha256([dict(record) for record in records]),
            "count": len(records),
        },
        "activations": activation_receipt,
        "patched_activations": patched_activation_receipt,
        "prerequisite_bindings": dict(prerequisite_bindings or {}),
        "selected_layer": selected_layer,
        "model_calls": len(records),
        "generation_used": False,
        "biological_model_calls": 0,
        "partial_resume_allowed": False,
    }


def _validate_execution_manifest(
    phase: str,
    *,
    expected_count: int,
    expected_activation_shape: tuple[int, ...] | None,
    expected_patched_activation_shape: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    paths = PHASE_PATHS[phase]
    manifest = _load_json(paths["manifest"])
    expected_manifest_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "design_file_sha256",
        "plan_manifest_file_sha256",
        "preregistration_sha256",
        "runner_sha256",
        "analyzer_sha256",
        "attempt",
        "records",
        "activations",
        "patched_activations",
        "prerequisite_bindings",
        "selected_layer",
        "model_calls",
        "generation_used",
        "biological_model_calls",
        "partial_resume_allowed",
    }
    if set(manifest) != expected_manifest_keys:
        raise CrossCodebookRunnerError(f"{phase} execution manifest schema changed")
    plan_manifest = _load_json(DEFAULT_PLAN_MANIFEST)
    design = _load_json(DEFAULT_DESIGN)
    if (
        manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA
        or manifest.get("status") != "EXECUTION_COMPLETE_NOT_ANALYZED"
        or manifest.get("phase") != phase
        or manifest.get("model_calls") != expected_count
        or manifest.get("generation_used") is not False
        or manifest.get("biological_model_calls") != 0
        or manifest.get("partial_resume_allowed") is not False
        or manifest.get("call_plan_sha256")
        != plan_manifest.get("call_plan_sha256")
        or manifest.get("analyzer_sha256")
        != design.get("locks", {}).get("analyzer", {}).get("sha256")
    ):
        raise CrossCodebookRunnerError(f"{phase} execution manifest changed")
    if (
        manifest.get("design_file_sha256") != file_sha256(DEFAULT_DESIGN)
        or manifest.get("plan_manifest_file_sha256")
        != file_sha256(DEFAULT_PLAN_MANIFEST)
        or manifest.get("runner_sha256") != file_sha256(Path(__file__))
        or manifest.get("preregistration_sha256") != FROZEN_PREREGISTRATION_SHA256
    ):
        raise CrossCodebookRunnerError(f"{phase} execution lock changed")
    attempt_path = paths["attempt"]
    attempt = _load_json(attempt_path)
    expected_attempt_keys = {
        "schema_version",
        "status",
        "phase",
        "call_plan_sha256",
        "design_file_sha256",
        "plan_manifest_file_sha256",
        "preregistration_sha256",
        "runner_sha256",
        "analyzer_sha256",
        "prerequisite_bindings",
        "model_calls_before_attempt",
        "generation_used",
        "biological_model_calls",
    }
    if (
        set(attempt) != expected_attempt_keys
        or attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("status") != "EXECUTION_ATTEMPT_STARTED_IMMUTABLE"
        or attempt.get("phase") != phase
        or attempt.get("call_plan_sha256") != manifest["call_plan_sha256"]
        or attempt.get("design_file_sha256") != file_sha256(DEFAULT_DESIGN)
        or attempt.get("plan_manifest_file_sha256")
        != file_sha256(DEFAULT_PLAN_MANIFEST)
        or attempt.get("preregistration_sha256")
        != FROZEN_PREREGISTRATION_SHA256
        or attempt.get("runner_sha256") != file_sha256(Path(__file__))
        or attempt.get("analyzer_sha256") != manifest["analyzer_sha256"]
        or attempt.get("prerequisite_bindings")
        != manifest.get("prerequisite_bindings")
        or attempt.get("model_calls_before_attempt") != 0
        or attempt.get("generation_used") is not False
        or attempt.get("biological_model_calls") != 0
        or manifest.get("attempt")
        != {"path": str(attempt_path), "file_sha256": file_sha256(attempt_path)}
    ):
        raise CrossCodebookRunnerError(f"{phase} execution attempt changed")
    prerequisite_paths = {
        "basis_lock_file_sha256": DEFAULT_BASIS_LOCK,
        "fit_execution_manifest_file_sha256": PHASE_PATHS["fit-baseline"][
            "manifest"
        ],
        "localization_baseline_execution_manifest_file_sha256": PHASE_PATHS[
            "localization-baseline"
        ]["manifest"],
        "localization_baseline_records_file_sha256": PHASE_PATHS[
            "localization-baseline"
        ]["records"],
        "localization_entry_file_sha256": DEFAULT_LOCALIZATION_ENTRY,
        "localization_patch_execution_manifest_file_sha256": PHASE_PATHS[
            "localization-patch"
        ]["manifest"],
        "layer_lock_file_sha256": DEFAULT_LAYER_LOCK,
        "holdout_entry_file_sha256": DEFAULT_HOLDOUT_ENTRY,
        "holdout_baseline_execution_manifest_file_sha256": PHASE_PATHS[
            "holdout-baseline"
        ]["manifest"],
        "holdout_baseline_records_file_sha256": PHASE_PATHS["holdout-baseline"]
        ["records"],
    }
    prerequisites = manifest.get("prerequisite_bindings")
    required_prerequisite_keys = {
        "fit-baseline": set(),
        "localization-baseline": {
            "basis_lock_file_sha256",
            "fit_execution_manifest_file_sha256",
        },
        "localization-patch": {
            "basis_lock_file_sha256",
            "localization_baseline_execution_manifest_file_sha256",
            "localization_baseline_records_file_sha256",
            "localization_entry_file_sha256",
        },
        "holdout-baseline": {
            "basis_lock_file_sha256",
            "layer_lock_file_sha256",
        },
        "holdout-patch": {
            "basis_lock_file_sha256",
            "layer_lock_file_sha256",
            "holdout_entry_file_sha256",
            "holdout_baseline_execution_manifest_file_sha256",
            "holdout_baseline_records_file_sha256",
        },
    }
    if (
        not isinstance(prerequisites, dict)
        or set(prerequisites) != required_prerequisite_keys[phase]
        or any(
        key not in prerequisite_paths
        or not prerequisite_paths[key].is_file()
        or value != file_sha256(prerequisite_paths[key])
        for key, value in prerequisites.items()
        )
    ):
        raise CrossCodebookRunnerError(f"{phase} prerequisite binding changed")
    selected_layer = manifest.get("selected_layer")
    if phase in {"holdout-baseline", "holdout-patch"}:
        if (
            isinstance(selected_layer, bool)
            or not isinstance(selected_layer, int)
            or selected_layer not in LAYER_GRID
        ):
            raise CrossCodebookRunnerError(f"{phase} selected layer is invalid")
    elif selected_layer is not None:
        raise CrossCodebookRunnerError(f"{phase} unexpectedly binds a layer")
    records_path = paths["records"]
    records = load_jsonl(records_path)
    receipt = manifest.get("records")
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {"path", "file_sha256", "canonical_sha256", "count"}
        or receipt.get("path") != str(records_path)
        or receipt.get("file_sha256") != file_sha256(records_path)
        or receipt.get("canonical_sha256") != canonical_sha256(records)
        or receipt.get("count") != expected_count
        or len(records) != expected_count
        or len({record.get("record_id") for record in records}) != expected_count
    ):
        raise CrossCodebookRunnerError(f"{phase} record artifact changed")
    activations = None
    if expected_activation_shape is None:
        if manifest.get("activations") is not None:
            raise CrossCodebookRunnerError(f"{phase} unexpectedly binds activations")
    else:
        activation_path = paths.get("activations")
        activation_receipt = manifest.get("activations")
        if not isinstance(activation_path, Path) or not isinstance(
            activation_receipt, dict
        ):
            raise CrossCodebookRunnerError(f"{phase} activation receipt is missing")
        activations = _load_activation_sidecar(activation_path, expected_activation_shape)
        expected_activation_map = {
            str(record["cell_id"]): {
                "activation_row": record["activation_row"],
                "activation_sha256": record["activation_sha256"],
                "activation_layer_sha256": record["activation_layer_sha256"],
            }
            for record in records
        }
        if (
            set(activation_receipt)
            != {
                "path",
                "file_sha256",
                "logical_sha256",
                "shape",
                "dtype",
                "logical_id_map",
            }
            or activation_receipt.get("path") != str(activation_path)
            or activation_receipt.get("file_sha256") != file_sha256(activation_path)
            or activation_receipt.get("logical_sha256") != f32_sha256(activations)
            or activation_receipt.get("shape") != list(expected_activation_shape)
            or activation_receipt.get("dtype") != "<f4"
            or activation_receipt.get("logical_id_map")
            != expected_activation_map
        ):
            raise CrossCodebookRunnerError(f"{phase} activation binding changed")
    patched = None
    if expected_patched_activation_shape is None:
        if manifest.get("patched_activations") is not None:
            raise CrossCodebookRunnerError(
                f"{phase} unexpectedly binds patched activations"
            )
    else:
        patched_path = paths.get("patched_activations")
        patched_receipt = manifest.get("patched_activations")
        if not isinstance(patched_path, Path) or not isinstance(
            patched_receipt, dict
        ):
            raise CrossCodebookRunnerError(
                f"{phase} patched activation receipt is missing"
            )
        patched = _load_activation_sidecar(
            patched_path, expected_patched_activation_shape
        )
        expected_patched_map = {
            str(record["record_id"]): {
                "patched_activation_row": record["patched_activation_row"],
                "patched_activation_sha256": record[
                    "patched_activation_sha256"
                ],
            }
            for record in records
        }
        if (
            set(patched_receipt)
            != {
                "path",
                "file_sha256",
                "logical_sha256",
                "shape",
                "dtype",
                "logical_id_map",
            }
            or patched_receipt.get("path") != str(patched_path)
            or patched_receipt.get("file_sha256") != file_sha256(patched_path)
            or patched_receipt.get("logical_sha256") != f32_sha256(patched)
            or patched_receipt.get("shape")
            != list(expected_patched_activation_shape)
            or patched_receipt.get("dtype") != "<f4"
            or patched_receipt.get("logical_id_map") != expected_patched_map
            or any(
                record.get("patched_activation_row") != index
                or record.get("patched_activation_sha256")
                != f32_sha256(patched[index])
                for index, record in enumerate(records)
            )
        ):
            raise CrossCodebookRunnerError(
                f"{phase} patched activation binding changed"
            )
    _validate_records_against_plan(
        phase,
        plan_manifest["plan"],
        records,
        activations,
        patched,
        selected_layer=selected_layer,
    )
    return manifest, records, activations


def _validate_records_against_plan(
    phase: str,
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray | None,
    patched_activations: np.ndarray | None,
    *,
    selected_layer: int | None,
) -> None:
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    prompts = {prompt["cell_id"]: prompt for prompt in plan["prompts"]}
    baseline_roles = {
        "fit-baseline": "direction_fit",
        "localization-baseline": "localization",
        "holdout-baseline": "holdout",
    }
    if phase in baseline_roles:
        if activations is None or patched_activations is not None:
            raise CrossCodebookRunnerError("baseline record sidecar contract changed")
        role = baseline_roles[phase]
        templates = _phase_baseline_templates(plan, role)
        layers = (
            list(LAYER_GRID)
            if role != "holdout"
            else [int(selected_layer)]
        )
        if len(records) != len(templates):
            raise CrossCodebookRunnerError("baseline record count changed")
        for index, (record, template) in enumerate(zip(records, templates, strict=True)):
            cell = cells[template["cell_id"]]
            prompt = prompts[template["cell_id"]]
            if role == "holdout":
                activation = np.ascontiguousarray(
                    activations[index][None, :], dtype="<f4"
                )
            else:
                activation = np.ascontiguousarray(activations[index], dtype="<f4")
            diagnostics = record.get("diagnostics")
            trace = record.get("forward_trace")
            if not isinstance(diagnostics, dict) or not isinstance(trace, dict):
                raise CrossCodebookRunnerError("baseline diagnostics or trace is invalid")
            _validate_stored_diagnostics(diagnostics)
            expected = _baseline_record(
                plan,
                prompt,
                cell,
                phase=phase,
                activation_row=index,
                layers=layers,
                activations=activation,
                diagnostics=diagnostics,
                trace=trace,
            )
            if dict(record) != expected:
                raise CrossCodebookRunnerError("baseline record reconstruction changed")
        return
    patch_roles = {
        "localization-patch": (
            "localization",
            "localization-baseline",
            (256, 5, MODEL_WIDTH),
        ),
        "holdout-patch": (
            "holdout",
            "holdout-baseline",
            (1024, MODEL_WIDTH),
        ),
    }
    if phase not in patch_roles:
        raise CrossCodebookRunnerError("record phase is outside the frozen registry")
    if activations is not None or patched_activations is None:
        raise CrossCodebookRunnerError("patch record sidecar contract changed")
    role, baseline_phase, baseline_shape = patch_roles[phase]
    _, baseline_records, baseline_activations = _validate_execution_manifest(
        baseline_phase,
        expected_count=baseline_shape[0],
        expected_activation_shape=baseline_shape,
    )
    if baseline_activations is None:
        raise CrossCodebookRunnerError("patch baseline sidecar is missing")
    baselines = {record["cell_id"]: record for record in baseline_records}
    basis_sidecar = _load_activation_sidecar(
        DEFAULT_BASIS_SIDECAR, (5, 43, MODEL_WIDTH)
    )
    basis_details = _load_json(DEFAULT_BASIS_DETAILS)
    templates = _phase_patch_templates(plan, role)
    if len(records) != len(templates) or len(patched_activations) != len(templates):
        raise CrossCodebookRunnerError("patch record count changed")
    for index, (record, template) in enumerate(zip(records, templates, strict=True)):
        layer = template["layer"] if role == "localization" else selected_layer
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise CrossCodebookRunnerError("patch record layer changed")
        recipient_cell = cells[template["recipient_cell_id"]]
        recipient_prompt = prompts[recipient_cell["cell_id"]]
        recipient_baseline = baselines[recipient_cell["cell_id"]]
        source_baseline = (
            None
            if template["source_cell_id"] is None
            else baselines[template["source_cell_id"]]
        )
        recipient_activation = _activation_for_record(
            recipient_baseline, baseline_activations, layer=layer, role=role
        )
        source_activation = (
            None
            if source_baseline is None
            else _activation_for_record(
                source_baseline, baseline_activations, layer=layer, role=role
            )
        )
        fit_center, directions = _basis_vectors_at_layer(
            basis_sidecar, basis_details, layer
        )
        direction = (
            None
            if template["direction_name"] is None
            else directions[template["direction_name"]]
        )
        center = (
            fit_center
            if template["operation"] in {"erasure", "rescue"}
            else None
        )
        reconstructed = expected_intervention_activation(
            recipient_activation,
            operation=template["operation"],
            source=source_activation,
            direction=direction,
            center=center,
        )
        actual = np.ascontiguousarray(patched_activations[index], dtype="<f4")
        reconstruction_error = float(
            np.linalg.norm(actual.astype(np.float64) - reconstructed.astype(np.float64))
        )
        reconstruction_tolerance = NUMERICAL_TOLERANCE * max(
            1.0, float(np.linalg.norm(reconstructed.astype(np.float64)))
        )
        if reconstruction_error > reconstruction_tolerance:
            raise CrossCodebookRunnerError("patched activation formula changed")
        diagnostics = record.get("diagnostics")
        trace = record.get("hook_trace")
        if not isinstance(diagnostics, dict) or not isinstance(trace, dict):
            raise CrossCodebookRunnerError("patch diagnostics or trace is invalid")
        _validate_stored_diagnostics(diagnostics)
        if (
            trace.get("post_activation_sha256") != f32_sha256(actual)
            or trace.get("expected_activation_sha256") != f32_sha256(reconstructed)
            or not isinstance(trace.get("post_expected_l2_error"), (int, float))
            or abs(float(trace["post_expected_l2_error"]) - reconstruction_error)
            > 1e-12
        ):
            raise CrossCodebookRunnerError("patch trace activation commitment changed")
        expected = _patch_record(
            plan,
            template,
            recipient_cell,
            recipient_prompt,
            phase=phase,
            layer=layer,
            patched_activation_row=index,
            recipient_baseline=recipient_baseline,
            source_baseline=source_baseline,
            recipient_activation=recipient_activation,
            source_activation=source_activation,
            direction=direction,
            center=center,
            diagnostics=diagnostics,
            trace=trace,
        )
        if dict(record) != expected:
            raise CrossCodebookRunnerError("patch record reconstruction changed")


def _load_basis_artifacts(
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not DEFAULT_BASIS_LOCK.is_file():
        raise CrossCodebookRunnerError("fit basis lock is missing")
    lock = _load_json(DEFAULT_BASIS_LOCK)
    if (
        lock.get("schema_version")
        != "coherent-readout-v3-cross-codebook-basis-lock-v1"
        or lock.get("status") != "FIT_BASIS_LOCKED_LOCALIZATION_AUTHORIZED"
        or lock.get("call_plan_sha256") != plan["call_plan_sha256"]
        or lock.get("engineering_pass") is not True
        or lock.get("behavioral_admission_pass") is not True
    ):
        raise CrossCodebookRunnerError("fit basis lock does not authorize localization")
    sidecar_receipt = lock.get("basis_sidecar")
    details_receipt = lock.get("basis_details")
    calculations_receipt = lock.get("basis_calculations")
    if (
        not isinstance(sidecar_receipt, dict)
        or not isinstance(details_receipt, dict)
        or not isinstance(calculations_receipt, dict)
    ):
        raise CrossCodebookRunnerError("fit basis artifact receipts are missing")
    sidecar = _load_activation_sidecar(DEFAULT_BASIS_SIDECAR, (5, 43, MODEL_WIDTH))
    calculations = _load_f64_sidecar(
        DEFAULT_BASIS_CALCULATIONS, (5, 62, MODEL_WIDTH)
    )
    details = _load_json(DEFAULT_BASIS_DETAILS)
    if (
        sidecar_receipt.get("path") != str(DEFAULT_BASIS_SIDECAR)
        or sidecar_receipt.get("file_sha256") != file_sha256(DEFAULT_BASIS_SIDECAR)
        or sidecar_receipt.get("logical_sha256") != f32_sha256(sidecar)
        or sidecar_receipt.get("shape") != [5, 43, MODEL_WIDTH]
        or details_receipt.get("path") != str(DEFAULT_BASIS_DETAILS)
        or details_receipt.get("file_sha256") != file_sha256(DEFAULT_BASIS_DETAILS)
        or details_receipt.get("canonical_sha256") != canonical_sha256(details)
        or calculations_receipt.get("path") != str(DEFAULT_BASIS_CALCULATIONS)
        or calculations_receipt.get("file_sha256")
        != file_sha256(DEFAULT_BASIS_CALCULATIONS)
        or calculations_receipt.get("logical_sha256") != f64_sha256(calculations)
        or calculations_receipt.get("shape") != [5, 62, MODEL_WIDTH]
        or calculations_receipt.get("dtype") != "<f8"
    ):
        raise CrossCodebookRunnerError("fit basis artifact binding changed")
    if (
        details.get("schema_version") != BASIS_SCHEMA
        or details.get("layer_grid") != list(LAYER_GRID)
        or details.get("sidecar_shape") != [5, 43, MODEL_WIDTH]
        or details.get("sidecar_logical_sha256") != f32_sha256(sidecar)
        or details.get("calculation_sidecar_shape") != [5, 62, MODEL_WIDTH]
        or details.get("calculation_sidecar_logical_sha256")
        != f64_sha256(calculations)
        or details.get("registry")
        != [
            "walsh_00_intercept",
            "fit_intercept",
            *[f"walsh_{mask:02d}" for mask in range(1, 32)],
            *[f"direction_{name}" for name in DIRECTION_NAMES],
        ]
    ):
        raise CrossCodebookRunnerError("fit basis details changed")
    fit_manifest, fit_records, fit_activations = _validate_execution_manifest(
        "fit-baseline",
        expected_count=512,
        expected_activation_shape=(512, 5, MODEL_WIDTH),
    )
    if (
        lock.get("fit_execution_manifest_file_sha256")
        != file_sha256(PHASE_PATHS["fit-baseline"]["manifest"])
        or fit_activations is None
    ):
        raise CrossCodebookRunnerError("fit basis lock lost its fit execution binding")
    rebuilt_sidecar, rebuilt_details, rebuilt_calculations = (
        _fit_basis_with_calculations(plan, fit_records, fit_activations)
    )
    if (
        not np.array_equal(rebuilt_sidecar, sidecar)
        or rebuilt_details != details
        or not np.array_equal(rebuilt_calculations, calculations)
    ):
        raise CrossCodebookRunnerError("fit basis independent reconstruction changed")
    if fit_manifest["call_plan_sha256"] != plan["call_plan_sha256"]:
        raise CrossCodebookRunnerError("fit basis plan binding changed")
    return sidecar, details, lock


def _basis_vectors_at_layer(
    sidecar: np.ndarray,
    details: Mapping[str, Any],
    layer: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if layer not in LAYER_GRID:
        raise CrossCodebookRunnerError("basis layer is outside the frozen grid")
    registry = details["registry"]
    position = LAYER_GRID.index(layer)
    center = np.ascontiguousarray(
        sidecar[position, registry.index("fit_intercept")], dtype="<f4"
    )
    directions = {
        name: np.ascontiguousarray(
            sidecar[position, registry.index(f"direction_{name}")], dtype="<f4"
        )
        for name in DIRECTION_NAMES
    }
    if any(
        abs(float(np.linalg.norm(direction.astype(np.float64))) - 1.0) > 1e-6
        for direction in directions.values()
    ):
        raise CrossCodebookRunnerError("applied fit direction is not unit norm")
    return center, directions


def _load_layer_lock(plan: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    if not DEFAULT_LAYER_LOCK.is_file():
        raise CrossCodebookRunnerError("localization layer lock is missing")
    lock = _load_json(DEFAULT_LAYER_LOCK)
    selected = lock.get("selected_layer")
    if (
        lock.get("schema_version")
        != "coherent-readout-v3-cross-codebook-layer-lock-v1"
        or lock.get("status")
        != "LOCALIZATION_LAYER_LOCKED_HOLDOUT_BASELINE_AUTHORIZED"
        or lock.get("call_plan_sha256") != plan["call_plan_sha256"]
        or lock.get("engineering_pass") is not True
        or lock.get("behavioral_admission_pass") is not True
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in LAYER_GRID
    ):
        raise CrossCodebookRunnerError("localization lock does not authorize holdout")
    for phase, count, shape, patched_shape in (
        ("localization-baseline", 256, (256, 5, MODEL_WIDTH), None),
        ("localization-patch", 4480, None, (4480, MODEL_WIDTH)),
    ):
        _validate_execution_manifest(
            phase,
            expected_count=count,
            expected_activation_shape=shape,
            expected_patched_activation_shape=patched_shape,
        )
    required_hashes = {
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_patch_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["localization-patch"]["manifest"]
        ),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "localization_entry_file_sha256": file_sha256(
            DEFAULT_LOCALIZATION_ENTRY
        ),
    }
    if any(lock.get(key) != value for key, value in required_hashes.items()):
        raise CrossCodebookRunnerError("localization lock prerequisite hash changed")
    return selected, lock


def _load_localization_entry(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not DEFAULT_LOCALIZATION_ENTRY.is_file():
        raise CrossCodebookRunnerError("localization baseline entry is missing")
    entry = _load_json(DEFAULT_LOCALIZATION_ENTRY)
    if (
        entry.get("schema_version")
        != "coherent-readout-v3-cross-codebook-localization-entry-v1"
        or entry.get("status")
        != "LOCALIZATION_BASELINE_ADMITTED_PATCH_AUTHORIZED"
        or entry.get("call_plan_sha256") != plan["call_plan_sha256"]
        or entry.get("behavioral_admission_pass") is not True
        or entry.get("engineering_pass") is not True
        or entry.get("localization_baseline_execution_manifest_file_sha256")
        != file_sha256(PHASE_PATHS["localization-baseline"]["manifest"])
        or entry.get("basis_lock_file_sha256") != file_sha256(DEFAULT_BASIS_LOCK)
    ):
        raise CrossCodebookRunnerError(
            "localization baseline did not authorize patch execution"
        )
    _validate_execution_manifest(
        "localization-baseline",
        expected_count=256,
        expected_activation_shape=(256, 5, MODEL_WIDTH),
    )
    return entry


def _recompute_analyzer_authorization(stage: str, path: Path) -> dict[str, Any]:
    """Re-run the frozen analyzer and require exact equality with its authority."""

    try:
        if __package__:
            from . import analyze_coherent_readout_v3_cross_codebook as analyzer
        else:
            import analyze_coherent_readout_v3_cross_codebook as analyzer
    except ImportError as error:
        raise CrossCodebookRunnerError("cannot import the frozen V3 analyzer") from error
    actions = {
        "fit_basis": analyzer.analyze_fit_basis,
        "localization_baseline": analyzer.analyze_localization_baseline,
        "localization": analyzer.analyze_localization,
        "holdout_baseline": analyzer.analyze_holdout_baseline,
    }
    if stage not in actions:
        raise CrossCodebookRunnerError("unknown analyzer authorization stage")
    try:
        result = actions[stage]()
    except Exception as error:
        raise CrossCodebookRunnerError(
            f"frozen analyzer could not recompute {stage} authorization"
        ) from error
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[1], dict)
    ):
        raise CrossCodebookRunnerError(
            f"frozen analyzer returned no {stage} authority object"
        )
    expected = result[1]
    if not path.is_file() or _load_json(path) != expected:
        raise CrossCodebookRunnerError(
            f"stored {stage} authority differs from independent recomputation"
        )
    return expected


def _load_holdout_entry(plan: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    if not DEFAULT_HOLDOUT_ENTRY.is_file():
        raise CrossCodebookRunnerError("holdout entry authorization is missing")
    entry = _load_json(DEFAULT_HOLDOUT_ENTRY)
    selected = entry.get("selected_layer")
    if (
        entry.get("schema_version")
        != "coherent-readout-v3-cross-codebook-holdout-entry-v1"
        or entry.get("status") != "HOLDOUT_BASELINE_ADMITTED_PATCH_AUTHORIZED"
        or entry.get("call_plan_sha256") != plan["call_plan_sha256"]
        or entry.get("behavioral_admission_pass") is not True
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in LAYER_GRID
    ):
        raise CrossCodebookRunnerError("holdout baseline did not authorize patching")
    locked_layer, _ = _load_layer_lock(plan)
    if selected != locked_layer:
        raise CrossCodebookRunnerError("holdout entry layer differs from localization lock")
    _validate_execution_manifest(
        "holdout-baseline",
        expected_count=1024,
        expected_activation_shape=(1024, MODEL_WIDTH),
    )
    required_hashes = {
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["holdout-baseline"]["manifest"]
        ),
        "layer_lock_file_sha256": file_sha256(DEFAULT_LAYER_LOCK),
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
    }
    if any(entry.get(key) != value for key, value in required_hashes.items()):
        raise CrossCodebookRunnerError("holdout entry prerequisite hash changed")
    return selected, entry


def _phase_baseline_templates(
    plan: Mapping[str, Any], role: str
) -> list[dict[str, Any]]:
    templates = [
        dict(template)
        for template in plan["baseline_templates"]
        if template["role"] == role
    ]
    expected = {"direction_fit": 512, "localization": 256, "holdout": 1024}
    if role not in expected or len(templates) != expected[role]:
        raise CrossCodebookRunnerError("phase baseline template count changed")
    return sorted(templates, key=lambda value: value["cell_id"])


def _phase_patch_templates(
    plan: Mapping[str, Any], role: str
) -> list[dict[str, Any]]:
    templates = [
        dict(template)
        for template in plan["patch_templates"]
        if template["role"] == role
    ]
    expected = {"localization": 4480, "holdout": 5376}
    if role not in expected or len(templates) != expected[role]:
        raise CrossCodebookRunnerError("phase patch template count changed")
    return templates


def _execute_baselines(
    model: Any,
    plan: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    layers: Sequence[int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    templates = _phase_baseline_templates(plan, role)
    prompts = {prompt["cell_id"]: prompt for prompt in plan["prompts"]}
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    records: list[dict[str, Any]] = []
    captured: list[np.ndarray] = []
    for activation_row, template in enumerate(templates):
        cell_id = template["cell_id"]
        prompt = prompts.get(cell_id)
        cell = cells.get(cell_id)
        if prompt is None or cell is None or cell["role"] != role:
            raise CrossCodebookRunnerError("baseline template does not resolve")
        if _baseline_template(cell, prompt) != template:
            raise CrossCodebookRunnerError("baseline template identity changed")
        diagnostics, activation, trace = _baseline_forward(
            model, prompt, layers=layers
        )
        record = _baseline_record(
            plan,
            prompt,
            cell,
            phase=phase,
            activation_row=activation_row,
            layers=layers,
            activations=activation,
            diagnostics=diagnostics,
            trace=trace,
        )
        records.append(record)
        captured.append(activation)
        if (activation_row + 1) % 32 == 0 or activation_row + 1 == len(templates):
            print(
                canonical_json(
                    {
                        "phase": phase,
                        "completed": activation_row + 1,
                        "total": len(templates),
                    }
                ),
                flush=True,
            )
    matrix = np.ascontiguousarray(np.stack(captured), dtype="<f4")
    expected_shape = (len(templates), len(layers), MODEL_WIDTH)
    if matrix.shape != expected_shape:
        raise CrossCodebookRunnerError("baseline activation matrix shape changed")
    if role == "holdout":
        if len(layers) != 1:
            raise CrossCodebookRunnerError("holdout must capture one selected layer")
        matrix = np.ascontiguousarray(matrix[:, 0, :], dtype="<f4")
    return records, matrix


def _activation_for_record(
    record: Mapping[str, Any],
    activations: np.ndarray,
    *,
    layer: int,
    role: str,
) -> np.ndarray:
    row = record.get("activation_row")
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(
        activations
    ):
        raise CrossCodebookRunnerError("baseline activation row is invalid")
    if role == "localization":
        if activations.ndim != 3 or layer not in LAYER_GRID:
            raise CrossCodebookRunnerError("localization activation lookup changed")
        value = activations[row, LAYER_GRID.index(layer)]
    elif role == "holdout":
        if activations.ndim != 2:
            raise CrossCodebookRunnerError("holdout activation lookup changed")
        value = activations[row]
    else:
        raise CrossCodebookRunnerError("patch role is invalid")
    value = np.ascontiguousarray(value, dtype="<f4")
    stored = record.get("activation_layer_sha256", {}).get(str(layer))
    if stored != f32_sha256(value):
        raise CrossCodebookRunnerError("baseline vector hash does not match sidecar")
    return value


def _execute_patches(
    model: Any,
    plan: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    templates: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    basis_sidecar: np.ndarray,
    basis_details: Mapping[str, Any],
    selected_layer: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    cells = {cell["cell_id"]: cell for cell in plan["cell_registry"]}
    prompts = {prompt["cell_id"]: prompt for prompt in plan["prompts"]}
    baselines = {record["cell_id"]: record for record in baseline_records}
    if len(baselines) != len(baseline_records):
        raise CrossCodebookRunnerError("baseline record cell IDs are duplicated")
    records: list[dict[str, Any]] = []
    patched_activations: list[np.ndarray] = []
    for index, raw_template in enumerate(templates):
        template = dict(raw_template)
        layer = template["layer"] if role == "localization" else selected_layer
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in LAYER_GRID:
            raise CrossCodebookRunnerError("patch template layer is invalid")
        recipient_id = template["recipient_cell_id"]
        source_id = template["source_cell_id"]
        recipient_cell = cells.get(recipient_id)
        recipient_prompt = prompts.get(recipient_id)
        recipient_baseline = baselines.get(recipient_id)
        if (
            recipient_cell is None
            or recipient_prompt is None
            or recipient_baseline is None
            or recipient_cell["role"] != role
            or recipient_cell["recipient_selected"] is not True
        ):
            raise CrossCodebookRunnerError("patch recipient does not resolve")
        source_baseline = None if source_id is None else baselines.get(source_id)
        if source_id is not None and source_baseline is None:
            raise CrossCodebookRunnerError("patch source does not resolve")
        expected_template = _patch_template(
            recipient_cell,
            condition=template["condition"],
            layer=template["layer"],
        )
        if expected_template != template:
            raise CrossCodebookRunnerError("patch template identity changed")
        recipient_activation = _activation_for_record(
            recipient_baseline, activations, layer=layer, role=role
        )
        source_activation = (
            None
            if source_baseline is None
            else _activation_for_record(
                source_baseline, activations, layer=layer, role=role
            )
        )
        fit_center, directions = _basis_vectors_at_layer(
            basis_sidecar, basis_details, layer
        )
        direction_name = template["direction_name"]
        direction = None if direction_name is None else directions[direction_name]
        center = (
            fit_center
            if template["operation"] in {"erasure", "rescue"}
            else None
        )
        diagnostics, trace, patched_activation = _patch_forward(
            model,
            recipient_prompt,
            layer=layer,
            operation=template["operation"],
            recipient_activation=recipient_activation,
            source_activation=source_activation,
            direction=direction,
            center=center,
        )
        trace["direction_name"] = direction_name
        record = _patch_record(
            plan,
            template,
            recipient_cell,
            recipient_prompt,
            phase=phase,
            layer=layer,
            patched_activation_row=index,
            recipient_baseline=recipient_baseline,
            source_baseline=source_baseline,
            recipient_activation=recipient_activation,
            source_activation=source_activation,
            direction=direction,
            center=center,
            diagnostics=diagnostics,
            trace=trace,
        )
        records.append(record)
        patched_activations.append(patched_activation)
        if (index + 1) % 32 == 0 or index + 1 == len(templates):
            print(
                canonical_json(
                    {"phase": phase, "completed": index + 1, "total": len(templates)}
                ),
                flush=True,
            )
    if len({record["record_id"] for record in records}) != len(templates):
        raise CrossCodebookRunnerError("patch record IDs are duplicated")
    matrix = np.ascontiguousarray(np.stack(patched_activations), dtype="<f4")
    if matrix.shape != (len(templates), MODEL_WIDTH):
        raise CrossCodebookRunnerError("patched activation sidecar shape changed")
    if any(
        record["patched_activation_row"] != index
        or record["patched_activation_sha256"] != f32_sha256(matrix[index])
        for index, record in enumerate(records)
    ):
        raise CrossCodebookRunnerError("patched activation row binding changed")
    return records, matrix


def _write_completed_baseline_phase(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    prerequisite_bindings: Mapping[str, str] | None = None,
    selected_layer: int | None = None,
) -> None:
    paths = PHASE_PATHS[phase]
    write_jsonl(paths["records"], records)
    write_array(paths["activations"], activations)
    manifest = _execution_manifest(
        phase=phase,
        plan=plan,
        design=design,
        attempt_path=paths["attempt"],
        records_path=paths["records"],
        records=records,
        activations_path=paths["activations"],
        activations=activations,
        prerequisite_bindings=prerequisite_bindings,
        selected_layer=selected_layer,
    )
    write_json(paths["manifest"], manifest)


def _write_completed_patch_phase(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    patched_activations: np.ndarray,
    prerequisite_bindings: Mapping[str, str],
    selected_layer: int | None,
) -> None:
    paths = PHASE_PATHS[phase]
    write_jsonl(paths["records"], records)
    write_array(paths["patched_activations"], patched_activations)
    manifest = _execution_manifest(
        phase=phase,
        plan=plan,
        design=design,
        attempt_path=paths["attempt"],
        records_path=paths["records"],
        records=records,
        activations_path=None,
        activations=None,
        patched_activations_path=paths["patched_activations"],
        patched_activations=patched_activations,
        prerequisite_bindings=prerequisite_bindings,
        selected_layer=selected_layer,
    )
    write_json(paths["manifest"], manifest)


def run_fit_baseline() -> None:
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts("fit-baseline")
    paths = PHASE_PATHS["fit-baseline"]
    _require_absent(
        [
            *paths.values(),
            DEFAULT_BASIS_LOCK,
            DEFAULT_BASIS_SIDECAR,
            DEFAULT_BASIS_DETAILS,
            DEFAULT_BASIS_CALCULATIONS,
        ],
        "fit-baseline",
    )
    _write_execution_attempt(
        paths["attempt"], phase="fit-baseline", plan=plan, design=design
    )
    model = _load_model()
    records, activations = _execute_baselines(
        model,
        plan,
        role="direction_fit",
        phase="fit-baseline",
        layers=LAYER_GRID,
    )
    _write_completed_baseline_phase(
        phase="fit-baseline",
        plan=plan,
        design=design,
        records=records,
        activations=activations,
    )


def run_localization_baseline() -> None:
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts("localization-baseline")
    _recompute_analyzer_authorization("fit_basis", DEFAULT_BASIS_LOCK)
    _load_basis_artifacts(plan)
    paths = PHASE_PATHS["localization-baseline"]
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "fit_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["fit-baseline"]["manifest"]
        ),
    }
    _require_absent(
        [
            *paths.values(),
            RESULT_ROOT / "localization_baseline_analysis.json",
            DEFAULT_LOCALIZATION_ENTRY,
            DEFAULT_LAYER_LOCK,
        ],
        "localization-baseline",
    )
    _write_execution_attempt(
        paths["attempt"],
        phase="localization-baseline",
        plan=plan,
        design=design,
        prerequisite_bindings=prerequisites,
    )
    model = _load_model()
    records, activations = _execute_baselines(
        model,
        plan,
        role="localization",
        phase="localization-baseline",
        layers=LAYER_GRID,
    )
    _write_completed_baseline_phase(
        phase="localization-baseline",
        plan=plan,
        design=design,
        records=records,
        activations=activations,
        prerequisite_bindings=prerequisites,
    )


def run_localization_patch() -> None:
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts("localization-patch")
    _recompute_analyzer_authorization("fit_basis", DEFAULT_BASIS_LOCK)
    _recompute_analyzer_authorization(
        "localization_baseline", DEFAULT_LOCALIZATION_ENTRY
    )
    basis, details, _ = _load_basis_artifacts(plan)
    _load_localization_entry(plan)
    baseline_manifest, baseline_records, activations = _validate_execution_manifest(
        "localization-baseline",
        expected_count=256,
        expected_activation_shape=(256, 5, MODEL_WIDTH),
    )
    if activations is None:
        raise CrossCodebookRunnerError("localization activation sidecar is missing")
    paths = PHASE_PATHS["localization-patch"]
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "localization_baseline_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["localization-baseline"]["manifest"]
        ),
        "localization_baseline_records_file_sha256": baseline_manifest["records"][
            "file_sha256"
        ],
        "localization_entry_file_sha256": file_sha256(
            DEFAULT_LOCALIZATION_ENTRY
        ),
    }
    _require_absent([*paths.values(), DEFAULT_LAYER_LOCK], "localization-patch")
    _write_execution_attempt(
        paths["attempt"],
        phase="localization-patch",
        plan=plan,
        design=design,
        prerequisite_bindings=prerequisites,
    )
    model = _load_model()
    records, patched_activations = _execute_patches(
        model,
        plan,
        role="localization",
        phase="localization-patch",
        templates=_phase_patch_templates(plan, "localization"),
        baseline_records=baseline_records,
        activations=activations,
        basis_sidecar=basis,
        basis_details=details,
        selected_layer=None,
    )
    _write_completed_patch_phase(
        phase="localization-patch",
        plan=plan,
        design=design,
        records=records,
        patched_activations=patched_activations,
        prerequisite_bindings=prerequisites,
        selected_layer=None,
    )


def run_holdout_baseline() -> None:
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts("holdout-baseline")
    _recompute_analyzer_authorization("fit_basis", DEFAULT_BASIS_LOCK)
    _recompute_analyzer_authorization("localization", DEFAULT_LAYER_LOCK)
    _load_basis_artifacts(plan)
    selected_layer, _ = _load_layer_lock(plan)
    paths = PHASE_PATHS["holdout-baseline"]
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(DEFAULT_LAYER_LOCK),
    }
    _require_absent([*paths.values(), DEFAULT_HOLDOUT_ENTRY], "holdout-baseline")
    _write_execution_attempt(
        paths["attempt"],
        phase="holdout-baseline",
        plan=plan,
        design=design,
        prerequisite_bindings=prerequisites,
    )
    model = _load_model()
    records, activations = _execute_baselines(
        model,
        plan,
        role="holdout",
        phase="holdout-baseline",
        layers=(selected_layer,),
    )
    _write_completed_baseline_phase(
        phase="holdout-baseline",
        plan=plan,
        design=design,
        records=records,
        activations=activations,
        prerequisite_bindings=prerequisites,
        selected_layer=selected_layer,
    )


def run_holdout_patch() -> None:
    plan, design = _load_frozen_plan()
    _require_no_downstream_artifacts("holdout-patch")
    _recompute_analyzer_authorization("fit_basis", DEFAULT_BASIS_LOCK)
    _recompute_analyzer_authorization("localization", DEFAULT_LAYER_LOCK)
    _recompute_analyzer_authorization("holdout_baseline", DEFAULT_HOLDOUT_ENTRY)
    basis, details, _ = _load_basis_artifacts(plan)
    selected_layer, _ = _load_holdout_entry(plan)
    baseline_manifest, baseline_records, activations = _validate_execution_manifest(
        "holdout-baseline",
        expected_count=1024,
        expected_activation_shape=(1024, MODEL_WIDTH),
    )
    if activations is None:
        raise CrossCodebookRunnerError("holdout activation sidecar is missing")
    paths = PHASE_PATHS["holdout-patch"]
    prerequisites = {
        "basis_lock_file_sha256": file_sha256(DEFAULT_BASIS_LOCK),
        "layer_lock_file_sha256": file_sha256(DEFAULT_LAYER_LOCK),
        "holdout_entry_file_sha256": file_sha256(DEFAULT_HOLDOUT_ENTRY),
        "holdout_baseline_execution_manifest_file_sha256": file_sha256(
            PHASE_PATHS["holdout-baseline"]["manifest"]
        ),
        "holdout_baseline_records_file_sha256": baseline_manifest["records"][
            "file_sha256"
        ],
    }
    _require_absent(paths.values(), "holdout-patch")
    _write_execution_attempt(
        paths["attempt"],
        phase="holdout-patch",
        plan=plan,
        design=design,
        prerequisite_bindings=prerequisites,
    )
    model = _load_model()
    records, patched_activations = _execute_patches(
        model,
        plan,
        role="holdout",
        phase="holdout-patch",
        templates=_phase_patch_templates(plan, "holdout"),
        baseline_records=baseline_records,
        activations=activations,
        basis_sidecar=basis,
        basis_details=details,
        selected_layer=selected_layer,
    )
    _write_completed_patch_phase(
        phase="holdout-patch",
        plan=plan,
        design=design,
        records=records,
        patched_activations=patched_activations,
        prerequisite_bindings=prerequisites,
        selected_layer=selected_layer,
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

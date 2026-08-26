"""Plan and run the frozen non-biological coherent-readout v2 syntax bank.

The runner is development-only.  It freezes all four syntax candidates and all
256 candidate-by-item-by-form requests before loading weights, then obtains one
raw, unprocessed next-token distribution per request.  It never generates text
and it does not analyze biology, knowledge, or activation claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import coherent_binary_readout as coherent
    from . import run_coherent_binary_readout as v1_runner
    from .causal_intervention import execution_input_sha256
except ImportError:  # direct execution from eval/
    import coherent_binary_readout as coherent
    import run_coherent_binary_readout as v1_runner
    from causal_intervention import execution_input_sha256


FIXTURE_SCHEMA = "coherent-readout-v2-syntax-bank-v1"
ITEM_SCHEMA = "coherent-readout-v2-syntax-item-v1"
RECORD_SCHEMA = "coherent-readout-v2-syntax-record-v1"
PLAN_SCHEMA = "coherent-readout-v2-syntax-call-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v2-syntax-design-v1"
RUN_MANIFEST_SCHEMA = "coherent-readout-v2-syntax-run-manifest-v1"
GATE_SCHEMA = "coherent-readout-v2-syntax-gates-v1"
LOGITS_SOURCE = "raw_model_output_before_processors"

FROZEN_PREREG_SHA256 = (
    "659bb45361697c0967d9339f2cdda6299021d12860fbbef7f04c8dcaebd604b7"
)
FROZEN_FIXTURE_SHA256 = (
    "d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8"
)
FROZEN_FIXTURE_MANIFEST_SHA256 = (
    "b7b8467c2595af5bf5cfb414cb052289062e721300ea3415c23d22de16c22ae5"
)

CANDIDATE_ORDER = (
    "assistant_prefill_answer_colon",
    "system_user_exact",
    "user_only_exact",
    "user_label_colon_v1_control",
)
FORM_KEYS = coherent.FORM_KEYS

SYSTEM_PREFILL = "Complete the assistant answer with exactly X or Y and no other text."
USER_PREFILL = (
    "Classify the record from its explicit declaration.\n"
    "{line_1}\n{line_2}\n{declaration}"
)
SYSTEM_EXACT = (
    "Follow the user's label mapping. Your entire response must be exactly one "
    "character: X or Y. Do not write any other text."
)
USER_SYSTEM_EXACT = (
    "Classify the record from its explicit declaration.\n"
    "{line_1}\n{line_2}\n{declaration}\nReturn exactly X or Y."
)
USER_ONLY_EXACT = (
    "Follow the label mapping. Answer with exactly one character: X or Y. Do not "
    "write any other text.\nClassify the record from its explicit declaration.\n"
    "{line_1}\n{line_2}\n{declaration}\nAnswer:"
)
USER_V1_CONTROL = (
    "Classify the record using exactly one label.\n"
    "{line_1}\n{line_2}\n{declaration}\nLabel:"
)

FROZEN_ROLE_LOCKS = {
    "smoke_only": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "model_weights_sha256": (
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
        ),
    },
    "syntax_selection": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "model_weights_sha256": (
            "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        ),
    },
}

PAIR_KEYS = {"pair_id", "cluster_id", "positive_class", "negative_class"}
ITEM_KEYS = {
    "schema_version",
    "item_id",
    "pair_id",
    "cluster_id",
    "positive_class",
    "negative_class",
    "declared_class",
    "truth_polarity",
    "declaration_text",
    "declaration_sha256",
    "firewall",
}
FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "mode",
    "purpose",
    "inferential_unit",
    "firewall",
    "pair_registry",
    "items",
    "model_calls_made_by_builder",
}

IDENTITY_KEYS = (
    "schema_version",
    "candidate_id",
    "candidate_rank",
    "candidate_definition_sha256",
    "item_id",
    "pair_id",
    "cluster_id",
    "fixture_item_sha256",
    "declaration_sha256",
    "positive_class",
    "negative_class",
    "declared_class",
    "truth_polarity",
    "order",
    "mapping",
    "messages_sha256",
    "user_content_sha256",
    "prompt_sha256",
    "render_mode",
    "add_generation_prompt",
    "continue_final_message",
    "enable_thinking",
    "x_answer_text",
    "y_answer_text",
    "x_token_id",
    "y_token_id",
    "positive_token_id",
    "negative_token_id",
    "correct_token_id",
    "wrong_token_id",
    "execution_input_sha256",
    "input_token_count",
    "model_role",
    "model_id",
    "model_revision",
    "model_weights_sha256",
    "tokenizer_id",
    "tokenizer_revision",
    "chat_template_sha256",
    "dtype",
    "device",
    "logits_source",
    "vocab_size",
    "runner_code_sha256",
    "analyzer_code_sha256",
    "environment_sha256",
    "full_vocab_logits_row",
)

OUTPUT_KEYS = {
    "x_logit",
    "y_logit",
    "full_vocab_logsumexp",
    "full_vocab_logits_sha256",
    "maximum_token_ids",
    "maximum_tie_count",
    "greedy_token_id",
    "greedy_logit",
    "forward_trace_sha256",
}
RECORD_KEYS = {"record_id", *IDENTITY_KEYS, *OUTPUT_KEYS}

RANKING = [
    {"metric": "min_form_native_correct_count", "direction": "max"},
    {"metric": "total_native_correct_count", "direction": "max"},
    {"metric": "min_form_truth_native_correct_count", "direction": "max"},
    {"metric": "min_form_native_count", "direction": "max"},
    {"metric": "total_native_count", "direction": "max"},
    {"metric": "max_item_range", "direction": "min"},
    {"metric": "max_abs_mean_pair_estimand", "direction": "min"},
    {"metric": "min_form_median_s", "direction": "max"},
    {"metric": "candidate_rank", "direction": "min"},
    {"metric": "candidate_id", "direction": "min"},
]


class SyntaxRunnerError(ValueError):
    """Raised when the syntax-bank plan or raw execution is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cached_model_weights(
    model_id: str,
    model_revision: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Resolve and hash the exact local safetensors file before model loading."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        raise SyntaxRunnerError(
            "cached-weight verification requires huggingface_hub"
        ) from error
    resolved = try_to_load_from_cache(
        model_id,
        "model.safetensors",
        revision=model_revision,
    )
    if not isinstance(resolved, str):
        raise SyntaxRunnerError(
            "the frozen model.safetensors file is not present in the local cache"
        )
    path = Path(resolved)
    if not path.is_file():
        raise SyntaxRunnerError(
            "the resolved cached model.safetensors path is not a file"
        )
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise SyntaxRunnerError(
            "cached model.safetensors SHA-256 differs from the frozen lock"
        )
    return {
        "filename": path.name,
        "sha256": observed_sha256,
        "size_bytes": path.stat().st_size,
    }


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyntaxRunnerError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    output = _require_string(value, label)
    if len(output) != 64 or any(character not in "0123456789abcdef" for character in output):
        raise SyntaxRunnerError(f"{label} must be a lowercase SHA-256 digest")
    return output


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SyntaxRunnerError(
            f"{label} schema mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_lock() -> dict[str, Any]:
    value = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: _package_version(name)
            for name in (
                "huggingface-hub",
                "numpy",
                "scipy",
                "torch",
                "transformers",
            )
        },
        "execution_source_sha256": {
            "causal_intervention": file_sha256(
                Path(sys.modules[execution_input_sha256.__module__].__file__)
            ),
            "coherent_binary_readout": file_sha256(Path(coherent.__file__)),
            "run_coherent_binary_readout": file_sha256(Path(v1_runner.__file__)),
        },
    }
    return {**value, "environment_sha256": canonical_sha256(value)}


def default_gate_config() -> dict[str, Any]:
    """Return the exact numeric gate and ranking frozen by the v2 preregistration."""

    return {
        "schema_version": GATE_SCHEMA,
        "records_per_candidate": 64,
        "items_per_candidate": 16,
        "pairs_per_candidate": 8,
        "forms_per_candidate": 4,
        "truth_polarities": ["negative", "positive"],
        "overall_native_min_count": 61,
        "per_form_native_min_count": 15,
        "per_form_truth_native_min_count": 7,
        "overall_native_correct_min_count": 61,
        "per_form_native_correct_min_count": 15,
        "per_form_truth_native_correct_min_count": 7,
        "per_form_positive_s_min_count": 15,
        "per_form_median_s_min": 0.50,
        "item_range_margin": 0.20,
        "item_range_required_count": 16,
        "equivalence_margin": 0.06,
        "equivalence_alpha": 0.05,
        "pair_estimands": ["D_I", "D_O", "D_R", "M_I", "M_O", "M_R"],
        "ranking": [dict(value) for value in RANKING],
        "argmax_tie_policy": "valid_behavior_but_native_and_native_correct_false",
        "sign_symmetry_assumption": "required_unverified",
        "inference_scope": "fixed_bank_engineering_only",
    }


def candidate_definitions() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": CANDIDATE_ORDER[0],
            "candidate_rank": 0,
            "message_roles": ["system", "user", "assistant"],
            "system_template": SYSTEM_PREFILL,
            "user_template": USER_PREFILL,
            "assistant_template": "Answer:",
            "render_mode": "continue_final_message",
            "add_generation_prompt": False,
            "continue_final_message": True,
            "enable_thinking": False,
            "x_answer_text": " X",
            "y_answer_text": " Y",
        },
        {
            "candidate_id": CANDIDATE_ORDER[1],
            "candidate_rank": 1,
            "message_roles": ["system", "user"],
            "system_template": SYSTEM_EXACT,
            "user_template": USER_SYSTEM_EXACT,
            "assistant_template": None,
            "render_mode": "add_generation_prompt",
            "add_generation_prompt": True,
            "continue_final_message": False,
            "enable_thinking": False,
            "x_answer_text": "X",
            "y_answer_text": "Y",
        },
        {
            "candidate_id": CANDIDATE_ORDER[2],
            "candidate_rank": 2,
            "message_roles": ["user"],
            "system_template": None,
            "user_template": USER_ONLY_EXACT,
            "assistant_template": None,
            "render_mode": "add_generation_prompt",
            "add_generation_prompt": True,
            "continue_final_message": False,
            "enable_thinking": False,
            "x_answer_text": "X",
            "y_answer_text": "Y",
        },
        {
            "candidate_id": CANDIDATE_ORDER[3],
            "candidate_rank": 3,
            "message_roles": ["user"],
            "system_template": None,
            "user_template": USER_V1_CONTROL,
            "assistant_template": None,
            "render_mode": "add_generation_prompt",
            "add_generation_prompt": True,
            "continue_final_message": False,
            "enable_thinking": False,
            "x_answer_text": "X",
            "y_answer_text": "Y",
        },
    ]


def candidate_bank_sha256() -> str:
    return canonical_sha256(candidate_definitions())


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntaxRunnerError("syntax fixture must be an object")
    _exact_keys(value, FIXTURE_KEYS, "syntax fixture")
    if value["schema_version"] != FIXTURE_SCHEMA:
        raise SyntaxRunnerError("unsupported syntax fixture schema")
    if value["mode"] != "development" or value["purpose"] != "syntax_selection_only":
        raise SyntaxRunnerError("syntax fixture mode or purpose changed")
    if value["inferential_unit"] != "pair_cluster":
        raise SyntaxRunnerError("syntax fixture inferential unit changed")
    if value["model_calls_made_by_builder"] != 0:
        raise SyntaxRunnerError("syntax fixture claims prior model execution")
    expected_firewall = {
        "scope": "syntax_selection_only",
        "biology_inference": "forbidden",
        "knowledge_inference": "forbidden",
        "activation_inference": "forbidden",
    }
    if value["firewall"] != expected_firewall:
        raise SyntaxRunnerError("syntax fixture firewall changed")
    pairs = value["pair_registry"]
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise SyntaxRunnerError("syntax fixture must contain eight pair clusters")
    pair_by_id: dict[str, dict[str, str]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise SyntaxRunnerError("pair registry rows must be objects")
        _exact_keys(pair, PAIR_KEYS, "pair registry row")
        pair_id = _require_string(pair["pair_id"], "pair_id")
        if pair["cluster_id"] != pair_id or pair_id in pair_by_id:
            raise SyntaxRunnerError("pair and cluster registry is not one-to-one")
        positive = _require_string(pair["positive_class"], "positive_class")
        negative = _require_string(pair["negative_class"], "negative_class")
        if positive == negative:
            raise SyntaxRunnerError("pair classes must be distinct")
        pair_by_id[pair_id] = dict(pair)
    items = value["items"]
    if not isinstance(items, list) or len(items) != 16:
        raise SyntaxRunnerError("syntax fixture must contain exactly 16 items")
    normalized: list[dict[str, Any]] = []
    coverage: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise SyntaxRunnerError("syntax items must be objects")
        _exact_keys(item, ITEM_KEYS, "syntax item")
        if item["schema_version"] != ITEM_SCHEMA or item["firewall"] != expected_firewall:
            raise SyntaxRunnerError("syntax item schema or firewall changed")
        pair = pair_by_id.get(item["pair_id"])
        if pair is None or item["cluster_id"] != item["pair_id"]:
            raise SyntaxRunnerError("syntax item has an unknown pair/cluster")
        if any(item[key] != pair[key] for key in ("positive_class", "negative_class")):
            raise SyntaxRunnerError("syntax item class orientation changed")
        polarity = item["truth_polarity"]
        if polarity not in {"positive", "negative"}:
            raise SyntaxRunnerError("truth_polarity must be positive or negative")
        if item["declared_class"] != pair[f"{polarity}_class"]:
            raise SyntaxRunnerError("declared class disagrees with truth polarity")
        declaration = _require_string(item["declaration_text"], "declaration_text")
        if "\n" in declaration or "\r" in declaration:
            raise SyntaxRunnerError("declaration_text must be one exact line")
        if item["declaration_sha256"] != text_sha256(declaration):
            raise SyntaxRunnerError("declaration digest mismatch")
        key = (item["pair_id"], polarity)
        if key in coverage:
            raise SyntaxRunnerError("duplicate pair-by-truth item")
        coverage.add(key)
        normalized.append({**dict(item), "fixture_item_sha256": canonical_sha256(item)})
    expected = {(pair_id, polarity) for pair_id in pair_by_id for polarity in ("positive", "negative")}
    if coverage != expected:
        raise SyntaxRunnerError("every pair must contain both truth polarities")
    if [item["item_id"] for item in items] != sorted(item["item_id"] for item in items):
        raise SyntaxRunnerError("syntax items must be sorted by item_id")
    return {
        **dict(value),
        "pair_registry": [dict(pair) for pair in pairs],
        "items": normalized,
        "fixture_canonical_sha256": canonical_sha256(value),
    }


def _mapping_lines(item: Mapping[str, Any], order: str, mapping: str) -> tuple[str, str]:
    if (order, mapping) not in FORM_KEYS:
        raise SyntaxRunnerError("unknown order-by-remapping form")
    positive_label, negative_label = (
        ("X", "Y") if mapping == "positive_is_x" else ("Y", "X")
    )
    positive_line = f"label {positive_label} means {item['positive_class']}"
    negative_line = f"label {negative_label} means {item['negative_class']}"
    return (
        (positive_line, negative_line)
        if order == "positive_first"
        else (negative_line, positive_line)
    )


def render_candidate_messages(
    candidate_id: str,
    item: Mapping[str, Any],
    order: str,
    mapping: str,
) -> list[dict[str, str]]:
    """Return the exact preregistered message list before chat templating."""

    definitions = {value["candidate_id"]: value for value in candidate_definitions()}
    candidate = definitions.get(candidate_id)
    if candidate is None:
        raise SyntaxRunnerError(f"unknown syntax candidate: {candidate_id}")
    line_1, line_2 = _mapping_lines(item, order, mapping)
    values = {
        "line_1": line_1,
        "line_2": line_2,
        "declaration": item["declaration_text"],
    }
    if candidate_id == CANDIDATE_ORDER[0]:
        return [
            {"role": "system", "content": SYSTEM_PREFILL},
            {"role": "user", "content": USER_PREFILL.format(**values)},
            {"role": "assistant", "content": "Answer:"},
        ]
    if candidate_id == CANDIDATE_ORDER[1]:
        return [
            {"role": "system", "content": SYSTEM_EXACT},
            {"role": "user", "content": USER_SYSTEM_EXACT.format(**values)},
        ]
    template = USER_ONLY_EXACT if candidate_id == CANDIDATE_ORDER[2] else USER_V1_CONTROL
    return [{"role": "user", "content": template.format(**values)}]


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise SyntaxRunnerError(f"{label} must contain one row")
        raw = list(raw[0])
    if not raw or any(isinstance(token, bool) or not isinstance(token, (int, np.integer)) for token in raw):
        raise SyntaxRunnerError(f"{label} must be a nonempty integer vector")
    return [int(token) for token in raw]


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    candidate: Mapping[str, Any],
) -> tuple[str, list[int], list[int]]:
    flags = {
        "add_generation_prompt": candidate["add_generation_prompt"],
        "continue_final_message": candidate["continue_final_message"],
        "enable_thinking": candidate["enable_thinking"],
    }
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **flags)
    if not isinstance(rendered, str) or not rendered:
        raise SyntaxRunnerError("chat template did not return rendered text")
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **flags,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise SyntaxRunnerError("tokenized chat template must return input_ids")
    input_ids = _as_int_vector(encoded["input_ids"], "input_ids")
    attention_mask = _as_int_vector(
        encoded.get("attention_mask", [1] * len(input_ids)), "attention_mask"
    )
    if len(input_ids) != len(attention_mask) or any(value != 1 for value in attention_mask):
        raise SyntaxRunnerError("syntax requests must be unpadded and fully attended")
    retokenized = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "retokenized chat"
    )
    if retokenized != input_ids:
        raise SyntaxRunnerError("tokenize-false rendering does not reproduce tokenize-true IDs")
    return rendered, input_ids, attention_mask


def record_id(record: Mapping[str, Any]) -> str:
    return canonical_sha256({key: record[key] for key in IDENTITY_KEYS})


def call_plan_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    entries = [
        {
            "record_id": record_id(record),
            "request_identity": {key: record[key] for key in IDENTITY_KEYS},
        }
        for record in records
    ]
    entries.sort(key=lambda value: value["record_id"])
    if not entries or len({value["record_id"] for value in entries}) != len(entries):
        raise SyntaxRunnerError("call plan must contain unique requests")
    return canonical_sha256({"schema_version": PLAN_SCHEMA, "entries": entries})


def forward_trace_sha256(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "coherent-readout-v2-syntax-forward-trace-v1",
            "record_id": record["record_id"],
            "candidate_id": record["candidate_id"],
            "item_id": record["item_id"],
            "order": record["order"],
            "mapping": record["mapping"],
            "prompt_sha256": record["prompt_sha256"],
            "execution_input_sha256": record["execution_input_sha256"],
            "full_vocab_logits_row": record["full_vocab_logits_row"],
            "full_vocab_logits_sha256": record["full_vocab_logits_sha256"],
            "x_token_id": record["x_token_id"],
            "y_token_id": record["y_token_id"],
            "vocab_size": record["vocab_size"],
            "model_id": record["model_id"],
            "model_revision": record["model_revision"],
            "model_weights_sha256": record["model_weights_sha256"],
            "tokenizer_id": record["tokenizer_id"],
            "tokenizer_revision": record["tokenizer_revision"],
            "device": record["device"],
            "dtype": record["dtype"],
            "runner_code_sha256": record["runner_code_sha256"],
            "analyzer_code_sha256": record["analyzer_code_sha256"],
            "environment_sha256": record["environment_sha256"],
            "logits_source": record["logits_source"],
        }
    )


def build_call_plan(
    fixture: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_role: str,
    model_id: str,
    model_revision: str,
    model_weights_sha256: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    model_vocab_size: int,
    dtype: str,
    device: str,
    source_fixture_sha256: str,
    source_manifest_sha256: str,
    preregistration_sha256: str,
    analyzer_code_sha256: str,
) -> dict[str, Any]:
    """Tokenize and freeze the complete 256-request plan without model weights."""

    locked = validate_fixture(fixture)
    if model_role not in FROZEN_ROLE_LOCKS:
        raise SyntaxRunnerError("model_role must be smoke_only or syntax_selection")
    model_id = _require_string(model_id, "model_id")
    model_revision = _require_string(model_revision, "model_revision")
    model_weights_sha256 = _require_sha256(model_weights_sha256, "model_weights_sha256")
    tokenizer_id = _require_string(tokenizer_id, "tokenizer_id")
    tokenizer_revision = _require_string(tokenizer_revision, "tokenizer_revision")
    if not isinstance(model_vocab_size, (int, np.integer)) or isinstance(model_vocab_size, bool) or int(model_vocab_size) < 2:
        raise SyntaxRunnerError("model_vocab_size must be an integer of at least two")
    model_vocab_size = int(model_vocab_size)
    if dtype != "float32":
        raise SyntaxRunnerError("v2 syntax preregistration freezes dtype=float32")
    if device not in {"cpu", "mps", "cuda"}:
        raise SyntaxRunnerError("device must be cpu, mps, or cuda")
    for value, label in (
        (source_fixture_sha256, "source_fixture_sha256"),
        (source_manifest_sha256, "source_manifest_sha256"),
        (preregistration_sha256, "preregistration_sha256"),
        (analyzer_code_sha256, "analyzer_code_sha256"),
    ):
        _require_sha256(value, label)
    try:
        tokenizer_vocab_size = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise SyntaxRunnerError("tokenizer must expose its complete length") from error
    if not isinstance(tokenizer_vocab_size, int) or tokenizer_vocab_size < 2:
        raise SyntaxRunnerError("tokenizer vocabulary size is invalid")
    template_getter = getattr(tokenizer, "get_chat_template", None)
    template = template_getter() if callable(template_getter) else getattr(tokenizer, "chat_template", None)
    template = _require_string(template, "effective chat template")
    chat_template_sha256 = text_sha256(template)
    runner_code_sha256 = file_sha256(Path(__file__))
    environment = environment_lock()
    definitions = candidate_definitions()
    definition_by_id = {value["candidate_id"]: value for value in definitions}
    records: list[dict[str, Any]] = []
    candidate_token_ids: dict[str, dict[str, int]] = {}
    for candidate_id in CANDIDATE_ORDER:
        candidate = definition_by_id[candidate_id]
        observed_x: set[int] = set()
        observed_y: set[int] = set()
        definition_sha256 = canonical_sha256(candidate)
        for item in locked["items"]:
            for order, mapping in FORM_KEYS:
                messages = render_candidate_messages(candidate_id, item, order, mapping)
                rendered, input_ids, attention_mask = _render_chat(tokenizer, messages, candidate)
                x_token_id = coherent.contextual_single_token_id(
                    tokenizer, rendered, candidate["x_answer_text"]
                )
                y_token_id = coherent.contextual_single_token_id(
                    tokenizer, rendered, candidate["y_answer_text"]
                )
                if x_token_id == y_token_id or max(x_token_id, y_token_id) >= min(
                    tokenizer_vocab_size, model_vocab_size
                ):
                    raise SyntaxRunnerError("candidate answer tokens are invalid")
                observed_x.add(x_token_id)
                observed_y.add(y_token_id)
                positive_token_id, negative_token_id = (
                    (x_token_id, y_token_id)
                    if mapping == "positive_is_x"
                    else (y_token_id, x_token_id)
                )
                correct_token_id, wrong_token_id = (
                    (positive_token_id, negative_token_id)
                    if item["truth_polarity"] == "positive"
                    else (negative_token_id, positive_token_id)
                )
                user_content = next(
                    message["content"] for message in messages if message["role"] == "user"
                )
                identity = {
                    "schema_version": RECORD_SCHEMA,
                    "candidate_id": candidate_id,
                    "candidate_rank": candidate["candidate_rank"],
                    "candidate_definition_sha256": definition_sha256,
                    "item_id": item["item_id"],
                    "pair_id": item["pair_id"],
                    "cluster_id": item["cluster_id"],
                    "fixture_item_sha256": item["fixture_item_sha256"],
                    "declaration_sha256": item["declaration_sha256"],
                    "positive_class": item["positive_class"],
                    "negative_class": item["negative_class"],
                    "declared_class": item["declared_class"],
                    "truth_polarity": item["truth_polarity"],
                    "order": order,
                    "mapping": mapping,
                    "messages_sha256": canonical_sha256(messages),
                    "user_content_sha256": text_sha256(user_content),
                    "prompt_sha256": text_sha256(rendered),
                    "render_mode": candidate["render_mode"],
                    "add_generation_prompt": candidate["add_generation_prompt"],
                    "continue_final_message": candidate["continue_final_message"],
                    "enable_thinking": candidate["enable_thinking"],
                    "x_answer_text": candidate["x_answer_text"],
                    "y_answer_text": candidate["y_answer_text"],
                    "x_token_id": x_token_id,
                    "y_token_id": y_token_id,
                    "positive_token_id": positive_token_id,
                    "negative_token_id": negative_token_id,
                    "correct_token_id": correct_token_id,
                    "wrong_token_id": wrong_token_id,
                    "execution_input_sha256": execution_input_sha256(input_ids, attention_mask),
                    "input_token_count": len(input_ids),
                    "model_role": model_role,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "model_weights_sha256": model_weights_sha256,
                    "tokenizer_id": tokenizer_id,
                    "tokenizer_revision": tokenizer_revision,
                    "chat_template_sha256": chat_template_sha256,
                    "dtype": dtype,
                    "device": device,
                    "logits_source": LOGITS_SOURCE,
                    "vocab_size": model_vocab_size,
                    "runner_code_sha256": runner_code_sha256,
                    "analyzer_code_sha256": analyzer_code_sha256,
                    "environment_sha256": environment["environment_sha256"],
                    "full_vocab_logits_row": len(records),
                }
                records.append(
                    {
                        "record_id": record_id(identity),
                        **identity,
                        "messages": messages,
                        "rendered_chat": rendered,
                        "execution_input_ids": input_ids,
                        "execution_attention_mask": attention_mask,
                    }
                )
        if len(observed_x) != 1 or len(observed_y) != 1:
            raise SyntaxRunnerError(f"candidate {candidate_id} answer token IDs are unstable")
        candidate_token_ids[candidate_id] = {
            "x_token_id": next(iter(observed_x)),
            "y_token_id": next(iter(observed_y)),
        }
    if len(records) != 256 or len({record["record_id"] for record in records}) != 256:
        raise SyntaxRunnerError("syntax call plan must contain exactly 256 unique records")
    candidate_registry = [
        {
            **definition,
            "candidate_definition_sha256": canonical_sha256(definition),
            **candidate_token_ids[definition["candidate_id"]],
        }
        for definition in definitions
    ]
    plan_sha256 = call_plan_sha256(records)
    return {
        "schema_version": PLAN_SCHEMA,
        "mode": "development",
        "purpose": "syntax_selection_only",
        "analysis_id": locked["analysis_id"],
        "model_role": model_role,
        "selection_eligible_model": model_role == "syntax_selection",
        "source_fixture_sha256": source_fixture_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_fixture_canonical_sha256": locked["fixture_canonical_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "runner_code_sha256": runner_code_sha256,
        "analyzer_code_sha256": analyzer_code_sha256,
        "environment": environment,
        "candidate_bank_sha256": candidate_bank_sha256(),
        "candidate_registry": candidate_registry,
        "pair_registry": locked["pair_registry"],
        "item_ids": [item["item_id"] for item in locked["items"]],
        "model_id": model_id,
        "model_revision": model_revision,
        "model_weights_sha256": model_weights_sha256,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "chat_template_sha256": chat_template_sha256,
        "dtype": dtype,
        "device": device,
        "vocab_size": model_vocab_size,
        "gate_config": default_gate_config(),
        "record_count": len(records),
        "expected_record_ids": sorted(record["record_id"] for record in records),
        "call_plan_sha256": plan_sha256,
        "records": records,
    }


def design_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("record_count") != 256:
        raise SyntaxRunnerError("invalid v2 syntax call plan")
    return {
        "schema_version": DESIGN_SCHEMA,
        "mode": "development",
        "purpose": "syntax_selection_only",
        "analysis_id": plan["analysis_id"],
        "model_role": plan["model_role"],
        "selection_eligible_model": plan["selection_eligible_model"],
        "source_fixture_sha256": plan["source_fixture_sha256"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_fixture_canonical_sha256": plan["source_fixture_canonical_sha256"],
        "preregistration_sha256": plan["preregistration_sha256"],
        "expected_runner_code_sha256": plan["runner_code_sha256"],
        "expected_analyzer_code_sha256": plan["analyzer_code_sha256"],
        "expected_environment": plan["environment"],
        "candidate_bank_sha256": plan["candidate_bank_sha256"],
        "candidate_registry": plan["candidate_registry"],
        "pair_registry": plan["pair_registry"],
        "expected_item_ids": plan["item_ids"],
        "expected_model_id": plan["model_id"],
        "expected_model_revision": plan["model_revision"],
        "expected_model_weights_sha256": plan["model_weights_sha256"],
        "expected_tokenizer_id": plan["tokenizer_id"],
        "expected_tokenizer_revision": plan["tokenizer_revision"],
        "expected_tokenizer_vocab_size": plan["tokenizer_vocab_size"],
        "expected_chat_template_sha256": plan["chat_template_sha256"],
        "expected_dtype": plan["dtype"],
        "expected_device": plan["device"],
        "expected_vocab_size": plan["vocab_size"],
        "gate_config": plan["gate_config"],
        "expected_record_count": plan["record_count"],
        "expected_record_ids": plan["expected_record_ids"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "confirmatory_execution_allowed": False,
        "claim_scope": "syntax_selection_only_no_biology_knowledge_or_activation_claim",
    }


def validate_plan_against_design(
    plan: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    expected = design_from_plan(plan)
    if design != expected:
        mismatches = sorted(
            key
            for key in set(expected) | set(design)
            if expected.get(key) != design.get(key)
        )
        raise SyntaxRunnerError(f"plan differs from frozen design: {mismatches}")
    return expected


def validate_loaded_model(model: Any, plan: Mapping[str, Any]) -> None:
    """Verify the observed state of a loaded torch model against the frozen plan."""

    try:
        import torch
    except ImportError:
        # Generic mock/library callers are permitted; the frozen CLI model loader
        # independently requires torch and can never reach execution without it.
        return
    if not isinstance(model, torch.nn.Module):
        return
    parameters = list(model.parameters())
    if not parameters:
        raise SyntaxRunnerError("loaded torch model has no parameters")
    tensors = [*parameters, *model.buffers()]
    observed_devices = {tensor.device.type for tensor in tensors}
    if observed_devices != {plan["device"]}:
        raise SyntaxRunnerError(
            "loaded model tensor devices differ from the frozen plan: "
            f"{sorted(observed_devices)}"
        )
    floating_tensors = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating_tensors:
        raise SyntaxRunnerError("loaded torch model has no floating tensors")
    expected_dtype = {"float32": torch.float32}[plan["dtype"]]
    observed_dtypes = {tensor.dtype for tensor in floating_tensors}
    if observed_dtypes != {expected_dtype}:
        raise SyntaxRunnerError(
            "loaded model floating-tensor dtypes differ from the frozen plan: "
            f"{sorted(str(value) for value in observed_dtypes)}"
        )


def _forward_next_logits(model: Any, planned: Mapping[str, Any]) -> np.ndarray:
    input_ids = [list(planned["execution_input_ids"])]
    attention_mask = [list(planned["execution_attention_mask"])]
    try:
        import torch
    except ImportError:
        torch = None
    is_torch = torch is not None and isinstance(model, torch.nn.Module)
    context = torch.inference_mode() if is_torch else nullcontext()
    if is_torch:
        device = v1_runner._model_device(model)
        input_ids_value = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_value = torch.tensor(attention_mask, dtype=torch.long, device=device)
    else:
        input_ids_value = input_ids
        attention_mask_value = attention_mask
    with context:
        output = model(
            input_ids=input_ids_value,
            attention_mask=attention_mask_value,
            use_cache=False,
            return_dict=True,
        )
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    shape = getattr(logits, "shape", None)
    if shape is None or len(shape) != 3 or shape[0] != 1 or shape[1] != planned["input_token_count"]:
        raise SyntaxRunnerError(f"model returned invalid raw logits shape: {shape}")
    row = logits[0, -1, :]
    if hasattr(row, "detach"):
        row = row.detach().float().cpu().numpy()
    try:
        return coherent.canonical_full_vocab_row(row)
    except coherent.CoherentReadoutError as error:
        raise SyntaxRunnerError("model returned an invalid next-token row") from error


def execute_call_plan(
    model: Any, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("mode") != "development":
        raise SyntaxRunnerError("invalid syntax call plan")
    if file_sha256(Path(__file__)) != plan["runner_code_sha256"]:
        raise SyntaxRunnerError("runner changed after call-plan freeze")
    if environment_lock() != plan["environment"]:
        raise SyntaxRunnerError("execution environment changed after call-plan freeze")
    if not callable(getattr(model, "eval", None)):
        raise SyntaxRunnerError("model must expose eval()")
    model.eval()
    validate_loaded_model(model, plan)
    records: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for row_index, planned in enumerate(plan["records"]):
        if planned["full_vocab_logits_row"] != row_index:
            raise SyntaxRunnerError("call-plan sidecar row order changed")
        row = _forward_next_logits(model, planned)
        if len(row) != planned["vocab_size"]:
            raise SyntaxRunnerError("model output vocabulary differs from frozen config")
        diagnostics = coherent.full_vocab_diagnostics(
            row,
            x_token_id=planned["x_token_id"],
            y_token_id=planned["y_token_id"],
        )
        maximum = np.max(row)
        maximum_token_ids = [int(value) for value in np.flatnonzero(row == maximum)]
        record = {
            "record_id": planned["record_id"],
            **{key: planned[key] for key in IDENTITY_KEYS},
            **diagnostics,
            "maximum_token_ids": maximum_token_ids,
            "maximum_tie_count": len(maximum_token_ids),
        }
        record["forward_trace_sha256"] = forward_trace_sha256(record)
        if set(record) != RECORD_KEYS or record_id(record) != record["record_id"]:
            raise SyntaxRunnerError("raw record does not preserve planned identity")
        records.append(record)
        rows.append(row)
    matrix = np.ascontiguousarray(np.stack(rows), dtype="<f4")
    if call_plan_sha256(records) != plan["call_plan_sha256"]:
        raise SyntaxRunnerError("raw records do not reproduce the call-plan hash")
    return records, matrix


def _serialize_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != value:
        raise SyntaxRunnerError(f"refusing to overwrite a different artifact: {path}")
    path.write_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, _serialize_json(value))


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    value = b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    _write_bytes(path, value)


def write_sidecar(path: Path, matrix: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(matrix, dtype="<f4"))
    if array.ndim != 2 or not np.isfinite(array).all():
        raise SyntaxRunnerError("full-vocabulary sidecar must be a finite f32 matrix")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _write_bytes(path, buffer.getvalue())


def manifest_payload(
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    matrix: np.ndarray | None = None,
    records_path: Path | None = None,
    logits_path: Path | None = None,
) -> dict[str, Any]:
    executed = records is not None
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "RAW_EXECUTION_COMPLETE_NOT_ANALYZED" if executed else "PLAN_AND_DESIGN_FROZEN_NO_FORWARD",
        "mode": "development",
        "purpose": "syntax_selection_only",
        "model_role": plan["model_role"],
        "selection_eligible_model": plan["selection_eligible_model"],
        "claim_scope": design["claim_scope"],
        "model_calls": 0 if records is None else len(records),
        "generation_used": False,
        "logit_processors_used": False,
        "confirmatory_execution": False,
        "locks": {
            key: design[key]
            for key in (
                "source_fixture_sha256",
                "source_manifest_sha256",
                "preregistration_sha256",
                "expected_runner_code_sha256",
                "expected_analyzer_code_sha256",
                "expected_model_id",
                "expected_model_revision",
                "expected_model_weights_sha256",
                "expected_tokenizer_id",
                "expected_tokenizer_revision",
                "expected_device",
                "expected_dtype",
                "expected_vocab_size",
                "call_plan_sha256",
            )
        },
        "environment": plan["environment"],
        "candidate_registry": plan["candidate_registry"],
        "gate_config": plan["gate_config"],
        "call_plan": {
            "schema_version": PLAN_SCHEMA,
            "record_count": plan["record_count"],
            "record_ids": plan["expected_record_ids"],
            "records": plan["records"],
        },
        "output": {
            "records_path": None if records_path is None else str(records_path),
            "records_file_sha256": None if records_path is None or not records_path.is_file() else file_sha256(records_path),
            "logits_path": None if logits_path is None else str(logits_path),
            "logits_file_sha256": None if logits_path is None or not logits_path.is_file() else file_sha256(logits_path),
            "full_vocab_matrix_sha256": None if matrix is None else coherent.full_vocab_matrix_sha256(matrix),
            "shape": None if matrix is None else list(matrix.shape),
            "row_record_ids": None if records is None else [record["record_id"] for record in records],
            "row_sha256": None if records is None else [record["full_vocab_logits_sha256"] for record in records],
        },
    }


def _validate_frozen_cli(
    args: argparse.Namespace,
    *,
    fixture_sha256: str,
    manifest_sha256: str,
    preregistration_sha256: str,
    tokenizer_id: str,
) -> None:
    expected = FROZEN_ROLE_LOCKS[args.model_role]
    observed = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_weights_sha256": args.model_weights_sha256,
    }
    if observed != expected:
        raise SyntaxRunnerError(f"model role does not match frozen lock: {args.model_role}")
    if tokenizer_id != args.model_id or args.tokenizer_revision != args.model_revision:
        raise SyntaxRunnerError("frozen tokenizer ID/revision must equal the model lock")
    if args.device != "mps" or args.dtype != "float32":
        raise SyntaxRunnerError("frozen syntax runs require device=mps and dtype=float32")
    if fixture_sha256 != FROZEN_FIXTURE_SHA256:
        raise SyntaxRunnerError("fixture bytes differ from the frozen preregistration")
    if manifest_sha256 != FROZEN_FIXTURE_MANIFEST_SHA256:
        raise SyntaxRunnerError("fixture manifest differs from the frozen preregistration")
    if preregistration_sha256 != FROZEN_PREREG_SHA256:
        raise SyntaxRunnerError("preregistration bytes differ from the frozen contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-provenance-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--model-role", choices=tuple(FROZEN_ROLE_LOCKS), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--dtype", choices=("float32",), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-design", type=Path)
    parser.add_argument("--output-records", type=Path)
    parser.add_argument("--output-logits", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for path, label in (
        (args.inputs, "inputs"),
        (args.input_provenance_manifest, "input provenance manifest"),
        (args.preregistration, "preregistration"),
        (args.analyzer, "analyzer"),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.plan_only:
        if args.output_design is None or any(
            value is not None for value in (args.design, args.output_records, args.output_logits)
        ):
            parser.error("plan-only requires --output-design and forbids execution outputs/design")
    elif any(value is None for value in (args.design, args.output_records, args.output_logits)) or args.output_design is not None:
        parser.error("execution requires --design/--output-records/--output-logits only")
    tokenizer_id = args.tokenizer_id or args.model_id
    fixture_sha256 = file_sha256(args.inputs)
    manifest_sha256 = file_sha256(args.input_provenance_manifest)
    preregistration_sha256 = file_sha256(args.preregistration)
    try:
        _validate_frozen_cli(
            args,
            fixture_sha256=fixture_sha256,
            manifest_sha256=manifest_sha256,
            preregistration_sha256=preregistration_sha256,
            tokenizer_id=tokenizer_id,
        )
        verify_cached_model_weights(
            args.model_id,
            args.model_revision,
            args.model_weights_sha256,
        )
        fixture = json.loads(args.inputs.read_text(encoding="utf-8"))
        tokenizer = v1_runner._load_hf_tokenizer(
            args.model_id,
            tokenizer_id,
            args.tokenizer_revision,
            local_files_only=True,
        )
        model_vocab_size = v1_runner._load_hf_config_vocab_size(
            args.model_id,
            args.model_revision,
            local_files_only=True,
        )
        plan = build_call_plan(
            fixture,
            tokenizer=tokenizer,
            model_role=args.model_role,
            model_id=args.model_id,
            model_revision=args.model_revision,
            model_weights_sha256=args.model_weights_sha256,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            model_vocab_size=model_vocab_size,
            dtype=args.dtype,
            device=args.device,
            source_fixture_sha256=fixture_sha256,
            source_manifest_sha256=manifest_sha256,
            preregistration_sha256=preregistration_sha256,
            analyzer_code_sha256=file_sha256(args.analyzer),
        )
        frozen_design = design_from_plan(plan)
        if args.plan_only:
            write_json(args.output_design, frozen_design)
            write_json(args.output_manifest, manifest_payload(plan, frozen_design))
            print(f"Wrote frozen syntax design: {args.output_design}")
            print(f"Wrote no-forward syntax plan: {args.output_manifest}")
            return
        supplied_design = json.loads(args.design.read_text(encoding="utf-8"))
        validate_plan_against_design(plan, supplied_design)
        model = v1_runner._load_hf_model(
            args.model_id,
            args.model_revision,
            args.dtype,
            local_files_only=True,
            device=args.device,
        )
        records, matrix = execute_call_plan(model, plan)
        write_jsonl(args.output_records, records)
        write_sidecar(args.output_logits, matrix)
        write_json(
            args.output_manifest,
            manifest_payload(
                plan,
                frozen_design,
                records=records,
                matrix=matrix,
                records_path=args.output_records,
                logits_path=args.output_logits,
            ),
        )
        print(f"Wrote raw syntax records: {args.output_records}")
        print(f"Wrote full-vocabulary sidecar: {args.output_logits}")
        print(f"Wrote execution manifest: {args.output_manifest}")
    except (SyntaxRunnerError, coherent.CoherentReadoutError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

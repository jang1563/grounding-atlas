"""Plan and execute the frozen coherent-readout v2 causal binding study.

This development-only runner performs full residual-state replacement at one
post-decoder-block, final-context-token site.  It never generates text and never
touches the biological fixture.  The complete discovery plan and held-out patch
template are frozen before model weights are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import coherent_binary_readout as coherent
    from . import model_hooks
    from . import run_coherent_binary_readout as v1_runner
    from . import run_coherent_readout_v2_syntax as syntax_runner
    from .causal_intervention import execution_input_sha256
except ImportError:  # direct execution from eval/
    import coherent_binary_readout as coherent
    import model_hooks
    import run_coherent_binary_readout as v1_runner
    import run_coherent_readout_v2_syntax as syntax_runner
    from causal_intervention import execution_input_sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v2_causal_binding"
    / "qwen2.5-1.5b-instruct"
)

PLAN_SCHEMA = "coherent-readout-v2-causal-binding-plan-v1"
DESIGN_SCHEMA = "coherent-readout-v2-causal-binding-design-v1"
PLAN_MANIFEST_SCHEMA = "coherent-readout-v2-causal-binding-plan-manifest-v1"
PROMPT_SCHEMA = "coherent-readout-v2-causal-binding-prompt-v1"
PATCH_TEMPLATE_SCHEMA = "coherent-readout-v2-causal-binding-patch-template-v1"
BASELINE_SCHEMA = "coherent-readout-v2-causal-binding-baseline-v1"
PATCH_SCHEMA = "coherent-readout-v2-causal-binding-patch-v1"
EXECUTION_MANIFEST_SCHEMA = "coherent-readout-v2-causal-binding-execution-v1"

FROZEN_PREREGISTRATION = ROOT / "docs" / "COHERENT_READOUT_V2_CAUSAL_BINDING_PREREG.md"
FROZEN_PREREGISTRATION_SHA256 = (
    "7cc0a11ee43490426de2338840cc2dfa9f62e0aecf259c16b9defb3da165eab0"
)
DISCOVERY_FIXTURE = ROOT / "signal" / "syntax" / "coherent_readout_v2_syntax_bank.json"
DISCOVERY_FIXTURE_SHA256 = (
    "d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8"
)
DISCOVERY_ANALYSIS = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "coherent_readout_v2_syntax"
    / "qwen2.5-1.5b-instruct"
    / "analysis.json"
)
DISCOVERY_ANALYSIS_SHA256 = (
    "dda99af4f3dfe4dda808e39b37f2e491cfd75af7766f1dc666c5fb577135415f"
)
DISCOVERY_RAW_RECORDS = DISCOVERY_ANALYSIS.with_name("raw_records.jsonl")
DISCOVERY_RAW_RECORDS_SHA256 = (
    "f085506de6bc1501cabfc43e25520f5397f4b97e486d2c5a31cbefd7e26ac453"
)
HOLDOUT_FIXTURE = (
    ROOT / "signal" / "syntax" / "coherent_readout_v2_causal_binding_bank.json"
)
HOLDOUT_FIXTURE_SHA256 = (
    "2c40ba0c796202059056aec4535fd7656eab2b446d8895816bbae2034ebcbcdb"
)
HOLDOUT_MANIFEST = HOLDOUT_FIXTURE.with_suffix(".manifest.json")
HOLDOUT_MANIFEST_SHA256 = (
    "fadc7eda8edc92e2b355a23db33898660265c2d2949cf1c7724172398b0dd8ac"
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
MODEL_VOCAB_SIZE = 151936
TOKENIZER_VOCAB_SIZE = 151665
X_TOKEN_ID = 55
Y_TOKEN_ID = 56
DEVICE = "mps"
DTYPE = "float32"
ATTENTION_IMPLEMENTATION = "sdpa"

CANDIDATE_ID = "system_user_exact"
SYSTEM_EXACT = (
    "Follow the user's label mapping. Your entire response must be exactly one "
    "character: X or Y. Do not write any other text."
)
USER_EXACT = (
    "Classify the record from its explicit declaration.\n"
    "{line_1}\n{line_2}\n{declaration}\nReturn exactly X or Y."
)
CHAT_FLAGS = {
    "add_generation_prompt": True,
    "continue_final_message": False,
    "enable_thinking": False,
}

LAYER_GRID = (0, 4, 8, 12, 16, 20, 24, 27)
PATCH_CONDITIONS = (
    "forward_paired",
    "forward_same_pair_x",
    "forward_unrelated_x",
    "reverse_paired",
    "reverse_same_pair",
    "reverse_unrelated",
    "identity_r",
    "identity_d",
)

DEFAULT_ANALYZER = ROOT / "eval" / "analyze_coherent_readout_v2_causal_binding.py"
DEFAULT_DESIGN = RESULT_ROOT / "design.json"
DEFAULT_PLAN_MANIFEST = RESULT_ROOT / "plan_manifest.json"
DEFAULT_LAYER_LOCK = RESULT_ROOT / "layer_lock.json"
DEFAULT_HOLDOUT_ENTRY = RESULT_ROOT / "holdout_entry.json"


class CausalBindingRunnerError(ValueError):
    """Raised when planning or execution violates the frozen contract."""


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
        raise CausalBindingRunnerError("float32 artifact contains non-finite values")
    header = canonical_json({"dtype": "<f4", "shape": list(array.shape)}).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise CausalBindingRunnerError(
            f"required distribution is unavailable: {distribution}"
        ) from error


def environment_lock(analyzer_path: Path) -> dict[str, Any]:
    if not analyzer_path.is_file():
        raise CausalBindingRunnerError(f"analyzer does not exist: {analyzer_path}")
    helpers = {
        "analyzer": file_sha256(analyzer_path),
        "causal_intervention": file_sha256(ROOT / "eval" / "causal_intervention.py"),
        "model_hooks": file_sha256(ROOT / "eval" / "model_hooks.py"),
        "run_coherent_binary_readout": file_sha256(
            ROOT / "eval" / "run_coherent_binary_readout.py"
        ),
        "run_coherent_readout_v2_syntax": file_sha256(
            ROOT / "eval" / "run_coherent_readout_v2_syntax.py"
        ),
        "runner": file_sha256(Path(__file__)),
    }
    try:
        import torch
    except ImportError as error:
        raise CausalBindingRunnerError("torch is required for the runtime lock") from error
    value = {
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
        "helper_file_sha256": helpers,
        "device": DEVICE,
        "dtype": DTYPE,
        "torch_runtime": {
            "mps_is_built": bool(torch.backends.mps.is_built()),
            "mps_is_available": bool(torch.backends.mps.is_available()),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "default_dtype": str(torch.get_default_dtype()),
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
    }
    return {**value, "environment_sha256": canonical_sha256(value)}


def _require_frozen_inputs() -> None:
    frozen = (
        (FROZEN_PREREGISTRATION, FROZEN_PREREGISTRATION_SHA256, "preregistration"),
        (DISCOVERY_FIXTURE, DISCOVERY_FIXTURE_SHA256, "discovery fixture"),
        (DISCOVERY_ANALYSIS, DISCOVERY_ANALYSIS_SHA256, "discovery analysis"),
        (
            DISCOVERY_RAW_RECORDS,
            DISCOVERY_RAW_RECORDS_SHA256,
            "discovery raw records",
        ),
        (HOLDOUT_FIXTURE, HOLDOUT_FIXTURE_SHA256, "holdout fixture"),
        (HOLDOUT_MANIFEST, HOLDOUT_MANIFEST_SHA256, "holdout manifest"),
    )
    for path, expected, label in frozen:
        if not path.is_file() or file_sha256(path) != expected:
            raise CausalBindingRunnerError(f"{label} differs from its frozen hash")


def _verify_cached_model_asset(filename: str, expected_sha256: str) -> Path:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        raise CausalBindingRunnerError(
            "cached model-asset verification requires huggingface_hub"
        ) from error
    resolved = try_to_load_from_cache(
        MODEL_ID, filename, revision=MODEL_REVISION
    )
    if not isinstance(resolved, str):
        raise CausalBindingRunnerError(
            f"the frozen cached model asset is unavailable: {filename}"
        )
    path = Path(resolved)
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise CausalBindingRunnerError(
            f"cached model asset differs from its frozen hash: {filename}"
        )
    return path


def _verify_cached_model_assets() -> None:
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    _verify_cached_model_asset("config.json", MODEL_CONFIG_SHA256)
    _verify_cached_model_asset("tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
    _verify_cached_model_asset("tokenizer.json", TOKENIZER_JSON_SHA256)


def _verify_cached_plan_assets() -> None:
    _verify_cached_model_asset("config.json", MODEL_CONFIG_SHA256)
    _verify_cached_model_asset("tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
    _verify_cached_model_asset("tokenizer.json", TOKENIZER_JSON_SHA256)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CausalBindingRunnerError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise CausalBindingRunnerError(f"JSON artifact must be an object: {path}")
    return value


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise CausalBindingRunnerError(f"{label} must contain exactly one row")
        raw = raw[0]
    if not raw or any(isinstance(x, bool) or not isinstance(x, int) for x in raw):
        raise CausalBindingRunnerError(f"{label} must be a nonempty integer vector")
    return [int(x) for x in raw]


def _contextual_token_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "rendered prompt"
    )
    combined = _as_int_vector(
        tokenizer.encode(rendered + answer, add_special_tokens=False),
        "prompt plus answer",
    )
    if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
        raise CausalBindingRunnerError(
            f"answer {answer!r} is not one contextual continuation token"
        )
    return combined[-1]


def _mapping_lines(item: Mapping[str, Any], order: str, mapping: str) -> tuple[str, str]:
    if order not in {"positive_first", "negative_first"}:
        raise CausalBindingRunnerError("unknown mapping-line order")
    if mapping not in {"positive_is_x", "positive_is_y"}:
        raise CausalBindingRunnerError("unknown label mapping")
    positive_label, negative_label = (
        ("X", "Y") if mapping == "positive_is_x" else ("Y", "X")
    )
    positive = f"label {positive_label} means {item['positive_class']}"
    negative = f"label {negative_label} means {item['negative_class']}"
    return (positive, negative) if order == "positive_first" else (negative, positive)


def _item_forms(item: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    polarity = item["truth_polarity"]
    if polarity == "positive":
        return {
            "mapping": "positive_is_x",
            "d_order": "positive_first",
            "r_order": "negative_first",
        }
    if polarity == "negative":
        return {
            "mapping": "positive_is_y",
            "d_order": "negative_first",
            "r_order": "positive_first",
        }
    raise CausalBindingRunnerError("truth polarity must be positive or negative")


def _render_prompt(
    tokenizer: Any,
    item: Mapping[str, Any],
    *,
    bank_role: str,
    prompt_role: str,
    source_fixture_sha256: str,
    prior_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prompt_role not in {"D", "R"}:
        raise CausalBindingRunnerError("prompt role must be D or R")
    forms = _item_forms(item)
    order = forms["d_order"] if prompt_role == "D" else forms["r_order"]
    line_1, line_2 = _mapping_lines(item, order, forms["mapping"])
    user = USER_EXACT.format(
        line_1=line_1,
        line_2=line_2,
        declaration=item["declaration_text"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_EXACT},
        {"role": "user", "content": user},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **CHAT_FLAGS)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **CHAT_FLAGS,
    )
    input_ids = _as_int_vector(encoded["input_ids"], "input_ids")
    attention_mask = _as_int_vector(
        encoded.get("attention_mask", [1] * len(input_ids)), "attention_mask"
    )
    if len(input_ids) != len(attention_mask) or any(x != 1 for x in attention_mask):
        raise CausalBindingRunnerError("prompts must be unpadded and fully attended")
    if _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False), "retokenized prompt"
    ) != input_ids:
        raise CausalBindingRunnerError("rendered chat does not reproduce planned IDs")
    if _contextual_token_id(tokenizer, rendered, "X") != X_TOKEN_ID:
        raise CausalBindingRunnerError("contextual X token ID changed")
    if _contextual_token_id(tokenizer, rendered, "Y") != Y_TOKEN_ID:
        raise CausalBindingRunnerError("contextual Y token ID changed")
    prior_identity = {
        "prior_syntax_record_id": None,
        "prior_full_vocab_logits_sha256": None,
        "prior_x_logit": None,
        "prior_y_logit": None,
    }
    if prior_record is not None:
        expected_prior = {
            "candidate_id": CANDIDATE_ID,
            "item_id": item["item_id"],
            "order": order,
            "mapping": forms["mapping"],
            "prompt_sha256": text_sha256(rendered),
            "execution_input_sha256": execution_input_sha256(
                input_ids, attention_mask
            ),
            "x_token_id": X_TOKEN_ID,
            "y_token_id": Y_TOKEN_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        }
        if any(prior_record.get(key) != value for key, value in expected_prior.items()):
            raise CausalBindingRunnerError("prior syntax record does not match prompt")
        prior_identity = {
            "prior_syntax_record_id": prior_record["record_id"],
            "prior_full_vocab_logits_sha256": prior_record[
                "full_vocab_logits_sha256"
            ],
            "prior_x_logit": prior_record["x_logit"],
            "prior_y_logit": prior_record["y_logit"],
        }
    identity = {
        "schema_version": PROMPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "bank_role": bank_role,
        "source_fixture_sha256": source_fixture_sha256,
        "fixture_item_sha256": canonical_sha256(item),
        "item_id": item["item_id"],
        "pair_id": item["pair_id"],
        "truth_polarity": item["truth_polarity"],
        "positive_class": item["positive_class"],
        "negative_class": item["negative_class"],
        "declared_class": item["declared_class"],
        "prompt_role": prompt_role,
        "order": order,
        "mapping": forms["mapping"],
        "messages_sha256": canonical_sha256(messages),
        "prompt_sha256": text_sha256(rendered),
        "execution_input_sha256": execution_input_sha256(
            input_ids, attention_mask
        ),
        "input_token_count": len(input_ids),
        "input_token_multiset_sha256": canonical_sha256(sorted(input_ids)),
        "x_token_id": X_TOKEN_ID,
        "y_token_id": Y_TOKEN_ID,
        "final_token_index": len(input_ids) - 1,
        **prior_identity,
    }
    return {
        "prompt_id": canonical_sha256(identity),
        **identity,
        "messages": messages,
        "rendered_chat": rendered,
        "execution_input_ids": input_ids,
        "execution_attention_mask": attention_mask,
    }


def _discovery_items() -> list[dict[str, Any]]:
    fixture = syntax_runner.validate_fixture(_load_json(DISCOVERY_FIXTURE))
    items = [dict(item) for item in fixture["items"]]
    pair_order = [pair["pair_id"] for pair in fixture["pair_registry"]]
    by_key = {(item["pair_id"], item["truth_polarity"]): item for item in items}
    by_pair = {pair_id: [] for pair_id in pair_order}
    for item in items:
        by_pair[item["pair_id"]].append(item)
    unrelated_pair = {
        pair_id: pair_order[index + 1] if index % 2 == 0 else pair_order[index - 1]
        for index, pair_id in enumerate(pair_order)
    }
    enriched = []
    for item in items:
        opposite = next(
            candidate
            for candidate in by_pair[item["pair_id"]]
            if candidate["item_id"] != item["item_id"]
        )
        unrelated_pair_id = unrelated_pair[item["pair_id"]]
        unrelated = by_key[(unrelated_pair_id, item["truth_polarity"])]
        enriched.append(
            {
                **item,
                "same_pair_counterfactual_item_id": opposite["item_id"],
                "unrelated_same_polarity_item_id": unrelated["item_id"],
                "unrelated_control_cluster_id": "control_dyad:"
                + ":".join(sorted((item["pair_id"], unrelated_pair_id))),
            }
        )
    return sorted(enriched, key=lambda item: item["item_id"])


def _holdout_items() -> list[dict[str, Any]]:
    fixture = _load_json(HOLDOUT_FIXTURE)
    if fixture.get("schema_version") != "coherent-readout-v2-causal-binding-bank-v1":
        raise CausalBindingRunnerError("holdout fixture schema changed")
    items = fixture.get("items")
    if not isinstance(items, list) or len(items) != 96:
        raise CausalBindingRunnerError("holdout fixture must contain 96 items")
    return [dict(item) for item in items]


def _patch_refs(
    item: Mapping[str, Any], condition: str
) -> tuple[str, str, str, str]:
    paired = item["item_id"]
    same = item["same_pair_counterfactual_item_id"]
    unrelated = item["unrelated_same_polarity_item_id"]
    values = {
        "forward_paired": (paired, "D", paired, "R"),
        "forward_same_pair_x": (same, "D", paired, "R"),
        "forward_unrelated_x": (unrelated, "D", paired, "R"),
        "reverse_paired": (paired, "R", paired, "D"),
        "reverse_same_pair": (same, "R", paired, "D"),
        "reverse_unrelated": (unrelated, "R", paired, "D"),
        "identity_r": (paired, "R", paired, "R"),
        "identity_d": (paired, "D", paired, "D"),
    }
    try:
        return values[condition]
    except KeyError as error:
        raise CausalBindingRunnerError(f"unknown patch condition: {condition}") from error


def _patch_template(
    *,
    bank_role: str,
    item: Mapping[str, Any],
    condition: str,
    source_prompt: Mapping[str, Any],
    target_prompt: Mapping[str, Any],
    layer: int | None,
) -> dict[str, Any]:
    identity = {
        "schema_version": PATCH_TEMPLATE_SCHEMA,
        "bank_role": bank_role,
        "item_id": item["item_id"],
        "pair_id": item["pair_id"],
        "truth_polarity": item["truth_polarity"],
        "condition": condition,
        "dependency_cluster_id": (
            item["unrelated_control_cluster_id"]
            if condition in {"forward_unrelated_x", "reverse_unrelated"}
            else item["pair_id"]
        ),
        "source_prompt_id": source_prompt["prompt_id"],
        "target_prompt_id": target_prompt["prompt_id"],
        "source_target_token_count_difference": (
            source_prompt["input_token_count"] - target_prompt["input_token_count"]
        ),
        "layer": layer,
        "layer_selector": "fixed" if layer is not None else "layer_lock",
        "hook_site": "decoder_block_output_resid_post",
        "token_site": "final_attended_context_token",
        "source_token_index": -1,
        "target_token_index": -1,
        "patch_kind": "full_state_replacement",
        "patch_strength": 1.0,
    }
    return {"template_id": canonical_sha256(identity), **identity}


def _control_matching_registry(
    items: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze output-independent exact-length specificity units."""

    by_key = {
        (template["item_id"], template["condition"]): template
        for template in templates
        if template["bank_role"] == "holdout"
    }
    pair_to_items: dict[str, list[str]] = {}
    pair_to_cluster: dict[str, str] = {}
    for item in items:
        item_id = item["item_id"]
        pair_id = item["pair_id"]
        pair_to_items.setdefault(pair_id, []).append(item_id)
        cluster_id = item["unrelated_control_cluster_id"]
        if pair_id in pair_to_cluster and pair_to_cluster[pair_id] != cluster_id:
            raise CausalBindingRunnerError(
                "one holdout pair maps to multiple unrelated-control dyads"
            )
        pair_to_cluster[pair_id] = cluster_id
    if len(pair_to_items) != 48 or any(
        len(member_ids) != 2 for member_ids in pair_to_items.values()
    ):
        raise CausalBindingRunnerError("holdout pair registry changed")

    same_pair_exact = sorted(
        pair_id
        for pair_id, member_ids in pair_to_items.items()
        if all(
            by_key[(item_id, condition)][
                "source_target_token_count_difference"
            ]
            == 0
            for item_id in member_ids
            for condition in ("forward_same_pair_x", "reverse_same_pair")
        )
    )
    dyad_to_pairs: dict[str, list[str]] = {}
    for pair_id, cluster_id in pair_to_cluster.items():
        dyad_to_pairs.setdefault(cluster_id, []).append(pair_id)
    if len(dyad_to_pairs) != 24 or any(
        len(pair_ids) != 2 for pair_ids in dyad_to_pairs.values()
    ):
        raise CausalBindingRunnerError(
            "unrelated controls must form 24 reciprocal two-pair dyads"
        )
    unrelated_exact = sorted(
        cluster_id
        for cluster_id, pair_ids in dyad_to_pairs.items()
        if all(
            by_key[(item_id, condition)][
                "source_target_token_count_difference"
            ]
            == 0
            for pair_id in pair_ids
            for item_id in pair_to_items[pair_id]
            for condition in ("forward_unrelated_x", "reverse_unrelated")
        )
    )
    if len(same_pair_exact) != 31 or len(unrelated_exact) != 22:
        raise CausalBindingRunnerError(
            "frozen exact-length specificity-unit counts changed"
        )
    return {
        "eligibility_basis": (
            "frozen_source_target_input_token_count_difference_equals_zero"
        ),
        "computed_before_first_new_forward": True,
        "outcome_dependent": False,
        "primary_transfer_lexical_pairs": sorted(pair_to_items),
        "same_pair_exact_length_lexical_pairs": same_pair_exact,
        "unrelated_control_reciprocal_dyads": {
            cluster_id: sorted(pair_ids)
            for cluster_id, pair_ids in sorted(dyad_to_pairs.items())
        },
        "unrelated_exact_length_reciprocal_dyads": unrelated_exact,
    }


def _config_and_tokenizer() -> tuple[Any, int]:
    _verify_cached_plan_assets()
    tokenizer = v1_runner._load_hf_tokenizer(
        MODEL_ID,
        MODEL_ID,
        MODEL_REVISION,
        local_files_only=True,
    )
    vocab_size = v1_runner._load_hf_config_vocab_size(
        MODEL_ID, MODEL_REVISION, local_files_only=True
    )
    if vocab_size != MODEL_VOCAB_SIZE or len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise CausalBindingRunnerError("model/tokenizer vocabulary lock changed")
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise CausalBindingRunnerError("effective tokenizer chat template changed")
    return tokenizer, vocab_size


def build_plan(tokenizer: Any, analyzer_path: Path) -> dict[str, Any]:
    """Build the complete zero-forward prompt and intervention plan."""

    _require_frozen_inputs()
    syntax_runner.verify_cached_model_weights(
        MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256
    )
    if v1_runner.chat_template_sha256(tokenizer) != CHAT_TEMPLATE_SHA256:
        raise CausalBindingRunnerError("effective tokenizer chat template changed")
    environment = environment_lock(analyzer_path)
    prompts: list[dict[str, Any]] = []
    items_by_bank: dict[str, list[dict[str, Any]]] = {
        "discovery": _discovery_items(),
        "holdout": _holdout_items(),
    }
    fixture_hashes = {
        "discovery": DISCOVERY_FIXTURE_SHA256,
        "holdout": HOLDOUT_FIXTURE_SHA256,
    }
    prior_rows = [
        row
        for row in load_jsonl(DISCOVERY_RAW_RECORDS)
        if row.get("candidate_id") == CANDIDATE_ID
    ]
    prior_by_key = {
        (row["item_id"], row["order"], row["mapping"]): row
        for row in prior_rows
    }
    if len(prior_rows) != 64 or len(prior_by_key) != 64:
        raise CausalBindingRunnerError("prior syntax record registry changed")
    item_registry: list[dict[str, Any]] = []
    for bank_role, items in items_by_bank.items():
        for item in items:
            item_registry.append(
                {
                    "bank_role": bank_role,
                    "item_id": item["item_id"],
                    "pair_id": item["pair_id"],
                    "truth_polarity": item["truth_polarity"],
                    "same_pair_counterfactual_item_id": item[
                        "same_pair_counterfactual_item_id"
                    ],
                    "unrelated_same_polarity_item_id": item[
                        "unrelated_same_polarity_item_id"
                    ],
                    "unrelated_control_cluster_id": item[
                        "unrelated_control_cluster_id"
                    ],
                    "fixture_item_sha256": canonical_sha256(item),
                }
            )
            pair_prompts = []
            item_forms = _item_forms(item)
            for role in ("D", "R"):
                order = (
                    item_forms["d_order"]
                    if role == "D"
                    else item_forms["r_order"]
                )
                prior = (
                    prior_by_key[(item["item_id"], order, item_forms["mapping"])]
                    if bank_role == "discovery"
                    else None
                )
                pair_prompts.append(
                    _render_prompt(
                        tokenizer,
                        item,
                        bank_role=bank_role,
                        prompt_role=role,
                        source_fixture_sha256=fixture_hashes[bank_role],
                        prior_record=prior,
                    )
                )
            d_prompt, r_prompt = pair_prompts
            if d_prompt["input_token_count"] != r_prompt["input_token_count"]:
                raise CausalBindingRunnerError("D/R token counts differ")
            if (
                d_prompt["input_token_multiset_sha256"]
                != r_prompt["input_token_multiset_sha256"]
            ):
                raise CausalBindingRunnerError("D/R token multisets differ")
            prompts.extend(pair_prompts)

    if len(prompts) != 224 or len({p["prompt_id"] for p in prompts}) != 224:
        raise CausalBindingRunnerError("plan must contain 224 unique baseline prompts")
    prompt_by_key = {
        (prompt["bank_role"], prompt["item_id"], prompt["prompt_role"]): prompt
        for prompt in prompts
    }
    templates: list[dict[str, Any]] = []
    for bank_role, items in items_by_bank.items():
        layers: Sequence[int | None] = LAYER_GRID if bank_role == "discovery" else (None,)
        for item in items:
            for condition in PATCH_CONDITIONS:
                source_item, source_role, target_item, target_role = _patch_refs(
                    item, condition
                )
                source = prompt_by_key[(bank_role, source_item, source_role)]
                target = prompt_by_key[(bank_role, target_item, target_role)]
                for layer in layers:
                    templates.append(
                        _patch_template(
                            bank_role=bank_role,
                            item=item,
                            condition=condition,
                            source_prompt=source,
                            target_prompt=target,
                            layer=layer,
                        )
                    )
    expected_templates = 16 * 8 * 8 + 96 * 8
    if len(templates) != expected_templates:
        raise CausalBindingRunnerError("patch-template count changed")
    if len({template["template_id"] for template in templates}) != len(templates):
        raise CausalBindingRunnerError("patch templates are not unique")

    control_matching_registry = _control_matching_registry(
        items_by_bank["holdout"], templates
    )

    plan_core = {
        "schema_version": PLAN_SCHEMA,
        "analysis_id": "coherent-readout-v2-causal-decision-state-transfer-v1",
        "freeze_date": "2026-08-02",
        "mode": "development",
        "purpose": "non_biological_causal_decision_state_transfer",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_weights_sha256": MODEL_WEIGHTS_SHA256,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "layers": MODEL_LAYERS,
            "hidden_width": MODEL_WIDTH,
            "vocab_size": MODEL_VOCAB_SIZE,
            "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
            "device": DEVICE,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "system_template": SYSTEM_EXACT,
            "user_template": USER_EXACT,
            "chat_flags": CHAT_FLAGS,
            "x_token_id": X_TOKEN_ID,
            "y_token_id": Y_TOKEN_ID,
        },
        "locks": {
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
            "discovery_fixture_sha256": DISCOVERY_FIXTURE_SHA256,
            "discovery_analysis_sha256": DISCOVERY_ANALYSIS_SHA256,
            "discovery_raw_records_sha256": DISCOVERY_RAW_RECORDS_SHA256,
            "holdout_fixture_sha256": HOLDOUT_FIXTURE_SHA256,
            "holdout_manifest_sha256": HOLDOUT_MANIFEST_SHA256,
            "runner_sha256": file_sha256(Path(__file__)),
            "analyzer_sha256": file_sha256(analyzer_path),
            "model_hooks_sha256": file_sha256(ROOT / "eval" / "model_hooks.py"),
        },
        "environment": environment,
        "layer_grid": list(LAYER_GRID),
        "patch_conditions": list(PATCH_CONDITIONS),
        "control_matching_registry": control_matching_registry,
        "item_registry": sorted(
            item_registry, key=lambda x: (x["bank_role"], x["item_id"])
        ),
        "prompts": sorted(
            prompts,
            key=lambda x: (x["bank_role"], x["item_id"], x["prompt_role"]),
        ),
        "patch_templates": sorted(
            templates,
            key=lambda x: (
                x["bank_role"],
                x["item_id"],
                -1 if x["layer"] is None else x["layer"],
                x["condition"],
            ),
        ),
        "expected_counts": {
            "discovery_pairs": 8,
            "discovery_items": 16,
            "discovery_prompts": 32,
            "discovery_patch_records": 1024,
            "holdout_pairs": 48,
            "holdout_items": 96,
            "holdout_prompts": 192,
            "holdout_patch_records": 768,
        },
        "model_calls_before_plan_freeze": 0,
        "generation_used": False,
        "biological_fixture_accessed": False,
    }
    return {**plan_core, "call_plan_sha256": canonical_sha256(plan_core)}


def design_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise CausalBindingRunnerError("invalid plan schema")
    return {
        "schema_version": DESIGN_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "freeze_date": plan["freeze_date"],
        "mode": plan["mode"],
        "purpose": plan["purpose"],
        "claim_scope": (
            "model_prompt_site_bank_specific_causal_state_transfer_only_no_biology_"
            "latent_knowledge_activation_gap_bottleneck_or_physical_law"
        ),
        "model": plan["model"],
        "candidate": plan["candidate"],
        "locks": plan["locks"],
        "environment": plan["environment"],
        "layer_grid": plan["layer_grid"],
        "patch_conditions": plan["patch_conditions"],
        "control_matching_registry": plan["control_matching_registry"],
        "expected_counts": plan["expected_counts"],
        "expected_prompt_ids": sorted(p["prompt_id"] for p in plan["prompts"]),
        "expected_patch_template_ids": sorted(
            p["template_id"] for p in plan["patch_templates"]
        ),
        "call_plan_sha256": plan["call_plan_sha256"],
        "selection_rule": {
            "specificity_used_for_layer_selection": False,
            "tie_break": "earliest_layer_in_frozen_grid",
            "selectable_layers": [0, 4, 8, 12, 16, 20, 24],
            "engineering_only_layer": 27,
            "transfer_fraction_floor": 0.30,
            "transfer_lodo_fraction_floor": 0.20,
            "minimum_positive_discovery_pairs": 7,
        },
        "holdout_gate": {
            "minimum_d_correct_rate": 0.95,
            "minimum_r_incorrect_rate": 0.75,
            "minimum_mean_label_mass": 0.95,
            "minimum_mean_gap": 1.0,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 260802,
        },
        "numerical_tolerance": 1e-4,
        "confirmatory_execution": False,
        "biological_execution_allowed": False,
    }


def _plan_payload_without_sha(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "call_plan_sha256"}


def validate_plan(plan: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    observed_sha = canonical_sha256(_plan_payload_without_sha(plan))
    if observed_sha != plan.get("call_plan_sha256"):
        raise CausalBindingRunnerError("call-plan SHA does not reproduce")
    holdout_items = [
        item for item in plan.get("item_registry", []) if item.get("bank_role") == "holdout"
    ]
    observed_registry = _control_matching_registry(
        holdout_items, plan.get("patch_templates", [])
    )
    if observed_registry != plan.get("control_matching_registry"):
        raise CausalBindingRunnerError("control-matching registry does not reconstruct")
    expected_design = design_from_plan(plan)
    if design != expected_design:
        raise CausalBindingRunnerError("plan differs from frozen design")
    if plan["locks"]["runner_sha256"] != file_sha256(Path(__file__)):
        raise CausalBindingRunnerError("runner changed after plan freeze")
    if plan["environment"] != environment_lock(DEFAULT_ANALYZER):
        raise CausalBindingRunnerError("execution environment changed after plan freeze")
    _require_frozen_inputs()
    return expected_design


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise CausalBindingRunnerError(f"refusing to overwrite different artifact: {path}")
    path.write_bytes(payload)


def _require_absent(paths: Sequence[Path], phase: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise CausalBindingRunnerError(
            f"refusing to re-enter {phase}; phase artifact already exists: {existing}"
        )


def _write_execution_attempt(
    path: Path, *, phase: str, plan: Mapping[str, Any]
) -> None:
    write_json(
        path,
        {
            "schema_version": "coherent-readout-v2-causal-binding-attempt-v1",
            "status": "EXECUTION_ARMED_NO_FORWARD_RECORDED",
            "phase": phase,
            "call_plan_sha256": plan["call_plan_sha256"],
            "runner_sha256": file_sha256(Path(__file__)),
            "environment_sha256": plan["environment"]["environment_sha256"],
            "model_calls_recorded_before_attempt": 0,
            "biological_model_calls": 0,
        },
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_bytes(path, payload)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for value in values
    )
    _write_bytes(path, payload)


def write_array(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if not np.isfinite(array).all():
        raise CausalBindingRunnerError("activation sidecar is not finite")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _write_bytes(path, stream.getvalue())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CausalBindingRunnerError(f"JSONL artifact does not exist: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CausalBindingRunnerError(f"invalid JSONL artifact: {path}") from error
        if not isinstance(value, dict):
            raise CausalBindingRunnerError("JSONL records must be objects")
        records.append(value)
    return records


def _full_vocab_diagnostics(row: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if value.shape != (MODEL_VOCAB_SIZE,) or not np.isfinite(value).all():
        raise CausalBindingRunnerError("model returned an invalid vocabulary row")
    x_logit = float(value[X_TOKEN_ID])
    y_logit = float(value[Y_TOKEN_ID])
    maximum = float(np.max(value))
    maximum_ids = [int(x) for x in np.flatnonzero(value == maximum)]
    peak = float(np.max(value.astype(np.float64)))
    logsumexp = peak + math.log(
        float(np.exp(value.astype(np.float64) - peak).sum())
    )
    label_logsumexp = float(np.logaddexp(x_logit, y_logit))
    return {
        "x_logit": x_logit,
        "y_logit": y_logit,
        "x_minus_y_margin": x_logit - y_logit,
        "full_vocab_logsumexp": logsumexp,
        "label_probability_mass": math.exp(label_logsumexp - logsumexp),
        "greedy_token_id": maximum_ids[0],
        "greedy_logit": maximum,
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": coherent.full_vocab_logits_sha256(value),
    }


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise CausalBindingRunnerError("model has no parameter device") from error


def validate_loaded_model(model: Any) -> None:
    import torch

    layers = model_hooks.resolve_decoder_layers(model)
    if len(layers) != MODEL_LAYERS:
        raise CausalBindingRunnerError("loaded decoder layer count changed")
    config = getattr(model, "config", None)
    if getattr(config, "hidden_size", None) != MODEL_WIDTH:
        raise CausalBindingRunnerError("loaded hidden width changed")
    if getattr(config, "vocab_size", None) != MODEL_VOCAB_SIZE:
        raise CausalBindingRunnerError("loaded vocabulary size changed")
    if (
        getattr(config, "_attn_implementation", None)
        != ATTENTION_IMPLEMENTATION
    ):
        raise CausalBindingRunnerError("loaded attention implementation changed")
    tensors = [*model.parameters(), *model.buffers()]
    if {tensor.device.type for tensor in tensors} != {DEVICE}:
        raise CausalBindingRunnerError("loaded model tensors are not all on MPS")
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if not floating or {tensor.dtype for tensor in floating} != {torch.float32}:
        raise CausalBindingRunnerError("loaded floating tensors are not all float32")


def _inputs(prompt: Mapping[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = _model_device(model)
    return {
        "input_ids": torch.tensor(
            [prompt["execution_input_ids"]], dtype=torch.long, device=device
        ),
        "attention_mask": torch.tensor(
            [prompt["execution_attention_mask"]], dtype=torch.long, device=device
        ),
    }


def _baseline_forward(
    model: Any, prompt: Mapping[str, Any], *, duplicate: bool
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    captures = [
        model_hooks.ResidualStreamCapture(model, layer, token_index=-1)
        for layer in range(MODEL_LAYERS)
    ]
    with torch.inference_mode(), ExitStack() as stack:
        for capture in captures:
            stack.enter_context(capture)
        output = model(**_inputs(prompt, model), use_cache=False, return_dict=True)
    if any(len(capture.values) != 1 or capture.active for capture in captures):
        raise CausalBindingRunnerError("a baseline capture did not fire exactly once")
    activations = np.stack(
        [
            capture.values[0][0].detach().float().cpu().numpy()
            for capture in captures
        ]
    ).astype("<f4", copy=False)
    if activations.shape != (MODEL_LAYERS, MODEL_WIDTH):
        raise CausalBindingRunnerError("captured activation shape changed")
    row = output.logits[0, -1, :].detach().float().cpu().numpy()
    diagnostics = _full_vocab_diagnostics(row)
    duplicate_diagnostics = None
    if duplicate:
        with torch.inference_mode():
            duplicate_output = model(
                **_inputs(prompt, model), use_cache=False, return_dict=True
            )
        duplicate_row = (
            duplicate_output.logits[0, -1, :].detach().float().cpu().numpy()
        )
        duplicate_diagnostics = _full_vocab_diagnostics(duplicate_row)
    return (
        {
            "diagnostics": diagnostics,
            "duplicate_diagnostics": duplicate_diagnostics,
            "activation_layer_sha256": [
                f32_sha256(activations[layer]) for layer in range(MODEL_LAYERS)
            ],
        },
        activations,
    )


def _patch_forward(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    source_activation: np.ndarray,
    layer: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    base_transform = model_hooks.patch_transform(
        source_activation, token_index=-1, strength=1.0
    )
    trace = {
        "hook_calls": 0,
        "non_target_tokens_unchanged": True,
        "patched_token_matches_source": True,
    }

    def traced_transform(hidden: Any) -> Any:
        trace["hook_calls"] += 1
        changed = base_transform(hidden)
        trace["non_target_tokens_unchanged"] = bool(
            trace["non_target_tokens_unchanged"]
            and torch.equal(changed[:, :-1, :], hidden[:, :-1, :])
        )
        expected = torch.as_tensor(
            source_activation, device=hidden.device, dtype=hidden.dtype
        ).reshape(1, -1)
        trace["patched_token_matches_source"] = bool(
            trace["patched_token_matches_source"]
            and torch.equal(changed[:, -1, :], expected)
        )
        return changed

    intervention = model_hooks.ResidualStreamIntervention(
        model, layer, traced_transform
    )
    with torch.inference_mode(), intervention:
        output = model(**_inputs(prompt, model), use_cache=False, return_dict=True)
    if intervention.active or trace["hook_calls"] != 1:
        raise CausalBindingRunnerError("patch hook count or cleanup gate failed")
    if not trace["non_target_tokens_unchanged"]:
        raise CausalBindingRunnerError("patch modified a non-target token")
    if not trace["patched_token_matches_source"]:
        raise CausalBindingRunnerError("patched token differs from source activation")
    row = output.logits[0, -1, :].detach().float().cpu().numpy()
    return _full_vocab_diagnostics(row), trace


def _baseline_record(
    prompt: Mapping[str, Any],
    *,
    phase: str,
    activation_row: int,
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": BASELINE_SCHEMA,
        "phase": phase,
        "bank_role": prompt["bank_role"],
        "prompt_id": prompt["prompt_id"],
        "prompt_role": prompt["prompt_role"],
        "item_id": prompt["item_id"],
        "pair_id": prompt["pair_id"],
        "truth_polarity": prompt["truth_polarity"],
        "order": prompt["order"],
        "mapping": prompt["mapping"],
        "execution_input_sha256": prompt["execution_input_sha256"],
        "activation_row": activation_row,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "runner_sha256": file_sha256(Path(__file__)),
    }
    return {
        "baseline_id": canonical_sha256(identity),
        **identity,
        "diagnostics": measurement["diagnostics"],
        "duplicate_diagnostics": measurement["duplicate_diagnostics"],
        "activation_layer_sha256": measurement["activation_layer_sha256"],
    }


def _patch_record(
    template: Mapping[str, Any],
    *,
    phase: str,
    layer: int,
    source_baseline: Mapping[str, Any],
    target_baseline: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": PATCH_SCHEMA,
        "phase": phase,
        "bank_role": template["bank_role"],
        "template_id": template["template_id"],
        "item_id": template["item_id"],
        "pair_id": template["pair_id"],
        "truth_polarity": template["truth_polarity"],
        "condition": template["condition"],
        "dependency_cluster_id": template["dependency_cluster_id"],
        "layer": layer,
        "hook_site": template["hook_site"],
        "token_site": template["token_site"],
        "source_prompt_id": template["source_prompt_id"],
        "target_prompt_id": template["target_prompt_id"],
        "source_target_token_count_difference": template[
            "source_target_token_count_difference"
        ],
        "source_baseline_id": source_baseline["baseline_id"],
        "target_baseline_id": target_baseline["baseline_id"],
        "source_activation_row": source_baseline["activation_row"],
        "source_activation_sha256": source_baseline[
            "activation_layer_sha256"
        ][layer],
        "patch_kind": template["patch_kind"],
        "patch_strength": template["patch_strength"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "runner_sha256": file_sha256(Path(__file__)),
        "model_hooks_sha256": file_sha256(ROOT / "eval" / "model_hooks.py"),
    }
    return {
        "patch_id": canonical_sha256(identity),
        **identity,
        "diagnostics": dict(diagnostics),
        "hook_trace": dict(trace),
    }


def _phase_prompts(plan: Mapping[str, Any], bank_role: str) -> list[dict[str, Any]]:
    return [dict(p) for p in plan["prompts"] if p["bank_role"] == bank_role]


def _phase_templates(
    plan: Mapping[str, Any], bank_role: str
) -> list[dict[str, Any]]:
    return [
        dict(template)
        for template in plan["patch_templates"]
        if template["bank_role"] == bank_role
    ]


def _execute_baselines(
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    duplicate: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = []
    activations = []
    for index, prompt in enumerate(prompts):
        measurement, captured = _baseline_forward(
            model, prompt, duplicate=duplicate
        )
        records.append(
            _baseline_record(
                prompt,
                phase=phase,
                activation_row=index,
                measurement=measurement,
            )
        )
        activations.append(captured)
    matrix = np.ascontiguousarray(np.stack(activations), dtype="<f4")
    expected = (len(prompts), MODEL_LAYERS, MODEL_WIDTH)
    if matrix.shape != expected:
        raise CausalBindingRunnerError("baseline activation matrix shape changed")
    return records, matrix


def _execute_patches(
    model: Any,
    templates: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    *,
    phase: str,
    selected_layer: int | None,
) -> list[dict[str, Any]]:
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    baseline_by_prompt = {row["prompt_id"]: row for row in baselines}
    records = []
    for template in templates:
        layer = template["layer"] if template["layer"] is not None else selected_layer
        if not isinstance(layer, int) or layer not in LAYER_GRID:
            raise CausalBindingRunnerError("patch layer is not frozen")
        source = baseline_by_prompt[template["source_prompt_id"]]
        target = baseline_by_prompt[template["target_prompt_id"]]
        source_activation = activations[source["activation_row"], layer]
        if f32_sha256(source_activation) != source["activation_layer_sha256"][layer]:
            raise CausalBindingRunnerError("source activation sidecar hash mismatch")
        diagnostics, trace = _patch_forward(
            model,
            prompt_by_id[target["prompt_id"]],
            source_activation=source_activation,
            layer=layer,
        )
        records.append(
            _patch_record(
                template,
                phase=phase,
                layer=layer,
                source_baseline=source,
                target_baseline=target,
                diagnostics=diagnostics,
                trace=trace,
            )
        )
    return records


def _execution_manifest(
    *,
    phase: str,
    plan: Mapping[str, Any],
    design: Mapping[str, Any],
    model_calls: int,
    records_path: Path,
    records_count: int,
    activations_path: Path | None = None,
    activations: np.ndarray | None = None,
    patch_records_path: Path | None = None,
    patch_records_count: int = 0,
    selected_layer: int | None = None,
    attempt_path: Path,
) -> dict[str, Any]:
    if not attempt_path.is_file():
        raise CausalBindingRunnerError("execution attempt receipt is missing")
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "status": "EXECUTION_COMPLETE_NOT_ANALYZED",
        "phase": phase,
        "mode": "development",
        "claim_scope": design["claim_scope"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "plan_manifest_file_sha256": file_sha256(DEFAULT_PLAN_MANIFEST),
        "selected_layer": selected_layer,
        "attempt_receipt": {
            "path": str(attempt_path),
            "file_sha256": file_sha256(attempt_path),
        },
        "model_calls": model_calls,
        "generation_used": False,
        "biological_model_calls": 0,
        "records": {
            "path": str(records_path),
            "count": records_count,
            "file_sha256": file_sha256(records_path),
        },
        "activations": None
        if activations_path is None or activations is None
        else {
            "path": str(activations_path),
            "file_sha256": file_sha256(activations_path),
            "logical_sha256": f32_sha256(activations),
            "shape": list(activations.shape),
            "dtype": "<f4",
        },
        "patch_records": None
        if patch_records_path is None
        else {
            "path": str(patch_records_path),
            "count": patch_records_count,
            "file_sha256": file_sha256(patch_records_path),
        },
        "locks": plan["locks"],
        "environment": plan["environment"],
    }


_RECONSTRUCTED_PLAN_CACHE: tuple[dict[str, Any], dict[str, Any]] | None = None


def _load_frozen_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    global _RECONSTRUCTED_PLAN_CACHE
    design = _load_json(DEFAULT_DESIGN)
    manifest = _load_json(DEFAULT_PLAN_MANIFEST)
    expected_manifest_keys = {
        "schema_version",
        "status",
        "mode",
        "model_calls",
        "generation_used",
        "biological_model_calls",
        "design_path",
        "design_file_sha256",
        "call_plan_sha256",
        "plan",
    }
    if set(manifest) != expected_manifest_keys:
        raise CausalBindingRunnerError("plan manifest fields changed")
    if manifest.get("schema_version") != PLAN_MANIFEST_SCHEMA:
        raise CausalBindingRunnerError("plan manifest schema changed")
    if manifest.get("status") != "PLAN_AND_DESIGN_FROZEN_NO_FORWARD":
        raise CausalBindingRunnerError("plan manifest is not zero-forward")
    if manifest.get("model_calls") != 0:
        raise CausalBindingRunnerError("plan manifest claims model execution")
    if manifest.get("mode") != "development":
        raise CausalBindingRunnerError("plan manifest mode changed")
    if manifest.get("generation_used") is not False:
        raise CausalBindingRunnerError("plan manifest claims generation")
    if manifest.get("biological_model_calls") != 0:
        raise CausalBindingRunnerError("plan manifest claims biological execution")
    if manifest.get("design_path") != str(DEFAULT_DESIGN):
        raise CausalBindingRunnerError("plan manifest design path changed")
    if manifest.get("design_file_sha256") != file_sha256(DEFAULT_DESIGN):
        raise CausalBindingRunnerError("plan manifest design hash changed")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise CausalBindingRunnerError("plan manifest does not embed a plan")
    if manifest.get("call_plan_sha256") != plan.get("call_plan_sha256"):
        raise CausalBindingRunnerError("plan manifest call-plan hash changed")
    if _RECONSTRUCTED_PLAN_CACHE is None:
        tokenizer, vocab_size = _config_and_tokenizer()
        if vocab_size != MODEL_VOCAB_SIZE:
            raise CausalBindingRunnerError("configured vocabulary changed")
        rebuilt_plan = build_plan(tokenizer, DEFAULT_ANALYZER)
        rebuilt_design = design_from_plan(rebuilt_plan)
        _RECONSTRUCTED_PLAN_CACHE = (rebuilt_plan, rebuilt_design)
    rebuilt_plan, rebuilt_design = _RECONSTRUCTED_PLAN_CACHE
    if plan != rebuilt_plan:
        raise CausalBindingRunnerError(
            "disk call plan does not reconstruct from frozen inputs"
        )
    if design != rebuilt_design:
        raise CausalBindingRunnerError(
            "disk design does not reconstruct from frozen inputs"
        )
    validate_plan(plan, design)
    return plan, design


def _load_activation_sidecar(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise CausalBindingRunnerError(f"activation sidecar does not exist: {path}")
    value = np.load(path, allow_pickle=False)
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise CausalBindingRunnerError("activation sidecar shape or values changed")
    return array


def _load_model() -> Any:
    _verify_cached_model_assets()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise CausalBindingRunnerError(
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


def _recompute_layer_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        if __package__:
            from . import analyze_coherent_readout_v2_causal_binding as analyzer
        else:
            import analyze_coherent_readout_v2_causal_binding as analyzer
    except ImportError as error:
        raise CausalBindingRunnerError("cannot import the frozen analyzer") from error
    analysis, expected_lock = analyzer.analyze_discovery()
    observed_lock = _load_json(DEFAULT_LAYER_LOCK)
    if observed_lock != expected_lock:
        raise CausalBindingRunnerError("layer-lock artifact does not recompute")
    if analysis.get("call_plan_sha256") != expected_lock.get("call_plan_sha256"):
        raise CausalBindingRunnerError("discovery analysis changed call-plan authority")
    return analysis, expected_lock


def _recompute_holdout_entry() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        if __package__:
            from . import analyze_coherent_readout_v2_causal_binding as analyzer
        else:
            import analyze_coherent_readout_v2_causal_binding as analyzer
    except ImportError as error:
        raise CausalBindingRunnerError("cannot import the frozen analyzer") from error
    analysis, expected_entry = analyzer.analyze_holdout_baseline()
    observed_entry = _load_json(DEFAULT_HOLDOUT_ENTRY)
    if observed_entry != expected_entry:
        raise CausalBindingRunnerError("holdout-entry artifact does not recompute")
    if analysis.get("call_plan_sha256") != expected_entry.get("call_plan_sha256"):
        raise CausalBindingRunnerError("holdout analysis changed call-plan authority")
    return analysis, expected_entry


def run_plan(analyzer_path: Path) -> None:
    tokenizer, vocab_size = _config_and_tokenizer()
    if vocab_size != MODEL_VOCAB_SIZE:
        raise CausalBindingRunnerError("configured vocabulary changed")
    plan = build_plan(tokenizer, analyzer_path)
    design = design_from_plan(plan)
    write_json(DEFAULT_DESIGN, design)
    manifest = {
        "schema_version": PLAN_MANIFEST_SCHEMA,
        "status": "PLAN_AND_DESIGN_FROZEN_NO_FORWARD",
        "mode": "development",
        "model_calls": 0,
        "generation_used": False,
        "biological_model_calls": 0,
        "design_path": str(DEFAULT_DESIGN),
        "design_file_sha256": file_sha256(DEFAULT_DESIGN),
        "call_plan_sha256": plan["call_plan_sha256"],
        "plan": plan,
    }
    write_json(DEFAULT_PLAN_MANIFEST, manifest)


def run_discovery() -> None:
    plan, design = _load_frozen_plan()
    baseline_path = RESULT_ROOT / "discovery_baselines.jsonl"
    activation_path = RESULT_ROOT / "discovery_activations.npy"
    patch_path = RESULT_ROOT / "discovery_patches.jsonl"
    manifest_path = RESULT_ROOT / "discovery_execution_manifest.json"
    attempt_path = RESULT_ROOT / "discovery_attempt.json"
    _require_absent(
        (
            attempt_path,
            baseline_path,
            activation_path,
            patch_path,
            manifest_path,
            RESULT_ROOT / "discovery_analysis.json",
            DEFAULT_LAYER_LOCK,
        ),
        "discovery",
    )
    _write_execution_attempt(attempt_path, phase="discovery", plan=plan)
    model = _load_model()
    prompts = _phase_prompts(plan, "discovery")
    templates = _phase_templates(plan, "discovery")
    baselines, activations = _execute_baselines(
        model, prompts, phase="discovery", duplicate=True
    )
    patches = _execute_patches(
        model,
        templates,
        prompts,
        baselines,
        activations,
        phase="discovery",
        selected_layer=None,
    )
    write_jsonl(baseline_path, baselines)
    write_array(activation_path, activations)
    write_jsonl(patch_path, patches)
    write_json(
        manifest_path,
        _execution_manifest(
            phase="discovery",
            plan=plan,
            design=design,
            model_calls=2 * len(baselines) + len(patches),
            records_path=baseline_path,
            records_count=len(baselines),
            activations_path=activation_path,
            activations=activations,
            patch_records_path=patch_path,
            patch_records_count=len(patches),
            attempt_path=attempt_path,
        ),
    )


def run_holdout_baseline() -> None:
    plan, design = _load_frozen_plan()
    _, layer_lock = _recompute_layer_authorization()
    selected_layer = layer_lock.get("selected_layer")
    if layer_lock.get("holdout_baseline_authorized") is not True:
        raise CausalBindingRunnerError("layer lock does not authorize holdout baseline")
    if (
        isinstance(selected_layer, bool)
        or not isinstance(selected_layer, int)
        or selected_layer not in design["selection_rule"]["selectable_layers"]
    ):
        raise CausalBindingRunnerError("layer lock selected an invalid layer")
    baseline_path = RESULT_ROOT / "holdout_baselines.jsonl"
    activation_path = RESULT_ROOT / "holdout_activations.npy"
    manifest_path = RESULT_ROOT / "holdout_baseline_execution_manifest.json"
    attempt_path = RESULT_ROOT / "holdout_baseline_attempt.json"
    _require_absent(
        (
            attempt_path,
            baseline_path,
            activation_path,
            manifest_path,
            RESULT_ROOT / "holdout_baseline_analysis.json",
            DEFAULT_HOLDOUT_ENTRY,
        ),
        "holdout-baseline",
    )
    _write_execution_attempt(attempt_path, phase="holdout_baseline", plan=plan)
    model = _load_model()
    prompts = _phase_prompts(plan, "holdout")
    baselines, activations = _execute_baselines(
        model, prompts, phase="holdout_baseline", duplicate=False
    )
    write_jsonl(baseline_path, baselines)
    write_array(activation_path, activations)
    write_json(
        manifest_path,
        _execution_manifest(
            phase="holdout_baseline",
            plan=plan,
            design=design,
            model_calls=len(baselines),
            records_path=baseline_path,
            records_count=len(baselines),
            activations_path=activation_path,
            activations=activations,
            selected_layer=selected_layer,
            attempt_path=attempt_path,
        ),
    )


def run_holdout_patch() -> None:
    plan, design = _load_frozen_plan()
    _, layer_lock = _recompute_layer_authorization()
    _, entry = _recompute_holdout_entry()
    selected_layer = layer_lock.get("selected_layer")
    if entry.get("holdout_patch_authorized") is not True:
        raise CausalBindingRunnerError("holdout admission gate did not authorize patching")
    if (
        entry.get("selected_layer") != selected_layer
        or isinstance(selected_layer, bool)
        or not isinstance(selected_layer, int)
        or selected_layer not in design["selection_rule"]["selectable_layers"]
    ):
        raise CausalBindingRunnerError("holdout entry and layer lock disagree")
    patch_path = RESULT_ROOT / "holdout_patches.jsonl"
    manifest_path = RESULT_ROOT / "holdout_patch_execution_manifest.json"
    attempt_path = RESULT_ROOT / "holdout_patch_attempt.json"
    _require_absent(
        (
            attempt_path,
            patch_path,
            manifest_path,
            RESULT_ROOT / "analysis.json",
            RESULT_ROOT / "analysis_manifest.json",
            RESULT_ROOT / "analysis.md",
        ),
        "holdout-patch",
    )
    baseline_path = RESULT_ROOT / "holdout_baselines.jsonl"
    activation_path = RESULT_ROOT / "holdout_activations.npy"
    baseline_manifest = _load_json(
        RESULT_ROOT / "holdout_baseline_execution_manifest.json"
    )
    if baseline_manifest["records"]["file_sha256"] != file_sha256(baseline_path):
        raise CausalBindingRunnerError("holdout baseline records changed")
    if baseline_manifest["activations"]["file_sha256"] != file_sha256(
        activation_path
    ):
        raise CausalBindingRunnerError("holdout activations changed")
    baselines = load_jsonl(baseline_path)
    activations = _load_activation_sidecar(
        activation_path, (192, MODEL_LAYERS, MODEL_WIDTH)
    )
    if f32_sha256(activations) != baseline_manifest["activations"]["logical_sha256"]:
        raise CausalBindingRunnerError("holdout activation logical hash changed")
    _write_execution_attempt(attempt_path, phase="holdout_patch", plan=plan)
    model = _load_model()
    prompts = _phase_prompts(plan, "holdout")
    templates = _phase_templates(plan, "holdout")
    patches = _execute_patches(
        model,
        templates,
        prompts,
        baselines,
        activations,
        phase="holdout_patch",
        selected_layer=selected_layer,
    )
    write_jsonl(patch_path, patches)
    write_json(
        manifest_path,
        _execution_manifest(
            phase="holdout_patch",
            plan=plan,
            design=design,
            model_calls=len(patches),
            records_path=baseline_path,
            records_count=len(baselines),
            patch_records_path=patch_path,
            patch_records_count=len(patches),
            selected_layer=selected_layer,
            attempt_path=attempt_path,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("plan", "discovery", "holdout-baseline", "holdout-patch"),
        required=True,
    )
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    args = parser.parse_args()
    try:
        if args.phase == "plan":
            run_plan(args.analyzer)
        elif args.phase == "discovery":
            run_discovery()
        elif args.phase == "holdout-baseline":
            run_holdout_baseline()
        else:
            run_holdout_patch()
    except (CausalBindingRunnerError, OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

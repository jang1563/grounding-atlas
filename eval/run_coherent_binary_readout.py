"""Run the coherent two-token readout on local open-weight models.

This is a development-only measurement runner for the Level-0 contract in
``docs/COHERENT_BINARY_READOUT_DESIGN.md``.  It expands every registered input
into the exact order-by-opaque-mapping 2x2, obtains both opaque-token logits
from one next-token distribution per form, and writes raw records plus an
execution manifest.  It does not analyze biology, adjudicate Level 0, generate
natural-language probabilities, or execute a confirmatory design.

The call-plan builder is deliberately independent of model execution.  It can
be used to freeze the expected record-ID registry before any forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import coherent_binary_readout as coherent
    from .causal_intervention import execution_input_sha256
except ImportError:  # direct execution from eval/
    import coherent_binary_readout as coherent
    from causal_intervention import execution_input_sha256


INPUT_SCHEMA = "coherent-readout-development-fixture-v1"
PLAN_SCHEMA = "coherent-binary-readout-call-plan-v1"
RUN_MANIFEST_SCHEMA = "coherent-binary-readout-development-run-v1"
PLAN_ONLY_STATUS = "DEVELOPMENT_CALL_PLAN_ONLY_NO_MODEL_FORWARD"
EXECUTED_STATUS = "DEVELOPMENT_RAW_MEASUREMENT_COMPLETE_NOT_ANALYZED"

REQUIRED_ITEM_FIELDS = {
    "item_id",
    "donor_id",
    "source_entity_id",
    "fixture_record_id",
    "input_family",
    "readout_id",
    "positive_class",
    "negative_class",
    "gene_sentence",
    "gene_sentence_sha256",
}

LOGITS_SOURCE = "raw_model_output_before_processors"


class CoherentReadoutRunnerError(ValueError):
    """Raised before execution when the development runner contract is invalid."""


def candidate_margin_lock_sha256() -> str:
    """Hash the unqualified design-default margin candidate used in development."""

    return coherent.canonical_sha256(
        {
            "schema_version": "coherent-binary-margin-candidate-v1",
            "status": "candidate_unqualified",
            "equivalence_margin": 0.06,
            "item_range_margin": 0.20,
            "strong_score_threshold": 0.20,
            "format_overall_min": 0.95,
            "format_per_group_min": 0.95,
            "format_per_donor_min": 0.90,
            "item_range_pass_min": 0.95,
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoherentReadoutRunnerError(f"{label} must be a nonempty string")
    return value


def _sha256_string(value: Any, label: str) -> str:
    output = _nonempty_string(value, label)
    if len(output) != 64 or any(character not in "0123456789abcdef" for character in output):
        raise CoherentReadoutRunnerError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return output


def _string_registry(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise CoherentReadoutRunnerError(f"{label} must be a nonempty list")
    output = [_nonempty_string(value, f"{label} item") for value in values]
    if len(output) != len(set(output)):
        raise CoherentReadoutRunnerError(f"{label} must contain unique values")
    return sorted(output)


def _as_int_vector(value: Any, label: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    raw = list(value)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise CoherentReadoutRunnerError(f"{label} must contain one batch row")
        raw = list(raw[0])
    if not raw:
        raise CoherentReadoutRunnerError(f"{label} cannot be empty")
    if any(
        isinstance(token, bool) or not isinstance(token, (int, np.integer))
        for token in raw
    ):
        raise CoherentReadoutRunnerError(f"{label} must contain integer tokens")
    return [int(token) for token in raw]


def _effective_chat_template(tokenizer: Any) -> str:
    template = None
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        try:
            template = getter()
        except TypeError as error:
            raise CoherentReadoutRunnerError(
                "tokenizer chat template requires an unregistered tool/template choice"
            ) from error
    if template is None:
        template = getattr(tokenizer, "chat_template", None)
    return _nonempty_string(template, "effective chat template")


def chat_template_sha256(tokenizer: Any) -> str:
    """Hash the exact effective Jinja chat-template text."""

    return coherent.text_sha256(_effective_chat_template(tokenizer))


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        value = len(tokenizer)
    except (TypeError, AttributeError):
        value = getattr(tokenizer, "vocab_size", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 2
    ):
        raise CoherentReadoutRunnerError(
            "tokenizer must expose a vocabulary size of at least two"
        )
    return int(value)


def _validated_vocab_size(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 2
    ):
        raise CoherentReadoutRunnerError(f"{label} must be an integer of at least two")
    return int(value)


def validate_input_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the development fixture without consuming provenance as model input."""

    if not isinstance(value, Mapping):
        raise CoherentReadoutRunnerError("input manifest must be an object")
    if value.get("schema_version") != INPUT_SCHEMA:
        raise CoherentReadoutRunnerError(
            f"input schema must be {INPUT_SCHEMA!r}"
        )
    if value.get("mode") != "development":
        raise CoherentReadoutRunnerError("runner accepts development fixtures only")
    _nonempty_string(value.get("analysis_id"), "input analysis_id")
    donor_ids = _string_registry(value.get("donor_ids"), "donor_ids")
    input_families = _string_registry(
        value.get("input_families"), "input_families"
    )
    readouts = value.get("readouts")
    if not isinstance(readouts, Mapping) or not readouts:
        raise CoherentReadoutRunnerError("readouts must be a nonempty object")
    normalized_readouts: dict[str, dict[str, str]] = {}
    for raw_name, raw_spec in readouts.items():
        name = _nonempty_string(raw_name, "readout ID")
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != {
            "positive_class",
            "negative_class",
        }:
            raise CoherentReadoutRunnerError(
                f"readout {name!r} must contain exactly positive_class/negative_class"
            )
        positive = _nonempty_string(
            raw_spec["positive_class"], f"{name} positive_class"
        )
        negative = _nonempty_string(
            raw_spec["negative_class"], f"{name} negative_class"
        )
        if positive == negative:
            raise CoherentReadoutRunnerError(
                f"readout {name!r} has identical classes"
            )
        normalized_readouts[name] = {
            "positive_class": positive,
            "negative_class": negative,
        }
    if list(readouts) != sorted(readouts):
        raise CoherentReadoutRunnerError("readout IDs must be sorted")

    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise CoherentReadoutRunnerError("items must be a nonempty list")
    normalized_items: list[dict[str, str]] = []
    observed_keys: set[tuple[str, str, str]] = set()
    item_donors: dict[str, str] = {}
    source_donors: dict[str, str] = {}
    item_by_source_family: dict[tuple[str, str], set[str]] = {}
    gene_hash_by_source_family: dict[tuple[str, str], set[str]] = {}
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, Mapping):
            raise CoherentReadoutRunnerError(f"item {index} must be an object")
        missing = sorted(REQUIRED_ITEM_FIELDS - set(raw_item))
        if missing:
            raise CoherentReadoutRunnerError(
                f"item {index} is missing runner fields: {missing}"
            )
        source = {
            field: _nonempty_string(raw_item[field], f"item {index} {field}")
            for field in REQUIRED_ITEM_FIELDS
        }
        item = {
            "item_id": source["item_id"],
            "donor_id": source["donor_id"],
            # The source entity is family-independent; item_id is the concrete
            # source-by-family unit shared across readouts.
            "source_item_id": source["source_entity_id"],
            "source_fixture_record_id": source["fixture_record_id"],
            "input_family": source["input_family"],
            "readout_id": source["readout_id"],
            "positive_class": source["positive_class"],
            "negative_class": source["negative_class"],
            "gene_sentence": source["gene_sentence"],
            "gene_sentence_sha256": source["gene_sentence_sha256"],
        }
        _sha256_string(
            item["source_fixture_record_id"],
            f"item {index} fixture_record_id",
        )
        _sha256_string(
            item["gene_sentence_sha256"],
            f"item {index} gene_sentence_sha256",
        )
        if item["donor_id"] not in donor_ids:
            raise CoherentReadoutRunnerError(
                f"item {index} donor is outside the donor registry"
            )
        if item["input_family"] not in input_families:
            raise CoherentReadoutRunnerError(
                f"item {index} family is outside the family registry"
            )
        readout = normalized_readouts.get(item["readout_id"])
        if readout is None:
            raise CoherentReadoutRunnerError(
                f"item {index} readout is outside the readout registry"
            )
        if (
            item["positive_class"] != readout["positive_class"]
            or item["negative_class"] != readout["negative_class"]
        ):
            raise CoherentReadoutRunnerError(
                f"item {index} class strings differ from its readout registry"
            )
        prior_donor = item_donors.setdefault(item["item_id"], item["donor_id"])
        if prior_donor != item["donor_id"]:
            raise CoherentReadoutRunnerError(
                f"item_id {item['item_id']!r} occurs under multiple donors"
            )
        source_prior = source_donors.setdefault(
            item["source_item_id"], item["donor_id"]
        )
        if source_prior != item["donor_id"]:
            raise CoherentReadoutRunnerError(
                f"source item {item['source_item_id']!r} occurs under multiple donors"
            )
        key = (item["item_id"], item["readout_id"], item["input_family"])
        if key in observed_keys:
            raise CoherentReadoutRunnerError(f"duplicate input item key: {key}")
        observed_keys.add(key)
        if item["gene_sentence_sha256"] != coherent.text_sha256(
            item["gene_sentence"]
        ):
            raise CoherentReadoutRunnerError(
                f"item {index} gene_sentence_sha256 is invalid"
            )
        if raw_item.get("confirmatory_eligibility", "prohibited") != "prohibited":
            raise CoherentReadoutRunnerError(
                f"item {index} is not firewalled from confirmation"
            )
        source_family = (item["source_item_id"], item["input_family"])
        item_by_source_family.setdefault(source_family, set()).add(item["item_id"])
        gene_hash_by_source_family.setdefault(source_family, set()).add(
            item["gene_sentence_sha256"]
        )
        normalized_items.append(item)

    if any(len(values) != 1 for values in item_by_source_family.values()):
        raise CoherentReadoutRunnerError(
            "readouts must share one concrete item per source/input family"
        )
    if any(len(values) != 1 for values in gene_hash_by_source_family.values()):
        raise CoherentReadoutRunnerError(
            "readouts must share one gene sentence per source/input family"
        )

    expected_cells = {
        (donor, readout, family)
        for donor in donor_ids
        for readout in normalized_readouts
        for family in input_families
    }
    observed_cells = {
        (item["donor_id"], item["readout_id"], item["input_family"])
        for item in normalized_items
    }
    if observed_cells != expected_cells:
        raise CoherentReadoutRunnerError(
            "fixture donor-by-readout-by-family coverage is incomplete: "
            f"missing={sorted(expected_cells-observed_cells)}, "
            f"extra={sorted(observed_cells-expected_cells)}"
        )
    firewall = value.get("firewall")
    if isinstance(firewall, Mapping) and firewall.get(
        "confirmatory_eligibility", "prohibited"
    ) != "prohibited":
        raise CoherentReadoutRunnerError(
            "input firewall permits confirmatory use"
        )
    return {
        "schema_version": INPUT_SCHEMA,
        "analysis_id": value["analysis_id"],
        "mode": "development",
        "donor_ids": donor_ids,
        "input_families": input_families,
        "readouts": dict(sorted(normalized_readouts.items())),
        "items": sorted(
            normalized_items,
            key=lambda item: (
                item["donor_id"],
                item["item_id"],
                item["readout_id"],
                item["input_family"],
            ),
        ),
        "input_canonical_sha256": coherent.canonical_sha256(value),
        "source_items": sorted(
            [
                {"source_item_id": source_item, "donor_id": donor}
                for source_item, donor in source_donors.items()
            ],
            key=lambda record: record["source_item_id"],
        ),
    }


def _chat_render(tokenizer: Any, user_prompt: str) -> tuple[str, list[int], list[int]]:
    messages = [{"role": "user", "content": user_prompt}]
    kwargs = {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise CoherentReadoutRunnerError(
            "apply_chat_template(tokenize=False) did not return text"
        )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        **kwargs,
    )
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise CoherentReadoutRunnerError(
                "tokenized chat template omitted input_ids"
            )
        input_ids = _as_int_vector(encoded["input_ids"], "input_ids")
        attention_mask = _as_int_vector(
            encoded.get("attention_mask", [1] * len(input_ids)),
            "attention_mask",
        )
    else:
        input_ids = _as_int_vector(encoded, "input_ids")
        attention_mask = [1] * len(input_ids)
    if len(attention_mask) != len(input_ids) or any(
        value != 1 for value in attention_mask
    ):
        raise CoherentReadoutRunnerError(
            "each development call must be one unpadded all-attended sequence"
        )
    retokenized = _as_int_vector(
        tokenizer.encode(rendered, add_special_tokens=False),
        "retokenized rendered chat",
    )
    if input_ids != retokenized:
        raise CoherentReadoutRunnerError(
            "tokenize=False rendering does not reconstruct tokenize=True input IDs"
        )
    return rendered, input_ids, attention_mask


def build_call_plan(
    input_manifest: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_vocab_size: int,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    dtype: str,
    source_fixture_sha256: str,
    source_manifest_sha256: str,
    preregistration_sha256: str,
    margin_lock_sha256: str,
    margin_lock_status: str = "candidate_unqualified",
    x_label: str = "X",
    y_label: str = "Y",
) -> dict[str, Any]:
    """Expand and tokenize the exact four-form development call registry."""

    inputs = validate_input_manifest(input_manifest)
    model_id = _nonempty_string(model_id, "model_id")
    model_revision = _nonempty_string(model_revision, "model_revision")
    tokenizer_id = _nonempty_string(tokenizer_id, "tokenizer_id")
    tokenizer_revision = _nonempty_string(
        tokenizer_revision, "tokenizer_revision"
    )
    dtype = _nonempty_string(dtype, "dtype")
    source_fixture_sha256 = _sha256_string(
        source_fixture_sha256, "source_fixture_sha256"
    )
    source_manifest_sha256 = _sha256_string(
        source_manifest_sha256, "source_manifest_sha256"
    )
    preregistration_sha256 = _sha256_string(
        preregistration_sha256, "preregistration_sha256"
    )
    margin_lock_sha256 = _sha256_string(
        margin_lock_sha256, "margin_lock_sha256"
    )
    if margin_lock_status not in {"candidate_unqualified", "phase0_qualified"}:
        raise CoherentReadoutRunnerError("invalid margin_lock_status")
    template_sha256 = chat_template_sha256(tokenizer)
    tokenizer_vocab_size = _tokenizer_vocab_size(tokenizer)
    vocab_size = _validated_vocab_size(model_vocab_size, "model_vocab_size")
    planned: list[dict[str, Any]] = []
    x_token_ids: set[int] = set()
    y_token_ids: set[int] = set()
    for item in inputs["items"]:
        for order, mapping in coherent.FORM_KEYS:
            user_prompt = coherent.render_binary_prompt(
                gene_sentence=item["gene_sentence"],
                positive_class=item["positive_class"],
                negative_class=item["negative_class"],
                order=order,
                mapping=mapping,
                x_label=x_label,
                y_label=y_label,
            )
            rendered_chat, input_ids, attention_mask = _chat_render(
                tokenizer,
                user_prompt,
            )
            x_token_id = coherent.contextual_single_token_id(
                tokenizer, rendered_chat, x_label
            )
            y_token_id = coherent.contextual_single_token_id(
                tokenizer, rendered_chat, y_label
            )
            if x_token_id == y_token_id:
                raise CoherentReadoutRunnerError(
                    "opaque labels resolve to the same answer token"
                )
            if max(x_token_id, y_token_id) >= min(
                tokenizer_vocab_size, vocab_size
            ):
                raise CoherentReadoutRunnerError(
                    "opaque answer-token ID lies outside the tokenizer or model vocabulary"
                )
            x_token_ids.add(x_token_id)
            y_token_ids.add(y_token_id)
            positive_token_id, negative_token_id = (
                (x_token_id, y_token_id)
                if mapping == "positive_is_x"
                else (y_token_id, x_token_id)
            )
            identity = {
                "schema_version": coherent.RECORD_SCHEMA,
                "donor_id": item["donor_id"],
                "source_item_id": item["source_item_id"],
                "item_id": item["item_id"],
                "readout_id": item["readout_id"],
                "input_family": item["input_family"],
                "source_fixture_record_id": item["source_fixture_record_id"],
                "gene_sentence_sha256": item["gene_sentence_sha256"],
                "positive_class": item["positive_class"],
                "negative_class": item["negative_class"],
                "order": order,
                "mapping": mapping,
                "x_token_id": x_token_id,
                "y_token_id": y_token_id,
                "positive_token_id": positive_token_id,
                "negative_token_id": negative_token_id,
                "user_prompt_sha256": coherent.text_sha256(user_prompt),
                "prompt_sha256": coherent.text_sha256(rendered_chat),
                "execution_input_sha256": execution_input_sha256(
                    input_ids,
                    attention_mask,
                ),
                "input_token_count": len(input_ids),
                "model_id": model_id,
                "model_revision": model_revision,
                "tokenizer_id": tokenizer_id,
                "tokenizer_revision": tokenizer_revision,
                "chat_template_sha256": template_sha256,
                "dtype": dtype,
                "logits_source": LOGITS_SOURCE,
                "vocab_size": vocab_size,
                "full_vocab_logits_row": len(planned),
            }
            planned.append(
                {
                    "record_id": coherent.record_id(identity),
                    **identity,
                    "planned_index": len(planned),
                    "user_prompt": user_prompt,
                    "rendered_chat": rendered_chat,
                    "execution_input_ids": input_ids,
                    "execution_attention_mask": attention_mask,
                }
            )
    if len(x_token_ids) != 1 or len(y_token_ids) != 1:
        raise CoherentReadoutRunnerError(
            "opaque answer-token IDs are not constant over the complete call plan"
        )
    if len({record["record_id"] for record in planned}) != len(planned):
        raise CoherentReadoutRunnerError("call plan contains duplicate record IDs")
    registry = planned
    runner_code_sha256 = _file_sha256(Path(__file__))
    call_plan_sha256 = coherent.call_plan_sha256(registry)
    return {
        "schema_version": PLAN_SCHEMA,
        "mode": "development",
        "input_analysis_id": inputs["analysis_id"],
        "input_canonical_sha256": inputs["input_canonical_sha256"],
        "source_fixture_sha256": source_fixture_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "preregistration_sha256": preregistration_sha256,
        "runner_code_sha256": runner_code_sha256,
        "call_plan_sha256": call_plan_sha256,
        "margin_lock_sha256": margin_lock_sha256,
        "margin_lock_status": margin_lock_status,
        "donor_ids": inputs["donor_ids"],
        "readouts": inputs["readouts"],
        "input_families": inputs["input_families"],
        "input_rows": len(inputs["items"]),
        "source_items": inputs["source_items"],
        "form_order": [list(form) for form in coherent.FORM_KEYS],
        "record_count": len(registry),
        "expected_record_ids": sorted(record["record_id"] for record in registry),
        "x_label": x_label,
        "y_label": y_label,
        "x_token_id": next(iter(x_token_ids)),
        "y_token_id": next(iter(y_token_ids)),
        "vocab_size": vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": template_sha256,
        "dtype": dtype,
        "records": registry,
    }


def validate_plan_against_design(
    plan: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    """Require every pre-forward registry lock to equal the frozen design."""

    locked = coherent.validate_design(design)
    if locked["mode"] != "development":
        raise CoherentReadoutRunnerError(
            "this runner refuses confirmatory designs and model execution"
        )
    comparisons = {
        "required_readouts": sorted(plan["readouts"]),
        "readout_classes": plan["readouts"],
        "required_input_families": sorted(plan["input_families"]),
        "source_fixture_sha256": plan["source_fixture_sha256"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "preregistration_sha256": plan["preregistration_sha256"],
        "runner_code_sha256": plan["runner_code_sha256"],
        "call_plan_sha256": plan["call_plan_sha256"],
        "margin_lock_sha256": plan["margin_lock_sha256"],
        "margin_lock_status": plan["margin_lock_status"],
        "expected_model_id": plan["model_id"],
        "expected_model_revision": plan["model_revision"],
        "expected_tokenizer_id": plan["tokenizer_id"],
        "expected_tokenizer_revision": plan["tokenizer_revision"],
        "expected_chat_template_sha256": plan["chat_template_sha256"],
        "expected_dtype": plan["dtype"],
        "expected_x_token_id": plan["x_token_id"],
        "expected_y_token_id": plan["y_token_id"],
        "expected_vocab_size": plan["vocab_size"],
        "expected_donor_ids": sorted(plan["donor_ids"]),
        "expected_source_items": plan["source_items"],
        "expected_record_ids": sorted(plan["expected_record_ids"]),
    }
    mismatches = [
        field for field, observed in comparisons.items() if locked[field] != observed
    ]
    if mismatches:
        raise CoherentReadoutRunnerError(
            f"call plan differs from frozen design: {mismatches}"
        )
    return locked


def design_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the exact frozen development design for a no-forward plan."""

    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("mode") != "development":
        raise CoherentReadoutRunnerError("invalid development call plan")
    design = coherent.default_design(
        mode="development",
        required_readouts=sorted(plan["readouts"]),
        readout_classes=plan["readouts"],
        required_input_families=sorted(plan["input_families"]),
        source_fixture_sha256=plan["source_fixture_sha256"],
        source_manifest_sha256=plan["source_manifest_sha256"],
        preregistration_sha256=plan["preregistration_sha256"],
        runner_code_sha256=plan["runner_code_sha256"],
        call_plan_sha256=plan["call_plan_sha256"],
        margin_lock_sha256=plan["margin_lock_sha256"],
        margin_lock_status=plan["margin_lock_status"],
        model_id=plan["model_id"],
        model_revision=plan["model_revision"],
        tokenizer_id=plan["tokenizer_id"],
        tokenizer_revision=plan["tokenizer_revision"],
        chat_template_sha256=plan["chat_template_sha256"],
        dtype=plan["dtype"],
        x_token_id=plan["x_token_id"],
        y_token_id=plan["y_token_id"],
        vocab_size=plan["vocab_size"],
        expected_donor_ids=plan["donor_ids"],
        expected_source_items=plan["source_items"],
        expected_record_ids=plan["expected_record_ids"],
    )
    locked = coherent.validate_design(design)
    validate_plan_against_design(plan, locked)
    return locked


def _torch_module_and_context(model: Any) -> tuple[bool, Any]:
    try:
        import torch
    except ImportError:
        return False, nullcontext()
    is_torch_module = isinstance(model, torch.nn.Module)
    return is_torch_module, torch.inference_mode() if is_torch_module else nullcontext()


def _model_device(model: Any) -> Any:
    embedding_getter = getattr(model, "get_input_embeddings", None)
    if callable(embedding_getter):
        embeddings = embedding_getter()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and hasattr(weight, "device"):
            return weight.device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def _forward_next_logits(model: Any, plan_record: Mapping[str, Any]) -> np.ndarray:
    input_ids = [list(plan_record["execution_input_ids"])]
    attention_mask = [list(plan_record["execution_attention_mask"])]
    is_torch_module, context = _torch_module_and_context(model)
    if is_torch_module:
        import torch

        device = _model_device(model)
        input_ids_value = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_value = torch.tensor(
            attention_mask,
            dtype=torch.long,
            device=device,
        )
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
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(
        output, "logits", None
    )
    if logits is None:
        raise CoherentReadoutRunnerError("model forward omitted logits")
    shape = getattr(logits, "shape", None)
    if shape is None or len(shape) != 3 or shape[0] != 1:
        raise CoherentReadoutRunnerError(
            f"model logits must have shape (1, sequence, vocabulary), got {shape}"
        )
    if shape[1] != plan_record["input_token_count"]:
        raise CoherentReadoutRunnerError(
            "model logit sequence length differs from the executed input"
        )
    next_logits = logits[0, -1, :]
    if hasattr(next_logits, "detach"):
        next_logits = next_logits.detach().float().cpu().numpy()
    try:
        return coherent.canonical_full_vocab_row(next_logits)
    except coherent.CoherentReadoutError as error:
        raise CoherentReadoutRunnerError(
            "next-token distribution must be a finite vocabulary vector"
        ) from error


def execute_call_plan(
    model: Any, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Execute exactly one full-vocabulary forward for every frozen plan record."""

    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("mode") != "development":
        raise CoherentReadoutRunnerError("invalid development call plan")
    eval_method = getattr(model, "eval", None)
    if not callable(eval_method):
        raise CoherentReadoutRunnerError("model must expose eval()")
    eval_method()
    output: list[dict[str, Any]] = []
    full_vocab_rows: list[np.ndarray] = []
    for planned_index, planned in enumerate(plan["records"]):
        if planned["planned_index"] != planned_index or planned[
            "full_vocab_logits_row"
        ] != planned_index:
            raise CoherentReadoutRunnerError(
                "call-plan and full-vocabulary sidecar row order differ"
            )
        next_logits = _forward_next_logits(model, planned)
        x_token_id = int(planned["x_token_id"])
        y_token_id = int(planned["y_token_id"])
        if max(x_token_id, y_token_id) >= len(next_logits):
            raise CoherentReadoutRunnerError(
                "opaque answer-token ID is outside the model vocabulary"
            )
        if len(next_logits) != planned["vocab_size"]:
            raise CoherentReadoutRunnerError(
                "model output vocabulary differs from the frozen config vocabulary"
            )
        diagnostics = coherent.full_vocab_diagnostics(
            next_logits,
            x_token_id=x_token_id,
            y_token_id=y_token_id,
        )
        identity_keys = (
            "donor_id",
            "source_item_id",
            "item_id",
            "readout_id",
            "input_family",
            "source_fixture_record_id",
            "gene_sentence_sha256",
            "positive_class",
            "negative_class",
            "order",
            "mapping",
            "x_token_id",
            "y_token_id",
            "positive_token_id",
            "negative_token_id",
            "user_prompt_sha256",
            "prompt_sha256",
            "execution_input_sha256",
            "input_token_count",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "chat_template_sha256",
            "dtype",
            "logits_source",
            "vocab_size",
            "full_vocab_logits_row",
        )
        record = coherent.make_record(
            **{key: planned[key] for key in identity_keys},
            **diagnostics,
        )
        if record["record_id"] != planned["record_id"]:
            raise CoherentReadoutRunnerError(
                "executed record identity differs from the frozen call plan"
            )
        output.append(record)
        full_vocab_rows.append(next_logits)
    if [record["record_id"] for record in output] != [
        record["record_id"] for record in plan["records"]
    ]:
        raise CoherentReadoutRunnerError("executed record order differs from the plan")
    sidecar = np.ascontiguousarray(np.stack(full_vocab_rows), dtype="<f4")
    if coherent.call_plan_sha256(output) != plan["call_plan_sha256"]:
        raise CoherentReadoutRunnerError(
            "executed records do not reproduce the frozen call-plan hash"
        )
    return output, sidecar


def _plan_registry(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "planned_index",
        "record_id",
        "donor_id",
        "source_item_id",
        "item_id",
        "readout_id",
        "input_family",
        "source_fixture_record_id",
        "gene_sentence_sha256",
        "positive_class",
        "negative_class",
        "order",
        "mapping",
        "user_prompt_sha256",
        "prompt_sha256",
        "execution_input_sha256",
        "input_token_count",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "x_token_id",
        "y_token_id",
        "positive_token_id",
        "negative_token_id",
        "vocab_size",
        "full_vocab_logits_row",
        "logits_source",
    )
    return [{key: record[key] for key in keys} for record in plan["records"]]


def _manifest_payload(
    *,
    plan: Mapping[str, Any],
    model_id: str,
    tokenizer_id: str,
    design: Mapping[str, Any] | None,
    records_path: Path | None,
    records: Sequence[Mapping[str, Any]] | None,
    logits_path: Path | None,
    full_vocab_logits: np.ndarray | None,
    input_path: Path | None,
    input_provenance_path: Path | None,
    model: Any | None,
) -> dict[str, Any]:
    if model_id != plan["model_id"] or tokenizer_id != plan["tokenizer_id"]:
        raise CoherentReadoutRunnerError(
            "manifest repository IDs differ from the frozen call plan"
        )
    executed = records is not None
    registry = _plan_registry(plan)
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": EXECUTED_STATUS if executed else PLAN_ONLY_STATUS,
        "mode": "development",
        "claim_scope": (
            "local Level-0 readout engineering only; no biological, knowledge, "
            "activation-gap, calibration, or confirmatory claim"
        ),
        "forward_passes_executed": 0 if not executed else len(records),
        "natural_language_generation_used": False,
        "biological_analysis_run": False,
        "confirmatory_execution": False,
        "input": {
            "analysis_id": plan["input_analysis_id"],
            "path": None if input_path is None else str(input_path),
            "file_sha256": (
                None if input_path is None else _file_sha256(input_path)
            ),
            "canonical_sha256": plan["source_manifest_sha256"],
            "fixture_body_canonical_sha256": plan["input_canonical_sha256"],
            "provenance_manifest_path": (
                None
                if input_provenance_path is None
                else str(input_provenance_path)
            ),
            "provenance_manifest_sha256": (
                None
                if input_provenance_path is None
                else _file_sha256(input_provenance_path)
            ),
        },
        "design": {
            "frozen": design is not None,
            "canonical_sha256": (
                None if design is None else coherent.canonical_sha256(design)
            ),
            "expected_record_ids_match": (
                None
                if design is None
                else sorted(design["expected_record_ids"])
                == plan["expected_record_ids"]
            ),
            "source_fixture_sha256": plan["source_fixture_sha256"],
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "preregistration_sha256": plan["preregistration_sha256"],
            "runner_code_sha256": plan["runner_code_sha256"],
            "call_plan_sha256": plan["call_plan_sha256"],
            "margin_lock_sha256": plan["margin_lock_sha256"],
            "margin_lock_status": plan["margin_lock_status"],
        },
        "model": {
            "model_id": plan["model_id"],
            "model_revision": plan["model_revision"],
            "class": None if model is None else type(model).__name__,
            "dtype": plan["dtype"],
            "output_vocab_size": plan["vocab_size"],
            "device": None if model is None else str(_model_device(model)),
        },
        "tokenizer": {
            "tokenizer_id": plan["tokenizer_id"],
            "tokenizer_revision": plan["tokenizer_revision"],
            "chat_template_sha256": plan["chat_template_sha256"],
            "x_label": plan["x_label"],
            "y_label": plan["y_label"],
            "x_token_id": plan["x_token_id"],
            "y_token_id": plan["y_token_id"],
            "tokenizer_vocab_size": plan["tokenizer_vocab_size"],
            "apply_chat_template_tokenize_true": True,
            "rendered_text_retokenization_identity_checked": True,
            "enable_thinking": False,
            "add_generation_prompt": True,
        },
        "coverage": {
            "donor_ids": plan["donor_ids"],
            "readouts": plan["readouts"],
            "input_families": plan["input_families"],
            "input_rows": plan["input_rows"],
            "source_items": plan["source_items"],
            "forms_per_input": len(coherent.FORM_KEYS),
            "planned_records": plan["record_count"],
        },
        "call_plan": {
            "schema_version": PLAN_SCHEMA,
            "canonical_sha256": plan["call_plan_sha256"],
            "registry_projection_sha256": coherent.canonical_sha256(registry),
            "form_order": plan["form_order"],
            "record_ids": plan["expected_record_ids"],
            "registry": registry,
        },
        "output": {
            "records_path": None if records_path is None else str(records_path),
            "records_file_sha256": (
                None
                if records_path is None or not records_path.is_file()
                else _file_sha256(records_path)
            ),
            "record_count": 0 if records is None else len(records),
            "canonical_records_sha256": (
                None
                if records is None
                else coherent.canonical_sha256(list(records))
            ),
            "full_vocab_logits_sidecar": {
                "path": None if logits_path is None else str(logits_path),
                "file_sha256": (
                    None
                    if logits_path is None or not logits_path.is_file()
                    else _file_sha256(logits_path)
                ),
                "format": "npy-v1-float32-le-c-order",
                "shape": (
                    None
                    if full_vocab_logits is None
                    else list(full_vocab_logits.shape)
                ),
                "row_record_ids": (
                    None
                    if records is None
                    else [record["record_id"] for record in records]
                ),
                "row_sha256": (
                    None
                    if records is None
                    else [record["full_vocab_logits_sha256"] for record in records]
                ),
                "matrix_sha256": (
                    None
                    if full_vocab_logits is None
                    else coherent.full_vocab_matrix_sha256(full_vocab_logits)
                ),
            },
        },
        "runtime": {
            "python_implementation": __import__("platform").python_implementation(),
            "python_version": __import__("platform").python_version(),
            "dependencies": {
                name: _package_version(name)
                for name in ("numpy", "torch", "transformers", "scipy")
            },
            "execution_code_path": str(Path(__file__)),
            "execution_code_sha256": _file_sha256(Path(__file__)),
            "coherent_analyzer_code_sha256": _file_sha256(
                Path(coherent.__file__)
            ),
        },
    }
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise CoherentReadoutRunnerError(
            f"refusing to overwrite a different artifact: {path}"
        )
    path.write_text(serialized, encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise CoherentReadoutRunnerError(
            f"refusing to overwrite a different artifact: {path}"
        )
    path.write_text(serialized, encoding="utf-8")


def _write_full_vocab_sidecar(path: Path, values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    if array.ndim != 2 or not np.isfinite(array).all():
        raise CoherentReadoutRunnerError(
            "full-vocabulary sidecar must be a finite record-by-vocabulary matrix"
        )
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    serialized = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != serialized:
        raise CoherentReadoutRunnerError(
            f"refusing to overwrite a different artifact: {path}"
        )
    path.write_bytes(serialized)


def run_development(
    *,
    input_manifest: Mapping[str, Any],
    design: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    model_id: str,
    tokenizer_id: str,
    model_vocab_size: int,
    model_revision: str,
    tokenizer_revision: str,
    dtype: str,
    source_fixture_sha256: str,
    source_manifest_sha256: str,
    preregistration_sha256: str,
    margin_lock_sha256: str,
    margin_lock_status: str,
    output_records_path: Path,
    output_logits_path: Path,
    output_manifest_path: Path,
    input_path: Path | None = None,
    input_provenance_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the frozen registry, then execute and write raw development artifacts."""

    plan = build_call_plan(
        input_manifest,
        tokenizer=tokenizer,
        model_vocab_size=model_vocab_size,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        dtype=dtype,
        source_fixture_sha256=source_fixture_sha256,
        source_manifest_sha256=source_manifest_sha256,
        preregistration_sha256=preregistration_sha256,
        margin_lock_sha256=margin_lock_sha256,
        margin_lock_status=margin_lock_status,
    )
    locked = validate_plan_against_design(plan, design)
    records, full_vocab_logits = execute_call_plan(model, plan)
    if sorted(record["record_id"] for record in records) != locked[
        "expected_record_ids"
    ]:
        raise CoherentReadoutRunnerError(
            "executed records differ from the frozen record registry"
        )
    coherent.verify_full_vocab_sidecar(records, locked, full_vocab_logits)
    _write_jsonl(output_records_path, records)
    _write_full_vocab_sidecar(output_logits_path, full_vocab_logits)
    manifest = _manifest_payload(
        plan=plan,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        design=locked,
        records_path=output_records_path,
        records=records,
        logits_path=output_logits_path,
        full_vocab_logits=full_vocab_logits,
        input_path=input_path,
        input_provenance_path=input_provenance_path,
        model=model,
    )
    _write_json(output_manifest_path, manifest)
    return records, manifest


def write_plan_manifest(
    *,
    plan: Mapping[str, Any],
    output_design_path: Path,
    output_manifest_path: Path,
    model_id: str,
    tokenizer_id: str,
    input_path: Path | None = None,
    input_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Write a no-forward plan and its exact analyzer-ready development design."""

    design = design_from_plan(plan)
    _write_json(output_design_path, design)
    manifest = _manifest_payload(
        plan=plan,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        design=design,
        records_path=None,
        records=None,
        logits_path=None,
        full_vocab_logits=None,
        input_path=input_path,
        input_provenance_path=input_provenance_path,
        model=None,
    )
    manifest["design_artifact"] = {
        "path": str(output_design_path),
        "file_sha256": _file_sha256(output_design_path),
        "canonical_sha256": coherent.canonical_sha256(design),
        "validated": True,
    }
    _write_json(output_manifest_path, manifest)
    return manifest


def _load_hf_tokenizer(
    model_id: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    *,
    local_files_only: bool,
) -> Any:
    del model_id
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise CoherentReadoutRunnerError(
            "Hugging Face execution requires the optional transformers dependency"
        ) from error
    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )


def _load_hf_config_vocab_size(
    model_id: str,
    model_revision: str,
    *,
    local_files_only: bool,
) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise CoherentReadoutRunnerError(
            "Hugging Face planning requires the optional transformers dependency"
        ) from error
    config = AutoConfig.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    return _validated_vocab_size(
        getattr(config, "vocab_size", None),
        "model config vocab_size",
    )


def _load_hf_model(
    model_id: str,
    model_revision: str,
    dtype: str,
    *,
    local_files_only: bool,
    device: str,
) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise CoherentReadoutRunnerError(
            "Hugging Face execution requires torch and transformers"
        ) from error
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise CoherentReadoutRunnerError(
            f"unsupported frozen dtype {dtype!r}; choose {sorted(dtype_map)}"
        )
    if device not in {"cpu", "mps", "cuda"}:
        raise CoherentReadoutRunnerError(
            "device must be one of: cpu, mps, cuda"
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise CoherentReadoutRunnerError("requested MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise CoherentReadoutRunnerError("requested CUDA device is unavailable")
    kwargs: dict[str, Any] = {
        "revision": model_revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "dtype": dtype_map[dtype],
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.to(torch.device(device)).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-provenance-manifest", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("docs/COHERENT_BINARY_READOUT_DESIGN.md"),
    )
    parser.add_argument("--design", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-design", type=Path)
    parser.add_argument("--output-records", type=Path)
    parser.add_argument("--output-logits", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--margin-lock-status",
        choices=("candidate_unqualified", "phase0_qualified"),
        default="candidate_unqualified",
    )
    parser.add_argument("--margin-lock-sha256")
    args = parser.parse_args()

    if not args.inputs.is_file():
        parser.error(f"inputs do not exist: {args.inputs}")
    if not args.input_provenance_manifest.is_file():
        parser.error(
            f"input provenance manifest does not exist: {args.input_provenance_manifest}"
        )
    if not args.preregistration.is_file():
        parser.error(f"preregistration does not exist: {args.preregistration}")
    if args.plan_only:
        if (
            args.design is not None
            or args.output_records is not None
            or args.output_logits is not None
        ):
            parser.error(
                "--plan-only does not accept --design, --output-records, or --output-logits"
            )
        if args.output_design is None:
            parser.error("--plan-only requires --output-design")
    elif (
        args.design is None
        or args.output_records is None
        or args.output_logits is None
    ):
        parser.error("execution requires --design, --output-records, and --output-logits")
    elif args.output_design is not None:
        parser.error("--output-design is only valid with --plan-only")
    elif args.device is None:
        parser.error("execution requires an explicit --device")
    margin_lock_sha256 = (
        candidate_margin_lock_sha256()
        if args.margin_lock_sha256 is None
        else _sha256_string(args.margin_lock_sha256, "margin_lock_sha256")
    )
    if (
        args.margin_lock_status == "phase0_qualified"
        and args.margin_lock_sha256 is None
    ):
        parser.error("phase0_qualified requires an explicit --margin-lock-sha256")

    input_manifest = json.loads(args.inputs.read_text(encoding="utf-8"))
    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer = _load_hf_tokenizer(
        args.model_id,
        tokenizer_id,
        args.tokenizer_revision,
        local_files_only=not args.allow_download,
    )
    model_vocab_size = _load_hf_config_vocab_size(
        args.model_id,
        args.model_revision,
        local_files_only=not args.allow_download,
    )
    plan = build_call_plan(
        input_manifest,
        tokenizer=tokenizer,
        model_vocab_size=model_vocab_size,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        dtype=args.dtype,
        source_fixture_sha256=_file_sha256(args.inputs),
        source_manifest_sha256=_file_sha256(args.input_provenance_manifest),
        preregistration_sha256=_file_sha256(args.preregistration),
        margin_lock_sha256=margin_lock_sha256,
        margin_lock_status=args.margin_lock_status,
    )
    if args.plan_only:
        write_plan_manifest(
            plan=plan,
            output_design_path=args.output_design,
            output_manifest_path=args.output_manifest,
            model_id=args.model_id,
            tokenizer_id=tokenizer_id,
            input_path=args.inputs,
            input_provenance_path=args.input_provenance_manifest,
        )
        print(f"Wrote development-only call plan: {args.output_manifest}")
        print(f"Wrote frozen development design: {args.output_design}")
        return

    design = json.loads(args.design.read_text(encoding="utf-8"))
    validate_plan_against_design(plan, design)
    model = _load_hf_model(
        args.model_id,
        args.model_revision,
        args.dtype,
        local_files_only=not args.allow_download,
        device=args.device,
    )
    records, full_vocab_logits = execute_call_plan(model, plan)
    locked = coherent.validate_design(design)
    if sorted(record["record_id"] for record in records) != locked[
        "expected_record_ids"
    ]:
        raise CoherentReadoutRunnerError(
            "executed records differ from the frozen record registry"
        )
    coherent.verify_full_vocab_sidecar(records, locked, full_vocab_logits)
    _write_jsonl(args.output_records, records)
    _write_full_vocab_sidecar(args.output_logits, full_vocab_logits)
    manifest = _manifest_payload(
        plan=plan,
        model_id=args.model_id,
        tokenizer_id=tokenizer_id,
        design=locked,
        records_path=args.output_records,
        records=records,
        logits_path=args.output_logits,
        full_vocab_logits=full_vocab_logits,
        input_path=args.inputs,
        input_provenance_path=args.input_provenance_manifest,
        model=model,
    )
    _write_json(args.output_manifest, manifest)
    print(f"Wrote raw development records: {args.output_records}")
    print(f"Wrote development execution manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()

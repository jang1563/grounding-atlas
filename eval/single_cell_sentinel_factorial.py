"""Run and analyze the held-out GSE96583 GNLY/NKG7/CCL5 factorial.

This arm is deliberately narrow.  It tests whether the model's CD8-versus-NK
output surface is dominated by a GNLY sentinel or retains material incremental
leverage from NKG7 and CCL5.  The eight held-out cells, nested matched-control
masks, canonical prompt form, donor-level estimands, and decision hierarchy are
frozen in ``docs/GSE96583_SENTINEL_FACTORIAL_PREREG.md``.

The raw JSONL is an append-only, resumable checkpoint.  Every record binds the
request to the frozen inputs, plan, preregistration, runner, parsing helper,
runtime manifest, decode settings, and provider response metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_grounding_eval import DECODE, parse_prob_with_status  # noqa: E402

ANALYSIS_ID = "gse96583-sentinel-factorial-holdout-v1"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
DATA_PATH = ROOT / "signal" / "single_cell" / "gse96583_sentinel_factorial.csv"
MANIFEST_PATH = (
    ROOT / "signal" / "single_cell" / "gse96583_sentinel_factorial.manifest.json"
)
PREREG_PATH = ROOT / "docs" / "GSE96583_SENTINEL_FACTORIAL_PREREG.md"
PARENT_DATA_PATH = (
    ROOT / "signal" / "single_cell" / "gse96583_cd8_nk_module_replication.csv"
)
PARENT_MANIFEST_PATH = (
    ROOT
    / "signal"
    / "single_cell"
    / "gse96583_cd8_nk_module_replication.manifest.json"
)
HELPER_PATH = HERE / "run_grounding_eval.py"
BENCHMARK_TASKS_PATH = HERE / "benchmark_tasks.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
RESULT_ROOT = ROOT / "results" / "benchmark" / "single_cell" / "sentinel_factorial"

EXPECTED_DATA_SHA256 = "673db8c8bcd6ba923e62891de6cd5f04f97967706ac3ade8a6e44ad2d14a4b95"
EXPECTED_MANIFEST_SHA256 = (
    "9f0035469c81852b11e2a36651b7892bf2dad4d30f8049b37cd1f655ca9bf0c4"
)
EXPECTED_PREREG_SHA256 = (
    "eaa0ad301d44dcb644e45deeb83a8de75ce6121a14635a8c5a36ee2fa3fbb707"
)
EXPECTED_PLAN_SHA256 = (
    "db5af60a5142c6f79cbac7abdba85520077efd75fab732db0bfda80f6843a20e"
)
EXPECTED_PARENT_DATA_SHA256 = (
    "f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f"
)
EXPECTED_PARENT_EXCLUSION_SHA256 = (
    "35d2ba1aa04f7e740f037d30b677001b5b05900e75a35da1b82fc2dd1f036950"
)
EXPECTED_INPUT_SENTENCE_PLAN_SHA256 = (
    "8e2e22a3e612f3383f6466d79fb6978e1c1bd0117cae87c6f0edff74f0026933"
)

CLASS_A = "CD8+ T cell"
CLASS_B = "NK cell"
CANONICAL_FORM = "ab_pa"
FORMS = {
    "ab_pa": {"first": CLASS_A, "second": CLASS_B, "target": CLASS_A},
    "ab_pb": {"first": CLASS_A, "second": CLASS_B, "target": CLASS_B},
    "ba_pa": {"first": CLASS_B, "second": CLASS_A, "target": CLASS_A},
    "ba_pb": {"first": CLASS_B, "second": CLASS_A, "target": CLASS_B},
}
FORM_ORDER = tuple(FORMS)
SYSTEM_PROMPT = (
    "You are a property predictor. Respond with ONLY a single decimal number "
    "between 0 and 1 (for example: 0.42). No words, no explanation, no analysis, "
    "no units. Your entire reply must be just the number."
)
PROMPT = (
    "These are the most distinctively expressed genes of one human immune cell "
    "(PBMC), highest first:\n{genes}\nIs it a {first} or a {second}? Reply with "
    "ONLY a number between 0 and 1 = the probability that it is a {target}."
)
FROZEN_DECODE = {"temperature": 0.0, "max_tokens": 16}
FROZEN_MAX_ATTEMPTS = 5
MARGIN = 0.03
EXPECTED_DONORS = 8
EXPECTED_CALLS = 480
EXPECTED_INPUTS = 120
SENTINELS = ("GNLY", "NKG7", "CCL5")
SUBSETS = (
    "GNLY",
    "NKG7",
    "CCL5",
    "GNLY+NKG7",
    "GNLY+CCL5",
    "NKG7+CCL5",
    "GNLY+NKG7+CCL5",
)
SHORT = {
    "GNLY": "G",
    "NKG7": "N",
    "CCL5": "C",
    "GNLY+NKG7": "GN",
    "GNLY+CCL5": "GC",
    "NKG7+CCL5": "NC",
    "GNLY+NKG7+CCL5": "GNC",
}
_THREAD_LOCAL = threading.local()


class SentinelFactorialError(ValueError):
    """Raised when a frozen artifact, checkpoint, or result violates its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return _text_sha256(_canonical_json(value))


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _raw_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}_raw.jsonl"


def _json_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}.json"


def _markdown_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}.md"


def _runtime_manifest_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}_execution_dependency_manifest.json"


def _verify_hash(
    path: Path,
    expected: str,
    label: str,
    *,
    allow_pending: bool = False,
) -> str:
    if not path.is_file():
        raise SentinelFactorialError(f"missing {label}: {path}")
    observed = _sha256(path)
    if expected.startswith("PENDING_"):
        if allow_pending:
            return observed
        raise SentinelFactorialError(
            f"{label} freeze constant is still pending; run --dry-run, freeze the "
            "reported plan/preregistration hashes, then execute"
        )
    if observed != expected:
        raise SentinelFactorialError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _build_prompt(genes: str, form: str) -> str:
    spec = FORMS[form]
    return PROMPT.format(
        genes=genes,
        first=spec["first"],
        second=spec["second"],
        target=spec["target"],
    )


def _condition_rows(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    states: list[dict[str, str]] = []
    for row in base_rows:
        states.append(
            {
                "entity_id": row["entity_id"],
                "condition": "unmasked",
                "subset_id": "none",
                "sentence": row["original_sentence"],
                "target_genes": "",
                "control_genes": "",
                "subset_size": "0",
                "subset_mask": "000",
            }
        )
    for row in intervention_rows:
        for condition, column in (
            ("target_mask", "module_mask_sentence"),
            ("control_mask", "control_mask_sentence"),
        ):
            states.append(
                {
                    "entity_id": row["entity_id"],
                    "condition": condition,
                    "subset_id": row["subset_id"],
                    "sentence": row[column],
                    "target_genes": row["target_genes"],
                    "control_genes": row["control_genes"],
                    "subset_size": row["subset_size"],
                    "subset_mask": row["subset_mask"],
                }
            )
    return states


def _parent_exclusion_hash(rows: list[dict[str, str]]) -> str:
    records = sorted(
        (
            {
                "donor_id": row["donor_id"],
                "cell_barcode": row["cell_barcode"],
            }
            for row in rows
            if row["row_type"] == "base"
        ),
        key=lambda record: (int(record["donor_id"]), record["cell_barcode"]),
    )
    return _canonical_sha256(records)


def _validate_rows(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    parent_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [row for row in rows if row["row_type"] == "intervention"]
    if len(base_rows) != EXPECTED_DONORS or len(intervention_rows) != 56:
        raise SentinelFactorialError(
            f"frozen row counts changed: base={len(base_rows)}, "
            f"intervention={len(intervention_rows)}"
        )
    donors = [row["donor_id"] for row in base_rows]
    if len(set(donors)) != EXPECTED_DONORS or any(
        count != 1 for count in Counter(donors).values()
    ):
        raise SentinelFactorialError("the design requires exactly one cell per donor")
    if len({row["entity_id"] for row in base_rows}) != EXPECTED_DONORS:
        raise SentinelFactorialError("base entity IDs are not unique")
    parent_barcodes = {
        row["cell_barcode"] for row in parent_rows if row["row_type"] == "base"
    }
    if parent_barcodes & {row["cell_barcode"] for row in base_rows}:
        raise SentinelFactorialError("held-out cells overlap the parent experiment")

    expected_subset_records = manifest["factorial"]["nonempty_subsets"]
    expected_by_id = {
        record["subset_id"]: (
            int(record["subset_size"]),
            str(record["subset_mask"]),
        )
        for record in expected_subset_records
    }
    if tuple(expected_by_id) != SUBSETS:
        raise SentinelFactorialError("manifest subset order or membership changed")
    by_entity = {row["entity_id"]: row for row in base_rows}
    intervention_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in intervention_rows:
        key = (row["entity_id"], row["subset_id"])
        if key in intervention_by_key:
            raise SentinelFactorialError(f"duplicate intervention row: {key}")
        intervention_by_key[key] = row
        if row["entity_id"] not in by_entity:
            raise SentinelFactorialError(f"orphan intervention row: {key}")
        if row["subset_id"] not in expected_by_id:
            raise SentinelFactorialError(f"unknown subset: {row['subset_id']}")
        size, mask = expected_by_id[row["subset_id"]]
        targets = row["target_genes"].split(";")
        controls = row["control_genes"].split(";")
        if (
            int(row["subset_size"]) != size
            or row["subset_mask"] != mask
            or len(targets) != size
            or len(controls) != size
        ):
            raise SentinelFactorialError(f"subset metadata changed: {key}")
        expected_targets = [
            sentinel
            for sentinel, bit in zip(SENTINELS, mask, strict=True)
            if bit == "1"
        ]
        if targets != expected_targets:
            raise SentinelFactorialError(f"target order changed: {key}")
        base = by_entity[row["entity_id"]]
        control_map = dict(
            zip(
                SENTINELS,
                base["joint_control_genes"].split(";"),
                strict=True,
            )
        )
        if controls != [control_map[target] for target in targets]:
            raise SentinelFactorialError(f"nested control assignment changed: {key}")
        original_tokens = row["original_sentence"].split()
        for column in ("module_mask_sentence", "control_mask_sentence"):
            masked_tokens = row[column].split()
            if len(masked_tokens) != len(original_tokens):
                raise SentinelFactorialError(f"mask changed sentence length: {key}")
            if masked_tokens.count("MASKED_GENE") != size:
                raise SentinelFactorialError(f"mask dose changed: {key}/{column}")
            if any(
                masked != original
                for masked, original in zip(masked_tokens, original_tokens, strict=True)
                if masked != "MASKED_GENE"
            ):
                raise SentinelFactorialError(f"mask altered retained tokens: {key}")

    expected_keys = {
        (row["entity_id"], subset) for row in base_rows for subset in SUBSETS
    }
    if set(intervention_by_key) != expected_keys:
        raise SentinelFactorialError("every held-out cell must receive all seven subsets")

    states = _condition_rows(base_rows, intervention_rows)
    if len(states) != EXPECTED_INPUTS or len({state["sentence"] for state in states}) != (
        EXPECTED_INPUTS
    ):
        raise SentinelFactorialError("the frozen design requires 120 unique input states")
    sentence_plan = [
        {
            **{
                key: state[key]
                for key in ("entity_id", "condition", "subset_id")
            },
            "sentence_sha256": _text_sha256(state["sentence"]),
        }
        for state in states
    ]
    observed_sentence_plan = _canonical_sha256(sentence_plan)
    if observed_sentence_plan != EXPECTED_INPUT_SENTENCE_PLAN_SHA256:
        raise SentinelFactorialError(
            "input-sentence plan changed: "
            f"{observed_sentence_plan}"
        )
    return base_rows, intervention_rows


def _load_inputs(
    *,
    allow_pending_prereg: bool = False,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
    dict[str, str],
]:
    hashes = {
        "input_csv_sha256": _verify_hash(
            DATA_PATH,
            EXPECTED_DATA_SHA256,
            "sentinel CSV",
        ),
        "input_manifest_sha256": _verify_hash(
            MANIFEST_PATH,
            EXPECTED_MANIFEST_SHA256,
            "sentinel manifest",
        ),
        "preregistration_sha256": _verify_hash(
            PREREG_PATH,
            EXPECTED_PREREG_SHA256,
            "sentinel preregistration",
            allow_pending=allow_pending_prereg,
        ),
        "parent_csv_sha256": _verify_hash(
            PARENT_DATA_PATH,
            EXPECTED_PARENT_DATA_SHA256,
            "parent exclusion CSV",
        ),
        "parent_manifest_sha256": _verify_hash(
            PARENT_MANIFEST_PATH,
            EXPECTED_PARENT_MANIFEST_SHA256,
            "parent exclusion manifest",
        ),
    }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != ANALYSIS_ID:
        raise SentinelFactorialError("manifest analysis_id changed")
    if manifest.get("status") != "built_not_executed":
        raise SentinelFactorialError("manifest pre-execution status changed")
    if manifest.get("artifacts", {}).get("csv_sha256") != hashes[
        "input_csv_sha256"
    ]:
        raise SentinelFactorialError("manifest does not bind the frozen CSV")
    budget = manifest.get("execution_budget", {})
    if (
        budget.get("base_rows") != EXPECTED_DONORS
        or budget.get("factorial_subset_rows") != 56
        or budget.get("unique_input_sentences") != EXPECTED_INPUTS
        or budget.get("logical_condition_form_observations") != EXPECTED_CALLS
        or budget.get("unique_api_calls_after_input_deduplication") != EXPECTED_CALLS
        or budget.get("input_sentence_plan_sha256")
        != EXPECTED_INPUT_SENTENCE_PLAN_SHA256
    ):
        raise SentinelFactorialError("manifest execution budget changed")
    parent_contract = manifest.get("parent_holdout_exclusion", {})
    for key, expected in (
        ("csv_sha256", hashes["parent_csv_sha256"]),
        ("manifest_sha256", hashes["parent_manifest_sha256"]),
        ("exclusion_records_sha256", EXPECTED_PARENT_EXCLUSION_SHA256),
    ):
        if parent_contract.get(key) != expected:
            raise SentinelFactorialError(f"manifest parent {key} changed")

    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with PARENT_DATA_PATH.open(newline="", encoding="utf-8") as handle:
        parent_rows = list(csv.DictReader(handle))
    exclusion_sha256 = _parent_exclusion_hash(parent_rows)
    if exclusion_sha256 != EXPECTED_PARENT_EXCLUSION_SHA256:
        raise SentinelFactorialError(
            f"parent exclusion-record hash changed: {exclusion_sha256}"
        )
    hashes["parent_exclusion_records_sha256"] = exclusion_sha256
    base_rows, intervention_rows = _validate_rows(rows, manifest, parent_rows)
    return base_rows, intervention_rows, manifest, hashes


def _plan(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    model: str,
    *,
    enforce_freeze: bool,
) -> tuple[list[dict[str, Any]], str]:
    if model != EXPECTED_MODEL:
        raise SentinelFactorialError(
            f"model differs from preregistration: {model}"
        )
    logical: list[dict[str, Any]] = []
    for state in _condition_rows(base_rows, intervention_rows):
        sentence_sha256 = _text_sha256(state["sentence"])
        for form in FORM_ORDER:
            prompt = _build_prompt(state["sentence"], form)
            identity = {
                "analysis_id": ANALYSIS_ID,
                "model": model,
                "entity_id": state["entity_id"],
                "condition": state["condition"],
                "subset_id": state["subset_id"],
                "form": form,
                "sentence_sha256": sentence_sha256,
                "prompt_sha256": _text_sha256(prompt),
            }
            request_id = _canonical_sha256(identity)
            logical.append(
                {
                    **identity,
                    "request_id": request_id,
                    "prompt": prompt,
                    "target_genes": state["target_genes"],
                    "control_genes": state["control_genes"],
                    "subset_size": int(state["subset_size"]),
                    "subset_mask": state["subset_mask"],
                    "shuffle_key": _text_sha256(
                        f"{ANALYSIS_ID}|request-plan|{model}|{request_id}"
                    ),
                }
            )
    logical.sort(key=lambda item: (item["shuffle_key"], item["request_id"]))
    plan = [
        {**item, "planned_index": index}
        for index, item in enumerate(logical)
    ]
    if len(plan) != EXPECTED_CALLS:
        raise SentinelFactorialError(f"call count changed: {len(plan)}")
    if len({item["request_id"] for item in plan}) != EXPECTED_CALLS:
        raise SentinelFactorialError("request IDs are not unique")
    if len({item["prompt"] for item in plan}) != EXPECTED_CALLS:
        raise SentinelFactorialError("prompt payloads are not unique")
    frozen_fields = [
        {
            key: item[key]
            for key in (
                "planned_index",
                "request_id",
                "entity_id",
                "condition",
                "subset_id",
                "form",
                "sentence_sha256",
                "prompt_sha256",
                "target_genes",
                "control_genes",
                "subset_size",
                "subset_mask",
            )
        }
        for item in plan
    ]
    plan_sha256 = _canonical_sha256(frozen_fields)
    if enforce_freeze:
        if EXPECTED_PLAN_SHA256.startswith("PENDING_"):
            raise SentinelFactorialError(
                "request-plan hash is not frozen; run --dry-run and update the "
                "preregistration and EXPECTED_PLAN_SHA256 first"
            )
        if plan_sha256 != EXPECTED_PLAN_SHA256:
            raise SentinelFactorialError(
                "call plan differs from frozen hash: "
                f"expected {EXPECTED_PLAN_SHA256}, observed {plan_sha256}"
            )
    return plan, plan_sha256


def _artifact_hashes(
    frozen_hashes: dict[str, str],
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        **frozen_hashes,
        "call_plan_sha256": plan_sha256,
        "execution_code_sha256": _sha256(Path(__file__)),
        "parsing_helper_sha256": _sha256(HELPER_PATH),
        "benchmark_tasks_sha256": _sha256(BENCHMARK_TASKS_PATH),
        "pyproject_sha256": _sha256(PYPROJECT_PATH),
        "requirements_sha256": _sha256(REQUIREMENTS_PATH),
        "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT),
        "prompt_template_sha256": _text_sha256(PROMPT),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_manifest_payload(
    model: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "provider": "anthropic",
        "decode": FROZEN_DECODE,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            name: _package_version(name)
            for name in ("anthropic", "httpx", "numpy", "scipy")
        },
        "artifact_hashes": artifacts,
        "provider_retry_policy": {
            "sdk_automatic_retries": 0,
            "runner_max_attempts": FROZEN_MAX_ATTEMPTS,
            "retryable_classes": [
                "APIConnectionError",
                "APITimeoutError",
                "RateLimitError",
                "InternalServerError",
                "APIStatusError(status in 408,409,429,500-599)",
            ],
            "temperature_fallback": (
                "one retry without temperature only for a BadRequestError whose "
                "message identifies temperature"
            ),
        },
    }


def _write_or_validate_runtime_manifest(
    model: str,
    artifacts: dict[str, Any],
) -> tuple[Path, str]:
    path = _runtime_manifest_path(model)
    payload = _runtime_manifest_payload(model, artifacts)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise SentinelFactorialError(
                "runtime/dependency manifest differs from the current frozen "
                "execution environment"
            )
    else:
        path.write_text(serialized, encoding="utf-8")
    return path, _sha256(path)


def _request_payload(item: dict[str, Any], model: str, *, temperature: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": FROZEN_DECODE["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": item["prompt"]}],
    }
    if temperature:
        payload["temperature"] = FROZEN_DECODE["temperature"]
    return payload


def _anthropic_client() -> Any:
    client = getattr(_THREAD_LOCAL, "anthropic_client", None)
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=0)
        _THREAD_LOCAL.anthropic_client = client
    return client


def _safe_error(error: Exception, attempt: int) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "class": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "request_id": getattr(error, "request_id", None),
        "retry_after": (
            getattr(getattr(error, "response", None), "headers", {}).get(
                "retry-after"
            )
            if getattr(error, "response", None) is not None
            else None
        ),
    }


def _is_retryable(error: Exception) -> bool:
    import anthropic

    if isinstance(
        error,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(error, anthropic.APIStatusError):
        status = int(getattr(error, "status_code", 0) or 0)
        return status in {408, 409, 429} or 500 <= status <= 599
    return False


def _call_with_retry(
    item: dict[str, Any],
    model: str,
    *,
    max_attempts: int,
) -> dict[str, Any]:
    import anthropic

    errors: list[dict[str, Any]] = []
    temperature_fallback_used = False
    temperature_sent = True
    for attempt in range(1, max_attempts + 1):
        payload = _request_payload(item, model, temperature=temperature_sent)
        try:
            raw_response = _anthropic_client().messages.with_raw_response.create(
                **payload
            )
            message = raw_response.parse()
            raw_output = "".join(
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text"
            )
            usage = (
                message.usage.model_dump(mode="json")
                if hasattr(message.usage, "model_dump")
                else dict(message.usage)
            )
            return {
                "raw_output": raw_output,
                "returned_model": str(message.model),
                "response_id": str(message.id),
                "provider_request_id": (
                    raw_response.request_id
                    or raw_response.headers.get("request-id")
                    or raw_response.headers.get("x-request-id")
                ),
                "http_status_code": int(raw_response.status_code),
                "stop_reason": message.stop_reason,
                "stop_sequence": message.stop_sequence,
                "usage": usage,
                "attempt_count": attempt,
                "retry_count": len(errors),
                "attempt_errors": errors,
                "temperature_requested": FROZEN_DECODE["temperature"],
                "temperature_sent": (
                    FROZEN_DECODE["temperature"] if temperature_sent else None
                ),
                "temperature_fallback_used": temperature_fallback_used,
                "effective_request_payload_sha256": _canonical_sha256(payload),
            }
        except anthropic.BadRequestError as error:
            safe = _safe_error(error, attempt)
            if (
                temperature_sent
                and not temperature_fallback_used
                and "temperature" in str(error).lower()
            ):
                errors.append({**safe, "action": "retry_without_temperature"})
                temperature_sent = False
                temperature_fallback_used = True
                continue
            raise
        except Exception as error:
            errors.append(
                {
                    **_safe_error(error, attempt),
                    "action": (
                        "retry_with_backoff"
                        if _is_retryable(error) and attempt < max_attempts
                        else "raise"
                    ),
                }
            )
            if not _is_retryable(error) or attempt == max_attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def _base_maps(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
]:
    return (
        {row["entity_id"]: row for row in base_rows},
        {(row["entity_id"], row["subset_id"]): row for row in intervention_rows},
    )


def _execute_item(
    item: dict[str, Any],
    model: str,
    base_by_entity: dict[str, dict[str, str]],
    artifacts: dict[str, Any],
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
    max_attempts: int,
) -> dict[str, Any]:
    base = base_by_entity[item["entity_id"]]
    started = datetime.now(UTC)
    response = _call_with_retry(item, model, max_attempts=max_attempts)
    finished = datetime.now(UTC)
    probability, parsed = parse_prob_with_status(response["raw_output"])
    queried_target = FORMS[item["form"]]["target"]
    aligned = probability if queried_target == CLASS_A else 1.0 - probability
    requested_payload = _request_payload(item, model, temperature=True)
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "provider": "anthropic",
        "requested_model": model,
        "returned_model": response["returned_model"],
        "entity_id": item["entity_id"],
        "cell_barcode": base["cell_barcode"],
        "donor_id": base["donor_id"],
        "reference_annotation_descriptive_only": base["reference_annotation"],
        "label_a_descriptive_only": int(base["label_a"]),
        "request_id": item["request_id"],
        "planned_index": item["planned_index"],
        "condition": item["condition"],
        "subset_id": item["subset_id"],
        "subset_short": SHORT.get(item["subset_id"], "none"),
        "subset_size": item["subset_size"],
        "subset_mask": item["subset_mask"],
        "target_genes": item["target_genes"],
        "control_genes": item["control_genes"],
        "form": item["form"],
        "answer_order": item["form"][:2],
        "queried_target": "A" if queried_target == CLASS_A else "B",
        "raw_output": response["raw_output"],
        "reported_probability": probability,
        "aligned_probability_a": aligned,
        "parsed": parsed,
        "sentence_sha256": item["sentence_sha256"],
        "prompt_sha256": item["prompt_sha256"],
        "system_prompt_sha256": artifacts["system_prompt_sha256"],
        "prompt_template_sha256": artifacts["prompt_template_sha256"],
        "requested_payload_sha256": _canonical_sha256(requested_payload),
        "effective_request_payload_sha256": response[
            "effective_request_payload_sha256"
        ],
        "decode_requested": FROZEN_DECODE,
        "response_id": response["response_id"],
        "provider_request_id": response["provider_request_id"],
        "http_status_code": response["http_status_code"],
        "stop_reason": response["stop_reason"],
        "stop_sequence": response["stop_sequence"],
        "usage": response["usage"],
        "attempt_count": response["attempt_count"],
        "max_attempts": max_attempts,
        "retry_count": response["retry_count"],
        "attempt_errors": response["attempt_errors"],
        "temperature_requested": response["temperature_requested"],
        "temperature_sent": response["temperature_sent"],
        "temperature_fallback_used": response["temperature_fallback_used"],
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        **artifacts,
        "runtime_dependency_manifest": str(runtime_manifest_path.relative_to(ROOT)),
        "runtime_dependency_manifest_sha256": runtime_manifest_sha256,
    }


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SentinelFactorialError(
                    f"invalid checkpoint JSON at line {line_number}"
                ) from error
            request_id = record.get("request_id")
            if not isinstance(request_id, str):
                raise SentinelFactorialError(
                    f"checkpoint request_id missing at line {line_number}"
                )
            if request_id in records:
                raise SentinelFactorialError(
                    f"duplicate checkpoint request_id: {request_id}"
                )
            records[request_id] = record
    return records


def _expected_item_metadata(
    item: dict[str, Any],
    model: str,
    base: dict[str, str],
    artifacts: dict[str, Any],
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
) -> dict[str, Any]:
    queried_target = FORMS[item["form"]]["target"]
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "provider": "anthropic",
        "requested_model": model,
        "entity_id": item["entity_id"],
        "cell_barcode": base["cell_barcode"],
        "donor_id": base["donor_id"],
        "reference_annotation_descriptive_only": base["reference_annotation"],
        "label_a_descriptive_only": int(base["label_a"]),
        "request_id": item["request_id"],
        "planned_index": item["planned_index"],
        "condition": item["condition"],
        "subset_id": item["subset_id"],
        "subset_short": SHORT.get(item["subset_id"], "none"),
        "subset_size": item["subset_size"],
        "subset_mask": item["subset_mask"],
        "target_genes": item["target_genes"],
        "control_genes": item["control_genes"],
        "form": item["form"],
        "answer_order": item["form"][:2],
        "queried_target": "A" if queried_target == CLASS_A else "B",
        "sentence_sha256": item["sentence_sha256"],
        "prompt_sha256": item["prompt_sha256"],
        "system_prompt_sha256": artifacts["system_prompt_sha256"],
        "prompt_template_sha256": artifacts["prompt_template_sha256"],
        "requested_payload_sha256": _canonical_sha256(
            _request_payload(item, model, temperature=True)
        ),
        "decode_requested": FROZEN_DECODE,
        "max_attempts": FROZEN_MAX_ATTEMPTS,
        **artifacts,
        "runtime_dependency_manifest": str(runtime_manifest_path.relative_to(ROOT)),
        "runtime_dependency_manifest_sha256": runtime_manifest_sha256,
    }


def _validate_checkpoint_record(
    record: dict[str, Any],
    item: dict[str, Any],
    *,
    model: str,
    base: dict[str, str],
    artifacts: dict[str, Any],
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
) -> None:
    expected = _expected_item_metadata(
        item,
        model,
        base,
        artifacts,
        runtime_manifest_path,
        runtime_manifest_sha256,
    )
    for field, value in expected.items():
        if record.get(field) != value:
            raise SentinelFactorialError(
                f"checkpoint mismatch for {item['request_id']}: {field}"
            )
    probability, parsed = parse_prob_with_status(record.get("raw_output"))
    queried_target = FORMS[item["form"]]["target"]
    aligned = probability if queried_target == CLASS_A else 1.0 - probability
    for field, expected_value in (
        ("reported_probability", probability),
        ("aligned_probability_a", aligned),
    ):
        try:
            observed = float(record[field])
        except (KeyError, TypeError, ValueError) as error:
            raise SentinelFactorialError(
                f"checkpoint probability missing: {item['request_id']}/{field}"
            ) from error
        if not np.isclose(observed, expected_value, rtol=0.0, atol=1e-15):
            raise SentinelFactorialError(
                f"checkpoint probability changed: {item['request_id']}/{field}"
            )
    if record.get("parsed") is not parsed:
        raise SentinelFactorialError(
            f"checkpoint parse flag changed: {item['request_id']}"
        )
    if record.get("returned_model") != model:
        raise SentinelFactorialError(
            f"provider returned a different model: {record.get('returned_model')}"
        )
    if not record.get("response_id") or not record.get("provider_request_id"):
        raise SentinelFactorialError(
            f"provider identifiers missing: {item['request_id']}"
        )
    if record.get("http_status_code") != 200:
        raise SentinelFactorialError(
            f"non-200 checkpoint response: {item['request_id']}"
        )
    if not isinstance(record.get("usage"), dict):
        raise SentinelFactorialError(f"usage missing: {item['request_id']}")
    if not isinstance(record.get("attempt_errors"), list):
        raise SentinelFactorialError(
            f"attempt metadata missing: {item['request_id']}"
        )
    if record.get("retry_count") != len(record["attempt_errors"]):
        raise SentinelFactorialError(
            f"retry count is incoherent: {item['request_id']}"
        )
    if not 1 <= int(record.get("attempt_count", 0)) <= FROZEN_MAX_ATTEMPTS:
        raise SentinelFactorialError(
            f"attempt count is incoherent: {item['request_id']}"
        )
    sent_temperature = record.get("temperature_sent")
    fallback = record.get("temperature_fallback_used")
    if fallback:
        if sent_temperature is not None:
            raise SentinelFactorialError(
                f"temperature fallback metadata changed: {item['request_id']}"
            )
    elif sent_temperature != FROZEN_DECODE["temperature"]:
        raise SentinelFactorialError(
            f"temperature metadata changed: {item['request_id']}"
        )
    expected_effective = _canonical_sha256(
        _request_payload(item, model, temperature=not bool(fallback))
    )
    if record.get("effective_request_payload_sha256") != expected_effective:
        raise SentinelFactorialError(
            f"effective request hash changed: {item['request_id']}"
        )
    try:
        started = datetime.fromisoformat(record["started_at_utc"])
        finished = datetime.fromisoformat(record["finished_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise SentinelFactorialError(
            f"checkpoint timestamp invalid: {item['request_id']}"
        ) from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise SentinelFactorialError(
            f"checkpoint timestamp order invalid: {item['request_id']}"
        )


def run(
    model: str,
    workers: int,
    max_attempts: int,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    frozen_hashes: dict[str, str],
) -> Path:
    if workers < 1:
        raise SentinelFactorialError("--workers must be at least 1")
    if max_attempts < 1:
        raise SentinelFactorialError("--max-attempts must be at least 1")
    if max_attempts != FROZEN_MAX_ATTEMPTS:
        raise SentinelFactorialError(
            f"retry policy is frozen at {FROZEN_MAX_ATTEMPTS} attempts"
        )
    if DECODE != FROZEN_DECODE:
        raise SentinelFactorialError(f"helper decode changed: {DECODE}")
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        model,
        enforce_freeze=True,
    )
    artifacts = _artifact_hashes(frozen_hashes, plan_sha256)
    runtime_path, runtime_sha256 = _write_or_validate_runtime_manifest(
        model,
        artifacts,
    )
    base_by_entity, _ = _base_maps(base_rows, intervention_rows)
    path = _raw_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_checkpoint(path)
    plan_by_id = {item["request_id"]: item for item in plan}
    if set(existing) - set(plan_by_id):
        raise SentinelFactorialError("checkpoint contains requests outside the plan")
    for request_id, record in existing.items():
        item = plan_by_id[request_id]
        _validate_checkpoint_record(
            record,
            item,
            model=model,
            base=base_by_entity[item["entity_id"]],
            artifacts=artifacts,
            runtime_manifest_path=runtime_path,
            runtime_manifest_sha256=runtime_sha256,
        )
    response_ids = [record["response_id"] for record in existing.values()]
    if len(response_ids) != len(set(response_ids)):
        raise SentinelFactorialError("checkpoint response IDs are not unique")
    pending = [item for item in plan if item["request_id"] not in existing]
    if not pending:
        print(f"checkpoint already complete: {len(existing)}/{len(plan)}", flush=True)
        return path

    failures: list[tuple[dict[str, Any], Exception]] = []
    completed = len(existing)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_item,
                item,
                model,
                base_by_entity,
                artifacts,
                runtime_path,
                runtime_sha256,
                max_attempts,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: PERF203
                failures.append((item, error))
                print(
                    f"FAILED {item['planned_index']} {item['entity_id']} "
                    f"{item['condition']} {item['subset_id']} {item['form']}: "
                    f"{type(error).__name__}",
                    flush=True,
                )
                continue
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed += 1
            if completed % 20 == 0 or completed == len(plan):
                print(
                    f"{model} sentinel checkpoint {completed}/{len(plan)} "
                    f"parsed={record['parsed']}",
                    flush=True,
                )
    if failures:
        item, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} calls failed; first={item['request_id']} "
            f"{type(error).__name__}"
        ) from error
    return path


def _require_donor_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (EXPECTED_DONORS,) or not np.isfinite(vector).all():
        raise SentinelFactorialError(
            "registered inference requires exactly eight finite donor values; "
            f"observed shape={vector.shape}"
        )
    return vector


def _t_summary(values: np.ndarray, *, confidence: float = 0.95) -> dict[str, Any]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise SentinelFactorialError("t summary requires at least two finite values")
    n = len(vector)
    mean = float(vector.mean())
    standard_deviation = float(vector.std(ddof=1))
    standard_error = standard_deviation / float(np.sqrt(n))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, n - 1))
    return {
        "n": n,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "degrees_freedom": n - 1,
        "confidence_level": confidence,
        "ci_lower": mean - critical * standard_error,
        "ci_upper": mean + critical * standard_error,
    }


def _one_sided_t_p(
    values: np.ndarray,
    *,
    null: float,
    direction: str,
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    standard_error = float(vector.std(ddof=1) / np.sqrt(len(vector)))
    centered_mean = float(vector.mean() - null)
    if standard_error == 0.0:
        if centered_mean == 0.0:
            statistic = 0.0
        else:
            statistic = float(np.copysign(np.inf, centered_mean))
    else:
        statistic = centered_mean / standard_error
    if direction == "greater":
        p_value = float(student_t.sf(statistic, len(vector) - 1))
    elif direction == "less":
        p_value = float(student_t.cdf(statistic, len(vector) - 1))
    else:
        raise SentinelFactorialError(f"unknown direction: {direction}")
    return statistic, p_value


def _exact_sign_flip_p(
    values: np.ndarray,
    direction: str,
    *,
    null: float = 0.0,
) -> float:
    """Exact one-sided Rademacher test, conditional on null symmetry."""
    vector = np.asarray(values, dtype=float) - null
    observed = float(vector.mean())
    permuted = np.asarray(
        [
            float(np.mean(vector * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(vector))
        ],
        dtype=float,
    )
    tolerance = 1e-15
    if direction == "greater":
        return float(np.mean(permuted >= observed - tolerance))
    if direction == "less":
        return float(np.mean(permuted <= observed + tolerance))
    raise SentinelFactorialError(f"unknown direction: {direction}")


def _exact_binomial_sign_p(
    values: np.ndarray,
    direction: str,
    *,
    null: float = 0.0,
) -> dict[str, Any]:
    """Conservative one-sided sign test; ties count against the direction."""
    vector = np.asarray(values, dtype=float)
    if direction == "greater":
        successes = int(np.sum(vector > null))
    elif direction == "less":
        successes = int(np.sum(vector < null))
    else:
        raise SentinelFactorialError(f"unknown direction: {direction}")
    return {
        "successes": successes,
        "trials": len(vector),
        "ties_counted_as_failures": int(np.sum(vector == null)),
        "p_value": float(
            binomtest(successes, len(vector), 0.5, alternative="greater").pvalue
        ),
    }


def _vector_summary(
    donor_ids: list[str],
    values: np.ndarray,
) -> dict[str, Any]:
    vector = _require_donor_vector(values)
    ci95 = _t_summary(vector, confidence=0.95)
    ci90 = _t_summary(vector, confidence=0.90)
    lodo = np.asarray(
        [float(np.delete(vector, index).mean()) for index in range(len(vector))],
        dtype=float,
    )
    statistic, two_sided_half = _one_sided_t_p(
        vector,
        null=0.0,
        direction="greater",
    )
    two_sided_p = (
        1.0
        if statistic == 0.0
        else float(2.0 * student_t.sf(abs(statistic), len(vector) - 1))
    )
    return {
        "n_donors": len(vector),
        "mean": ci95["mean"],
        "standard_deviation": ci95["standard_deviation"],
        "standard_error": ci95["standard_error"],
        "degrees_freedom": ci95["degrees_freedom"],
        "ci95_lower": ci95["ci_lower"],
        "ci95_upper": ci95["ci_upper"],
        "ci90_lower": ci90["ci_lower"],
        "ci90_upper": ci90["ci_upper"],
        "two_sided_student_t_p_value": two_sided_p,
        "donor_values": {
            donor: float(value)
            for donor, value in zip(donor_ids, vector, strict=True)
        },
        "leave_one_donor_out_means": {
            donor: float(value)
            for donor, value in zip(donor_ids, lodo, strict=True)
        },
    }


def _material_negative_summary(
    donor_ids: list[str],
    values: np.ndarray,
    *,
    margin: float = MARGIN,
) -> dict[str, Any]:
    vector = _require_donor_vector(values)
    summary = _vector_summary(donor_ids, vector)
    shifted = vector + margin
    statistic, student_p = _one_sided_t_p(
        shifted,
        null=0.0,
        direction="less",
    )
    exact_p = _exact_sign_flip_p(shifted, "less")
    sign_test = _exact_binomial_sign_p(shifted, "less")
    lodo = np.asarray(
        list(summary["leave_one_donor_out_means"].values()),
        dtype=float,
    )
    ci_pass = bool(summary["ci95_upper"] < -margin)
    sign_count = int(np.sum(vector < -margin))
    output = {
        "margin": margin,
        "null_boundary": -margin,
        "shifted_student_t_statistic": statistic,
        "one_sided_student_t_p_value": student_p,
        "exact_one_sided_rademacher_p_value": exact_p,
        "exact_binomial_sign_test": sign_test,
        "donors_below_negative_margin": sign_count,
        "ci95_below_negative_margin": ci_pass,
        "exact_sign_flip_pass": bool(exact_p < 0.05),
        "donor_sign_pass": bool(sign_count >= 7),
        "all_lodo_below_negative_margin": bool(np.all(lodo < -margin)),
    }
    output["pass"] = bool(
        output["ci95_below_negative_margin"]
        and output["exact_sign_flip_pass"]
        and output["donor_sign_pass"]
        and output["all_lodo_below_negative_margin"]
    )
    return output


def _equivalence_summary(
    donor_ids: list[str],
    values: np.ndarray,
    *,
    margin: float = MARGIN,
) -> dict[str, Any]:
    vector = _require_donor_vector(values)
    summary = _vector_summary(donor_ids, vector)
    lower_statistic, lower_student_p = _one_sided_t_p(
        vector,
        null=-margin,
        direction="greater",
    )
    upper_statistic, upper_student_p = _one_sided_t_p(
        vector,
        null=margin,
        direction="less",
    )
    lower_exact_p = _exact_sign_flip_p(vector, "greater", null=-margin)
    upper_exact_p = _exact_sign_flip_p(vector, "less", null=margin)
    lower_sign = _exact_binomial_sign_p(vector, "greater", null=-margin)
    upper_sign = _exact_binomial_sign_p(vector, "less", null=margin)
    lodo = np.asarray(
        list(summary["leave_one_donor_out_means"].values()),
        dtype=float,
    )
    output = {
        "margin": margin,
        "ci95_strictly_within_margin": bool(
            summary["ci95_lower"] > -margin
            and summary["ci95_upper"] < margin
        ),
        "exact_lower_shift_p_value": lower_exact_p,
        "exact_upper_shift_p_value": upper_exact_p,
        "exact_shifted_tost_pass": bool(
            lower_exact_p < 0.05 and upper_exact_p < 0.05
        ),
        "exact_binomial_lower_sign_test": lower_sign,
        "exact_binomial_upper_sign_test": upper_sign,
        "student_t_lower_statistic": lower_statistic,
        "student_t_upper_statistic": upper_statistic,
        "student_t_lower_p_value": lower_student_p,
        "student_t_upper_p_value": upper_student_p,
        "student_t_tost_p_value": max(lower_student_p, upper_student_p),
        "student_t_tost_pass": bool(
            lower_student_p < 0.05 and upper_student_p < 0.05
        ),
        "ci90_strictly_within_margin": bool(
            summary["ci90_lower"] > -margin
            and summary["ci90_upper"] < margin
        ),
        "all_lodo_strictly_within_margin": bool(
            np.all(lodo > -margin) and np.all(lodo < margin)
        ),
    }
    output["pass"] = bool(
        output["ci95_strictly_within_margin"]
        and output["exact_shifted_tost_pass"]
        and output["all_lodo_strictly_within_margin"]
    )
    return output


def _registered_summary(
    donor_ids: list[str],
    values: np.ndarray,
    *,
    margin: float = MARGIN,
) -> dict[str, Any]:
    return {
        **_vector_summary(donor_ids, values),
        "material_negative": _material_negative_summary(
            donor_ids,
            values,
            margin=margin,
        ),
        "equivalent_to_zero": _equivalence_summary(
            donor_ids,
            values,
            margin=margin,
        ),
    }


def _registered_vectors(
    surface: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    g = surface["GNLY"]
    n = surface["NKG7"]
    c = surface["CCL5"]
    gn = surface["GNLY+NKG7"]
    gc = surface["GNLY+CCL5"]
    nc = surface["NKG7+CCL5"]
    gnc = surface["GNLY+NKG7+CCL5"]
    return {
        "A_GNLY": g,
        "T_full_triple": gnc,
        "J_increment_NKG7_CCL5_after_GNLY": gnc - g,
        "U_NKG7_CCL5_without_GNLY": nc,
        "Q_GNLY_on_NKG7_CCL5_background": gnc - nc,
        "K_nonadditivity_GNLY_vs_NKG7_CCL5": gnc - g - nc,
        "surface_NKG7": n,
        "surface_CCL5": c,
        "surface_NKG7_CCL5": nc,
        "increment_NKG7_after_GNLY": gn - g,
        "increment_CCL5_after_GNLY": gc - g,
        "increment_NKG7_CCL5_after_GNLY": gnc - g,
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def _conditional_shapley(
    surface: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "NKG7": 0.5
        * (
            surface["GNLY+NKG7"]
            - surface["GNLY"]
            + surface["GNLY+NKG7+CCL5"]
            - surface["GNLY+CCL5"]
        ),
        "CCL5": 0.5
        * (
            surface["GNLY+CCL5"]
            - surface["GNLY"]
            + surface["GNLY+NKG7+CCL5"]
            - surface["GNLY+NKG7"]
        ),
    }


def _decision(
    donor_ids: list[str],
    r_surface: dict[str, np.ndarray],
    a_surface: dict[str, np.ndarray],
    h_surface: dict[str, np.ndarray],
    *,
    margin: float = MARGIN,
) -> dict[str, Any]:
    r_vectors = _registered_vectors(r_surface)
    a_vectors = _registered_vectors(a_surface)
    r_summaries = {
        name: _registered_summary(donor_ids, values, margin=margin)
        for name, values in r_vectors.items()
    }
    a_summaries = {
        name: _registered_summary(donor_ids, values, margin=margin)
        for name, values in a_vectors.items()
    }
    anchor_names = ("A_GNLY", "T_full_triple")
    anchor_components = {
        f"{scale}:{name}": summaries[name]["material_negative"]
        for scale, summaries in (("r", r_summaries), ("a", a_summaries))
        for name in anchor_names
    }
    anchor_pass = all(record["pass"] for record in anchor_components.values())
    anchor_iut_exact_p = max(
        record["exact_one_sided_rademacher_p_value"]
        for record in anchor_components.values()
    )

    j_name = "J_increment_NKG7_CCL5_after_GNLY"
    j_equivalent = (
        r_summaries[j_name]["equivalent_to_zero"]["pass"]
        and a_summaries[j_name]["equivalent_to_zero"]["pass"]
    )
    j_material = (
        r_summaries[j_name]["material_negative"]["pass"]
        and a_summaries[j_name]["material_negative"]["pass"]
    )
    if not anchor_pass:
        endpoint = "anchor_gate_failed"
    elif j_equivalent:
        endpoint = "GNLY_endpoint_capture"
    elif j_material:
        endpoint = "distributed_joint_residual"
    else:
        endpoint = "inconclusive_hybrid"

    six_names = (
        "surface_NKG7",
        "surface_CCL5",
        "surface_NKG7_CCL5",
        "increment_NKG7_after_GNLY",
        "increment_CCL5_after_GNLY",
        "increment_NKG7_CCL5_after_GNLY",
    )
    q_name = "Q_GNLY_on_NKG7_CCL5_background"
    q_pass = (
        r_summaries[q_name]["material_negative"]["pass"]
        and a_summaries[q_name]["material_negative"]["pass"]
    )
    equivalence_components = {
        f"{scale}:{name}": summaries[name]["equivalent_to_zero"]
        for scale, summaries in (("r", r_summaries), ("a", a_summaries))
        for name in six_names
    }
    strong_sparse = bool(
        anchor_pass
        and q_pass
        and all(record["pass"] for record in equivalence_components.values())
    )

    h_vectors = _registered_vectors(h_surface)
    h_summaries = {
        name: _registered_summary(donor_ids, values, margin=margin)
        for name, values in h_vectors.items()
    }
    relevant_shams = {
        name: h_summaries[name]
        for name in (
            "A_GNLY",
            "T_full_triple",
            "J_increment_NKG7_CCL5_after_GNLY",
        )
    }
    specificity_wording = all(
        record["equivalent_to_zero"]["pass"] for record in relevant_shams.values()
    )

    shapley: dict[str, Any] = {
        "status": (
            "tested_after_dual_scale_distributed_endpoint"
            if endpoint == "distributed_joint_residual"
            else "not_tested_by_hierarchy"
        ),
        "individual_gene_localization_allowed": False,
    }
    if endpoint == "distributed_joint_residual":
        r_phi = _conditional_shapley(r_surface)
        a_phi = _conditional_shapley(a_surface)
        phi_summaries = {
            f"{scale}:{gene}": _registered_summary(
                donor_ids,
                values,
                margin=margin,
            )
            for scale, allocation in (("r", r_phi), ("a", a_phi))
            for gene, values in allocation.items()
        }
        shifted_exact = {
            name: record["material_negative"][
                "exact_one_sided_rademacher_p_value"
            ]
            for name, record in phi_summaries.items()
        }
        holm = _holm_adjust(shifted_exact)
        allowed_by_gene = {
            gene: bool(
                phi_summaries[f"r:{gene}"]["material_negative"]["pass"]
                and phi_summaries[f"a:{gene}"]["material_negative"]["pass"]
                and holm[f"r:{gene}"] < 0.05
                and holm[f"a:{gene}"] < 0.05
            )
            for gene in ("NKG7", "CCL5")
        }
        shapley.update(
            {
                "allocations": phi_summaries,
                "holm_family": (
                    "four shifted exact tests: r:NKG7, r:CCL5, "
                    "a:NKG7, a:CCL5"
                ),
                "holm_adjusted_exact_p_values": holm,
                "individual_gene_localization_allowed": any(
                    allowed_by_gene.values()
                ),
                "individual_gene_localization_allowed_by_gene": allowed_by_gene,
                "localized_genes": [
                    gene for gene, passed in allowed_by_gene.items() if passed
                ],
                "sum_identity_max_absolute_error": float(
                    max(
                        np.max(
                            np.abs(
                                r_phi["NKG7"]
                                + r_phi["CCL5"]
                                - r_vectors[j_name]
                            )
                        ),
                        np.max(
                            np.abs(
                                a_phi["NKG7"]
                                + a_phi["CCL5"]
                                - a_vectors[j_name]
                            )
                        ),
                    )
                ),
            }
        )

    return {
        "margin": margin,
        "anchor_gate": {
            "pass": anchor_pass,
            "required_components": anchor_components,
            "exact_intersection_union_p_value": anchor_iut_exact_p,
        },
        "endpoint_classification": endpoint,
        "endpoint_discriminator": {
            "dual_scale_equivalence_pass": j_equivalent,
            "dual_scale_material_negative_pass": j_material,
            "r_J": r_summaries[j_name],
            "a_J": a_summaries[j_name],
        },
        "strong_sparse_GNLY_gate": {
            "pass": strong_sparse,
            "dual_scale_Q_material_negative_pass": q_pass,
            "r_Q": r_summaries[q_name],
            "a_Q": a_summaries[q_name],
            "required_dual_scale_equivalences": equivalence_components,
        },
        "matched_control_sham_guard": {
            "reference_prevalence_balanced_control_language": True,
            "relevant_shams": relevant_shams,
            "target_erasure_specific_wording_allowed": specificity_wording,
            "all_registered_h_vectors": h_summaries,
        },
        "registered_r_vectors": r_summaries,
        "registered_a_vectors": a_summaries,
        "distributed_shapley_localization": shapley,
    }


def _compute_surfaces(
    donor_ids: list[str],
    probability_by_key: dict[tuple[str, str, str, str], float],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Map aligned probabilities to the frozen a/h/r factorial surfaces.

    ``probability_by_key`` is keyed by
    ``(donor_id, form, condition, subset_id)``.  The unmasked condition uses
    subset ``none``; target and control conditions use all seven subsets.
    """
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for form in FORM_ORDER:
        a_surface: dict[str, np.ndarray] = {}
        h_surface: dict[str, np.ndarray] = {}
        r_surface: dict[str, np.ndarray] = {}
        for subset in SUBSETS:
            unmasked = np.asarray(
                [
                    probability_by_key[(donor, form, "unmasked", "none")]
                    for donor in donor_ids
                ],
                dtype=float,
            )
            target = np.asarray(
                [
                    probability_by_key[(donor, form, "target_mask", subset)]
                    for donor in donor_ids
                ],
                dtype=float,
            )
            control = np.asarray(
                [
                    probability_by_key[(donor, form, "control_mask", subset)]
                    for donor in donor_ids
                ],
                dtype=float,
            )
            a_surface[subset] = unmasked - target
            h_surface[subset] = unmasked - control
            r_surface[subset] = control - target
            if not np.allclose(
                r_surface[subset],
                a_surface[subset] - h_surface[subset],
                rtol=0.0,
                atol=1e-15,
            ):
                raise SentinelFactorialError(
                    f"a-h=r identity failed for {form}/{subset}"
                )
        output[form] = {"a": a_surface, "h": h_surface, "r": r_surface}
    return output


def _average_form_surfaces(
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        scale: {
            subset: np.mean(
                np.stack(
                    [surfaces[form][scale][subset] for form in FORM_ORDER],
                    axis=0,
                ),
                axis=0,
            )
            for subset in SUBSETS
        }
        for scale in ("a", "h", "r")
    }


def _surface_report(
    donor_ids: list[str],
    surface: dict[str, dict[str, np.ndarray]],
    *,
    margin: float,
) -> dict[str, Any]:
    return {
        subset: {
            scale: _registered_summary(
                donor_ids,
                surface[scale][subset],
                margin=margin,
            )
            for scale in ("r", "a", "h")
        }
        for subset in SUBSETS
    }


def _nonadditivity_vectors(
    surface: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    g = surface["GNLY"]
    n = surface["NKG7"]
    c = surface["CCL5"]
    gn = surface["GNLY+NKG7"]
    gc = surface["GNLY+CCL5"]
    nc = surface["NKG7+CCL5"]
    gnc = surface["GNLY+NKG7+CCL5"]
    return {
        "beta_GN": gn - g - n,
        "beta_GC": gc - g - c,
        "beta_NC": nc - n - c,
        "beta_GNC": gnc - gn - gc - nc + g + n + c,
    }


def _prompt_metric_vectors(
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for scale in ("r", "a"):
        for name, subset in (
            (f"{scale}_G", "GNLY"),
            (f"{scale}_T", "GNLY+NKG7+CCL5"),
        ):
            metrics[name] = {
                form: surfaces[form][scale][subset] for form in FORM_ORDER
            }
        metrics[f"{scale}_J"] = {
            form: (
                surfaces[form][scale]["GNLY+NKG7+CCL5"]
                - surfaces[form][scale]["GNLY"]
            )
            for form in FORM_ORDER
        }
    return metrics


def _prompt_interactions(
    donor_ids: list[str],
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    margin: float,
) -> dict[str, Any]:
    metrics = _prompt_metric_vectors(surfaces)
    output: dict[str, Any] = {}
    for name, form_values in metrics.items():
        order = (
            (form_values["ab_pa"] + form_values["ab_pb"]) / 2.0
            - (form_values["ba_pa"] + form_values["ba_pb"]) / 2.0
        )
        queried_target = (
            (form_values["ab_pa"] + form_values["ba_pa"]) / 2.0
            - (form_values["ab_pb"] + form_values["ba_pb"]) / 2.0
        )
        order_summary = _vector_summary(donor_ids, order)
        target_summary = _vector_summary(donor_ids, queried_target)
        order_summary["ci95_strictly_within_margin"] = bool(
            order_summary["ci95_lower"] > -margin
            and order_summary["ci95_upper"] < margin
        )
        target_summary["ci95_strictly_within_margin"] = bool(
            target_summary["ci95_lower"] > -margin
            and target_summary["ci95_upper"] < margin
        )
        output[name] = {
            "answer_order_ab_minus_ba": order_summary,
            "queried_target_A_minus_B": target_summary,
            "both_interactions_within_margin": bool(
                order_summary["ci95_strictly_within_margin"]
                and target_summary["ci95_strictly_within_margin"]
            ),
        }
    return {
        "formulas": {
            "answer_order": (
                "[v_ab_pa + v_ab_pb]/2 - [v_ba_pa + v_ba_pb]/2"
            ),
            "queried_target": (
                "[v_ab_pa + v_ba_pa]/2 - [v_ab_pb + v_ba_pb]/2"
            ),
            "probability_alignment": "all forms aligned to P(CD8) first",
        },
        "registered_metrics": output,
        "all_registered_interactions_within_margin": all(
            record["both_interactions_within_margin"]
            for record in output.values()
        ),
    }


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = sorted(
        {
            key
            for record in records
            for key, value in record.get("usage", {}).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    )
    return {
        key: int(sum(int(record.get("usage", {}).get(key, 0)) for record in records))
        for key in keys
    }


def analyze(
    model: str,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    manifest: dict[str, Any],
    frozen_hashes: dict[str, str],
) -> dict[str, Any]:
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        model,
        enforce_freeze=True,
    )
    artifacts = _artifact_hashes(frozen_hashes, plan_sha256)
    runtime_path, runtime_sha256 = _write_or_validate_runtime_manifest(
        model,
        artifacts,
    )
    raw_path = _raw_path(model)
    checkpoint = _read_checkpoint(raw_path)
    plan_by_id = {item["request_id"]: item for item in plan}
    if set(checkpoint) != set(plan_by_id):
        missing = len(set(plan_by_id) - set(checkpoint))
        extra = len(set(checkpoint) - set(plan_by_id))
        raise SentinelFactorialError(
            f"checkpoint incomplete: observed={len(checkpoint)}, "
            f"missing={missing}, extra={extra}"
        )
    base_by_entity, _ = _base_maps(base_rows, intervention_rows)
    probability_by_key: dict[tuple[str, str, str, str], float] = {}
    parse_flags: list[bool] = []
    records: list[dict[str, Any]] = []
    for request_id, item in plan_by_id.items():
        record = checkpoint[request_id]
        _validate_checkpoint_record(
            record,
            item,
            model=model,
            base=base_by_entity[item["entity_id"]],
            artifacts=artifacts,
            runtime_manifest_path=runtime_path,
            runtime_manifest_sha256=runtime_sha256,
        )
        key = (
            record["donor_id"],
            record["form"],
            record["condition"],
            record["subset_id"],
        )
        if key in probability_by_key:
            raise SentinelFactorialError(f"duplicate probability assignment: {key}")
        probability_by_key[key] = float(record["aligned_probability_a"])
        parse_flags.append(bool(record["parsed"]))
        records.append(record)
    if len(probability_by_key) != EXPECTED_CALLS:
        raise SentinelFactorialError("checkpoint does not map to 480 unique states")
    if not all(parse_flags):
        raise SentinelFactorialError(
            "confirmatory analysis requires 100% exact parsing: "
            f"{sum(parse_flags)}/{len(parse_flags)}"
        )
    response_ids = [record["response_id"] for record in records]
    provider_request_ids = [record["provider_request_id"] for record in records]
    if len(set(response_ids)) != EXPECTED_CALLS:
        raise SentinelFactorialError("response IDs are not unique")
    if len(set(provider_request_ids)) != EXPECTED_CALLS:
        raise SentinelFactorialError("provider request IDs are not unique")

    donors = sorted(
        {row["donor_id"] for row in base_rows},
        key=lambda value: int(value),
    )
    surfaces = _compute_surfaces(donors, probability_by_key)
    canonical = surfaces[CANONICAL_FORM]
    canonical_decision = _decision(
        donors,
        canonical["r"],
        canonical["a"],
        canonical["h"],
        margin=MARGIN,
    )
    form_decisions = {
        form: _decision(
            donors,
            surfaces[form]["r"],
            surfaces[form]["a"],
            surfaces[form]["h"],
            margin=MARGIN,
        )
        for form in FORM_ORDER
    }
    ensemble = _average_form_surfaces(surfaces)
    ensemble_decision = _decision(
        donors,
        ensemble["r"],
        ensemble["a"],
        ensemble["h"],
        margin=MARGIN,
    )
    interactions = _prompt_interactions(
        donors,
        surfaces,
        margin=MARGIN,
    )
    canonical_anchor = canonical_decision["anchor_gate"]["pass"]
    canonical_endpoint = canonical_decision["endpoint_classification"]
    form_classification_preserved = all(
        decision["anchor_gate"]["pass"] == canonical_anchor
        and decision["endpoint_classification"] == canonical_endpoint
        for decision in form_decisions.values()
    )
    prompt_robust = bool(
        form_classification_preserved
        and interactions["all_registered_interactions_within_margin"]
    )

    secondary_margin = 0.05
    secondary_decision = _decision(
        donors,
        canonical["r"],
        canonical["a"],
        canonical["h"],
        margin=secondary_margin,
    )
    subset_surfaces = {
        form: _surface_report(donors, surfaces[form], margin=MARGIN)
        for form in FORM_ORDER
    }
    ensemble_surface_report = _surface_report(
        donors,
        ensemble,
        margin=MARGIN,
    )
    nonadditivities = {
        form: {
            scale: {
                name: _vector_summary(donors, values)
                for name, values in _nonadditivity_vectors(
                    surfaces[form][scale]
                ).items()
            }
            for scale in ("r", "a")
        }
        for form in FORM_ORDER
    }
    nonadditivities["four_form_ensemble"] = {
        scale: {
            name: _vector_summary(donors, values)
            for name, values in _nonadditivity_vectors(ensemble[scale]).items()
        }
        for scale in ("r", "a")
    }

    annotation_by_donor = {
        row["donor_id"]: row["reference_annotation"] for row in base_rows
    }
    annotation_summary: dict[str, Any] = {}
    for annotation in sorted(set(annotation_by_donor.values())):
        indices = [
            index
            for index, donor in enumerate(donors)
            if annotation_by_donor[donor] == annotation
        ]
        annotation_summary[annotation] = {
            "n_cells_and_donors": len(indices),
            "canonical_surface_means": {
                scale: {
                    subset: float(canonical[scale][subset][indices].mean())
                    for subset in SUBSETS
                }
                for scale in ("r", "a", "h")
            },
        }

    records_by_start = sorted(records, key=lambda record: record["started_at_utc"])
    raw_counts = dict(sorted(Counter(record["raw_output"] for record in records).items()))
    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "canonical_prompt_form": CANONICAL_FORM,
        "n_cells": len(base_rows),
        "n_donors": len(donors),
        "donors": donors,
        "model_calls": len(records),
        "parse_rate": float(np.mean(parse_flags)),
        "inference_unit": "one held-out cell per donor; donor is the unit",
        "estimands": {
            "a": "P(CD8|unmasked) - P(CD8|target mask)",
            "h": (
                "P(CD8|unmasked) - P(CD8|reference-prevalence-balanced "
                "control mask)"
            ),
            "r": "P(CD8|control mask) - P(CD8|target mask) = a-h",
            "negative_direction": (
                "intact target names have greater NK-directed output leverage"
            ),
        },
        "primary_margin": MARGIN,
        "canonical_decision": canonical_decision,
        "canonical_subset_surfaces": subset_surfaces[CANONICAL_FORM],
        "prompt_surface_analysis": {
            "form_decisions": form_decisions,
            "all_form_subset_surfaces": subset_surfaces,
            "four_form_ensemble": {
                "decision": ensemble_decision,
                "subset_surfaces": ensemble_surface_report,
            },
            "interactions": interactions,
            "all_forms_preserve_canonical_anchor_and_endpoint": (
                form_classification_preserved
            ),
            "prompt_robust": prompt_robust,
        },
        "secondary_margin_sensitivity": {
            "margin": secondary_margin,
            "canonical_decision": secondary_decision,
            "can_rescue_primary": False,
        },
        "subset_nonadditivities_descriptive_only": nonadditivities,
        "deposited_annotation_stratification_descriptive_only": (
            annotation_summary
        ),
        "raw_output_distribution": raw_counts,
        "execution": {
            "run_started_at_utc": records_by_start[0]["started_at_utc"],
            "run_finished_at_utc": max(
                record["finished_at_utc"] for record in records
            ),
            "total_retries": int(sum(record["retry_count"] for record in records)),
            "temperature_fallback_calls": int(
                sum(bool(record["temperature_fallback_used"]) for record in records)
            ),
            "usage_totals": _usage_totals(records),
            "returned_models": dict(
                Counter(record["returned_model"] for record in records)
            ),
            "response_ids_unique": True,
            "provider_request_ids_unique": True,
        },
        "inference_assumptions": {
            "student_t": (
                "Student-t7 intervals treat the eight frozen donor effects as "
                "independent donor-level observations"
            ),
            "exact_rademacher": (
                "exact enumeration over 2^8 sign assignments, conditional on "
                "donor-effect sign symmetry; not distribution-free"
            ),
            "exact_binomial_sign": (
                "one-sided exact binomial sign test with ties counted against "
                "the registered direction"
            ),
        },
        "claim_scope": manifest["claim_scope"],
        "interpretation_boundary": (
            "held-out-cell text-interface dependence in one model and one "
            "eight-donor SLE control cohort; not a biological perturbation, "
            "pathway mechanism, annotation-truth validation, latent-knowledge "
            "proof, hidden-state activation-gap test, mathematical invariant, "
            "or physical law"
        ),
        "provenance": {
            **artifacts,
            "raw_checkpoint": str(raw_path.relative_to(ROOT)),
            "raw_checkpoint_sha256": _sha256(raw_path),
            "runtime_dependency_manifest": str(runtime_path.relative_to(ROOT)),
            "runtime_dependency_manifest_sha256": runtime_sha256,
            "decode": FROZEN_DECODE,
            "max_attempts": FROZEN_MAX_ATTEMPTS,
            "plan_construction": (
                "SHA256-sort by analysis ID, model, and stable request ID"
            ),
        },
    }


def _format_effect(record: dict[str, Any]) -> str:
    return (
        f"{record['mean']:+.4f} "
        f"[{record['ci95_lower']:+.4f}, {record['ci95_upper']:+.4f}]"
    )


def _pass_label(value: bool) -> str:
    return "PASS" if value else "FAIL"


def render_markdown(result: dict[str, Any]) -> str:
    decision = result["canonical_decision"]
    r_vectors = decision["registered_r_vectors"]
    a_vectors = decision["registered_a_vectors"]
    anchor = decision["anchor_gate"]
    endpoint = decision["endpoint_discriminator"]
    sparse = decision["strong_sparse_GNLY_gate"]
    sham = decision["matched_control_sham_guard"]
    lines = [
        "# GSE96583 held-out GNLY/NKG7/CCL5 factorial result",
        "",
        (
            f"`{result['n_donors']}` SLE donors, one held-out cell per donor, "
            f"`{result['model_calls']}` unique model calls, exact parse rate "
            f"`{result['parse_rate']:.1%}`."
        ),
        "",
        (
            "The primary interface is `ab_pa`. Negative effects indicate "
            "NK-directed output leverage. `a` is absolute target-mask movement, "
            "`h` is reference-prevalence-balanced control-mask movement, and "
            "`r=a-h` is their matched contrast."
        ),
        "",
        "## Primary decision",
        "",
        (
            f"- Dual-scale anchor gate: **{_pass_label(anchor['pass'])}** "
            f"(exact IUT p={anchor['exact_intersection_union_p_value']:.6g})"
        ),
        (
            f"- Endpoint: **{decision['endpoint_classification']}** "
            f"(rJ equivalence={endpoint['r_J']['equivalent_to_zero']['pass']}, "
            f"aJ equivalence={endpoint['a_J']['equivalent_to_zero']['pass']}; "
            f"rJ material={endpoint['r_J']['material_negative']['pass']}, "
            f"aJ material={endpoint['a_J']['material_negative']['pass']})"
        ),
        (
            f"- Strong sparse-GNLY gate: **{_pass_label(sparse['pass'])}**"
        ),
        (
            "- Isolated target-erasure-specific wording: "
            f"**{'allowed' if sham['target_erasure_specific_wording_allowed'] else 'not allowed'}**"
        ),
        "",
        "| scale/vector | donor mean [95% CI] | material < -0.03 | equivalent ±0.03 |",
        "|---|---:|---:|---:|",
    ]
    vector_rows = (
        ("r", r_vectors, "A_GNLY"),
        ("a", a_vectors, "A_GNLY"),
        ("r", r_vectors, "T_full_triple"),
        ("a", a_vectors, "T_full_triple"),
        ("r", r_vectors, "J_increment_NKG7_CCL5_after_GNLY"),
        ("a", a_vectors, "J_increment_NKG7_CCL5_after_GNLY"),
        ("r", r_vectors, "Q_GNLY_on_NKG7_CCL5_background"),
        ("a", a_vectors, "Q_GNLY_on_NKG7_CCL5_background"),
    )
    for scale, records, name in vector_rows:
        record = records[name]
        lines.append(
            f"| `{scale}:{name}` | {_format_effect(record)} | "
            f"{_pass_label(record['material_negative']['pass'])} | "
            f"{_pass_label(record['equivalent_to_zero']['pass'])} |"
        )

    lines.extend(
        [
            "",
            "## Canonical seven-subset surface",
            "",
            "| target subset | r mean [95% CI] | a mean [95% CI] | h mean [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for subset in SUBSETS:
        record = result["canonical_subset_surfaces"][subset]
        lines.append(
            f"| `{subset}` | {_format_effect(record['r'])} | "
            f"{_format_effect(record['a'])} | {_format_effect(record['h'])} |"
        )

    shapley = decision["distributed_shapley_localization"]
    lines.extend(["", "## Conditional localization", ""])
    if shapley["status"] == "not_tested_by_hierarchy":
        lines.append(
            "Conditional Shapley localization was not tested because the "
            "dual-scale distributed endpoint did not pass."
        )
    else:
        lines.append(
            "Holm correction covers four shifted exact tests "
            "(`r:NKG7`, `r:CCL5`, `a:NKG7`, `a:CCL5`)."
        )
        lines.append("")
        lines.append("| allocation | mean [95% CI] | Holm p | material pass |")
        lines.append("|---|---:|---:|---:|")
        for name, record in shapley["allocations"].items():
            lines.append(
                f"| `{name}` | {_format_effect(record)} | "
                f"{shapley['holm_adjusted_exact_p_values'][name]:.6g} | "
                f"{_pass_label(record['material_negative']['pass'])} |"
            )
        localized = shapley.get("localized_genes", [])
        lines.append("")
        lines.append(
            "Individual target localization allowed: "
            + (", ".join(localized) if localized else "none; joint residual only")
            + "."
        )

    prompt = result["prompt_surface_analysis"]
    lines.extend(
        [
            "",
            "## Prompt-surface analysis",
            "",
            "| form | anchor | endpoint |",
            "|---|---:|---|",
        ]
    )
    for form in FORM_ORDER:
        form_decision = prompt["form_decisions"][form]
        lines.append(
            f"| `{form}` | {_pass_label(form_decision['anchor_gate']['pass'])} | "
            f"`{form_decision['endpoint_classification']}` |"
        )
    lines.extend(
        [
            "",
            (
                f"Prompt robust: **{_pass_label(prompt['prompt_robust'])}**. "
                "This requires every form to preserve the canonical anchor and "
                "endpoint and all twelve registered interaction intervals "
                "(six vectors × two prompt factors) to lie within ±0.03."
            ),
            "",
            "## Sensitivity and inference",
            "",
            (
                "The secondary ±0.05 analysis classified the canonical endpoint "
                f"as `{result['secondary_margin_sensitivity']['canonical_decision']['endpoint_classification']}`. "
                "It cannot rescue the ±0.03 primary decision."
            ),
            (
                "Student-t7 intervals use eight unweighted donor effects. Exact "
                "Rademacher p-values enumerate all 2^8 sign assignments and are "
                "exact only under donor-effect sign symmetry. Exact binomial sign "
                "tests are also reported, with ties counted against the direction."
            ),
            "",
            "## Provenance and boundary",
            "",
            (
                f"- Raw checkpoint SHA-256: "
                f"`{result['provenance']['raw_checkpoint_sha256']}`"
            ),
            (
                f"- Request-plan SHA-256: "
                f"`{result['provenance']['call_plan_sha256']}`"
            ),
            (
                f"- Runner SHA-256: "
                f"`{result['provenance']['execution_code_sha256']}`"
            ),
            (
                f"- Runtime/dependency manifest SHA-256: "
                f"`{result['provenance']['runtime_dependency_manifest_sha256']}`"
            ),
            "",
            result["interpretation_boundary"] + ".",
            "",
        ]
    )
    return "\n".join(lines)


def _write_results(model: str, result: dict[str, Any]) -> tuple[Path, Path]:
    json_path = _json_path(model)
    markdown_path = _markdown_path(model)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _dry_run(
    model: str,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    frozen_hashes: dict[str, str],
) -> dict[str, Any]:
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        model,
        enforce_freeze=False,
    )
    output: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "cells": len(base_rows),
        "subsets_per_cell": len(SUBSETS),
        "input_states": EXPECTED_INPUTS,
        "prompt_forms": len(FORMS),
        "unique_calls": len(plan),
        "call_plan_sha256": plan_sha256,
        "expected_call_plan_sha256": EXPECTED_PLAN_SHA256,
        "plan_matches_frozen": (
            not EXPECTED_PLAN_SHA256.startswith("PENDING_")
            and plan_sha256 == EXPECTED_PLAN_SHA256
        ),
        "preregistration_sha256": frozen_hashes["preregistration_sha256"],
        "expected_preregistration_sha256": EXPECTED_PREREG_SHA256,
        "decode": FROZEN_DECODE,
        "max_attempts": FROZEN_MAX_ATTEMPTS,
        "model_calls_made": 0,
    }
    freeze_complete = not (
        EXPECTED_PLAN_SHA256.startswith("PENDING_")
        or EXPECTED_PREREG_SHA256.startswith("PENDING_")
    )
    if freeze_complete:
        if plan_sha256 != EXPECTED_PLAN_SHA256:
            raise SentinelFactorialError(
                f"dry-run plan mismatch: {plan_sha256}"
            )
        artifacts = _artifact_hashes(frozen_hashes, plan_sha256)
        runtime_path, runtime_sha256 = _write_or_validate_runtime_manifest(
            model,
            artifacts,
        )
        output.update(
            {
                "runtime_dependency_manifest": str(
                    runtime_path.relative_to(ROOT)
                ),
                "runtime_dependency_manifest_sha256": runtime_sha256,
                "execution_code_sha256": artifacts["execution_code_sha256"],
                "parsing_helper_sha256": artifacts["parsing_helper_sha256"],
            }
        )
    else:
        output["runtime_dependency_manifest"] = (
            "not written until preregistration and plan constants are frozen"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all frozen inputs and print the request-plan hash; no API calls",
    )
    mode.add_argument(
        "--analyze-only",
        action="store_true",
        help="validate and analyze an already complete raw checkpoint",
    )
    args = parser.parse_args()

    base_rows, intervention_rows, manifest, frozen_hashes = _load_inputs(
        allow_pending_prereg=args.dry_run,
    )
    if args.dry_run:
        print(
            json.dumps(
                _dry_run(
                    args.model,
                    base_rows,
                    intervention_rows,
                    frozen_hashes,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.analyze_only:
        run(
            args.model,
            args.workers,
            FROZEN_MAX_ATTEMPTS,
            base_rows,
            intervention_rows,
            frozen_hashes,
        )
    result = analyze(
        args.model,
        base_rows,
        intervention_rows,
        manifest,
        frozen_hashes,
    )
    json_path, markdown_path = _write_results(args.model, result)
    print(
        json.dumps(
            {
                "result_json": str(json_path),
                "result_markdown": str(markdown_path),
                "endpoint": result["canonical_decision"][
                    "endpoint_classification"
                ],
                "strong_sparse": result["canonical_decision"][
                    "strong_sparse_GNLY_gate"
                ]["pass"],
                "prompt_robust": result["prompt_surface_analysis"][
                    "prompt_robust"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

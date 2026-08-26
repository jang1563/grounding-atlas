"""Run the GSE96583 donor-aware expression-context module experiment.

Reference labels define only the upstream CD8/NK task frame and descriptive
summaries. Expression support selects one of two disjoint contexts, and every
selected cell receives both modules paired within that context. Donors, not
cells or prompt forms, are the inferential units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_grounding_eval import DECODE, complete, parse_prob_with_status  # noqa: E402

ANALYSIS_ID = "gse96583-cd8-nk-context-module-donor-replication-v1"
DATA_PATH = ROOT / "signal" / "single_cell" / "gse96583_cd8_nk_module_replication.csv"
MANIFEST_PATH = (
    ROOT / "signal" / "single_cell" / "gse96583_cd8_nk_module_replication.manifest.json"
)
PREREG_PATH = ROOT / "docs" / "GSE96583_CONTEXT_MODULE_PREREG.md"
RESULT_ROOT = ROOT / "results" / "benchmark" / "single_cell" / "donor_context_factorial"
EXPECTED_DATA_SHA256 = "f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321"
EXPECTED_MANIFEST_SHA256 = "3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f"
EXPECTED_PREREG_SHA256 = "6096d8ca84f0eb72327bf6f44773e62baa909d8dd6715268f5fd099dce42d62e"
EXPECTED_PLAN_SHA256 = "bb046113f08eac0e69a12dbcca63ecbbd26fdbea1d70aa1d42db5d6ebd801615"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"

CLASS_A = "CD8+ T cell"
CLASS_B = "NK cell"
T_CONTEXT = "T_plus_cytotoxic"
RECEPTOR_CONTEXT = "NK_receptor_plus_cytotoxic"
CONTEXT_MODULES = {
    T_CONTEXT: ("T_TCR_CD8", "cytotoxic_effector"),
    RECEPTOR_CONTEXT: ("NK_receptor_identity", "cytotoxic_effector"),
}
EXPECTED_DIRECTION = {
    (T_CONTEXT, "T_TCR_CD8"): "greater",
    (T_CONTEXT, "cytotoxic_effector"): "less",
    (RECEPTOR_CONTEXT, "NK_receptor_identity"): "less",
    (RECEPTOR_CONTEXT, "cytotoxic_effector"): "less",
}
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
FORMS = {
    "ab_pa": {"first": CLASS_A, "second": CLASS_B, "target": CLASS_A},
    "ab_pb": {"first": CLASS_A, "second": CLASS_B, "target": CLASS_B},
    "ba_pa": {"first": CLASS_B, "second": CLASS_A, "target": CLASS_A},
    "ba_pb": {"first": CLASS_B, "second": CLASS_A, "target": CLASS_B},
}
EQUIVALENCE_MARGIN = 0.03
STRICT_MAX_RANK_DISTANCE = 10.0
STRICT_MAX_EXPRESSION_DISTANCE = 1.0
EXPECTED_DONORS = 8


class DonorContextFactorialError(ValueError):
    """Raised when the frozen plan, checkpoint, or analysis is incoherent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join((ANALYSIS_ID, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _raw_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}_raw.jsonl"


def _load_inputs() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
]:
    for path, expected, label in (
        (DATA_PATH, EXPECTED_DATA_SHA256, "factorial CSV"),
        (MANIFEST_PATH, EXPECTED_MANIFEST_SHA256, "factorial manifest"),
        (PREREG_PATH, EXPECTED_PREREG_SHA256, "preregistration"),
    ):
        if not path.exists():
            raise DonorContextFactorialError(f"{label} missing: {path}")
        if _sha256(path) != expected:
            raise DonorContextFactorialError(f"{label} differs from the frozen expected hash")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != ANALYSIS_ID:
        raise DonorContextFactorialError("factorial manifest analysis_id mismatch")
    if manifest["artifacts"]["csv_sha256"] != EXPECTED_DATA_SHA256:
        raise DonorContextFactorialError("manifest CSV hash mismatch")
    budget = manifest.get("execution_budget", {})
    if (
        budget.get("logical_condition_form_observations") != 1120
        or budget.get("unique_api_calls_after_input_deduplication") != 1120
        or budget.get("prompt_forms_per_input") != 4
    ):
        raise DonorContextFactorialError("manifest execution budget changed")
    if manifest.get("sampled_context_counts") != {
        RECEPTOR_CONTEXT: 24,
        T_CONTEXT: 32,
    }:
        raise DonorContextFactorialError("manifest context counts changed")

    rows = list(csv.DictReader(DATA_PATH.open(newline="", encoding="utf-8")))
    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [row for row in rows if row["row_type"] == "intervention"]
    if len(base_rows) != 56 or len(intervention_rows) != 112:
        raise DonorContextFactorialError(
            f"frozen row counts changed: base={len(base_rows)}, "
            f"intervention={len(intervention_rows)}"
        )
    base_ids = {row["entity_id"] for row in base_rows}
    if len(base_ids) != 56:
        raise DonorContextFactorialError("base entity IDs are not unique")
    if Counter(row["sampling_context"] for row in base_rows) != Counter(
        {T_CONTEXT: 32, RECEPTOR_CONTEXT: 24}
    ):
        raise DonorContextFactorialError("CSV context counts changed")
    if Counter(row["donor_id"] for row in base_rows) != Counter(
        {donor: 7 for donor in {row["donor_id"] for row in base_rows}}
    ):
        raise DonorContextFactorialError("every donor must contribute exactly seven cells")
    if len({row["donor_id"] for row in base_rows}) != EXPECTED_DONORS:
        raise DonorContextFactorialError("the frozen design requires eight donors")
    if Counter(int(row["label_a"]) for row in base_rows) != Counter({1: 33, 0: 23}):
        raise DonorContextFactorialError("descriptive deposited-label counts changed")

    by_entity = Counter(row["entity_id"] for row in intervention_rows)
    by_entity_module = Counter((row["entity_id"], row["module"]) for row in intervention_rows)
    if set(by_entity) != base_ids or set(by_entity.values()) != {2}:
        raise DonorContextFactorialError("every selected cell must have two interventions")
    if set(by_entity_module.values()) != {1}:
        raise DonorContextFactorialError("entity-module intervention rows must be unique")
    base_by_entity = {row["entity_id"]: row for row in base_rows}
    for entity_id in base_ids:
        context = base_by_entity[entity_id]["sampling_context"]
        observed = {
            module
            for current_entity, module in by_entity_module
            if current_entity == entity_id
        }
        if observed != set(CONTEXT_MODULES[context]):
            raise DonorContextFactorialError(
                f"{entity_id} lacks its expression-context module pair: {observed}"
            )
    return base_rows, intervention_rows, manifest


def _build_prompt(genes: str, form: str) -> str:
    spec = FORMS[form]
    return PROMPT.format(
        genes=genes,
        first=spec["first"],
        second=spec["second"],
        target=spec["target"],
    )


def _plan(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    model: str,
) -> tuple[list[dict[str, Any]], str]:
    if model != EXPECTED_MODEL:
        raise DonorContextFactorialError(
            f"model differs from frozen preregistration: {model}"
        )
    logical_items: list[dict[str, Any]] = []
    for row in base_rows:
        for form in FORMS:
            prompt = _build_prompt(row["original_sentence"], form)
            logical_items.append(
                {
                    "entity_id": row["entity_id"],
                    "module": "none",
                    "condition": "unmasked",
                    "form": form,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    for row in intervention_rows:
        for condition, column in (
            ("module_mask", "module_mask_sentence"),
            ("neutral_mask", "control_mask_sentence"),
        ):
            for form in FORMS:
                prompt = _build_prompt(row[column], form)
                logical_items.append(
                    {
                        "entity_id": row["entity_id"],
                        "module": row["module"],
                        "condition": condition,
                        "form": form,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
    if len(logical_items) != 1120:
        raise DonorContextFactorialError(
            f"expected 1120 logical observations, observed {len(logical_items)}"
        )

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for logical in logical_items:
        key = (logical["entity_id"], logical["form"], logical["prompt"])
        request = grouped.setdefault(
            key,
            {
                "entity_id": logical["entity_id"],
                "form": logical["form"],
                "prompt": logical["prompt"],
                "prompt_sha256": logical["prompt_sha256"],
                "assignments": [],
            },
        )
        request["assignments"].append(
            {
                "module": logical["module"],
                "condition": logical["condition"],
            }
        )

    items = list(grouped.values())
    for item in items:
        item["assignments"] = sorted(
            item["assignments"],
            key=lambda assignment: (
                assignment["module"],
                assignment["condition"],
            ),
        )
        if len(item["assignments"]) != len(
            {
                (assignment["module"], assignment["condition"])
                for assignment in item["assignments"]
            }
        ):
            raise DonorContextFactorialError("a request contains duplicate assignments")
        request_body = json.dumps(
            {
                "entity_id": item["entity_id"],
                "form": item["form"],
                "prompt_sha256": item["prompt_sha256"],
                "assignments": item["assignments"],
            },
            sort_keys=True,
        )
        item["request_id"] = hashlib.sha256(request_body.encode()).hexdigest()

    if (
        len(items) != 1120
        or sum(len(item["assignments"]) for item in items) != 1120
        or any(len(item["assignments"]) != 1 for item in items)
    ):
        raise DonorContextFactorialError(
            f"frozen unique-request structure changed: requests={len(items)}"
        )

    generator = np.random.default_rng(_seed(model, "call-order"))
    generator.shuffle(items)
    for index, item in enumerate(items):
        item["planned_index"] = index
    body = "\n".join(
        json.dumps(
            {
                key: item[key]
                for key in (
                    "planned_index",
                    "request_id",
                    "entity_id",
                    "form",
                    "prompt_sha256",
                    "assignments",
                )
            },
            sort_keys=True,
        )
        for item in items
    )
    plan_sha256 = hashlib.sha256((body + "\n").encode()).hexdigest()
    if plan_sha256 != EXPECTED_PLAN_SHA256:
        raise DonorContextFactorialError(
            f"call plan differs from the frozen hash: {plan_sha256}"
        )
    return items, plan_sha256


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        request_id = record["request_id"]
        if request_id in records:
            raise DonorContextFactorialError(
                f"duplicate checkpoint request_id: {request_id}"
            )
        records[request_id] = record
    return records


def _call_with_retry(model: str, prompt: str, attempts: int = 5) -> str:
    for attempt in range(attempts):
        try:
            return complete(model, prompt, system=SYSTEM_PROMPT)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _metadata_maps(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
]:
    return (
        {row["entity_id"]: row for row in base_rows},
        {(row["entity_id"], row["module"]): row for row in intervention_rows},
    )


def _assignment_metadata(
    item: dict[str, Any],
    intervention_by_key: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    assignments = []
    for assignment in item["assignments"]:
        module = assignment["module"]
        intervention = (
            None
            if module == "none"
            else intervention_by_key[(item["entity_id"], module)]
        )
        assignments.append(
            {
                **assignment,
                "target_genes": (
                    "" if intervention is None else intervention["target_genes"]
                ),
                "control_genes": (
                    "" if intervention is None else intervention["control_genes"]
                ),
            }
        )
    return assignments


def _execute_item(
    item: dict[str, Any],
    model: str,
    base_by_entity: dict[str, dict[str, str]],
    intervention_by_key: dict[tuple[str, str], dict[str, str]],
    data_sha256: str,
    manifest_sha256: str,
    prereg_sha256: str,
    plan_sha256: str,
    execution_code_sha256: str,
) -> dict[str, Any]:
    base = base_by_entity[item["entity_id"]]
    assignments = _assignment_metadata(item, intervention_by_key)
    started = datetime.now(UTC)
    raw_output = _call_with_retry(model, item["prompt"])
    finished = datetime.now(UTC)
    probability, parsed = parse_prob_with_status(raw_output)
    target = FORMS[item["form"]]["target"]
    aligned_probability_a = probability if target == CLASS_A else 1.0 - probability
    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "entity_id": item["entity_id"],
        "cell_barcode": base["cell_barcode"],
        "donor_id": base["donor_id"],
        "sampling_context": base["sampling_context"],
        "technical_group": base["technical_group"],
        "reference_annotation": base["reference_annotation"],
        "label_a": int(base["label_a"]),
        "request_id": item["request_id"],
        "assignments": assignments,
        "logical_observation_count": len(assignments),
        "form": item["form"],
        "order": item["form"][:2],
        "queried_target": "A" if target == CLASS_A else "B",
        "reported_probability": probability,
        "aligned_probability_a": aligned_probability_a,
        "parsed": parsed,
        "raw_output": raw_output,
        "planned_index": item["planned_index"],
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "prompt_sha256": item["prompt_sha256"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_csv_sha256": data_sha256,
        "input_manifest_sha256": manifest_sha256,
        "preregistration_sha256": prereg_sha256,
        "call_plan_sha256": plan_sha256,
        "execution_code_sha256": execution_code_sha256,
    }


def _validate_checkpoint_record(
    record: dict[str, Any],
    planned: dict[str, Any],
    *,
    model: str,
    base: dict[str, str],
    intervention_by_key: dict[tuple[str, str], dict[str, str]],
    data_sha256: str,
    manifest_sha256: str,
    prereg_sha256: str,
    plan_sha256: str,
    execution_code_sha256: str,
) -> None:
    expected_assignments = _assignment_metadata(planned, intervention_by_key)
    target = FORMS[planned["form"]]["target"]
    expected_target = "A" if target == CLASS_A else "B"
    probability, parsed = parse_prob_with_status(record.get("raw_output"))
    expected_aligned = probability if target == CLASS_A else 1.0 - probability
    exact_checks = {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "entity_id": planned["entity_id"],
        "cell_barcode": base["cell_barcode"],
        "donor_id": base["donor_id"],
        "sampling_context": base["sampling_context"],
        "technical_group": base["technical_group"],
        "reference_annotation": base["reference_annotation"],
        "label_a": int(base["label_a"]),
        "request_id": planned["request_id"],
        "assignments": expected_assignments,
        "logical_observation_count": len(expected_assignments),
        "form": planned["form"],
        "order": planned["form"][:2],
        "queried_target": expected_target,
        "parsed": parsed,
        "planned_index": planned["planned_index"],
        "prompt_sha256": planned["prompt_sha256"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_csv_sha256": data_sha256,
        "input_manifest_sha256": manifest_sha256,
        "preregistration_sha256": prereg_sha256,
        "call_plan_sha256": plan_sha256,
        "execution_code_sha256": execution_code_sha256,
    }
    for field, expected in exact_checks.items():
        if record.get(field) != expected:
            raise DonorContextFactorialError(
                f"checkpoint field mismatch for {planned['request_id']}: {field}"
            )
    for field, expected in (
        ("reported_probability", probability),
        ("aligned_probability_a", expected_aligned),
    ):
        try:
            observed = float(record[field])
        except (KeyError, TypeError, ValueError) as error:
            raise DonorContextFactorialError(
                f"checkpoint field is not numeric for "
                f"{planned['request_id']}: {field}"
            ) from error
        if not np.isclose(observed, expected, rtol=0.0, atol=1e-15):
            raise DonorContextFactorialError(
                f"checkpoint probability mismatch for "
                f"{planned['request_id']}: {field}"
            )
    try:
        started = datetime.fromisoformat(record["started_at_utc"])
        finished = datetime.fromisoformat(record["finished_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise DonorContextFactorialError(
            f"checkpoint timestamp invalid for {planned['request_id']}"
        ) from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise DonorContextFactorialError(
            f"checkpoint timestamp order invalid for {planned['request_id']}"
        )


def run(
    model: str,
    workers: int,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> Path:
    data_sha256 = _sha256(DATA_PATH)
    manifest_sha256 = _sha256(MANIFEST_PATH)
    prereg_sha256 = _sha256(PREREG_PATH)
    execution_code_sha256 = _sha256(Path(__file__))
    plan, plan_sha256 = _plan(base_rows, intervention_rows, model)
    base_by_entity, intervention_by_key = _metadata_maps(
        base_rows,
        intervention_rows,
    )
    path = _raw_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_checkpoint(path)
    plan_by_id = {item["request_id"]: item for item in plan}
    if set(existing) - set(plan_by_id):
        raise DonorContextFactorialError("checkpoint contains keys outside the frozen plan")
    for request_id, record in existing.items():
        planned = plan_by_id[request_id]
        _validate_checkpoint_record(
            record,
            planned,
            model=model,
            base=base_by_entity[planned["entity_id"]],
            intervention_by_key=intervention_by_key,
            data_sha256=data_sha256,
            manifest_sha256=manifest_sha256,
            prereg_sha256=prereg_sha256,
            plan_sha256=plan_sha256,
            execution_code_sha256=execution_code_sha256,
        )
    pending = [item for item in plan if item["request_id"] not in existing]
    if not pending:
        print(f"checkpoint already complete: {len(existing)}/{len(plan)}", flush=True)
        return path

    failures: list[tuple[dict[str, Any], Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_item,
                item,
                model,
                base_by_entity,
                intervention_by_key,
                data_sha256,
                manifest_sha256,
                prereg_sha256,
                plan_sha256,
                execution_code_sha256,
            ): item
            for item in pending
        }
        completed = len(existing)
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: PERF203
                failures.append((item, error))
                print(
                    f"FAILED {item['planned_index']} {item['entity_id']} "
                    f"{item['form']} {item['assignments']}: {error}",
                    flush=True,
                )
                continue
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(plan):
                print(
                    f"{model} donor-context checkpoint "
                    f"{completed}/{len(plan)} parsed={record['parsed']}",
                    flush=True,
                )
    if failures:
        first_item, first_error = failures[0]
        raise RuntimeError(
            f"{len(failures)} calls failed; first failure "
            f"{first_item['entity_id']} {first_item['form']} "
            f"{first_item['assignments']}: {first_error}"
        )
    return path


def _require_complete_parsing(parse_flags: list[bool]) -> None:
    if not parse_flags or not all(parse_flags):
        parsed = sum(parse_flags)
        raise DonorContextFactorialError(
            "confirmatory analysis requires 100% exact-output parsing: "
            f"parsed={parsed}/{len(parse_flags)}"
        )


def _direction_holds(value: float, direction: str) -> bool:
    if direction == "greater":
        return value > 0.0
    if direction == "less":
        return value < 0.0
    raise DonorContextFactorialError(f"unsupported direction: {direction}")


def _t_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) != EXPECTED_DONORS or not np.isfinite(values).all():
        raise DonorContextFactorialError(
            f"donor inference requires eight finite effects, observed {len(values)}"
        )
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    if standard_error == 0.0:
        statistic = (
            0.0 if mean == 0.0 else float(np.copysign(np.inf, mean))
        )
    else:
        statistic = mean / standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "statistic": statistic,
        "degrees_freedom": len(values) - 1,
        "ci95_lower": mean - critical * standard_error,
        "ci95_upper": mean + critical * standard_error,
        "two_sided_p_value": (
            1.0
            if statistic == 0.0
            else float(2.0 * student_t.sf(abs(statistic), len(values) - 1))
        ),
    }


def _exact_sign_flip_p(values: np.ndarray, direction: str) -> float:
    observed = float(values.mean())
    permuted = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    tolerance = 1e-15
    if direction == "greater":
        return float(np.mean(permuted >= observed - tolerance))
    if direction == "less":
        return float(np.mean(permuted <= observed + tolerance))
    raise DonorContextFactorialError(f"unsupported direction: {direction}")


def _directional_donor_summary(
    donor_ids: list[str],
    values: np.ndarray,
    direction: str,
) -> dict[str, Any]:
    summary = _t_summary(values)
    statistic = float(summary["statistic"])
    one_sided_p = (
        float(student_t.sf(statistic, EXPECTED_DONORS - 1))
        if direction == "greater"
        else float(student_t.cdf(statistic, EXPECTED_DONORS - 1))
    )
    exact_p = _exact_sign_flip_p(values, direction)
    signed_count = sum(_direction_holds(float(value), direction) for value in values)
    lodo = np.asarray(
        [float(np.delete(values, index).mean()) for index in range(len(values))]
    )
    ci_direction = (
        summary["ci95_lower"] > 0.0
        if direction == "greater"
        else summary["ci95_upper"] < 0.0
    )
    output = {
        **summary,
        "expected_direction": direction,
        "one_sided_student_t_p_value": one_sided_p,
        "exact_one_sided_rademacher_p_value": exact_p,
        "donor_effects": {
            donor: float(value)
            for donor, value in zip(donor_ids, values, strict=True)
        },
        "donors_with_expected_sign": int(signed_count),
        "leave_one_donor_out_means": {
            donor: float(value)
            for donor, value in zip(donor_ids, lodo, strict=True)
        },
        "ci_direction_pass": bool(ci_direction),
        "student_p_pass": bool(one_sided_p < 0.05),
        "exact_sign_flip_pass": bool(exact_p < 0.05),
        "donor_sign_pass": bool(signed_count >= 7),
        "lodo_direction_pass": bool(
            all(_direction_holds(float(value), direction) for value in lodo)
        ),
    }
    output["component_pass"] = bool(
        output["ci_direction_pass"]
        and output["exact_sign_flip_pass"]
        and output["donor_sign_pass"]
        and output["lodo_direction_pass"]
    )
    return output


def _donor_vector(
    cell_values: dict[str, np.ndarray | float],
    base_by_entity: dict[str, dict[str, str]],
    context: str,
    donors: list[str],
) -> np.ndarray:
    donor_values: dict[str, list[np.ndarray | float]] = {donor: [] for donor in donors}
    for entity_id, value in cell_values.items():
        base = base_by_entity[entity_id]
        if base["sampling_context"] == context:
            donor_values[base["donor_id"]].append(value)
    if any(not values for values in donor_values.values()):
        raise DonorContextFactorialError(
            f"incomplete donor support in expression context {context}"
        )
    return np.asarray(
        [np.asarray(donor_values[donor], dtype=float).mean(axis=0) for donor in donors],
        dtype=float,
    )


def _component_analysis(
    *,
    context: str,
    module: str,
    donors: list[str],
    base_by_entity: dict[str, dict[str, str]],
    effect_by_key: dict[tuple[str, str], np.ndarray],
    absolute_by_key: dict[tuple[str, str], np.ndarray],
    sham_by_key: dict[tuple[str, str], np.ndarray],
    selected_entities: set[str] | None = None,
) -> dict[str, Any]:
    def select(source: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
        return {
            entity_id: values
            for (entity_id, current_module), values in source.items()
            if current_module == module
            and base_by_entity[entity_id]["sampling_context"] == context
            and (selected_entities is None or entity_id in selected_entities)
        }

    effect_cells = select(effect_by_key)
    absolute_cells = select(absolute_by_key)
    sham_cells = select(sham_by_key)
    effect_donor_forms = _donor_vector(
        effect_cells,
        base_by_entity,
        context,
        donors,
    )
    absolute_donor_forms = _donor_vector(
        absolute_cells,
        base_by_entity,
        context,
        donors,
    )
    sham_donor_forms = _donor_vector(
        sham_cells,
        base_by_entity,
        context,
        donors,
    )
    direction = EXPECTED_DIRECTION[(context, module)]
    effect = _directional_donor_summary(
        donors,
        effect_donor_forms.mean(axis=1),
        direction,
    )
    form_summaries = {
        form: _t_summary(effect_donor_forms[:, index])
        for index, form in enumerate(FORMS)
    }
    all_forms_expected = all(
        _direction_holds(record["mean"], direction)
        for record in form_summaries.values()
    )
    order = _t_summary(
        effect_donor_forms[:, [0, 1]].mean(axis=1)
        - effect_donor_forms[:, [2, 3]].mean(axis=1)
    )
    target = _t_summary(
        effect_donor_forms[:, [0, 2]].mean(axis=1)
        - effect_donor_forms[:, [1, 3]].mean(axis=1)
    )
    interactions_equivalent = bool(
        order["ci95_lower"] > -EQUIVALENCE_MARGIN
        and order["ci95_upper"] < EQUIVALENCE_MARGIN
        and target["ci95_lower"] > -EQUIVALENCE_MARGIN
        and target["ci95_upper"] < EQUIVALENCE_MARGIN
    )
    sham = _t_summary(sham_donor_forms.mean(axis=1))
    sham["equivalent_within_margin"] = bool(
        sham["ci95_lower"] > -EQUIVALENCE_MARGIN
        and sham["ci95_upper"] < EQUIVALENCE_MARGIN
    )
    return {
        "context": context,
        "module": module,
        "n_cells": len(effect_cells),
        "matched_neutral_minus_module_mask": effect,
        "unmasked_minus_module_mask": _t_summary(
            absolute_donor_forms.mean(axis=1)
        ),
        "unmasked_minus_neutral_mask": sham,
        "form_effects": form_summaries,
        "prompt_factor_interactions": {
            "order_ab_minus_ba": order,
            "queried_target_a_minus_b": target,
            "all_forms_expected_direction": bool(all_forms_expected),
            "interactions_equivalent": interactions_equivalent,
            "prompt_robust": bool(all_forms_expected and interactions_equivalent),
        },
    }


def _difference_summary(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    base_by_entity: dict[str, dict[str, str]],
    context: str,
    donors: list[str],
) -> dict[str, Any]:
    left_donor = _donor_vector(left, base_by_entity, context, donors).mean(axis=1)
    right_donor = _donor_vector(right, base_by_entity, context, donors).mean(axis=1)
    values = left_donor - right_donor
    return {
        **_t_summary(values),
        "donor_effects": {
            donor: float(value)
            for donor, value in zip(donors, values, strict=True)
        },
    }


def _strict_entities(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> set[str]:
    by_entity: dict[str, list[dict[str, str]]] = {
        row["entity_id"]: [] for row in base_rows
    }
    for row in intervention_rows:
        by_entity[row["entity_id"]].append(row)
    return {
        entity_id
        for entity_id, rows in by_entity.items()
        if all(
            float(row["control_rank_distance"]) <= STRICT_MAX_RANK_DISTANCE
            and float(row["control_expression_distance"])
            <= STRICT_MAX_EXPRESSION_DISTANCE
            for row in rows
        )
    }


def analyze(
    model: str,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    raw_path = _raw_path(model)
    checkpoint = _read_checkpoint(raw_path)
    plan, plan_sha256 = _plan(base_rows, intervention_rows, model)
    plan_by_id = {item["request_id"]: item for item in plan}
    if set(checkpoint) != set(plan_by_id):
        raise DonorContextFactorialError(
            f"checkpoint incomplete: expected {len(plan_by_id)}, "
            f"observed {len(checkpoint)}"
        )

    data_sha256 = _sha256(DATA_PATH)
    manifest_sha256 = _sha256(MANIFEST_PATH)
    prereg_sha256 = _sha256(PREREG_PATH)
    execution_code_sha256 = _sha256(Path(__file__))
    base_by_entity, intervention_by_key = _metadata_maps(
        base_rows,
        intervention_rows,
    )
    donors = sorted(
        {row["donor_id"] for row in base_rows},
        key=lambda value: int(value),
    )
    unmasked = {
        row["entity_id"]: np.full(len(FORMS), np.nan) for row in base_rows
    }
    module_probabilities = {
        key: np.full(len(FORMS), np.nan) for key in intervention_by_key
    }
    neutral_probabilities = {
        key: np.full(len(FORMS), np.nan) for key in intervention_by_key
    }
    parse_flags: list[bool] = []
    for request_id, record in checkpoint.items():
        planned = plan_by_id[request_id]
        _validate_checkpoint_record(
            record,
            planned,
            model=model,
            base=base_by_entity[planned["entity_id"]],
            intervention_by_key=intervention_by_key,
            data_sha256=data_sha256,
            manifest_sha256=manifest_sha256,
            prereg_sha256=prereg_sha256,
            plan_sha256=plan_sha256,
            execution_code_sha256=execution_code_sha256,
        )
        form_index = list(FORMS).index(record["form"])
        value = float(record["aligned_probability_a"])
        for assignment in record["assignments"]:
            key = (record["entity_id"], assignment["module"])
            condition = assignment["condition"]
            if condition == "unmasked":
                unmasked[record["entity_id"]][form_index] = value
            elif condition == "module_mask":
                module_probabilities[key][form_index] = value
            elif condition == "neutral_mask":
                neutral_probabilities[key][form_index] = value
            else:
                raise DonorContextFactorialError(
                    f"unknown condition: {condition}"
                )
        parse_flags.append(bool(record["parsed"]))
    _require_complete_parsing(parse_flags)
    matrices = [
        *unmasked.values(),
        *module_probabilities.values(),
        *neutral_probabilities.values(),
    ]
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise DonorContextFactorialError(
            "checkpoint assignments did not fill every probability vector"
        )

    effect_by_key = {
        key: neutral_probabilities[key] - module_probabilities[key]
        for key in intervention_by_key
    }
    absolute_by_key = {
        key: unmasked[key[0]] - module_probabilities[key]
        for key in intervention_by_key
    }
    sham_by_key = {
        key: unmasked[key[0]] - neutral_probabilities[key]
        for key in intervention_by_key
    }
    components = {
        f"{context}:{module}": _component_analysis(
            context=context,
            module=module,
            donors=donors,
            base_by_entity=base_by_entity,
            effect_by_key=effect_by_key,
            absolute_by_key=absolute_by_key,
            sham_by_key=sham_by_key,
        )
        for context, modules in CONTEXT_MODULES.items()
        for module in modules
    }
    t_component = components[f"{T_CONTEXT}:T_TCR_CD8"]
    t_cyt_component = components[f"{T_CONTEXT}:cytotoxic_effector"]
    receptor_component = components[
        f"{RECEPTOR_CONTEXT}:NK_receptor_identity"
    ]
    receptor_cyt_component = components[
        f"{RECEPTOR_CONTEXT}:cytotoxic_effector"
    ]

    def passed(component: dict[str, Any]) -> bool:
        return bool(
            component["matched_neutral_minus_module_mask"]["component_pass"]
        )

    gate_a = passed(t_component) and passed(t_cyt_component)
    gate_b_raw = passed(receptor_component) and passed(receptor_cyt_component)
    gate_b_confirmatory = gate_a and gate_b_raw

    t_cells = {
        entity_id: values
        for (entity_id, module), values in effect_by_key.items()
        if module == "T_TCR_CD8"
    }
    t_cyt_cells = {
        entity_id: values
        for (entity_id, module), values in effect_by_key.items()
        if module == "cytotoxic_effector"
        and base_by_entity[entity_id]["sampling_context"] == T_CONTEXT
    }
    receptor_cells = {
        entity_id: values
        for (entity_id, module), values in effect_by_key.items()
        if module == "NK_receptor_identity"
    }
    receptor_cyt_cells = {
        entity_id: values
        for (entity_id, module), values in effect_by_key.items()
        if module == "cytotoxic_effector"
        and base_by_entity[entity_id]["sampling_context"] == RECEPTOR_CONTEXT
    }
    separations = {
        "T_minus_cytotoxic_within_T_plus_cytotoxic": _difference_summary(
            t_cells,
            t_cyt_cells,
            base_by_entity,
            T_CONTEXT,
            donors,
        ),
        "NK_receptor_minus_cytotoxic_within_receptor_plus_cytotoxic": (
            _difference_summary(
                receptor_cells,
                receptor_cyt_cells,
                base_by_entity,
                RECEPTOR_CONTEXT,
                donors,
            )
        ),
    }
    t_cyt_donor = _donor_vector(
        t_cyt_cells,
        base_by_entity,
        T_CONTEXT,
        donors,
    ).mean(axis=1)
    receptor_cyt_donor = _donor_vector(
        receptor_cyt_cells,
        base_by_entity,
        RECEPTOR_CONTEXT,
        donors,
    ).mean(axis=1)
    exploratory_cross_context = {
        **_t_summary(t_cyt_donor - receptor_cyt_donor),
        "contrast": (
            "cytotoxic effect in T_plus_cytotoxic minus cytotoxic effect "
            "in NK_receptor_plus_cytotoxic"
        ),
        "boundary": (
            "exploratory between-cohort contrast; context and sampled cells "
            "change together"
        ),
    }

    strict_entities = _strict_entities(base_rows, intervention_rows)
    strict_context_donor_counts = Counter(
        (row["donor_id"], row["sampling_context"])
        for row in base_rows
        if row["entity_id"] in strict_entities
    )
    if (
        len(strict_context_donor_counts) != EXPECTED_DONORS * len(CONTEXT_MODULES)
        or min(strict_context_donor_counts.values()) < 2
    ):
        raise DonorContextFactorialError(
            "strict matching sensitivity requires at least two cells "
            "per donor and expression context"
        )
    strict_components = {
        key: _component_analysis(
            context=component["context"],
            module=component["module"],
            donors=donors,
            base_by_entity=base_by_entity,
            effect_by_key=effect_by_key,
            absolute_by_key=absolute_by_key,
            sham_by_key=sham_by_key,
            selected_entities=strict_entities,
        )
        for key, component in components.items()
    }

    target_gene_results = []
    for (context, module), direction in EXPECTED_DIRECTION.items():
        genes: dict[str, dict[str, list[float]]] = {}
        for (entity_id, current_module), values in effect_by_key.items():
            base = base_by_entity[entity_id]
            if (
                current_module != module
                or base["sampling_context"] != context
            ):
                continue
            gene = intervention_by_key[(entity_id, module)]["target_genes"]
            genes.setdefault(gene, {}).setdefault(base["donor_id"], []).append(
                float(values.mean())
            )
        for gene, donor_values in sorted(genes.items()):
            if len(donor_values) < 6:
                continue
            donor_means = np.asarray(
                [np.mean(values) for values in donor_values.values()],
                dtype=float,
            )
            target_gene_results.append(
                {
                    "context": context,
                    "module": module,
                    "gene": gene,
                    "expected_direction": direction,
                    "n_donors": len(donor_values),
                    "n_cells": sum(len(values) for values in donor_values.values()),
                    "mean_effect_on_p_cd8": float(donor_means.mean()),
                    "boundary": (
                        "expression-selected descriptive localization; "
                        "not individual-gene causal attribution"
                    ),
                }
            )

    annotation_results = []
    for key, component in components.items():
        context, module = key.split(":", maxsplit=1)
        by_label: dict[str, list[float]] = {}
        for (entity_id, current_module), values in effect_by_key.items():
            base = base_by_entity[entity_id]
            if (
                current_module == module
                and base["sampling_context"] == context
            ):
                by_label.setdefault(
                    base["reference_annotation"],
                    [],
                ).append(float(values.mean()))
        annotation_results.append(
            {
                "context": context,
                "module": module,
                "labels": {
                    label: {
                        "n_cells": len(values),
                        "mean_effect_on_p_cd8": float(np.mean(values)),
                    }
                    for label, values in sorted(by_label.items())
                },
            }
        )

    all_records = list(checkpoint.values())
    prompt_robust = all(
        component["prompt_factor_interactions"]["prompt_robust"]
        for component in components.values()
    )
    sham_equivalent = all(
        component["unmasked_minus_neutral_mask"]["equivalent_within_margin"]
        for component in components.values()
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "claim_scope": manifest["claim_scope"],
        "n_cells": len(base_rows),
        "n_donors": len(donors),
        "donors": donors,
        "context_counts": dict(Counter(row["sampling_context"] for row in base_rows)),
        "deposited_label_counts_descriptive_only": {
            "CD8": sum(int(row["label_a"]) == 1 for row in base_rows),
            "NK": sum(int(row["label_a"]) == 0 for row in base_rows),
        },
        "model_calls": len(all_records),
        "logical_condition_form_observations": sum(
            int(record["logical_observation_count"]) for record in all_records
        ),
        "parse_rate": float(np.mean(parse_flags)),
        "parse_policy": (
            "abort confirmatory analysis unless all 1120 responses parse exactly"
        ),
        "run_started_at_utc": min(
            record["started_at_utc"] for record in all_records
        ),
        "run_finished_at_utc": max(
            record["finished_at_utc"] for record in all_records
        ),
        "inference_unit": "donor",
        "component_effect_definition": (
            "matched-neutral mask P(CD8) minus module mask P(CD8), "
            "averaged forms then cells then unweighted donors"
        ),
        "components": components,
        "hierarchical_gates": {
            "gate_a_T_plus_cytotoxic": {
                "pass": gate_a,
                "exact_iut_p_value": max(
                    t_component["matched_neutral_minus_module_mask"][
                        "exact_one_sided_rademacher_p_value"
                    ],
                    t_cyt_component["matched_neutral_minus_module_mask"][
                        "exact_one_sided_rademacher_p_value"
                    ],
                ),
            },
            "gate_b_receptor_plus_cytotoxic": {
                "raw_component_gate_pass": gate_b_raw,
                "confirmatory_pass_after_gate_a": gate_b_confirmatory,
                "status": (
                    "tested_confirmatorily"
                    if gate_a
                    else "hierarchically_blocked_by_gate_a"
                ),
                "exact_iut_p_value": max(
                    receptor_component["matched_neutral_minus_module_mask"][
                        "exact_one_sided_rademacher_p_value"
                    ],
                    receptor_cyt_component["matched_neutral_minus_module_mask"][
                        "exact_one_sided_rademacher_p_value"
                    ],
                ),
            },
            "full_hierarchical_gate_pass": gate_b_confirmatory,
        },
        "within_context_module_differences": separations,
        "prompt_robust": prompt_robust,
        "neutral_sham_equivalent_all_components": sham_equivalent,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "strict_matching_sensitivity": {
            "n_cells": len(strict_entities),
            "cells_by_donor_context": {
                f"{donor}:{context}": count
                for (donor, context), count in sorted(
                    strict_context_donor_counts.items(),
                    key=lambda item: (
                        int(item[0][0]),
                        item[0][1],
                    ),
                )
            },
            "criteria": {
                "maximum_rank_distance": STRICT_MAX_RANK_DISTANCE,
                "maximum_expression_distance": STRICT_MAX_EXPRESSION_DISTANCE,
            },
            "components": strict_components,
            "boundary": "sensitivity only; cannot rescue a primary gate",
        },
        "exploratory": {
            "cross_context_cytotoxic_difference": exploratory_cross_context,
            "deposited_label_summaries": annotation_results,
            "target_genes_with_at_least_six_donors": target_gene_results,
        },
        "interpretation_boundary": (
            "causal sensitivity to rendered marker-name deletion in one model "
            "and one eight-donor SLE control cohort; not annotation truth, a "
            "gene perturbation, pathway causality, a hidden-state activation "
            "route, latent-knowledge proof, or a physical law"
        ),
        "provenance": {
            "input_csv": str(DATA_PATH.relative_to(ROOT)),
            "input_csv_sha256": data_sha256,
            "input_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "input_manifest_sha256": manifest_sha256,
            "preregistration": str(PREREG_PATH.relative_to(ROOT)),
            "preregistration_sha256": prereg_sha256,
            "raw_checkpoint": str(raw_path.relative_to(ROOT)),
            "raw_checkpoint_sha256": _sha256(raw_path),
            "call_plan_sha256": plan_sha256,
            "execution_code_sha256": execution_code_sha256,
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "prompt_template_sha256": hashlib.sha256(
                PROMPT.encode()
            ).hexdigest(),
            "decode": DECODE,
            "component_test": (
                "unweighted donor mean, two-sided Student-t7 95% CI, "
                "one-sided Student-t7 p, and exact 2^8 Rademacher sign-flip p"
            ),
        },
    }


def _format_interval(
    record: dict[str, Any],
    *,
    key: str = "mean",
) -> str:
    return (
        f"{record[key]:+.3f} "
        f"[{record['ci95_lower']:+.3f}, {record['ci95_upper']:+.3f}]"
    )


def render_markdown(result: dict[str, Any]) -> str:
    gate_a = result["hierarchical_gates"]["gate_a_T_plus_cytotoxic"]
    gate_b = result["hierarchical_gates"]["gate_b_receptor_plus_cytotoxic"]
    labels = {
        "T_TCR_CD8": "TCR/CD8",
        "NK_receptor_identity": "NK receptor/identity",
        "cytotoxic_effector": "cytotoxic effector",
    }
    lines = [
        "# GSE96583 donor-aware expression-context module result",
        "",
        (
            f"Eight SLE donors, `{result['n_cells']}` cells, "
            f"`{result['model_calls']}` unique calls, parse rate "
            f"`{result['parse_rate']:.1%}`."
        ),
        "",
        "Positive effects push P(CD8) upward relative to a matched neutral",
        "deletion; negative effects push it toward NK. Intervals and tests use",
        "eight unweighted donor effects.",
        "",
        "| context | module | donor effect [95% CI] | t p | exact p | signs | pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for component in result["components"].values():
        effect = component["matched_neutral_minus_module_mask"]
        lines.append(
            f"| {component['context']} | {labels[component['module']]} "
            f"| {_format_interval(effect)} "
            f"| {effect['one_sided_student_t_p_value']:.5g} "
            f"| {effect['exact_one_sided_rademacher_p_value']:.5g} "
            f"| {effect['donors_with_expected_sign']}/8 "
            f"| {str(effect['component_pass']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                f"- Gate A (`T_plus_cytotoxic`) passed: "
                f"**{str(gate_a['pass']).lower()}**, exact IUT "
                f"p=`{gate_a['exact_iut_p_value']:.5g}`"
            ),
            (
                "- Gate B (`NK_receptor_plus_cytotoxic`) raw component gate: "
                f"**{str(gate_b['raw_component_gate_pass']).lower()}**; "
                "hierarchical confirmatory pass: "
                f"**{str(gate_b['confirmatory_pass_after_gate_a']).lower()}**"
            ),
            (
                f"- All prompt robustness checks passed: "
                f"**{str(result['prompt_robust']).lower()}**"
            ),
            (
                "- All unmasked-minus-neutral sham intervals lie within "
                f"±{result['equivalence_margin']:.2f}: "
                f"**{str(result['neutral_sham_equivalent_all_components']).lower()}**"
            ),
            "",
            "## Interpretation boundary",
            "",
            result["interpretation_boundary"] + ".",
            "",
            (
                "Deposited labels are descriptive only. Expression support "
                "selected the paired modules; the experiment does not estimate "
                "T-versus-receptor effects on the same cells."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyze a complete checkpoint without model calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the frozen plan without model calls",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise DonorContextFactorialError("--workers must be between 1 and 16")

    base_rows, intervention_rows, manifest = _load_inputs()
    plan, plan_sha256 = _plan(base_rows, intervention_rows, args.model)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": ANALYSIS_ID,
                    "model": args.model,
                    "cells": len(base_rows),
                    "donors": len({row["donor_id"] for row in base_rows}),
                    "context_counts": dict(
                        Counter(row["sampling_context"] for row in base_rows)
                    ),
                    "deposited_label_counts_descriptive_only": {
                        "CD8": sum(
                            int(row["label_a"]) == 1 for row in base_rows
                        ),
                        "NK": sum(
                            int(row["label_a"]) == 0 for row in base_rows
                        ),
                    },
                    "unique_api_calls": len(plan),
                    "logical_condition_form_observations": sum(
                        len(item["assignments"]) for item in plan
                    ),
                    "shared_response_reuses": sum(
                        len(item["assignments"]) - 1 for item in plan
                    ),
                    "input_csv_sha256": _sha256(DATA_PATH),
                    "input_manifest_sha256": _sha256(MANIFEST_PATH),
                    "preregistration_sha256": _sha256(PREREG_PATH),
                    "plan_sha256": plan_sha256,
                    "execution_code_sha256": _sha256(Path(__file__)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not args.analyze_only:
        run(args.model, args.workers, base_rows, intervention_rows)
    result = analyze(
        args.model,
        base_rows,
        intervention_rows,
        manifest,
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = _safe_model_name(args.model)
    json_path = RESULT_ROOT / f"{stem}.json"
    markdown_path = RESULT_ROOT / f"{stem}.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(json_path),
                "markdown_out": str(markdown_path),
                "hierarchical_gates": result["hierarchical_gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

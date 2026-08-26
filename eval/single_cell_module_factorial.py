"""Run the PBMC68k two-category ablation crossed with a prompt factorial.

The same common-support cells receive both module interventions, each paired
with its own within-cell matched-neutral deletion, plus an unmasked input.
Every input is crossed with answer order and queried target.  Reference labels
define the upstream CD8/NK task frame and stratified analysis, but do not
select common-support membership, a module, or a mask within that frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_grounding_eval import DECODE, complete, parse_prob_with_status  # noqa: E402

ANALYSIS_ID = "pbmc68k-cd8-nk-module-factorial-v1"
DATA_PATH = ROOT / "signal" / "single_cell" / "pbmc68k_cd8_nk_module_factorial.csv"
MANIFEST_PATH = ROOT / "signal" / "single_cell" / "pbmc68k_cd8_nk_module_factorial.manifest.json"
PREREG_PATH = ROOT / "docs" / "PBMC68K_MODULE_FACTORIAL_PREREG.md"
RESULT_ROOT = ROOT / "results" / "benchmark" / "single_cell" / "module_factorial"
EXPECTED_DATA_SHA256 = "fadd90b9aa1249c2691943287962a381329bcba8f91f0b392636ba80ce9f1d9b"
EXPECTED_MANIFEST_SHA256 = "b849e34b94f0004568ca289f1997f39b15f25eb991200b8b3d0f0c3e62869f79"
EXPECTED_PREREG_SHA256 = "a7f38447ceaae932b87492b15d667af02c21b006fedb7beb7469d7448b17e605"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
EXPECTED_PLAN_SHA256 = "25d083894f54b17ee0dca72a04b19a30f03e076403dde7f9ae05d05c7c42ae96"

CLASS_A = "CD8+ T cell"
CLASS_B = "NK cell"
MODULES = ("T_TCR_CD8", "cytotoxic_effector")
EXPECTED_DIRECTION = {
    "T_TCR_CD8": "greater",
    "cytotoxic_effector": "less",
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
N_BOOTSTRAP = 20_000
EQUIVALENCE_MARGIN = 0.03
STRICT_MAX_RANK_DISTANCE = 10
STRICT_MAX_EXPRESSION_DISTANCE = 1.0


class ModuleFactorialError(ValueError):
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
    if not PREREG_PATH.exists():
        raise ModuleFactorialError(f"preregistration missing: {PREREG_PATH}")
    prereg_sha256 = _sha256(PREREG_PATH)
    if prereg_sha256 != EXPECTED_PREREG_SHA256:
        raise ModuleFactorialError("preregistration differs from the frozen expected hash")
    data_sha256 = _sha256(DATA_PATH)
    manifest_sha256 = _sha256(MANIFEST_PATH)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ModuleFactorialError("factorial CSV differs from the frozen expected hash")
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ModuleFactorialError("factorial manifest differs from the frozen expected hash")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != ANALYSIS_ID:
        raise ModuleFactorialError("factorial manifest analysis_id mismatch")
    if manifest["artifacts"]["csv_sha256"] != data_sha256:
        raise ModuleFactorialError("manifest CSV hash mismatch")
    construction = manifest.get("construction", {})
    if (
        construction.get("logical_condition_form_observations") != 1300
        or construction.get("expected_unique_api_calls") != 1256
        or construction.get("shared_neutral_control_cells") != 11
    ):
        raise ModuleFactorialError("manifest request de-duplication contract changed")

    rows = list(csv.DictReader(DATA_PATH.open(newline="", encoding="utf-8")))
    base_rows = [row for row in rows if row["row_type"] == "base"]
    intervention_rows = [row for row in rows if row["row_type"] == "intervention"]
    if len(base_rows) != 65 or len(intervention_rows) != 130:
        raise ModuleFactorialError(
            f"frozen row counts changed: base={len(base_rows)}, intervention={len(intervention_rows)}"
        )
    base_ids = {row["entity_id"] for row in base_rows}
    if len(base_ids) != 65:
        raise ModuleFactorialError("base entity IDs are not unique")
    counts = Counter(int(row["label_a"]) for row in base_rows)
    if counts != Counter({1: 55, 0: 10}):
        raise ModuleFactorialError(f"frozen class counts changed: {dict(counts)}")
    by_entity = Counter(row["entity_id"] for row in intervention_rows)
    by_entity_module = Counter((row["entity_id"], row["module"]) for row in intervention_rows)
    if set(by_entity) != base_ids or set(by_entity.values()) != {2}:
        raise ModuleFactorialError("every common-support cell must have two interventions")
    if set(by_entity_module.values()) != {1}:
        raise ModuleFactorialError("entity-module intervention rows must be unique")
    for entity_id in base_ids:
        observed = {module for current_entity, module in by_entity_module if current_entity == entity_id}
        if observed != set(MODULES):
            raise ModuleFactorialError(f"{entity_id} lacks common module support: {observed}")
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
        raise ModuleFactorialError(f"model differs from frozen preregistration: {model}")
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
    if len(logical_items) != 1300:
        raise ModuleFactorialError(f"expected 1300 logical observations, observed {len(logical_items)}")

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for logical in logical_items:
        key = (
            logical["entity_id"],
            logical["form"],
            logical["prompt"],
        )
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
            {(assignment["module"], assignment["condition"]) for assignment in item["assignments"]}
        ):
            raise ModuleFactorialError("a request contains duplicate logical assignments")
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

    shared_requests = [item for item in items if len(item["assignments"]) == 2]
    expected_shared_assignments = {
        ("T_TCR_CD8", "neutral_mask"),
        ("cytotoxic_effector", "neutral_mask"),
    }
    if (
        len(items) != 1256
        or len(shared_requests) != 44
        or sum(len(item["assignments"]) for item in items) != 1300
        or any(
            {(assignment["module"], assignment["condition"]) for assignment in item["assignments"]}
            != expected_shared_assignments
            for item in shared_requests
        )
    ):
        raise ModuleFactorialError(
            f"frozen unique-request structure changed: requests={len(items)}, shared={len(shared_requests)}"
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
        raise ModuleFactorialError("call plan differs from the frozen preregistered hash")
    return items, plan_sha256


def _read_checkpoint(
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        request_id = record["request_id"]
        if request_id in records:
            raise ModuleFactorialError(f"duplicate checkpoint request_id: {request_id}")
        records[request_id] = record
    return records


def _call_with_retry(
    model: str,
    prompt: str,
    attempts: int = 5,
) -> str:
    for attempt in range(attempts):
        try:
            return complete(
                model,
                prompt,
                system=SYSTEM_PROMPT,
            )
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
        intervention = None if module == "none" else intervention_by_key[(item["entity_id"], module)]
        assignments.append(
            {
                **assignment,
                "target_genes": ("" if intervention is None else intervention["target_genes"]),
                "control_genes": ("" if intervention is None else intervention["control_genes"]),
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
    assignments = _assignment_metadata(
        item,
        intervention_by_key,
    )
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
    expected_assignments = _assignment_metadata(
        planned,
        intervention_by_key,
    )
    target = FORMS[planned["form"]]["target"]
    expected_target = "A" if target == CLASS_A else "B"
    probability, parsed = parse_prob_with_status(record.get("raw_output"))
    expected_aligned = probability if target == CLASS_A else 1.0 - probability
    exact_checks = {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "entity_id": planned["entity_id"],
        "cell_barcode": base["cell_barcode"],
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
            raise ModuleFactorialError(f"checkpoint field mismatch for {planned['request_id']}: {field}")
    for field, expected in (
        ("reported_probability", probability),
        ("aligned_probability_a", expected_aligned),
    ):
        try:
            observed = float(record[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ModuleFactorialError(
                f"checkpoint field is not numeric for {planned['request_id']}: {field}"
            ) from error
        if not np.isclose(
            observed,
            expected,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ModuleFactorialError(f"checkpoint probability mismatch for {planned['request_id']}: {field}")
    try:
        started = datetime.fromisoformat(record["started_at_utc"])
        finished = datetime.fromisoformat(record["finished_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModuleFactorialError(f"checkpoint timestamp invalid for {planned['request_id']}") from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise ModuleFactorialError(f"checkpoint timestamp order invalid for {planned['request_id']}")


def run(
    model: str,
    workers: int,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> Path:
    if not PREREG_PATH.exists():
        raise ModuleFactorialError(f"preregistration missing: {PREREG_PATH}")
    data_sha256 = _sha256(DATA_PATH)
    manifest_sha256 = _sha256(MANIFEST_PATH)
    prereg_sha256 = _sha256(PREREG_PATH)
    if prereg_sha256 != EXPECTED_PREREG_SHA256:
        raise ModuleFactorialError("preregistration differs from the frozen expected hash")
    execution_code_sha256 = _sha256(Path(__file__))
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        model,
    )
    base_by_entity, intervention_by_key = _metadata_maps(
        base_rows,
        intervention_rows,
    )
    path = _raw_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_checkpoint(path)
    plan_by_id = {item["request_id"]: item for item in plan}
    expected_keys = set(plan_by_id)
    if set(existing) - expected_keys:
        raise ModuleFactorialError("checkpoint contains keys outside the frozen plan")
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
        print(
            f"checkpoint already complete: {len(existing)}/{len(plan)}",
            flush=True,
        )
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
                    f"FAILED {item['planned_index']} {item['entity_id']} {item['form']} {item['assignments']}: {error}",
                    flush=True,
                )
                continue
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(plan):
                print(
                    f"{model} module-factorial checkpoint {completed}/{len(plan)} parsed={record['parsed']}",
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


def _equal_class_mean(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    return float(0.5 * (values[labels == 1].mean() + values[labels == 0].mean()))


def _interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _require_complete_parsing(
    parse_flags: list[bool],
) -> None:
    if not parse_flags or not all(parse_flags):
        parsed = sum(parse_flags)
        raise ModuleFactorialError(
            f"confirmatory analysis requires 100% exact-output parsing: parsed={parsed}/{len(parse_flags)}"
        )


def _equal_class_welch_test(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    alternative: str,
) -> dict[str, float]:
    positive = values[labels == 1]
    negative = values[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        raise ModuleFactorialError("Welch-Satterthwaite test requires at least two cells per class")
    effect = _equal_class_mean(values, labels)
    variance_terms = np.asarray(
        [
            np.var(positive, ddof=1) / len(positive),
            np.var(negative, ddof=1) / len(negative),
        ],
        dtype=float,
    )
    standard_error = float(0.5 * np.sqrt(variance_terms.sum()))
    denominator = float(variance_terms[0] ** 2 / (len(positive) - 1) + variance_terms[1] ** 2 / (len(negative) - 1))
    if standard_error == 0.0 or denominator == 0.0:
        statistic = 0.0 if effect == 0.0 else float(np.copysign(np.inf, effect))
        degrees_freedom = float("inf")
        if alternative == "greater":
            p_value = 0.0 if effect > 0.0 else 1.0 if effect < 0.0 else 0.5
        elif alternative == "less":
            p_value = 0.0 if effect < 0.0 else 1.0 if effect > 0.0 else 0.5
        else:
            raise ModuleFactorialError(f"unsupported alternative: {alternative}")
    else:
        statistic = effect / standard_error
        degrees_freedom = float(variance_terms.sum() ** 2 / denominator)
        if alternative == "greater":
            p_value = float(student_t.sf(statistic, degrees_freedom))
        elif alternative == "less":
            p_value = float(student_t.cdf(statistic, degrees_freedom))
        else:
            raise ModuleFactorialError(f"unsupported alternative: {alternative}")
    return {
        "effect": effect,
        "standard_error": standard_error,
        "statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
    }


def _bootstrap(
    module_effects: dict[str, np.ndarray],
    absolute_changes: dict[str, np.ndarray],
    neutral_changes: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    positive_draws = positive[
        generator.integers(
            0,
            len(positive),
            size=(N_BOOTSTRAP, len(positive)),
        )
    ]
    negative_draws = negative[
        generator.integers(
            0,
            len(negative),
            size=(N_BOOTSTRAP, len(negative)),
        )
    ]

    def draw_equal(values: np.ndarray) -> np.ndarray:
        return 0.5 * (values[positive_draws].mean(axis=1) + values[negative_draws].mean(axis=1))

    output: dict[str, Any] = {"modules": {}}
    module_draws = {}
    for module in MODULES:
        effects = module_effects[module]
        cell_effect = effects.mean(axis=1)
        equal_draws = draw_equal(cell_effect)
        module_draws[module] = equal_draws
        order_draws = draw_equal(effects[:, [0, 1]].mean(axis=1)) - draw_equal(effects[:, [2, 3]].mean(axis=1))
        target_draws = draw_equal(effects[:, [0, 2]].mean(axis=1)) - draw_equal(effects[:, [1, 3]].mean(axis=1))
        output["modules"][module] = {
            "equal_class_ci95": _interval(equal_draws),
            "cd8_ci95": _interval(cell_effect[positive_draws].mean(axis=1)),
            "nk_ci95": _interval(cell_effect[negative_draws].mean(axis=1)),
            "form_ci95": {form: _interval(draw_equal(effects[:, form_index])) for form_index, form in enumerate(FORMS)},
            "order_interaction_ci95": _interval(order_draws),
            "target_interaction_ci95": _interval(target_draws),
            "absolute_change_ci95": _interval(draw_equal(absolute_changes[module].mean(axis=1))),
            "neutral_change_ci95": _interval(draw_equal(neutral_changes[module].mean(axis=1))),
        }
    output["separation_ci95"] = _interval(module_draws["T_TCR_CD8"] - module_draws["cytotoxic_effector"])
    return output


def _strict_cells(
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
) -> np.ndarray:
    by_entity: dict[str, list[dict[str, str]]] = {row["entity_id"]: [] for row in base_rows}
    for row in intervention_rows:
        by_entity[row["entity_id"]].append(row)
    return np.asarray(
        [
            all(
                float(row["control_rank_distance"]) <= STRICT_MAX_RANK_DISTANCE
                and float(row["control_expression_distance"]) <= STRICT_MAX_EXPRESSION_DISTANCE
                for row in by_entity[base["entity_id"]]
            )
            for base in base_rows
        ],
        dtype=bool,
    )


def _condition_metrics(
    labels: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, float]:
    averaged = matrix.mean(axis=1)
    return {
        "orientation_averaged_auroc": float(roc_auc_score(labels, averaged)),
        "orientation_averaged_brier": float(brier_score_loss(labels, averaged)),
        "mean_aligned_probability_a": float(averaged.mean()),
    }


def analyze(
    model: str,
    base_rows: list[dict[str, str]],
    intervention_rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not PREREG_PATH.exists():
        raise ModuleFactorialError("current preregistration is missing")
    raw_path = _raw_path(model)
    checkpoint = _read_checkpoint(raw_path)
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        model,
    )
    plan_by_id = {item["request_id"]: item for item in plan}
    expected_keys = set(plan_by_id)
    if set(checkpoint) != expected_keys:
        raise ModuleFactorialError(f"checkpoint incomplete: expected {len(expected_keys)}, observed {len(checkpoint)}")

    data_sha256 = _sha256(DATA_PATH)
    manifest_sha256 = _sha256(MANIFEST_PATH)
    prereg_sha256 = _sha256(PREREG_PATH)
    if prereg_sha256 != EXPECTED_PREREG_SHA256:
        raise ModuleFactorialError("preregistration differs from the frozen expected hash")
    execution_code_sha256 = _sha256(Path(__file__))
    entity_to_index = {row["entity_id"]: index for index, row in enumerate(base_rows)}
    base_by_entity, intervention_by_key = _metadata_maps(
        base_rows,
        intervention_rows,
    )
    labels = np.asarray(
        [int(row["label_a"]) for row in base_rows],
        dtype=int,
    )
    shape = (len(base_rows), len(FORMS))
    unmasked = np.full(shape, np.nan)
    module_probabilities = {module: np.full(shape, np.nan) for module in MODULES}
    neutral_probabilities = {module: np.full(shape, np.nan) for module in MODULES}
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
        row_index = entity_to_index[record["entity_id"]]
        form_index = list(FORMS).index(record["form"])
        value = float(record["aligned_probability_a"])
        for assignment in record["assignments"]:
            module = assignment["module"]
            condition = assignment["condition"]
            if condition == "unmasked":
                unmasked[row_index, form_index] = value
            elif condition == "module_mask":
                module_probabilities[module][
                    row_index,
                    form_index,
                ] = value
            elif condition == "neutral_mask":
                neutral_probabilities[module][
                    row_index,
                    form_index,
                ] = value
            else:
                raise ModuleFactorialError(f"unknown condition: {condition}")
        parse_flags.append(bool(record["parsed"]))
    _require_complete_parsing(parse_flags)
    matrices = [
        unmasked,
        *module_probabilities.values(),
        *neutral_probabilities.values(),
    ]
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ModuleFactorialError("checkpoint assignments did not fill every probability matrix")

    module_effects = {module: neutral_probabilities[module] - module_probabilities[module] for module in MODULES}
    absolute_changes = {module: unmasked - module_probabilities[module] for module in MODULES}
    neutral_changes = {module: unmasked - neutral_probabilities[module] for module in MODULES}
    bootstrap = _bootstrap(
        module_effects,
        absolute_changes,
        neutral_changes,
        labels,
        seed=_seed(model, "bootstrap"),
    )

    module_results = {}
    for module in MODULES:
        effects = module_effects[module]
        cell_effect = effects.mean(axis=1)
        module_bootstrap = bootstrap["modules"][module]
        effect = _equal_class_mean(cell_effect, labels)
        welch_test = _equal_class_welch_test(
            cell_effect,
            labels,
            alternative=EXPECTED_DIRECTION[module],
        )
        cd8_effect = float(cell_effect[labels == 1].mean())
        nk_effect = float(cell_effect[labels == 0].mean())
        class_direction_consistent = (
            cd8_effect > 0 and nk_effect > 0
            if EXPECTED_DIRECTION[module] == "greater"
            else cd8_effect < 0 and nk_effect < 0
        )
        form_effects = {
            form: _equal_class_mean(
                effects[:, form_index],
                labels,
            )
            for form_index, form in enumerate(FORMS)
        }
        order_interaction = _equal_class_mean(
            effects[:, [0, 1]].mean(axis=1),
            labels,
        ) - _equal_class_mean(
            effects[:, [2, 3]].mean(axis=1),
            labels,
        )
        target_interaction = _equal_class_mean(
            effects[:, [0, 2]].mean(axis=1),
            labels,
        ) - _equal_class_mean(
            effects[:, [1, 3]].mean(axis=1),
            labels,
        )
        order_ci = module_bootstrap["order_interaction_ci95"]
        target_ci = module_bootstrap["target_interaction_ci95"]
        interactions_equivalent = (
            order_ci[0] > -EQUIVALENCE_MARGIN
            and order_ci[1] < EQUIVALENCE_MARGIN
            and target_ci[0] > -EQUIVALENCE_MARGIN
            and target_ci[1] < EQUIVALENCE_MARGIN
        )
        expected_forms = (
            all(value > 0 for value in form_effects.values())
            if EXPECTED_DIRECTION[module] == "greater"
            else all(value < 0 for value in form_effects.values())
        )
        module_results[module] = {
            "effect_on_p_cd8": effect,
            "ci95_lower": module_bootstrap["equal_class_ci95"][0],
            "ci95_upper": module_bootstrap["equal_class_ci95"][1],
            "expected_direction": EXPECTED_DIRECTION[module],
            "one_sided_welch_p_value": welch_test["p_value"],
            "welch_satterthwaite": {
                "standard_error": welch_test["standard_error"],
                "statistic": welch_test["statistic"],
                "degrees_freedom": welch_test["degrees_freedom"],
            },
            "class_direction_consistent": class_direction_consistent,
            "class_effects": {
                "CD8": {
                    "n": int((labels == 1).sum()),
                    "effect_on_p_cd8": cd8_effect,
                    "ci95_lower": module_bootstrap["cd8_ci95"][0],
                    "ci95_upper": module_bootstrap["cd8_ci95"][1],
                },
                "NK": {
                    "n": int((labels == 0).sum()),
                    "effect_on_p_cd8": nk_effect,
                    "ci95_lower": module_bootstrap["nk_ci95"][0],
                    "ci95_upper": module_bootstrap["nk_ci95"][1],
                },
            },
            "form_effects": {
                form: {
                    "effect_on_p_cd8": value,
                    "ci95_lower": module_bootstrap["form_ci95"][form][0],
                    "ci95_upper": module_bootstrap["form_ci95"][form][1],
                }
                for form, value in form_effects.items()
            },
            "prompt_factor_interactions": {
                "order_ab_minus_ba": {
                    "effect": order_interaction,
                    "ci95_lower": order_ci[0],
                    "ci95_upper": order_ci[1],
                },
                "queried_target_a_minus_b": {
                    "effect": target_interaction,
                    "ci95_lower": target_ci[0],
                    "ci95_upper": target_ci[1],
                },
                "interactions_equivalent": interactions_equivalent,
                "all_forms_expected_direction": expected_forms,
            },
            "unmasked_minus_module_mask": {
                "effect_on_p_cd8": _equal_class_mean(
                    absolute_changes[module].mean(axis=1),
                    labels,
                ),
                "ci95_lower": module_bootstrap["absolute_change_ci95"][0],
                "ci95_upper": module_bootstrap["absolute_change_ci95"][1],
            },
            "unmasked_minus_neutral_mask": {
                "effect_on_p_cd8": _equal_class_mean(
                    neutral_changes[module].mean(axis=1),
                    labels,
                ),
                "ci95_lower": module_bootstrap["neutral_change_ci95"][0],
                "ci95_upper": module_bootstrap["neutral_change_ci95"][1],
            },
        }

    t_result = module_results["T_TCR_CD8"]
    c_result = module_results["cytotoxic_effector"]
    primary_iut_pass = (
        t_result["effect_on_p_cd8"] > 0
        and c_result["effect_on_p_cd8"] < 0
        and t_result["one_sided_welch_p_value"] < 0.05
        and c_result["one_sided_welch_p_value"] < 0.05
    )
    class_direction_consistency_pass = t_result["class_direction_consistent"] and c_result["class_direction_consistent"]
    primary_gate_pass = primary_iut_pass and class_direction_consistency_pass
    separation = t_result["effect_on_p_cd8"] - c_result["effect_on_p_cd8"]

    strict_mask = _strict_cells(
        base_rows,
        intervention_rows,
    )
    strict_labels = labels[strict_mask]
    strict_result: dict[str, Any]
    if set(strict_labels.tolist()) == {0, 1}:
        strict_effects = {module: values[strict_mask] for module, values in module_effects.items()}
        strict_bootstrap = _bootstrap(
            strict_effects,
            {module: values[strict_mask] for module, values in absolute_changes.items()},
            {module: values[strict_mask] for module, values in neutral_changes.items()},
            strict_labels,
            seed=_seed(model, "strict-bootstrap"),
        )
        strict_module = {}
        for module in MODULES:
            effect = _equal_class_mean(
                strict_effects[module].mean(axis=1),
                strict_labels,
            )
            ci = strict_bootstrap["modules"][module]["equal_class_ci95"]
            strict_module[module] = {
                "effect_on_p_cd8": effect,
                "ci95_lower": ci[0],
                "ci95_upper": ci[1],
            }
        strict_result = {
            "n": int(strict_mask.sum()),
            "class_counts": {
                "CD8": int((strict_labels == 1).sum()),
                "NK": int((strict_labels == 0).sum()),
            },
            "modules": strict_module,
            "separation": (
                strict_module["T_TCR_CD8"]["effect_on_p_cd8"] - strict_module["cytotoxic_effector"]["effect_on_p_cd8"]
            ),
            "separation_ci95_lower": strict_bootstrap["separation_ci95"][0],
            "separation_ci95_upper": strict_bootstrap["separation_ci95"][1],
        }
    else:
        strict_result = {
            "n": int(strict_mask.sum()),
            "not_estimable": True,
        }

    condition_metrics = {
        "unmasked": _condition_metrics(labels, unmasked),
    }
    for module in MODULES:
        condition_metrics[f"{module}:module_mask"] = _condition_metrics(
            labels,
            module_probabilities[module],
        )
        condition_metrics[f"{module}:neutral_mask"] = _condition_metrics(
            labels,
            neutral_probabilities[module],
        )

    annotations = np.asarray(
        [row["reference_annotation"] for row in base_rows],
        dtype=object,
    )
    annotation_results = []
    for annotation in sorted(set(annotations.tolist())):
        selected = annotations == annotation
        annotation_results.append(
            {
                "reference_annotation": annotation,
                "n": int(selected.sum()),
                "module_effects_on_p_cd8": {
                    module: float(module_effects[module][selected].mean(axis=1).mean()) for module in MODULES
                },
            }
        )

    intervention_by_key = {(row["entity_id"], row["module"]): row for row in intervention_rows}
    target_gene_results = {}
    for module in MODULES:
        gene_values: dict[str, list[float]] = {}
        cell_effect = module_effects[module].mean(axis=1)
        for index, base in enumerate(base_rows):
            gene = intervention_by_key[(base["entity_id"], module)]["target_genes"]
            gene_values.setdefault(gene, []).append(float(cell_effect[index]))
        target_gene_results[module] = [
            {
                "gene": gene,
                "n": len(values),
                "mean_effect_on_p_cd8": float(np.mean(values)),
            }
            for gene, values in sorted(
                gene_values.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ]

    all_records = list(checkpoint.values())
    started = min(record["started_at_utc"] for record in all_records)
    finished = max(record["finished_at_utc"] for record in all_records)
    prompt_robust = all(
        result["prompt_factor_interactions"]["interactions_equivalent"]
        and result["prompt_factor_interactions"]["all_forms_expected_direction"]
        for result in module_results.values()
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "claim_scope": manifest["claim_scope"],
        "n_cells": len(base_rows),
        "class_counts": {
            "CD8": int((labels == 1).sum()),
            "NK": int((labels == 0).sum()),
        },
        "model_calls": len(all_records),
        "logical_condition_form_observations": sum(int(record["logical_observation_count"]) for record in all_records),
        "shared_neutral_response_reuses": sum(int(record["logical_observation_count"]) - 1 for record in all_records),
        "parse_rate": float(np.mean(parse_flags)),
        "parse_policy": "abort_confirmatory_analysis_unless_all_unique_responses_parse",
        "run_started_at_utc": started,
        "run_finished_at_utc": finished,
        "primary_estimand": (
            "intersection-union gate: equal-class matched-neutral-adjusted "
            "T_TCR_CD8 effect on P(CD8) > 0 and cytotoxic_effector effect "
            "on P(CD8) < 0, averaged over four prompt forms; both class-specific "
            "means must also have the expected sign"
        ),
        "primary_iut_pass": primary_iut_pass,
        "primary_iut_p_value": max(
            t_result["one_sided_welch_p_value"],
            c_result["one_sided_welch_p_value"],
        ),
        "class_direction_consistency_pass": class_direction_consistency_pass,
        "primary_gate_pass": primary_gate_pass,
        "module_effects": module_results,
        "module_separation": {
            "effect": separation,
            "ci95_lower": bootstrap["separation_ci95"][0],
            "ci95_upper": bootstrap["separation_ci95"][1],
        },
        "prompt_robust": prompt_robust,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "strict_matching_sensitivity": strict_result,
        "condition_metrics": condition_metrics,
        "exploratory_localization": {
            "reference_annotations": annotation_results,
            "masked_target_genes": target_gene_results,
            "boundary": (
                "post-hoc descriptions; co-occurrence and cell-state "
                "selection prevent individual-gene causal attribution"
            ),
        },
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
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "prompt_template_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "decode": DECODE,
            "bootstrap_replicates": N_BOOTSTRAP,
            "component_test": ("one-sided Welch-Satterthwaite test of the equal-class linear contrast"),
        },
    }


def _format_interval(
    record: dict[str, Any],
    *,
    key: str = "effect_on_p_cd8",
) -> str:
    return f"{record[key]:+.3f} [{record['ci95_lower']:+.3f}, {record['ci95_upper']:+.3f}]"


def render_markdown(result: dict[str, Any]) -> str:
    t_result = result["module_effects"]["T_TCR_CD8"]
    c_result = result["module_effects"]["cytotoxic_effector"]
    separation = result["module_separation"]
    strict = result["strict_matching_sensitivity"]
    lines = [
        "# PBMC68k within-frame label-blind module comparison",
        "",
        "This experiment tests the direction of two recognizable evidence",
        "categories on the same common-support cells. Within the label-defined",
        "CD8/NK task frame, every cell receives both TCR/CD8 and frozen",
        "cytotoxic/NK-enriched marker masks, separately matched neutral masks,",
        "and an unmasked input. Target labels never select common-support",
        "inclusion or a mask.",
        "",
        "## Primary mechanistic result",
        "",
        (
            f"- Cells: `{result['n_cells']}` "
            f"(CD8 `{result['class_counts']['CD8']}`, "
            f"NK `{result['class_counts']['NK']}`)"
        ),
        (
            f"- Unique model calls: `{result['model_calls']}`; logical "
            f"condition-form observations: "
            f"`{result['logical_condition_form_observations']}`; shared-neutral "
            f"response reuses: `{result['shared_neutral_response_reuses']}`; "
            f"parse rate: `{result['parse_rate']:.1%}`"
        ),
        (
            "- TCR/CD8 effect on P(CD8): "
            f"**{_format_interval(t_result)}**, "
            f"one-sided Welch p=`{t_result['one_sided_welch_p_value']:.5g}`"
        ),
        (
            "- Cytotoxic/NK-enriched marker effect on P(CD8): "
            f"**{_format_interval(c_result)}**, "
            f"one-sided Welch p=`{c_result['one_sided_welch_p_value']:.5g}`"
        ),
        (
            "- TCR-minus-cytotoxic separation: "
            f"**{separation['effect']:+.3f} "
            f"[{separation['ci95_lower']:+.3f}, "
            f"{separation['ci95_upper']:+.3f}]**"
        ),
        (
            "- Conjunctive intersection-union gate passed: "
            f"**{str(result['primary_iut_pass']).lower()}** "
            f"(IUT p=`{result['primary_iut_p_value']:.5g}`)"
        ),
        (
            "- Both modules have the expected sign within both annotation "
            f"classes: **{str(result['class_direction_consistency_pass']).lower()}**"
        ),
        (f"- Full preregistered directional gate passed: **{str(result['primary_gate_pass']).lower()}**"),
        "",
        "Positive values mean the module pushes the model toward CD8 relative",
        "to its module-specific neutral deletion; negative values mean it",
        "pushes toward NK.",
        "",
        "| module | effect in CD8 cells | effect in NK cells | equal-class effect |",
        "|---|---:|---:|---:|",
        (
            "| TCR/CD8 | "
            f"{_format_interval(t_result['class_effects']['CD8'])} | "
            f"{_format_interval(t_result['class_effects']['NK'])} | "
            f"{_format_interval(t_result)} |"
        ),
        (
            "| cytotoxic/NK-enriched marker | "
            f"{_format_interval(c_result['class_effects']['CD8'])} | "
            f"{_format_interval(c_result['class_effects']['NK'])} | "
            f"{_format_interval(c_result)} |"
        ),
        "",
        "## Unmasked and neutral controls",
        "",
        "| module | unmasked - module mask | unmasked - neutral mask | adjusted |",
        "|---|---:|---:|---:|",
    ]
    for module, label in (
        ("T_TCR_CD8", "TCR/CD8"),
        ("cytotoxic_effector", "cytotoxic/NK-enriched marker"),
    ):
        record = result["module_effects"][module]
        lines.append(
            f"| {label} "
            f"| {_format_interval(record['unmasked_minus_module_mask'])} "
            f"| {_format_interval(record['unmasked_minus_neutral_mask'])} "
            f"| {_format_interval(record)} |"
        )
    lines.extend(
        [
            "",
            "## Prompt-factor boundary",
            "",
            "| module | order interaction | queried-target interaction | all forms expected sign |",
            "|---|---:|---:|---:|",
        ]
    )
    for module, label in (
        ("T_TCR_CD8", "TCR/CD8"),
        ("cytotoxic_effector", "cytotoxic/NK-enriched marker"),
    ):
        interactions = result["module_effects"][module]["prompt_factor_interactions"]
        lines.append(
            f"| {label} "
            f"| {_format_interval(interactions['order_ab_minus_ba'], key='effect')} "
            f"| {_format_interval(interactions['queried_target_a_minus_b'], key='effect')} "
            f"| {str(interactions['all_forms_expected_direction']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                f"Both modules pass the full ±{result['equivalence_margin']:.2f} "
                "prompt-equivalence and sign gate: "
                f"**{str(result['prompt_robust']).lower()}**."
            ),
            "",
            "## Matching sensitivity",
            "",
        ]
    )
    if strict.get("not_estimable"):
        lines.append(f"- Strict subset n=`{strict['n']}`; both classes not estimable.")
    else:
        lines.extend(
            [
                (
                    f"- Strict subset n=`{strict['n']}` "
                    f"(CD8 `{strict['class_counts']['CD8']}`, "
                    f"NK `{strict['class_counts']['NK']}`)"
                ),
                (f"- Strict TCR/CD8 effect: {_format_interval(strict['modules']['T_TCR_CD8'])}"),
                (
                    "- Strict cytotoxic/NK-enriched marker effect: "
                    f"{_format_interval(strict['modules']['cytotoxic_effector'])}"
                ),
                (
                    "- Strict separation: "
                    f"{strict['separation']:+.3f} "
                    f"[{strict['separation_ci95_lower']:+.3f}, "
                    f"{strict['separation_ci95_upper']:+.3f}]"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A passed gate supports opposing token-level evidence use on an",
            "equal-class-weighted average in one model and one external cohort:",
            "highest-ranked TCR/CD8-category symbols push output toward CD8,",
            "while highest-ranked frozen cytotoxic/NK-enriched marker symbols",
            "push it toward NK. The within-class sign guard is descriptive,",
            "not donor-level replication.",
            "",
            "The 10 included NK-labeled cells were selected because they also",
            "contain CD3/CD8-category evidence. They are atypical mixed-marker",
            "cells and may include contamination or doublets. This is a paired",
            "single-token two-category comparison crossed with a prompt",
            "factorial, not a biological 2x2 factorial; it cannot estimate",
            "module interaction, additivity, or mediation.",
            "It does not isolate an NK-receptor mechanism because the reduced",
            "feature panel censors that module. It is not a gene knockout,",
            "biological causality, a hidden-state activation route, latent-knowledge",
            "proof, multi-donor generalization, or a physical law.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
    )
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
        raise ModuleFactorialError("--workers must be between 1 and 16")

    base_rows, intervention_rows, manifest = _load_inputs()
    plan, plan_sha256 = _plan(
        base_rows,
        intervention_rows,
        args.model,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": ANALYSIS_ID,
                    "model": args.model,
                    "cells": len(base_rows),
                    "class_counts": {
                        "CD8": sum(int(row["label_a"]) == 1 for row in base_rows),
                        "NK": sum(int(row["label_a"]) == 0 for row in base_rows),
                    },
                    "unique_api_calls": len(plan),
                    "logical_condition_form_observations": sum(len(item["assignments"]) for item in plan),
                    "shared_neutral_response_reuses": sum(len(item["assignments"]) - 1 for item in plan),
                    "input_csv_sha256": _sha256(DATA_PATH),
                    "input_manifest_sha256": _sha256(MANIFEST_PATH),
                    "plan_sha256": plan_sha256,
                    "preregistration_exists": PREREG_PATH.exists(),
                    "preregistration_sha256": (_sha256(PREREG_PATH) if PREREG_PATH.exists() else None),
                    "execution_code_sha256": _sha256(Path(__file__)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not args.analyze_only:
        run(
            args.model,
            args.workers,
            base_rows,
            intervention_rows,
        )
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
    markdown_path.write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json_out": str(json_path),
                "markdown_out": str(markdown_path),
                "primary_gate_pass": result["primary_gate_pass"],
                "module_separation": result["module_separation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

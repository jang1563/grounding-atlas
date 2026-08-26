"""Run and analyze the PBMC68k CD8/NK marker-program masking intervention.

The primary comparison is label-matched marker-program masking versus an
equal-size, within-cell rank/expression-matched control mask.  Program and
control prompts are contemporaneously randomized across a full 2 x 2 nuisance
factorial:

* answer order: CD8/NK versus NK/CD8
* queried probability: P(CD8) versus P(NK)

All responses are aligned to P(CD8).  A positive correct-oriented
control-minus-program contrast means that masking the frozen biological program
harms the annotated-class probability more than masking matched non-program
genes.  This is an output-level intervention on one external cohort, not
hidden-state causality or multi-donor generalization.
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
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_grounding_eval import DECODE, complete, parse_prob_with_status  # noqa: E402

DATA_PATH = ROOT / "signal" / "single_cell" / "pbmc68k_cd8_nk_program_mask.csv"
MANIFEST_PATH = ROOT / "signal" / "single_cell" / "pbmc68k_cd8_nk_program_mask.manifest.json"
RESULT_ROOT = ROOT / "results" / "benchmark" / "single_cell" / "program_mask_transfer"
ANALYSIS_ID = "pbmc68k-cd8-nk-program-mask-v1"
CLASS_A = "CD8+ T cell"
CLASS_B = "NK cell"
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
CONDITION_COLUMNS = {
    "program": "program_mask_sentence",
    "control": "control_mask_sentence",
}
N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 50_000
EQUIVALENCE_MARGIN = 0.03
STRICT_MAX_RANK_DISTANCE = 10
STRICT_MAX_EXPRESSION_DISTANCE = 1.0


class ProgramMaskError(ValueError):
    """Raised when the intervention plan or checkpoint is incoherent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join((ANALYSIS_ID, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _raw_path(model: str) -> Path:
    return RESULT_ROOT / f"{_safe_model_name(model)}_raw.jsonl"


def _load_inputs() -> tuple[list[dict[str, str]], dict[str, Any], str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("analysis_id") != ANALYSIS_ID:
        raise ProgramMaskError("program-mask manifest analysis_id mismatch")
    data_sha256 = _sha256(DATA_PATH)
    if data_sha256 != manifest["artifacts"]["csv_sha256"]:
        raise ProgramMaskError("derived CSV does not match the frozen manifest hash")
    rows = list(csv.DictReader(DATA_PATH.open(newline="", encoding="utf-8")))
    if len(rows) != 119:
        raise ProgramMaskError(f"expected 119 eligible PBMC68k cells, observed {len(rows)}")
    if len({row["entity_id"] for row in rows}) != len(rows):
        raise ProgramMaskError("derived CSV entity IDs are not unique")
    counts = {label: sum(int(row["label_a"]) == label for row in rows) for label in (0, 1)}
    if counts != {0: 31, 1: 88}:
        raise ProgramMaskError(f"frozen class counts changed: {counts}")
    return rows, manifest, data_sha256, _sha256(MANIFEST_PATH)


def _build_prompt(row: dict[str, str], condition: str, form: str) -> str:
    spec = FORMS[form]
    return PROMPT.format(
        genes=row[CONDITION_COLUMNS[condition]],
        first=spec["first"],
        second=spec["second"],
        target=spec["target"],
    )


def _plan(
    rows: list[dict[str, str]],
    model: str,
) -> tuple[list[dict[str, Any]], str]:
    items = []
    for row_index, row in enumerate(rows):
        for condition in CONDITION_COLUMNS:
            for form in FORMS:
                prompt = _build_prompt(row, condition, form)
                items.append(
                    {
                        "row_index": row_index,
                        "entity_id": row["entity_id"],
                        "condition": condition,
                        "form": form,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
    generator = np.random.default_rng(_seed(model, "call-order"))
    generator.shuffle(items)
    for planned_index, item in enumerate(items):
        item["planned_index"] = planned_index
    plan_body = "\n".join(
        json.dumps(
            {
                key: item[key]
                for key in (
                    "planned_index",
                    "entity_id",
                    "condition",
                    "form",
                    "prompt_sha256",
                )
            },
            sort_keys=True,
        )
        for item in items
    )
    return items, hashlib.sha256((plan_body + "\n").encode()).hexdigest()


def _read_checkpoint(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        key = (record["entity_id"], record["condition"], record["form"])
        if key in records:
            raise ProgramMaskError(f"duplicate checkpoint key: {key}")
        records[key] = record
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


def _execute_item(
    item: dict[str, Any],
    row: dict[str, str],
    model: str,
    data_sha256: str,
    manifest_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    raw_output = _call_with_retry(model, item["prompt"])
    finished_at = datetime.now(UTC)
    probability, parsed = parse_prob_with_status(raw_output)
    target = FORMS[item["form"]]["target"]
    aligned_probability_a = probability if target == CLASS_A else 1.0 - probability
    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "entity_id": row["entity_id"],
        "cell_barcode": row["cell_barcode"],
        "technical_group": row["technical_group"],
        "reference_annotation": row["reference_annotation"],
        "label_a": int(row["label_a"]),
        "program": row["program"],
        "mask_k": int(row["mask_k"]),
        "target_genes": row["target_genes"],
        "control_genes": row["control_genes"],
        "condition": item["condition"],
        "form": item["form"],
        "order": item["form"][:2],
        "queried_target": "A" if target == CLASS_A else "B",
        "reported_probability": probability,
        "aligned_probability_a": aligned_probability_a,
        "parsed": parsed,
        "raw_output": raw_output,
        "planned_index": item["planned_index"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "prompt_sha256": item["prompt_sha256"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_csv_sha256": data_sha256,
        "input_manifest_sha256": manifest_sha256,
        "call_plan_sha256": plan_sha256,
    }


def run(
    model: str,
    workers: int,
    rows: list[dict[str, str]],
    data_sha256: str,
    manifest_sha256: str,
) -> Path:
    plan, plan_sha256 = _plan(rows, model)
    path = _raw_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_checkpoint(path)
    expected_keys = {(item["entity_id"], item["condition"], item["form"]) for item in plan}
    if set(existing) - expected_keys:
        raise ProgramMaskError("checkpoint contains keys outside the frozen call plan")
    for record in existing.values():
        if (
            record["analysis_id"] != ANALYSIS_ID
            or record["model"] != model
            or record["input_csv_sha256"] != data_sha256
            or record["input_manifest_sha256"] != manifest_sha256
            or record["call_plan_sha256"] != plan_sha256
        ):
            raise ProgramMaskError("checkpoint provenance does not match this frozen run")

    pending = [item for item in plan if (item["entity_id"], item["condition"], item["form"]) not in existing]
    if not pending:
        print(f"checkpoint already complete: {len(existing)}/{len(plan)}", flush=True)
        return path

    failures: list[tuple[dict[str, Any], Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_item,
                item,
                rows[item["row_index"]],
                model,
                data_sha256,
                manifest_sha256,
                plan_sha256,
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
                    f"FAILED {item['planned_index']} {item['entity_id']} {item['condition']} {item['form']}: {error}",
                    flush=True,
                )
                continue
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(plan):
                print(
                    f"{model} program-mask checkpoint {completed}/{len(plan)} parsed={record['parsed']}",
                    flush=True,
                )
    if failures:
        first_item, first_error = failures[0]
        raise RuntimeError(
            f"{len(failures)} model calls failed; first failure "
            f"{first_item['entity_id']} {first_item['condition']} "
            f"{first_item['form']}: {first_error}"
        )
    return path


def _equal_class_mean(values: np.ndarray, labels: np.ndarray) -> float:
    return float(0.5 * (values[labels == 1].mean() + values[labels == 0].mean()))


def _interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _bootstrap_statistics(
    effects: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    positive_draws = positive[generator.integers(0, len(positive), size=(N_BOOTSTRAP, len(positive)))]
    negative_draws = negative[generator.integers(0, len(negative), size=(N_BOOTSTRAP, len(negative)))]

    def draw_equal(values: np.ndarray) -> np.ndarray:
        return 0.5 * (values[positive_draws].mean(axis=1) + values[negative_draws].mean(axis=1))

    cell_effect = effects.mean(axis=1)
    primary_draws = draw_equal(cell_effect)
    class_a_draws = cell_effect[positive_draws].mean(axis=1)
    class_b_draws = cell_effect[negative_draws].mean(axis=1)
    form_draws = {form: draw_equal(effects[:, form_index]) for form_index, form in enumerate(FORMS)}
    order_ab_draws = draw_equal(effects[:, [0, 1]].mean(axis=1))
    order_ba_draws = draw_equal(effects[:, [2, 3]].mean(axis=1))
    target_a_draws = draw_equal(effects[:, [0, 2]].mean(axis=1))
    target_b_draws = draw_equal(effects[:, [1, 3]].mean(axis=1))
    order_interaction_draws = order_ab_draws - order_ba_draws
    target_interaction_draws = target_a_draws - target_b_draws

    primary_lower, primary_upper = _interval(primary_draws)
    class_a_lower, class_a_upper = _interval(class_a_draws)
    class_b_lower, class_b_upper = _interval(class_b_draws)
    order_lower, order_upper = _interval(order_interaction_draws)
    target_lower, target_upper = _interval(target_interaction_draws)
    return {
        "primary_ci95_lower": primary_lower,
        "primary_ci95_upper": primary_upper,
        "class_a_ci95_lower": class_a_lower,
        "class_a_ci95_upper": class_a_upper,
        "class_b_ci95_lower": class_b_lower,
        "class_b_ci95_upper": class_b_upper,
        "form_ci95": {
            form: {
                "lower": _interval(draws)[0],
                "upper": _interval(draws)[1],
            }
            for form, draws in form_draws.items()
        },
        "order_interaction_ci95_lower": order_lower,
        "order_interaction_ci95_upper": order_upper,
        "queried_target_interaction_ci95_lower": target_lower,
        "queried_target_interaction_ci95_upper": target_upper,
    }


def _sign_flip_p_value(
    cell_effect: np.ndarray,
    labels: np.ndarray,
    *,
    observed: float,
    seed: int,
) -> float:
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(N_PERMUTATIONS):
        signs = generator.choice((-1.0, 1.0), size=len(cell_effect))
        permuted = _equal_class_mean(cell_effect * signs, labels)
        exceedances += abs(permuted) >= abs(observed)
    return float((exceedances + 1) / (N_PERMUTATIONS + 1))


def _strict_matching(row: dict[str, str]) -> bool:
    genes = row["original_sentence"].split()
    positions = {gene: rank for rank, gene in enumerate(genes)}
    target_genes = row["target_genes"].split(";")
    control_genes = row["control_genes"].split(";")
    target_values = list(map(float, row["target_values"].split(";")))
    control_values = list(map(float, row["control_values"].split(";")))
    return all(
        abs(positions[target] - positions[control]) <= STRICT_MAX_RANK_DISTANCE
        and abs(target_value - control_value) <= STRICT_MAX_EXPRESSION_DISTANCE
        for target, control, target_value, control_value in zip(
            target_genes,
            control_genes,
            target_values,
            control_values,
        )
    )


def analyze(
    model: str,
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    data_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    raw_path = _raw_path(model)
    checkpoint = _read_checkpoint(raw_path)
    expected_keys = {
        (row["entity_id"], condition, form) for row in rows for condition in CONDITION_COLUMNS for form in FORMS
    }
    if set(checkpoint) != expected_keys:
        raise ProgramMaskError(f"checkpoint incomplete: expected {len(expected_keys)}, observed {len(checkpoint)}")

    entity_to_index = {row["entity_id"]: index for index, row in enumerate(rows)}
    labels = np.asarray([int(row["label_a"]) for row in rows], dtype=int)
    mask_k = np.asarray([int(row["mask_k"]) for row in rows], dtype=int)
    strict_mask = np.asarray([_strict_matching(row) for row in rows], dtype=bool)
    probabilities = {condition: np.empty((len(rows), len(FORMS)), dtype=float) for condition in CONDITION_COLUMNS}
    parse_flags = []
    for (entity_id, condition, form), record in checkpoint.items():
        if record["input_csv_sha256"] != data_sha256 or record["input_manifest_sha256"] != manifest_sha256:
            raise ProgramMaskError("checkpoint input hash mismatch")
        row_index = entity_to_index[entity_id]
        form_index = list(FORMS).index(form)
        probabilities[condition][row_index, form_index] = record["aligned_probability_a"]
        parse_flags.append(bool(record["parsed"]))

    correct_sign = np.where(labels == 1, 1.0, -1.0)
    effects = correct_sign[:, None] * (probabilities["control"] - probabilities["program"])
    cell_effect = effects.mean(axis=1)
    correct_probabilities = {
        condition: np.where(
            labels[:, None] == 1,
            matrix,
            1.0 - matrix,
        ).mean(axis=1)
        for condition, matrix in probabilities.items()
    }

    def effect_counts(selected: np.ndarray) -> dict[str, int]:
        selected_effects = cell_effect[selected]
        tolerance = 1e-12
        return {
            "negative": int((selected_effects < -tolerance).sum()),
            "zero": int((np.abs(selected_effects) <= tolerance).sum()),
            "positive": int((selected_effects > tolerance).sum()),
        }

    primary = _equal_class_mean(cell_effect, labels)
    bootstrap = _bootstrap_statistics(
        effects,
        labels,
        seed=_seed(model, "bootstrap"),
    )
    p_value = _sign_flip_p_value(
        cell_effect,
        labels,
        observed=primary,
        seed=_seed(model, "sign-flip"),
    )

    form_effects = {form: _equal_class_mean(effects[:, form_index], labels) for form_index, form in enumerate(FORMS)}
    order_ab = _equal_class_mean(effects[:, [0, 1]].mean(axis=1), labels)
    order_ba = _equal_class_mean(effects[:, [2, 3]].mean(axis=1), labels)
    target_a = _equal_class_mean(effects[:, [0, 2]].mean(axis=1), labels)
    target_b = _equal_class_mean(effects[:, [1, 3]].mean(axis=1), labels)
    order_interaction = order_ab - order_ba
    target_interaction = target_a - target_b
    prompt_robust_equivalence = (
        bootstrap["order_interaction_ci95_lower"] > -EQUIVALENCE_MARGIN
        and bootstrap["order_interaction_ci95_upper"] < EQUIVALENCE_MARGIN
        and bootstrap["queried_target_interaction_ci95_lower"] > -EQUIVALENCE_MARGIN
        and bootstrap["queried_target_interaction_ci95_upper"] < EQUIVALENCE_MARGIN
    )

    condition_metrics = {}
    for condition, matrix in probabilities.items():
        averaged = matrix.mean(axis=1)
        condition_metrics[condition] = {
            "orientation_averaged_auroc": float(roc_auc_score(labels, averaged)),
            "orientation_averaged_brier": float(brier_score_loss(labels, averaged)),
            "mean_aligned_probability_a": float(averaged.mean()),
        }

    strict_result: dict[str, Any]
    strict_labels = labels[strict_mask]
    if set(strict_labels.tolist()) == {0, 1}:
        strict_effects = effects[strict_mask]
        strict_primary = _equal_class_mean(strict_effects.mean(axis=1), strict_labels)
        strict_bootstrap = _bootstrap_statistics(
            strict_effects,
            strict_labels,
            seed=_seed(model, "strict-bootstrap"),
        )
        strict_result = {
            "n": int(strict_mask.sum()),
            "class_counts": {
                "CD8": int((strict_labels == 1).sum()),
                "NK": int((strict_labels == 0).sum()),
            },
            "effect": strict_primary,
            "ci95_lower": strict_bootstrap["primary_ci95_lower"],
            "ci95_upper": strict_bootstrap["primary_ci95_upper"],
        }
    else:
        strict_result = {"n": int(strict_mask.sum()), "not_estimable": True}

    dose_results = []
    for dose in sorted(set(mask_k.tolist())):
        selected = mask_k == dose
        dose_labels = labels[selected]
        result = {
            "mask_k": int(dose),
            "n": int(selected.sum()),
            "class_counts": {
                "CD8": int((dose_labels == 1).sum()),
                "NK": int((dose_labels == 0).sum()),
            },
        }
        if set(dose_labels.tolist()) == {0, 1}:
            result["equal_class_effect"] = _equal_class_mean(
                cell_effect[selected],
                dose_labels,
            )
        dose_results.append(result)

    technical_group_results = []
    groups = np.asarray([row["technical_group"] for row in rows], dtype=object)
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_labels = labels[selected]
        record = {
            "technical_group": group,
            "n": int(selected.sum()),
            "class_counts": {
                "CD8": int((group_labels == 1).sum()),
                "NK": int((group_labels == 0).sum()),
            },
            "mean_correct_oriented_effect": float(cell_effect[selected].mean()),
        }
        if set(group_labels.tolist()) == {0, 1}:
            record["equal_class_effect"] = _equal_class_mean(
                cell_effect[selected],
                group_labels,
            )
        technical_group_results.append(record)

    annotation_results = []
    annotations = np.asarray([row["reference_annotation"] for row in rows], dtype=object)
    for annotation in sorted(set(annotations.tolist())):
        selected = annotations == annotation
        annotation_results.append(
            {
                "reference_annotation": annotation,
                "n": int(selected.sum()),
                "control_mean_correct_probability": float(correct_probabilities["control"][selected].mean()),
                "program_mean_correct_probability": float(correct_probabilities["program"][selected].mean()),
                "mean_effect": float(cell_effect[selected].mean()),
                "cell_effect_counts": effect_counts(selected),
            }
        )

    masked_gene_frequencies = {}
    for program in sorted({row["program"] for row in rows}):
        counts = Counter(gene for row in rows if row["program"] == program for gene in row["target_genes"].split(";"))
        masked_gene_frequencies[program] = [
            {"gene": gene, "cells": count}
            for gene, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    all_records = list(checkpoint.values())
    started = min(record["started_at_utc"] for record in all_records)
    finished = max(record["finished_at_utc"] for record in all_records)

    return {
        "analysis_id": ANALYSIS_ID,
        "model": model,
        "claim_scope": manifest["claim_scope"],
        "n_cells": len(rows),
        "class_counts": {
            "CD8": int((labels == 1).sum()),
            "NK": int((labels == 0).sum()),
        },
        "model_calls": len(all_records),
        "parse_rate": float(np.mean(parse_flags)),
        "run_started_at_utc": started,
        "run_finished_at_utc": finished,
        "primary_estimand": (
            "equal-class mean over four prompt forms of signed P(A)_control_mask minus P(A)_program_mask"
        ),
        "primary_effect": primary,
        "primary_ci95_lower": bootstrap["primary_ci95_lower"],
        "primary_ci95_upper": bootstrap["primary_ci95_upper"],
        "primary_sign_flip_p_value": p_value,
        "class_effects": {
            "CD8": {
                "effect": float(cell_effect[labels == 1].mean()),
                "ci95_lower": bootstrap["class_a_ci95_lower"],
                "ci95_upper": bootstrap["class_a_ci95_upper"],
                "control_mean_correct_probability": float(correct_probabilities["control"][labels == 1].mean()),
                "program_mean_correct_probability": float(correct_probabilities["program"][labels == 1].mean()),
                "cell_effect_counts": effect_counts(labels == 1),
            },
            "NK": {
                "effect": float(cell_effect[labels == 0].mean()),
                "ci95_lower": bootstrap["class_b_ci95_lower"],
                "ci95_upper": bootstrap["class_b_ci95_upper"],
                "control_mean_correct_probability": float(correct_probabilities["control"][labels == 0].mean()),
                "program_mean_correct_probability": float(correct_probabilities["program"][labels == 0].mean()),
                "cell_effect_counts": effect_counts(labels == 0),
            },
        },
        "cell_effect_counts": effect_counts(np.ones(len(rows), dtype=bool)),
        "form_effects": {
            form: {
                "effect": effect,
                "ci95_lower": bootstrap["form_ci95"][form]["lower"],
                "ci95_upper": bootstrap["form_ci95"][form]["upper"],
            }
            for form, effect in form_effects.items()
        },
        "prompt_factor_interactions": {
            "order_ab_minus_ba": {
                "effect": order_interaction,
                "ci95_lower": bootstrap["order_interaction_ci95_lower"],
                "ci95_upper": bootstrap["order_interaction_ci95_upper"],
            },
            "queried_target_a_minus_b": {
                "effect": target_interaction,
                "ci95_lower": bootstrap["queried_target_interaction_ci95_lower"],
                "ci95_upper": bootstrap["queried_target_interaction_ci95_upper"],
            },
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "both_interactions_equivalent": prompt_robust_equivalence,
        },
        "condition_metrics": condition_metrics,
        "strict_matching_sensitivity": strict_result,
        "mask_dose": dose_results,
        "exploratory_biological_localization": {
            "reference_annotations": annotation_results,
            "masked_gene_frequencies": masked_gene_frequencies,
            "interpretation": (
                "post-hoc descriptive localization only; co-masking and "
                "class-dependent mask dose preclude single-gene or dose-response attribution"
            ),
        },
        "technical_groups": technical_group_results,
        "provenance": {
            "input_csv": str(DATA_PATH.relative_to(ROOT)),
            "input_csv_sha256": data_sha256,
            "input_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "input_manifest_sha256": manifest_sha256,
            "raw_checkpoint": str(raw_path.relative_to(ROOT)),
            "raw_checkpoint_sha256": _sha256(raw_path),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "prompt_template_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "decode": DECODE,
            "bootstrap_replicates": N_BOOTSTRAP,
            "sign_flip_permutations": N_PERMUTATIONS,
        },
    }


def _format_interval(record: dict[str, Any]) -> str:
    return f"{record['effect']:+.3f} [{record['ci95_lower']:+.3f}, {record['ci95_upper']:+.3f}]"


def render_markdown(result: dict[str, Any]) -> str:
    primary = {
        "effect": result["primary_effect"],
        "ci95_lower": result["primary_ci95_lower"],
        "ci95_upper": result["primary_ci95_upper"],
    }
    interactions = result["prompt_factor_interactions"]
    strict = result["strict_matching_sensitivity"]
    localization = result["exploratory_biological_localization"]
    lines = [
        "# PBMC68k transfer: marker-program masking versus matched deletion",
        "",
        "This experiment moves from prompt auditing to a biological intervention.",
        "A label-matched CD8 or NK marker program frozen from the earlier",
        "PBMC3k builder was masked in an independently annotated PBMC68k Donor A",
        "cell. The control arm",
        "masked the same number of source-neutral genes matched within that cell.",
        "Program and control calls were contemporaneously randomized across all four",
        "order × queried-target prompt forms.",
        "",
        "## Primary result",
        "",
        (
            f"- Cells: `{result['n_cells']}` "
            f"(CD8 `{result['class_counts']['CD8']}`, NK `{result['class_counts']['NK']}`)"
        ),
        f"- Model calls: `{result['model_calls']}`; parse rate: `{result['parse_rate']:.1%}`",
        f"- Control-adjusted program effect: **{_format_interval(primary)}**",
        f"- Paired sign-flip p-value: `{result['primary_sign_flip_p_value']:.5g}`",
        "",
        "Positive values mean that deleting the frozen marker program harms the",
        "reference-class probability more than deleting equally many matched",
        "non-program genes.",
        "",
        "| class | control P(correct) | program P(correct) | effect (95% CI) | positive cells |",
        "|---|---:|---:|---:|---:|",
        (
            f"| CD8 | {result['class_effects']['CD8']['control_mean_correct_probability']:.3f} "
            f"| {result['class_effects']['CD8']['program_mean_correct_probability']:.3f} "
            f"| {_format_interval(result['class_effects']['CD8'])} "
            f"| {result['class_effects']['CD8']['cell_effect_counts']['positive']}/"
            f"{result['class_counts']['CD8']} |"
        ),
        (
            f"| NK | {result['class_effects']['NK']['control_mean_correct_probability']:.3f} "
            f"| {result['class_effects']['NK']['program_mean_correct_probability']:.3f} "
            f"| {_format_interval(result['class_effects']['NK'])} "
            f"| {result['class_effects']['NK']['cell_effect_counts']['positive']}/"
            f"{result['class_counts']['NK']} |"
        ),
        "",
        (
            "Across cells, the paired four-form effect was positive for "
            f"`{result['cell_effect_counts']['positive']}/{result['n_cells']}`, "
            f"zero for `{result['cell_effect_counts']['zero']}`, and negative for "
            f"`{result['cell_effect_counts']['negative']}`."
        ),
        "",
        "## Prompt-factor boundary",
        "",
        "| prompt form | effect (95% CI) |",
        "|---|---:|",
    ]
    for form in FORMS:
        lines.append(f"| `{form}` | {_format_interval(result['form_effects'][form])} |")
    lines.extend(
        [
            "",
            ("Order interaction (AB minus BA): " + _format_interval(interactions["order_ab_minus_ba"])),
            (
                "Queried-target interaction (P(A) minus P(B)): "
                + _format_interval(interactions["queried_target_a_minus_b"])
            ),
            (
                f"Both interactions inside the preregistered ±{EQUIVALENCE_MARGIN:.2f} "
                "equivalence margin: "
                f"**{str(interactions['both_interactions_equivalent']).lower()}**."
            ),
            "",
            "## Controls and sensitivities",
            "",
            (
                f"- Strict matching sensitivity (all rank distances ≤"
                f"{STRICT_MAX_RANK_DISTANCE}, expression distances ≤"
                f"{STRICT_MAX_EXPRESSION_DISTANCE:.1f}): n=`{strict['n']}` "
                f"(CD8 `{strict['class_counts']['CD8']}`, "
                f"NK `{strict['class_counts']['NK']}`)"
                + (f", effect {_format_interval(strict)}" if "effect" in strict else ", not estimable in both classes")
            ),
            (
                "- Orientation-averaged AUROC, control/program masks: "
                f"`{result['condition_metrics']['control']['orientation_averaged_auroc']:.3f}`/"
                f"`{result['condition_metrics']['program']['orientation_averaged_auroc']:.3f}`"
            ),
            (
                "- Orientation-averaged Brier, control/program masks: "
                f"`{result['condition_metrics']['control']['orientation_averaged_brier']:.3f}`/"
                f"`{result['condition_metrics']['program']['orientation_averaged_brier']:.3f}`"
            ),
            "",
            "## Exploratory biological localization",
            "",
            "These post-hoc summaries localize the preregistered aggregate; they are",
            "not independent tests.",
            "",
            "| reference annotation | n | control P(correct) | program P(correct) | effect |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for annotation in localization["reference_annotations"]:
        lines.append(
            f"| {annotation['reference_annotation']} | {annotation['n']} "
            f"| {annotation['control_mean_correct_probability']:.3f} "
            f"| {annotation['program_mean_correct_probability']:.3f} "
            f"| {annotation['mean_effect']:+.3f} |"
        )
    frequency = localization["masked_gene_frequencies"]
    top_cd8 = ", ".join(f"`{record['gene']}` {record['cells']}/88" for record in frequency["CD8_identity"][:3])
    top_nk = ", ".join(f"`{record['gene']}` {record['cells']}/31" for record in frequency["NK_identity"][:3])
    lines.extend(
        [
            "",
            f"- Most frequently masked CD8-program genes: {top_cd8}.",
            f"- Most frequently masked NK-program genes: {top_nk}.",
            "- Because genes were co-masked and mask count is confounded with program/class,",
            "  these frequencies do not identify a necessary gene or a dose-response curve.",
            "",
            "## Interpretation boundary",
            "",
            "This is a causal input-to-output masking contrast for one external cohort.",
            "The operational programs mix lineage-identity and cytotoxic-effector",
            "markers, especially in the NK arm, so this result does not isolate a",
            "pure lineage mechanism from a cytotoxic-state mechanism.",
            "It is not an unassisted annotation benchmark because the reference label",
            "selects which frozen program is masked. It does not establish a hidden-state",
            "activation route, corpus exposure, a physical law, or multi-donor",
            "generalization. Technical barcode suffixes are not treated as donors.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyze a complete checkpoint without making model calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the frozen call plan without making model calls",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ProgramMaskError("--workers must be between 1 and 16")

    rows, manifest, data_sha256, manifest_sha256 = _load_inputs()
    plan, plan_sha256 = _plan(rows, args.model)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": ANALYSIS_ID,
                    "cells": len(rows),
                    "calls": len(plan),
                    "class_counts": {
                        "CD8": sum(int(row["label_a"]) == 1 for row in rows),
                        "NK": sum(int(row["label_a"]) == 0 for row in rows),
                    },
                    "input_csv_sha256": data_sha256,
                    "input_manifest_sha256": manifest_sha256,
                    "model": args.model,
                    "plan_sha256": plan_sha256,
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
            rows,
            data_sha256,
            manifest_sha256,
        )
    result = analyze(
        args.model,
        rows,
        manifest,
        data_sha256,
        manifest_sha256,
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
                "primary_effect": result["primary_effect"],
                "primary_ci95": [
                    result["primary_ci95_lower"],
                    result["primary_ci95_upper"],
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

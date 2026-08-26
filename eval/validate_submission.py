"""Validate a GroundBench submission before opening a PR.

Usage:  python eval/validate_submission.py results/benchmark/<model> [--allow-partial]

Checks that a results/benchmark/<model>/ directory is a well-formed, comparable, complete submission:
valid scorecard / manifest / raw.jsonl, the full CORE task set (unless --allow-partial), the current
prompt version, per-task fields present and in range, and raw<->scorecard consistency. Exit 0 = pass,
1 = fail, 2 = usage. See SUBMITTING.md.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import (  # noqa: E402
    BIOLOGICAL_SPLIT_SCOPES,
    CORE,
    EXPLORATORY,
    QUARANTINED,
    SPLIT_SCOPE_ENTITY_PROXY,
    SPLIT_SCOPE_UNAVAILABLE,
    TASKS,
    TRUTH_LEVEL_CODES,
    TRUTH_TAXONOMY_VERSION,
    GroundBenchSampler,
)
from run_grounding_eval import (  # noqa: E402
    DECODE,
    METADATA_CONTRACT_VERSION,
    PROMPT_VERSION,
    SCORE_SCHEMA,
    _brier,
    _ceilings,
    _log_loss,
    _repository_state,
    _sha256_file,
    _task_data_checksums,
    _task_rng,
    aurc,
    auroc,
    build_pair_group_comparisons,
    ci,
    ece,
    paired_auroc_delta_ci,
    parse_prob_with_status,
    probability_ece,
    sel_acc,
)

ROOT = os.path.dirname(HERE)
ALLOWED_CONDITIONS = {"matched", "scrambled", "re_notation"}

REQUIRED_TASK_FIELDS = [
    "n", "output_auroc", "output_auroc_ci", "brier", "brier_ci", "log_loss", "log_loss_ci",
    "probability_ece", "confidence_ece",
    "valid_response_rate", "n_invalid", "output_auroc_valid_only", "aurc", "sel_acc_50",
    "web_exposure", "orientation",
    "truth_level_code", "target_source_kind", "truth_level",
    "task_status", "reference_comparability", "sample_prevalence",
    "calibration_scope", "uncertainty_method", "uncertainty_unit", "reference_score",
    "reference_gap", "biological_question_id", "task_family_id", "split_group_scope",
    "intervention_pair_field", "intervention_pair_id", "factor_levels", "pair_group",
]
REQUIRED_MANIFEST = [
    "model", "prompt_version", "score_schema", "metadata_contract_version", "decode", "seed",
    "n_per_task", "dry_run",
    "data_commit", "tasks", "truth_taxonomy_version", "task_truth_metadata",
    "sampling_contract", "registry", "working_tree_clean", "code_data_fingerprint",
    "provider", "model_revision", "dataset_sha256", "task_data_sha256",
    "reference_registry_sha256", "environment", "calibration_scope", "uncertainty_method",
    "uncertainty_unit", "pair_group_comparisons_file",
]
REQUIRED_RAW_FIELDS = [
    "task", "id", "entity_id", "source_id", "entity_id_scope", "biological_question_id",
    "task_family_id", "truth_level_code", "target_source_kind", "truth_level",
    "split_group_id", "split_group_scope", "intervention_pair_id",
    "factor_levels", "condition", "source_label", "target_label", "prob", "prompt_version",
    "parse_valid", "parse_status", "output",
]


def _load_json(path, errs):
    if not os.path.exists(path):
        errs.append(f"missing {os.path.basename(path)}")
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        errs.append(f"invalid JSON {os.path.basename(path)}: {e}")
        return None


def validate(d, allow_partial=False):
    errs, warns = [], []
    sc = _load_json(os.path.join(d, "scorecard.json"), errs)
    man = _load_json(os.path.join(d, "manifest.json"), errs)
    rawp = os.path.join(d, "raw.jsonl")
    if not os.path.exists(rawp):
        errs.append("missing raw.jsonl")
    if sc is None or man is None:
        return errs, warns

    for k in REQUIRED_MANIFEST:
        if k not in man:
            errs.append(f"manifest missing '{k}'")
    if man.get("prompt_version") != PROMPT_VERSION:
        errs.append(f"prompt_version {man.get('prompt_version')!r} != current {PROMPT_VERSION!r} "
                    "(not comparable; re-run on the current harness)")
    if man.get("score_schema") != SCORE_SCHEMA:
        errs.append(
            f"score_schema {man.get('score_schema')!r} != current {SCORE_SCHEMA!r}"
        )
    if man.get("metadata_contract_version") != METADATA_CONTRACT_VERSION:
        errs.append(
            "metadata_contract_version "
            f"{man.get('metadata_contract_version')!r} != current "
            f"{METADATA_CONTRACT_VERSION!r}"
        )
    if man.get("truth_taxonomy_version") != TRUTH_TAXONOMY_VERSION:
        errs.append(
            "truth_taxonomy_version "
            f"{man.get('truth_taxonomy_version')!r} != current "
            f"{TRUTH_TAXONOMY_VERSION!r}"
        )
    if man.get("decode") != DECODE:
        errs.append(f"decode {man.get('decode')!r} != current fixed decode {DECODE!r}")
    if man.get("pair_group_comparisons_file") != "pair_group_comparisons.json":
        errs.append("manifest pair_group_comparisons_file must be 'pair_group_comparisons.json'")
    if not isinstance(man.get("seed"), int):
        errs.append("manifest seed must be an integer")
    if not isinstance(man.get("n_per_task"), int) or man.get("n_per_task", 0) <= 0:
        errs.append("manifest n_per_task must be a positive integer")
    expected_registry = {
        "core": list(CORE),
        "exploratory": list(EXPLORATORY),
        "quarantined": list(QUARANTINED),
    }
    if man.get("registry") != expected_registry:
        errs.append("manifest registry does not exactly match the current task registry")
    if man.get("sampling_contract") != (
        "shared entity intersection within pair_group; balanced labels"
    ):
        errs.append("manifest sampling_contract is not the current fixed contract")
    if man.get("calibration_scope") != "balanced_benchmark_distribution_only":
        errs.append("manifest calibration_scope is not the current balanced-sample contract")
    if man.get("uncertainty_method") != "iid_entity_item_bootstrap_pilot":
        errs.append("manifest uncertainty_method is not the current pilot contract")
    if man.get("uncertainty_unit") != (
        "entity_id_row_not_biological_dependency_cluster"
    ):
        errs.append("manifest uncertainty_unit is not the current pilot unit")
    if man.get("dry_run"):
        errs.append("manifest dry_run=true (synthetic results, not a real submission)")
    if man.get("working_tree_clean") is not True:
        errs.append("manifest working_tree_clean is not true (commit code/data before a release run)")
    if man.get("model_revision") in (None, "", "unspecified"):
        errs.append("manifest model_revision must identify an immutable provider/checkpoint revision")
    if not man.get("provider"):
        errs.append("manifest provider is missing")
    if not man.get("dataset_sha256"):
        errs.append("manifest dataset_sha256 is missing")
    if set(man.get("task_data_sha256", {})) != set(TASKS):
        errs.append("manifest task_data_sha256 does not cover the current task registry")
    current_state = _repository_state()
    for field in ("data_commit", "working_tree_clean", "code_data_fingerprint"):
        if man.get(field) != current_state[field]:
            errs.append(f"manifest {field} does not match the current checkout")
    dataset_path = os.path.join(ROOT, "dataset", "groundbench.parquet")
    current_dataset_sha = _sha256_file(dataset_path) if os.path.isfile(dataset_path) else None
    if man.get("dataset_sha256") != current_dataset_sha:
        errs.append("manifest dataset_sha256 does not match the current dataset")
    if man.get("task_data_sha256") != _task_data_checksums():
        errs.append("manifest task_data_sha256 values do not match the current task sources")
    reference_path = os.path.join(ROOT, "results", "benchmark", "ceilings.json")
    current_reference_sha = (
        _sha256_file(reference_path) if os.path.isfile(reference_path) else None
    )
    if man.get("reference_registry_sha256") != current_reference_sha:
        errs.append(
            "manifest reference_registry_sha256 does not match the current reference registry"
        )

    missing = [t for t in CORE if t not in sc]
    if missing:
        msg = f"missing {len(missing)} CORE task(s): {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        (warns if allow_partial else errs).append(msg)
    extra = [t for t in sc if t not in TASKS]
    if extra:
        errs.append(f"{len(extra)} task(s) not in the registry: {', '.join(extra[:5])}")
    if set(man.get("tasks", [])) != set(sc):
        errs.append("manifest tasks do not match scorecard task keys")
    expected_truth_metadata = {
        task_id: {
            "truth_level_code": TASKS[task_id]["truth_level_code"],
            "target_source_kind": TASKS[task_id]["target_source_kind"],
        }
        for task_id in sc
        if task_id in TASKS
    }
    if man.get("task_truth_metadata") != expected_truth_metadata:
        errs.append(
            "manifest task_truth_metadata does not exactly match the current task registry"
        )

    references = _ceilings()
    for t, rec in sc.items():
        if t not in TASKS:
            continue
        miss = [f for f in REQUIRED_TASK_FIELDS if f not in rec]
        if miss:
            errs.append(f"{t}: missing field(s) {', '.join(miss)}")
            continue
        a = rec["output_auroc"]
        if not (isinstance(a, (int, float)) and 0.0 <= a <= 1.0):
            errs.append(f"{t}: output_auroc {a} out of [0,1]")
        reported_output_ci = rec["output_auroc_ci"]
        if not (isinstance(reported_output_ci, list) and len(reported_output_ci) == 2):
            errs.append(
                f"{t}: output_auroc_ci must be a [lo, hi] pair, got "
                f"{reported_output_ci!r}"
            )
        if rec["web_exposure"] not in ("rich", "zero", "mixed"):
            errs.append(f"{t}: web_exposure {rec['web_exposure']!r} not in rich/zero/mixed")
        if rec["truth_level_code"] not in TRUTH_LEVEL_CODES:
            errs.append(
                f"{t}: truth_level_code {rec['truth_level_code']!r} is outside T0-T5"
            )
        if rec["truth_level"] != rec["truth_level_code"]:
            errs.append(
                f"{t}: deprecated truth_level alias must equal truth_level_code"
            )
        expected_task_metadata = {
            "web_exposure": TASKS[t]["web"],
            "orientation": TASKS[t]["orient"],
            "truth_level_code": TASKS[t]["truth_level_code"],
            "target_source_kind": TASKS[t]["target_source_kind"],
            "truth_level": TASKS[t]["truth_level"],
            "task_status": TASKS[t]["status"],
            "biological_question_id": TASKS[t]["biological_question_id"],
            "task_family_id": TASKS[t]["task_family_id"],
            "split_group_scope": TASKS[t]["split_group_scope"],
            "intervention_pair_field": TASKS[t]["intervention_pair_field"],
            "intervention_pair_id": TASKS[t]["intervention_pair_id"],
            "factor_levels": TASKS[t]["factor_levels"],
            "pair_group": TASKS[t].get("pair_group"),
        }
        for field, expected in expected_task_metadata.items():
            if rec.get(field) != expected:
                errs.append(f"{t}: {field} disagrees with the task registry")
        if rec["reference_comparability"] != TASKS[t]["reference_comparability"]:
            errs.append(f"{t}: reference_comparability disagrees with the task registry")
        reference_key = TASKS[t]["reference"]
        expected_reference = (
            references.get(reference_key)
            if isinstance(reference_key, str)
            else reference_key
        )
        if rec["reference_score"] != expected_reference:
            errs.append(f"{t}: reference_score disagrees with the current reference registry")
        if rec["reference_comparability"] != "same_entities_same_split" and rec.get("reference_gap") is not None:
            errs.append(f"{t}: context-only reference must not be serialized as a score gap")
        if rec["calibration_scope"] != "balanced_benchmark_distribution_only":
            errs.append(f"{t}: calibration_scope must identify the balanced benchmark population")
        if rec["uncertainty_method"] != "iid_entity_item_bootstrap_pilot":
            errs.append(f"{t}: uncertainty_method is not the current pilot contract")

    if os.path.exists(rawp):
        raw_tasks = set()
        raw_by_task_condition = {}
        raw_rows = []
        raw_ids = set()
        try:
            for line in open(rawp):
                line = line.strip()
                if line:
                    row = json.loads(line)
                    raw_tasks.add(row.get("task"))
                    missing_raw = [field for field in REQUIRED_RAW_FIELDS if field not in row]
                    if missing_raw:
                        errs.append(f"raw.jsonl row missing field(s) {', '.join(missing_raw)}")
                        break
                    task_id = row["task"]
                    if task_id not in TASKS:
                        errs.append(f"raw.jsonl row has unknown task {task_id!r}")
                        break
                    if row["condition"] not in ALLOWED_CONDITIONS:
                        errs.append(f"{task_id}: invalid condition {row['condition']!r}")
                    if not row["id"] or row["id"] in raw_ids:
                        errs.append(f"{task_id}: raw row id is empty or duplicated")
                    raw_ids.add(row["id"])
                    if not row["entity_id"]:
                        errs.append(f"{task_id}: entity_id is empty")
                    if row["source_id"] in (None, ""):
                        errs.append(f"{task_id}: source_id is empty")
                    if row["entity_id_scope"] != TASKS[task_id]["entity_id_scope"]:
                        errs.append(f"{task_id}: entity_id_scope disagrees with the task registry")
                    task = TASKS[task_id]
                    if row["truth_level_code"] not in TRUTH_LEVEL_CODES:
                        errs.append(
                            f"{task_id}: raw truth_level_code "
                            f"{row['truth_level_code']!r} is outside T0-T5"
                        )
                    if row["truth_level"] != row["truth_level_code"]:
                        errs.append(
                            f"{task_id}: raw deprecated truth_level alias must equal "
                            "truth_level_code"
                        )
                    for field in (
                        "biological_question_id",
                        "task_family_id",
                        "truth_level_code",
                        "target_source_kind",
                        "truth_level",
                        "split_group_scope",
                    ):
                        if row[field] != task[field]:
                            errs.append(f"{task_id}: raw {field} disagrees with the task registry")
                    if task["intervention_pair_field"] is None:
                        if row["intervention_pair_id"] is not None:
                            errs.append(
                                f"{task_id}: non-T5 task requires a null raw "
                                "intervention_pair_id"
                            )
                    elif row["intervention_pair_id"] in (None, ""):
                        errs.append(
                            f"{task_id}: T5 task requires a row-specific raw "
                            "intervention_pair_id"
                        )
                    expected_factor_levels = {
                        **task["factor_levels"],
                        "input_condition": row["condition"],
                    }
                    if row["factor_levels"] != expected_factor_levels:
                        errs.append(
                            f"{task_id}: raw factor_levels disagree with the descriptive "
                            "interface contract"
                        )
                    split_group_scope = row["split_group_scope"]
                    split_group_id = row["split_group_id"]
                    if split_group_scope == SPLIT_SCOPE_UNAVAILABLE:
                        if split_group_id is not None:
                            errs.append(
                                f"{task_id}: unavailable split-group scope requires a null "
                                "split_group_id"
                            )
                    elif split_group_scope == SPLIT_SCOPE_ENTITY_PROXY:
                        if split_group_id != row["entity_id"]:
                            errs.append(
                                f"{task_id}: exact-entity proxy split_group_id must equal entity_id"
                            )
                    elif split_group_scope in BIOLOGICAL_SPLIT_SCOPES:
                        if split_group_id in (None, ""):
                            errs.append(
                                f"{task_id}: biological split-group scope requires a non-null "
                                "split_group_id"
                            )
                    else:
                        errs.append(
                            f"{task_id}: unregistered split_group_scope={split_group_scope!r}"
                        )
                    if row["source_label"] not in (0, 1) or row["target_label"] not in (0, 1):
                        errs.append(f"{task_id}: source_label and target_label must be binary")
                    expected_target = (
                        1 - row["source_label"]
                        if TASKS[task_id]["orient"] == "oppose"
                        else row["source_label"]
                    )
                    if row["target_label"] != expected_target:
                        errs.append(f"{task_id}: target_label does not match registered orientation")
                    probability = row["prob"]
                    if not (
                        isinstance(probability, (int, float))
                        and math.isfinite(probability)
                        and 0.0 <= probability <= 1.0
                    ):
                        errs.append(f"{task_id}: raw probability {probability!r} is outside [0,1]")
                    if row["prompt_version"] != PROMPT_VERSION:
                        errs.append(f"{task_id}: raw prompt_version is not {PROMPT_VERSION}")
                    if not isinstance(row["parse_valid"], bool):
                        errs.append(f"{task_id}: parse_valid must be boolean")
                    expected_parse_status = "parsed" if row["parse_valid"] else "invalid_or_refusal"
                    if row["parse_status"] != expected_parse_status:
                        errs.append(f"{task_id}: parse_status disagrees with parse_valid")
                    if not row["parse_valid"] and probability != 0.5:
                        errs.append(f"{task_id}: invalid responses must use the declared neutral 0.5 fallback")
                    reparsed_probability, reparsed_valid = parse_prob_with_status(row["output"])
                    if reparsed_valid != row["parse_valid"] or not np.isclose(
                        reparsed_probability,
                        probability,
                    ):
                        errs.append(f"{task_id}: stored parse result disagrees with raw output")
                    key = (row["task"], row["condition"])
                    raw_by_task_condition.setdefault(key, []).append(row)
                    raw_rows.append(row)
        except Exception as e:
            errs.append(f"raw.jsonl parse error: {e}")
        sc_only = set(sc) - raw_tasks
        if sc_only:
            warns.append(f"{len(sc_only)} task(s) in scorecard but absent from raw.jsonl: "
                         f"{', '.join(sorted(sc_only)[:5])}")
        if isinstance(man.get("seed"), int) and isinstance(man.get("n_per_task"), int):
            sampler = GroundBenchSampler(seed=man["seed"])
            for task_id in sc:
                if task_id not in TASKS:
                    continue
                expected_conditions = {
                    condition: rows
                    for condition, rows in sampler.task_condition_items(
                        task_id,
                        man["n_per_task"],
                    ).items()
                    if rows
                }
                actual_conditions = {
                    condition: rows
                    for (raw_task, condition), rows in raw_by_task_condition.items()
                    if raw_task == task_id and rows
                }
                if set(actual_conditions) != set(expected_conditions):
                    errs.append(
                        f"{task_id}: raw condition set does not match the registered sampler"
                    )
                    continue
                for condition, expected_rows in expected_conditions.items():
                    expected_by_entity = {
                        row["entity_id"]: row for row in expected_rows
                    }
                    actual_by_entity = {
                        row["entity_id"]: row for row in actual_conditions[condition]
                    }
                    if set(actual_by_entity) != set(expected_by_entity):
                        errs.append(
                            f"{task_id}: {condition} entity IDs do not match the registered sampler"
                        )
                        continue
                    for entity_id, expected in expected_by_entity.items():
                        actual = actual_by_entity[entity_id]
                        expected_target = (
                            1 - expected["label"]
                            if TASKS[task_id]["orient"] == "oppose"
                            else expected["label"]
                        )
                        contract_fields = {
                            "id": expected["id"],
                            "source_id": expected["source_id"],
                            "entity_id_scope": expected["entity_id_scope"],
                            "truth_level_code": expected["truth_level_code"],
                            "target_source_kind": expected["target_source_kind"],
                            "truth_level": expected["truth_level"],
                            "biological_question_id": expected["biological_question_id"],
                            "task_family_id": expected["task_family_id"],
                            "split_group_id": expected["split_group_id"],
                            "split_group_scope": expected["split_group_scope"],
                            "intervention_pair_id": expected["intervention_pair_id"],
                            "factor_levels": expected["factor_levels"],
                            "source_label": expected["label"],
                            "target_label": expected_target,
                        }
                        for field, value in contract_fields.items():
                            if actual.get(field) != value:
                                errs.append(
                                    f"{task_id}: {condition} {field} disagrees with "
                                    "the registered sampler"
                                )
                                break
        for task_id, rec in sc.items():
            matched = raw_by_task_condition.get((task_id, "matched"), [])
            if len(matched) != rec.get("n"):
                errs.append(f"{task_id}: scorecard n={rec.get('n')} but raw matched rows={len(matched)}")
            if len({row["entity_id"] for row in matched}) != len(matched):
                errs.append(f"{task_id}: duplicate entity_id in matched raw rows")
            valid_rate = round(
                sum(row["parse_valid"] for row in matched) / len(matched),
                3,
            ) if matched else None
            if valid_rate != rec.get("valid_response_rate"):
                errs.append(
                    f"{task_id}: valid_response_rate={rec.get('valid_response_rate')} "
                    f"but raw rows imply {valid_rate}"
                )
            if sum(not row["parse_valid"] for row in matched) != rec.get("n_invalid"):
                errs.append(f"{task_id}: n_invalid disagrees with raw rows")
            if matched:
                raw_y = np.asarray([row["target_label"] for row in matched], dtype=int)
                raw_prob = np.asarray([row["prob"] for row in matched], dtype=float)
                score_rng = _task_rng(man.get("seed", 0), task_id, "bootstrap")
                recomputed = {
                    "output_auroc": auroc(raw_prob, raw_y),
                    "brier": _brier(raw_prob, raw_y),
                    "log_loss": _log_loss(raw_prob, raw_y),
                    "probability_ece": probability_ece(raw_prob, raw_y),
                    "confidence_ece": ece(raw_prob, raw_y),
                    "aurc": aurc(raw_prob, raw_y),
                    "sel_acc_50": sel_acc(raw_prob, raw_y),
                    "sample_prevalence": float(raw_y.mean()),
                }
                for field, value in recomputed.items():
                    reported = rec.get(field)
                    if not (
                        isinstance(reported, (int, float))
                        and math.isfinite(reported)
                        and abs(float(reported) - value) <= 0.002
                    ):
                        errs.append(
                            f"{task_id}: {field}={reported!r} disagrees with raw rows "
                            f"({value:.4f})"
                        )
                expected_cis = {
                    "output_auroc_ci": ci(auroc, raw_prob, raw_y, score_rng),
                    "brier_ci": ci(_brier, raw_prob, raw_y, score_rng),
                    "log_loss_ci": ci(_log_loss, raw_prob, raw_y, score_rng),
                }
                for field, expected in expected_cis.items():
                    reported = rec.get(field)
                    if not (
                        isinstance(reported, list)
                        and len(reported) == 2
                        and np.allclose(reported, expected, atol=0.002, equal_nan=True)
                    ):
                        errs.append(f"{task_id}: {field} disagrees with raw rows")
                valid_mask = np.asarray(
                    [row["parse_valid"] for row in matched],
                    dtype=bool,
                )
                expected_valid_auc = (
                    round(auroc(raw_prob[valid_mask], raw_y[valid_mask]), 3)
                    if valid_mask.any() and len(set(raw_y[valid_mask])) > 1
                    else None
                )
                if rec.get("output_auroc_valid_only") != expected_valid_auc:
                    errs.append(
                        f"{task_id}: output_auroc_valid_only disagrees with raw rows"
                    )
            matched_ids = {row["entity_id"] for row in matched}
            matched_prob_by_id = {row["entity_id"]: row["prob"] for row in matched}
            matched_labels_by_id = {
                row["entity_id"]: (row["source_label"], row["target_label"])
                for row in matched
            }
            controls = {
                condition: rows
                for (raw_task, condition), rows in raw_by_task_condition.items()
                if raw_task == task_id and condition != "matched"
            }
            for condition, control_rows in controls.items():
                if not {row["entity_id"] for row in control_rows}.issubset(matched_ids):
                    errs.append(
                        f"{task_id}: {condition} rows are not a subset of matched entity IDs"
                    )
                    continue
                if len({row["entity_id"] for row in control_rows}) != len(control_rows):
                    errs.append(f"{task_id}: duplicate entity_id in {condition} raw rows")
                label_mismatches = [
                    row["entity_id"]
                    for row in control_rows
                    if (row["source_label"], row["target_label"])
                    != matched_labels_by_id[row["entity_id"]]
                ]
                if label_mismatches:
                    errs.append(
                        f"{task_id}: {condition} source/target labels do not match paired "
                        "matched rows"
                    )
                if rec.get(f"{condition}_n") != len(control_rows):
                    errs.append(
                        f"{task_id}: {condition}_n={rec.get(f'{condition}_n')} "
                        f"but raw rows={len(control_rows)}"
                    )
                control_valid_rate = round(
                    sum(row["parse_valid"] for row in control_rows) / len(control_rows),
                    3,
                )
                if rec.get(f"{condition}_valid_response_rate") != control_valid_rate:
                    errs.append(f"{task_id}: {condition}_valid_response_rate disagrees with raw rows")
                control_y = np.asarray([row["target_label"] for row in control_rows], dtype=int)
                matched_control_prob = np.asarray(
                    [matched_prob_by_id[row["entity_id"]] for row in control_rows],
                    dtype=float,
                )
                control_prob = np.asarray([row["prob"] for row in control_rows], dtype=float)
                if len(set(control_y)) < 2:
                    if any(rec.get(field) is not None for field in (
                        f"{condition}_auroc",
                        f"matched_minus_{condition}_auroc",
                        f"matched_minus_{condition}_auroc_ci",
                    )):
                        errs.append(
                            f"{task_id}: single-class {condition} rows must have null AUROC fields"
                        )
                    continue
                condition_auc = auroc(control_prob, control_y)
                condition_delta = auroc(matched_control_prob, control_y) - condition_auc
                if not (
                    isinstance(rec.get(f"{condition}_auroc"), (int, float))
                    and abs(float(rec[f"{condition}_auroc"]) - condition_auc) <= 0.002
                ):
                    errs.append(f"{task_id}: {condition}_auroc disagrees with raw rows")
                delta_field = f"matched_minus_{condition}_auroc"
                if not (
                    isinstance(rec.get(delta_field), (int, float))
                    and abs(float(rec[delta_field]) - condition_delta) <= 0.002
                ):
                    errs.append(f"{task_id}: {delta_field} disagrees with paired raw rows")
                expected_ci = paired_auroc_delta_ci(
                    matched_control_prob,
                    control_prob,
                    control_y,
                    _task_rng(man.get("seed", 0), task_id, f"{condition}-bootstrap"),
                )
                reported_ci = rec.get(f"{delta_field}_ci")
                if not (
                    isinstance(reported_ci, list)
                    and len(reported_ci) == 2
                    and np.allclose(reported_ci, expected_ci, atol=0.002, equal_nan=True)
                ):
                    errs.append(f"{task_id}: {delta_field}_ci disagrees with paired raw rows")

            scrambled = controls.get("scrambled", [])
            if scrambled:
                scrambled_y = np.asarray([row["target_label"] for row in scrambled], dtype=int)
                matched_scrambled_prob = np.asarray(
                    [matched_prob_by_id[row["entity_id"]] for row in scrambled],
                    dtype=float,
                )
                scrambled_prob = np.asarray([row["prob"] for row in scrambled], dtype=float)
                memo_delta = (
                    auroc(matched_scrambled_prob, scrambled_y)
                    - auroc(scrambled_prob, scrambled_y)
                )
                if not (
                    isinstance(rec.get("memo_delta"), (int, float))
                    and abs(float(rec["memo_delta"]) - memo_delta) <= 0.002
                ):
                    errs.append(f"{task_id}: memo_delta disagrees with paired raw rows")
            if rec.get("memo_n", 0) != len(scrambled):
                errs.append(f"{task_id}: memo_n={rec.get('memo_n')} but raw scrambled rows={len(scrambled)}")

        pair_groups = {}
        for task_id in sc:
            group = TASKS.get(task_id, {}).get("pair_group")
            if group:
                rows = raw_by_task_condition.get((task_id, "matched"), [])
                pair_groups.setdefault(group, {})[task_id] = {
                    row["entity_id"]: row["source_label"] for row in rows
                }
        for group, task_maps in pair_groups.items():
            if len(task_maps) < 2:
                continue
            first_task, first_map = next(iter(task_maps.items()))
            for task_id, entity_map in list(task_maps.items())[1:]:
                if entity_map != first_map:
                    errs.append(
                        f"pair_group {group}: {task_id} does not use the same entity-label map "
                        f"as {first_task}"
                    )
        pair_path = os.path.join(d, "pair_group_comparisons.json")
        pair_artifact = _load_json(pair_path, errs)
        if pair_artifact is not None:
            expected_pair_artifact = json.loads(json.dumps(
                build_pair_group_comparisons(sc, raw_rows, man.get("seed", 0))
            ))
            if pair_artifact != expected_pair_artifact:
                errs.append(
                    "pair_group_comparisons.json disagrees with entity-paired raw predictions"
                )
    return errs, warns


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pos:
        print("usage: python eval/validate_submission.py results/benchmark/<model> [--allow-partial]")
        sys.exit(2)
    d = pos[0]
    errs, warns = validate(d, allow_partial="--allow-partial" in sys.argv)
    for w in warns:
        print(f"WARN  {w}")
    for e in errs:
        print(f"FAIL  {e}")
    if errs:
        print(f"\n{len(errs)} error(s) in {d}. Not a valid submission.")
        sys.exit(1)
    print(f"\nOK: valid submission ({d}). {len(warns)} warning(s).")
    sys.exit(0)


if __name__ == "__main__":
    main()

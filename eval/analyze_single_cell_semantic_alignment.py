"""Decompose the paired single-cell gene-name effect into biological alignment and class bias.

The same 200 cells were scored with real gene symbols and globally consistent
anonymous feature IDs.  This analysis asks a stricter question than whether AUROC
increases: does revealing gene symbols move each cell's probability in the correct
direction, and does that movement track graded out-of-fold specialist evidence
within a cell class?

The analysis is deterministic and uses only existing per-item predictions.  It does
not infer training-corpus exposure, latent knowledge, or a causal activation route.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm, rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SINGLE_CELL_RESULTS = ROOT / "results" / "benchmark" / "single_cell"
SINGLE_CELL_DATA = ROOT / "signal" / "single_cell"

MODEL_SPECS = (
    ("claude-haiku-4-5-20251001", "Haiku 4.5"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-opus-4-8", "Opus 4.8"),
    ("gpt-4o", "GPT-4o"),
)
TASK_SPECS = (
    {
        "task": "cd8t_nk",
        "label": "CD8+ T vs NK",
        "first_class": "CD8+ T",
        "comparator_class": "NK",
        "data": SINGLE_CELL_DATA / "cd8t_nk.csv",
        "result_subdir": "",
    },
    {
        "task": "mono",
        "label": "CD14+ vs CD16+ monocyte",
        "first_class": "CD14+",
        "comparator_class": "CD16+",
        "data": SINGLE_CELL_DATA / "mono_cd14_fcgr3a.csv",
        "result_subdir": "mono",
    },
)

N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 50_000
ANALYSIS_SEED_NAMESPACE = "single-cell-semantic-alignment-v1"


class SemanticAlignmentError(ValueError):
    """Raised when paired predictions or source data are not internally coherent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join((ANALYSIS_SEED_NAMESPACE, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _load_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"label", "cell_sentence", "anon"}:
        raise SemanticAlignmentError(f"unexpected single-cell source schema: {path}")
    return rows


def _select_original_items(
    rows: list[dict[str, str]],
    *,
    n: int = 200,
) -> list[tuple[int, dict[str, str]]]:
    indexed = list(enumerate(rows))
    positive = [item for item in indexed if item[1]["label"] == "1"]
    negative = [item for item in indexed if item[1]["label"] == "0"]
    generator = np.random.default_rng(0)
    generator.shuffle(positive)
    generator.shuffle(negative)
    per_class = min(n // 2, len(positive), len(negative))
    selected = positive[:per_class] + negative[:per_class]
    if len(selected) != n:
        raise SemanticAlignmentError("single-cell source cannot reproduce the locked balanced sample")
    return selected


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_cell: dict[int, dict[str, Any]] = {}
    for record in records:
        if set(record) != {"cell", "condition", "label", "prob"}:
            raise SemanticAlignmentError(f"unexpected raw prediction schema: {path}")
        cell = record["cell"]
        condition = record["condition"]
        if type(cell) is not int or condition not in {"name", "anon"}:
            raise SemanticAlignmentError(f"invalid paired prediction identity: {path}")
        entry = by_cell.setdefault(cell, {"label": record["label"]})
        if entry["label"] != record["label"] or condition in entry:
            raise SemanticAlignmentError(f"duplicate or label-incoherent pair: {path}")
        entry[condition] = record["prob"]
    expected_cells = list(range(200))
    if sorted(by_cell) != expected_cells:
        raise SemanticAlignmentError(f"raw predictions do not contain the locked 200 cells: {path}")
    if any(set(by_cell[cell]) != {"label", "name", "anon"} for cell in expected_cells):
        raise SemanticAlignmentError(f"raw predictions are not complete pairs: {path}")
    labels = np.asarray([by_cell[cell]["label"] for cell in expected_cells], dtype=int)
    name = np.asarray([by_cell[cell]["name"] for cell in expected_cells], dtype=float)
    anon = np.asarray([by_cell[cell]["anon"] for cell in expected_cells], dtype=float)
    if not np.isfinite(name).all() or not np.isfinite(anon).all():
        raise SemanticAlignmentError(f"non-finite probability in {path}")
    if ((name < 0) | (name > 1) | (anon < 0) | (anon > 1)).any():
        raise SemanticAlignmentError(f"probability outside [0,1] in {path}")
    return labels, name, anon


def _oof_specialist_margin(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, float]:
    texts = np.asarray([row["cell_sentence"] for row in rows], dtype=object)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    margins = np.empty(len(rows), dtype=float)
    for train, test in splitter.split(texts, labels):
        vectorizer = CountVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            binary=True,
        )
        train_matrix = vectorizer.fit_transform(texts[train])
        test_matrix = vectorizer.transform(texts[test])
        classifier = LogisticRegression(
            max_iter=2_000,
            solver="liblinear",
            random_state=0,
        )
        classifier.fit(train_matrix, labels[train])
        margins[test] = classifier.decision_function(test_matrix)
    return margins, float(roc_auc_score(labels, margins))


def _delong_auc_difference(
    labels: np.ndarray,
    name: np.ndarray,
    anon: np.ndarray,
) -> dict[str, float]:
    positive = labels == 1
    negative = labels == 0
    score_matrix = np.vstack((name, anon))
    v10_rows = []
    v01_rows = []
    aucs = []
    for scores in score_matrix:
        pos = scores[positive]
        neg = scores[negative]
        comparisons = (pos[:, None] > neg[None, :]).astype(float)
        comparisons += 0.5 * (pos[:, None] == neg[None, :])
        v10 = comparisons.mean(axis=1)
        v01 = comparisons.mean(axis=0)
        v10_rows.append(v10)
        v01_rows.append(v01)
        aucs.append(float(v10.mean()))
    covariance = np.cov(np.vstack(v10_rows), ddof=1) / positive.sum()
    covariance += np.cov(np.vstack(v01_rows), ddof=1) / negative.sum()
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    standard_error = math.sqrt(max(variance, 0.0))
    difference = aucs[0] - aucs[1]
    if standard_error == 0:
        p_value = 0.0 if difference != 0 else 1.0
    else:
        p_value = float(2 * norm.sf(abs(difference / standard_error)))
    return {
        "name_auroc": aucs[0],
        "anon_auroc": aucs[1],
        "auroc_difference": difference,
        "auroc_difference_ci95_lower": difference - 1.96 * standard_error,
        "auroc_difference_ci95_upper": difference + 1.96 * standard_error,
        "auroc_difference_p_value": p_value,
    }


def _stratified_rank_correlation(
    specialist_margin: np.ndarray,
    probability_delta: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    specialist_ranks = np.zeros(len(labels), dtype=float)
    delta_ranks = np.zeros(len(labels), dtype=float)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        specialist_ranks[indices] = rankdata(
            specialist_margin[indices],
            method="average",
        )
        delta_ranks[indices] = rankdata(
            probability_delta[indices],
            method="average",
        )
        specialist_ranks[indices] -= specialist_ranks[indices].mean()
        delta_ranks[indices] -= delta_ranks[indices].mean()
    denominator = np.linalg.norm(specialist_ranks) * np.linalg.norm(delta_ranks)
    correlation = float(specialist_ranks @ delta_ranks / denominator) if denominator > 0 else 0.0
    return correlation, specialist_ranks, delta_ranks


def _within_class_permutation_p_value(
    specialist_ranks: np.ndarray,
    delta_ranks: np.ndarray,
    labels: np.ndarray,
    *,
    observed: float,
    seed: int,
) -> float:
    denominator = np.linalg.norm(specialist_ranks) * np.linalg.norm(delta_ranks)
    if denominator == 0:
        return 1.0
    generator = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    exceedances = 0
    permuted = delta_ranks.copy()
    for _ in range(N_PERMUTATIONS):
        for indices in class_indices:
            permuted[indices] = generator.permutation(delta_ranks[indices])
        statistic = float(specialist_ranks @ permuted / denominator)
        exceedances += abs(statistic) >= abs(observed)
    return (exceedances + 1) / (N_PERMUTATIONS + 1)


def _bootstrap_contrasts(
    probability_delta: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    positive = probability_delta[labels == 1]
    negative = probability_delta[labels == 0]
    generator = np.random.default_rng(seed)
    positive_draws = generator.integers(
        0,
        len(positive),
        size=(N_BOOTSTRAP, len(positive)),
    )
    negative_draws = generator.integers(
        0,
        len(negative),
        size=(N_BOOTSTRAP, len(negative)),
    )
    positive_means = positive[positive_draws].mean(axis=1)
    negative_means = negative[negative_draws].mean(axis=1)
    separation = positive_means - negative_means
    prior = 0.5 * (positive_means + negative_means)

    def interval(values: np.ndarray) -> tuple[float, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return float(lower), float(upper)

    separation_lower, separation_upper = interval(separation)
    prior_lower, prior_upper = interval(prior)
    positive_lower, positive_upper = interval(positive_means)
    comparator_lower, comparator_upper = interval(-negative_means)
    return {
        "first_class_correct_shift_ci95_lower": positive_lower,
        "first_class_correct_shift_ci95_upper": positive_upper,
        "comparator_class_correct_shift_ci95_lower": comparator_lower,
        "comparator_class_correct_shift_ci95_upper": comparator_upper,
        "class_separation_gain_ci95_lower": separation_lower,
        "class_separation_gain_ci95_upper": separation_upper,
        "class_prior_shift_ci95_lower": prior_lower,
        "class_prior_shift_ci95_upper": prior_upper,
    }


def _reference_class_strength(
    specialist_margin: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    first_class = specialist_margin[labels == 1]
    comparator_class = -specialist_margin[labels == 0]
    first_mean = float(first_class.mean())
    comparator_mean = float(comparator_class.mean())
    generator = np.random.default_rng(seed)
    first_draws = generator.integers(
        0,
        len(first_class),
        size=(N_BOOTSTRAP, len(first_class)),
    )
    comparator_draws = generator.integers(
        0,
        len(comparator_class),
        size=(N_BOOTSTRAP, len(comparator_class)),
    )
    difference_draws = first_class[first_draws].mean(axis=1) - comparator_class[comparator_draws].mean(axis=1)
    lower, upper = np.quantile(difference_draws, [0.025, 0.975])
    return {
        "first_class_specialist_correct_margin_mean": first_mean,
        "comparator_class_specialist_correct_margin_mean": comparator_mean,
        "first_minus_comparator_specialist_margin_difference": (first_mean - comparator_mean),
        "first_minus_comparator_specialist_margin_ci95_lower": float(lower),
        "first_minus_comparator_specialist_margin_ci95_upper": float(upper),
    }


def _holm_adjust(records: list[dict[str, Any]], field: str, output: str) -> None:
    ordered = sorted(
        enumerate(records),
        key=lambda item: item[1][field],
    )
    adjusted = [1.0] * len(records)
    running = 0.0
    total = len(records)
    for rank, (original_index, record) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * record[field])
        running = max(running, candidate)
        adjusted[original_index] = running
    for record, value in zip(records, adjusted, strict=True):
        record[output] = value


def _proper_scoring(
    labels: np.ndarray,
    name: np.ndarray,
    anon: np.ndarray,
) -> dict[str, float | int]:
    epsilon = 1e-6

    def brier(probabilities: np.ndarray) -> float:
        return float(np.mean((probabilities - labels) ** 2))

    def log_loss(probabilities: np.ndarray) -> float:
        clipped = np.clip(probabilities, epsilon, 1 - epsilon)
        return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))

    name_correct = (name > 0.5).astype(int) == labels
    anon_correct = (anon > 0.5).astype(int) == labels
    corrected = int(np.sum(name_correct & ~anon_correct))
    corrupted = int(np.sum(~name_correct & anon_correct))
    discordant = corrected + corrupted
    mcnemar_p = (
        float(
            binomtest(
                min(corrected, corrupted),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "name_accuracy": float(name_correct.mean()),
        "anon_accuracy": float(anon_correct.mean()),
        "name_brier": brier(name),
        "anon_brier": brier(anon),
        "brier_improvement_anon_minus_name": brier(anon) - brier(name),
        "name_log_loss": log_loss(name),
        "anon_log_loss": log_loss(anon),
        "log_loss_improvement_anon_minus_name": log_loss(anon) - log_loss(name),
        "threshold_corrected_items": corrected,
        "threshold_corrupted_items": corrupted,
        "mcnemar_exact_p_value": mcnemar_p,
    }


def analyze() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    task_aggregates: list[dict[str, Any]] = []
    input_files: dict[str, str] = {}
    specialist_by_task: dict[str, dict[str, Any]] = {}
    for task_spec in TASK_SPECS:
        source_path = task_spec["data"]
        source_rows = _load_source_rows(source_path)
        selected = _select_original_items(source_rows)
        selected_indices = np.asarray([index for index, _ in selected], dtype=int)
        selected_labels = np.asarray(
            [int(row["label"]) for _, row in selected],
            dtype=int,
        )
        specialist_margin, specialist_auroc = _oof_specialist_margin(source_rows)
        selected_specialist_margin = specialist_margin[selected_indices]
        specialist_by_task[task_spec["task"]] = {
            "bag_of_gene_identity_oof_auroc": specialist_auroc,
            "source_rows": len(source_rows),
        }
        input_files[str(source_path.relative_to(ROOT))] = _sha256(source_path)
        task_name_probabilities = []
        task_anon_probabilities = []

        for model_id, model_label in MODEL_SPECS:
            result_path = SINGLE_CELL_RESULTS / task_spec["result_subdir"] / f"{model_id}_raw.jsonl"
            labels, name, anon = _load_predictions(result_path)
            if not np.array_equal(labels, selected_labels):
                raise SemanticAlignmentError(f"raw labels do not replay the locked source sampling: {result_path}")
            input_files[str(result_path.relative_to(ROOT))] = _sha256(result_path)
            task_name_probabilities.append(name)
            task_anon_probabilities.append(anon)
            delta = name - anon
            positive_delta = float(delta[labels == 1].mean())
            negative_delta = float(delta[labels == 0].mean())
            comparator_correct_shift = -negative_delta
            separation_gain = positive_delta - negative_delta
            prior_shift = 0.5 * (positive_delta + negative_delta)
            comparator_share = comparator_correct_shift / separation_gain if separation_gain > 0 else math.nan
            rank_correlation, specialist_ranks, delta_ranks = _stratified_rank_correlation(
                selected_specialist_margin,
                delta,
                labels,
            )
            record: dict[str, Any] = {
                "task": task_spec["task"],
                "task_label": task_spec["label"],
                "first_class": task_spec["first_class"],
                "comparator_class": task_spec["comparator_class"],
                "model": model_id,
                "model_label": model_label,
                "n": len(labels),
                **_delong_auc_difference(labels, name, anon),
                "first_class_correct_shift": positive_delta,
                "comparator_class_correct_shift": comparator_correct_shift,
                "class_separation_gain": separation_gain,
                "class_prior_shift_toward_first_class": prior_shift,
                "comparator_share_of_class_separation_gain": comparator_share,
                **_bootstrap_contrasts(
                    delta,
                    labels,
                    seed=_seed(task_spec["task"], model_id, "bootstrap"),
                ),
                "within_class_specialist_delta_rank_correlation": rank_correlation,
                "within_class_specialist_delta_permutation_p_value": (
                    _within_class_permutation_p_value(
                        specialist_ranks,
                        delta_ranks,
                        labels,
                        observed=rank_correlation,
                        seed=_seed(task_spec["task"], model_id, "permutation"),
                    )
                ),
                **_proper_scoring(labels, name, anon),
            }
            records.append(record)

        model_average_name = np.mean(np.vstack(task_name_probabilities), axis=0)
        model_average_anon = np.mean(np.vstack(task_anon_probabilities), axis=0)
        model_average_delta = model_average_name - model_average_anon
        model_deltas = [
            name - anon
            for name, anon in zip(
                task_name_probabilities,
                task_anon_probabilities,
                strict=True,
            )
        ]
        pairwise_model_correlations = [
            _stratified_rank_correlation(
                model_deltas[first],
                model_deltas[second],
                selected_labels,
            )[0]
            for first in range(len(model_deltas))
            for second in range(first)
        ]
        positive_delta = float(model_average_delta[selected_labels == 1].mean())
        negative_delta = float(model_average_delta[selected_labels == 0].mean())
        comparator_correct_shift = -negative_delta
        separation_gain = positive_delta - negative_delta
        rank_correlation, specialist_ranks, delta_ranks = _stratified_rank_correlation(
            selected_specialist_margin,
            model_average_delta,
            selected_labels,
        )
        task_aggregates.append(
            {
                "task": task_spec["task"],
                "task_label": task_spec["label"],
                "first_class": task_spec["first_class"],
                "comparator_class": task_spec["comparator_class"],
                "models_averaged": len(MODEL_SPECS),
                "n_shared_cells": len(selected_labels),
                **_delong_auc_difference(
                    selected_labels,
                    model_average_name,
                    model_average_anon,
                ),
                "first_class_correct_shift": positive_delta,
                "comparator_class_correct_shift": comparator_correct_shift,
                "class_separation_gain": separation_gain,
                "class_prior_shift_toward_first_class": (0.5 * (positive_delta + negative_delta)),
                "comparator_share_of_class_separation_gain": (
                    comparator_correct_shift / separation_gain if separation_gain > 0 else math.nan
                ),
                **_bootstrap_contrasts(
                    model_average_delta,
                    selected_labels,
                    seed=_seed(task_spec["task"], "model-average", "bootstrap"),
                ),
                "within_class_specialist_delta_rank_correlation": rank_correlation,
                "within_class_specialist_delta_permutation_p_value": (
                    _within_class_permutation_p_value(
                        specialist_ranks,
                        delta_ranks,
                        selected_labels,
                        observed=rank_correlation,
                        seed=_seed(task_spec["task"], "model-average", "permutation"),
                    )
                ),
                "mean_pairwise_cross_model_delta_rank_correlation": float(np.mean(pairwise_model_correlations)),
                "pairwise_cross_model_delta_rank_correlations": (pairwise_model_correlations),
                **_reference_class_strength(
                    selected_specialist_margin,
                    selected_labels,
                    seed=_seed(task_spec["task"], "reference-strength", "bootstrap"),
                ),
            }
        )

    _holm_adjust(
        records,
        "auroc_difference_p_value",
        "auroc_difference_holm_p_value",
    )
    _holm_adjust(
        records,
        "within_class_specialist_delta_permutation_p_value",
        "within_class_specialist_delta_holm_p_value",
    )
    _holm_adjust(
        task_aggregates,
        "auroc_difference_p_value",
        "auroc_difference_holm_p_value",
    )
    _holm_adjust(
        task_aggregates,
        "within_class_specialist_delta_permutation_p_value",
        "within_class_specialist_delta_holm_p_value",
    )
    total_comparator_alignment = sum(record["comparator_class_correct_shift"] for record in records)
    total_separation_gain = sum(record["class_separation_gain"] for record in records)
    significant_auroc = sum(record["auroc_difference_holm_p_value"] < 0.05 for record in records)
    significant_within_class = sum(record["within_class_specialist_delta_holm_p_value"] < 0.05 for record in records)
    comparator_dominant = sum(record["comparator_share_of_class_separation_gain"] >= 0.65 for record in records)
    first_class_wrong_direction = sum(record["first_class_correct_shift"] < 0 for record in records)
    return {
        "analysis_id": ANALYSIS_SEED_NAMESPACE,
        "claim_scope": (
            "paired_output_probability_decomposition_only_no_training_exposure_"
            "latent_knowledge_or_causal_activation_claim"
        ),
        "bootstrap_draws": N_BOOTSTRAP,
        "within_class_permutations": N_PERMUTATIONS,
        "input_sha256": input_files,
        "specialist_reference": specialist_by_task,
        "task_aggregates": task_aggregates,
        "records": records,
        "summary": {
            "model_task_cells": len(records),
            "positive_auroc_differences": sum(record["auroc_difference"] > 0 for record in records),
            "holm_significant_auroc_differences": significant_auroc,
            "comparator_dominant_cells_at_least_65_percent": comparator_dominant,
            "first_class_wrong_direction_cells": first_class_wrong_direction,
            "aggregate_comparator_share_of_class_separation_gain": (total_comparator_alignment / total_separation_gain),
            "holm_significant_within_class_graded_alignment_cells": (significant_within_class),
        },
    }


def _format_interval(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Single-cell semantic grounding is graded but polarity-asymmetric",
        "",
        "This paired reanalysis asks whether replacing anonymous feature IDs with real gene",
        "symbols moves each cell toward its correct class, rather than merely increasing AUROC.",
        "It uses the existing 200-cell predictions for four models on two PBMC contrasts.",
        "",
        "## Main result",
        "",
        (
            f"All {summary['positive_auroc_differences']}/{summary['model_task_cells']} "
            "model-task cells have a positive name-minus-anonymous AUROC difference; "
            f"{summary['holm_significant_auroc_differences']} remain significant after "
            "Holm correction. The probability update is strongly class-asymmetric: "
            f"{100 * summary['aggregate_comparator_share_of_class_separation_gain']:.1f}% "
            "of the aggregate mean class-separation gain comes from lowering the "
            "first-class probability on comparator-class cells (NK or CD16+)."
        ),
        "",
        (
            f"In {summary['comparator_dominant_cells_at_least_65_percent']}/"
            f"{summary['model_task_cells']} cells, the comparator class contributes at "
            "least 65% of the separation gain. In "
            f"{summary['first_class_wrong_direction_cells']} cells, revealing gene names "
            "actually lowers the mean probability of the correct first class; AUROC still "
            "improves because the comparator-class probability falls more."
        ),
        "",
        (
            "After removing between-class differences, "
            f"{summary['holm_significant_within_class_graded_alignment_cells']}/"
            f"{summary['model_task_cells']} model-task cells show a Holm-significant "
            "correlation between the name-induced probability update and graded "
            "out-of-fold bag-of-gene specialist evidence."
        ),
        "",
        (
            "The 8 model-task cells reuse the same 200 cells within each biological task; "
            "they are not 8 independent datasets. The model-averaged task summaries below "
            "therefore provide the cleaner unit for describing the shared pattern."
        ),
        "",
        "## Model-averaged task decomposition",
        "",
        (
            "| task | ensemble AUROC gap (95% CI) | first-class correct shift | "
            "comparator correct shift | comparator share | prior shift | "
            "within-class evidence rho | Holm p |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["task_aggregates"]:
        lines.append(
            "| {task} | {gap:.3f} {gap_ci} | {first:.3f} {first_ci} | "
            "{comparator:.3f} {comparator_ci} | {share:.1%} | {prior:.3f} | "
            "{rho:.3f} | {p:.2g} |".format(
                task=record["task_label"],
                gap=record["auroc_difference"],
                gap_ci=_format_interval(
                    record["auroc_difference_ci95_lower"],
                    record["auroc_difference_ci95_upper"],
                ),
                first=record["first_class_correct_shift"],
                first_ci=_format_interval(
                    record["first_class_correct_shift_ci95_lower"],
                    record["first_class_correct_shift_ci95_upper"],
                ),
                comparator=record["comparator_class_correct_shift"],
                comparator_ci=_format_interval(
                    record["comparator_class_correct_shift_ci95_lower"],
                    record["comparator_class_correct_shift_ci95_upper"],
                ),
                share=record["comparator_share_of_class_separation_gain"],
                prior=record["class_prior_shift_toward_first_class"],
                rho=record["within_class_specialist_delta_rank_correlation"],
                p=record["within_class_specialist_delta_holm_p_value"],
            )
        )
    lines.extend(
        [
            "",
            (
                "A comparator share above 100% is a signed decomposition, not a variance "
                "fraction: it occurs when the first-class mean update points in the wrong "
                "direction while the larger comparator update still improves separation."
            ),
            "",
            (
                "The out-of-fold reference did not show weaker evidence for the first class. "
                "Its mean correct-oriented margins were "
                + "; ".join(
                    (
                        f"{record['task_label']}: "
                        f"{record['first_class_specialist_correct_margin_mean']:.2f} "
                        f"({record['first_class']}) versus "
                        f"{record['comparator_class_specialist_correct_margin_mean']:.2f} "
                        f"({record['comparator_class']})"
                    )
                    for record in result["task_aggregates"]
                )
                + ". The comparator-dominant LLM update therefore is not explained by a "
                "weaker first class under this model-independent reference."
            ),
            "",
            (
                "The item-level update is also moderately shared across models: the mean "
                "within-class pairwise rank correlation of name-minus-anonymous deltas is "
                + " and ".join(
                    f"{record['mean_pairwise_cross_model_delta_rank_correlation']:.3f} for {record['task_label']}"
                    for record in result["task_aggregates"]
                )
                + "."
            ),
            "",
            "## Per-model decomposition",
            "",
            (
                "| task | model | AUROC gap (95% CI) | first-class correct shift | "
                "comparator correct shift | comparator share | prior shift | "
                "within-class evidence rho | Brier improvement |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in result["records"]:
        lines.append(
            "| {task} | {model} | {gap:.3f} {gap_ci} | {first:.3f} {first_ci} | "
            "{comparator:.3f} {comparator_ci} | {share:.1%} | {prior:.3f} | "
            "{rho:.3f} | {brier:.3f} |".format(
                task=record["task_label"],
                model=record["model_label"],
                gap=record["auroc_difference"],
                gap_ci=_format_interval(
                    record["auroc_difference_ci95_lower"],
                    record["auroc_difference_ci95_upper"],
                ),
                first=record["first_class_correct_shift"],
                first_ci=_format_interval(
                    record["first_class_correct_shift_ci95_lower"],
                    record["first_class_correct_shift_ci95_upper"],
                ),
                comparator=record["comparator_class_correct_shift"],
                comparator_ci=_format_interval(
                    record["comparator_class_correct_shift_ci95_lower"],
                    record["comparator_class_correct_shift_ci95_upper"],
                ),
                share=record["comparator_share_of_class_separation_gain"],
                prior=record["class_prior_shift_toward_first_class"],
                rho=record["within_class_specialist_delta_rank_correlation"],
                brier=record["brier_improvement_anon_minus_name"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The named-symbol advantage is real in these paired outputs, but it is not a",
            "uniform increase in cell-level biological correctness. Much of the effect is",
            "a one-sided probability update toward the comparator class. For CD8/NK in particular, some",
            "models reduce P(CD8) for both CD8 and NK cells, but reduce it much more for NK.",
            "That improves ranking and aggregate proper scores while harming the mean",
            "direction of the CD8 update.",
            "",
            "This separates two hypotheses that AUROC alone conflates:",
            "",
            "1. **Class anchoring:** recognizable markers trigger a class prototype or prompt-side",
            "   prior shift.",
            "2. **Graded semantic grounding:** within a class, probability updates track the",
            "   strength of model-independent, out-of-fold bag-of-gene evidence for each cell.",
            "",
            "The within-class permutation result reports how much evidence supports the second",
            "hypothesis. Neither result identifies training exposure, latent knowledge, or a",
            "causal activation-to-output route.",
            "",
            "## Follow-up intervention",
            "",
            "The answer roles and probability orientation were subsequently reversed on the",
            "exact same cells for Haiku 4.5. The pilot identifies a positive graded semantic",
            "correction together with a substantially larger prompt-role prior; see",
            "[the orientation-swap result](orientation_swap/claude-haiku-4-5-20251001.md).",
            "",
            "That two-form swap changes both list position and which class is queried. A pure",
            "position test still requires four contemporaneous forms crossing order (A/B, B/A)",
            "with queried target (P(A), P(B)).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired gene-name probability updates in the single-cell arm")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=SINGLE_CELL_RESULTS / "semantic_alignment.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=SINGLE_CELL_RESULTS / "semantic_alignment.md",
    )
    args = parser.parse_args()

    result = analyze()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "markdown_out": str(args.markdown_out),
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

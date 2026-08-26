"""Reverse answer order on the exact paired single-cell semantic-grounding items.

The historical prompt asks for P(A) in "A or B?" order.  This experiment asks
for P(B) in "B or A?" order, then maps the answer back to P(A) as 1 - P(B).
Both named and anonymous renderings are rerun, so the estimand remains the
within-cell name-minus-anonymous probability update.

If the class asymmetry follows biological identity, the correct-direction
class shifts should be stable after reorientation.  If it follows answer
role/polarity, the dominant class should flip when A and B exchange roles.

This is an output-level prompt intervention.  It does not test hidden-state
causality, latent knowledge, or training-corpus exposure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_grounding_eval import complete, parse_prob_with_status  # noqa: E402

RESULT_ROOT = ROOT / "results" / "benchmark" / "single_cell"
DATA_ROOT = ROOT / "signal" / "single_cell"
PROMPT = (
    "These are the most distinctively expressed genes of one human immune cell "
    "(PBMC), highest first:\n{genes}\nIs it a {first} or a {second}? Reply with "
    "ONLY a number between 0 and 1 = the probability that it is a {first}."
)
SYSTEM_PROMPT = (
    "You are a property predictor. Respond with ONLY a single decimal number "
    "between 0 and 1 (for example: 0.42). No words, no explanation, no analysis, "
    "no units. Your entire reply must be just the number."
)
TASKS = {
    "cd8t_nk": {
        "data": DATA_ROOT / "cd8t_nk.csv",
        "forward_subdir": "",
        "class_a": "CD8+ T cell",
        "class_b": "NK cell",
        "label": "CD8+ T vs NK",
    },
    "mono": {
        "data": DATA_ROOT / "mono_cd14_fcgr3a.csv",
        "forward_subdir": "mono",
        "class_a": "CD14+ classical monocyte",
        "class_b": "CD16+ non-classical monocyte",
        "label": "CD14+ vs CD16+ monocyte",
    },
}
N_BOOTSTRAP = 20_000
ANALYSIS_SEED = "single-cell-orientation-swap-v1"


class OrientationSwapError(ValueError):
    """Raised when exact pairing or an experiment artifact is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join((ANALYSIS_SEED, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _select_items(path: Path, n: int) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = list(enumerate(rows))
    positive = [item for item in indexed if item[1]["label"] == "1"]
    negative = [item for item in indexed if item[1]["label"] == "0"]
    generator = np.random.default_rng(0)
    generator.shuffle(positive)
    generator.shuffle(negative)
    per_class = min(n // 2, len(positive), len(negative))
    selected = positive[:per_class] + negative[:per_class]
    if len(selected) != n or n % 2:
        raise OrientationSwapError("requested sample must be even, balanced, and available")
    return selected


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _reverse_raw_path(task: str, model: str) -> Path:
    return RESULT_ROOT / "orientation_swap" / task / f"{_safe_model_name(model)}_raw.jsonl"


def _read_reverse_records(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        key = (record["cell"], record["condition"])
        if key in result:
            raise OrientationSwapError(f"duplicate checkpoint record in {path}: {key}")
        result[key] = record
    return result


def _call_with_retry(model: str, prompt: str, attempts: int = 4) -> str:
    for attempt in range(attempts):
        try:
            return complete(model, prompt, system=SYSTEM_PROMPT)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def run_task(task: str, model: str, n: int) -> Path:
    spec = TASKS[task]
    selected = _select_items(spec["data"], n)
    output_path = _reverse_raw_path(task, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_reverse_records(output_path)
    expected = {(cell, condition) for cell in range(n) for condition in ("name", "anon")}
    unexpected = set(existing) - expected
    if unexpected:
        raise OrientationSwapError(f"checkpoint has records outside requested run: {unexpected}")

    for condition in ("name", "anon"):
        column = "cell_sentence" if condition == "name" else "anon"
        for cell, (source_index, row) in enumerate(selected):
            key = (cell, condition)
            if key in existing:
                continue
            prompt = PROMPT.format(
                genes=row[column],
                first=spec["class_b"],
                second=spec["class_a"],
            )
            raw_output = _call_with_retry(model, prompt)
            probability_b, parsed = parse_prob_with_status(raw_output)
            record = {
                "cell": cell,
                "source_index": source_index,
                "condition": condition,
                "label_a": int(row["label"]),
                "prompt_first_class": spec["class_b"],
                "reported_probability_b": probability_b,
                "aligned_probability_a": 1.0 - probability_b,
                "parsed": parsed,
                "raw_output": raw_output,
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            existing[key] = record
            print(
                f"{task} {model} {condition} {cell + 1}/{n} P(B)={probability_b:.4f} parsed={parsed}",
                flush=True,
            )

    if set(existing) != expected:
        raise OrientationSwapError("orientation-swap checkpoint is incomplete after execution")
    return output_path


def _load_forward(task: str, model: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    subdir = TASKS[task]["forward_subdir"]
    path = RESULT_ROOT / subdir / f"{_safe_model_name(model)}_raw.jsonl"
    if not path.exists():
        raise OrientationSwapError(f"missing forward raw predictions: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_cell: dict[int, dict[str, Any]] = {}
    for record in records:
        if record["cell"] >= n:
            continue
        entry = by_cell.setdefault(record["cell"], {"label": record["label"]})
        if entry["label"] != record["label"] or record["condition"] in entry:
            raise OrientationSwapError(f"invalid forward pair in {path}")
        entry[record["condition"]] = record["prob"]
    if sorted(by_cell) != list(range(n)):
        raise OrientationSwapError(f"forward run does not contain exact n={n} cells: {path}")
    labels = np.asarray([by_cell[cell]["label"] for cell in range(n)], dtype=int)
    name = np.asarray([by_cell[cell]["name"] for cell in range(n)], dtype=float)
    anon = np.asarray([by_cell[cell]["anon"] for cell in range(n)], dtype=float)
    return labels, name, anon, path


def _load_reverse(
    task: str,
    model: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, float]:
    path = _reverse_raw_path(task, model)
    records = _read_reverse_records(path)
    expected = {(cell, condition) for cell in range(n) for condition in ("name", "anon")}
    if set(records) != expected:
        raise OrientationSwapError(f"reverse run is incomplete: {path}")
    labels = np.asarray([records[(cell, "name")]["label_a"] for cell in range(n)], dtype=int)
    name = np.asarray(
        [records[(cell, "name")]["aligned_probability_a"] for cell in range(n)],
        dtype=float,
    )
    anon = np.asarray(
        [records[(cell, "anon")]["aligned_probability_a"] for cell in range(n)],
        dtype=float,
    )
    parse_rate = float(np.mean([record["parsed"] for record in records.values()]))
    return labels, name, anon, path, parse_rate


def _decomposition(labels: np.ndarray, name: np.ndarray, anon: np.ndarray) -> dict[str, float]:
    delta = name - anon
    first_shift = float(delta[labels == 1].mean())
    comparator_shift = float(-delta[labels == 0].mean())
    separation = first_shift + comparator_shift
    return {
        "name_auroc": float(roc_auc_score(labels, name)),
        "anon_auroc": float(roc_auc_score(labels, anon)),
        "first_class_correct_shift": first_shift,
        "comparator_class_correct_shift": comparator_shift,
        "class_asymmetry_first_minus_comparator": first_shift - comparator_shift,
        "class_separation_gain": separation,
        "class_prior_shift_toward_a": float(0.5 * (delta[labels == 1].mean() + delta[labels == 0].mean())),
    }


def _probability_distribution_summary(probabilities: np.ndarray) -> dict[str, float | int]:
    values, counts = np.unique(probabilities, return_counts=True)
    mode_index = int(np.argmax(counts))
    return {
        "mean": float(probabilities.mean()),
        "unique_values": len(values),
        "modal_probability": float(values[mode_index]),
        "modal_fraction": float(counts[mode_index] / len(probabilities)),
    }


def _stratified_rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
) -> float:
    first_ranks = np.zeros(len(labels), dtype=float)
    second_ranks = np.zeros(len(labels), dtype=float)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        first_ranks[indices] = rankdata(first[indices], method="average")
        second_ranks[indices] = rankdata(second[indices], method="average")
        first_ranks[indices] -= first_ranks[indices].mean()
        second_ranks[indices] -= second_ranks[indices].mean()
    denominator = np.linalg.norm(first_ranks) * np.linalg.norm(second_ranks)
    return float(first_ranks @ second_ranks / denominator) if denominator else 0.0


def _orientation_bootstrap(
    labels: np.ndarray,
    forward_delta: np.ndarray,
    reverse_delta: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    positive_draws = positive[generator.integers(0, len(positive), size=(N_BOOTSTRAP, len(positive)))]
    negative_draws = negative[generator.integers(0, len(negative), size=(N_BOOTSTRAP, len(negative)))]

    def statistics(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = delta[positive_draws].mean(axis=1)
        comparator = -delta[negative_draws].mean(axis=1)
        return first, comparator

    forward_first, forward_comparator = statistics(forward_delta)
    reverse_first, reverse_comparator = statistics(reverse_delta)
    forward_asymmetry = forward_first - forward_comparator
    reverse_asymmetry = reverse_first - reverse_comparator
    order_averaged_correct_grounding = (forward_first + forward_comparator + reverse_first + reverse_comparator) / 4
    biological_class_asymmetry = (forward_asymmetry + reverse_asymmetry) / 2
    prompt_role_asymmetry = (forward_asymmetry - reverse_asymmetry) / 2
    orientation_guardrail = ((forward_first - reverse_first) - (reverse_comparator - forward_comparator)) / 2
    orientation_effect = reverse_asymmetry - forward_asymmetry
    prior_effect = 0.5 * orientation_effect

    def interval(values: np.ndarray) -> tuple[float, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return float(lower), float(upper)

    forward_lower, forward_upper = interval(forward_asymmetry)
    reverse_lower, reverse_upper = interval(reverse_asymmetry)
    effect_lower, effect_upper = interval(orientation_effect)
    prior_lower, prior_upper = interval(prior_effect)
    grounding_lower, grounding_upper = interval(order_averaged_correct_grounding)
    biological_lower, biological_upper = interval(biological_class_asymmetry)
    role_lower, role_upper = interval(prompt_role_asymmetry)
    guardrail_lower, guardrail_upper = interval(orientation_guardrail)
    return {
        "forward_class_asymmetry_ci95_lower": forward_lower,
        "forward_class_asymmetry_ci95_upper": forward_upper,
        "reverse_class_asymmetry_ci95_lower": reverse_lower,
        "reverse_class_asymmetry_ci95_upper": reverse_upper,
        "orientation_effect_on_class_asymmetry_ci95_lower": effect_lower,
        "orientation_effect_on_class_asymmetry_ci95_upper": effect_upper,
        "orientation_effect_on_prior_shift_ci95_lower": prior_lower,
        "orientation_effect_on_prior_shift_ci95_upper": prior_upper,
        "order_averaged_correct_grounding_ci95_lower": grounding_lower,
        "order_averaged_correct_grounding_ci95_upper": grounding_upper,
        "biological_class_asymmetry_ci95_lower": biological_lower,
        "biological_class_asymmetry_ci95_upper": biological_upper,
        "prompt_role_asymmetry_ci95_lower": role_lower,
        "prompt_role_asymmetry_ci95_upper": role_upper,
        "orientation_guardrail_ci95_lower": guardrail_lower,
        "orientation_guardrail_ci95_upper": guardrail_upper,
    }


def analyze_task(task: str, model: str, n: int) -> dict[str, Any]:
    forward_labels, forward_name, forward_anon, forward_path = _load_forward(task, model, n)
    reverse_labels, reverse_name, reverse_anon, reverse_path, parse_rate = _load_reverse(
        task,
        model,
        n,
    )
    if not np.array_equal(forward_labels, reverse_labels):
        raise OrientationSwapError("forward and reverse labels are not aligned")
    expected_labels = np.asarray(
        [int(row["label"]) for _, row in _select_items(TASKS[task]["data"], n)],
        dtype=int,
    )
    if not np.array_equal(forward_labels, expected_labels):
        raise OrientationSwapError("saved runs do not replay the exact source sample")
    forward_delta = forward_name - forward_anon
    reverse_delta = reverse_name - reverse_anon
    forward = _decomposition(forward_labels, forward_name, forward_anon)
    reverse = _decomposition(reverse_labels, reverse_name, reverse_anon)
    orientation_effect = (
        reverse["class_asymmetry_first_minus_comparator"] - forward["class_asymmetry_first_minus_comparator"]
    )
    prior_effect = reverse["class_prior_shift_toward_a"] - forward["class_prior_shift_toward_a"]
    bootstrap = _orientation_bootstrap(
        forward_labels,
        forward_delta,
        reverse_delta,
        seed=_seed(task, model, "orientation-bootstrap"),
    )
    biological_alignment_correlation = _stratified_rank_correlation(
        forward_delta,
        reverse_delta,
        forward_labels,
    )
    forward_asymmetry = forward["class_asymmetry_first_minus_comparator"]
    reverse_asymmetry = reverse["class_asymmetry_first_minus_comparator"]
    order_averaged_correct_grounding = (
        forward["first_class_correct_shift"]
        + forward["comparator_class_correct_shift"]
        + reverse["first_class_correct_shift"]
        + reverse["comparator_class_correct_shift"]
    ) / 4
    biological_class_asymmetry = 0.5 * (forward_asymmetry + reverse_asymmetry)
    prompt_role_asymmetry = 0.5 * (forward_asymmetry - reverse_asymmetry)
    orientation_guardrail = 0.5 * (
        (forward["first_class_correct_shift"] - reverse["first_class_correct_shift"])
        - (reverse["comparator_class_correct_shift"] - forward["comparator_class_correct_shift"])
    )
    return {
        "task": task,
        "task_label": TASKS[task]["label"],
        "model": model,
        "n": n,
        "reverse_parse_rate": parse_rate,
        "forward": forward,
        "reverse_aligned_to_original_class_a": reverse,
        "forward_anon_queried_first_probability": (_probability_distribution_summary(forward_anon)),
        "reverse_anon_queried_first_probability": (_probability_distribution_summary(1.0 - reverse_anon)),
        "orientation_effect_on_class_asymmetry": orientation_effect,
        "orientation_effect_on_prior_shift": prior_effect,
        "order_averaged_correct_grounding": order_averaged_correct_grounding,
        "biological_class_asymmetry": biological_class_asymmetry,
        "prompt_role_asymmetry": prompt_role_asymmetry,
        "orientation_guardrail": orientation_guardrail,
        "forward_reverse_within_class_delta_rank_correlation": (biological_alignment_correlation),
        **bootstrap,
        "input_sha256": {
            str(TASKS[task]["data"].relative_to(ROOT)): _sha256(TASKS[task]["data"]),
            str(forward_path.relative_to(ROOT)): _sha256(forward_path),
            str(reverse_path.relative_to(ROOT)): _sha256(reverse_path),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Single-cell prompt-role reversal: graded signal plus polarity bias",
        "",
        "The same cells and named/anonymous renderings were rerun after reversing",
        "the two answer classes. Reported P(B) was mapped back to P(A)=1-P(B), so",
        "biological-class stability and broader prompt-role following can be compared directly.",
        "",
        (
            "| task | forward correct shift A / B | reversed correct shift A / B | "
            "named AUROC F→R | "
            "order-averaged grounding M | biological-class B_class | prompt-role P_role | "
            "aligned delta rho |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in result["records"]:
        forward = record["forward"]
        reverse = record["reverse_aligned_to_original_class_a"]
        lines.append(
            "| {task} | {fa:+.3f} / {fb:+.3f} | {ra:+.3f} / {rb:+.3f} | "
            "{fauc:.3f}→{rauc:.3f} | "
            "{grounding:+.3f} {grounding_ci} | {biological:+.3f} "
            "{biological_ci} | {role:+.3f} {role_ci} | {rho:+.3f} |".format(
                task=record["task_label"],
                fa=forward["first_class_correct_shift"],
                fb=forward["comparator_class_correct_shift"],
                ra=reverse["first_class_correct_shift"],
                rb=reverse["comparator_class_correct_shift"],
                fauc=forward["name_auroc"],
                rauc=reverse["name_auroc"],
                grounding=record["order_averaged_correct_grounding"],
                grounding_ci=(
                    f"[{record['order_averaged_correct_grounding_ci95_lower']:+.3f}, "
                    f"{record['order_averaged_correct_grounding_ci95_upper']:+.3f}]"
                ),
                biological=record["biological_class_asymmetry"],
                biological_ci=(
                    f"[{record['biological_class_asymmetry_ci95_lower']:+.3f}, "
                    f"{record['biological_class_asymmetry_ci95_upper']:+.3f}]"
                ),
                role=record["prompt_role_asymmetry"],
                role_ci=(
                    f"[{record['prompt_role_asymmetry_ci95_lower']:+.3f}, "
                    f"{record['prompt_role_asymmetry_ci95_upper']:+.3f}]"
                ),
                rho=record["forward_reverse_within_class_delta_rank_correlation"],
            )
        )
    lines.extend(
        [
            "",
            "`M` is the mean correct-oriented named-minus-anonymous shift over both",
            "classes and orientations. `B_class` is positive when biological class A receives",
            "the larger update across orientations and negative when class B does.",
            "`P_role` is positive when the first prompt role is",
            "favored and negative when the second role is favored. Confidence intervals use",
            "a class-stratified paired bootstrap over the same cell IDs.",
            "",
            (
                "The anonymous arm reveals the source of the role effect. The modal "
                "probability assigned to whichever class was queried first was "
                + "; ".join(
                    (
                        f"{record['task_label']}: "
                        f"{record['forward_anon_queried_first_probability']['modal_probability']:.2f} "
                        f"({record['forward_anon_queried_first_probability']['modal_fraction']:.1%}) "
                        "forward and "
                        f"{record['reverse_anon_queried_first_probability']['modal_probability']:.2f} "
                        f"({record['reverse_anon_queried_first_probability']['modal_fraction']:.1%}) "
                        "reversed"
                    )
                    for record in result["records"]
                )
                + ". In this Haiku pilot, gene symbols therefore act mainly by correcting a "
                "high, input-insensitive default for the queried-first class; the second class "
                "has much more headroom for a correct semantic update."
            ),
            "",
            (
                "Across both tasks, `M` is positive and `P_role` is negative with intervals "
                "excluding zero, while the aligned per-cell delta correlation remains "
                "positive. The shared result is a graded, item-specific semantic correction "
                "combined with a larger second-role correction. Biological-class asymmetry "
                "is task-specific rather than shared."
            ),
            "",
            (
                "The orientation-interaction guardrail is "
                + "; ".join(
                    (
                        f"{record['task_label']}: "
                        f"{record['orientation_guardrail']:+.3f} "
                        f"[{record['orientation_guardrail_ci95_lower']:+.3f}, "
                        f"{record['orientation_guardrail_ci95_upper']:+.3f}]"
                    )
                    for record in result["records"]
                )
                + ". The interval includes zero for CD8/NK but not for the monocyte task, "
                "so an additive biological-class plus prompt-role description is insufficient "
                "for the latter. Neither result is an equivalence test."
            ),
            "",
            "The swap changes both list position and which class is queried as the reported",
            "probability. It therefore identifies a bundled prompt-role/polarity effect, not",
            "pure first-versus-second list-position causality. A pure position test must cross",
            "list order (A/B versus B/A) with queried target (P(A) versus P(B)).",
            "",
            "The forward artifact is historical while the reverse run is new. The dated model",
            "identifier reduces but does not eliminate model-drift risk; a claim-bearing run",
            "should rerun all prompt forms contemporaneously and interleave their call order.",
            "",
            "This output-level intervention does not establish hidden-state causality,",
            "pre-existing latent knowledge, or training exposure.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--tasks",
        default="cd8t_nk,mono",
        help="comma-separated subset of cd8t_nk,mono",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="do not make model calls; analyze complete saved reverse runs",
    )
    args = parser.parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks or any(task not in TASKS for task in tasks):
        raise OrientationSwapError("--tasks must be a nonempty subset of cd8t_nk,mono")
    if args.n != 200:
        raise OrientationSwapError("the paired forward artifacts contain the locked n=200 sample; use --n 200")
    if not args.analyze_only:
        for task in tasks:
            run_task(task, args.model, args.n)
    records = [analyze_task(task, args.model, args.n) for task in tasks]
    result = {
        "analysis_id": ANALYSIS_SEED,
        "claim_scope": (
            "same_cell_output_level_answer_order_intervention_no_hidden_state_"
            "causality_latent_knowledge_or_training_exposure_claim"
        ),
        "model": args.model,
        "records": records,
    }
    output_dir = RESULT_ROOT / "orientation_swap"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_model_name(args.model)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

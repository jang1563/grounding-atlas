"""Validate and analyze the frozen v2 causal decision-state transfer study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import run_coherent_readout_v2_causal_binding as runner
except ImportError:  # direct execution from eval/
    import run_coherent_readout_v2_causal_binding as runner


DISCOVERY_ANALYSIS_SCHEMA = "coherent-readout-v2-causal-binding-discovery-analysis-v1"
LAYER_LOCK_SCHEMA = "coherent-readout-v2-causal-binding-layer-lock-v1"
HOLDOUT_BASELINE_ANALYSIS_SCHEMA = (
    "coherent-readout-v2-causal-binding-holdout-baseline-analysis-v1"
)
HOLDOUT_ENTRY_SCHEMA = "coherent-readout-v2-causal-binding-holdout-entry-v1"
FINAL_ANALYSIS_SCHEMA = "coherent-readout-v2-causal-binding-final-analysis-v1"


class CausalBindingAnalysisError(ValueError):
    """Raised when a raw causal-binding artifact is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CausalBindingAnalysisError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise CausalBindingAnalysisError("JSON artifact must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalBindingAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CausalBindingAnalysisError(f"{label} must be finite")
    return result


def _validate_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "x_logit",
        "y_logit",
        "x_minus_y_margin",
        "full_vocab_logsumexp",
        "label_probability_mass",
        "greedy_token_id",
        "greedy_logit",
        "maximum_token_ids",
        "maximum_tie_count",
        "full_vocab_logits_sha256",
    }
    if set(value) != expected:
        raise CausalBindingAnalysisError("diagnostic schema changed")
    x = _finite(value["x_logit"], "x_logit")
    y = _finite(value["y_logit"], "y_logit")
    margin = _finite(value["x_minus_y_margin"], "margin")
    logsumexp = _finite(value["full_vocab_logsumexp"], "full_vocab_logsumexp")
    greedy_logit = _finite(value["greedy_logit"], "greedy_logit")
    if abs(margin - (x - y)) > 1e-7:
        raise CausalBindingAnalysisError("stored margin does not equal X minus Y")
    mass = _finite(value["label_probability_mass"], "label_probability_mass")
    if not 0.0 <= mass <= 1.0:
        raise CausalBindingAnalysisError("label probability mass is outside [0,1]")
    expected_mass = math.exp(float(np.logaddexp(x, y)) - logsumexp)
    if abs(mass - expected_mass) > 1e-12:
        raise CausalBindingAnalysisError("stored label probability mass is inconsistent")
    if logsumexp < greedy_logit or greedy_logit < max(x, y):
        raise CausalBindingAnalysisError("greedy logit or log-sum-exp is inconsistent")
    maximum_ids = value["maximum_token_ids"]
    tie_count = value["maximum_tie_count"]
    greedy_token_id = value["greedy_token_id"]
    if (
        not isinstance(maximum_ids, list)
        or not maximum_ids
        or any(isinstance(x, bool) or not isinstance(x, int) for x in maximum_ids)
        or any(x < 0 or x >= runner.MODEL_VOCAB_SIZE for x in maximum_ids)
        or maximum_ids != sorted(set(maximum_ids))
        or isinstance(tie_count, bool)
        or not isinstance(tie_count, int)
        or tie_count != len(maximum_ids)
        or isinstance(greedy_token_id, bool)
        or not isinstance(greedy_token_id, int)
        or greedy_token_id != maximum_ids[0]
    ):
        raise CausalBindingAnalysisError("global-maximum diagnostics are inconsistent")
    if runner.X_TOKEN_ID in maximum_ids and abs(greedy_logit - x) > 1e-7:
        raise CausalBindingAnalysisError("greedy logit disagrees with maximal X logit")
    if runner.Y_TOKEN_ID in maximum_ids and abs(greedy_logit - y) > 1e-7:
        raise CausalBindingAnalysisError("greedy logit disagrees with maximal Y logit")
    digest = value["full_vocab_logits_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CausalBindingAnalysisError("full-vocabulary digest is invalid")
    return dict(value)


def _load_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return runner._load_frozen_plan()
    except runner.CausalBindingRunnerError as error:
        raise CausalBindingAnalysisError(str(error)) from error


def _validate_execution_manifest(
    path: Path,
    *,
    phase: str,
    plan: Mapping[str, Any],
    baseline_path: Path,
    baseline_count: int,
    patch_path: Path | None = None,
    patch_count: int = 0,
    activation_path: Path | None = None,
    activation_shape: tuple[int, ...] | None = None,
    selected_layer: int | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    manifest = _load_json(path)
    expected_manifest_keys = {
        "schema_version",
        "status",
        "phase",
        "mode",
        "claim_scope",
        "call_plan_sha256",
        "design_file_sha256",
        "plan_manifest_file_sha256",
        "selected_layer",
        "attempt_receipt",
        "model_calls",
        "generation_used",
        "biological_model_calls",
        "records",
        "activations",
        "patch_records",
        "locks",
        "environment",
    }
    if set(manifest) != expected_manifest_keys:
        raise CausalBindingAnalysisError("execution manifest fields changed")
    if manifest.get("schema_version") != runner.EXECUTION_MANIFEST_SCHEMA:
        raise CausalBindingAnalysisError("execution manifest schema changed")
    if manifest.get("phase") != phase:
        raise CausalBindingAnalysisError("execution phase changed")
    if manifest.get("status") != "EXECUTION_COMPLETE_NOT_ANALYZED":
        raise CausalBindingAnalysisError("execution status changed")
    if manifest.get("mode") != "development":
        raise CausalBindingAnalysisError("execution mode changed")
    if manifest.get("claim_scope") != runner.design_from_plan(plan)["claim_scope"]:
        raise CausalBindingAnalysisError("execution claim scope changed")
    if manifest.get("call_plan_sha256") != plan["call_plan_sha256"]:
        raise CausalBindingAnalysisError("execution manifest changed call plan")
    if manifest.get("locks") != plan["locks"]:
        raise CausalBindingAnalysisError("execution input locks changed")
    if manifest.get("environment") != plan["environment"]:
        raise CausalBindingAnalysisError("execution environment lock changed")
    if manifest.get("design_file_sha256") != runner.file_sha256(
        runner.DEFAULT_DESIGN
    ):
        raise CausalBindingAnalysisError("execution design binding changed")
    if manifest.get("plan_manifest_file_sha256") != runner.file_sha256(
        runner.DEFAULT_PLAN_MANIFEST
    ):
        raise CausalBindingAnalysisError("execution plan-manifest binding changed")
    if manifest.get("biological_model_calls") != 0:
        raise CausalBindingAnalysisError("execution claims biological model calls")
    if manifest.get("generation_used") is not False:
        raise CausalBindingAnalysisError("execution claims text generation")
    expected_model_calls = {
        "discovery": 1088,
        "holdout_baseline": 192,
        "holdout_patch": 768,
    }
    if manifest.get("model_calls") != expected_model_calls.get(phase):
        raise CausalBindingAnalysisError("execution model-call count changed")
    if manifest.get("selected_layer") != selected_layer:
        raise CausalBindingAnalysisError("execution selected layer changed")

    expected_attempt_path = runner.RESULT_ROOT / f"{phase}_attempt.json"
    attempt_binding = manifest.get("attempt_receipt")
    if (
        not isinstance(attempt_binding, Mapping)
        or attempt_binding.get("path") != str(expected_attempt_path)
        or attempt_binding.get("file_sha256")
        != runner.file_sha256(expected_attempt_path)
    ):
        raise CausalBindingAnalysisError("execution attempt receipt is not bound")
    attempt = _load_json(expected_attempt_path)
    expected_attempt = {
        "schema_version": "coherent-readout-v2-causal-binding-attempt-v1",
        "status": "EXECUTION_ARMED_NO_FORWARD_RECORDED",
        "phase": phase,
        "call_plan_sha256": plan["call_plan_sha256"],
        "runner_sha256": plan["locks"]["runner_sha256"],
        "environment_sha256": plan["environment"]["environment_sha256"],
        "model_calls_recorded_before_attempt": 0,
        "biological_model_calls": 0,
    }
    if attempt != expected_attempt:
        raise CausalBindingAnalysisError("execution attempt receipt changed")
    records = manifest.get("records")
    if (
        not isinstance(records, Mapping)
        or records.get("path") != str(baseline_path)
        or records.get("count") != baseline_count
        or records.get("file_sha256") != runner.file_sha256(baseline_path)
    ):
        raise CausalBindingAnalysisError("baseline records are not manifest-bound")
    if patch_path is None:
        if manifest.get("patch_records") is not None:
            raise CausalBindingAnalysisError("unexpected patch records in manifest")
    else:
        patches = manifest.get("patch_records")
        if (
            not isinstance(patches, Mapping)
            or patches.get("path") != str(patch_path)
            or patches.get("count") != patch_count
            or patches.get("file_sha256") != runner.file_sha256(patch_path)
        ):
            raise CausalBindingAnalysisError("patch records are not manifest-bound")
    activations = None
    if activation_path is not None:
        if activation_shape is None:
            raise CausalBindingAnalysisError("activation shape was not supplied")
        try:
            activations = runner._load_activation_sidecar(
                activation_path, activation_shape
            )
        except runner.CausalBindingRunnerError as error:
            raise CausalBindingAnalysisError(str(error)) from error
        activation_manifest = manifest.get("activations")
        if (
            not isinstance(activation_manifest, Mapping)
            or activation_manifest.get("path") != str(activation_path)
            or activation_manifest.get("file_sha256")
            != runner.file_sha256(activation_path)
            or activation_manifest.get("logical_sha256")
            != runner.f32_sha256(activations)
            or activation_manifest.get("shape") != list(activation_shape)
            or activation_manifest.get("dtype") != "<f4"
        ):
            raise CausalBindingAnalysisError("activation sidecar is not manifest-bound")
    return manifest, activations


def _validate_baselines(
    records: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    activations: np.ndarray,
    *,
    phase: str,
    require_duplicate: bool,
) -> list[dict[str, Any]]:
    if len(records) != len(prompts) or activations.shape[0] != len(records):
        raise CausalBindingAnalysisError("baseline prompt/record counts disagree")
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    if len(prompt_by_id) != len(prompts):
        raise CausalBindingAnalysisError("planned prompt IDs are not unique")
    observed = []
    for index, raw in enumerate(records):
        row = dict(raw)
        prompt = prompt_by_id.get(row.get("prompt_id"))
        if prompt is None:
            raise CausalBindingAnalysisError("baseline prompt is not in the plan")
        if row.get("activation_row") != index:
            raise CausalBindingAnalysisError("baseline activation row order changed")
        if row.get("phase") != phase:
            raise CausalBindingAnalysisError("baseline phase changed")
        diagnostics = _validate_diagnostics(row.get("diagnostics", {}))
        duplicate = row.get("duplicate_diagnostics")
        if require_duplicate:
            if not isinstance(duplicate, Mapping):
                raise CausalBindingAnalysisError("discovery duplicate is missing")
            duplicate = _validate_diagnostics(duplicate)
        elif duplicate is not None:
            raise CausalBindingAnalysisError("unexpected holdout duplicate diagnostics")
        layer_hashes = row.get("activation_layer_sha256")
        if not isinstance(layer_hashes, list) or len(layer_hashes) != runner.MODEL_LAYERS:
            raise CausalBindingAnalysisError("baseline layer-hash registry changed")
        for layer in range(runner.MODEL_LAYERS):
            if runner.f32_sha256(activations[index, layer]) != layer_hashes[layer]:
                raise CausalBindingAnalysisError("baseline activation hash mismatch")
        expected = runner._baseline_record(
            prompt,
            phase=phase,
            activation_row=index,
            measurement={
                "diagnostics": diagnostics,
                "duplicate_diagnostics": duplicate,
                "activation_layer_sha256": layer_hashes,
            },
        )
        if row != expected:
            raise CausalBindingAnalysisError("baseline identity does not reconstruct")
        observed.append(row)
    if {row["prompt_id"] for row in observed} != set(prompt_by_id):
        raise CausalBindingAnalysisError("baseline prompt set changed")
    return observed


def _validate_patches(
    records: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    selected_layer: int | None,
) -> list[dict[str, Any]]:
    if len(records) != len(templates):
        raise CausalBindingAnalysisError("patch record/template counts disagree")
    template_by_id = {template["template_id"]: template for template in templates}
    baseline_by_prompt = {row["prompt_id"]: row for row in baselines}
    observed = []
    for raw in records:
        row = dict(raw)
        template = template_by_id.get(row.get("template_id"))
        if template is None:
            raise CausalBindingAnalysisError("patch template is not in the plan")
        expected_layer = (
            template["layer"] if template["layer"] is not None else selected_layer
        )
        if row.get("layer") != expected_layer or row.get("phase") != phase:
            raise CausalBindingAnalysisError("patch phase or layer changed")
        source = baseline_by_prompt.get(template["source_prompt_id"])
        target = baseline_by_prompt.get(template["target_prompt_id"])
        if source is None or target is None:
            raise CausalBindingAnalysisError("patch source or target baseline is missing")
        diagnostics = _validate_diagnostics(row.get("diagnostics", {}))
        trace = row.get("hook_trace")
        if trace != {
            "hook_calls": 1,
            "non_target_tokens_unchanged": True,
            "patched_token_matches_source": True,
        }:
            raise CausalBindingAnalysisError("patch hook trace failed")
        expected = runner._patch_record(
            template,
            phase=phase,
            layer=int(expected_layer),
            source_baseline=source,
            target_baseline=target,
            diagnostics=diagnostics,
            trace=trace,
        )
        if row != expected:
            raise CausalBindingAnalysisError("patch identity does not reconstruct")
        observed.append(row)
    if len({row["patch_id"] for row in observed}) != len(observed):
        raise CausalBindingAnalysisError("patch IDs are not unique")
    return observed


def _pair_means(
    item_values: Mapping[str, float], item_to_pair: Mapping[str, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item_id, value in item_values.items():
        grouped.setdefault(item_to_pair[item_id], []).append(float(value))
    if not grouped or any(len(values) != 2 for values in grouped.values()):
        raise CausalBindingAnalysisError("each pair must contribute exactly two items")
    return {
        pair_id: float(np.mean(values)) for pair_id, values in sorted(grouped.items())
    }


def _dependency_cluster_means(
    pair_values: Mapping[str, float], pair_to_cluster: Mapping[str, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for pair_id, value in pair_values.items():
        grouped.setdefault(pair_to_cluster[pair_id], []).append(float(value))
    if not grouped or any(len(values) != 2 for values in grouped.values()):
        raise CausalBindingAnalysisError(
            "each unrelated-control dependency cluster must contain two pairs"
        )
    return {
        cluster_id: float(np.mean(values))
        for cluster_id, values in sorted(grouped.items())
    }


def _effect_summary(
    pair_values: Mapping[str, float],
    *,
    gap_pair_values: Mapping[str, float],
    inference_unit: str = "lexical_pair",
) -> dict[str, Any]:
    if list(pair_values) != list(gap_pair_values):
        raise CausalBindingAnalysisError("effect and gap pair order differs")
    values = np.asarray(list(pair_values.values()), dtype=float)
    gap_values = np.asarray(list(gap_pair_values.values()), dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(gap_values).all():
        raise CausalBindingAnalysisError("effect or gap values are non-finite")
    if not len(values):
        return {
            "pair_values": {},
            "n_pairs": 0,
            "n_inference_clusters": 0,
            "inference_unit": inference_unit,
            "summary_defined": False,
            "mean": None,
            "median": None,
            "positive_pair_count": 0,
            "leave_one_pair_out_means": [],
            "gap_mean": None,
            "gap_denominator_positive": False,
            "lodo_gap_means": [],
            "all_lodo_gap_denominators_positive": False,
            "recovery_fraction_defined": False,
            "mean_over_gap": None,
            "lodo_over_gap": [],
        }
    lodo = (
        [float(np.mean(np.delete(values, index))) for index in range(len(values))]
        if len(values) > 1
        else []
    )
    lodo_gap = (
        [
            float(np.mean(np.delete(gap_values, index)))
            for index in range(len(values))
        ]
        if len(values) > 1
        else []
    )
    gap_mean = float(np.mean(gap_values))
    gap_positive = gap_mean > 0.0
    lodo_gaps_positive = bool(lodo_gap) and all(value > 0.0 for value in lodo_gap)
    lodo_over_gap = [
        (float(value / gap) if gap > 0.0 else None)
        for value, gap in zip(lodo, lodo_gap, strict=True)
    ]
    return {
        "pair_values": dict(pair_values),
        "n_pairs": len(values),
        "n_inference_clusters": len(values),
        "inference_unit": inference_unit,
        "summary_defined": True,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_pair_count": int(np.sum(values > 0.0)),
        "leave_one_pair_out_means": lodo,
        "gap_mean": gap_mean,
        "gap_denominator_positive": gap_positive,
        "lodo_gap_means": lodo_gap,
        "all_lodo_gap_denominators_positive": lodo_gaps_positive,
        "recovery_fraction_defined": bool(gap_positive and lodo_gaps_positive),
        "mean_over_gap": (
            float(np.mean(values) / gap_mean) if gap_positive else None
        ),
        "lodo_over_gap": lodo_over_gap,
    }


def _bootstrap_summary(
    pair_values: Mapping[str, float], *, draws: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(list(pair_values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "draws": draws,
        "seed": seed,
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _metrics_for_layer(
    baselines: Sequence[Mapping[str, Any]],
    patches: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    bootstrap: bool,
) -> dict[str, Any]:
    baseline = {
        (row["item_id"], row["prompt_role"]): row for row in baselines
    }
    patch = {
        (row["item_id"], row["condition"]): row
        for row in patches
        if row["layer"] == layer
    }
    item_ids = sorted({row["item_id"] for row in baselines})
    item_to_pair = {
        row["item_id"]: row["pair_id"] for row in baselines
    }
    required = set(runner.PATCH_CONDITIONS)
    if any(
        {condition for candidate, condition in patch if candidate == item_id}
        != required
        for item_id in item_ids
    ):
        raise CausalBindingAnalysisError("an item is missing a patch condition")

    def margin(row: Mapping[str, Any]) -> float:
        return float(row["diagnostics"]["x_minus_y_margin"])

    item_values: dict[str, dict[str, float]] = {}
    for item_id in item_ids:
        d = margin(baseline[(item_id, "D")])
        r = margin(baseline[(item_id, "R")])
        forward_paired = margin(patch[(item_id, "forward_paired")]) - r
        forward_same = margin(patch[(item_id, "forward_same_pair_x")]) - r
        forward_unrelated = margin(patch[(item_id, "forward_unrelated_x")]) - r
        reverse_paired = d - margin(patch[(item_id, "reverse_paired")])
        reverse_same = d - margin(patch[(item_id, "reverse_same_pair")])
        reverse_unrelated = d - margin(patch[(item_id, "reverse_unrelated")])
        item_values[item_id] = {
            "G": d - r,
            "F": forward_paired,
            "N": reverse_paired,
            "F_same": forward_same,
            "F_unrelated": forward_unrelated,
            "N_same": reverse_same,
            "N_unrelated": reverse_unrelated,
            "S_F_same": forward_paired - forward_same,
            "S_F_unrelated": forward_paired - forward_unrelated,
            "S_N_same": reverse_paired - reverse_same,
            "S_N_unrelated": reverse_paired - reverse_unrelated,
            "identity_R": margin(patch[(item_id, "identity_r")]) - r,
            "identity_D": margin(patch[(item_id, "identity_d")]) - d,
        }
    raw_pair_values = {
        metric: _pair_means(
            {item_id: values[metric] for item_id, values in item_values.items()},
            item_to_pair,
        )
        for metric in next(iter(item_values.values()))
    }
    pair_to_unrelated_cluster: dict[str, str] = {}
    for item_id in item_ids:
        cluster_id = patch[(item_id, "forward_unrelated_x")][
            "dependency_cluster_id"
        ]
        pair_id = item_to_pair[item_id]
        if (
            pair_id in pair_to_unrelated_cluster
            and pair_to_unrelated_cluster[pair_id] != cluster_id
        ):
            raise CausalBindingAnalysisError(
                "one pair maps to multiple unrelated-control clusters"
            )
        pair_to_unrelated_cluster[pair_id] = cluster_id
        if (
            patch[(item_id, "reverse_unrelated")]["dependency_cluster_id"]
            != cluster_id
        ):
            raise CausalBindingAnalysisError(
                "forward and reverse unrelated clusters disagree"
            )
    gap_mean = float(np.mean(list(raw_pair_values["G"].values())))
    inference_values: dict[str, dict[str, float]] = {}
    inference_gaps: dict[str, dict[str, float]] = {}
    inference_units: dict[str, str] = {}
    for metric, values in raw_pair_values.items():
        if "unrelated" in metric:
            inference_values[metric] = _dependency_cluster_means(
                values, pair_to_unrelated_cluster
            )
            inference_gaps[metric] = _dependency_cluster_means(
                raw_pair_values["G"], pair_to_unrelated_cluster
            )
            inference_units[metric] = "reciprocal_unrelated_control_dyad"
        else:
            inference_values[metric] = values
            inference_gaps[metric] = raw_pair_values["G"]
            inference_units[metric] = "lexical_pair"
    summaries = {
        metric: _effect_summary(
            values,
            gap_pair_values=inference_gaps[metric],
            inference_unit=inference_units[metric],
        )
        for metric, values in inference_values.items()
    }

    pair_to_items: dict[str, list[str]] = {}
    for item_id in item_ids:
        pair_to_items.setdefault(item_to_pair[item_id], []).append(item_id)
    same_exact_pairs = sorted(
        pair_id
        for pair_id, members in pair_to_items.items()
        if all(
            patch[(item_id, "forward_same_pair_x")][
                "source_target_token_count_difference"
            ]
            == 0
            and patch[(item_id, "reverse_same_pair")][
                "source_target_token_count_difference"
            ]
            == 0
            for item_id in members
        )
    )
    dyad_to_pairs: dict[str, list[str]] = {}
    for pair_id, cluster_id in pair_to_unrelated_cluster.items():
        dyad_to_pairs.setdefault(cluster_id, []).append(pair_id)
    unrelated_exact_dyads = sorted(
        cluster_id
        for cluster_id, pair_ids in dyad_to_pairs.items()
        if len(pair_ids) == 2
        and all(
            patch[(item_id, "forward_unrelated_x")][
                "source_target_token_count_difference"
            ]
            == 0
            and patch[(item_id, "reverse_unrelated")][
                "source_target_token_count_difference"
            ]
            == 0
            for pair_id in pair_ids
            for item_id in pair_to_items[pair_id]
        )
    )
    for metric in ("S_F_same", "S_N_same"):
        exact_values = {
            pair_id: raw_pair_values[metric][pair_id]
            for pair_id in same_exact_pairs
        }
        exact_gaps = {
            pair_id: raw_pair_values["G"][pair_id]
            for pair_id in same_exact_pairs
        }
        summaries[f"{metric}_exact_length"] = _effect_summary(
            exact_values,
            gap_pair_values=exact_gaps,
            inference_unit="exact_length_lexical_pair",
        )
    for metric in ("S_F_unrelated", "S_N_unrelated"):
        dyad_values = inference_values[metric]
        dyad_gaps = inference_gaps[metric]
        exact_values = {
            cluster_id: dyad_values[cluster_id]
            for cluster_id in unrelated_exact_dyads
        }
        exact_gaps = {
            cluster_id: dyad_gaps[cluster_id]
            for cluster_id in unrelated_exact_dyads
        }
        summaries[f"{metric}_exact_length"] = _effect_summary(
            exact_values,
            gap_pair_values=exact_gaps,
            inference_unit="exact_length_reciprocal_control_dyad",
        )
    if bootstrap:
        for metric in summaries:
            summaries[metric]["cluster_bootstrap_95"] = _bootstrap_summary(
                summaries[metric]["pair_values"],
                draws=10000,
                seed=260802,
            )
    return {
        "layer": layer,
        "n_items": len(item_ids),
        "n_pairs": len(raw_pair_values["G"]),
        "gap_mean": gap_mean,
        "item_values": item_values,
        "length_matched_specificity_units": {
            "same_pair_lexical_pairs": same_exact_pairs,
            "unrelated_control_dyads": unrelated_exact_dyads,
        },
        "metrics": summaries,
    }


def _discovery_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = metrics["metrics"]

    def transfer_component(name: str) -> bool:
        summary = values[name]
        ratios = summary["lodo_over_gap"]
        return bool(
            summary["recovery_fraction_defined"]
            and summary["mean_over_gap"] is not None
            and summary["mean_over_gap"] >= 0.30
            and summary["positive_pair_count"] >= 7
            and ratios
            and all(value is not None and value >= 0.20 for value in ratios)
        )

    transfer = all(
        transfer_component(name)
        for name in ("F", "N")
    )
    specificity_metrics = (
        "S_F_same_exact_length",
        "S_F_unrelated_exact_length",
        "S_N_same_exact_length",
        "S_N_unrelated_exact_length",
    )
    specificity = transfer and all(
        bool(
            values[name]["summary_defined"]
            and values[name]["gap_denominator_positive"]
            and values[name]["mean_over_gap"] is not None
            and values[name]["mean_over_gap"] >= 0.20
            and values[name]["positive_pair_count"]
            >= math.ceil(0.75 * values[name]["n_inference_clusters"])
            and values[name]["leave_one_pair_out_means"]
            and min(values[name]["leave_one_pair_out_means"]) > 0.0
        )
        for name in specificity_metrics
    )
    return {
        "transfer_pass": transfer,
        "specificity_diagnostic_pass": specificity,
        "specificity_used_for_layer_selection": False,
    }


def _engineering_gates(
    baselines: Sequence[Mapping[str, Any]],
    patches: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tolerance = 1e-4
    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    prior_xy_differences = []
    prior_full_vocab_pass = True
    for row in baselines:
        prompt = prompt_by_id[row["prompt_id"]]
        prior_full_vocab_pass = bool(
            prior_full_vocab_pass
            and prompt["prior_syntax_record_id"] is not None
            and row["diagnostics"]["full_vocab_logits_sha256"]
            == prompt["prior_full_vocab_logits_sha256"]
        )
        prior_xy_differences.extend(
            [
                abs(row["diagnostics"]["x_logit"] - prompt["prior_x_logit"]),
                abs(row["diagnostics"]["y_logit"] - prompt["prior_y_logit"]),
            ]
        )
    duplicate_differences = []
    for row in baselines:
        duplicate = row["duplicate_diagnostics"]
        duplicate_differences.extend(
            [
                abs(row["diagnostics"]["x_logit"] - duplicate["x_logit"]),
                abs(row["diagnostics"]["y_logit"] - duplicate["y_logit"]),
            ]
        )
    baseline = {
        (row["item_id"], row["prompt_role"]): row for row in baselines
    }
    identity_differences = []
    final_copy_differences = []
    for row in patches:
        if row["condition"] in {"identity_r", "identity_d"}:
            role = "R" if row["condition"] == "identity_r" else "D"
            target = baseline[(row["item_id"], role)]["diagnostics"]
            identity_differences.extend(
                [
                    abs(row["diagnostics"]["x_logit"] - target["x_logit"]),
                    abs(row["diagnostics"]["y_logit"] - target["y_logit"]),
                ]
            )
        if row["layer"] == 27 and row["condition"] in {
            "forward_paired",
            "reverse_paired",
        }:
            role = "D" if row["condition"] == "forward_paired" else "R"
            source = baseline[(row["item_id"], role)]["diagnostics"]
            final_copy_differences.extend(
                [
                    abs(row["diagnostics"]["x_logit"] - source["x_logit"]),
                    abs(row["diagnostics"]["y_logit"] - source["y_logit"]),
                ]
            )
    maximum_duplicate = max(duplicate_differences, default=math.inf)
    maximum_prior_xy = max(prior_xy_differences, default=math.inf)
    maximum_identity = max(identity_differences, default=math.inf)
    maximum_final_copy = max(final_copy_differences, default=math.inf)
    return {
        "tolerance": tolerance,
        "maximum_prior_xy_difference": maximum_prior_xy,
        "prior_xy_pass": maximum_prior_xy <= tolerance,
        "prior_full_vocab_row_sha256_pass": prior_full_vocab_pass,
        "maximum_duplicate_xy_difference": maximum_duplicate,
        "maximum_identity_xy_difference": maximum_identity,
        "maximum_final_block_source_xy_difference": maximum_final_copy,
        "duplicate_pass": maximum_duplicate <= tolerance,
        "identity_pass": maximum_identity <= tolerance,
        "final_block_source_copy_pass": maximum_final_copy <= tolerance,
        "hook_trace_pass": all(
            row["hook_trace"]
            == {
                "hook_calls": 1,
                "non_target_tokens_unchanged": True,
                "patched_token_matches_source": True,
            }
            for row in patches
        ),
    }


def analyze_discovery() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    root = runner.RESULT_ROOT
    baseline_path = root / "discovery_baselines.jsonl"
    activation_path = root / "discovery_activations.npy"
    patch_path = root / "discovery_patches.jsonl"
    _, activations = _validate_execution_manifest(
        root / "discovery_execution_manifest.json",
        phase="discovery",
        plan=plan,
        baseline_path=baseline_path,
        baseline_count=32,
        patch_path=patch_path,
        patch_count=1024,
        activation_path=activation_path,
        activation_shape=(32, runner.MODEL_LAYERS, runner.MODEL_WIDTH),
    )
    assert activations is not None
    prompts = runner._phase_prompts(plan, "discovery")
    templates = runner._phase_templates(plan, "discovery")
    baselines = _validate_baselines(
        runner.load_jsonl(baseline_path),
        prompts,
        activations,
        phase="discovery",
        require_duplicate=True,
    )
    patches = _validate_patches(
        runner.load_jsonl(patch_path),
        templates,
        baselines,
        phase="discovery",
        selected_layer=None,
    )
    engineering = _engineering_gates(baselines, patches, prompts)
    engineering_pass = all(
        engineering[key]
        for key in (
            "duplicate_pass",
            "prior_xy_pass",
            "prior_full_vocab_row_sha256_pass",
            "identity_pass",
            "final_block_source_copy_pass",
            "hook_trace_pass",
        )
    )
    layer_metrics = [
        _metrics_for_layer(baselines, patches, layer=layer, bootstrap=False)
        for layer in runner.LAYER_GRID
    ]
    decisions = {
        metric["layer"]: _discovery_decision(metric) for metric in layer_metrics
    }
    transfer_layers = [
        layer
        for layer, decision in decisions.items()
        if layer != 27 and decision["transfer_pass"]
    ]
    selected_layer = None
    selection_class = None
    if engineering_pass and transfer_layers:
        selected_layer = min(transfer_layers)
        selection_class = "transfer_only_earliest_nonfinal_grid_layer"
    if not engineering_pass:
        status = "ENGINEERING_STOP_INVALID_PATCH_EXECUTION"
    elif selected_layer is None:
        status = (
            "LOCALIZATION_STOP_ONLY_FINAL_COPY_CONTROL"
            if decisions[27]["transfer_pass"]
            else "LOCALIZATION_STOP_NO_TRANSFER_LAYER"
        )
    else:
        status = "LOCALIZATION_COMPLETE_HOLDOUT_BASELINE_AUTHORIZED"
    analysis = {
        "schema_version": DISCOVERY_ANALYSIS_SCHEMA,
        "status": status,
        "mode": "development",
        "claim_authority": "layer_localization_only",
        "selected_layer": selected_layer,
        "selection_class": selection_class,
        "engineering_gates": engineering,
        "layer_decisions": {str(key): value for key, value in decisions.items()},
        "layer_metrics": layer_metrics,
        "call_plan_sha256": plan["call_plan_sha256"],
        "design_sha256": runner.file_sha256(runner.DEFAULT_DESIGN),
        "analyzer_sha256": plan["locks"]["analyzer_sha256"],
        "raw_artifacts": {
            "baselines_sha256": runner.file_sha256(baseline_path),
            "activations_sha256": runner.file_sha256(activation_path),
            "patches_sha256": runner.file_sha256(patch_path),
        },
        "biological_model_calls": 0,
    }
    analysis_path = root / "discovery_analysis.json"
    runner.write_json(analysis_path, analysis)
    layer_lock = {
        "schema_version": LAYER_LOCK_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "selection_class": selection_class,
        "holdout_baseline_authorized": selected_layer is not None and engineering_pass,
        "selection_rule": design["selection_rule"],
        "layer_grid": list(runner.LAYER_GRID),
        "discovery_analysis_path": str(analysis_path),
        "discovery_analysis_file_sha256": runner.file_sha256(analysis_path),
        "call_plan_sha256": plan["call_plan_sha256"],
        "analyzer_sha256": plan["locks"]["analyzer_sha256"],
        "holdout_patch_authorized": False,
        "biological_execution_allowed": False,
    }
    runner.write_json(runner.DEFAULT_LAYER_LOCK, layer_lock)
    return analysis, layer_lock


def _holdout_baseline_metrics(
    baselines: Sequence[Mapping[str, Any]], selected_layer: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    del selected_layer
    by_item = {
        (row["item_id"], row["prompt_role"]): row for row in baselines
    }
    item_ids = sorted({row["item_id"] for row in baselines})
    item_to_pair = {row["item_id"]: row["pair_id"] for row in baselines}
    gap_items = {
        item_id: by_item[(item_id, "D")]["diagnostics"]["x_minus_y_margin"]
        - by_item[(item_id, "R")]["diagnostics"]["x_minus_y_margin"]
        for item_id in item_ids
    }
    gap_pairs = _pair_means(gap_items, item_to_pair)
    gap_mean = float(np.mean(list(gap_pairs.values())))
    gap_bootstrap = _bootstrap_summary(gap_pairs, draws=10000, seed=260802)
    d_rows = [by_item[(item_id, "D")] for item_id in item_ids]
    r_rows = [by_item[(item_id, "R")] for item_id in item_ids]
    d_correct = sum(row["diagnostics"]["greedy_token_id"] == runner.X_TOKEN_ID for row in d_rows)
    r_incorrect = sum(row["diagnostics"]["greedy_token_id"] == runner.Y_TOKEN_ID for row in r_rows)
    no_ties = all(row["diagnostics"]["maximum_tie_count"] == 1 for row in baselines)
    d_mass = float(np.mean([row["diagnostics"]["label_probability_mass"] for row in d_rows]))
    r_mass = float(np.mean([row["diagnostics"]["label_probability_mass"] for row in r_rows]))
    cohort = [
        item_id
        for item_id in item_ids
        if by_item[(item_id, "D")]["diagnostics"]["greedy_token_id"]
        == runner.X_TOKEN_ID
        and by_item[(item_id, "R")]["diagnostics"]["greedy_token_id"]
        == runner.Y_TOKEN_ID
    ]
    metrics = {
        "items": len(item_ids),
        "pairs": len(gap_pairs),
        "d_correct_count": d_correct,
        "d_correct_rate": d_correct / len(d_rows),
        "r_incorrect_count": r_incorrect,
        "r_incorrect_rate": r_incorrect / len(r_rows),
        "no_global_argmax_ties": no_ties,
        "d_mean_label_probability_mass": d_mass,
        "r_mean_label_probability_mass": r_mass,
        "gap_pair_values": gap_pairs,
        "gap_mean": gap_mean,
        "gap_cluster_bootstrap_95": gap_bootstrap,
        "secondary_failure_cohort_count": len(cohort),
    }
    gate = {
        "d_correct_rate_pass": metrics["d_correct_rate"] >= 0.95,
        "r_incorrect_rate_pass": metrics["r_incorrect_rate"] >= 0.75,
        "no_ties_pass": no_ties,
        "d_label_mass_pass": d_mass >= 0.95,
        "r_label_mass_pass": r_mass >= 0.95,
        "gap_point_pass": gap_mean >= 1.0,
        "gap_bootstrap_pass": gap_bootstrap["lower_95"] > 0.0,
    }
    gate["all_pass"] = all(gate.values())
    return {"metrics": metrics, "gate": gate}, {"item_ids": cohort}


def _control_length_summary(
    templates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for condition in (
        "forward_same_pair_x",
        "forward_unrelated_x",
        "reverse_same_pair",
        "reverse_unrelated",
    ):
        differences = [
            int(template["source_target_token_count_difference"])
            for template in templates
            if template["condition"] == condition
        ]
        if not differences:
            raise CausalBindingAnalysisError("control length registry is empty")
        result[condition] = {
            "records": len(differences),
            "exact_length_matches": sum(value == 0 for value in differences),
            "difference_counts": {
                str(value): differences.count(value)
                for value in sorted(set(differences))
            },
            "maximum_absolute_difference": max(abs(value) for value in differences),
        }
    return result


def analyze_holdout_baseline() -> tuple[dict[str, Any], dict[str, Any]]:
    plan, design = _load_plan()
    _, recomputed_lock = analyze_discovery()
    layer_lock = _load_json(runner.DEFAULT_LAYER_LOCK)
    if layer_lock != recomputed_lock:
        raise CausalBindingAnalysisError("layer lock does not recompute")
    if layer_lock.get("holdout_baseline_authorized") is not True:
        raise CausalBindingAnalysisError("layer lock did not authorize holdout baseline")
    selected_layer = layer_lock.get("selected_layer")
    if (
        isinstance(selected_layer, bool)
        or not isinstance(selected_layer, int)
        or selected_layer not in design["selection_rule"]["selectable_layers"]
    ):
        raise CausalBindingAnalysisError("selected layer is invalid")
    root = runner.RESULT_ROOT
    baseline_path = root / "holdout_baselines.jsonl"
    activation_path = root / "holdout_activations.npy"
    _, activations = _validate_execution_manifest(
        root / "holdout_baseline_execution_manifest.json",
        phase="holdout_baseline",
        plan=plan,
        baseline_path=baseline_path,
        baseline_count=192,
        activation_path=activation_path,
        activation_shape=(192, runner.MODEL_LAYERS, runner.MODEL_WIDTH),
        selected_layer=selected_layer,
    )
    assert activations is not None
    baselines = _validate_baselines(
        runner.load_jsonl(baseline_path),
        runner._phase_prompts(plan, "holdout"),
        activations,
        phase="holdout_baseline",
        require_duplicate=False,
    )
    adjudication, cohort = _holdout_baseline_metrics(baselines, selected_layer)
    authorized = bool(adjudication["gate"]["all_pass"])
    status = (
        "HOLDOUT_BEHAVIOR_REPLICATED_PATCH_AUTHORIZED"
        if authorized
        else "HOLDOUT_STOP_BEHAVIOR_NOT_REPLICATED"
    )
    analysis = {
        "schema_version": HOLDOUT_BASELINE_ANALYSIS_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        **adjudication,
        "secondary_failure_cohort": cohort,
        "control_length_matching": _control_length_summary(
            runner._phase_templates(plan, "holdout")
        ),
        "call_plan_sha256": plan["call_plan_sha256"],
        "analyzer_sha256": plan["locks"]["analyzer_sha256"],
        "baseline_artifacts": {
            "records_sha256": runner.file_sha256(baseline_path),
            "activations_sha256": runner.file_sha256(activation_path),
        },
        "biological_model_calls": 0,
    }
    analysis_path = root / "holdout_baseline_analysis.json"
    runner.write_json(analysis_path, analysis)
    entry = {
        "schema_version": HOLDOUT_ENTRY_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "holdout_patch_authorized": authorized,
        "secondary_failure_cohort_item_ids": cohort["item_ids"],
        "holdout_baseline_analysis_path": str(analysis_path),
        "holdout_baseline_analysis_file_sha256": runner.file_sha256(analysis_path),
        "layer_lock_file_sha256": runner.file_sha256(runner.DEFAULT_LAYER_LOCK),
        "call_plan_sha256": plan["call_plan_sha256"],
        "analyzer_sha256": plan["locks"]["analyzer_sha256"],
        "adaptation_after_baseline": False,
        "biological_execution_allowed": False,
    }
    runner.write_json(runner.DEFAULT_HOLDOUT_ENTRY, entry)
    return analysis, entry


def _holdout_engineering(
    baselines: Sequence[Mapping[str, Any]], patches: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline = {
        (row["item_id"], row["prompt_role"]): row for row in baselines
    }
    differences = []
    for row in patches:
        if row["condition"] not in {"identity_r", "identity_d"}:
            continue
        role = "R" if row["condition"] == "identity_r" else "D"
        target = baseline[(row["item_id"], role)]["diagnostics"]
        differences.extend(
            [
                abs(row["diagnostics"]["x_logit"] - target["x_logit"]),
                abs(row["diagnostics"]["y_logit"] - target["y_logit"]),
            ]
        )
    maximum = max(differences, default=math.inf)
    return {
        "tolerance": 1e-4,
        "maximum_identity_xy_difference": maximum,
        "identity_pass": maximum <= 1e-4,
        "hook_trace_pass": all(
            row["hook_trace"]["hook_calls"] == 1
            and row["hook_trace"]["non_target_tokens_unchanged"] is True
            and row["hook_trace"]["patched_token_matches_source"] is True
            for row in patches
        ),
    }


def _flip_rate_summary(
    item_values: Mapping[str, float],
    item_to_pair: Mapping[str, str],
    *,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for item_id, value in item_values.items():
        grouped.setdefault(item_to_pair[item_id], []).append(float(value))
    if not grouped:
        raise CausalBindingAnalysisError("secondary failure cohort is empty")
    pair_rates = {
        pair_id: float(np.mean(values))
        for pair_id, values in sorted(grouped.items())
    }
    summary = _bootstrap_summary(pair_rates, draws=10000, seed=seed)
    return {
        "point": float(np.mean(list(pair_rates.values()))),
        "pair_values": pair_rates,
        "cluster_bootstrap_95": summary,
    }


def analyze_holdout_final() -> dict[str, Any]:
    plan, _ = _load_plan()
    _, recomputed_lock = analyze_discovery()
    _, recomputed_entry = analyze_holdout_baseline()
    layer_lock = _load_json(runner.DEFAULT_LAYER_LOCK)
    entry = _load_json(runner.DEFAULT_HOLDOUT_ENTRY)
    if layer_lock != recomputed_lock:
        raise CausalBindingAnalysisError("layer lock does not recompute")
    if entry != recomputed_entry:
        raise CausalBindingAnalysisError("holdout entry does not recompute")
    if entry.get("holdout_patch_authorized") is not True:
        raise CausalBindingAnalysisError("holdout patching was not authorized")
    selected_layer = entry.get("selected_layer")
    if selected_layer != layer_lock.get("selected_layer"):
        raise CausalBindingAnalysisError("layer lock and holdout entry disagree")
    root = runner.RESULT_ROOT
    baseline_path = root / "holdout_baselines.jsonl"
    activation_path = root / "holdout_activations.npy"
    patch_path = root / "holdout_patches.jsonl"
    _, activation_array = _validate_execution_manifest(
        root / "holdout_baseline_execution_manifest.json",
        phase="holdout_baseline",
        plan=plan,
        baseline_path=baseline_path,
        baseline_count=192,
        activation_path=activation_path,
        activation_shape=(192, runner.MODEL_LAYERS, runner.MODEL_WIDTH),
        selected_layer=selected_layer,
    )
    _, activations = _validate_execution_manifest(
        root / "holdout_patch_execution_manifest.json",
        phase="holdout_patch",
        plan=plan,
        baseline_path=baseline_path,
        baseline_count=192,
        patch_path=patch_path,
        patch_count=768,
        selected_layer=selected_layer,
    )
    del activations
    assert activation_array is not None
    baselines = _validate_baselines(
        runner.load_jsonl(baseline_path),
        runner._phase_prompts(plan, "holdout"),
        activation_array,
        phase="holdout_baseline",
        require_duplicate=False,
    )
    holdout_templates = runner._phase_templates(plan, "holdout")
    patches = _validate_patches(
        runner.load_jsonl(patch_path),
        holdout_templates,
        baselines,
        phase="holdout_patch",
        selected_layer=selected_layer,
    )
    metrics = _metrics_for_layer(
        baselines, patches, layer=selected_layer, bootstrap=True
    )
    expected_matching_units = {
        "same_pair_lexical_pairs": plan["control_matching_registry"][
            "same_pair_exact_length_lexical_pairs"
        ],
        "unrelated_control_dyads": plan["control_matching_registry"][
            "unrelated_exact_length_reciprocal_dyads"
        ],
    }
    if metrics["length_matched_specificity_units"] != expected_matching_units:
        raise CausalBindingAnalysisError(
            "analyzed exact-length specificity units differ from the frozen plan"
        )
    values = metrics["metrics"]
    engineering = _holdout_engineering(baselines, patches)
    paired_forward = [row for row in patches if row["condition"] == "forward_paired"]
    paired_reverse = [row for row in patches if row["condition"] == "reverse_paired"]
    baseline_r = [row for row in baselines if row["prompt_role"] == "R"]
    baseline_d = [row for row in baselines if row["prompt_role"] == "D"]
    forward_native_rate = float(
        np.mean(
            [
                row["diagnostics"]["greedy_token_id"]
                in {runner.X_TOKEN_ID, runner.Y_TOKEN_ID}
                and row["diagnostics"]["maximum_tie_count"] == 1
                for row in paired_forward
            ]
        )
    )
    reverse_native_rate = float(
        np.mean(
            [
                row["diagnostics"]["greedy_token_id"]
                in {runner.X_TOKEN_ID, runner.Y_TOKEN_ID}
                and row["diagnostics"]["maximum_tie_count"] == 1
                for row in paired_reverse
            ]
        )
    )
    forward_mass = float(
        np.mean([row["diagnostics"]["label_probability_mass"] for row in paired_forward])
    )
    reverse_mass = float(
        np.mean([row["diagnostics"]["label_probability_mass"] for row in paired_reverse])
    )
    r_mass = float(
        np.mean([row["diagnostics"]["label_probability_mass"] for row in baseline_r])
    )
    d_mass = float(
        np.mean([row["diagnostics"]["label_probability_mass"] for row in baseline_d])
    )
    transfer_gate = {
        "forward_fraction_denominator_pass": values["F"][
            "gap_denominator_positive"
        ],
        "reverse_fraction_denominator_pass": values["N"][
            "gap_denominator_positive"
        ],
        "forward_fraction_pass": (
            values["F"]["mean_over_gap"] is not None
            and values["F"]["mean_over_gap"] >= 0.30
        ),
        "reverse_fraction_pass": (
            values["N"]["mean_over_gap"] is not None
            and values["N"]["mean_over_gap"] >= 0.30
        ),
        "forward_ci_pass": values["F"]["cluster_bootstrap_95"]["lower_95"] > 0.0,
        "reverse_ci_pass": values["N"]["cluster_bootstrap_95"]["lower_95"] > 0.0,
        "forward_positive_pairs_pass": values["F"]["positive_pair_count"] >= 36,
        "reverse_positive_pairs_pass": values["N"]["positive_pair_count"] >= 36,
        "forward_native_channel_pass": forward_native_rate >= 0.95,
        "reverse_native_channel_pass": reverse_native_rate >= 0.95,
        "forward_label_mass_pass": forward_mass >= r_mass - 0.02,
        "reverse_label_mass_pass": reverse_mass >= d_mass - 0.02,
        "identity_pass": engineering["identity_pass"],
        "hook_trace_pass": engineering["hook_trace_pass"],
    }
    transfer_gate["all_pass"] = all(transfer_gate.values())
    specificity_requirements = {
        "S_F_same_exact_length": (31, 24),
        "S_N_same_exact_length": (31, 24),
        "S_F_unrelated_exact_length": (22, 17),
        "S_N_unrelated_exact_length": (22, 17),
    }
    specificity_gate = {
        name: {
            "expected_inference_clusters_pass": (
                values[name]["n_inference_clusters"] == expected_clusters
            ),
            "summary_defined_pass": values[name]["summary_defined"],
            "effect_fraction_denominator_pass": values[name][
                "gap_denominator_positive"
            ],
            "effect_fraction_pass": (
                values[name]["mean_over_gap"] is not None
                and values[name]["mean_over_gap"] >= 0.20
            ),
            "ci_pass": values[name]["cluster_bootstrap_95"]["lower_95"] > 0.0,
            "positive_clusters_pass": (
                values[name]["positive_pair_count"] >= minimum_positive
            ),
        }
        for name, (expected_clusters, minimum_positive) in (
            specificity_requirements.items()
        )
    }
    control_specificity_pass = all(
        component
        for gate in specificity_gate.values()
        for component in gate.values()
    )
    specificity_pass = bool(
        transfer_gate["all_pass"] and control_specificity_pass
    )
    secondary_ids = set(entry["secondary_failure_cohort_item_ids"])
    patch_by_item = {
        (row["item_id"], row["condition"]): row for row in patches
    }
    baseline_by_item = {
        (row["item_id"], row["prompt_role"]): row for row in baselines
    }
    item_to_pair = {row["item_id"]: row["pair_id"] for row in baselines}
    forward_flips = {
        item_id: float(
            patch_by_item[(item_id, "forward_paired")]["diagnostics"]
            ["x_minus_y_margin"]
            > 0.0
        )
        for item_id in secondary_ids
    }
    reverse_flips = {
        item_id: float(
            patch_by_item[(item_id, "reverse_paired")]["diagnostics"]
            ["x_minus_y_margin"]
            < 0.0
        )
        for item_id in secondary_ids
    }
    secondary = {
        "cohort_item_count": len(secondary_ids),
        "forward_correction": _flip_rate_summary(
            forward_flips,
            {item_id: item_to_pair[item_id] for item_id in secondary_ids},
            seed=260802,
        ),
        "reverse_damage": _flip_rate_summary(
            reverse_flips,
            {item_id: item_to_pair[item_id] for item_id in secondary_ids},
            seed=260802,
        ),
    }
    for value in (secondary["forward_correction"], secondary["reverse_damage"]):
        value["descriptive_rescue_pass"] = (
            value["point"] >= 0.25
            and value["cluster_bootstrap_95"]["lower_95"] > 0.10
        )
    if not engineering["identity_pass"] or not engineering["hook_trace_pass"]:
        status = "ENGINEERING_STOP_INVALID_PATCH_EXECUTION"
    elif not transfer_gate["all_pass"]:
        status = "NO_REPLICATED_CAUSAL_TRANSFER_AT_FROZEN_SITE"
    elif specificity_pass:
        status = "ITEM_SPECIFIC_DECISION_STATE_TRANSFER_REPLICATED"
    else:
        status = "NONSPECIFIC_DECISION_STATE_TRANSFER_REPLICATED"
    analysis = {
        "schema_version": FINAL_ANALYSIS_SCHEMA,
        "status": status,
        "selected_layer": selected_layer,
        "selection_class": layer_lock["selection_class"],
        "metrics": metrics,
        "engineering_gates": engineering,
        "native_channel": {
            "forward_paired_rate": forward_native_rate,
            "reverse_paired_rate": reverse_native_rate,
            "forward_paired_mean_label_mass": forward_mass,
            "reverse_paired_mean_label_mass": reverse_mass,
            "recipient_baseline_mean_label_mass": r_mass,
            "source_baseline_mean_label_mass": d_mass,
        },
        "transfer_gate": transfer_gate,
        "specificity_gate": specificity_gate,
        "control_specificity_pass": control_specificity_pass,
        "specificity_pass": specificity_pass,
        "secondary_failure_cohort": secondary,
        "secondary_flip_definition": {
            "forward_correction": "patched_forward_x_minus_y_margin_strictly_positive",
            "reverse_damage": "patched_reverse_x_minus_y_margin_strictly_negative",
            "scope": "within_xy_channel_sign_not_global_argmax",
            "bootstrap_seed": 260802,
        },
        "control_length_matching": _control_length_summary(holdout_templates),
        "claim_boundary": (
            "model_prompt_site_bank_specific_causal_state_transfer_only_no_necessity_"
            "bottleneck_biology_latent_knowledge_activation_gap_or_physical_law"
        ),
        "call_plan_sha256": plan["call_plan_sha256"],
        "analyzer_sha256": plan["locks"]["analyzer_sha256"],
        "raw_artifacts": {
            "baselines_sha256": runner.file_sha256(baseline_path),
            "activations_sha256": runner.file_sha256(activation_path),
            "patches_sha256": runner.file_sha256(patch_path),
        },
        "biological_model_calls": 0,
    }
    analysis_path = root / "analysis.json"
    runner.write_json(analysis_path, analysis)
    report_path = root / "analysis.md"
    markdown = _render_final_markdown(analysis)
    runner._write_bytes(report_path, markdown.encode("utf-8"))
    runner.write_json(
        root / "analysis_manifest.json",
        {
            "schema_version": "coherent-readout-v2-causal-binding-analysis-manifest-v1",
            "status": status,
            "analysis_path": str(analysis_path),
            "analysis_file_sha256": runner.file_sha256(analysis_path),
            "report_path": str(report_path),
            "report_file_sha256": runner.file_sha256(report_path),
            "analyzer_sha256": runner.file_sha256(Path(__file__)),
            "runner_sha256": runner.file_sha256(Path(runner.__file__)),
            "call_plan_sha256": plan["call_plan_sha256"],
            "biological_model_calls": 0,
        },
    )
    return analysis


def _render_final_markdown(analysis: Mapping[str, Any]) -> str:
    metrics = analysis["metrics"]["metrics"]
    selected = analysis["selected_layer"]
    forward = metrics["F"]["mean_over_gap"]
    reverse = metrics["N"]["mean_over_gap"]
    forward_text = "undefined" if forward is None else f"{forward:.3f}"
    reverse_text = "undefined" if reverse is None else f"{reverse:.3f}"
    status = analysis["status"]
    if status == "ITEM_SPECIFIC_DECISION_STATE_TRANSFER_REPLICATED":
        english = (
            f"Forward paired replacement recovered {forward_text} of the held-out "
            f"behavioral gap and reverse replacement transferred {reverse_text} in "
            "the damaging direction. Transfer and preregistered exact-length "
            "specificity gates passed. This supports a bounded item-specific causal "
            "state-transfer claim at the frozen site."
        )
        korean = (
            f"Paired forward replacement는 holdout 행동 격차의 {forward_text}를 "
            f"복구했고 reverse replacement는 손상 방향으로 {reverse_text}를 "
            "전달했다. Transfer와 사전 등록된 exact-length specificity gate가 "
            "통과했으므로, frozen site에서 제한적인 item-specific causal "
            "state-transfer를 지지한다."
        )
    elif status == "NONSPECIFIC_DECISION_STATE_TRANSFER_REPLICATED":
        english = (
            f"Forward and reverse transfer gates passed ({forward_text}, "
            f"{reverse_text}), but the preregistered specificity gate did not. The "
            "result supports only nonspecific decision-state transfer, not "
            "item-specific binding."
        )
        korean = (
            f"Forward·reverse transfer gate는 통과했다({forward_text}, "
            f"{reverse_text}). 그러나 사전 등록된 specificity gate는 통과하지 "
            "못했으므로 item-specific binding이 아니라 nonspecific "
            "decision-state transfer만 지지한다."
        )
    elif status == "NO_REPLICATED_CAUSAL_TRANSFER_AT_FROZEN_SITE":
        english = (
            "The preregistered transfer gate did not pass. No replicated causal "
            "transfer was found at the frozen site; this does not rule out relevant "
            "states at untested sites."
        )
        korean = (
            "사전 등록된 transfer gate가 통과하지 않았다. Frozen site에서 "
            "replicated causal transfer를 발견하지 못했으며, 검사하지 않은 다른 "
            "site의 관련 상태까지 부정하지는 않는다."
        )
    else:
        english = (
            "An engineering gate failed, so the patch execution has no scientific "
            "interpretation."
        )
        korean = (
            "Engineering gate가 실패했으므로 patch 실행에는 과학적 해석 권한이 "
            "없다."
        )
    return f"""# Causal decision-state transfer result

Status: **`{status}`**

Frozen decoder block: **{selected}** (`resid_post`, final context token)

## English

{english}

No status establishes necessity, a unique binding circuit, biology, latent
knowledge, an activation gap, or a physical law.
Any positive result is specific to Qwen2.5-1.5B, this exact prompt, the frozen
final-token site, and this lexical bank. It does not generalize to a model family
or establish a general variable-binding mechanism.

## 한국어

{korean}

어떤 status도 necessity, 고유 binding circuit, 생물학, 잠재지식, activation gap
또는 물리 법칙을 확립하지 않는다.
양성 결과가 있더라도 Qwen2.5-1.5B, 이 exact prompt, frozen final-token site와
해당 lexical bank에만 한정된다. Model-family 일반성이나 보편적 variable-binding
mechanism을 확립하지 않는다.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("discovery", "holdout-baseline", "holdout-final"), required=True
    )
    args = parser.parse_args()
    try:
        if args.phase == "discovery":
            analysis, lock = analyze_discovery()
            print(json.dumps({"status": analysis["status"], "layer_lock": lock}, sort_keys=True))
        elif args.phase == "holdout-baseline":
            analysis, entry = analyze_holdout_baseline()
            print(json.dumps({"status": analysis["status"], "entry": entry}, sort_keys=True))
        else:
            analysis = analyze_holdout_final()
            print(json.dumps({"status": analysis["status"]}, sort_keys=True))
    except (
        CausalBindingAnalysisError,
        runner.CausalBindingRunnerError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

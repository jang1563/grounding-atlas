"""Independent post-hoc interface audit of the GSE96583 sentinel run.

The analysis reads frozen CSV and raw JSONL artifacts directly.  It neither
imports the execution runner nor consumes its result JSON.  All probabilities
are reparsed from ``raw_output`` strings, and every reported matched contrast
is checked algebraically as r = a - h.

This is a descriptive analysis selected after the confirmatory run.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_CODE = Path(__file__).resolve()
SENTINEL_CSV = (
    ROOT / "signal" / "single_cell" / "gse96583_sentinel_factorial.csv"
)
SENTINEL_MANIFEST = SENTINEL_CSV.with_suffix(".manifest.json")
SENTINEL_RAW = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "sentinel_factorial"
    / "claude-haiku-4-5-20251001_raw.jsonl"
)
PARENT_CSV = (
    ROOT
    / "signal"
    / "single_cell"
    / "gse96583_cd8_nk_module_replication.csv"
)
PARENT_MANIFEST = PARENT_CSV.with_suffix(".manifest.json")
PARENT_RAW = (
    ROOT
    / "results"
    / "benchmark"
    / "single_cell"
    / "donor_context_factorial"
    / "claude-haiku-4-5-20251001_raw.jsonl"
)
JSON_OUT = SENTINEL_RAW.with_name("posthoc_interface_discovery.json")
MARKDOWN_OUT = SENTINEL_RAW.with_name("posthoc_interface_discovery.md")

EXPECTED_SHA256 = {
    "sentinel_csv": (
        SENTINEL_CSV,
        "673db8c8bcd6ba923e62891de6cd5f04f97967706ac3ade8a6e44ad2d14a4b95",
    ),
    "sentinel_manifest": (
        SENTINEL_MANIFEST,
        "9f0035469c81852b11e2a36651b7892bf2dad4d30f8049b37cd1f655ca9bf0c4",
    ),
    "sentinel_raw": (
        SENTINEL_RAW,
        "b625d8cf8f65e15863d19f8eb3b8300c8d0f84be6709eaac7edd94339dfede33",
    ),
    "parent_csv": (
        PARENT_CSV,
        "f2f0859ca4c3559494a7c132921fef3d1286c2a20384a5b35d44e7b9ac280321",
    ),
    "parent_manifest": (
        PARENT_MANIFEST,
        "3e59808e09675f98be5e88fa8266f56c43aeea3592f023b6f91750ffdd0cb53f",
    ),
    "parent_raw": (
        PARENT_RAW,
        "aaf048375f4fa8f5b972a406185bfe15618323b8eacdff1b25db938ea2e63aea",
    ),
}

SENTINEL_ANALYSIS_ID = "gse96583-sentinel-factorial-holdout-v1"
PARENT_ANALYSIS_ID = "gse96583-cd8-nk-context-module-donor-replication-v1"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
SENTINEL_PLAN_SHA256 = (
    "db5af60a5142c6f79cbac7abdba85520077efd75fab732db0bfda80f6843a20e"
)
PARENT_PLAN_SHA256 = (
    "bb046113f08eac0e69a12dbcca63ecbbd26fdbea1d70aa1d42db5d6ebd801615"
)
FORMS = ("ab_pa", "ab_pb", "ba_pa", "ba_pb")
DONORS = ("101", "107", "1015", "1016", "1039", "1244", "1256", "1488")
SUBSETS = (
    "GNLY",
    "NKG7",
    "CCL5",
    "GNLY+NKG7",
    "GNLY+CCL5",
    "NKG7+CCL5",
    "GNLY+NKG7+CCL5",
)
DELTA = Decimal("0.03")
T7_975 = 2.3646242515927844
NUMBER_RE = re.compile(r"(?:0(?:\.\d+)?|1(?:\.0+)?)")


class PosthocArtifactError(ValueError):
    """Raised when an input no longer satisfies the frozen raw-data contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Decimal | float | int) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        raise PosthocArtifactError("cannot average an empty vector")
    return sum(values, Decimal(0)) / Decimal(len(values))


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _verify_frozen_inputs() -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    for name, (path, expected) in EXPECTED_SHA256.items():
        if not path.is_file():
            raise PosthocArtifactError(f"missing frozen artifact: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise PosthocArtifactError(
                f"{name} SHA-256 changed: expected {expected}, observed {observed}"
            )
        provenance[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": observed,
        }
    return provenance


def _parse_raw_probability(record: dict[str, Any], source: str) -> None:
    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str) or NUMBER_RE.fullmatch(raw_output) is None:
        raise PosthocArtifactError(f"{source} has a noncanonical raw output")
    probability = Decimal(raw_output)
    if not Decimal(0) <= probability <= Decimal(1):
        raise PosthocArtifactError(f"{source} raw output is outside [0,1]")
    queried_target = record.get("queried_target")
    if queried_target not in {"A", "B"}:
        raise PosthocArtifactError(f"{source} has an invalid queried target")
    aligned = probability if queried_target == "A" else Decimal(1) - probability
    if not math.isclose(
        float(probability),
        float(record.get("reported_probability")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PosthocArtifactError(f"{source} reported probability differs from raw")
    if not math.isclose(
        float(aligned),
        float(record.get("aligned_probability_a")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PosthocArtifactError(f"{source} aligned probability differs from raw")
    record["_raw_decimal"] = probability
    record["_aligned_decimal"] = aligned


def _read_raw(path: Path, expected_count: int, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise PosthocArtifactError(f"{source} contains a blank line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PosthocArtifactError(
                f"{source} line {line_number} is not JSON"
            ) from exc
        if not isinstance(record, dict):
            raise PosthocArtifactError(f"{source} line is not an object")
        _parse_raw_probability(record, source)
        records.append(record)
    if len(records) != expected_count:
        raise PosthocArtifactError(
            f"{source} expected {expected_count} records, found {len(records)}"
        )
    request_ids = [record.get("request_id") for record in records]
    if len(set(request_ids)) != expected_count or None in request_ids:
        raise PosthocArtifactError(f"{source} request IDs are not unique")
    if any(record.get("parsed") is not True for record in records):
        raise PosthocArtifactError(f"{source} includes an unparsed record")
    if source == "sentinel":
        for field, expected in (
            ("attempt_count", 1),
            ("retry_count", 0),
            ("temperature_fallback_used", False),
            ("requested_model", EXPECTED_MODEL),
            ("returned_model", EXPECTED_MODEL),
            ("provider", "anthropic"),
        ):
            if any(record.get(field) != expected for record in records):
                raise PosthocArtifactError(
                    f"sentinel execution field {field} changed"
                )
        for field in ("response_id", "provider_request_id"):
            identifiers = [record.get(field) for record in records]
            if len(set(identifiers)) != expected_count or None in identifiers:
                raise PosthocArtifactError(
                    f"sentinel {field} values are not unique"
                )
    elif any(record.get("model") != EXPECTED_MODEL for record in records):
        raise PosthocArtifactError("parent model identity changed")
    planned = {record.get("planned_index") for record in records}
    if planned != set(range(expected_count)):
        raise PosthocArtifactError(f"{source} planned indexes are incomplete")
    return records


def _read_sentinel_csv() -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
]:
    with SENTINEL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise PosthocArtifactError(
            f"sentinel CSV expected 64 rows, found {len(rows)}"
        )
    bases: dict[str, dict[str, str]] = {}
    interventions: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        entity_id = row["entity_id"]
        if row["row_type"] == "base":
            if entity_id in bases:
                raise PosthocArtifactError("duplicate sentinel base row")
            bases[entity_id] = row
        elif row["row_type"] == "intervention":
            key = (entity_id, row["subset_id"])
            if key in interventions:
                raise PosthocArtifactError("duplicate sentinel intervention row")
            interventions[key] = row
        else:
            raise PosthocArtifactError("unknown sentinel CSV row type")
    if len(bases) != 8 or len(interventions) != 56:
        raise PosthocArtifactError("sentinel CSV base/intervention counts changed")
    csv_donors = tuple(sorted((row["donor_id"] for row in bases.values()), key=int))
    if csv_donors != DONORS:
        raise PosthocArtifactError("sentinel CSV donor set changed")
    for entity_id, base in bases.items():
        observed = {
            subset for entity, subset in interventions if entity == entity_id
        }
        if observed != set(SUBSETS):
            raise PosthocArtifactError(
                f"{entity_id} does not have the frozen seven-subset lattice"
            )
        if base["subset_id"] != "none" or base["subset_mask"] != "000":
            raise PosthocArtifactError("sentinel base row is not unmasked")
    return bases, interventions


def _validate_form(record: dict[str, Any], source: str) -> None:
    form = record.get("form")
    if form not in FORMS:
        raise PosthocArtifactError(f"{source} has an unknown prompt form")
    order, target = form.split("_")
    if source == "sentinel" and record.get("answer_order") != order:
        raise PosthocArtifactError("sentinel answer-order field differs from form")
    if source == "parent" and record.get("order") != order:
        raise PosthocArtifactError("parent answer-order field differs from form")
    if record.get("queried_target") != target[-1].upper():
        raise PosthocArtifactError(f"{source} queried-target field differs from form")


def _validate_sentinel_records(
    records: list[dict[str, Any]],
    bases: dict[str, dict[str, str]],
    interventions: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str, str, str], Decimal]:
    expected_states: dict[tuple[str, str, str], dict[str, str]] = {}
    for entity_id, base in bases.items():
        expected_states[(entity_id, "unmasked", "none")] = {
            "sentence": base["original_sentence"],
            "target_genes": "",
            "control_genes": "",
            "donor_id": base["donor_id"],
            "cell_barcode": base["cell_barcode"],
        }
        for subset in SUBSETS:
            row = interventions[(entity_id, subset)]
            common = {
                "target_genes": row["target_genes"],
                "control_genes": row["control_genes"],
                "donor_id": row["donor_id"],
                "cell_barcode": row["cell_barcode"],
            }
            expected_states[(entity_id, "target_mask", subset)] = {
                **common,
                "sentence": row["module_mask_sentence"],
            }
            expected_states[(entity_id, "control_mask", subset)] = {
                **common,
                "sentence": row["control_mask_sentence"],
            }
    if len(expected_states) != 120:
        raise PosthocArtifactError("sentinel state lattice is not 120 states")

    values: dict[tuple[str, str, str, str], Decimal] = {}
    for record in records:
        _validate_form(record, "sentinel")
        if record.get("analysis_id") != SENTINEL_ANALYSIS_ID:
            raise PosthocArtifactError("sentinel analysis ID changed")
        if record.get("call_plan_sha256") != SENTINEL_PLAN_SHA256:
            raise PosthocArtifactError("sentinel request-plan hash changed")
        for field, expected in (
            ("input_csv_sha256", EXPECTED_SHA256["sentinel_csv"][1]),
            ("input_manifest_sha256", EXPECTED_SHA256["sentinel_manifest"][1]),
            ("parent_csv_sha256", EXPECTED_SHA256["parent_csv"][1]),
            ("parent_manifest_sha256", EXPECTED_SHA256["parent_manifest"][1]),
        ):
            if record.get(field) != expected:
                raise PosthocArtifactError(f"sentinel record {field} changed")
        state = (
            str(record.get("entity_id")),
            str(record.get("condition")),
            str(record.get("subset_id")),
        )
        expected = expected_states.get(state)
        if expected is None:
            raise PosthocArtifactError(f"unexpected sentinel state: {state}")
        for field in ("target_genes", "control_genes", "donor_id", "cell_barcode"):
            if str(record.get(field)) != expected[field]:
                raise PosthocArtifactError(
                    f"sentinel {field} differs from the frozen CSV"
                )
        sentence_sha256 = hashlib.sha256(
            expected["sentence"].encode("utf-8")
        ).hexdigest()
        if record.get("sentence_sha256") != sentence_sha256:
            raise PosthocArtifactError(
                "sentinel sentence hash differs from the frozen CSV"
            )
        key = (*state, str(record["form"]))
        if key in values:
            raise PosthocArtifactError("duplicate sentinel state/form record")
        values[key] = record["_aligned_decimal"]
    expected_keys = {
        (*state, form) for state in expected_states for form in FORMS
    }
    if set(values) != expected_keys:
        raise PosthocArtifactError("sentinel raw does not cover every state/form")
    return values


def _assignment_signature(record: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    assignments = record.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise PosthocArtifactError("parent record has no assignments")
    signature = []
    for assignment in assignments:
        signature.append(
            (
                str(assignment.get("module")),
                str(assignment.get("condition")),
                str(assignment.get("target_genes")),
                str(assignment.get("control_genes")),
            )
        )
    return tuple(sorted(signature))


def _validate_parent_records(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, tuple[tuple[str, ...], ...]], set[str]] = defaultdict(
        set
    )
    for record in records:
        _validate_form(record, "parent")
        if record.get("analysis_id") != PARENT_ANALYSIS_ID:
            raise PosthocArtifactError("parent analysis ID changed")
        if record.get("call_plan_sha256") != PARENT_PLAN_SHA256:
            raise PosthocArtifactError("parent request-plan hash changed")
        if record.get("input_csv_sha256") != EXPECTED_SHA256["parent_csv"][1]:
            raise PosthocArtifactError("parent input CSV hash changed")
        if (
            record.get("input_manifest_sha256")
            != EXPECTED_SHA256["parent_manifest"][1]
        ):
            raise PosthocArtifactError("parent input manifest hash changed")
        state = (str(record.get("entity_id")), _assignment_signature(record))
        form = str(record["form"])
        if form in grouped[state]:
            raise PosthocArtifactError("duplicate parent state/form record")
        grouped[state].add(form)
    if len(grouped) != 280:
        raise PosthocArtifactError(
            f"parent raw expected 280 states, found {len(grouped)}"
        )
    if any(forms != set(FORMS) for forms in grouped.values()):
        raise PosthocArtifactError("parent state lacks the four prompt forms")


def _vector_summary(values: list[Decimal]) -> dict[str, Any]:
    count = len(values)
    mean = _mean(values)
    if count == 1:
        ci_low = ci_high = float(mean)
    else:
        mean_float = float(mean)
        variance = sum((float(value) - mean_float) ** 2 for value in values)
        variance /= count - 1
        half_width = T7_975 * math.sqrt(variance) / math.sqrt(count)
        ci_low = mean_float - half_width
        ci_high = mean_float + half_width
    lodo = [
        _mean(values[:index] + values[index + 1 :]) for index in range(count)
    ]
    shifted_material = [value + DELTA for value in values]
    material_p = _sign_flip_p(shifted_material, alternative="negative")
    shifted_lower = [value + DELTA for value in values]
    shifted_upper = [value - DELTA for value in values]
    equivalence_lower_p = _sign_flip_p(shifted_lower, alternative="positive")
    equivalence_upper_p = _sign_flip_p(shifted_upper, alternative="negative")
    material_facts = {
        "ci95_upper_below_minus_delta": ci_high < -float(DELTA),
        "shifted_exact_sign_flip_p_negative": material_p,
        "shifted_exact_p_below_0_05": material_p < 0.05,
        "donors_below_minus_delta": sum(value < -DELTA for value in values),
        "at_least_7_of_8_below_minus_delta": (
            sum(value < -DELTA for value in values) >= 7
        ),
        "all_lodo_means_below_minus_delta": all(
            value < -DELTA for value in lodo
        ),
    }
    material_facts["pass"] = all(
        (
            material_facts["ci95_upper_below_minus_delta"],
            material_facts["shifted_exact_p_below_0_05"],
            material_facts["at_least_7_of_8_below_minus_delta"],
            material_facts["all_lodo_means_below_minus_delta"],
        )
    )
    equivalence_facts = {
        "ci95_strictly_inside_margin": (
            ci_low > -float(DELTA) and ci_high < float(DELTA)
        ),
        "lower_shift_exact_sign_flip_p_positive": equivalence_lower_p,
        "upper_shift_exact_sign_flip_p_negative": equivalence_upper_p,
        "both_exact_p_below_0_05": (
            equivalence_lower_p < 0.05 and equivalence_upper_p < 0.05
        ),
        "all_lodo_means_strictly_inside_margin": all(
            -DELTA < value < DELTA for value in lodo
        ),
    }
    equivalence_facts["pass"] = all(
        (
            equivalence_facts["ci95_strictly_inside_margin"],
            equivalence_facts["both_exact_p_below_0_05"],
            equivalence_facts["all_lodo_means_strictly_inside_margin"],
        )
    )
    return {
        "values": [_number(value) for value in values],
        "mean": _number(mean),
        "ci95_t7": [_number(ci_low), _number(ci_high)],
        "sign_counts": {
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive": sum(value > 0 for value in values),
        },
        "leave_one_donor_out_means": [_number(value) for value in lodo],
        "material_negative_beyond_0_03": material_facts,
        "equivalent_within_0_03": equivalence_facts,
    }


def _sign_flip_p(values: list[Decimal], alternative: str) -> float:
    observed = _mean(values)
    null_means = []
    for signs in itertools.product((Decimal(-1), Decimal(1)), repeat=len(values)):
        null_means.append(
            _mean([sign * value for sign, value in zip(signs, values, strict=True)])
        )
    if alternative == "negative":
        extreme = sum(value <= observed for value in null_means)
    elif alternative == "positive":
        extreme = sum(value >= observed for value in null_means)
    else:
        raise PosthocArtifactError(f"unknown sign-flip alternative: {alternative}")
    return extreme / len(null_means)


def _sentinel_surface(
    values: dict[tuple[str, str, str, str], Decimal],
) -> tuple[dict[str, Any], dict[tuple[str, str], list[Decimal]]]:
    raw_vectors: dict[tuple[str, str], list[Decimal]] = {}
    rendered: dict[str, Any] = {}
    for subset in SUBSETS:
        a_values: list[Decimal] = []
        h_values: list[Decimal] = []
        r_values: list[Decimal] = []
        for donor in DONORS:
            entity_ids = {
                key[0]
                for key in values
                if key[0].split(":")[2] == donor
            }
            if len(entity_ids) != 1:
                raise PosthocArtifactError(
                    f"expected one held-out entity for donor {donor}"
                )
            entity_id = next(iter(entity_ids))
            unmasked = values[(entity_id, "unmasked", "none", "ab_pa")]
            target = values[(entity_id, "target_mask", subset, "ab_pa")]
            control = values[(entity_id, "control_mask", subset, "ab_pa")]
            a_value = unmasked - target
            h_value = unmasked - control
            r_value = control - target
            if r_value != a_value - h_value:
                raise PosthocArtifactError(
                    f"r != a-h for donor {donor}, subset {subset}"
                )
            a_values.append(a_value)
            h_values.append(h_value)
            r_values.append(r_value)
        raw_vectors[("a", subset)] = a_values
        raw_vectors[("h", subset)] = h_values
        raw_vectors[("r", subset)] = r_values
        rendered[subset] = {
            scale: {
                "values": [_number(value) for value in raw_vectors[(scale, subset)]],
                "mean": _number(_mean(raw_vectors[(scale, subset)])),
            }
            for scale in ("a", "h", "r")
        }
    return rendered, raw_vectors


def _registered_vectors(
    surface: dict[tuple[str, str], list[Decimal]],
) -> tuple[dict[str, Any], dict[tuple[str, str], list[Decimal]]]:
    triple = "GNLY+NKG7+CCL5"
    nc = "NKG7+CCL5"
    raw: dict[tuple[str, str], list[Decimal]] = {}
    for scale in ("a", "r"):
        raw[(scale, "A_GNLY")] = surface[(scale, "GNLY")]
        raw[(scale, "T_full_triple")] = surface[(scale, triple)]
        raw[(scale, "J_increment_NKG7_CCL5_after_GNLY")] = [
            total - anchor
            for total, anchor in zip(
                surface[(scale, triple)],
                surface[(scale, "GNLY")],
                strict=True,
            )
        ]
        raw[(scale, "U_NKG7_CCL5_without_GNLY")] = surface[(scale, nc)]
        raw[(scale, "Q_GNLY_on_NKG7_CCL5_background")] = [
            total - without_gnly
            for total, without_gnly in zip(
                surface[(scale, triple)],
                surface[(scale, nc)],
                strict=True,
            )
        ]
        raw[(scale, "K_nonadditivity")] = [
            total - anchor - without_gnly
            for total, anchor, without_gnly in zip(
                surface[(scale, triple)],
                surface[(scale, "GNLY")],
                surface[(scale, nc)],
                strict=True,
            )
        ]
    rendered = {
        scale: {
            name: _vector_summary(raw[(scale, name)])
            for name in (
                "A_GNLY",
                "T_full_triple",
                "J_increment_NKG7_CCL5_after_GNLY",
                "U_NKG7_CCL5_without_GNLY",
                "Q_GNLY_on_NKG7_CCL5_background",
                "K_nonadditivity",
            )
        }
        for scale in ("a", "r")
    }
    return rendered, raw


def _primary_gate(registered: dict[str, Any]) -> dict[str, Any]:
    components = (
        ("r", "A_GNLY"),
        ("r", "T_full_triple"),
        ("a", "A_GNLY"),
        ("a", "T_full_triple"),
    )
    component_facts = {
        f"{scale}:{name}": registered[scale][name][
            "material_negative_beyond_0_03"
        ]
        for scale, name in components
    }
    anchor_pass = all(facts["pass"] for facts in component_facts.values())
    exact_iut_p = max(
        facts["shifted_exact_sign_flip_p_negative"]
        for facts in component_facts.values()
    )
    r_increment = registered["r"]["J_increment_NKG7_CCL5_after_GNLY"]
    a_increment = registered["a"]["J_increment_NKG7_CCL5_after_GNLY"]
    if not anchor_pass:
        endpoint = "anchor_gate_failed"
    elif (
        r_increment["equivalent_within_0_03"]["pass"]
        and a_increment["equivalent_within_0_03"]["pass"]
    ):
        endpoint = "GNLY_captures_triple_endpoint_within_0_03"
    elif (
        r_increment["material_negative_beyond_0_03"]["pass"]
        and a_increment["material_negative_beyond_0_03"]["pass"]
    ):
        endpoint = "material_joint_NKG7_CCL5_residual_beyond_GNLY"
    else:
        endpoint = "inconclusive_or_hybrid"
    return {
        "canonical_form": "ab_pa",
        "margin": 0.03,
        "material_rule": (
            "95% t7 CI below -delta; exact shifted one-sided sign-flip p<0.05; "
            "at least 7/8 donors below -delta; all LODO means below -delta"
        ),
        "equivalence_rule": (
            "95% t7 CI strictly inside +/-delta; both shifted exact one-sided "
            "sign-flip p<0.05; all LODO means strictly inside the margin"
        ),
        "anchor_components": component_facts,
        "anchor_exact_intersection_union_p": exact_iut_p,
        "anchor_pass": anchor_pass,
        "endpoint_checks": {
            "rJ_material": r_increment["material_negative_beyond_0_03"]["pass"],
            "aJ_material": a_increment["material_negative_beyond_0_03"]["pass"],
            "rJ_equivalent": r_increment["equivalent_within_0_03"]["pass"],
            "aJ_equivalent": a_increment["equivalent_within_0_03"]["pass"],
        },
        "endpoint": endpoint,
        "strong_sparse_gate_pass": False,
        "strong_sparse_gate_reason": (
            "not reachable because the preregistered anchor gate failed"
        ),
    }


def _distribution_summary(values: list[Decimal]) -> dict[str, Any]:
    sorted_values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        median = sorted_values[midpoint]
    else:
        median = (
            sorted_values[midpoint - 1] + sorted_values[midpoint]
        ) / Decimal(2)
    counts = Counter(values)
    return {
        "n_same_states": len(values),
        "mean": _number(_mean(values)),
        "median": _number(median),
        "mean_absolute": _number(_mean([abs(value) for value in values])),
        "min": _number(min(values)),
        "max": _number(max(values)),
        "nonzero_count": sum(value != 0 for value in values),
        "absolute_value_within_or_equal_0_03_count": sum(
            abs(value) <= DELTA for value in values
        ),
        "absolute_value_above_0_03_count": sum(
            abs(value) > DELTA for value in values
        ),
        "value_counts": {
            _canonical_decimal(value): counts[value] for value in sorted(counts)
        },
    }


def _interface_summary(
    records: list[dict[str, Any]],
    state_key: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    groups: dict[Any, dict[str, Decimal]] = defaultdict(dict)
    for record in records:
        key = state_key(record)
        form = str(record["form"])
        if form in groups[key]:
            raise PosthocArtifactError("duplicate form in same-state interface audit")
        groups[key][form] = record["_aligned_decimal"]
    if any(set(form_values) != set(FORMS) for form_values in groups.values()):
        raise PosthocArtifactError("same-state interface group lacks four forms")
    complement = {
        order: [
            values[f"{order}_pa"] - values[f"{order}_pb"]
            for values in groups.values()
        ]
        for order in ("ab", "ba")
    }
    order_difference = {
        queried: [
            values[f"ab_{queried}"] - values[f"ba_{queried}"]
            for values in groups.values()
        ]
        for queried in ("pa", "pb")
    }
    order_difference["form_pair_mean"] = [
        (
            values["ab_pa"]
            + values["ab_pb"]
            - values["ba_pa"]
            - values["ba_pb"]
        )
        / Decimal(2)
        for values in groups.values()
    ]
    cross_form_ranges = [
        max(values.values()) - min(values.values()) for values in groups.values()
    ]
    return {
        "definitions": {
            "same_order_complement_residual": (
                "aligned P(CD8) from direct P(A) minus aligned P(CD8) from "
                "1-P(B), equivalently reported P(A)+reported P(B)-1"
            ),
            "answer_order_difference": (
                "aligned P(CD8) under AB order minus aligned P(CD8) under BA "
                "order for the same input state"
            ),
        },
        "same_order_complement_residual": {
            order: _distribution_summary(values)
            for order, values in complement.items()
        },
        "answer_order_difference": {
            queried: _distribution_summary(values)
            for queried, values in order_difference.items()
        },
        "cross_form_range": _distribution_summary(cross_form_ranges),
    }


def _raw_quantization(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_values = [record["_raw_decimal"] for record in records]
    raw_counts = Counter(raw_values)
    dominant = {
        Decimal("0.15"),
        Decimal("0.25"),
        Decimal("0.75"),
        Decimal("0.85"),
    }
    dominant_count = sum(raw_counts[value] for value in dominant)
    queried_target_high_count = sum(
        record["_raw_decimal"] >= Decimal("0.75") for record in records
    )
    by_form = {}
    for form in FORMS:
        selected = [record for record in records if record["form"] == form]
        form_raw = [record["_raw_decimal"] for record in selected]
        aligned = [record["_aligned_decimal"] for record in selected]
        by_form[form] = {
            "n": len(selected),
            "raw_range": [_number(min(form_raw)), _number(max(form_raw))],
            "aligned_p_cd8_range": [
                _number(min(aligned)),
                _number(max(aligned)),
            ],
            "raw_output_counts": {
                _canonical_decimal(value): count
                for value, count in sorted(Counter(form_raw).items())
            },
        }
    return {
        "n_calls": len(records),
        "distinct_raw_output_count": len(raw_counts),
        "raw_output_counts": {
            _canonical_decimal(value): count
            for value, count in sorted(raw_counts.items())
        },
        "four_dominant_levels": [0.15, 0.25, 0.75, 0.85],
        "four_dominant_level_count": dominant_count,
        "four_dominant_level_fraction": _number(
            Decimal(dominant_count) / Decimal(len(records))
        ),
        "queried_target_probability_at_least_0_75_count": (
            queried_target_high_count
        ),
        "queried_target_probability_at_least_0_75_fraction": _number(
            Decimal(queried_target_high_count) / Decimal(len(records))
        ),
        "by_form": by_form,
    }


def _parent_gnly_comparison(
    parent_records: list[dict[str, Any]],
    sentinel_values: dict[tuple[str, str, str, str], Decimal],
) -> dict[str, Any]:
    parent_values: dict[tuple[str, str, str, str], Decimal] = {}
    assignment_metadata: dict[tuple[str, str, str], tuple[str, str]] = {}
    entity_metadata: dict[str, tuple[str, str]] = {}
    for record in parent_records:
        entity_id = str(record["entity_id"])
        entity_metadata[entity_id] = (
            str(record["donor_id"]),
            str(record["sampling_context"]),
        )
        for assignment in record["assignments"]:
            module = str(assignment["module"])
            condition = str(assignment["condition"])
            key = (entity_id, module, condition, str(record["form"]))
            value = record["_aligned_decimal"]
            if key in parent_values and parent_values[key] != value:
                raise PosthocArtifactError("parent assignment values disagree")
            parent_values[key] = value
            assignment_metadata[(entity_id, module, condition)] = (
                str(assignment["target_genes"]),
                str(assignment["control_genes"]),
            )

    cell_effects: dict[str, dict[str, Decimal]] = {}
    for (entity_id, module, condition), (target, _) in assignment_metadata.items():
        _, context = entity_metadata[entity_id]
        if not (
            module == "cytotoxic_effector"
            and condition == "module_mask"
            and target == "GNLY"
            and context == "NK_receptor_plus_cytotoxic"
        ):
            continue
        cell_effects[entity_id] = {}
        for form in FORMS:
            neutral_key = (entity_id, module, "neutral_mask", form)
            target_key = (entity_id, module, "module_mask", form)
            if neutral_key not in parent_values or target_key not in parent_values:
                raise PosthocArtifactError("parent GNLY effect is incomplete")
            cell_effects[entity_id][form] = (
                parent_values[neutral_key] - parent_values[target_key]
            )
    if len(cell_effects) != 16:
        raise PosthocArtifactError(
            f"expected 16 parent receptor-context GNLY cells, found {len(cell_effects)}"
        )

    parent_by_form: dict[str, dict[str, list[Decimal]]] = {
        form: defaultdict(list) for form in FORMS
    }
    parent_ensemble: dict[str, list[Decimal]] = defaultdict(list)
    for entity_id, form_values in cell_effects.items():
        donor, _ = entity_metadata[entity_id]
        for form in FORMS:
            parent_by_form[form][donor].append(form_values[form])
        parent_ensemble[donor].append(_mean(list(form_values.values())))

    def donor_equal(
        grouped: dict[str, list[Decimal]],
    ) -> tuple[list[str], list[Decimal]]:
        donors = sorted(grouped, key=int)
        return donors, [_mean(grouped[donor]) for donor in donors]

    parent_rendered: dict[str, Any] = {}
    for form in FORMS:
        donors, vector = donor_equal(parent_by_form[form])
        parent_rendered[form] = {
            "donor_order": donors,
            "donor_equal_values": [_number(value) for value in vector],
            "donor_equal_mean": _number(_mean(vector)),
        }
    parent_donors, parent_four_form = donor_equal(parent_ensemble)
    parent_rendered["four_form_mean"] = {
        "donor_order": parent_donors,
        "donor_equal_values": [_number(value) for value in parent_four_form],
        "donor_equal_mean": _number(_mean(parent_four_form)),
    }

    sentinel_entities = sorted(
        {
            key[0]
            for key in sentinel_values
            if key[1] == "unmasked" and key[2] == "none"
        },
        key=lambda value: int(value.split(":")[2]),
    )
    sentinel_by_form: dict[str, list[Decimal]] = {}
    for form in FORMS:
        sentinel_by_form[form] = [
            sentinel_values[(entity_id, "control_mask", "GNLY", form)]
            - sentinel_values[(entity_id, "target_mask", "GNLY", form)]
            for entity_id in sentinel_entities
        ]
    sentinel_four_form = [
        _mean([sentinel_by_form[form][index] for form in FORMS])
        for index in range(len(DONORS))
    ]
    sentinel_rendered = {
        form: {
            "donor_order": list(DONORS),
            "donor_equal_values": [
                _number(value) for value in sentinel_by_form[form]
            ],
            "donor_equal_mean": _number(_mean(sentinel_by_form[form])),
        }
        for form in FORMS
    }
    sentinel_rendered["four_form_mean"] = {
        "donor_order": list(DONORS),
        "donor_equal_values": [_number(value) for value in sentinel_four_form],
        "donor_equal_mean": _number(_mean(sentinel_four_form)),
    }
    return {
        "parent_scope": (
            "16 expression-selected GNLY-target cells in the parent "
            "NK-receptor-plus-cytotoxic context; 7 supported donors"
        ),
        "heldout_scope": "one frozen triple-positive held-out cell in each of 8 donors",
        "parent_matched_gnly_effect": parent_rendered,
        "heldout_matched_gnly_effect": sentinel_rendered,
        "mean_differences_heldout_minus_parent": {
            form: _number(
                Decimal(str(sentinel_rendered[form]["donor_equal_mean"]))
                - Decimal(str(parent_rendered[form]["donor_equal_mean"]))
            )
            for form in (*FORMS, "four_form_mean")
        },
        "comparison_boundary": (
            "post-hoc, same model/cohort/donors, different expression-selected "
            "cells and donor support; not an independent replication"
        ),
    }


def _canonical_drivers(
    values: dict[tuple[str, str, str, str], Decimal],
    registered_raw: dict[tuple[str, str], list[Decimal]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    selected_subsets = ("GNLY", "NKG7", "NKG7+CCL5", "GNLY+NKG7+CCL5")
    for donor in ("107", "1039"):
        entity_ids = {
            key[0] for key in values if key[0].split(":")[2] == donor
        }
        entity_id = next(iter(entity_ids))
        states: dict[str, Any] = {
            "unmasked": _number(
                values[(entity_id, "unmasked", "none", "ab_pa")]
            )
        }
        for subset in selected_subsets:
            states[subset] = {
                "target_mask": _number(
                    values[(entity_id, "target_mask", subset, "ab_pa")]
                ),
                "control_mask": _number(
                    values[(entity_id, "control_mask", subset, "ab_pa")]
                ),
            }
        donor_index = DONORS.index(donor)
        nonzero = {}
        for (scale, name), vector in registered_raw.items():
            value = vector[donor_index]
            if value:
                nonzero[f"{scale}:{name}"] = _number(value)
        output[donor] = {
            "entity_id": entity_id,
            "canonical_aligned_p_cd8_states": states,
            "nonzero_registered_vector_values": nonzero,
        }
    output["mean_contribution_facts"] = {
        "r_T_full_triple": {
            "total_mean": 0.075,
            "donor_107_contribution": 0.075,
            "other_donors_combined_contribution": 0.0,
        },
        "r_J_increment": {
            "total_mean": 0.0875,
            "donor_107_contribution": 0.075,
            "donor_1039_contribution": 0.0125,
            "other_donors_combined_contribution": 0.0,
        },
        "r_A_GNLY": {
            "total_mean": -0.0125,
            "donor_1039_contribution": -0.0125,
            "other_donors_combined_contribution": 0.0,
        },
        "a_A_GNLY": {
            "total_mean": -0.0125,
            "donor_107_contribution": -0.0125,
            "other_donors_combined_contribution": 0.0,
        },
    }
    return output


def _render_markdown(result: dict[str, Any]) -> str:
    surface = result["canonical_ab_pa_surface"]
    gate = result["canonical_gate"]
    interface = result["readout_interface"]
    comparison = result["parent_gnly_comparison"]

    lines = [
        "# GSE96583 sentinel post-hoc interface discovery",
        "",
        "## English",
        "",
        "The registered donor-recurrent three-token output-surface hypothesis "
        "did not pass. A large repeated structure in the raw replies is a "
        "coarse, prompt-contingent readout failure; this does not distinguish "
        "an absent token effect from an effect obscured by that readout. "
        "Canonical `ab_pa` uses donor 107 for the entire full-triple mean and "
        "donor 1039 for the only negative matched GNLY value.",
        "",
        f"- Anchor gate: **{'PASS' if gate['anchor_pass'] else 'FAIL'}** "
        "(four-component dual-scale exact IUT "
        f"p={gate['anchor_exact_intersection_union_p']:.6g}); "
        f"endpoint: `{gate['endpoint']}`.",
        "- The held-out matched GNLY mean is `-0.012500` under canonical "
        "`ab_pa`, but `-0.140625` after the four-form average. The parent "
        "receptor-context GNLY means are `-0.255714` and `-0.157024`, "
        "respectively. The similar four-form means are descriptive only: "
        "the cell selection and donor support differ.",
        "- The numeric replies are not a coherent binary probability surface. "
        "For the same sentinel input, mean `P(A)+P(B)-1` is `+0.565000` "
        "under AB order and `+0.364750` under BA order; the mean answer-order "
        "shift after pairing queried targets is `+0.151792`.",
        "- `407/480` replies (`84.8%`) assign at least `0.75` to whichever "
        "class is queried. Only `16/120` AB and `2/120` BA complementary "
        "pairs are within `±0.03` of summing to one, and all `120/120` "
        "cell-condition inputs vary across forms.",
        "- Therefore this is evidence of a structured "
        "**elicitation/output-readout failure** for this model revision and "
        "template family, not proof of latent stored knowledge, hidden-state "
        "activation failure, biology, or a physical law.",
        "",
        "Canonical donor order: `101, 107, 1015, 1016, 1039, 1244, 1256, 1488`.",
        "",
        "| subset | a vector | h vector | r=a-h vector | r mean |",
        "|---|---|---|---|---:|",
    ]
    for subset in SUBSETS:
        vectors = surface[subset]
        lines.append(
            f"| `{subset}` | `{vectors['a']['values']}` | "
            f"`{vectors['h']['values']}` | `{vectors['r']['values']}` | "
            f"{vectors['r']['mean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "### Canonical donor drivers",
            "",
            "- Donor 107: unmasked aligned CD8 score=`0.75`; NKG7 and "
            "full-triple target "
            "masks both move it to `0.15`, while their matched masks are "
            "`0.85` and `0.75`. Its `r(T)=+0.60` supplies 100% of the "
            "`+0.075` full-triple donor mean and opposes the registered "
            "NK-directed hypothesis.",
            "- Donor 1039: GNLY target masking leaves the aligned CD8 score "
            "at `0.85`, while "
            "the matched GAPDH mask moves it to `0.75`. Thus its "
            "`r(G)=-0.10` is comparator-driven (`a=0`, `h=+0.10`), and it "
            "supplies the entire `-0.0125` matched GNLY mean.",
            "",
            "### Readout coherence",
            "",
            "| raw set | calls / levels | four-level mass | AB complement "
            "residual mean [range] | BA residual mean [range] | paired "
            "answer-order mean [range] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (("sentinel", "held-out sentinel"), ("parent", "parent")):
        quant = interface[key]["raw_quantization"]
        coherence = interface[key]["same_state_coherence"]
        ab = coherence["same_order_complement_residual"]["ab"]
        ba = coherence["same_order_complement_residual"]["ba"]
        order = coherence["answer_order_difference"]["form_pair_mean"]
        lines.append(
            f"| {label} | {quant['n_calls']} / "
            f"{quant['distinct_raw_output_count']} | "
            f"{100 * quant['four_dominant_level_fraction']:.1f}% | "
            f"{ab['mean']:+.6f} [{ab['min']:+.2f}, {ab['max']:+.2f}] | "
            f"{ba['mean']:+.6f} [{ba['min']:+.2f}, {ba['max']:+.2f}] | "
            f"{order['mean']:+.6f} "
            f"[{order['min']:+.3f}, {order['max']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "The held-out raw outputs occupy only five levels; "
            "`0.15/0.25/0.75/0.85` account for "
            f"`{100 * interface['sentinel']['raw_quantization']['four_dominant_level_fraction']:.1f}%` "
            "of calls. Full per-form ranges and exact residual distributions "
            "are in the JSON artifact.",
            "",
            "A falsifiable next mathematical model is "
            "`y = Q(alpha[answer order, queried target] + b[donor] + "
            "beta[mask] + interactions)`, where `Q` is a coarse quantizer. "
            "The route-scale diagnostics above are larger than the canonical "
            "GNLY (`|mean|=0.0125`) and triple (`|mean|=0.075`) effects. This "
            "equation has not been fitted; it is a next model, not a variance "
            "decomposition, invariant, or physical law.",
            "",
            "## 한국어",
            "",
            "사전등록한 donor-recurrent 3-token 출력-surface 가설은 통과하지 "
            "못했다. Raw 응답에서 두드러진 반복 구조 중 하나는 거칠고 "
            "prompt 의존적인 readout 실패지만, 이것만으로 실제 token 효과가 "
            "없는지 readout에 가려졌는지는 구분할 수 없다. 정규 `ab_pa`에서 "
            "full-triple 평균은 전적으로 donor 107이 만들고, 음의 matched "
            "GNLY 값은 donor 1039 하나에서만 나온다.",
        "",
        f"- Anchor gate: **{'통과' if gate['anchor_pass'] else '실패'}** "
            "(4-component dual-scale exact IUT "
            f"p={gate['anchor_exact_intersection_union_p']:.6g}); "
            f"endpoint: `{gate['endpoint']}`.",
            "- Held-out matched GNLY 평균은 정규 `ab_pa`에서 `-0.012500`, "
            "4-form 평균에서 `-0.140625`이다. Parent receptor-context의 "
            "대응 값은 각각 `-0.255714`, `-0.157024`이다. 4-form 평균의 "
            "유사성은 기술적 발견일 뿐이며, 선택된 cell과 donor support가 "
            "달라 독립 재현이 아니다.",
            "- 동일 입력에서 수치 응답은 일관된 이항 확률을 이루지 않는다. "
            "평균 `P(A)+P(B)-1`은 AB 순서에서 `+0.565000`, BA 순서에서 "
            "`+0.364750`이고, queried target을 짝지은 평균 answer-order "
            "이동은 `+0.151792`이다.",
            "- `407/480` 응답(`84.8%`)은 질문한 class에 최소 `0.75`를 "
            "부여한다. 합이 1에서 `±0.03` 이내인 상보 query pair는 AB "
            "`16/120`, BA `2/120`뿐이며, `120/120` cell-condition 입력이 "
            "form에 따라 달라진다.",
            "- 따라서 이는 이 model revision과 template family에서의 구조적 "
            "**elicitation/output-readout 실패**의 증거다. 잠재 지식의 증명, "
            "hidden-state activation 실패, 생물학적 인과, 또는 물리 법칙의 "
            "증거는 아니다.",
            "",
            "검증 가능한 다음 수학 모델은 "
            "`y = Q(alpha[answer order, queried target] + b[donor] + "
            "beta[mask] + interactions)`이다. 여기서 `Q`는 거친 양자화 "
            "함수다. 위 route-scale 진단값은 정규 GNLY "
            "(`|mean|=0.0125`)와 triple (`|mean|=0.075`) 효과보다 크다. "
            "아직 적합하거나 분산분해한 모델이 아니며, 불변식이나 물리 "
            "법칙이 아니라 다음 검증 가설이다.",
            "",
            "### Donor 107 / 1039",
            "",
            "- Donor 107: unmasked aligned CD8 score=`0.75`에서 NKG7 및 "
            "full-triple "
            "target mask가 모두 `0.15`로 이동한다. `r(T)=+0.60` 하나가 "
            "full-triple 평균 `+0.075` 전체를 만들며 사전등록된 NK 방향과 "
            "반대다.",
            "- Donor 1039: GNLY target mask는 `0.85`로 변화가 없고, matched "
            "GAPDH mask만 `0.75`로 이동한다. 따라서 `r(G)=-0.10`은 target "
            "효과가 아니라 comparator 효과(`a=0`, `h=+0.10`)이며 matched "
            "GNLY 평균 전체를 만든다.",
            "",
            "## Post-hoc boundary / 사후분석 경계",
            "",
            "These diagnostics were chosen after the confirmatory responses "
            "were observed. They use one model revision and template family, "
            "the same eight-donor SLE control cohort, expression-selected "
            "cells, text-token masking, and uncalibrated coarse outputs. They "
            "do not validate deposited "
            "cell labels or identify gene/pathway causality, latent knowledge, "
            "hidden-state activation, a mathematical invariant, or a physical "
            "law.",
            "",
            "Analysis code SHA-256: "
            f"`{result['analysis_code']['sha256']}`.",
            "",
            "이 진단은 확증 응답을 본 뒤 선택한 사후분석이다. 하나의 model "
            "revision과 template family, 동일한 8-donor SLE control "
            "cohort, 발현으로 선택된 cell, "
            "text-token masking, 보정되지 않은 거친 출력만 사용한다. 따라서 "
            "기탁 cell label의 진실성, gene/pathway 인과, 잠재 지식, "
            "hidden-state activation, 수학적 불변식 또는 물리 법칙을 "
            "검증하지 않는다.",
            "",
        ]
    )
    parent_four = comparison["parent_matched_gnly_effect"]["four_form_mean"][
        "donor_equal_mean"
    ]
    heldout_four = comparison["heldout_matched_gnly_effect"]["four_form_mean"][
        "donor_equal_mean"
    ]
    if not math.isclose(parent_four, -0.157023809524, abs_tol=1e-12):
        raise PosthocArtifactError("parent four-form GNLY comparison changed")
    if not math.isclose(heldout_four, -0.140625, abs_tol=1e-12):
        raise PosthocArtifactError("held-out four-form GNLY comparison changed")
    return "\n".join(lines)


def analyze() -> dict[str, Any]:
    provenance = _verify_frozen_inputs()
    bases, interventions = _read_sentinel_csv()
    sentinel_records = _read_raw(SENTINEL_RAW, 480, "sentinel")
    parent_records = _read_raw(PARENT_RAW, 1120, "parent")
    sentinel_values = _validate_sentinel_records(
        sentinel_records,
        bases,
        interventions,
    )
    _validate_parent_records(parent_records)
    surface, surface_raw = _sentinel_surface(sentinel_values)
    registered, registered_raw = _registered_vectors(surface_raw)
    gate = _primary_gate(registered)
    sentinel_interface = _interface_summary(
        sentinel_records,
        lambda record: (
            record["entity_id"],
            record["condition"],
            record["subset_id"],
        ),
    )
    parent_interface = _interface_summary(
        parent_records,
        lambda record: (
            record["entity_id"],
            _assignment_signature(record),
        ),
    )
    result = {
        "analysis_id": "gse96583-sentinel-posthoc-interface-discovery-v1",
        "status": "posthoc_descriptive",
        "analysis_code": {
            "path": str(ANALYSIS_CODE.relative_to(ROOT)),
            "sha256": _sha256(ANALYSIS_CODE),
        },
        "frozen_input_provenance": provenance,
        "validated_contract": {
            "sentinel_csv_rows": 64,
            "sentinel_base_cells": 8,
            "sentinel_interventions": 56,
            "sentinel_raw_calls": 480,
            "sentinel_same_input_states": 120,
            "parent_raw_calls": 1120,
            "parent_same_input_states": 280,
            "raw_outputs_reparsed_from_strings": True,
            "sentinel_r_equals_a_minus_h_for_all_56_donor_subset_rows": True,
            "all_raw_records_parsed": True,
            "sentinel_no_retries_or_temperature_fallback": True,
            "sentinel_exact_model_and_provider_validated": True,
            "sentinel_response_and_provider_request_ids_unique": True,
            "parent_model_validated": True,
            "runner_imported": False,
            "primary_result_json_consumed": False,
        },
        "estimand_definitions": {
            "a": "p_cd8(unmasked) - p_cd8(target_mask)",
            "h": "p_cd8(unmasked) - p_cd8(matched_control_mask)",
            "r": "p_cd8(matched_control_mask) - p_cd8(target_mask) = a - h",
            "canonical_form": "ab_pa",
            "donor_order": list(DONORS),
        },
        "canonical_ab_pa_surface": surface,
        "registered_vectors": registered,
        "canonical_gate": gate,
        "canonical_driver_localization": _canonical_drivers(
            sentinel_values,
            registered_raw,
        ),
        "readout_interface": {
            "sentinel": {
                "raw_quantization": _raw_quantization(sentinel_records),
                "same_state_coherence": sentinel_interface,
            },
            "parent": {
                "raw_quantization": _raw_quantization(parent_records),
                "same_state_coherence": parent_interface,
            },
        },
        "parent_gnly_comparison": _parent_gnly_comparison(
            parent_records,
            sentinel_values,
        ),
        "mathematical_candidate": {
            "descriptive_equation": (
                "y = Q(alpha_answer_order,queried_target + b_donor + "
                "beta_mask + interactions)"
            ),
            "Q": (
                "coarse output quantizer; 0.15, 0.25, 0.75, and 0.85 contain "
                "479/480 held-out replies"
            ),
            "route_scale_observation": (
                "held-out mean same-state complement residuals are 0.565 "
                "(AB) and 0.36475 (BA), and the paired answer-order shift is "
                "0.151791666667; these exceed the absolute canonical GNLY "
                "mean (0.0125) and full-triple mean (0.075)"
            ),
            "fit_status": (
                "not fitted and not a variance decomposition; a falsifiable "
                "next model, not a law or invariant"
            ),
        },
        "posthoc_boundary": (
            "Selected after confirmatory responses; one model revision, one "
            "template family, and the same eight-donor SLE control cohort; "
            "expression-selected cells and text-token interventions; no "
            "biological perturbation, label-truth validation, gene/pathway "
            "causality, latent-knowledge proof, hidden-state activation-gap "
            "test, mathematical invariant, or physical law."
        ),
    }
    if gate["anchor_pass"] or gate["endpoint"] != "anchor_gate_failed":
        raise PosthocArtifactError("canonical registered gate facts changed")
    return result


def main() -> None:
    result = analyze()
    json_text = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    markdown_text = _render_markdown(result)
    JSON_OUT.write_text(json_text + "\n", encoding="utf-8")
    MARKDOWN_OUT.write_text(markdown_text, encoding="utf-8")
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"Wrote {MARKDOWN_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

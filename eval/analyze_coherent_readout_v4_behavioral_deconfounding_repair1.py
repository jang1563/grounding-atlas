"""Apply the disclosed V4 analysis-adapter repair without changing frozen code.

``plan`` freezes this repair, its tests and memo, and every original input hash
without inspecting aggregate outcomes. ``analyze`` validates that lock, calls
the untouched runner validator, suppresses only the fixture-cell schema tag in
an in-memory merge view, and delegates raw replay and all decisions to the
untouched frozen analyzer. No phase loads model weights or executes a forward.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from . import analyze_coherent_readout_v4_behavioral_deconfounding as frozen_analyzer
    from . import run_coherent_readout_v4_behavioral_deconfounding as frozen_runner
except ImportError:  # direct execution from eval/
    import analyze_coherent_readout_v4_behavioral_deconfounding as frozen_analyzer
    import run_coherent_readout_v4_behavioral_deconfounding as frozen_runner


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = frozen_runner.RESULT_ROOT
REPAIR_ID = "coherent-readout-v4-analysis-adapter-repair1"
REPAIR_PLAN_SCHEMA = "coherent-readout-v4-analysis-adapter-repair-plan-v1"
REPAIR_MANIFEST_SCHEMA = "coherent-readout-v4-analysis-adapter-repair-manifest-v1"
REPAIR_DATE = "2026-08-03"
EXPECTED_ROWS = 448

REPAIR_MEMO = ROOT / "docs" / "COHERENT_READOUT_V4_ANALYSIS_ADAPTER_REPAIR1.md"
REPAIR_TEST = (
    ROOT / "tests" / "test_coherent_readout_behavioral_deconfounding_repair1.py"
)
DEFAULT_REPAIR_PLAN = RESULT_ROOT / "analysis_adapter_repair1_plan.json"
DEFAULT_ANALYSIS = RESULT_ROOT / "behavior_analysis.repair1.json"
DEFAULT_MARKDOWN = RESULT_ROOT / "analysis.repair1.md"
DEFAULT_MANIFEST = RESULT_ROOT / "analysis_manifest.repair1.json"

ORIGINAL_SOURCE_HASHES = {
    "preregistration": "af63bb4fabcff96486adf9715c5ade276a44fe5f07dc07add02cb452836e0bdb",
    "runner": "4773f48ec1fa6d7e9b6a685456e3729183d1ff66682f14c41954243f4d60d28a",
    "analyzer": "7b8b460714d146371d6a24a2459bb6a19c7fa979bb4a4ecd86ecec6bfc2b1175",
    "test_builder": "a56dd83388e9d38000f005ca036ec75efbd75bb4ee5d93f529c4070ee2eb7c54",
    "test_runner": "ec7541a1633e5086aeb841e1f1677d57a1338c615bcc0cd7c0d7e8363c239e55",
    "test_analyzer": "f20d303f1a6f8d6147af6d7ba15fe9d802232f80c7a0b5d62300dc7f1b3c5a3d",
}
ORIGINAL_ARTIFACT_HASHES = {
    "dependency_lock.json": "14e3cc08b61df3d5e05036897640e925c8f0ee54e7e767591dde486aa49c5bbb",
    "design.json": "1b6d841db45025452d6bc3f845ea3872718cd15e4d8aa339aac6b4056827de23",
    "plan_manifest.json": "fc0446884ff51c9fb2b9fc4d1f99490adbf99faaf3637b6463ebdc82b73ffb1a",
    "tokenization_receipt.json": "a7c600a7af28366f2dc369b60adcf2277b86bf0121e862306bfb527af8f84543",
    "behavior_attempt.json": "56e06041ae81b88b7dfda1cb928fdeef15f117b0124a68b147961a1bf4f59b53",
    "behavior_records.jsonl": "a96554156f91f7b6721cd5abdb21f3694f30ab6efd3f55f0aec4de840e14d041",
    "behavior_full_vocab_logits.npy": "4f281510fb70f5f678eee3680999193b559970cf823e67d7c03f6b50bde80928",
    "behavior_execution_manifest.json": "3ebf2262111376d0870c0189016d62eed83e46ef9b231cf12f3edc28433c4132",
}
LOGICAL_F32_SHA256 = (
    "3beb4b6e09ef91d23c94e961008a99044b49798d5b9747d9750192761bba9a23"
)
FROZEN_EXCEPTION = "record differs from frozen cell field: schema_version"
EXPECTED_SHARED_FIELDS = frozenset(
    {
        "schema_version",
        "cell_id",
        "world_id",
        "family_id",
        "stratum_id",
        "target_property",
        "mapping_id",
        "target_fact_order",
        "rule_order",
        "option_order",
        "answer_labels",
        "correct_answer",
        "correct_option_position",
        "displayed_options",
        "rule_lines",
        "prompt_lines",
        "prompt_text",
        "biological_model_calls",
    }
)

SOURCE_PATHS = {
    "preregistration": frozen_runner.FROZEN_PREREGISTRATION,
    "runner": Path(frozen_runner.__file__).resolve(),
    "analyzer": Path(frozen_analyzer.__file__).resolve(),
    "test_builder": ROOT
    / "tests"
    / "test_build_coherent_readout_v4_behavioral_deconfounding_bank.py",
    "test_runner": ROOT
    / "tests"
    / "test_run_coherent_readout_v4_behavioral_deconfounding.py",
    "test_analyzer": ROOT
    / "tests"
    / "test_analyze_coherent_readout_v4_behavioral_deconfounding.py",
}


class AnalysisAdapterRepairError(ValueError):
    """Raised when the repair contract cannot be applied exactly."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisAdapterRepairError(f"cannot load JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisAdapterRepairError(f"JSON artifact is not an object: {path}")
    return value


def _source_bindings() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, path in SOURCE_PATHS.items():
        expected = ORIGINAL_SOURCE_HASHES[name]
        if not path.is_file() or file_sha256(path) != expected:
            raise AnalysisAdapterRepairError(f"frozen source hash changed: {name}")
        bindings[name] = {"path": str(path), "file_sha256": expected}
    return bindings


def _artifact_bindings(result_root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for filename, expected in ORIGINAL_ARTIFACT_HASHES.items():
        path = result_root / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise AnalysisAdapterRepairError(f"frozen artifact hash changed: {filename}")
        bindings[filename] = {"path": str(path), "file_sha256": expected}
    execution = _load_json(result_root / "behavior_execution_manifest.json")
    full_vocab = execution.get("full_vocab_logits")
    if not isinstance(full_vocab, Mapping):
        raise AnalysisAdapterRepairError("execution manifest omitted full-vocabulary lock")
    if (
        full_vocab.get("logical_f32_sha256") != LOGICAL_F32_SHA256
        or full_vocab.get("shape") != [EXPECTED_ROWS, frozen_runner.MODEL_VOCAB_SIZE]
        or full_vocab.get("dtype") != "float32"
    ):
        raise AnalysisAdapterRepairError("logical full-vocabulary lock changed")
    return bindings


def _repair_implementation_bindings() -> dict[str, dict[str, str]]:
    paths = {
        "repair_adapter": Path(__file__).resolve(),
        "repair_test": REPAIR_TEST,
        "repair_memo": REPAIR_MEMO,
    }
    bindings: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise AnalysisAdapterRepairError(f"repair implementation missing: {name}")
        bindings[name] = {"path": str(path), "file_sha256": file_sha256(path)}
    return bindings


def repair_plan_payload(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root).resolve()
    core = {
        "schema_version": REPAIR_PLAN_SCHEMA,
        "repair_id": REPAIR_ID,
        "repair_date": REPAIR_DATE,
        "status": "V4_ANALYSIS_ADAPTER_REPAIR1_PLAN_FROZEN_ZERO_FORWARD",
        "result_root": str(result_root),
        "frozen_exception": FROZEN_EXCEPTION,
        "record_schema": frozen_runner.RECORD_SCHEMA,
        "fixture_cell_schema": frozen_runner.FIXTURE_CELL_SCHEMA,
        "expected_rows": EXPECTED_ROWS,
        "expected_shared_fields": sorted(EXPECTED_SHARED_FIELDS),
        "sole_suppressed_merge_key": "schema_version",
        "transformation_scope": "in_memory_fixture_cell_merge_view_only",
        "original_sources": _source_bindings(),
        "original_artifacts": _artifact_bindings(result_root),
        "repair_implementation": _repair_implementation_bindings(),
        "raw_artifacts_modified": False,
        "decision_rules_changed": False,
        "aggregate_metrics_inspected_by_plan": False,
        "outcome_exposure_before_repair_lock_disclosed": True,
        "model_forwards_by_plan": 0,
        "biological_model_calls": 0,
        "activation_authorized": False,
    }
    return {**core, "canonical_sha256": canonical_sha256(core)}


def _atomic_frozen_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise AnalysisAdapterRepairError(f"refusing to overwrite artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise AnalysisAdapterRepairError(f"stale atomic temporary exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_frozen_bundle(payloads: Mapping[Path, bytes]) -> None:
    for path, payload in payloads.items():
        if path.exists() and path.read_bytes() != payload:
            raise AnalysisAdapterRepairError(f"refusing to overwrite artifact: {path}")
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists():
            raise AnalysisAdapterRepairError(f"stale atomic temporary exists: {temporary}")
    for path, payload in payloads.items():
        _atomic_frozen_write(path, payload)


def run_plan(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root).resolve()
    downstream = [
        result_root / DEFAULT_ANALYSIS.name,
        result_root / DEFAULT_MARKDOWN.name,
        result_root / DEFAULT_MANIFEST.name,
    ]
    if any(path.exists() for path in downstream):
        raise AnalysisAdapterRepairError("repair analysis artifacts already exist")
    plan = repair_plan_payload(result_root)
    path = result_root / DEFAULT_REPAIR_PLAN.name
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_frozen_write(path, payload)
    return {
        "status": plan["status"],
        "repair_plan_canonical_sha256": plan["canonical_sha256"],
        "model_forwards": 0,
        "biological_model_calls": 0,
    }


def load_and_validate_repair_plan(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root).resolve()
    path = result_root / DEFAULT_REPAIR_PLAN.name
    observed = _load_json(path)
    expected = repair_plan_payload(result_root)
    if observed != expected:
        raise AnalysisAdapterRepairError("repair plan differs from current frozen inputs")
    return observed


def adapt_schema_namespace(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suppress only the verified fixture-cell type tag in an in-memory copy."""

    records = bundle.get("records")
    plan = bundle.get("plan_manifest")
    if not isinstance(records, list) or len(records) != EXPECTED_ROWS:
        raise AnalysisAdapterRepairError("validated record registry changed")
    if not isinstance(plan, Mapping):
        raise AnalysisAdapterRepairError("validated plan manifest changed")
    cells = plan.get("cell_registry")
    if not isinstance(cells, list) or len(cells) != EXPECTED_ROWS:
        raise AnalysisAdapterRepairError("validated cell registry changed")
    if any(not isinstance(record, Mapping) for record in records) or any(
        not isinstance(cell, Mapping) for cell in cells
    ):
        raise AnalysisAdapterRepairError("record/cell registry contains a non-object")

    record_ids = [record.get("cell_id") for record in records]
    cell_ids = [cell.get("cell_id") for cell in cells]
    if (
        record_ids != cell_ids
        or len(set(record_ids)) != EXPECTED_ROWS
        or None in record_ids
    ):
        raise AnalysisAdapterRepairError("record/cell IDs are not one-to-one and ordered")

    conflict_count = 0
    for index, (record, cell) in enumerate(zip(records, cells, strict=True)):
        if record.get("schema_version") != frozen_runner.RECORD_SCHEMA:
            raise AnalysisAdapterRepairError(f"record schema changed at row {index}")
        if cell.get("schema_version") != frozen_runner.FIXTURE_CELL_SCHEMA:
            raise AnalysisAdapterRepairError(f"fixture-cell schema changed at row {index}")
        shared = set(record).intersection(cell)
        if shared != EXPECTED_SHARED_FIELDS:
            raise AnalysisAdapterRepairError(f"shared-field registry changed at row {index}")
        for field in EXPECTED_SHARED_FIELDS - {"schema_version"}:
            if record[field] != cell[field]:
                raise AnalysisAdapterRepairError(
                    f"non-schema record/cell disagreement at row {index}: {field}"
                )
        if record["schema_version"] == cell["schema_version"]:
            raise AnalysisAdapterRepairError(f"expected schema distinction vanished at row {index}")
        conflict_count += 1

    adapted_plan = copy.deepcopy(dict(plan))
    adapted_cells = adapted_plan["cell_registry"]
    for cell in adapted_cells:
        removed = cell.pop("schema_version", None)
        if removed != frozen_runner.FIXTURE_CELL_SCHEMA:
            raise AnalysisAdapterRepairError("in-memory schema suppression changed")
    adapted_bundle = dict(bundle)
    adapted_bundle["plan_manifest"] = adapted_plan
    audit = {
        "record_count": len(records),
        "cell_count": len(cells),
        "record_schema": frozen_runner.RECORD_SCHEMA,
        "fixture_cell_schema": frozen_runner.FIXTURE_CELL_SCHEMA,
        "shared_field_count": len(EXPECTED_SHARED_FIELDS),
        "equal_shared_field_count": len(EXPECTED_SHARED_FIELDS) - 1,
        "schema_conflict_count": conflict_count,
        "sole_suppressed_merge_key": "schema_version",
        "disk_objects_mutated": False,
        "adapted_cell_registry_canonical_sha256": canonical_sha256(adapted_cells),
        "pass": True,
    }
    return adapted_bundle, audit


def _analysis_provenance(
    result_root: Path,
    repair_plan: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "result_root": str(result_root),
        "repair_id": REPAIR_ID,
        "repair_qualifier": "disclosed_post_freeze_implementation_repair",
        "repair_plan_canonical_sha256": repair_plan["canonical_sha256"],
        "frozen_exception": FROZEN_EXCEPTION,
        "adapter_audit": dict(audit),
        "original_runner_file_sha256": ORIGINAL_SOURCE_HASHES["runner"],
        "original_analyzer_file_sha256": ORIGINAL_SOURCE_HASHES["analyzer"],
        "original_execution_manifest_file_sha256": ORIGINAL_ARTIFACT_HASHES[
            "behavior_execution_manifest.json"
        ],
        "raw_full_vocab_file_sha256": ORIGINAL_ARTIFACT_HASHES[
            "behavior_full_vocab_logits.npy"
        ],
        "raw_full_vocab_logical_f32_sha256": LOGICAL_F32_SHA256,
        "outcome_exposure_before_repair_lock_disclosed": True,
        "raw_artifacts_modified": False,
        "decision_rules_changed": False,
        "model_forwards_executed_by_repair": 0,
        "biological_model_calls": 0,
    }


def _render_markdown(analysis: Mapping[str, Any]) -> str:
    gates = analysis["component_gates"]
    accuracy = analysis["accuracy"]
    policy = analysis["composition_policy_comparison"]

    def rate(family: str) -> float:
        return float(
            accuracy[family]["overall_confirmatory_full_vocab_accuracy"]["rate"]
        )

    return "\n".join(
        [
            "# Coherent Readout V4 analysis — repair1",
            "",
            "## English",
            "",
            f"Terminal status: `{analysis['status']}`.",
            "",
            "Qualifier: disclosed post-freeze implementation repair. The frozen "
            "analyzer did not complete unchanged, and a provisional outcome was exposed "
            "during diagnosis before the repair lock.",
            "",
            f"- Output engineering: **{'pass' if gates['engineering'] else 'fail'}**",
            f"- Property retrieval: **{'pass' if gates['property_retrieval'] else 'fail'}** "
            f"(accuracy {rate('property_retrieval'):.6f})",
            f"- Codebook lookup: **{'pass' if gates['codebook_lookup'] else 'fail'}** "
            f"(accuracy {rate('codebook_lookup'):.6f})",
            f"- Composition: **{'pass' if gates['composition'] else 'fail'}** "
            f"(accuracy {rate('composition'):.6f})",
            "- Registered failure-policy classification: "
            f"`{policy['dominant_failure_classification']}`",
            "",
            "No raw artifact or decision rule changed, and repair1 executed zero model "
            "forwards. This behavior-only result does not establish a causal mechanism, "
            "latent knowledge, an activation gap, biology, a physical law, or model-family "
            "generality, and it authorizes no activation experiment.",
            "",
            "## 한국어",
            "",
            f"최종 상태: `{analysis['status']}`.",
            "",
            "한정어: 공개된 post-freeze implementation repair. 원 frozen analyzer가 "
            "그대로 완주한 결과가 아니며, repair lock 전에 진단용 잠정 결과가 한 번 "
            "노출되었다.",
            "",
            f"- 출력 engineering: **{'통과' if gates['engineering'] else '실패'}**",
            f"- 속성 retrieval: **{'통과' if gates['property_retrieval'] else '실패'}** "
            f"(정확도 {rate('property_retrieval'):.6f})",
            f"- codebook lookup: **{'통과' if gates['codebook_lookup'] else '실패'}** "
            f"(정확도 {rate('codebook_lookup'):.6f})",
            f"- composition: **{'통과' if gates['composition'] else '실패'}** "
            f"(정확도 {rate('composition'):.6f})",
            "- 등록 실패-policy 분류: "
            f"`{policy['dominant_failure_classification']}`",
            "",
            "Raw artifact와 decision rule은 변하지 않았고 repair1 model forward는 "
            "0회다. 이 행동 결과는 인과 메커니즘, 잠재지식, activation gap, 생물학, "
            "물리 법칙, 모델 계열 일반화를 증명하지 않으며 activation 실험을 승인하지 "
            "않는다.",
            "",
        ]
    )


def _write_outputs(
    result_root: Path,
    analysis: Mapping[str, Any],
    repair_plan: Mapping[str, Any],
) -> None:
    analysis_path = result_root / DEFAULT_ANALYSIS.name
    markdown_path = result_root / DEFAULT_MARKDOWN.name
    manifest_path = result_root / DEFAULT_MANIFEST.name
    plan_path = result_root / DEFAULT_REPAIR_PLAN.name
    analysis_bytes = (json.dumps(analysis, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    markdown_bytes = _render_markdown(analysis).encode("utf-8")
    manifest = {
        "schema_version": REPAIR_MANIFEST_SCHEMA,
        "repair_id": REPAIR_ID,
        "status": analysis["status"],
        "repair_qualifier": "disclosed_post_freeze_implementation_repair",
        "original_frozen_analyzer_completed_unchanged": False,
        "outcome_exposure_before_repair_lock_disclosed": True,
        "repair_plan": {
            "path": str(plan_path),
            "file_sha256": file_sha256(plan_path),
            "canonical_sha256": repair_plan["canonical_sha256"],
        },
        "analysis": {
            "path": str(analysis_path),
            "file_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
            "canonical_sha256": canonical_sha256(analysis),
        },
        "markdown": {
            "path": str(markdown_path),
            "file_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
        },
        "original_sources": repair_plan["original_sources"],
        "original_artifacts": repair_plan["original_artifacts"],
        "repair_implementation": repair_plan["repair_implementation"],
        "raw_full_vocab_logical_f32_sha256": LOGICAL_F32_SHA256,
        "model_forwards_executed_by_repair": 0,
        "biological_model_calls": 0,
        "raw_artifacts_modified": False,
        "decision_rules_changed": False,
        "activation_authorized": False,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_frozen_bundle(
        {
            analysis_path: analysis_bytes,
            markdown_path: markdown_bytes,
            manifest_path: manifest_bytes,
        }
    )


def run_analysis(result_root: Path = RESULT_ROOT) -> dict[str, Any]:
    result_root = Path(result_root).resolve()
    repair_plan = load_and_validate_repair_plan(result_root)
    output_paths = [
        result_root / DEFAULT_ANALYSIS.name,
        result_root / DEFAULT_MARKDOWN.name,
        result_root / DEFAULT_MANIFEST.name,
    ]
    if any(path.exists() for path in output_paths):
        raise AnalysisAdapterRepairError("repair analysis re-entry is forbidden")
    try:
        bundle = frozen_runner.validate_behavior_artifacts(result_root=result_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise AnalysisAdapterRepairError(str(error)) from error
    adapted_bundle, audit = adapt_schema_namespace(bundle)
    replayed_rows = frozen_analyzer._bundle_records(adapted_bundle)
    bindings = _artifact_bindings(result_root)
    analysis = frozen_analyzer.analyze_records(
        replayed_rows,
        artifact_validation={
            "pass": True,
            "validator": (
                "untouched frozen runner.validate_behavior_artifacts + repair1 "
                "schema audit + untouched frozen analyzer raw-sidecar replay"
            ),
            "input_artifacts": bindings,
            "repair_adapter_audit": audit,
        },
        provenance=_analysis_provenance(result_root, repair_plan, audit),
    )
    analysis = dict(analysis)
    analysis["claim_authority"] = (
        "terminal_behavioral_component_localization_only_with_disclosed_"
        "post_freeze_implementation_repair"
    )
    analysis["implementation_repair"] = {
        "repair_id": REPAIR_ID,
        "qualifier": "disclosed_post_freeze_implementation_repair",
        "original_frozen_analyzer_completed_unchanged": False,
        "outcome_exposure_before_repair_lock_disclosed": True,
        "frozen_exception": FROZEN_EXCEPTION,
        "adapter_audit": audit,
        "raw_artifacts_modified": False,
        "decision_rules_changed": False,
        "model_forwards_executed_by_repair": 0,
    }
    _write_outputs(result_root, analysis, repair_plan)
    return {
        "status": analysis["status"],
        "repair_qualifier": analysis["implementation_repair"]["qualifier"],
        "model_forwards": 0,
        "biological_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "analyze"), required=True)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    arguments = parser.parse_args()
    try:
        if arguments.phase == "plan":
            result = run_plan(arguments.result_root)
        else:
            result = run_analysis(arguments.result_root)
    except AnalysisAdapterRepairError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

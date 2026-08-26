from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from eval import analyze_coherent_readout_v6a_r2_topology as analyzer
from eval import run_coherent_readout_v6a_r2_topology as runner


def _answer_pair(split: str, family: str, world_index: int, sign: int) -> tuple[str, str]:
    split_offset = 0 if split == "discovery" else 32
    family_offset = {
        analyzer.PROPERTY_RETRIEVAL: 0,
        analyzer.CODEBOOK_LOOKUP: 16,
        analyzer.SINGLE_TARGET: 16,
        analyzer.TWO_FACT: 16,
    }[family]
    base = 0x400 + split_offset + family_offset + 2 * world_index
    labels = (chr(base), chr(base + 1))
    return labels[0 if sign == -1 else 1], labels[1 if sign == -1 else 0]


def _diagnostics(
    margin: float = 1.0,
    *,
    expected_id: int = 1,
    distractor_id: int = 2,
    global_other: bool = False,
) -> dict:
    expected = margin
    distractor = 0.0
    if global_other:
        maximum_ids = [3]
    elif expected > distractor:
        maximum_ids = [expected_id]
    elif distractor > expected:
        maximum_ids = [distractor_id]
    else:
        maximum_ids = [expected_id, distractor_id]
    return {
        "expected_logit": expected,
        "distractor_logit": distractor,
        "expected_minus_distractor_margin": margin,
        "pairwise_correct": expected > distractor,
        "pairwise_tie": expected == distractor,
        "strict_unique_global_correct": maximum_ids == [expected_id],
        "maximum_token_ids": maximum_ids,
        "maximum_tie_count": len(maximum_ids),
        "full_vocab_logits_sha256": "0" * 64,
    }


def _record(
    call_index: int,
    *,
    split: str,
    family: str,
    world_index: int,
    factors: dict[str, int | None],
    execution_block: str,
    margin: float = 1.0,
) -> dict:
    if family == analyzer.PROPERTY_RETRIEVAL:
        answer_sign = int(factors["p"])
    else:
        answer_sign = -int(factors["p"]) * int(factors["m"])
    expected, distractor = _answer_pair(split, family, world_index, answer_sign)
    return {
        "record_id": f"record-{call_index:04d}",
        "call_index": call_index,
        "cell_id": f"cell-{call_index:04d}",
        "prompt_id": f"prompt-{call_index:04d}",
        "split": split,
        "role": split,
        "world_id": f"v6a_r2_{split}_world_{world_index + 1:03d}",
        "family": family,
        "execution_block": execution_block,
        "factors": factors,
        "expected_answer": expected,
        "distractor_answer": distractor,
        "expected_token_id": 1,
        "distractor_token_id": 2,
        "diagnostics": _diagnostics(margin),
    }


def _component_rows(split: str, start: int) -> list[dict]:
    rows: list[dict] = []
    index = start
    block = f"{split}-components"
    for world in range(8):
        g = -1 if world % 2 == 0 else 1
        for p, v, o, q, a in itertools.product((-1, 1), repeat=5):
            rows.append(
                _record(
                    index,
                    split=split,
                    family=analyzer.PROPERTY_RETRIEVAL,
                    world_index=world,
                    factors={
                        "p": p,
                        "m": None,
                        "r": None,
                        "v": v,
                        "o": o,
                        "q": q,
                        "a": a,
                        "u": None,
                        "w": None,
                    },
                    execution_block=block,
                )
            )
            index += 1
        for p, m, r, v in itertools.product((-1, 1), repeat=4):
            rows.append(
                _record(
                    index,
                    split=split,
                    family=analyzer.CODEBOOK_LOOKUP,
                    world_index=world,
                    factors={
                        "p": p,
                        "m": m,
                        "r": r,
                        "v": v,
                        "o": None,
                        "q": None,
                        "a": None,
                        "u": -p * r,
                        "w": p * m * v,
                    },
                    execution_block=block,
                )
            )
            index += 1
        for p, m, u, q, a in itertools.product((-1, 1), repeat=5):
            w = g * p * m * u
            r = -p * u
            v = p * m * w
            rows.append(
                _record(
                    index,
                    split=split,
                    family=analyzer.SINGLE_TARGET,
                    world_index=world,
                    factors={
                        "p": p,
                        "m": m,
                        "r": r,
                        "v": v,
                        "o": None,
                        "q": q,
                        "a": a,
                        "u": u,
                        "w": w,
                    },
                    execution_block=block,
                    margin=0.25,
                )
            )
            index += 1
    assert len(rows) == 640
    return rows


def _topology_rows(split: str, start: int) -> list[dict]:
    rows: list[dict] = []
    index = start
    for world in range(8):
        g = -1 if world % 2 == 0 else 1
        for p, m, u, o, q, a in itertools.product((-1, 1), repeat=6):
            w = g * p * m * u
            r = -p * u
            v = p * m * w
            if (q, a) == (-1, -1):
                margin = float(o)
            elif (q, a) == (1, 1):
                margin = float(-o)
            else:
                margin = 0.5
            rows.append(
                _record(
                    index,
                    split=split,
                    family=analyzer.TWO_FACT,
                    world_index=world,
                    factors={
                        "p": p,
                        "m": m,
                        "r": r,
                        "v": v,
                        "o": o,
                        "q": q,
                        "a": a,
                        "u": u,
                        "w": w,
                    },
                    execution_block=f"{split}-topology",
                    margin=margin,
                )
            )
            index += 1
    assert len(rows) == 512
    return rows


def _full_staged_rows() -> tuple[list[dict], list[dict]]:
    discovery_components = _component_rows("discovery", 0)
    discovery_topology = _topology_rows("discovery", 640)
    confirmation_components = _component_rows("confirmation", 1152)
    confirmation_topology = _topology_rows("confirmation", 1792)
    return discovery_components, [
        *discovery_topology,
        *confirmation_components,
        *confirmation_topology,
    ]


def _make_incorrect(row: dict, *, tie: bool = False, global_only: bool = False) -> None:
    if global_only:
        row["diagnostics"] = _diagnostics(1.0, global_other=True)
    else:
        row["diagnostics"] = _diagnostics(0.0 if tie else -1.0)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _binding(path: Path) -> dict:
    return {
        "path": str(path),
        "file_sha256": analyzer.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _add_planning_artifacts(paths: dict[str, Path], tmp_path: Path) -> None:
    paths.update(
        {
            "design": tmp_path / "design.json",
            "plan_manifest": tmp_path / "plan_manifest.json",
            "dependency_lock": tmp_path / "dependency_lock.json",
            "tokenization_receipt": tmp_path / "tokenization_receipt.json",
        }
    )
    for key in ("design", "plan_manifest", "dependency_lock", "tokenization_receipt"):
        _write_json(paths[key], {"artifact": key})


def _source_locks(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "plan_manifest_file_sha256": analyzer.file_sha256(paths["plan_manifest"]),
        "design_file_sha256": analyzer.file_sha256(paths["design"]),
        "dependency_lock_file_sha256": analyzer.file_sha256(paths["dependency_lock"]),
        "tokenization_receipt_file_sha256": analyzer.file_sha256(
            paths["tokenization_receipt"]
        ),
        "fixture_file_sha256": analyzer.file_sha256(runner.FIXTURE),
        "fixture_manifest_file_sha256": analyzer.file_sha256(runner.FIXTURE_MANIFEST),
        "runner_file_sha256": analyzer.file_sha256(Path(runner.__file__)),
        "analyzer_file_sha256": analyzer.file_sha256(Path(analyzer.__file__)),
        "design_document_file_sha256": analyzer.file_sha256(runner.DESIGN_DOCUMENT),
        "sealed_v2_runner_file_sha256": analyzer.file_sha256(runner.SEALED_V2_RUNNER),
    }


def _preflight(plan: dict, stage: str, start: int, stop: int) -> dict:
    specs = plan["raw_logits_shards"][stage]
    stage_bytes = sum(int(np.prod(spec["shape"])) * 4 for spec in specs)
    return {
        "execution_stage": stage,
        "planned_calls": stop - start,
        "global_start_row": start,
        "global_stop_row": stop,
        "stage_raw_logits_expected_bytes": stage_bytes,
        "total_raw_logits_expected_bytes": plan["disk_requirements"][
            "raw_logits_expected_bytes"
        ],
        "model_safetensor_bytes": 1,
        "mps_recommended_max_memory_bytes": 100,
        "mps_allocated_before_bytes": 0,
        "required_mps_headroom_bytes": 1,
        "mps_bfloat16_kernel_pass": True,
        "disk_free_bytes": 100,
        "required_disk_free_bytes": 1,
        "stored_logits_dtype": "float32_little_endian",
        "no_model_loaded": True,
        "no_model_forward_performed": True,
        "model_calls": 0,
        "generation_calls": 0,
        "pass": True,
    }


def _discovery_artifact_bundle(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    rows = _component_rows("discovery", 0)
    raw_root = tmp_path / "raw_logits" / analyzer.DISCOVERY_STAGE
    raw_root.mkdir(parents=True)
    stage_hash = "b" * 64
    call_hash = "a" * 64
    scientific_hash = "d" * 64
    prompts: list[dict] = []
    records: list[dict] = []
    raw_specs: list[dict] = []
    raw_bindings: list[dict] = []
    for shard_index in range(10):
        start = shard_index * 64
        stop = start + 64
        array = np.full((64, 4), -10.0, dtype="<f4")
        for local_index, source in enumerate(rows[start:stop]):
            diagnostics = source["diagnostics"]
            array[local_index, 1] = diagnostics["expected_logit"]
            array[local_index, 2] = diagnostics["distractor_logit"]
        path = raw_root / f"shard_{shard_index:03d}.npy"
        np.save(path, array, allow_pickle=False)
        spec = {
            "execution_stage": analyzer.DISCOVERY_STAGE,
            "index": shard_index,
            "phase_start_row": start,
            "phase_stop_row": stop,
            "global_start_row": start,
            "global_stop_row": stop,
            "rows": 64,
            "shape": [64, 4],
            "dtype": "<f4",
            "path": str(path),
        }
        raw_specs.append(spec)
        raw_bindings.append(
            {
                **spec,
                **_binding(path),
                "logical_sha256": analyzer._logical_array_sha256(array),
            }
        )

    for stage_index, source in enumerate(rows):
        shard_index, row_index = divmod(stage_index, 64)
        raw = np.load(raw_root / f"shard_{shard_index:03d}.npy", allow_pickle=False)[
            row_index
        ]
        prompt = {
            "global_call_index": stage_index,
            "block_call_index": stage_index,
            "stage_call_index": stage_index,
            "execution_stage": analyzer.DISCOVERY_STAGE,
            "execution_block": "discovery-components",
            "cell_id": source["cell_id"],
            "prompt_id": source["prompt_id"],
            "world_id": source["world_id"],
            "role": "discovery",
            "family": source["family"],
            "factors": source["factors"],
            "factor_levels": {},
            "semantic_answer": source["expected_answer"],
            "semantic_distractor": source["distractor_answer"],
            "expected_token_id": 1,
            "distractor_token_id": 2,
        }
        prompt["record_identity_id"] = runner.record_identity_id(prompt)
        prompts.append(prompt)
        diagnostics = analyzer.diagnostics_from_full_vocab(raw, 1, 2)
        core = {
            "schema_version": runner.RECORD_SCHEMA,
            "registration_state": runner.REGISTRATION_STATE,
            "execution_revision": runner.EXECUTION_REVISION,
            "record_identity_id": prompt["record_identity_id"],
            "execution_stage": analyzer.DISCOVERY_STAGE,
            "execution_block": "discovery-components",
            "global_call_index": stage_index,
            "block_call_index": stage_index,
            "stage_call_index": stage_index,
            "call_plan_sha256": call_hash,
            "stage_plan_sha256": stage_hash,
            "scientific_registry_sha256": scientific_hash,
            "cell_id": source["cell_id"],
            "prompt_id": source["prompt_id"],
            "world_id": source["world_id"],
            "role": "discovery",
            "family": source["family"],
            "factors": source["factors"],
            "factor_levels": {},
            "expected_token_id": 1,
            "distractor_token_id": 2,
            "raw_logits_shard_index": shard_index,
            "raw_logits_row_in_shard": row_index,
            "raw_logits_stage_row": stage_index,
            "raw_logits_global_row": stage_index,
            "raw_logits_row_sha256": analyzer.f32_sha256(raw),
            "diagnostics": diagnostics,
            "teacher_forced_prompt_forward": True,
            "generation_used": False,
            "model_calls": 1,
        }
        records.append({**core, "record_id": analyzer.canonical_sha256(core)})

    paths = {
        "discovery_attempt": tmp_path / "discovery_components_attempt.json",
        "discovery_records": tmp_path / "discovery_components_records.jsonl",
        "discovery_execution_manifest": tmp_path
        / "discovery_components_execution_manifest.json",
        "discovery_raw_root": raw_root,
        "remaining_attempt": tmp_path / "remaining_main_attempt.json",
        "remaining_records": tmp_path / "remaining_main_records.jsonl",
        "remaining_execution_manifest": tmp_path
        / "remaining_main_execution_manifest.json",
        "remaining_raw_root": tmp_path / "raw_logits" / analyzer.REMAINING_STAGE,
    }
    _add_planning_artifacts(paths, tmp_path)
    total_raw_bytes = sum(int(np.prod(spec["shape"])) * 4 for spec in raw_specs)
    plan = {
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "call_plan_sha256": call_hash,
        "scientific_registry_sha256": scientific_hash,
        "stage_plan_sha256": {analyzer.DISCOVERY_STAGE: stage_hash},
        "raw_logits_shards": {analyzer.DISCOVERY_STAGE: raw_specs},
        "disk_requirements": {"raw_logits_expected_bytes": total_raw_bytes},
        "model": {"vocab_size": 4},
        "prompts": prompts,
    }
    attempt = {
        "schema_version": runner.ATTEMPT_SCHEMA,
        "status": "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "execution_stage": analyzer.DISCOVERY_STAGE,
        "call_plan_sha256": call_hash,
        "stage_plan_sha256": stage_hash,
        "scientific_registry_sha256": scientific_hash,
        "expected_calls": 640,
        "global_start_row": 0,
        "global_stop_row": 640,
        "stage_start_row": 0,
        "stage_stop_row": 640,
        "block_order": ["discovery-components"],
        "block_counts": {"discovery-components": 640},
        "partial_resume_allowed": False,
        "generation_used": False,
        "model": plan["model"],
        "forward_contract": analyzer._expected_forward_contract(plan),
        "preflight": _preflight(plan, analyzer.DISCOVERY_STAGE, 0, 640),
        "source_locks": _source_locks(paths),
    }
    _write_json(paths["discovery_attempt"], attempt)
    paths["discovery_records"].write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": runner.EXECUTION_SCHEMA,
        "status": "V6A_R2_DISCOVERY_COMPONENTS_EXECUTION_COMPLETE_NOT_ANALYZED",
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "execution_stage": analyzer.DISCOVERY_STAGE,
        "call_plan_sha256": call_hash,
        "stage_plan_sha256": stage_hash,
        "scientific_registry_sha256": scientific_hash,
        "phase_model_calls": 640,
        "cumulative_model_calls": 640,
        "global_start_row": 0,
        "global_stop_row": 640,
        "block_order": ["discovery-components"],
        "block_counts": {"discovery-components": 640},
        "generation_used": False,
        "partial_resume_allowed": False,
        "attempt": _binding(paths["discovery_attempt"]),
        "records": {
            **_binding(paths["discovery_records"]),
            "count": 640,
            "canonical_sha256": analyzer.canonical_sha256(records),
        },
        "raw_logits_shards": raw_bindings,
        "source_locks": _source_locks(paths),
    }
    _write_json(paths["discovery_execution_manifest"], manifest)
    bundle = {"runner": runner, "paths": paths, "plan": plan}
    return bundle, paths


def _remaining_opaque_envelope_bundle(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    call_hash = "a" * 64
    stage_hash = "c" * 64
    scientific_hash = "d" * 64
    raw_root = tmp_path / "raw_logits" / analyzer.REMAINING_STAGE
    raw_root.mkdir(parents=True)
    paths = {
        "remaining_attempt": tmp_path / "remaining_main_attempt.json",
        "remaining_discovery_topology_records": tmp_path
        / "remaining_main_discovery_topology_records.jsonl",
        "remaining_confirmation_components_records": tmp_path
        / "remaining_main_confirmation_components_records.jsonl",
        "remaining_confirmation_topology_records": tmp_path
        / "remaining_main_confirmation_topology_records.jsonl",
        "remaining_execution_manifest": tmp_path
        / "remaining_main_execution_manifest.json",
        "remaining_raw_root": raw_root,
    }
    _add_planning_artifacts(paths, tmp_path)

    record_blocks: dict[str, dict] = {}
    prompts: list[dict] = []
    for block, registry in analyzer.REMAINING_BLOCK_REGISTRY.items():
        path = paths[registry["path_key"]]
        path.write_text(f"opaque-{block}\n", encoding="utf-8")
        record_blocks[block] = {
            **_binding(path),
            "execution_block": block,
            "row_count": registry["stage_stop_row"] - registry["stage_start_row"],
            "canonical_sha256": "e" * 64,
            "global_start_row": registry["global_start_row"],
            "global_stop_row": registry["global_stop_row"],
            "stage_start_row": registry["stage_start_row"],
            "stage_stop_row": registry["stage_stop_row"],
            "raw_shard_indices": list(
                range(registry["shard_start"], registry["shard_stop"])
            ),
        }
        prompts.extend(
            {
                "execution_stage": analyzer.REMAINING_STAGE,
                "execution_block": block,
            }
            for _ in range(registry["stage_stop_row"] - registry["stage_start_row"])
        )

    raw_specs: list[dict] = []
    raw_bindings: list[dict] = []
    for shard_index in range(26):
        stage_start = shard_index * 64
        block = next(
            name
            for name, registry in analyzer.REMAINING_BLOCK_REGISTRY.items()
            if registry["stage_start_row"] <= stage_start < registry["stage_stop_row"]
        )
        array = np.zeros((64, 4), dtype="<f4")
        path = raw_root / f"shard_{shard_index:03d}.npy"
        np.save(path, array, allow_pickle=False)
        spec = {
            "execution_stage": analyzer.REMAINING_STAGE,
            "execution_block": block,
            "index": shard_index,
            "phase_start_row": stage_start,
            "phase_stop_row": stage_start + 64,
            "global_start_row": 640 + stage_start,
            "global_stop_row": 640 + stage_start + 64,
            "rows": 64,
            "shape": [64, 4],
            "dtype": "<f4",
            "path": str(path),
        }
        raw_specs.append(spec)
        raw_bindings.append(
            {
                **spec,
                **_binding(path),
                "logical_sha256": analyzer._logical_array_sha256(array),
            }
        )

    total_raw_bytes = sum(int(np.prod(spec["shape"])) * 4 for spec in raw_specs)
    plan = {
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "call_plan_sha256": call_hash,
        "scientific_registry_sha256": scientific_hash,
        "stage_plan_sha256": {analyzer.REMAINING_STAGE: stage_hash},
        "raw_logits_shards": {analyzer.REMAINING_STAGE: raw_specs},
        "disk_requirements": {"raw_logits_expected_bytes": total_raw_bytes},
        "model": {"vocab_size": 4},
        "prompts": prompts,
    }
    prior = {
        "execution_manifest_file_sha256": "1" * 64,
        "execution_manifest_canonical_sha256": "2" * 64,
        "authorization_file_sha256": "3" * 64,
        "authorization_canonical_sha256": "4" * 64,
        "authorization_status": analyzer.DISCOVERY_COMPONENT_PASS,
    }
    attempt = {
        "schema_version": runner.ATTEMPT_SCHEMA,
        "status": "V6A_R2_REMAINING_MAIN_EXECUTION_ATTEMPT_STARTED_IMMUTABLE",
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "execution_stage": analyzer.REMAINING_STAGE,
        "call_plan_sha256": call_hash,
        "stage_plan_sha256": stage_hash,
        "scientific_registry_sha256": scientific_hash,
        "expected_calls": 1664,
        "global_start_row": 640,
        "global_stop_row": 2304,
        "stage_start_row": 0,
        "stage_stop_row": 1664,
        "block_order": list(analyzer.REMAINING_BLOCK_REGISTRY),
        "block_counts": {
            block: registry["stage_stop_row"] - registry["stage_start_row"]
            for block, registry in analyzer.REMAINING_BLOCK_REGISTRY.items()
        },
        "partial_resume_allowed": False,
        "generation_used": False,
        "model": plan["model"],
        "forward_contract": analyzer._expected_forward_contract(plan),
        "preflight": _preflight(plan, analyzer.REMAINING_STAGE, 640, 2304),
        "source_locks": _source_locks(paths),
        "prior_discovery_authorization": prior,
    }
    _write_json(paths["remaining_attempt"], attempt)
    manifest = {
        "schema_version": runner.EXECUTION_SCHEMA,
        "status": "V6A_R2_REMAINING_MAIN_EXECUTION_COMPLETE_NOT_ANALYZED",
        "registration_state": runner.REGISTRATION_STATE,
        "execution_revision": runner.EXECUTION_REVISION,
        "execution_stage": analyzer.REMAINING_STAGE,
        "call_plan_sha256": call_hash,
        "stage_plan_sha256": stage_hash,
        "scientific_registry_sha256": scientific_hash,
        "phase_model_calls": 1664,
        "cumulative_model_calls": 2304,
        "global_start_row": 640,
        "global_stop_row": 2304,
        "block_order": list(analyzer.REMAINING_BLOCK_REGISTRY),
        "block_counts": attempt["block_counts"],
        "generation_used": False,
        "partial_resume_allowed": False,
        "attempt": _binding(paths["remaining_attempt"]),
        "record_blocks": record_blocks,
        "raw_logits_shards": raw_bindings,
        "source_locks": _source_locks(paths),
        "prior_discovery_authorization": prior,
    }
    _write_json(paths["remaining_execution_manifest"], manifest)
    return {"runner": runner, "paths": paths, "plan": plan}, paths


def _rewrite_discovery_records(
    paths: dict[str, Path],
    mutate,
) -> None:
    records = [
        json.loads(line)
        for line in paths["discovery_records"].read_text(encoding="utf-8").splitlines()
    ]
    mutate(records[0])
    core = {key: value for key, value in records[0].items() if key != "record_id"}
    records[0]["record_id"] = analyzer.canonical_sha256(core)
    paths["discovery_records"].write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = json.loads(
        paths["discovery_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["records"] = {
        **_binding(paths["discovery_records"]),
        "count": 640,
        "canonical_sha256": analyzer.canonical_sha256(records),
    }
    _write_json(paths["discovery_execution_manifest"], manifest)


def test_component_gates_pass_the_exact_balanced_panel() -> None:
    rows = _component_rows("discovery", 0)

    result = analyzer.discovery_gate_from_validated_records(rows)

    assert result["status"] == analyzer.DISCOVERY_COMPONENT_PASS
    assert result["remaining_main_authorized"] is True
    assert result["topology_rows_read"] == 0
    assert result["components"]["gates"]["pass"] is True


def test_one_lookup_tie_is_ordinary_incorrect_and_within_every_budget() -> None:
    rows = _component_rows("discovery", 0)
    lookup = next(row for row in rows if row["family"] == analyzer.CODEBOOK_LOOKUP)
    _make_incorrect(lookup, tie=True)

    result = analyzer.component_summary(rows, split="discovery")

    family = result["families"][analyzer.CODEBOOK_LOOKUP]
    assert family["pairwise_overall"]["correct"] == 127
    assert family["strict_global_overall"]["correct"] == 127
    assert family["pass"] is True
    assert result["gates"]["pass"] is True


def test_two_failures_in_one_lookup_label_fail_registered_label_budget() -> None:
    rows = _component_rows("discovery", 0)
    lookup = [row for row in rows if row["family"] == analyzer.CODEBOOK_LOOKUP]
    label = lookup[0]["expected_answer"]
    affected = [row for row in lookup if row["expected_answer"] == label][:2]
    for row in affected:
        _make_incorrect(row)

    result = analyzer.component_summary(rows, split="discovery")

    family = result["families"][analyzer.CODEBOOK_LOOKUP]
    assert family["pairwise_by_answer_label"][label]["correct"] == 6
    assert family["pairwise_by_answer_label"][label]["pass"] is False
    assert result["gates"]["pass"] is False


def test_discovery_gate_rejects_any_topology_row_before_summarizing() -> None:
    rows = _component_rows("discovery", 0)
    rows[-1] = _topology_rows("discovery", 640)[0]

    with pytest.raises(analyzer.R2TopologyAnalysisError, match="topology row"):
        analyzer.discovery_gate_from_validated_records(rows)


def test_confirmation_failure_suppresses_topology_function() -> None:
    discovery, remaining = _full_staged_rows()
    confirmation_lookup = [
        row
        for row in remaining[512:1152]
        if row["family"] == analyzer.CODEBOOK_LOOKUP
    ]
    label = confirmation_lookup[0]["expected_answer"]
    for row in [
        row for row in confirmation_lookup if row["expected_answer"] == label
    ][:2]:
        _make_incorrect(row)

    def forbidden_topology(_rows):
        raise AssertionError("topology must not run before confirmation admission")

    result = analyzer.final_from_validated_records(
        discovery,
        remaining,
        topology_function=forbidden_topology,
    )

    assert result["status"] == analyzer.CONFIRMATION_COMPONENT_FAIL
    assert result["topology_suppressed"] is True
    assert result["topology_inference_performed"] is False


def test_topology_estimands_bootstrap_and_accuracy_support_reversal() -> None:
    discovery, remaining = _full_staged_rows()

    result = analyzer.final_from_validated_records(discovery, remaining)

    assert result["status"] == analyzer.REVERSAL_SUPPORTED
    assert result["activation_design_authorized"] is True
    assert result["activation_execution_authorized"] is False
    for split in ("discovery", "confirmation"):
        topology = result["topology"]["splits"][split]
        assert topology["D"]["q=-1__a=-1"]["mean"] == pytest.approx(2.0)
        assert topology["D"]["q=+1__a=+1"]["mean"] == pytest.approx(-2.0)
        assert topology["R"]["mean"] == pytest.approx(4.0)
        assert topology["Q"]["mean"] == pytest.approx(2.0)
        assert topology["A"]["mean"] == pytest.approx(2.0)
        assert topology["residual_o_by_q_by_a_interaction"]["mean"] == pytest.approx(0.0)
        assert topology["accuracy_endpoints"]["v5_like_second_minus_first"] == 1.0
        assert topology["accuracy_endpoints"]["v4_like_first_minus_second"] == 1.0
        assert topology["maximum_reversal_gates"]["pass"] is True
        assert topology["R"]["bootstrap_95"]["draws"] == 10_000
        assert topology["R"]["bootstrap_95"]["seed"] == 260806


def test_corrupt_margin_and_missing_match_are_engineering_errors() -> None:
    discovery, remaining = _full_staged_rows()
    broken_margin = copy.deepcopy(discovery[0])
    broken_margin["diagnostics"]["expected_minus_distractor_margin"] = 9.0
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="does not reconstruct"):
        analyzer._diagnostics(broken_margin)

    all_rows = [*discovery, *remaining]
    corrupt = copy.deepcopy(all_rows)
    single = next(row for row in corrupt if row["family"] == analyzer.SINGLE_TARGET)
    single["factors"]["u"] *= -1
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="matched"):
        analyzer.topology_summary(corrupt)


def test_full_vocab_tie_and_outside_pair_are_metric_failures_not_engineering_errors() -> None:
    tie_logits = np.asarray([0.0, 3.0, 3.0, 1.0], dtype="<f4")
    outside_logits = np.asarray([0.0, 3.0, 1.0, 4.0], dtype="<f4")

    tied = analyzer.diagnostics_from_full_vocab(tie_logits, 1, 2)
    outside = analyzer.diagnostics_from_full_vocab(outside_logits, 1, 2)

    assert tied["pairwise_correct"] is False
    assert tied["pairwise_tie"] is True
    assert tied["strict_unique_global_correct"] is False
    assert outside["pairwise_correct"] is True
    assert outside["strict_unique_global_correct"] is False


def test_discovery_artifact_replay_validates_640_rows_without_remaining_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    observed_json_paths: list[Path] = []
    original = analyzer._load_json

    def tracked(path: Path):
        observed_json_paths.append(path)
        return original(path)

    monkeypatch.setattr(analyzer, "_load_json", tracked)

    execution = analyzer._validate_stage_execution(
        bundle,
        stage=analyzer.DISCOVERY_STAGE,
    )

    assert len(execution["records"]) == 640
    assert all(row["family"] != analyzer.TWO_FACT for row in execution["records"])
    assert paths["remaining_attempt"] not in observed_json_paths
    assert paths["remaining_execution_manifest"] not in observed_json_paths
    assert paths["remaining_records"] not in observed_json_paths


def test_discovery_artifact_replay_rejects_raw_shard_corruption(tmp_path: Path) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    shard = paths["discovery_raw_root"] / "shard_000.npy"
    array = np.load(shard, allow_pickle=False)
    array[0, 1] += 1.0
    np.save(shard, array, allow_pickle=False)

    with pytest.raises(analyzer.R2TopologyAnalysisError, match="file binding changed"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)


def test_discovery_artifact_replay_rejects_a_topology_path_in_raw_registry(
    tmp_path: Path,
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    manifest = json.loads(
        paths["discovery_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["raw_logits_shards"][0]["path"] = str(
        paths["remaining_raw_root"] / "shard_000.npy"
    )
    _write_json(paths["discovery_execution_manifest"], manifest)

    with pytest.raises(analyzer.R2TopologyAnalysisError, match="plan binding changed"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)


def test_closed_attempt_and_manifest_schemas_reject_unknown_keys(
    tmp_path: Path,
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    manifest = json.loads(
        paths["discovery_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["unregistered"] = False
    _write_json(paths["discovery_execution_manifest"], manifest)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="unknown=.*unregistered"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)

    bundle, paths = _discovery_artifact_bundle(tmp_path / "attempt")
    attempt = json.loads(paths["discovery_attempt"].read_text(encoding="utf-8"))
    attempt["unregistered"] = False
    _write_json(paths["discovery_attempt"], attempt)
    manifest = json.loads(
        paths["discovery_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["attempt"] = _binding(paths["discovery_attempt"])
    _write_json(paths["discovery_execution_manifest"], manifest)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="unknown=.*unregistered"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)


def test_exact_execution_revision_and_record_identity_echo_are_enforced(
    tmp_path: Path,
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    attempt = json.loads(paths["discovery_attempt"].read_text(encoding="utf-8"))
    attempt["execution_revision"] = "changed"
    _write_json(paths["discovery_attempt"], attempt)
    manifest = json.loads(
        paths["discovery_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["attempt"] = _binding(paths["discovery_attempt"])
    _write_json(paths["discovery_execution_manifest"], manifest)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="execution_revision"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)

    bundle, paths = _discovery_artifact_bundle(tmp_path / "identity")
    _rewrite_discovery_records(
        paths,
        lambda record: record.__setitem__("record_identity_id", "f" * 64),
    )
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="record_identity_id"):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda record: record.__setitem__("unregistered", False), "stage record schema"),
        (
            lambda record: record["diagnostics"].__setitem__("unregistered", False),
            "stored component diagnostics schema",
        ),
    ],
)
def test_record_and_component_diagnostic_schemas_are_closed(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    _rewrite_discovery_records(paths, mutation)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match=error):
        analyzer._validate_stage_execution(bundle, stage=analyzer.DISCOVERY_STAGE)


def test_remaining_record_block_binding_schema_is_closed(tmp_path: Path) -> None:
    bundle, paths = _remaining_opaque_envelope_bundle(tmp_path)
    manifest = json.loads(
        paths["remaining_execution_manifest"].read_text(encoding="utf-8")
    )
    manifest["record_blocks"]["discovery-topology"]["unregistered"] = False
    _write_json(paths["remaining_execution_manifest"], manifest)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="unknown=.*unregistered"):
        analyzer._validate_remaining_envelope(bundle)


def test_remaining_envelope_hashes_opaque_files_without_loading_any_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _paths = _remaining_opaque_envelope_bundle(tmp_path)

    monkeypatch.setattr(
        analyzer,
        "_load_jsonl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("record payload must remain unopened")
        ),
    )
    monkeypatch.setattr(
        analyzer.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw-logit matrix must remain unloaded")
        ),
    )

    envelope = analyzer._validate_remaining_envelope(bundle)

    assert set(envelope["record_blocks"]) == set(analyzer.REMAINING_BLOCK_REGISTRY)
    assert len(envelope["shard_registry"]) == 26
    assert envelope["loaded_blocks"] == []


def test_generic_remaining_stage_validation_rejects_before_any_payload_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed: list[str] = []

    def forbidden(label):
        def invoke(*_args, **_kwargs):
            accessed.append(label)
            raise AssertionError(f"remaining-stage firewall opened {label}")

        return invoke

    monkeypatch.setattr(
        analyzer,
        "_validate_remaining_envelope",
        forbidden("remaining envelope"),
    )
    monkeypatch.setattr(
        analyzer,
        "_load_remaining_block",
        forbidden("remaining record block"),
    )
    monkeypatch.setattr(analyzer, "_load_jsonl", forbidden("record stream"))
    monkeypatch.setattr(analyzer.np, "load", forbidden("raw-logit shard"))

    with pytest.raises(analyzer.R2TopologyAnalysisError, match="discovery-only"):
        analyzer._validate_stage_execution({}, stage=analyzer.REMAINING_STAGE)

    assert accessed == []


def _actual_shape_identity_pair() -> tuple[dict, dict]:
    prompt = {
        "block_call_index": 0,
        "cell_id": "cell-0000",
        "execution_block": "discovery-components",
        "execution_stage": analyzer.DISCOVERY_STAGE,
        "global_call_index": 0,
        "prompt_id": "prompt-0000",
        "stage_call_index": 0,
    }
    prompt["record_identity_id"] = runner.record_identity_id(prompt)
    cell = {
        key: prompt[key] for key in analyzer.CELL_SHARED_IDENTITY_FIELDS
    }
    cell["record_identity_id"] = prompt["record_identity_id"]
    return prompt, cell


def test_planned_identity_accepts_real_cell_shape_without_prompt_id() -> None:
    prompt, cell = _actual_shape_identity_pair()
    assert "prompt_id" not in cell
    assert set(analyzer.RECORD_IDENTITY_FIELDS) == {
        *analyzer.CELL_SHARED_IDENTITY_FIELDS,
        "prompt_id",
    }
    analyzer._validate_planned_record_identities([prompt], [cell])


@pytest.mark.parametrize("field", analyzer.CELL_SHARED_IDENTITY_FIELDS)
def test_planned_identity_rejects_each_shared_cell_field_drift(field: str) -> None:
    prompt, cell = _actual_shape_identity_pair()
    original = cell[field]
    cell[field] = original + 1 if isinstance(original, int) else original + "-drift"
    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="planned record identity changed",
    ):
        analyzer._validate_planned_record_identities([prompt], [cell])


@pytest.mark.parametrize("target", ["prompt_id", "prompt_identity", "cell_identity"])
def test_planned_identity_rejects_prompt_or_identity_echo_drift(target: str) -> None:
    prompt, cell = _actual_shape_identity_pair()
    if target == "prompt_id":
        prompt["prompt_id"] = "prompt-drift"
    elif target == "prompt_identity":
        prompt["record_identity_id"] = "0" * 64
    else:
        cell["record_identity_id"] = "0" * 64
    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="planned record identity changed",
    ):
        analyzer._validate_planned_record_identities([prompt], [cell])


def test_independent_plan_replay_rejects_any_rebuild_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = []
    cells = []
    for index in range(2304):
        stage = analyzer.DISCOVERY_STAGE if index < 640 else analyzer.REMAINING_STAGE
        if index < 640:
            block = "discovery-components"
            block_index = index
            stage_index = index
        elif index < 1152:
            block = "discovery-topology"
            block_index = index - 640
            stage_index = index - 640
        elif index < 1792:
            block = "confirmation-components"
            block_index = index - 1152
            stage_index = index - 640
        else:
            block = "confirmation-topology"
            block_index = index - 1792
            stage_index = index - 640
        prompt = {
            "global_call_index": index,
            "block_call_index": block_index,
            "stage_call_index": stage_index,
            "execution_stage": stage,
            "execution_block": block,
            "cell_id": f"cell-{index}",
            "prompt_id": f"prompt-{index}",
        }
        prompt["record_identity_id"] = runner.record_identity_id(prompt)
        prompts.append(prompt)
        cells.append(
            {
                key: value
                for key, value in prompt.items()
                if key != "prompt_id"
            }
        )
    plan = {
        "execution_revision": analyzer.EXECUTION_REVISION,
        "expected_calls": 2304,
        "prompts": prompts,
        "cells": cells,
        "execution_stage_counts": {
            analyzer.DISCOVERY_STAGE: 640,
            analyzer.REMAINING_STAGE: 1664,
        },
        "block_counts": {
            "discovery-components": 640,
            "discovery-topology": 512,
            "confirmation-components": 640,
            "confirmation-topology": 512,
        },
    }
    receipt = {"receipt": "frozen"}
    paths = {"plan_manifest": tmp_path / "plan_manifest.json"}
    _write_json(paths["plan_manifest"], {"plan": plan})

    class _FakeRunner:
        EXECUTION_REVISION = analyzer.EXECUTION_REVISION

        @staticmethod
        def record_identity_id(value):
            return runner.record_identity_id(value)

        @staticmethod
        def validate_frozen_plan(_root):
            return plan, {"design": True}, {"dependency": True}, receipt

        @staticmethod
        def load_and_rebuild_fixture():
            return {"fixture": True}, {"manifest": True}

        @staticmethod
        def dependency_lock(_analyzer_path):
            return {"dependency": True}

        @staticmethod
        def load_tokenizer_from_sealed_v2(_dependency):
            return object()

        @staticmethod
        def build_plan(_tokenizer, _fixture, _dependency, *, result_root):
            assert result_root == tmp_path
            return plan, receipt

        @staticmethod
        def artifact_paths(_root):
            return paths

    monkeypatch.setattr(analyzer, "_runner_module", lambda: _FakeRunner)
    assert analyzer.replay_frozen_plan(tmp_path)["plan"] == plan

    changed_receipt = {"receipt": "changed"}
    monkeypatch.setattr(
        _FakeRunner,
        "build_plan",
        staticmethod(
            lambda *_args, **_kwargs: (plan, changed_receipt)
        ),
    )
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="does not independently rebuild"):
        analyzer.replay_frozen_plan(tmp_path)


def test_runner_authorization_api_transitively_replays_exact_frozen_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, paths = _discovery_artifact_bundle(tmp_path)
    paths["discovery_analysis"] = tmp_path / "discovery_components_analysis.json"
    execution = analyzer._validate_stage_execution(
        bundle,
        stage=analyzer.DISCOVERY_STAGE,
    )
    authorization = analyzer._discovery_analysis_payload(bundle, execution)
    assert authorization["status"] == analyzer.DISCOVERY_COMPONENT_PASS
    _write_json(paths["discovery_analysis"], authorization)

    def replay(_root, *, dependency_replay_override=None):
        assert dependency_replay_override is None
        return bundle

    monkeypatch.setattr(analyzer, "replay_frozen_plan", replay)
    monkeypatch.setattr(
        analyzer,
        "_write_frozen_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization validation must never write an artifact")
        ),
    )

    receipt = analyzer.validate_remaining_authorization_for_runner(tmp_path)

    assert receipt["authorization"] == authorization
    assert receipt["authorization_status"] == analyzer.DISCOVERY_COMPONENT_PASS
    assert receipt["replay_equal"] is True
    assert receipt["model_calls_issued_by_validator"] == 0
    assert receipt["generation_calls_issued_by_validator"] == 0
    assert set(receipt) == {
        "authorization",
        "authorization_file_sha256",
        "authorization_canonical_sha256",
        "discovery_execution_manifest_file_sha256",
        "discovery_execution_manifest_canonical_sha256",
        "authorization_status",
        "call_plan_sha256",
        "stage_plan_sha256",
        "replay_equal",
        "model_calls_issued_by_validator",
        "generation_calls_issued_by_validator",
    }

    changed = copy.deepcopy(authorization)
    changed["claim_scope"] = "changed"
    _write_json(paths["discovery_analysis"], changed)
    with pytest.raises(analyzer.R2TopologyAnalysisError, match="replay exactly"):
        analyzer.validate_remaining_authorization_for_runner(tmp_path)


def test_discovery_behavioral_failure_is_not_engineering_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _component_rows("discovery", 0)
    lookup = [row for row in rows if row["family"] == analyzer.CODEBOOK_LOOKUP]
    label = lookup[0]["expected_answer"]
    for row in [row for row in lookup if row["expected_answer"] == label][:2]:
        _make_incorrect(row)
    bundle = {
        "plan": {
            "call_plan_sha256": "a" * 64,
            "stage_plan_sha256": {analyzer.DISCOVERY_STAGE: "b" * 64},
        }
    }
    monkeypatch.setattr(analyzer, "replay_frozen_plan", lambda _root: bundle)
    monkeypatch.setattr(
        analyzer,
        "_validate_stage_execution",
        lambda _bundle, *, stage: {
            "records": rows,
            "artifact_hashes": {"stage": stage},
        },
    )

    result = analyzer.analyze_discovery_gate(write=False)

    assert result["status"] == analyzer.DISCOVERY_COMPONENT_FAIL
    assert result["engineering_valid"] is True
    assert result["component_qualified"] is False
    assert result["authorization_issued"] is False


@pytest.mark.parametrize("state", ["DRAFT_ZERO_FORWARD", "UNKNOWN_STATE"])
@pytest.mark.parametrize(
    ("stage", "entrypoint"),
    [
        ("discovery-gate", analyzer.analyze_discovery_gate),
        ("final", analyzer.analyze_final),
    ],
)
def test_analysis_write_rejects_nonfrozen_state_before_replay_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    stage: str,
    entrypoint,
) -> None:
    target_root = tmp_path / f"blocked-{stage}-{state}"
    calls: list[str] = []

    monkeypatch.setattr(runner, "REGISTRATION_STATE", state)
    monkeypatch.setattr(
        analyzer,
        "replay_frozen_plan",
        lambda *_args, **_kwargs: calls.append("replay"),
    )
    monkeypatch.setattr(
        analyzer,
        "_write_frozen_json",
        lambda *_args, **_kwargs: calls.append("write"),
    )

    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="exact frozen registration state",
    ):
        entrypoint(target_root, write=True)
    assert calls == []
    assert not target_root.exists()


@pytest.mark.parametrize(
    "entrypoint", [analyzer.analyze_discovery_gate, analyzer.analyze_final]
)
def test_analysis_write_rejects_frozen_alternate_root_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
) -> None:
    target_root = tmp_path / "alternate-root"
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "REGISTRATION_STATE",
        runner.EXECUTION_ENABLED_REGISTRATION_STATE,
    )
    monkeypatch.setattr(
        analyzer,
        "replay_frozen_plan",
        lambda *_args, **_kwargs: calls.append("replay"),
    )

    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="exact registered result root",
    ):
        entrypoint(target_root, write=True)
    assert calls == []
    assert not target_root.exists()


@pytest.mark.parametrize(
    ("entrypoint", "analysis_key"),
    [
        (analyzer.analyze_discovery_gate, "discovery_analysis"),
        (analyzer.analyze_final, "final_analysis"),
    ],
)
@pytest.mark.parametrize("manifest_mode", ["missing", "wrong"])
def test_analysis_write_requires_exact_registered_main_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
    analysis_key: str,
    manifest_mode: str,
) -> None:
    registered_root = tmp_path / "registered-root"
    monkeypatch.setattr(
        runner,
        "REGISTRATION_STATE",
        runner.EXECUTION_ENABLED_REGISTRATION_STATE,
    )
    monkeypatch.setattr(runner, "RESULT_ROOT", registered_root)
    monkeypatch.setattr(analyzer, "RESULT_ROOT", registered_root)
    if manifest_mode == "wrong":
        paths = runner.artifact_paths(registered_root)
        paths["plan_manifest"].parent.mkdir(parents=True)
        paths["plan_manifest"].write_text(
            json.dumps(
                {
                    "artifact_scope": "integration_test_only",
                    "status": "R2_PLAN_FROZEN_ZERO_FORWARD",
                    "registration_state": "DRAFT_ZERO_FORWARD",
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="registered-main plan",
    ):
        entrypoint(registered_root, write=True)
    assert not runner.artifact_paths(registered_root)[analysis_key].exists()


@pytest.mark.parametrize(
    ("entrypoint", "analysis_key"),
    [
        (analyzer.analyze_discovery_gate, "discovery_analysis"),
        (analyzer.analyze_final, "final_analysis"),
    ],
)
def test_analysis_write_requires_completed_stage_manifest_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
    analysis_key: str,
) -> None:
    registered_root = tmp_path / "registered-root"
    paths = runner.artifact_paths(registered_root)
    paths["plan_manifest"].parent.mkdir(parents=True)
    paths["plan_manifest"].write_text(
        json.dumps(
            {
                "artifact_scope": "registered_main",
                "status": "R2_PLAN_FROZEN_BEFORE_ANY_R2_MODEL_FORWARD",
                "registration_state": runner.EXECUTION_ENABLED_REGISTRATION_STATE,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "REGISTRATION_STATE",
        runner.EXECUTION_ENABLED_REGISTRATION_STATE,
    )
    monkeypatch.setattr(runner, "RESULT_ROOT", registered_root)
    monkeypatch.setattr(analyzer, "RESULT_ROOT", registered_root)
    monkeypatch.setattr(
        analyzer,
        "replay_frozen_plan",
        lambda *_args, **_kwargs: calls.append("replay"),
    )

    with pytest.raises(
        analyzer.R2TopologyAnalysisError,
        match="completed execution manifest",
    ):
        entrypoint(registered_root, write=True)
    assert calls == []
    assert not paths[analysis_key].exists()


def test_discovery_corruption_is_engineering_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "replay_frozen_plan", lambda _root: {"plan": {}})
    monkeypatch.setattr(
        analyzer,
        "_validate_stage_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            analyzer.R2TopologyAnalysisError("raw hash corruption")
        ),
    )

    result = analyzer.analyze_discovery_gate(write=False)

    assert result["status"] == analyzer.ENGINEERING_INVALID
    assert result["engineering_valid"] is False
    assert result["component_qualified"] is False


def test_final_wrapper_does_not_materialize_topology_before_confirmation_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery, remaining = _full_staged_rows()
    confirmation_lookup = [
        row
        for row in remaining[512:1152]
        if row["family"] == analyzer.CODEBOOK_LOOKUP
    ]
    label = confirmation_lookup[0]["expected_answer"]
    for row in [
        row for row in confirmation_lookup if row["expected_answer"] == label
    ][:2]:
        _make_incorrect(row)
    bundle = {
        "plan": {
            "call_plan_sha256": "a" * 64,
            "stage_plan_sha256": {
                analyzer.DISCOVERY_STAGE: "b" * 64,
                analyzer.REMAINING_STAGE: "c" * 64,
            },
        },
        "paths": {"discovery_analysis": Path("unused-authorization.json")},
    }
    monkeypatch.setattr(analyzer, "replay_frozen_plan", lambda _root: bundle)
    prior = {
        "execution_manifest_file_sha256": "1" * 64,
        "execution_manifest_canonical_sha256": "2" * 64,
        "authorization_file_sha256": "3" * 64,
        "authorization_canonical_sha256": "4" * 64,
        "authorization_status": analyzer.DISCOVERY_COMPONENT_PASS,
    }
    monkeypatch.setattr(
        analyzer,
        "_replay_discovery_authorization",
        lambda _bundle: (
            {"status": analyzer.DISCOVERY_COMPONENT_PASS},
            {"records": discovery, "artifact_hashes": {}},
            prior,
        ),
    )
    envelope = {"artifact_hashes": {}}

    def validate_envelope(_bundle, *, prior_discovery_authorization):
        assert prior_discovery_authorization == prior
        return envelope

    monkeypatch.setattr(analyzer, "_validate_remaining_envelope", validate_envelope)
    opened_blocks: list[str] = []

    def load_block(_bundle, _envelope, block):
        opened_blocks.append(block)
        if block != "confirmation-components":
            raise AssertionError("topology record streams must remain unopened")
        return {"records": remaining[512:1152]}

    monkeypatch.setattr(
        analyzer,
        "_load_remaining_block",
        load_block,
    )
    monkeypatch.setattr(analyzer, "file_sha256", lambda _path: "f" * 64)

    result = analyzer.analyze_final(write=False)

    assert result["status"] == analyzer.CONFIRMATION_COMPONENT_FAIL
    assert result["topology_inference_performed"] is False
    assert result["topology_record_streams_opened"] == 0
    assert result["topology_raw_shards_loaded"] == 0
    assert opened_blocks == ["confirmation-components"]

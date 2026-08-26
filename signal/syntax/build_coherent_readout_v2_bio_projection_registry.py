"""Freeze outcome-blind biological projections for every v2 syntax envelope.

The registry is created before syntax selection.  It contains all four candidates
and only deterministic prompt templates: no biological item, gold label, model
output, metric, selected winner, or model call enters this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ID = "coherent-readout-v2-biological-answer-envelope-projections-v1"
REGISTRY_SCHEMA = "coherent-readout-v2-biological-projection-registry-v1"
MANIFEST_SCHEMA = "coherent-readout-v2-biological-projection-registry-manifest-v1"
FREEZE_DATE = "2026-08-02"
PURPOSE = "outcome_blind_biological_answer_envelope_projection"

BIO_CORE = (
    "Classify the cell using exactly one label.\n"
    "{line_1}\n{line_2}\n"
    "Genes, highest expression rank first: {gene_sentence}"
)
USER_ONLY_OUTPUT_INSTRUCTION = (
    "Follow the label mapping. Answer with exactly one character: X or Y. "
    "Do not write any other text."
)

CANDIDATE_ORDER = (
    "assistant_prefill_answer_colon",
    "system_user_exact",
    "user_only_exact",
    "user_label_colon_v1_control",
)

SYSTEM_PREFILL = "Complete the assistant answer with exactly X or Y and no other text."
SYSTEM_EXACT = (
    "Follow the user's label mapping. Your entire response must be exactly one "
    "character: X or Y. Do not write any other text."
)

SOURCE_SYNTAX_CANDIDATES = (
    {
        "candidate_id": CANDIDATE_ORDER[0],
        "candidate_rank": 0,
        "message_roles": ["system", "user", "assistant"],
        "system_template": SYSTEM_PREFILL,
        "user_template": (
            "Classify the record from its explicit declaration.\n"
            "{line_1}\n{line_2}\n{declaration}"
        ),
        "assistant_template": "Answer:",
        "render_mode": "continue_final_message",
        "add_generation_prompt": False,
        "continue_final_message": True,
        "enable_thinking": False,
        "x_answer_text": " X",
        "y_answer_text": " Y",
    },
    {
        "candidate_id": CANDIDATE_ORDER[1],
        "candidate_rank": 1,
        "message_roles": ["system", "user"],
        "system_template": SYSTEM_EXACT,
        "user_template": (
            "Classify the record from its explicit declaration.\n"
            "{line_1}\n{line_2}\n{declaration}\nReturn exactly X or Y."
        ),
        "assistant_template": None,
        "render_mode": "add_generation_prompt",
        "add_generation_prompt": True,
        "continue_final_message": False,
        "enable_thinking": False,
        "x_answer_text": "X",
        "y_answer_text": "Y",
    },
    {
        "candidate_id": CANDIDATE_ORDER[2],
        "candidate_rank": 2,
        "message_roles": ["user"],
        "system_template": None,
        "user_template": (
            f"{USER_ONLY_OUTPUT_INSTRUCTION}\n"
            "Classify the record from its explicit declaration.\n"
            "{line_1}\n{line_2}\n{declaration}\nAnswer:"
        ),
        "assistant_template": None,
        "render_mode": "add_generation_prompt",
        "add_generation_prompt": True,
        "continue_final_message": False,
        "enable_thinking": False,
        "x_answer_text": "X",
        "y_answer_text": "Y",
    },
    {
        "candidate_id": CANDIDATE_ORDER[3],
        "candidate_rank": 3,
        "message_roles": ["user"],
        "system_template": None,
        "user_template": (
            "Classify the record using exactly one label.\n"
            "{line_1}\n{line_2}\n{declaration}\nLabel:"
        ),
        "assistant_template": None,
        "render_mode": "add_generation_prompt",
        "add_generation_prompt": True,
        "continue_final_message": False,
        "enable_thinking": False,
        "x_answer_text": "X",
        "y_answer_text": "Y",
    },
)

FIREWALL = {
    "partition": "permanent_development_only",
    "permanent": True,
    "gold_labels": "absent",
    "syntax_outcomes_at_freeze": "unobserved",
    "confirmatory_eligibility": "prohibited",
    "promotion_to_confirmation": "forbidden",
    "claim_scope": (
        "answer_envelope_projection_only_no_biology_knowledge_or_activation_inference"
    ),
}

DEFAULT_OUT = Path(__file__).with_name(
    "coherent_readout_v2_bio_projection_registry.json"
)
DEFAULT_MANIFEST = Path(__file__).with_name(
    "coherent_readout_v2_bio_projection_registry.manifest.json"
)
SOURCE_SYNTAX_RUNNER = ROOT / "eval" / "run_coherent_readout_v2_syntax.py"

DEFINITION_KEYS = {
    "candidate_id",
    "candidate_rank",
    "message_roles",
    "system_template",
    "user_template",
    "assistant_template",
    "render_mode",
    "add_generation_prompt",
    "continue_final_message",
    "enable_thinking",
    "x_answer_text",
    "y_answer_text",
}
PROJECTION_KEYS = {
    "candidate_id",
    "candidate_rank",
    "source_syntax_candidate_definition",
    "source_syntax_candidate_definition_sha256",
    "biological_projection",
    "biological_projection_sha256",
}
REGISTRY_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "freeze_date",
    "mode",
    "purpose",
    "firewall",
    "biological_core",
    "biological_core_sha256",
    "candidate_priority",
    "source_syntax_candidate_registry_sha256",
    "projection_entries_canonical_sha256",
    "projections",
    "model_calls_made_by_builder",
}
MANIFEST_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "freeze_date",
    "mode",
    "purpose",
    "claim_scope",
    "firewall",
    "contract",
    "provenance",
    "artifacts",
}


class BioProjectionRegistryError(ValueError):
    """Raised when the outcome-blind projection registry violates its contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BioProjectionRegistryError(
            f"{label} schema mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _project_candidate(source: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = source["candidate_id"]
    user_templates = {
        CANDIDATE_ORDER[0]: BIO_CORE,
        CANDIDATE_ORDER[1]: BIO_CORE + "\nReturn exactly X or Y.",
        CANDIDATE_ORDER[2]: (
            USER_ONLY_OUTPUT_INSTRUCTION + "\n" + BIO_CORE + "\nAnswer:"
        ),
        CANDIDATE_ORDER[3]: BIO_CORE + "\nLabel:",
    }
    if candidate_id not in user_templates:
        raise BioProjectionRegistryError(f"unknown syntax candidate: {candidate_id}")
    return {**dict(source), "user_template": user_templates[candidate_id]}


def _projection_entries() -> list[dict[str, Any]]:
    entries = []
    for source in SOURCE_SYNTAX_CANDIDATES:
        source_definition = dict(source)
        projection = _project_candidate(source_definition)
        entries.append(
            {
                "candidate_id": source_definition["candidate_id"],
                "candidate_rank": source_definition["candidate_rank"],
                "source_syntax_candidate_definition": source_definition,
                "source_syntax_candidate_definition_sha256": _canonical_sha256(
                    source_definition
                ),
                "biological_projection": projection,
                "biological_projection_sha256": _canonical_sha256(projection),
            }
        )
    return entries


def _validate_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(registry, REGISTRY_KEYS, "registry")
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise BioProjectionRegistryError("registry schema changed")
    if registry["analysis_id"] != ANALYSIS_ID:
        raise BioProjectionRegistryError("registry analysis ID changed")
    if registry["status"] != "frozen_before_syntax_outcomes_no_model_execution":
        raise BioProjectionRegistryError("registry status changed")
    if registry["freeze_date"] != FREEZE_DATE:
        raise BioProjectionRegistryError("registry freeze date changed")
    if registry["mode"] != "development" or registry["purpose"] != PURPOSE:
        raise BioProjectionRegistryError("registry mode or purpose changed")
    if registry["firewall"] != FIREWALL:
        raise BioProjectionRegistryError("registry firewall changed")
    if registry["model_calls_made_by_builder"] != 0:
        raise BioProjectionRegistryError("registry claims model execution")
    if registry["biological_core"] != BIO_CORE or BIO_CORE.endswith("\n"):
        raise BioProjectionRegistryError("biological core changed or gained a terminal newline")
    if registry["biological_core_sha256"] != _text_sha256(BIO_CORE):
        raise BioProjectionRegistryError("biological core digest mismatch")
    if registry["candidate_priority"] != list(CANDIDATE_ORDER):
        raise BioProjectionRegistryError("candidate priority changed")
    expected_source_sha = _canonical_sha256(list(SOURCE_SYNTAX_CANDIDATES))
    if registry["source_syntax_candidate_registry_sha256"] != expected_source_sha:
        raise BioProjectionRegistryError("source syntax candidate registry digest mismatch")

    entries = registry["projections"]
    expected_entries = _projection_entries()
    if not isinstance(entries, list) or len(entries) != len(CANDIDATE_ORDER):
        raise BioProjectionRegistryError("registry must contain exactly four projections")
    for index, (entry, expected) in enumerate(zip(entries, expected_entries, strict=True)):
        if not isinstance(entry, Mapping):
            raise BioProjectionRegistryError(f"projection {index} must be an object")
        _exact_keys(entry, PROJECTION_KEYS, f"projection {index}")
        for definition_key in (
            "source_syntax_candidate_definition",
            "biological_projection",
        ):
            definition = entry[definition_key]
            if not isinstance(definition, Mapping):
                raise BioProjectionRegistryError(
                    f"projection {index} {definition_key} must be an object"
                )
            _exact_keys(definition, DEFINITION_KEYS, definition_key)
        if dict(entry) != expected:
            raise BioProjectionRegistryError(
                f"projection {index} differs from its exact frozen transformation"
            )
    if [entry["candidate_id"] for entry in entries] != list(CANDIDATE_ORDER):
        raise BioProjectionRegistryError("projection order changed")
    if registry["projection_entries_canonical_sha256"] != _canonical_sha256(entries):
        raise BioProjectionRegistryError("projection entries digest mismatch")
    return dict(registry)


def build_registry() -> dict[str, Any]:
    """Return the validated winner-independent four-candidate registry."""

    entries = _projection_entries()
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "frozen_before_syntax_outcomes_no_model_execution",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "purpose": PURPOSE,
        "firewall": FIREWALL,
        "biological_core": BIO_CORE,
        "biological_core_sha256": _text_sha256(BIO_CORE),
        "candidate_priority": list(CANDIDATE_ORDER),
        "source_syntax_candidate_registry_sha256": _canonical_sha256(
            list(SOURCE_SYNTAX_CANDIDATES)
        ),
        "projection_entries_canonical_sha256": _canonical_sha256(entries),
        "projections": entries,
        "model_calls_made_by_builder": 0,
    }
    return _validate_registry(registry)


def _validate_manifest(
    manifest: Mapping[str, Any], registry: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    _exact_keys(manifest, MANIFEST_KEYS, "manifest")
    locked = _validate_registry(registry)
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise BioProjectionRegistryError("manifest schema changed")
    if manifest["analysis_id"] != ANALYSIS_ID:
        raise BioProjectionRegistryError("manifest analysis ID changed")
    if manifest["status"] != "built_not_selected_not_executed":
        raise BioProjectionRegistryError("manifest status changed")
    if manifest["freeze_date"] != FREEZE_DATE:
        raise BioProjectionRegistryError("manifest freeze date changed")
    if manifest["mode"] != "development" or manifest["purpose"] != PURPOSE:
        raise BioProjectionRegistryError("manifest mode or purpose changed")
    if manifest["firewall"] != FIREWALL or manifest["claim_scope"] != FIREWALL[
        "claim_scope"
    ]:
        raise BioProjectionRegistryError("manifest firewall or claim scope changed")
    expected_contract = {
        "candidates": 4,
        "candidate_priority": list(CANDIDATE_ORDER),
        "projections": 4,
        "winner_selected": False,
        "syntax_outcomes_observed_by_builder": False,
        "biological_items_present": False,
        "gold_labels_present": False,
        "model_calls_made_by_builder": 0,
    }
    if manifest["contract"] != expected_contract:
        raise BioProjectionRegistryError("manifest contract changed")
    if not SOURCE_SYNTAX_RUNNER.is_file():
        raise BioProjectionRegistryError("source syntax runner is missing")
    expected_provenance = {
        "construction": "deterministic_in_code_four_candidate_projection",
        "external_data_sources": [],
        "source_syntax_runner_path": _relative_path(SOURCE_SYNTAX_RUNNER),
        "source_syntax_runner_sha256": _file_sha256(SOURCE_SYNTAX_RUNNER),
        "source_syntax_candidate_registry_sha256": locked[
            "source_syntax_candidate_registry_sha256"
        ],
        "syntax_outcomes_consulted": False,
    }
    if manifest["provenance"] != expected_provenance:
        raise BioProjectionRegistryError("manifest provenance changed")
    if not registry_path.is_file() or registry_path.read_bytes() != _serialized(locked):
        raise BioProjectionRegistryError(
            "registry path does not contain the frozen deterministic bytes"
        )
    expected_artifacts = {
        "builder_path": _relative_path(Path(__file__)),
        "builder_sha256": _file_sha256(Path(__file__)),
        "registry_path": _relative_path(registry_path),
        "registry_sha256": _file_sha256(registry_path),
        "registry_canonical_sha256": _canonical_sha256(locked),
        "projection_entries_canonical_sha256": locked[
            "projection_entries_canonical_sha256"
        ],
    }
    if manifest["artifacts"] != expected_artifacts:
        raise BioProjectionRegistryError("manifest artifact locks changed")
    return dict(manifest)


def build_manifest(
    registry: Mapping[str, Any], registry_path: Path = DEFAULT_OUT
) -> dict[str, Any]:
    """Return a manifest binding the registry, builder, and source syntax code."""

    locked = _validate_registry(registry)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "built_not_selected_not_executed",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "purpose": PURPOSE,
        "claim_scope": FIREWALL["claim_scope"],
        "firewall": FIREWALL,
        "contract": {
            "candidates": 4,
            "candidate_priority": list(CANDIDATE_ORDER),
            "projections": 4,
            "winner_selected": False,
            "syntax_outcomes_observed_by_builder": False,
            "biological_items_present": False,
            "gold_labels_present": False,
            "model_calls_made_by_builder": 0,
        },
        "provenance": {
            "construction": "deterministic_in_code_four_candidate_projection",
            "external_data_sources": [],
            "source_syntax_runner_path": _relative_path(SOURCE_SYNTAX_RUNNER),
            "source_syntax_runner_sha256": _file_sha256(SOURCE_SYNTAX_RUNNER),
            "source_syntax_candidate_registry_sha256": locked[
                "source_syntax_candidate_registry_sha256"
            ],
            "syntax_outcomes_consulted": False,
        },
        "artifacts": {
            "builder_path": _relative_path(Path(__file__)),
            "builder_sha256": _file_sha256(Path(__file__)),
            "registry_path": _relative_path(registry_path),
            "registry_sha256": _file_sha256(registry_path),
            "registry_canonical_sha256": _canonical_sha256(locked),
            "projection_entries_canonical_sha256": locked[
                "projection_entries_canonical_sha256"
            ],
        },
    }
    return _validate_manifest(manifest, locked, registry_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialized(value))


def build_and_write(
    *, out: Path = DEFAULT_OUT, manifest_out: Path = DEFAULT_MANIFEST
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write deterministic registry and provenance artifacts without model calls."""

    if out.resolve() == manifest_out.resolve():
        raise BioProjectionRegistryError("registry and manifest paths must be distinct")
    registry = build_registry()
    _write_json(out, registry)
    manifest = build_manifest(registry, out)
    _write_json(manifest_out, manifest)
    return registry, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry, _ = build_and_write(out=args.out, manifest_out=args.manifest)
    print(
        f"wrote {args.out} ({len(registry['projections'])} projections) and "
        f"{args.manifest}; winner_selected=false; model_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

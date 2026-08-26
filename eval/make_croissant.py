"""Emit Croissant (MLCommons) JSON-LD metadata for the GroundBench parquet (Phase C / discoverability).

Croissant is the machine-readable dataset description that Google Dataset Search and Hugging Face index.
Reads dataset/tasks.json + dataset/groundbench.parquet (from eval/export_dataset.py) and writes
dataset/croissant.json describing the columns, license, and provenance. No API.

Run (after export_dataset.py):  python eval/make_croissant.py
"""
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset")
PARQUET = "groundbench.parquet"

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column",
    "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field", "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty", "fileSet": "cr:fileSet", "format": "cr:format",
    "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
    "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
    "rai": "http://mlcommons.org/croissant/RAI/", "recordSet": "cr:recordSet",
    "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
    "replace": "cr:replace", "sc": "https://schema.org/", "samplingRate": "cr:samplingRate",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
    "transform": "cr:transform",
}
COLUMNS = [
    ("task", "sc:Text"), ("biological_question_id", "sc:Text"),
    ("task_family_id", "sc:Text"), ("modality", "sc:Text"), ("kind", "sc:Text"),
    ("status", "sc:Text"),
    ("truth_level_code", "sc:Text"), ("target_source_kind", "sc:Text"),
    ("truth_level", "sc:Text"), ("web", "sc:Text"), ("orientation", "sc:Text"),
    ("condition", "sc:Text"), ("pair_group", "sc:Text"),
    ("split_group_id", "sc:Text"), ("split_group_scope", "sc:Text"),
    ("intervention_pair_id", "sc:Text"), ("factor_levels", "sc:Text"),
    ("reference_score", "sc:Float"), ("reference_comparability", "sc:Text"),
    ("ceiling", "sc:Float"),
    ("rep_type", "sc:Text"), ("representation", "sc:Text"), ("source_label", "sc:Integer"),
    ("target_label", "sc:Integer"), ("label", "sc:Integer"), ("source_id", "sc:Text"),
    ("entity_id", "sc:Text"), ("entity_id_scope", "sc:Text"), ("id", "sc:Text"),
]

FIELD_DESCRIPTIONS = {
    "truth_level_code": "Machine-facing target truth level; exact enum T0, T1, T2, T3, T4, or T5.",
    "target_source_kind": "Descriptive provenance category for the target source.",
    "truth_level": "Deprecated compatibility alias, exactly equal to truth_level_code.",
}


def _field(col, dtype):
    field = {
        "@type": "cr:Field",
        "@id": f"items/{col}",
        "name": col,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": PARQUET},
            "extract": {"column": col},
        },
    }
    if col in FIELD_DESCRIPTIONS:
        field["description"] = FIELD_DESCRIPTIONS[col]
    return field


def main():
    meta = json.load(open(os.path.join(DATASET, "tasks.json")))
    allowed_truth_codes = {"T0", "T1", "T2", "T3", "T4", "T5"}
    for task_id, task_meta in meta.items():
        code = task_meta.get("truth_level_code")
        if code not in allowed_truth_codes:
            raise ValueError(
                f"{task_id}: tasks.json truth_level_code={code!r} is outside T0-T5"
            )
        if task_meta.get("truth_level") != code:
            raise ValueError(
                f"{task_id}: deprecated truth_level alias must equal truth_level_code"
            )
        if not task_meta.get("target_source_kind"):
            raise ValueError(f"{task_id}: tasks.json target_source_kind is missing")
    nmod = len({m["modality"] for m in meta.values()})
    sha = hashlib.sha256(open(os.path.join(DATASET, PARQUET), "rb").read()).hexdigest()
    commit = subprocess.getoutput(f"git -C {ROOT} rev-parse --short HEAD")
    date = subprocess.getoutput(f"git -C {ROOT} log -1 --format=%cs")  # commit date, deterministic
    status = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "eval", "signal", "dataset", "pyproject.toml",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    working_tree_clean = not status.strip()
    croissant = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "GroundBench",
        "description": (
            f"GroundBench: {len(meta)} registered renderings across {nmod} modalities measuring native "
            "model output on operational representation-label targets. The companion hidden-state analysis "
            "measures a supervised decodability-native-output gap; it does not establish latent knowledge or "
            "a causal expression bottleneck. Biological-question and task-family IDs group related interfaces. "
            "Entity IDs are deterministic within their declared scope; snapshot-local row IDs are not "
            "upstream-resolvable records. Split-group scope distinguishes a real protein-family group from "
            "exact-entity proxies and unavailable biological grouping; proxy or null groups are not "
            "confirmatory dependency units. Current intervention-pair IDs are null, and factor levels are "
            "descriptive interface metadata only. Truth-level codes classify targets under the exact "
            "T0-T5 enum; target-source kind separately preserves descriptive provenance, and the legacy "
            "truth_level field is a deprecated code alias. No current task is T5. The web tag is "
            "descriptive metadata, "
            f"not a law. Data git commit: {commit}; working tree clean at export: "
            f"{str(working_tree_clean).lower()}."),
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://github.com/jang1563/grounding-atlas",
        "version": "2.1.0",
        "datePublished": date,
        "citeAs": (
            "JangKeun Kim. grounding-atlas / GroundBench: a pilot benchmark of supervised "
            "biological decodability and native model output. 2026."
        ),
        "distribution": [
            {"@type": "cr:FileObject", "@id": PARQUET, "name": PARQUET, "contentUrl": PARQUET,
             "encodingFormat": "application/x-parquet", "sha256": sha},
            {"@type": "cr:FileSet", "@id": "images", "name": "Packaged histopathology images",
             "includes": "images/**/*.png", "encodingFormat": "image/png"},
        ],
        "recordSet": [
            {"@type": "cr:RecordSet", "@id": "items", "name": "items",
             "description": "One row per (representation, property) item.",
             "field": [_field(c, dt) for c, dt in COLUMNS]},
        ],
    }
    json.dump(croissant, open(os.path.join(DATASET, "croissant.json"), "w"), indent=2)
    print(f"wrote dataset/croissant.json ({len(COLUMNS)} fields, {len(meta)} tasks, "
          f"parquet sha256 {sha[:12]}..., version {commit}, clean={working_tree_clean})")


if __name__ == "__main__":
    main()

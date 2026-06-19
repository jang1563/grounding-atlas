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
COLUMNS = [("task", "sc:Text"), ("modality", "sc:Text"), ("kind", "sc:Text"), ("web", "sc:Text"),
           ("orientation", "sc:Text"), ("ceiling", "sc:Float"), ("rep_type", "sc:Text"),
           ("representation", "sc:Text"), ("label", "sc:Integer"), ("id", "sc:Text")]


def _field(col, dtype):
    return {"@type": "cr:Field", "@id": f"items/{col}", "name": col, "dataType": dtype,
            "source": {"fileObject": {"@id": PARQUET}, "extract": {"column": col}}}


def main():
    meta = json.load(open(os.path.join(DATASET, "tasks.json")))
    nmod = len({m["modality"] for m in meta.values()})
    sha = hashlib.sha256(open(os.path.join(DATASET, PARQUET), "rb").read()).hexdigest()
    commit = subprocess.getoutput(f"git -C {ROOT} rev-parse --short HEAD")
    date = subprocess.getoutput(f"git -C {ROOT} log -1 --format=%cs")  # commit date, deterministic
    croissant = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "GroundBench",
        "description": (
            f"GroundBench: {len(meta)} tasks across {nmod} modalities measuring whether a language model "
            "can verbalize a property from a specialist representation it is shown. Each row is a "
            "(representation, verifiable binary property) item with an a-priori web-exposure tag "
            "(rich/zero/mixed). The web tag predicts the snap-verbalization floor; its mechanism is a "
            f"capability-dependent mix of token-familiarity and mapping-documentation. Data git commit: "
            f"{commit}."),
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        "url": "https://github.com/jang1563/grounding-atlas",
        "version": "1.0.0",
        "datePublished": date,
        "citeAs": ("JangKeun Kim. grounding-atlas / GroundBench: a measurement-first map of biological "
                   "content-grounding in language models. 2026."),
        "distribution": [
            {"@type": "cr:FileObject", "@id": PARQUET, "name": PARQUET, "contentUrl": PARQUET,
             "encodingFormat": "application/x-parquet", "sha256": sha},
        ],
        "recordSet": [
            {"@type": "cr:RecordSet", "@id": "items", "name": "items",
             "description": "One row per (representation, property) item.",
             "field": [_field(c, dt) for c, dt in COLUMNS]},
        ],
    }
    json.dump(croissant, open(os.path.join(DATASET, "croissant.json"), "w"), indent=2)
    print(f"wrote dataset/croissant.json ({len(COLUMNS)} fields, {len(meta)} tasks, "
          f"parquet sha256 {sha[:12]}..., version {commit})")


if __name__ == "__main__":
    main()

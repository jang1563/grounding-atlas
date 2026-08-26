"""Build the deterministic non-biological v2 syntax-selection fixture.

The bank contains direct declarations over arbitrary class words.  It exists only
to test whether a coherent two-token interface can select the class explicitly
declared in the prompt.  It contains no biological observations, labels, facts,
or model outputs, and it cannot support biology, knowledge, or activation claims.

Every class pair contributes exactly two items: one declaring the registered
positive class and one declaring the registered negative class.  This builder is
pure apart from writing its requested JSON artifacts and makes zero model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ID = "coherent-readout-v2-syntax-selection-bank-v1"
FIXTURE_SCHEMA = "coherent-readout-v2-syntax-bank-v1"
ITEM_SCHEMA = "coherent-readout-v2-syntax-item-v1"
MANIFEST_SCHEMA = "coherent-readout-v2-syntax-bank-manifest-v1"
FREEZE_DATE = "2026-08-02"
PURPOSE = "syntax_selection_only"
DECLARATION_TEMPLATE = "The declared class is {declared_class}."

FIREWALL = {
    "scope": PURPOSE,
    "biology_inference": "forbidden",
    "knowledge_inference": "forbidden",
    "activation_inference": "forbidden",
}

PAIR_DEFINITIONS = (
    ("pair_01_amber_cobalt", "amber", "cobalt"),
    ("pair_02_cedar_maple", "cedar", "maple"),
    ("pair_03_north_south", "north", "south"),
    ("pair_04_circle_square", "circle", "square"),
    ("pair_05_copper_silver", "copper", "silver"),
    ("pair_06_violin_trumpet", "violin", "trumpet"),
    ("pair_07_river_mountain", "river", "mountain"),
    ("pair_08_winter_summer", "winter", "summer"),
)

DEFAULT_OUT = Path(__file__).with_name("coherent_readout_v2_syntax_bank.json")
DEFAULT_MANIFEST = Path(__file__).with_name(
    "coherent_readout_v2_syntax_bank.manifest.json"
)

FIXTURE_KEYS = {
    "schema_version",
    "analysis_id",
    "mode",
    "purpose",
    "firewall",
    "inferential_unit",
    "pair_registry",
    "items",
    "model_calls_made_by_builder",
}
PAIR_KEYS = {"pair_id", "cluster_id", "positive_class", "negative_class"}
ITEM_KEYS = {
    "schema_version",
    "item_id",
    "pair_id",
    "cluster_id",
    "positive_class",
    "negative_class",
    "declared_class",
    "truth_polarity",
    "declaration_text",
    "declaration_sha256",
    "firewall",
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


class SyntaxBankError(ValueError):
    """Raised when the v2 syntax bank violates its frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SyntaxBankError(
            f"{label} schema mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _pair_registry() -> list[dict[str, str]]:
    return [
        {
            "pair_id": pair_id,
            "cluster_id": pair_id,
            "positive_class": positive_class,
            "negative_class": negative_class,
        }
        for pair_id, positive_class, negative_class in PAIR_DEFINITIONS
    ]


def _build_item(
    pair_id: str,
    positive_class: str,
    negative_class: str,
    truth_polarity: str,
) -> dict[str, Any]:
    if truth_polarity not in {"positive", "negative"}:
        raise SyntaxBankError(f"unknown truth polarity: {truth_polarity}")
    declared_class = (
        positive_class if truth_polarity == "positive" else negative_class
    )
    declaration_text = DECLARATION_TEMPLATE.format(declared_class=declared_class)
    return {
        "schema_version": ITEM_SCHEMA,
        "item_id": f"syntax:{pair_id}:{declared_class}",
        "pair_id": pair_id,
        "cluster_id": pair_id,
        "positive_class": positive_class,
        "negative_class": negative_class,
        "declared_class": declared_class,
        "truth_polarity": truth_polarity,
        "declaration_text": declaration_text,
        "declaration_sha256": hashlib.sha256(
            declaration_text.encode("utf-8")
        ).hexdigest(),
        "firewall": FIREWALL,
    }


def _validate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(fixture, FIXTURE_KEYS, "fixture")
    if fixture["schema_version"] != FIXTURE_SCHEMA:
        raise SyntaxBankError("fixture schema changed")
    if fixture["analysis_id"] != ANALYSIS_ID:
        raise SyntaxBankError("fixture analysis ID changed")
    if fixture["mode"] != "development" or fixture["purpose"] != PURPOSE:
        raise SyntaxBankError("fixture mode or purpose changed")
    if fixture["inferential_unit"] != "pair_cluster":
        raise SyntaxBankError("fixture inferential unit changed")
    if fixture["firewall"] != FIREWALL:
        raise SyntaxBankError("fixture firewall changed")
    if fixture["model_calls_made_by_builder"] != 0:
        raise SyntaxBankError("fixture claims model execution")

    registry = fixture["pair_registry"]
    expected_registry = _pair_registry()
    if registry != expected_registry:
        raise SyntaxBankError("pair registry or its deterministic order changed")
    if any(set(pair) != PAIR_KEYS for pair in registry):
        raise SyntaxBankError("pair registry schema changed")

    items = fixture["items"]
    if not isinstance(items, list) or len(items) != 2 * len(PAIR_DEFINITIONS):
        raise SyntaxBankError("fixture must contain exactly 16 items")
    item_ids = [item.get("item_id") for item in items if isinstance(item, Mapping)]
    if len(item_ids) != len(items) or item_ids != sorted(item_ids):
        raise SyntaxBankError("items must be mappings sorted by unique item_id")
    if len(set(item_ids)) != len(item_ids):
        raise SyntaxBankError("item IDs are not unique")

    registered = {pair["pair_id"]: pair for pair in expected_registry}
    coverage: Counter[tuple[str, str]] = Counter()
    for item in items:
        _exact_keys(item, ITEM_KEYS, f"item {item.get('item_id')}")
        if item["schema_version"] != ITEM_SCHEMA:
            raise SyntaxBankError("item schema changed")
        pair_id = item["pair_id"]
        pair = registered.get(pair_id)
        if pair is None:
            raise SyntaxBankError(f"unknown pair ID: {pair_id}")
        if item["cluster_id"] != pair_id:
            raise SyntaxBankError("cluster ID must equal its pair ID")
        if item["positive_class"] != pair["positive_class"] or item[
            "negative_class"
        ] != pair["negative_class"]:
            raise SyntaxBankError("item class orientation changed")
        polarity = item["truth_polarity"]
        if polarity not in {"positive", "negative"}:
            raise SyntaxBankError("truth polarity must be positive or negative")
        expected_declared = pair[f"{polarity}_class"]
        if item["declared_class"] != expected_declared:
            raise SyntaxBankError("declared class disagrees with truth polarity")
        expected_item_id = f"syntax:{pair_id}:{expected_declared}"
        if item["item_id"] != expected_item_id:
            raise SyntaxBankError("item ID does not bind pair and declared class")
        expected_text = DECLARATION_TEMPLATE.format(declared_class=expected_declared)
        if item["declaration_text"] != expected_text:
            raise SyntaxBankError("declaration text changed")
        expected_digest = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
        if item["declaration_sha256"] != expected_digest:
            raise SyntaxBankError("declaration digest mismatch")
        if item["firewall"] != FIREWALL:
            raise SyntaxBankError("item escaped the syntax-selection firewall")
        coverage[(pair_id, polarity)] += 1

    expected_coverage = Counter(
        (pair_id, polarity)
        for pair_id, _, _ in PAIR_DEFINITIONS
        for polarity in ("positive", "negative")
    )
    if coverage != expected_coverage:
        raise SyntaxBankError("each pair must represent both truth polarities once")
    return dict(fixture)


def build_fixture() -> dict[str, Any]:
    """Return the validated deterministic syntax-selection fixture."""

    items = sorted(
        (
            _build_item(pair_id, positive_class, negative_class, polarity)
            for pair_id, positive_class, negative_class in PAIR_DEFINITIONS
            for polarity in ("positive", "negative")
        ),
        key=lambda item: item["item_id"],
    )
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "mode": "development",
        "purpose": PURPOSE,
        "firewall": FIREWALL,
        "inferential_unit": "pair_cluster",
        "pair_registry": _pair_registry(),
        "items": items,
        "model_calls_made_by_builder": 0,
    }
    return _validate_fixture(fixture)


def _validate_manifest(
    manifest: Mapping[str, Any], fixture: Mapping[str, Any], fixture_path: Path
) -> dict[str, Any]:
    _exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise SyntaxBankError("manifest schema changed")
    if manifest["analysis_id"] != ANALYSIS_ID:
        raise SyntaxBankError("manifest analysis ID changed")
    if manifest["status"] != "built_not_executed":
        raise SyntaxBankError("manifest status changed")
    if manifest["mode"] != "development" or manifest["purpose"] != PURPOSE:
        raise SyntaxBankError("manifest mode or purpose changed")
    if manifest["firewall"] != FIREWALL:
        raise SyntaxBankError("manifest firewall changed")
    if manifest["contract"]["model_calls_made_by_builder"] != 0:
        raise SyntaxBankError("manifest claims model execution")
    if manifest["artifacts"]["builder_sha256"] != _file_sha256(Path(__file__)):
        raise SyntaxBankError("manifest does not bind the builder")
    if manifest["artifacts"]["fixture_sha256"] != _file_sha256(fixture_path):
        raise SyntaxBankError("manifest does not bind the fixture bytes")
    if manifest["artifacts"]["fixture_canonical_sha256"] != _canonical_sha256(
        fixture
    ):
        raise SyntaxBankError("manifest does not bind the fixture content")
    return dict(manifest)


def build_manifest(
    fixture: Mapping[str, Any], fixture_path: Path = DEFAULT_OUT
) -> dict[str, Any]:
    """Return a manifest binding the validated fixture and its deterministic source."""

    locked = _validate_fixture(fixture)
    if not fixture_path.is_file():
        raise SyntaxBankError(f"fixture artifact does not exist: {fixture_path}")
    on_disk = json.loads(fixture_path.read_text(encoding="utf-8"))
    if on_disk != locked:
        raise SyntaxBankError("fixture path does not contain the supplied fixture")
    pair_registry = _pair_registry()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "built_not_executed",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "purpose": PURPOSE,
        "claim_scope": (
            "syntax_selection_only_no_biology_knowledge_or_activation_inference"
        ),
        "firewall": FIREWALL,
        "contract": {
            "pairs": len(PAIR_DEFINITIONS),
            "inferential_unit": "pair_cluster",
            "inferential_units": len(PAIR_DEFINITIONS),
            "items": len(locked["items"]),
            "items_per_pair": 2,
            "truth_polarity_counts": {"negative": 8, "positive": 8},
            "model_calls_made_by_builder": 0,
        },
        "provenance": {
            "source_type": "deterministic_in_code_direct_declarations",
            "external_data_sources": [],
            "pair_registry_constant": "PAIR_DEFINITIONS",
            "pair_registry_canonical_sha256": _canonical_sha256(pair_registry),
            "declaration_template": DECLARATION_TEMPLATE,
            "declaration_encoding": "utf-8",
        },
        "artifacts": {
            "builder_path": _relative_path(Path(__file__)),
            "builder_sha256": _file_sha256(Path(__file__)),
            "fixture_path": _relative_path(fixture_path),
            "fixture_sha256": _file_sha256(fixture_path),
            "fixture_canonical_sha256": _canonical_sha256(locked),
        },
    }
    return _validate_manifest(manifest, locked, fixture_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_and_write(
    *, out: Path = DEFAULT_OUT, manifest_out: Path = DEFAULT_MANIFEST
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the fixture and a manifest that authenticates its exact bytes."""

    if out.resolve() == manifest_out.resolve():
        raise SyntaxBankError("fixture and manifest paths must be distinct")
    fixture = build_fixture()
    _write_json(out, fixture)
    manifest = build_manifest(fixture, out)
    _write_json(manifest_out, manifest)
    return fixture, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture, manifest = build_and_write(out=args.out, manifest_out=args.manifest)
    print(
        f"wrote {args.out} ({len(fixture['items'])} items) and {args.manifest}; "
        f"scope={manifest['firewall']['scope']}; model_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

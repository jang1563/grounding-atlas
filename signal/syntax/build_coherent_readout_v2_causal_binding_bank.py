"""Build the frozen lexical holdout for the v2 causal binding experiment.

The bank is non-biological and contains only explicit class declarations.  It is
prospective with respect to every activation-patching outcome and every native
model output on these lexical pairs.  The original syntax bank is referenced
only as a disjoint layer-localization set; none of its pairs contribute to the
held-out intervention estimands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ID = "coherent-readout-v2-causal-binding-bank-v1"
FIXTURE_SCHEMA = "coherent-readout-v2-causal-binding-bank-v1"
ITEM_SCHEMA = "coherent-readout-v2-causal-binding-item-v1"
MANIFEST_SCHEMA = "coherent-readout-v2-causal-binding-bank-manifest-v1"
FREEZE_DATE = "2026-08-02"
PURPOSE = "causal_syntax_binding_holdout"
DECLARATION_TEMPLATE = "The declared class is {declared_class}."

DISCOVERY_FIXTURE = Path(__file__).with_name(
    "coherent_readout_v2_syntax_bank.json"
)
DISCOVERY_FIXTURE_SHA256 = (
    "d00e27d9e4130ff7d0d4ab32b1e26d31f40482cb1f4654204fd8a748ed06f4f8"
)

FIREWALL = {
    "scope": PURPOSE,
    "biology_inference": "forbidden",
    "latent_knowledge_inference": "forbidden",
    "activation_gap_inference": "forbidden",
    "physical_law_inference": "forbidden",
    "model_family_generalization": "forbidden",
}

# These 48 pairs are fully disjoint from the eight outcome-exposed discovery
# pairs. Their order groups prompt-length signatures and freezes 24 reciprocal
# unrelated-pair control dyads.
PAIR_DEFINITIONS = (
    ("holdout_02_oak_pine", "oak", "pine"),
    ("holdout_03_east_west", "east", "west"),
    ("holdout_05_brass_gold", "brass", "gold"),
    ("holdout_07_ocean_forest", "ocean", "forest"),
    ("holdout_08_autumn_spring", "autumn", "spring"),
    ("holdout_09_marble_granite", "marble", "granite"),
    ("holdout_10_comet_planet", "comet", "planet"),
    ("holdout_12_linen_velvet", "linen", "velvet"),
    ("holdout_15_dawn_dusk", "dawn", "dusk"),
    ("holdout_16_oval_diamond", "oval", "diamond"),
    ("holdout_17_bronze_platinum", "bronze", "platinum"),
    ("holdout_24_cotton_silk", "cotton", "silk"),
    ("holdout_29_iron_tin", "iron", "tin"),
    ("holdout_31_canyon_valley", "canyon", "valley"),
    ("holdout_32_frost_thunder", "frost", "thunder"),
    ("holdout_33_slate_limestone", "slate", "limestone"),
    ("holdout_36_wool_satin", "wool", "satin"),
    ("holdout_39_sunrise_sunset", "sunrise", "sunset"),
    ("holdout_40_sphere_cube", "sphere", "cube"),
    ("holdout_41_nickel_zinc", "nickel", "zinc"),
    ("holdout_44_breeze_storm", "breeze", "storm"),
    ("holdout_48_denim_suede", "denim", "suede"),
    ("holdout_01_coral_indigo", "coral", "indigo"),
    ("holdout_04_triangle_hexagon", "triangle", "hexagon"),
    ("holdout_13_ruby_sapphire", "ruby", "sapphire"),
    ("holdout_18_piano_clarinet", "piano", "clarinet"),
    ("holdout_19_desert_meadow", "desert", "meadow"),
    ("holdout_21_quartz_basalt", "quartz", "basalt"),
    ("holdout_22_asteroid_nebula", "asteroid", "nebula"),
    ("holdout_23_eagle_badger", "eagle", "badger"),
    ("holdout_27_zenith_nadir", "zenith", "nadir"),
    ("holdout_34_galaxy_pulsar", "galaxy", "pulsar"),
    ("holdout_46_meteor_quasar", "meteor", "quasar"),
    ("holdout_11_falcon_otter", "falcon", "otter"),
    ("holdout_14_birch_willow", "birch", "willow"),
    ("holdout_28_pentagon_octagon", "pentagon", "octagon"),
    ("holdout_35_raven_beaver", "raven", "beaver"),
    ("holdout_38_acacia_poplar", "acacia", "poplar"),
    ("holdout_42_banjo_bassoon", "banjo", "bassoon"),
    ("holdout_43_lagoon_tundra", "lagoon", "tundra"),
    ("holdout_45_obsidian_sandstone", "obsidian", "sandstone"),
    ("holdout_47_heron_lynx", "heron", "lynx"),
    ("holdout_06_cello_flute", "cello", "flute"),
    ("holdout_20_monsoon_drought", "monsoon", "drought"),
    ("holdout_25_scarlet_turquoise", "scarlet", "turquoise"),
    ("holdout_26_spruce_elm", "spruce", "elm"),
    ("holdout_37_magenta_teal", "magenta", "teal"),
    ("holdout_30_harp_oboe", "harp", "oboe"),
)

DEFAULT_OUT = Path(__file__).with_name(
    "coherent_readout_v2_causal_binding_bank.json"
)
DEFAULT_MANIFEST = Path(__file__).with_name(
    "coherent_readout_v2_causal_binding_bank.manifest.json"
)


class CausalBindingBankError(ValueError):
    """Raised when the frozen lexical holdout contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_discovery_reference(path: Path = DISCOVERY_FIXTURE) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != DISCOVERY_FIXTURE_SHA256:
        raise CausalBindingBankError("discovery syntax fixture does not match its frozen hash")
    value = json.loads(path.read_text(encoding="utf-8"))
    pairs = value.get("pair_registry")
    items = value.get("items")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise CausalBindingBankError("discovery reference must contain eight pairs")
    if not isinstance(items, list) or len(items) != 16:
        raise CausalBindingBankError("discovery reference must contain sixteen items")
    return {
        "path": relative_path(path),
        "file_sha256": DISCOVERY_FIXTURE_SHA256,
        "canonical_sha256": canonical_sha256(value),
        "pair_ids": [pair["pair_id"] for pair in pairs],
        "class_words": sorted(
            {
                word
                for pair in pairs
                for word in (pair["positive_class"], pair["negative_class"])
            }
        ),
        "pair_count": len(pairs),
        "item_count": len(items),
        "role": "layer_localization_only_no_holdout_inference",
    }


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


def _item_id(pair_id: str, declared_class: str) -> str:
    return f"causal-binding:{pair_id}:{declared_class}"


def _build_items() -> list[dict[str, Any]]:
    pair_ids = [pair_id for pair_id, _, _ in PAIR_DEFINITIONS]
    by_pair = {
        pair_id: (positive_class, negative_class)
        for pair_id, positive_class, negative_class in PAIR_DEFINITIONS
    }
    unrelated_pair = {
        pair_id: pair_ids[index + 1] if index % 2 == 0 else pair_ids[index - 1]
        for index, pair_id in enumerate(pair_ids)
    }
    items: list[dict[str, Any]] = []
    for pair_id, positive_class, negative_class in PAIR_DEFINITIONS:
        for polarity, declared_class, opposite_class in (
            ("positive", positive_class, negative_class),
            ("negative", negative_class, positive_class),
        ):
            unrelated_pair_id = unrelated_pair[pair_id]
            unrelated_classes = by_pair[unrelated_pair_id]
            unrelated_class = unrelated_classes[0 if polarity == "positive" else 1]
            control_cluster_id = "control_dyad:" + ":".join(
                sorted((pair_id, unrelated_pair_id))
            )
            declaration = DECLARATION_TEMPLATE.format(declared_class=declared_class)
            items.append(
                {
                    "schema_version": ITEM_SCHEMA,
                    "item_id": _item_id(pair_id, declared_class),
                    "pair_id": pair_id,
                    "cluster_id": pair_id,
                    "positive_class": positive_class,
                    "negative_class": negative_class,
                    "declared_class": declared_class,
                    "truth_polarity": polarity,
                    "declaration_text": declaration,
                    "declaration_sha256": text_sha256(declaration),
                    "same_pair_counterfactual_item_id": _item_id(
                        pair_id, opposite_class
                    ),
                    "unrelated_same_polarity_item_id": _item_id(
                        unrelated_pair_id, unrelated_class
                    ),
                    "unrelated_control_cluster_id": control_cluster_id,
                    "firewall": FIREWALL,
                }
            )
    return sorted(items, key=lambda item: item["item_id"])


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "analysis_id",
        "freeze_date",
        "mode",
        "purpose",
        "firewall",
        "inferential_unit",
        "discovery_reference",
        "pair_registry",
        "items",
        "control_derangement",
        "outcome_exposure",
        "model_calls_made_by_builder",
    }
    if set(value) != expected_keys:
        raise CausalBindingBankError("fixture schema changed")
    if value["schema_version"] != FIXTURE_SCHEMA or value["analysis_id"] != ANALYSIS_ID:
        raise CausalBindingBankError("fixture identity changed")
    if value["freeze_date"] != FREEZE_DATE:
        raise CausalBindingBankError("freeze date changed")
    if value["mode"] != "development" or value["purpose"] != PURPOSE:
        raise CausalBindingBankError("fixture mode or purpose changed")
    if value["firewall"] != FIREWALL or value["inferential_unit"] != "lexical_pair":
        raise CausalBindingBankError("fixture firewall or inferential unit changed")
    if value["model_calls_made_by_builder"] != 0:
        raise CausalBindingBankError("builder claims model execution")
    if value["outcome_exposure"] != {
        "prior_syntax_behavior_used_to_define_contrast": True,
        "holdout_native_behavior_observed": False,
        "intervention_outcomes_observed": False,
    }:
        raise CausalBindingBankError("outcome-exposure declaration changed")

    discovery = _load_discovery_reference()
    if value["discovery_reference"] != discovery:
        raise CausalBindingBankError("discovery reference changed")
    registry = _pair_registry()
    if value["pair_registry"] != registry:
        raise CausalBindingBankError("pair registry changed")

    class_words = [
        word
        for _, positive_class, negative_class in PAIR_DEFINITIONS
        for word in (positive_class, negative_class)
    ]
    if len(class_words) != len(set(class_words)):
        raise CausalBindingBankError("holdout class words must be globally unique")
    if set(class_words) & set(discovery["class_words"]):
        raise CausalBindingBankError("holdout class words overlap discovery")

    items = value["items"]
    expected_items = _build_items()
    if items != expected_items or len(items) != 96:
        raise CausalBindingBankError("held-out items changed")
    if [item["item_id"] for item in items] != sorted(item["item_id"] for item in items):
        raise CausalBindingBankError("items are not deterministically sorted")
    item_ids = {item["item_id"] for item in items}
    coverage = Counter((item["pair_id"], item["truth_polarity"]) for item in items)
    expected_coverage = Counter(
        (pair_id, polarity)
        for pair_id, _, _ in PAIR_DEFINITIONS
        for polarity in ("positive", "negative")
    )
    if coverage != expected_coverage:
        raise CausalBindingBankError("each pair must contain both truth polarities")
    for item in items:
        if item["same_pair_counterfactual_item_id"] not in item_ids:
            raise CausalBindingBankError("same-pair control does not resolve")
        if item["unrelated_same_polarity_item_id"] not in item_ids:
            raise CausalBindingBankError("unrelated control does not resolve")
        if item["firewall"] != FIREWALL:
            raise CausalBindingBankError("item escaped the firewall")

    expected_derangement = [
        {
            "source_pair_id": pair_id,
            "unrelated_pair_id": PAIR_DEFINITIONS[
                index + 1 if index % 2 == 0 else index - 1
            ][0],
            "control_cluster_id": "control_dyad:"
            + ":".join(
                sorted(
                    (
                        pair_id,
                        PAIR_DEFINITIONS[
                            index + 1 if index % 2 == 0 else index - 1
                        ][0],
                    )
                )
            ),
        }
        for index, (pair_id, _, _) in enumerate(PAIR_DEFINITIONS)
    ]
    if value["control_derangement"] != expected_derangement:
        raise CausalBindingBankError("unrelated-pair derangement changed")
    return dict(value)


def build_fixture() -> dict[str, Any]:
    registry = _pair_registry()
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "purpose": PURPOSE,
        "firewall": FIREWALL,
        "inferential_unit": "lexical_pair",
        "discovery_reference": _load_discovery_reference(),
        "pair_registry": registry,
        "items": _build_items(),
        "control_derangement": [
            {
                "source_pair_id": pair_id,
                "unrelated_pair_id": PAIR_DEFINITIONS[
                    index + 1 if index % 2 == 0 else index - 1
                ][0],
                "control_cluster_id": "control_dyad:"
                + ":".join(
                    sorted(
                        (
                            pair_id,
                            PAIR_DEFINITIONS[
                                index + 1 if index % 2 == 0 else index - 1
                            ][0],
                        )
                    )
                ),
            }
            for index, (pair_id, _, _) in enumerate(PAIR_DEFINITIONS)
        ],
        "outcome_exposure": {
            "prior_syntax_behavior_used_to_define_contrast": True,
            "holdout_native_behavior_observed": False,
            "intervention_outcomes_observed": False,
        },
        "model_calls_made_by_builder": 0,
    }
    return validate_fixture(fixture)


def build_manifest(fixture: Mapping[str, Any], fixture_path: Path) -> dict[str, Any]:
    locked = validate_fixture(fixture)
    if not fixture_path.is_file():
        raise CausalBindingBankError("fixture must be written before its manifest")
    if json.loads(fixture_path.read_text(encoding="utf-8")) != locked:
        raise CausalBindingBankError("fixture path does not contain the supplied fixture")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "status": "FROZEN_NO_HOLDOUT_FORWARD",
        "freeze_date": FREEZE_DATE,
        "mode": "development",
        "purpose": PURPOSE,
        "claim_scope": (
            "prompt_protocol_causal_transfer_only_no_biology_latent_knowledge_"
            "activation_gap_physical_law_or_model_family_claim"
        ),
        "firewall": FIREWALL,
        "contract": {
            "holdout_pairs": len(PAIR_DEFINITIONS),
            "holdout_items": len(locked["items"]),
            "items_per_pair": 2,
            "inferential_unit": "lexical_pair",
            "fixed_unrelated_control": "reciprocal_pair_dyad_same_truth_polarity",
            "unrelated_control_inferential_units": len(PAIR_DEFINITIONS) // 2,
            "model_calls_made_by_builder": 0,
        },
        "provenance": {
            "source_type": "deterministic_in_code_direct_declarations",
            "external_data_sources": [],
            "discovery_reference": locked["discovery_reference"],
            "pair_registry_canonical_sha256": canonical_sha256(
                locked["pair_registry"]
            ),
            "control_derangement_canonical_sha256": canonical_sha256(
                locked["control_derangement"]
            ),
        },
        "artifacts": {
            "builder_path": relative_path(Path(__file__)),
            "builder_sha256": file_sha256(Path(__file__)),
            "fixture_path": relative_path(fixture_path),
            "fixture_sha256": file_sha256(fixture_path),
            "fixture_canonical_sha256": canonical_sha256(locked),
        },
    }


def _write_frozen_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise CausalBindingBankError(f"refusing to overwrite frozen artifact: {path}")
    path.write_bytes(payload)


def build_and_write(
    out: Path = DEFAULT_OUT, manifest_out: Path = DEFAULT_MANIFEST
) -> tuple[dict[str, Any], dict[str, Any]]:
    if out.resolve() == manifest_out.resolve():
        raise CausalBindingBankError("fixture and manifest paths must differ")
    fixture = build_fixture()
    _write_frozen_json(out, fixture)
    manifest = build_manifest(fixture, out)
    _write_frozen_json(manifest_out, manifest)
    return fixture, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    fixture, manifest = build_and_write(args.out, args.manifest_out)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "holdout_pairs": len(fixture["pair_registry"]),
                "holdout_items": len(fixture["items"]),
                "fixture_canonical_sha256": canonical_sha256(fixture),
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

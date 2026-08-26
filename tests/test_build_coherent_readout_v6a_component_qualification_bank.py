from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "signal"
    / "syntax"
    / "coherent_readout_v6a_component_qualification_bank.json"
)
MANIFEST = FIXTURE.with_suffix(".manifest.json")
BUILDER = (
    ROOT
    / "signal"
    / "syntax"
    / "build_coherent_readout_v6a_component_qualification_bank.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("v6a_qualification_builder_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_fixture_rebuilds_exactly(builder) -> None:
    observed = _load(FIXTURE)
    rebuilt = builder.build_fixture()

    assert observed == rebuilt
    assert builder.canonical_sha256(observed) == _load(MANIFEST)[
        "fixture_canonical_sha256"
    ]
    assert builder.file_sha256(FIXTURE) == _load(MANIFEST)["fixture_file_sha256"]
    assert builder.file_sha256(Path(builder.__file__)) == _load(MANIFEST)[
        "builder_file_sha256"
    ]


def test_qualification_is_component_only_and_prompt_unique() -> None:
    fixture = _load(FIXTURE)
    cells = fixture["cells"]

    assert len(cells) == 384
    assert Counter(cell["family"] for cell in cells) == {
        "property_retrieval": 256,
        "codebook_lookup": 128,
    }
    assert all(cell["family"] != "composition" for cell in cells)
    assert len({cell["cell_id"] for cell in cells}) == len(cells)
    assert len({cell["prompt_text"] for cell in cells}) == len(cells)
    assert fixture["calibration_firewall"]["composition_calls"] == 0
    assert fixture["model_calls_made_by_builder"] == 0
    assert fixture["tokenizer_calls_made_by_builder"] == 0


def test_symbols_are_globally_unique_and_firewalled(builder) -> None:
    fixture = _load(FIXTURE)
    symbols = [symbol for world in fixture["worlds"] for symbol in world["symbols"]]

    assert len(symbols) == 32
    assert len(set(symbols)) == 32
    assert set(symbols) == set(builder.QUALIFICATION_SYMBOLS)
    assert not set(symbols) & builder.V6A_RESERVED_SYMBOLS
    assert not set(symbols) & builder.PRIOR_ASCII_SYMBOLS


def test_retrieval_topology_moves_only_registered_blocks() -> None:
    fixture = _load(FIXTURE)
    cells = [cell for cell in fixture["cells"] if cell["family"] == "property_retrieval"]
    for cell in cells:
        text = cell["prompt_text"]
        answer = text.index("VALID OUTPUTS")
        task = text.index("TASK:")
        facts = text.index("LABELED FACTS")
        q = cell["factors"]["q"]
        a = cell["factors"]["a"]
        assert (task < facts) is (q == -1)
        assert (answer < facts) is (a == -1)


def test_answer_ledger_reconstructs_from_worlds() -> None:
    fixture = _load(FIXTURE)
    worlds = {world["world_id"]: world for world in fixture["worlds"]}
    for cell in fixture["cells"]:
        world = worlds[cell["world_id"]]
        p = cell["factors"]["p"]
        prop = world["property_symbols"]["negative" if p == -1 else "positive"]
        if cell["family"] == "property_retrieval":
            expected = prop
        else:
            expected = world["codebooks"][cell["mapping_id"]][prop]
        assert cell["correct_answer"] == expected
        assert cell["correct_answer"] in cell["answer_options"]
        assert cell["distractor_answer"] in cell["answer_options"]
        assert cell["correct_answer"] != cell["distractor_answer"]


def test_write_fixture_is_deterministic_in_fresh_directory(builder, tmp_path: Path) -> None:
    output = tmp_path / "qualification.json"
    fixture_path, manifest_path = builder.write_fixture(output)

    assert _load(fixture_path) == _load(FIXTURE)
    manifest = _load(manifest_path)
    assert manifest["fixture_file_sha256"] == builder.file_sha256(fixture_path)
    assert manifest["composition_cell_count"] == 0
    assert manifest["model_calls"] == 0

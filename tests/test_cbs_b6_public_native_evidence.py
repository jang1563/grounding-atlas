from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from eval import cbs_b6_public_native_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "signal" / "dms" / "cbs_b6_pair_registry.v1.json"
LOCK_PATH = ROOT / "signal" / "dms" / "cbs_b6_public_native_evidence_lock.v1.json"


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def test_persisted_public_evidence_lock_is_exact_and_deterministic() -> None:
    registry = _registry()
    built = evidence.build_cbs_b6_public_native_evidence_lock(registry)
    persisted = evidence.load_cbs_b6_public_native_evidence_lock(
        LOCK_PATH,
        pair_registry=registry,
    )

    assert persisted == built
    assert persisted["evidence_ceiling"] == evidence.EVIDENCE_CEILING
    assert persisted["native_replay_ready"] is False
    assert persisted["uncertainty_ready"] is False
    assert persisted["outcome_status"] == "not_derived"
    assert persisted["confirmatory_eligible"] is False
    assert persisted["automatic_promotion"] is False
    assert persisted["lock_sha256"] == evidence.lock_sha256(persisted)


def test_workbook_lock_records_only_ordinal_partial_sample_evidence() -> None:
    lock = evidence.build_cbs_b6_public_native_evidence_lock(_registry())
    workbook = lock["supplement_workbook"]
    sample_evidence = lock["public_sample_role_evidence"]
    low, high = workbook["sheets"][:2]

    assert workbook["body_bytes"] == 10_005_946
    assert workbook["body_sha256"] == "878365975f62da42c4113214958d4bd60ba7f4d3fe269eee1a722465ca6470aa"
    assert low["ordinal_replicate_columns_per_measurement_role"] == 8
    assert high["ordinal_replicate_columns_per_measurement_role"] == 4
    assert workbook["defined_name_count"] == 0
    assert workbook["formula_cell_count"] == 0
    assert workbook["external_link_count"] == 0
    assert workbook["custom_xml_part_count"] == 0
    assert sample_evidence["status"] == "partial_ordinal_columns_only"
    assert "authenticated_facts" not in sample_evidence
    assert "source_reported_and_locally_inspected_facts" in sample_evidence
    assert sample_evidence["pairing_claim"] is False
    assert sample_evidence["column_order_inference_allowed"] is False
    assert "cross_condition_independence_or_pairing" in sample_evidence["unresolved_bindings"]


def test_public_repository_fixture_cannot_be_promoted_to_paper_parameter_sheet() -> None:
    lock = evidence.build_cbs_b6_public_native_evidence_lock(_registry())
    fixture = lock["paper_linked_software"]["tileseq_mave"]["cbs_repository_test_fixture"]

    assert fixture["gene"] == "CBS"
    assert fixture["tile_count"] == 15
    assert fixture["declared_replicates_per_generic_role"] == 2
    assert "not_authenticated" in fixture["evidence_status"]
    admission = lock["joint_bootstrap_admission"]
    assert admission["method_contract_definition_sha256"] == (evidence.uncertainty.method_contract_definition_sha256())
    assert admission["seed"] == evidence.uncertainty.JOINT_BOOTSTRAP_SEED
    assert admission["paper_reported_biological_cultures_per_condition"] == 2
    assert admission["minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci"] == 8
    assert admission["public_evidence_meets_independent_block_minimum"] is False
    assert admission["claim_bearing_percentile_ci_allowed_at_n_equals_2"] is False
    assert admission["public_evidence_contains_only_post_count_normalized_frequencies"] is True
    assert lock["joint_bootstrap_admission"]["public_evidence_sufficient_to_construct_resampling_graph"] is False
    assert lock["joint_bootstrap_admission"]["execution_allowed"] is False


def test_public_lock_is_bound_to_uncertainty_method_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)
    monkeypatch.setattr(
        evidence.uncertainty,
        "method_contract_definition_sha256",
        lambda: "f" * 64,
    )

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="differs"):
        evidence.validate_cbs_b6_public_native_evidence_lock(
            lock,
            pair_registry=registry,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("public_sample_role_evidence", "pairing_claim"), True),
        (
            (
                "joint_bootstrap_admission",
                "public_evidence_sufficient_to_construct_resampling_graph",
            ),
            True,
        ),
        (("native_replay_ready",), True),
        (("uncertainty_ready",), True),
        (("outcome_status",), "derived"),
        (("confirmatory_eligible",), True),
        (
            (
                "paper_linked_software",
                "tileseq_mave",
                "cbs_repository_test_fixture",
                "evidence_status",
            ),
            "authenticated_paper_parameter_sheet",
        ),
    ],
)
def test_coherent_rehash_cannot_promote_public_evidence(
    path: tuple[str, ...],
    value: object,
) -> None:
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)
    target = lock
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    lock["lock_sha256"] = evidence.lock_sha256(lock)

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="differs"):
        evidence.validate_cbs_b6_public_native_evidence_lock(
            lock,
            pair_registry=registry,
        )


def test_exact_supplement_bytes_are_optional_but_replayed_when_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"small exact supplement fixture"
    monkeypatch.setattr(evidence, "SUPPLEMENT_BYTES", len(body))
    monkeypatch.setattr(evidence, "SUPPLEMENT_SHA256", evidence._bytes_sha256(body))
    monkeypatch.setattr(
        evidence,
        "inspect_supplement_workbook_bytes",
        lambda _: evidence._expected_workbook_structure(),
    )
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)

    evidence.validate_cbs_b6_public_native_evidence_lock(
        lock,
        pair_registry=registry,
        supplement_workbook_bytes=body,
    )
    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="identity differs"):
        evidence.validate_cbs_b6_public_native_evidence_lock(
            lock,
            pair_registry=registry,
            supplement_workbook_bytes=body[:-1],
        )
    with pytest.raises(TypeError, match="exact bytes"):
        evidence.validate_cbs_b6_public_native_evidence_lock(
            lock,
            pair_registry=registry,
            supplement_workbook_bytes=bytearray(body),
        )


def test_ooxml_structural_inspector_recomputes_headers_and_package_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence, "HEADER_ROW_COUNTS", {"test sheet": 1})
    workbook_xml = f"""
    <workbook xmlns="{evidence.SPREADSHEET_NAMESPACE}"
      xmlns:r="{evidence.OFFICE_RELATIONSHIP_NAMESPACE}">
      <sheets><sheet name="test sheet" sheetId="1" state="hidden" r:id="rId1"/></sheets>
      <definedNames><definedName name="named">test!$A$1</definedName></definedNames>
    </workbook>
    """
    relationships_xml = f"""
    <Relationships xmlns="{evidence.PACKAGE_RELATIONSHIP_NAMESPACE}">
      <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
    </Relationships>
    """
    shared_strings_xml = f"""
    <sst xmlns="{evidence.SPREADSHEET_NAMESPACE}">
      <si><t>header-a</t></si><si><t>header-b</t></si>
    </sst>
    """
    worksheet_xml = f"""
    <worksheet xmlns="{evidence.SPREADSHEET_NAMESPACE}">
      <dimension ref="A1:B2"/>
      <sheetData>
        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
        <row r="2"><c r="A2"><v>1</v></c><c r="B2"><f>A2+1</f><v>2</v></c></row>
      </sheetData>
    </worksheet>
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        archive.writestr("xl/sharedStrings.xml", shared_strings_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
        archive.writestr("xl/externalLinks/externalLink1.xml", "<externalLink/>")
        archive.writestr("customXml/item1.xml", "<custom/>")

    observed = evidence.inspect_supplement_workbook_bytes(buffer.getvalue())

    assert observed == {
        "sheet_count": 1,
        "sheets": [
            {
                "name": "test sheet",
                "visibility": "hidden",
                "rows": 2,
                "columns": 2,
                "header_row_count": 1,
                "header_rows_sha256": evidence.canonical_sha256([["header-a", "header-b"]]),
                "formula_cell_count": 1,
            }
        ],
        "all_sheets_visible": False,
        "defined_name_count": 1,
        "formula_cell_count": 1,
        "external_link_count": 1,
        "custom_xml_part_count": 1,
    }


def test_exact_body_identity_cannot_hide_structural_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"identity-matched-structure-drift"
    monkeypatch.setattr(evidence, "SUPPLEMENT_BYTES", len(body))
    monkeypatch.setattr(evidence, "SUPPLEMENT_SHA256", evidence._bytes_sha256(body))
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)
    drifted = evidence._expected_workbook_structure()
    drifted["sheets"][0]["header_rows_sha256"] = "f" * 64
    monkeypatch.setattr(
        evidence,
        "inspect_supplement_workbook_bytes",
        lambda _: drifted,
    )

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="structure differs"):
        evidence.validate_cbs_b6_public_native_evidence_lock(
            lock,
            pair_registry=registry,
            supplement_workbook_bytes=body,
        )


def test_registry_tamper_is_rejected() -> None:
    registry = _registry()
    registry["confirmatory_eligible"] = True

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="registry validation"):
        evidence.build_cbs_b6_public_native_evidence_lock(registry)


def test_duplicate_key_loader_rejects(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="duplicate key"):
        evidence.load_cbs_b6_public_native_evidence_lock(
            duplicate,
            pair_registry=_registry(),
        )


def test_atomic_writer_no_clobber_and_replace(tmp_path: Path) -> None:
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)
    destination = tmp_path / "lock.json"

    evidence.write_cbs_b6_public_native_evidence_lock(
        destination,
        lock,
        pair_registry=registry,
    )
    original = destination.read_bytes()
    with pytest.raises(evidence.CbsB6PublicEvidenceError, match="already exists"):
        evidence.write_cbs_b6_public_native_evidence_lock(
            destination,
            lock,
            pair_registry=registry,
        )
    assert destination.read_bytes() == original
    evidence.write_cbs_b6_public_native_evidence_lock(
        destination,
        lock,
        pair_registry=registry,
        replace=True,
    )
    assert destination.read_bytes() == evidence.canonical_json_bytes(lock) + b"\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_writer_rejects_wrong_suffix(tmp_path: Path) -> None:
    registry = _registry()
    lock = evidence.build_cbs_b6_public_native_evidence_lock(registry)

    with pytest.raises(evidence.CbsB6PublicEvidenceError, match=r"\.json"):
        evidence.write_cbs_b6_public_native_evidence_lock(
            tmp_path / "lock.txt",
            lock,
            pair_registry=registry,
        )


def test_cli_emits_same_lock(tmp_path: Path) -> None:
    destination = tmp_path / "lock.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.cbs_b6_public_native_evidence",
            "--registry",
            str(REGISTRY_PATH),
            "--lock-out",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    persisted = json.loads(destination.read_text())
    assert persisted == evidence.build_cbs_b6_public_native_evidence_lock(_registry())
    assert output["lock_sha256"] == persisted["lock_sha256"]
    assert output["native_replay_ready"] is False

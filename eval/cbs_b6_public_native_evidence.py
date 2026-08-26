"""Exact public-evidence inventory for CBS low/high-B6 post-count reconstruction.

This module locks what the paper, its official spreadsheet supplement, and the
paper-linked software repositories expose publicly.  It deliberately does not
promote ordinal ``replicate 1..N`` spreadsheet columns into biological,
sequencing-run, tile, or cross-condition pairing claims.

The artifact is an evidence ceiling and gap inventory.  It cannot authenticate
the paper-executed software revisions, construct a resampling graph, replay
TileSeq, select outcomes, or make the CBS candidate confirmatory-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval import cbs_b6_pair_contract as pair
from eval import cbs_b6_uncertainty_contract as uncertainty

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "groundbench.dms_cbs_b6_public_native_evidence_lock"
LOCK_ID = "cbs-b6-public-native-evidence-2026-07-26-v1"
CLAIM_SCOPE = "public_evidence_inventory_only_no_native_replay_sample_pairing_uncertainty_or_outcome_claim"
EVIDENCE_CEILING = "PUBLIC_NATIVE_EVIDENCE_PARTIAL"

PAPER_DOI = "10.1186/s13073-020-0711-1"
PAPER_URL = "https://doi.org/10.1186/s13073-020-0711-1"
SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1186%2Fs13073-020-0711-1/"
    "MediaObjects/13073_2020_711_MOESM3_ESM.xlsx"
)
SUPPLEMENT_BYTES = 10_005_946
SUPPLEMENT_SHA256 = "878365975f62da42c4113214958d4bd60ba7f4d3fe269eee1a722465ca6470aa"

TILESEQ_PACKAGE_REPOSITORY = "https://github.com/rothlab/tileseq_package"
TILESEQ_PACKAGE_PUBLICATION_WINDOW_COMMIT = "6d05ca2e49e7810bf4ca782a2538552fd648409e"
TILESEQ_PACKAGE_PUBLICATION_WINDOW_TREE = "733427bc208454564a3822f8b23b73ed47be265d"

TILESEQ_MAVE_PAPER_CITED_REPOSITORY = "https://github.com/jweile/tileseqMave"
TILESEQ_MAVE_RESOLVED_REPOSITORY = "https://github.com/rothlab/tileseqMave"
TILESEQ_MAVE_PUBLICATION_WINDOW_COMMIT = "c9f6121686f69054c0e7f3c6241edc140574897f"
TILESEQ_MAVE_PUBLICATION_WINDOW_TREE = "dced914b58ea388f2f1308fb362de3c1cc81d8f2"
TILESEQ_MAVE_CBS_TEST_PARAMETERS_BLOB = "d00bd2b5a63a232926652e1384caa74b8d282225"
TILESEQ_MAVE_CBS_TEST_PARAMETER_CSV_BLOB = "e8bcf2662aed8606c2d3a9708225093abb0c1a49"

ACTIVE_BLOCKER_CODES = (
    "CBS_B6_EXACT_PAPER_EXECUTED_SOFTWARE_REVISIONS_NOT_REPORTED",
    "CBS_B6_CBS_SPECIFIC_B6_PARAMETER_SHEET_NOT_AUTHENTICATED",
    "CBS_B6_ORDINAL_REPLICATE_COLUMNS_LACK_CULTURE_RUN_TILE_BINDING",
    "CBS_B6_CROSS_CONDITION_DEPENDENCY_GRAPH_NOT_AUTHENTICATED",
    "CBS_B6_PUBLIC_SUPPLEMENT_EXPOSES_ONLY_POST_COUNT_NORMALIZED_FREQUENCIES",
    "CBS_B6_PAPER_REPORTED_TWO_CULTURES_FAIL_CONSERVATIVE_EIGHT_BLOCK_PERCENTILE_CI_GATE",
    "CBS_B6_POST_COUNT_TILESEQ_SCORE_RECONSTRUCTION_NOT_AUTHENTICATED",
    "CBS_B6_PUBLIC_EVIDENCE_CANNOT_PROMOTE_OUTCOMES",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CELL_REFERENCE_PATTERN = re.compile(r"^\$?([A-Z]+)\$?([1-9][0-9]*)$")
EXTERNAL_LINK_PART_PATTERN = re.compile(r"^xl/externalLinks/externalLink[1-9][0-9]*\.xml$")
CUSTOM_XML_DATA_PART_PATTERN = re.compile(r"^customXml/item[1-9][0-9]*\.xml$")
SPREADSHEET_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
HEADER_ROW_COUNTS = {
    "raw read counts lowB6": 2,
    "raw read counts highB6": 2,
    "experimental scores lowB6": 2,
    "experimental scores highB6": 2,
    "refined scores lowB6": 1,
    "refined scores highB6": 1,
}


class CbsB6PublicEvidenceError(ValueError):
    """Raised when the CBS public native-evidence lock fails closed."""


def _strict_json(value: Any, context: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise CbsB6PublicEvidenceError(f"{context} keys must be strings")
            _strict_json(item, f"{context}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _strict_json(item, f"{context}[{index}]")
        return
    if value is None or type(value) in {str, int, bool}:
        return
    raise CbsB6PublicEvidenceError(f"{context} must use strict JSON scalar types")


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict deterministic JSON bytes."""

    _strict_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CbsB6PublicEvidenceError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CbsB6PublicEvidenceError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _column_index(column_letters: str) -> int:
    result = 0
    for character in column_letters:
        result = result * 26 + (ord(character) - ord("A") + 1)
    return result


def _dimension_shape(reference: str) -> tuple[int, int]:
    final_reference = reference.split(":")[-1]
    match = CELL_REFERENCE_PATTERN.fullmatch(final_reference)
    if match is None:
        raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet dimension is invalid")
    return int(match.group(2)), _column_index(match.group(1))


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except (KeyError, ET.ParseError) as exc:
        raise CbsB6PublicEvidenceError("CBS B6 supplement shared strings are invalid") from exc
    return [
        "".join(node.text or "" for node in item.iter(f"{{{SPREADSHEET_NAMESPACE}}}t"))
        for item in root.findall(f"{{{SPREADSHEET_NAMESPACE}}}si")
    ]


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{SPREADSHEET_NAMESPACE}}}t"))
    value = cell.find(f"{{{SPREADSHEET_NAMESPACE}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)]
        except (IndexError, ValueError) as exc:
            raise CbsB6PublicEvidenceError("CBS B6 supplement shared-string index is invalid") from exc
    return value.text


def _inspect_worksheet(
    archive: zipfile.ZipFile,
    *,
    member: str,
    name: str,
    visibility: str,
    shared_strings: list[str],
) -> dict[str, Any]:
    if name not in HEADER_ROW_COUNTS:
        raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet name is unexpected")
    header_row_count = HEADER_ROW_COUNTS[name]
    dimension_reference: str | None = None
    header_values: dict[tuple[int, int], str] = {}
    formula_cell_count = 0
    try:
        with archive.open(member) as worksheet:
            for _, element in ET.iterparse(worksheet, events=("end",)):
                if element.tag == f"{{{SPREADSHEET_NAMESPACE}}}dimension":
                    dimension_reference = element.attrib.get("ref")
                    element.clear()
                    continue
                if element.tag != f"{{{SPREADSHEET_NAMESPACE}}}row":
                    continue
                try:
                    row_index = int(element.attrib["r"])
                except (KeyError, ValueError) as exc:
                    raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet row index is invalid") from exc
                formulas = element.findall(f".//{{{SPREADSHEET_NAMESPACE}}}f")
                formula_cell_count += len(formulas)
                if row_index <= header_row_count:
                    for cell in element.findall(f"{{{SPREADSHEET_NAMESPACE}}}c"):
                        reference = cell.attrib.get("r", "")
                        match = CELL_REFERENCE_PATTERN.fullmatch(reference)
                        if match is None:
                            raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet cell reference is invalid")
                        header_values[(row_index, _column_index(match.group(1)))] = _cell_text(
                            cell,
                            shared_strings,
                        )
                element.clear()
    except (KeyError, ET.ParseError, OSError, zipfile.BadZipFile) as exc:
        raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet XML is invalid") from exc
    if dimension_reference is None:
        raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet dimension is missing")
    rows, columns = _dimension_shape(dimension_reference)
    header_rows = [
        [header_values.get((row_index, column_index), "") for column_index in range(1, columns + 1)]
        for row_index in range(1, header_row_count + 1)
    ]
    return {
        "name": name,
        "visibility": visibility,
        "rows": rows,
        "columns": columns,
        "header_row_count": header_row_count,
        "header_rows_sha256": canonical_sha256(header_rows),
        "formula_cell_count": formula_cell_count,
    }


def inspect_supplement_workbook_bytes(workbook_bytes: bytes) -> dict[str, Any]:
    """Recompute the locked OOXML workbook structure from exact bytes."""

    if type(workbook_bytes) is not bytes:
        raise TypeError("workbook_bytes must be exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as archive:
            members = archive.namelist()
            workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
            relationship_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            relationships = {
                relation.attrib["Id"]: relation.attrib["Target"]
                for relation in relationship_root.findall(f"{{{PACKAGE_RELATIONSHIP_NAMESPACE}}}Relationship")
            }
            shared_strings = _shared_strings(archive)
            sheets: list[dict[str, Any]] = []
            for sheet in workbook_root.findall(f".//{{{SPREADSHEET_NAMESPACE}}}sheet"):
                relationship_id = sheet.attrib.get(f"{{{OFFICE_RELATIONSHIP_NAMESPACE}}}id")
                if relationship_id not in relationships:
                    raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet relationship is missing")
                member = posixpath.normpath(posixpath.join("xl", relationships[relationship_id]))
                if member.startswith("../") or member not in members:
                    raise CbsB6PublicEvidenceError("CBS B6 supplement worksheet member is invalid")
                sheets.append(
                    _inspect_worksheet(
                        archive,
                        member=member,
                        name=sheet.attrib.get("name", ""),
                        visibility=sheet.attrib.get("state", "visible"),
                        shared_strings=shared_strings,
                    )
                )
            defined_name_count = len(workbook_root.findall(f".//{{{SPREADSHEET_NAMESPACE}}}definedName"))
    except (KeyError, ET.ParseError, OSError, zipfile.BadZipFile) as exc:
        raise CbsB6PublicEvidenceError("CBS B6 supplement is not valid OOXML") from exc
    return {
        "sheet_count": len(sheets),
        "sheets": sheets,
        "all_sheets_visible": all(sheet["visibility"] == "visible" for sheet in sheets),
        "defined_name_count": defined_name_count,
        "formula_cell_count": sum(sheet["formula_cell_count"] for sheet in sheets),
        "external_link_count": sum(EXTERNAL_LINK_PART_PATTERN.fullmatch(member) is not None for member in members),
        "custom_xml_part_count": sum(CUSTOM_XML_DATA_PART_PATTERN.fullmatch(member) is not None for member in members),
    }


def _expected_sheets() -> list[dict[str, Any]]:
    return [
        {
            "name": "raw read counts lowB6",
            "visibility": "visible",
            "rows_including_two_header_rows": 22_538,
            "columns": 34,
            "data_rows": 22_536,
            "header_row_count": 2,
            "header_rows_sha256": "b8ba92ecef46cb788c7bffa9ebffff7368da6e5e81dee201956825e11107370d",
            "ordinal_replicate_columns_per_measurement_role": 8,
            "measurement_roles": [
                "pre_selection_library_relative_read_frequency_per_1M_total_reads",
                "post_selection_library_relative_read_frequency_per_1M_total_reads",
                "wildtype_control_pre_selection_relative_read_frequency_per_1M_total_reads",
                "wildtype_control_selection_relative_read_frequency_per_1M_total_reads",
            ],
        },
        {
            "name": "raw read counts highB6",
            "visibility": "visible",
            "rows_including_two_header_rows": 22_057,
            "columns": 18,
            "data_rows": 22_055,
            "header_row_count": 2,
            "header_rows_sha256": "8ec2eb29e3adb12bb981fc201d1de08587a097aaaf54a15d18fb4197370a751b",
            "ordinal_replicate_columns_per_measurement_role": 4,
            "measurement_roles": [
                "pre_selection_library_relative_read_frequency_per_1M_total_reads",
                "post_selection_library_relative_read_frequency_per_1M_total_reads",
                "wildtype_control_pre_selection_relative_read_frequency_per_1M_total_reads",
                "wildtype_control_selection_relative_read_frequency_per_1M_total_reads",
            ],
        },
        {
            "name": "experimental scores lowB6",
            "visibility": "visible",
            "rows_including_two_header_rows": 11_480,
            "columns": 5,
            "data_rows": 11_478,
            "header_row_count": 2,
            "header_rows_sha256": "f0ae43bbc22d0ec7ac1cf2a8e880b771e490d96872612e5417c6465e1c697ced",
            "score_columns": ["score", "stdev", "stderr"],
        },
        {
            "name": "experimental scores highB6",
            "visibility": "visible",
            "rows_including_two_header_rows": 10_804,
            "columns": 5,
            "data_rows": 10_802,
            "header_row_count": 2,
            "header_rows_sha256": "f0ae43bbc22d0ec7ac1cf2a8e880b771e490d96872612e5417c6465e1c697ced",
            "score_columns": ["score", "stdev", "stderr"],
        },
        {
            "name": "refined scores lowB6",
            "visibility": "visible",
            "rows_including_header_row": 11_551,
            "columns": 4,
            "data_rows": 11_550,
            "header_row_count": 1,
            "header_rows_sha256": "9015fdd32b8e23f2fbc9b8691364450d2728142db02a564f55de2851cc288fac",
            "score_columns": ["score", "stdev", "stderr"],
        },
        {
            "name": "refined scores highB6",
            "visibility": "visible",
            "rows_including_header_row": 11_551,
            "columns": 4,
            "data_rows": 11_550,
            "header_row_count": 1,
            "header_rows_sha256": "9015fdd32b8e23f2fbc9b8691364450d2728142db02a564f55de2851cc288fac",
            "score_columns": ["score", "stdev", "stderr"],
        },
    ]


def _expected_workbook_structure() -> dict[str, Any]:
    sheets = []
    for sheet in _expected_sheets():
        rows = sheet.get(
            "rows_including_two_header_rows",
            sheet.get("rows_including_header_row"),
        )
        sheets.append(
            {
                "name": sheet["name"],
                "visibility": sheet["visibility"],
                "rows": rows,
                "columns": sheet["columns"],
                "header_row_count": sheet["header_row_count"],
                "header_rows_sha256": sheet["header_rows_sha256"],
                "formula_cell_count": 0,
            }
        )
    return {
        "sheet_count": 6,
        "sheets": sheets,
        "all_sheets_visible": True,
        "defined_name_count": 0,
        "formula_cell_count": 0,
        "external_link_count": 0,
        "custom_xml_part_count": 0,
    }


def _expected_lock() -> dict[str, Any]:
    lock: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "lock_id": LOCK_ID,
        "claim_scope": CLAIM_SCOPE,
        "accessed_on": "2026-07-26",
        "gene": pair.GENE,
        "pair_registry_sha256": pair.EXPECTED_PAIR_REGISTRY_SHA256,
        "paper": {
            "doi": PAPER_DOI,
            "url": PAPER_URL,
            "title": "A proactive genotype-to-patient-phenotype map for cystathionine beta-synthase",
            "publication_year": 2020,
            "public_data_parent_urn": "urn:mavedb:00000005-a",
            "reported_design": {
                "vitamin_b6_conditions_ng_per_mL": [0, 1, 400],
                "biological_replicate_cultures_per_selective_and_nonselective_condition": 2,
                "separate_sequencing_runs": 2,
                "low_b6_composite": "average_of_0_and_1_ng_per_mL_due_to_reported_high_agreement",
            },
            "reported_processing": [
                "paired_read_forward_reverse_agreement",
                "sequencing_depth_normalization_to_allele_frequency",
                "three_standard_deviations_above_corresponding_wildtype_error_control_filter",
                "equivalent_codon_join",
                "wildtype_error_control_frequency_subtraction",
                "selection_to_nonselection_enrichment_ratio",
                "baldi_long_error_regularization_with_two_pseudocounts",
                "nonsense_and_synonymous_median_scaling",
            ],
            "method_locators": {
                "experimental_and_replicate_design": {
                    "pdf_page": 4,
                    "subsection": "High-throughput yeast-based complementation",
                },
                "read_processing_and_normalization_boundary": {
                    "pdf_page": 4,
                    "subsection": "Detecting variant effects on fitness using TileSeq",
                },
                "score_processing": {
                    "pdf_page": 4,
                    "subsection": "Scoring fitness and vitamin B6 remediability",
                },
                "empirical_null_fdr_and_threshold_rules": {
                    "pdf_page": 5,
                    "subsection": "Classifying vitamin B6-remediable and non-remediable variants",
                },
            },
            "exact_paper_executed_software_commit_status": "not_reported",
        },
        "supplement_workbook": {
            "filename": "13073_2020_711_MOESM3_ESM.xlsx",
            "url": SUPPLEMENT_URL,
            "body_bytes": SUPPLEMENT_BYTES,
            "body_sha256": SUPPLEMENT_SHA256,
            "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sheet_count": 6,
            "sheets": _expected_sheets(),
            "all_sheets_visible": True,
            "defined_name_count": 0,
            "formula_cell_count": 0,
            "external_link_count": 0,
            "custom_xml_part_count": 0,
            "header_locator_rule": ("raw_and_experimental_sheet_headers_rows_1_2_refined_sheet_headers_row_1"),
            "structural_replay_fields": [
                "sheet_order_name_visibility",
                "sheet_dimensions",
                "header_row_hashes",
                "formula_cells",
                "defined_names",
                "external_links",
                "custom_xml_parts",
            ],
            "body_replay_requirement": (
                "exact_workbook_bytes_required_to_replay_structural_inspection_"
                "summary_lock_alone_is_not_native_replay_evidence"
            ),
        },
        "public_sample_role_evidence": {
            "status": "partial_ordinal_columns_only",
            "source_reported_and_locally_inspected_facts": [
                "low_workbook_has_8_ordinal_columns_for_each_measurement_role",
                "high_workbook_has_4_ordinal_columns_for_each_measurement_role",
                "paper_reports_two_biological_replicate_cultures_per_condition",
                "paper_reports_two_separate_sequencing_runs",
                "public_repository_test_fixture_describes_two_replicates_and_15_CBS_tiles",
            ],
            "unresolved_bindings": [
                "ordinal_column_to_0_or_1_ng_per_mL_low_b6_branch",
                "ordinal_column_to_biological_culture",
                "ordinal_column_to_sequencing_library",
                "ordinal_column_to_sequencing_run",
                "ordinal_column_to_tile_or_tile_aggregation",
                "shared_preselection_and_wildtype_control_ancestry_across_conditions",
                "cross_condition_independence_or_pairing",
            ],
            "pairing_claim": False,
            "column_order_inference_allowed": False,
        },
        "paper_linked_software": {
            "tileseq_package": {
                "repository_url": TILESEQ_PACKAGE_REPOSITORY,
                "repository_readme_version": "1.5",
                "publication_window_candidate_commit_sha1": (TILESEQ_PACKAGE_PUBLICATION_WINDOW_COMMIT),
                "publication_window_candidate_tree_sha1": (TILESEQ_PACKAGE_PUBLICATION_WINDOW_TREE),
                "candidate_status": (
                    "latest_repository_commit_observed_on_or_before_2020_01_30_"
                    "not_authenticated_as_exact_paper_executed_revision"
                ),
                "license_status": "no_repository_license_detected",
            },
            "tileseq_mave": {
                "paper_cited_repository_url": TILESEQ_MAVE_PAPER_CITED_REPOSITORY,
                "resolved_repository_url": TILESEQ_MAVE_RESOLVED_REPOSITORY,
                "publication_window_candidate_commit_sha1": (TILESEQ_MAVE_PUBLICATION_WINDOW_COMMIT),
                "publication_window_candidate_tree_sha1": (TILESEQ_MAVE_PUBLICATION_WINDOW_TREE),
                "candidate_status": (
                    "latest_repository_commit_observed_on_or_before_2020_01_30_"
                    "not_authenticated_as_exact_paper_executed_revision"
                ),
                "license": "GPL-3.0",
                "cbs_repository_test_fixture": {
                    "parameters_json_blob_sha1": (TILESEQ_MAVE_CBS_TEST_PARAMETERS_BLOB),
                    "parameter_csv_blob_sha1": (TILESEQ_MAVE_CBS_TEST_PARAMETER_CSV_BLOB),
                    "fixture_commit_sha1": TILESEQ_MAVE_PUBLICATION_WINDOW_COMMIT,
                    "gene": pair.GENE,
                    "tile_count": 15,
                    "declared_replicates_per_generic_role": 2,
                    "condition_names": [
                        "nonselect",
                        "select",
                        "wtSelect",
                        "wtNonselect",
                    ],
                    "evidence_status": (
                        "repository_test_fixture_not_authenticated_as_the_CBS_B6_"
                        "paper_parameter_sheet_or_condition_specific_sample_map"
                    ),
                },
            },
        },
        "joint_bootstrap_admission": {
            "selected_method": uncertainty.SELECTED_METHOD,
            "method_contract_definition_sha256": (uncertainty.method_contract_definition_sha256()),
            "draw_count": uncertainty.JOINT_BOOTSTRAP_DRAW_COUNT,
            "seed": uncertainty.JOINT_BOOTSTRAP_SEED,
            "paper_reported_biological_cultures_per_condition": 2,
            "minimum_effective_independent_blocks_per_condition_branch_for_percentile_ci": 8,
            "public_evidence_meets_independent_block_minimum": False,
            "claim_bearing_percentile_ci_allowed_at_n_equals_2": False,
            "public_evidence_contains_only_post_count_normalized_frequencies": True,
            "public_evidence_sufficient_to_construct_resampling_graph": False,
            "execution_allowed": False,
            "required_next_artifact": (
                "authenticated_CBS_B6_specific_native_input_parameter_sample_role_"
                "dependency_runtime_and_label_null_manifests"
            ),
        },
        "evidence_ceiling": EVIDENCE_CEILING,
        "native_replay_ready": False,
        "uncertainty_ready": False,
        "outcome_status": "not_derived",
        "confirmatory_eligible": False,
        "automatic_promotion": False,
        "active_blocker_codes": list(ACTIVE_BLOCKER_CODES),
        "lock_sha256": "",
    }
    lock["lock_sha256"] = lock_sha256(lock)
    return lock


def lock_sha256(lock: Mapping[str, Any]) -> str:
    payload = dict(lock)
    payload.pop("lock_sha256", None)
    return canonical_sha256(payload)


def validate_cbs_b6_public_native_evidence_lock(
    lock: Mapping[str, Any],
    *,
    pair_registry: Mapping[str, Any],
    supplement_workbook_bytes: bytes | None = None,
) -> Mapping[str, Any]:
    """Validate the exact public-evidence lock and optionally replay workbook identity."""

    try:
        pair.validate_cbs_b6_pair_registry(pair_registry)
    except pair.CbsB6PairError as exc:
        raise CbsB6PublicEvidenceError("CBS B6 pair registry validation failed") from exc
    _strict_json(lock, "CBS B6 public native-evidence lock")
    expected = _expected_lock()
    if canonical_json_bytes(lock) != canonical_json_bytes(expected):
        raise CbsB6PublicEvidenceError("CBS B6 public native-evidence lock differs")
    observed_hash = lock.get("lock_sha256")
    if (
        type(observed_hash) is not str
        or SHA256_PATTERN.fullmatch(observed_hash) is None
        or observed_hash != lock_sha256(lock)
    ):
        raise CbsB6PublicEvidenceError("CBS B6 public native-evidence self-hash differs")
    if supplement_workbook_bytes is not None:
        if type(supplement_workbook_bytes) is not bytes:
            raise TypeError("supplement_workbook_bytes must be exact bytes")
        if (
            len(supplement_workbook_bytes) != SUPPLEMENT_BYTES
            or _bytes_sha256(supplement_workbook_bytes) != SUPPLEMENT_SHA256
        ):
            raise CbsB6PublicEvidenceError("CBS B6 supplement workbook identity differs")
        observed_structure = inspect_supplement_workbook_bytes(supplement_workbook_bytes)
        if canonical_json_bytes(observed_structure) != canonical_json_bytes(_expected_workbook_structure()):
            raise CbsB6PublicEvidenceError("CBS B6 supplement workbook structure differs")
    return lock


def build_cbs_b6_public_native_evidence_lock(
    pair_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic public-evidence ceiling artifact."""

    lock = _expected_lock()
    validate_cbs_b6_public_native_evidence_lock(
        lock,
        pair_registry=pair_registry,
    )
    return lock


def _load_json(path: str | Path) -> Mapping[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CbsB6PublicEvidenceError("cannot read CBS B6 public native-evidence lock") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_hook)
    except CbsB6PublicEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CbsB6PublicEvidenceError("CBS B6 public native-evidence lock must be duplicate-free UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise CbsB6PublicEvidenceError("CBS B6 public native-evidence lock must be an object")
    return value


def load_cbs_b6_public_native_evidence_lock(
    path: str | Path,
    *,
    pair_registry: Mapping[str, Any],
    supplement_workbook_bytes: bytes | None = None,
) -> Mapping[str, Any]:
    lock = _load_json(path)
    return validate_cbs_b6_public_native_evidence_lock(
        lock,
        pair_registry=pair_registry,
        supplement_workbook_bytes=supplement_workbook_bytes,
    )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_cbs_b6_public_native_evidence_lock(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    pair_registry: Mapping[str, Any],
    supplement_workbook_bytes: bytes | None = None,
    replace: bool = False,
) -> Path:
    """Atomically write the validated evidence lock and fsync its directory."""

    validate_cbs_b6_public_native_evidence_lock(
        lock,
        pair_registry=pair_registry,
        supplement_workbook_bytes=supplement_workbook_bytes,
    )
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise CbsB6PublicEvidenceError("CBS B6 public native-evidence lock must use .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(lock) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary_name, destination)
        else:
            try:
                os.link(temporary_name, destination)
            except FileExistsError as exc:
                raise CbsB6PublicEvidenceError(
                    f"CBS B6 public native-evidence lock already exists: {destination}"
                ) from exc
            os.unlink(temporary_name)
        _fsync_parent(destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _main() -> None:
    parser = argparse.ArgumentParser(description="Emit the deterministic CBS B6 public native-evidence ceiling")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--lock-out", type=Path, required=True)
    parser.add_argument("--supplement-workbook", type=Path)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()

    try:
        registry = pair.load_cbs_b6_pair_registry(arguments.registry)
        supplement_bytes = (
            arguments.supplement_workbook.read_bytes() if arguments.supplement_workbook is not None else None
        )
        lock = build_cbs_b6_public_native_evidence_lock(registry)
        write_cbs_b6_public_native_evidence_lock(
            arguments.lock_out,
            lock,
            pair_registry=registry,
            supplement_workbook_bytes=supplement_bytes,
            replace=arguments.replace,
        )
    except (pair.CbsB6PairError, CbsB6PublicEvidenceError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "lock_path": str(arguments.lock_out),
                "lock_sha256": lock["lock_sha256"],
                "evidence_ceiling": lock["evidence_ceiling"],
                "native_replay_ready": lock["native_replay_ready"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from copy import deepcopy

import pytest

from eval import mavedb_source_lock as msl

URN = "urn:mavedb:00000001-a-1"
API_BASE = "https://api.example.test/api/v1"
OPENAPI_URL = "https://api.example.test/openapi.json"
METADATA_URL = f"{API_BASE}/score-sets/{URN}"
SCORES_URL = f"{METADATA_URL}/scores"
COUNTS_URL = f"{METADATA_URL}/counts"
MAPPED_URL = f"{METADATA_URL}/mapped-variants"


class MockTransport:
    def __init__(self, responses):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls = defaultdict(int)
        self.request_headers = {}

    def __call__(self, url, headers):
        self.calls[url] += 1
        self.request_headers[url] = dict(headers)
        if url not in self.responses or not self.responses[url]:
            raise AssertionError(f"unexpected transport URL: {url}")
        queue = self.responses[url]
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _response(
    body,
    content_type,
    *,
    status=200,
    gzip_encoded=False,
    declared_length=None,
    chunked=False,
):
    if isinstance(body, str):
        body = body.encode()
    wire_body = gzip.compress(body, mtime=0) if gzip_encoded else body
    headers = {"Content-Type": content_type}
    if chunked:
        headers["Transfer-Encoding"] = "chunked"
    else:
        headers["Content-Length"] = str(len(wire_body) if declared_length is None else declared_length)
    if gzip_encoded:
        headers["Content-Encoding"] = "gzip"
    return msl.HttpResponse(status=status, headers=headers, body=wire_body)


def _metadata(*, count_columns=None, score_columns=None, target=None):
    if count_columns is None:
        count_columns = ["count_rep1"]
    if score_columns is None:
        score_columns = ["score", "score_rep1"]
    if target is None:
        target = {
            "name": "GENE1",
            "mappedHgncName": "GENE1",
            "externalIdentifiers": [],
            "targetSequence": {
                "sequenceType": "dna",
                "sequence": "ATGAAA",
                "recordType": "TargetSequence",
            },
            "targetAccession": None,
        }
    return {
        "urn": URN,
        "title": "Synthetic MaveDB score set",
        "numVariants": 2,
        "private": False,
        "processingState": "success",
        "supersededScoreSet": None,
        "supersedingScoreSet": None,
        "license": {
            "id": 1,
            "shortName": "CC0",
            "longName": "CC0 (Public domain)",
            "active": True,
            "link": "https://creativecommons.org/publicdomain/zero/1.0/",
            "version": "1.0",
            "recordType": "ShortLicense",
        },
        "dataUsagePolicy": None,
        "datasetColumns": {
            "scoreColumns": score_columns,
            "countColumns": count_columns,
            "recordType": "DatasetColumns",
        },
        "scoreCalibrations": [
            {
                "urn": "urn:mavedb:calibration-test",
                "title": "Test calibration",
                "baselineScore": 0.0,
            }
        ],
        "targetGenes": [target],
    }


def _scores(score_columns=None):
    if score_columns is None:
        score_columns = ["score", "score_rep1"]
    header = [*msl.FIXED_VARIANT_COLUMNS, *score_columns]
    values = {
        "score": ("0.2", "-1.1"),
        "score_rep1": ("0.1", "-1.0"),
        "annotation": ("benign", "pathogenic"),
    }
    rows = []
    for index in range(2):
        fixed = [f"{URN}#{index + 1}", f"c.{index + 1}A>G", "NA", f"p.Ala{index + 1}Gly"]
        rows.append(fixed + [values[column][index] for column in score_columns])
    return ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()


def _counts(count_columns=None):
    if count_columns is None:
        count_columns = ["count_rep1"]
    header = [*msl.FIXED_VARIANT_COLUMNS, *count_columns]
    rows = []
    for index in range(2):
        fixed = [f"{URN}#{index + 1}", f"c.{index + 1}A>G", "NA", f"p.Ala{index + 1}Gly"]
        rows.append(fixed + [str(10 + index) for _ in count_columns])
    return ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()


def _mappings(*, second_error=False, include_history=False):
    records = []
    if include_history:
        records.append(
            {
                "variantUrn": f"{URN}#1",
                "current": False,
                "preMapped": None,
                "postMapped": None,
                "errorMessage": "historical failure",
                "alignmentLevel": "protein",
                "mappingApiVersion": "old",
                "vrsVersion": "1",
            }
        )
    records.append(
        {
            "variantUrn": f"{URN}#1",
            "current": True,
            "preMapped": {"id": "pre-1"},
            "postMapped": {"id": "post-1"},
            "errorMessage": None,
            "alignmentLevel": "protein",
            "mappingApiVersion": "2026.2.1",
            "vrsVersion": "2",
            "atMismatchedLocus": False,
            "nearGap": False,
        }
    )
    records.append(
        {
            "variantUrn": f"{URN}#2",
            "current": True,
            "preMapped": None if second_error else {"id": "pre-2"},
            "postMapped": None if second_error else {"id": "post-2"},
            "errorMessage": "mapping failed" if second_error else None,
            "alignmentLevel": "protein",
            "mappingApiVersion": "2026.2.1",
            "vrsVersion": "2",
            "atMismatchedLocus": False,
            "nearGap": False,
        }
    )
    return records


def _responses(
    *,
    metadata=None,
    scores=None,
    counts=None,
    mappings=None,
    openapi_response=None,
):
    metadata = _metadata() if metadata is None else metadata
    score_columns = metadata["datasetColumns"]["scoreColumns"]
    count_columns = metadata["datasetColumns"]["countColumns"]
    scores = _scores(score_columns) if scores is None else scores
    counts = _counts(count_columns) if counts is None else counts
    mappings = _mappings() if mappings is None else mappings
    if openapi_response is None:
        openapi_response = _response(
            _json_bytes({"openapi": "3.1.0", "info": {"version": "2026.2.7"}}),
            "application/json",
        )
    return {
        OPENAPI_URL: [openapi_response],
        METADATA_URL: [_response(_json_bytes(metadata), "application/json")],
        SCORES_URL: [_response(scores, "text/csv; charset=utf-8")],
        COUNTS_URL: [_response(counts, "text/csv; charset=utf-8")],
        MAPPED_URL: [_response(_json_bytes(mappings), "application/json")],
    }


def _config(
    *,
    readiness=msl.Readiness.PROCESSED_REPLICATES,
    evidence=None,
    max_attempts=3,
    expected_sha256=None,
    expected_reference_sha256=None,
):
    if evidence is None:
        evidence = msl.ReadinessEvidence(
            aggregate_score_column="score",
            processed_replicate_columns=("score_rep1",),
        )
    return msl.SourceLockConfig(
        urn=URN,
        readiness=readiness,
        readiness_evidence=evidence,
        api_base_url=API_BASE,
        openapi_url=OPENAPI_URL,
        expected_api_version="2026.2.7",
        expected_sha256=expected_sha256 or {},
        expected_reference_sha256=expected_reference_sha256 or {},
        max_attempts=max_attempts,
    )


def _build(responses=None, config=None):
    transport = MockTransport(_responses() if responses is None else responses)
    lock = msl.build_candidate_source_lock(config or _config(), transport=transport)
    return lock, transport


def _refresh_integrity_hashes(lock):
    metadata = lock["metadata_contract"]
    metadata["license_sha256"] = msl.canonical_sha256(metadata["license"])
    metadata["dataset_columns_sha256"] = msl.canonical_sha256(metadata["dataset_columns"])
    lock["metadata_tabular_license_binding_sha256"] = msl.canonical_sha256(msl._license_binding_payload(lock))
    lock["source_bundle_sha256"] = msl.canonical_sha256(msl._source_bundle_payload(lock))
    return lock


def test_builds_candidate_only_lock_and_atomic_writer_is_canonical(tmp_path):
    lock, transport = _build()

    assert lock["claim_scope"] == msl.CLAIM_SCOPE
    assert lock["ingestion_status"] == "not_ingested"
    assert lock["outcome_status"] == "not_derived"
    assert lock["readiness"]["state"] == "PROCESSED_REPLICATES"
    assert lock["readiness"]["automatic_promotion"] is False
    assert lock["tabular_contract"]["counts_status"] == "substantive"
    assert lock["tabular_contract"]["substantive_count_column_count"] == 1
    assert lock["metadata_contract"]["calibration_count"] == 1
    assert transport.request_headers[MAPPED_URL]["Accept-Encoding"] == "gzip"

    first = msl.write_source_lock(tmp_path / "one" / "lock.json", lock)
    second = msl.write_source_lock(tmp_path / "two" / "lock.json", lock)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert msl.load_source_lock(first) == lock


def test_lower_readiness_is_never_inferred_upward():
    evidence = msl.ReadinessEvidence(aggregate_score_column="score")
    lock, _ = _build(
        config=_config(
            readiness=msl.Readiness.AGGREGATE_ONLY,
            evidence=evidence,
        )
    )
    assert lock["readiness"]["state"] == "AGGREGATE_ONLY"
    assert lock["tabular_contract"]["counts_status"] == "substantive"
    assert "score_rep1" in lock["metadata_contract"]["dataset_columns"]["scoreColumns"]


def test_truncated_http_200_is_rejected_by_content_length():
    responses = _responses()
    good = responses[METADATA_URL][0]
    responses[METADATA_URL] = [
        msl.HttpResponse(
            status=200,
            headers={
                **good.headers,
                "Content-Length": str(len(good.body) + 5),
            },
            body=good.body,
        )
    ]
    with pytest.raises(msl.SourceLockError, match="Content-Length mismatch"):
        _build(responses, _config(max_attempts=1))


def test_chunked_http_framing_is_explicitly_supported():
    responses = _responses()
    scores = responses[SCORES_URL][0]
    responses[SCORES_URL] = [
        msl.HttpResponse(
            status=200,
            headers={
                "Content-Type": scores.headers["Content-Type"],
                "Transfer-Encoding": "chunked",
            },
            body=scores.body,
        )
    ]
    lock, _ = _build(responses, _config(max_attempts=1))
    artifact = lock["source_artifacts"]["scores"]
    assert artifact["transfer_encoding"] == "chunked"
    assert artifact["declared_content_length"] is None
    assert artifact["wire_byte_count"] == len(scores.body)


def test_unframed_or_ambiguously_framed_http_200_is_rejected():
    responses = _responses()
    scores = responses[SCORES_URL][0]
    responses[SCORES_URL] = [
        msl.HttpResponse(
            status=200,
            headers={"Content-Type": scores.headers["Content-Type"]},
            body=scores.body,
        )
    ]
    with pytest.raises(msl.SourceLockError, match="supported HTTP response framing"):
        _build(responses, _config(max_attempts=1))

    responses = _responses()
    scores = responses[SCORES_URL][0]
    responses[SCORES_URL] = [
        msl.HttpResponse(
            status=200,
            headers={
                **scores.headers,
                "Transfer-Encoding": "chunked",
            },
            body=scores.body,
        )
    ]
    with pytest.raises(msl.SourceLockError, match="both HTTP framing headers"):
        _build(responses, _config(max_attempts=1))


def test_unsupported_transfer_encoding_is_rejected():
    responses = _responses()
    scores = responses[SCORES_URL][0]
    responses[SCORES_URL] = [
        msl.HttpResponse(
            status=200,
            headers={
                "Content-Type": scores.headers["Content-Type"],
                "Transfer-Encoding": "compress",
            },
            body=scores.body,
        )
    ]
    with pytest.raises(msl.SourceLockError, match="supported HTTP response framing"):
        _build(responses, _config(max_attempts=1))


def test_matching_length_but_truncated_json_is_rejected_as_incomplete():
    responses = _responses()
    incomplete = b'[{"variantUrn":'
    responses[MAPPED_URL] = [_response(incomplete, "application/json")]
    with pytest.raises(msl.SourceLockError, match="incomplete or invalid JSON"):
        _build(responses, _config(max_attempts=1))


def test_retry_recovers_from_502_and_records_attempt_count():
    responses = _responses()
    successful = responses[SCORES_URL][0]
    responses[SCORES_URL] = [
        _response("<html>bad gateway</html>", "text/html", status=502),
        successful,
    ]
    lock, transport = _build(responses, _config(max_attempts=2))
    assert transport.calls[SCORES_URL] == 2
    assert lock["source_artifacts"]["scores"]["attempts"] == 2


def test_gzip_wire_and_decoded_hashes_are_both_recorded():
    openapi_body = _json_bytes(
        {
            "openapi": "3.1.0",
            "info": {"version": "2026.2.7"},
            "padding": "A" * 1_000,
        }
    )
    responses = _responses(
        openapi_response=_response(
            openapi_body,
            "application/json",
            gzip_encoded=True,
        )
    )
    lock, _ = _build(responses)
    artifact = lock["api"]["openapi_artifact"]
    assert artifact["content_encoding"] == "gzip"
    assert artifact["wire_byte_count"] < artifact["decoded_byte_count"]
    assert artifact["sha256"] == hashlib.sha256(openapi_body).hexdigest()
    assert artifact["wire_sha256"] != artifact["sha256"]


def test_identifier_only_counts_are_not_promoted_from_nonempty_rows():
    metadata = _metadata(count_columns=[])
    responses = _responses(metadata=metadata, counts=_counts([]))
    lock, _ = _build(
        responses,
        _config(
            readiness=msl.Readiness.AGGREGATE_ONLY,
            evidence=msl.ReadinessEvidence(aggregate_score_column="score"),
        ),
    )
    counts_artifact = lock["source_artifacts"]["counts"]
    assert counts_artifact["decoded_byte_count"] > 0
    assert lock["tabular_contract"]["count_row_count"] == 2
    assert lock["tabular_contract"]["counts_status"] == "identifier_only"
    assert lock["tabular_contract"]["substantive_count_column_count"] == 0
    assert lock["tabular_contract"]["substantive_count_value_count"] == 0


def test_mapping_history_filters_current_and_summarizes_current_errors():
    mappings = _mappings(second_error=True, include_history=True)
    responses = _responses(mappings=mappings)
    lock, _ = _build(
        responses,
        _config(
            readiness=msl.Readiness.AGGREGATE_ONLY,
            evidence=msl.ReadinessEvidence(aggregate_score_column="score"),
        ),
    )
    summary = lock["mapping_contract"]
    assert summary["history_record_count"] == 3
    assert summary["current_record_count"] == 2
    assert summary["current_unique_variant_count"] == 2
    assert summary["current_error_count"] == 1
    assert summary["current_post_mapped_null_count"] == 1
    assert summary["mapping_api_versions"] == ["2026.2.1"]
    assert summary["vrs_versions"] == ["2"]


def test_duplicate_current_mapping_is_rejected():
    mappings = _mappings()
    mappings.append(deepcopy(mappings[0]))
    responses = _responses(mappings=mappings)
    with pytest.raises(msl.SourceLockError, match="duplicate current mappings"):
        _build(responses)


def test_metadata_and_tabular_bytes_are_bound_to_active_license():
    first, _ = _build()
    metadata = _metadata()
    metadata["license"] = {
        **metadata["license"],
        "shortName": "CC BY 4.0",
        "longName": "Creative Commons Attribution 4.0",
        "link": "https://creativecommons.org/licenses/by/4.0/",
    }
    second, _ = _build(_responses(metadata=metadata))

    assert first["source_artifacts"]["scores"]["sha256"] == second["source_artifacts"]["scores"]["sha256"]
    assert first["metadata_contract"]["license_sha256"] != second["metadata_contract"]["license_sha256"]
    assert first["metadata_tabular_license_binding_sha256"] != second["metadata_tabular_license_binding_sha256"]

    tampered = deepcopy(first)
    tampered["metadata_contract"]["license"]["shortName"] = "changed"
    with pytest.raises(msl.SourceLockError, match="license hash differs"):
        msl.validate_source_lock(tampered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.__setitem__("private", True), "public/non-private"),
        (
            lambda metadata: metadata.__setitem__("processingState", "failed"),
            "processingState",
        ),
        (
            lambda metadata: metadata["license"].__setitem__("active", False),
            "license must be active",
        ),
        (
            lambda metadata: metadata.__setitem__(
                "supersededScoreSet",
                {"urn": "urn:mavedb:00000001-a-0"},
            ),
            "superseded",
        ),
    ],
)
def test_rejects_nonpublic_inactive_unprocessed_or_superseded_metadata(
    mutation,
    message,
):
    metadata = _metadata()
    mutation(metadata)
    with pytest.raises(msl.SourceLockError, match=message):
        _build(_responses(metadata=metadata))


def test_expected_source_hash_mismatch_is_rejected():
    expected = {"scores": "0" * 64}
    with pytest.raises(msl.SourceLockError, match="scores hash mismatch"):
        _build(config=_config(expected_sha256=expected))


def test_expected_license_binding_mismatch_is_rejected():
    first, _ = _build()
    expected = {"metadata_tabular_license_binding": first["metadata_tabular_license_binding_sha256"]}
    metadata = _metadata()
    metadata["license"] = {
        **metadata["license"],
        "shortName": "CC BY-SA 4.0",
        "longName": "Creative Commons Attribution-ShareAlike 4.0",
        "link": "https://creativecommons.org/licenses/by-sa/4.0/",
    }
    with pytest.raises(msl.SourceLockError, match="license binding hash mismatch"):
        _build(
            _responses(metadata=metadata),
            _config(expected_sha256=expected),
        )


@pytest.mark.parametrize(
    ("readiness", "evidence", "message"),
    [
        (
            msl.Readiness.PROCESSED_REPLICATES,
            msl.ReadinessEvidence(aggregate_score_column="score"),
            "requires processed replicate columns",
        ),
        (
            msl.Readiness.COUNT_RECOMPUTABLE,
            msl.ReadinessEvidence(
                aggregate_score_column="score",
                count_lineage_columns=("count_rep1",),
            ),
            "requires identity_resolution_spec_sha256",
        ),
        (
            msl.Readiness.CONFIRMATORY_READY,
            msl.ReadinessEvidence(
                aggregate_score_column="score",
                processed_replicate_columns=("score_rep1",),
                count_lineage_columns=("count_rep1",),
                identity_resolution_spec_sha256="1" * 64,
                transformation_spec_sha256="2" * 64,
                wt_control_spec_sha256="3" * 64,
            ),
            "candidate-stage MaveDB source locks reject CONFIRMATORY_READY",
        ),
    ],
)
def test_readiness_overclaims_are_rejected(readiness, evidence, message):
    with pytest.raises(msl.SourceLockError, match=message):
        _build(config=_config(readiness=readiness, evidence=evidence))


def test_candidate_lock_rejects_confirmatory_even_with_all_caller_hashes():
    evidence = msl.ReadinessEvidence(
        aggregate_score_column="score",
        processed_replicate_columns=("score_rep1",),
        count_lineage_columns=("count_rep1",),
        identity_resolution_spec_sha256="1" * 64,
        transformation_spec_sha256="2" * 64,
        wt_control_spec_sha256="3" * 64,
        confirmatory_preregistration_sha256="4" * 64,
        independent_replication_spec_sha256="5" * 64,
    )
    with pytest.raises(
        msl.SourceLockError,
        match="caller-supplied hashes cannot authorize confirmatory execution",
    ):
        _build(
            _responses(mappings=_mappings(second_error=False)),
            _config(readiness=msl.Readiness.CONFIRMATORY_READY, evidence=evidence),
        )


def test_loaded_lock_cannot_be_coherently_escalated_to_confirmatory(tmp_path):
    lock, _ = _build()
    forged = deepcopy(lock)
    forged["readiness"]["state"] = msl.Readiness.CONFIRMATORY_READY.value
    forged["source_bundle_sha256"] = msl.canonical_sha256(msl._source_bundle_payload(forged))

    with pytest.raises(
        msl.SourceLockError,
        match="candidate-stage MaveDB source locks reject CONFIRMATORY_READY",
    ):
        msl.validate_source_lock(forged)
    with pytest.raises(msl.SourceLockError, match="reject CONFIRMATORY_READY"):
        msl.write_source_lock(tmp_path / "forged.json", forged)


def test_recomputable_readiness_rejects_current_mapping_errors():
    evidence = msl.ReadinessEvidence(
        aggregate_score_column="score",
        count_lineage_columns=("count_rep1",),
        identity_resolution_spec_sha256="1" * 64,
        transformation_spec_sha256="2" * 64,
        wt_control_spec_sha256="3" * 64,
    )
    with pytest.raises(msl.SourceLockError, match="error-free current mappings"):
        _build(
            _responses(mappings=_mappings(second_error=True)),
            _config(readiness=msl.Readiness.COUNT_RECOMPUTABLE, evidence=evidence),
        )


def test_recomputable_readiness_requires_values_in_every_count_column():
    count_columns = ["count_rep1", "count_rep2"]
    metadata = _metadata(count_columns=count_columns)
    header = [*msl.FIXED_VARIANT_COLUMNS, *count_columns]
    rows = [
        [f"{URN}#1", "c.1A>G", "NA", "p.Ala1Gly", "10", "NA"],
        [f"{URN}#2", "c.2A>G", "NA", "p.Ala2Gly", "NA", "NA"],
    ]
    sparse_counts = ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()
    evidence = msl.ReadinessEvidence(
        aggregate_score_column="score",
        count_lineage_columns=tuple(count_columns),
        identity_resolution_spec_sha256="1" * 64,
        transformation_spec_sha256="2" * 64,
        wt_control_spec_sha256="3" * 64,
    )

    with pytest.raises(
        msl.SourceLockError,
        match="count_rep2.*has no finite values",
    ):
        _build(
            _responses(
                metadata=metadata,
                counts=sparse_counts,
                mappings=_mappings(second_error=False),
            ),
            _config(
                readiness=msl.Readiness.COUNT_RECOMPUTABLE,
                evidence=evidence,
            ),
        )


def test_exact_csv_schema_and_row_count_are_enforced():
    bad_schema = _scores().replace(b"score,score_rep1", b"score,unexpected")
    with pytest.raises(msl.SourceLockError, match="schema differs"):
        _build(_responses(scores=bad_schema))

    truncated_rows = b"\n".join(_scores().splitlines()[:2]) + b"\n"
    with pytest.raises(msl.SourceLockError, match="row count differs"):
        _build(_responses(scores=truncated_rows))


def test_target_accession_is_resolved_and_hash_locked():
    accession = "NM_000001.1"
    target = {
        "name": "GENE1",
        "mappedHgncName": "GENE1",
        "externalIdentifiers": [],
        "targetSequence": None,
        "targetAccession": {
            "accession": accession,
            "assembly": "GRCh38",
            "gene": "GENE1",
            "isBaseEditor": False,
            "recordType": "TargetAccession",
        },
    }
    metadata = _metadata(target=target)
    responses = _responses(metadata=metadata)
    reference_url = f"{API_BASE}/refget/sequence/{accession}"
    reference = b"ATGAAA"
    responses[reference_url] = [_response(reference, "text/plain")]
    expected_reference = {accession: hashlib.sha256(reference).hexdigest()}
    lock, transport = _build(
        responses,
        _config(expected_reference_sha256=expected_reference),
    )
    locked_target = lock["metadata_contract"]["targets"][0]
    assert transport.calls[reference_url] == 1
    assert locked_target["target_kind"] == "accession"
    assert locked_target["accession"] == accession
    assert locked_target["sequence_length"] == 6
    assert locked_target["sequence_sha256"] == expected_reference[accession]
    assert locked_target["reference_artifact"]["sha256"] == expected_reference[accession]


def test_embedded_target_sequence_requires_its_configured_hash():
    expected = hashlib.sha256(b"ATGAAA").hexdigest()
    lock, _ = _build(
        config=_config(
            expected_reference_sha256={"target:GENE1": expected},
        )
    )
    assert lock["metadata_contract"]["targets"][0]["sequence_sha256"] == expected

    with pytest.raises(msl.SourceLockError, match="embedded target sequence hash mismatch"):
        _build(
            config=_config(
                expected_reference_sha256={"target:GENE1": "0" * 64},
            )
        )

    with pytest.raises(msl.SourceLockError, match="unused expected reference"):
        _build(
            config=_config(
                expected_reference_sha256={"target:WRONG": expected},
            )
        )


def test_persisted_recomputable_rewrite_replays_mapping_invariants(tmp_path):
    lock, _ = _build(
        _responses(mappings=_mappings(second_error=True)),
        _config(
            readiness=msl.Readiness.AGGREGATE_ONLY,
            evidence=msl.ReadinessEvidence(aggregate_score_column="score"),
        ),
    )
    forged = deepcopy(lock)
    forged["readiness"]["state"] = msl.Readiness.COUNT_RECOMPUTABLE.value
    forged["readiness"]["evidence"].update(
        {
            "count_lineage_columns": ["count_rep1"],
            "identity_resolution_spec_sha256": "1" * 64,
            "transformation_spec_sha256": "2" * 64,
            "wt_control_spec_sha256": "3" * 64,
        }
    )
    _refresh_integrity_hashes(forged)

    with pytest.raises(msl.SourceLockError, match="error-free current mappings"):
        msl.validate_source_lock(forged)
    with pytest.raises(msl.SourceLockError, match="error-free current mappings"):
        msl.write_source_lock(tmp_path / "forged-write.json", forged)

    raw_path = tmp_path / "forged-load.json"
    raw_path.write_bytes(msl.canonical_json_bytes(forged) + b"\n")
    with pytest.raises(msl.SourceLockError, match="error-free current mappings"):
        msl.load_source_lock(raw_path)


def test_persisted_recomputable_rewrite_replays_per_column_finite_counts():
    count_columns = ["count_rep1", "count_rep2"]
    metadata = _metadata(count_columns=count_columns)
    header = [*msl.FIXED_VARIANT_COLUMNS, *count_columns]
    rows = [
        [f"{URN}#1", "c.1A>G", "NA", "p.Ala1Gly", "10", "NA"],
        [f"{URN}#2", "c.2A>G", "NA", "p.Ala2Gly", "11", "NA"],
    ]
    counts = ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()
    lock, _ = _build(
        _responses(metadata=metadata, counts=counts),
        _config(
            readiness=msl.Readiness.AGGREGATE_ONLY,
            evidence=msl.ReadinessEvidence(aggregate_score_column="score"),
        ),
    )
    assert lock["tabular_contract"]["count_finite_value_counts"] == {
        "count_rep1": 2,
        "count_rep2": 0,
    }

    forged = deepcopy(lock)
    forged["readiness"]["state"] = msl.Readiness.COUNT_RECOMPUTABLE.value
    forged["readiness"]["evidence"].update(
        {
            "count_lineage_columns": count_columns,
            "identity_resolution_spec_sha256": "1" * 64,
            "transformation_spec_sha256": "2" * 64,
            "wt_control_spec_sha256": "3" * 64,
        }
    )
    _refresh_integrity_hashes(forged)
    with pytest.raises(
        msl.SourceLockError,
        match="count_rep2.*has no finite values",
    ):
        msl.validate_source_lock(forged)


def test_persisted_readiness_rejects_mixed_numeric_annotation_column():
    score_columns = ["score", "score_rep1", "mixed_annotation"]
    metadata = _metadata(score_columns=score_columns)
    header = [*msl.FIXED_VARIANT_COLUMNS, *score_columns]
    rows = [
        [f"{URN}#1", "c.1A>G", "NA", "p.Ala1Gly", "0.2", "0.1", "1"],
        [
            f"{URN}#2",
            "c.2A>G",
            "NA",
            "p.Ala2Gly",
            "-1.1",
            "-1.0",
            "pathogenic",
        ],
    ]
    scores = ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()
    lock, _ = _build(_responses(metadata=metadata, scores=scores))
    assert lock["tabular_contract"]["score_finite_value_counts"]["mixed_annotation"] == 1
    assert lock["tabular_contract"]["score_invalid_value_counts"]["mixed_annotation"] == 1

    forged = deepcopy(lock)
    forged["readiness"]["evidence"]["aggregate_score_column"] = "mixed_annotation"
    _refresh_integrity_hashes(forged)
    with pytest.raises(
        msl.SourceLockError,
        match="aggregate score column has invalid values",
    ):
        msl.validate_source_lock(forged)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda lock: lock["metadata_contract"].__setitem__(
                "processing_state",
                "failed",
            ),
            "processing_state must remain success",
        ),
        (
            lambda lock: lock["metadata_contract"].__setitem__(
                "superseding_score_set",
                {"urn": "urn:mavedb:99999999-a-1"},
            ),
            "superseding record",
        ),
        (
            lambda lock: lock["metadata_contract"]["license"].__setitem__(
                "active",
                False,
            ),
            "active license",
        ),
        (
            lambda lock: lock["metadata_contract"].__setitem__(
                "private",
                True,
            ),
            "public/non-private",
        ),
    ],
)
def test_coherent_metadata_state_rewrites_are_rejected(mutation, message):
    lock, _ = _build()
    forged = deepcopy(lock)
    mutation(forged)
    _refresh_integrity_hashes(forged)
    with pytest.raises(msl.SourceLockError, match=message):
        msl.validate_source_lock(forged)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda artifact: artifact.__setitem__("status", 404),
            "status must be HTTP 200",
        ),
        (
            lambda artifact: artifact.__setitem__(
                "content_type",
                "text/html",
            ),
            "content_type must be text/csv",
        ),
        (
            lambda artifact: artifact.__setitem__(
                "transfer_encoding",
                "chunked",
            ),
            "both HTTP framing headers",
        ),
        (
            lambda artifact: artifact.__setitem__("attempts", 0),
            "attempts must be a positive integer",
        ),
        (
            lambda artifact: artifact.__setitem__("wire_sha256", "0" * 63),
            "wire_sha256 must be a lowercase SHA-256",
        ),
    ],
)
def test_coherent_artifact_rewrites_are_rejected(mutation, message):
    lock, _ = _build()
    forged = deepcopy(lock)
    mutation(forged["source_artifacts"]["scores"])
    _refresh_integrity_hashes(forged)
    with pytest.raises(msl.SourceLockError, match=message):
        msl.validate_source_lock(forged)


def test_exact_nested_schemas_reject_coherently_rehashed_extensions():
    lock, _ = _build()
    nested_paths = (
        ("readiness",),
        ("readiness", "evidence"),
        ("api",),
        ("api", "openapi_artifact"),
        ("source_artifacts",),
        ("source_artifacts", "scores"),
        ("metadata_contract",),
        ("metadata_contract", "license"),
        ("metadata_contract", "dataset_columns"),
        ("metadata_contract", "targets", 0),
        ("tabular_contract",),
        ("mapping_contract",),
    )
    for path in nested_paths:
        forged = deepcopy(lock)
        target = forged
        for part in path:
            target = target[part]
        target["unexpected_field"] = "forged"
        _refresh_integrity_hashes(forged)
        with pytest.raises(
            msl.SourceLockError,
            match="exact nested schema",
        ):
            msl.validate_source_lock(forged)


def test_source_bundle_binds_every_field_except_itself():
    lock, _ = _build()
    payload = msl._source_bundle_payload(lock)
    assert set(payload) == set(lock) - {"source_bundle_sha256"}
    assert payload["source_artifacts"]["scores"]["attempts"] == 1
    assert payload["metadata_contract"]["processing_state"] == "success"

    tampered = deepcopy(lock)
    tampered["source_artifacts"]["scores"]["attempts"] = 2
    assert msl.canonical_sha256(msl._source_bundle_payload(tampered)) != lock["source_bundle_sha256"]
    with pytest.raises(msl.SourceLockError, match="source bundle hash differs"):
        msl.validate_source_lock(tampered)

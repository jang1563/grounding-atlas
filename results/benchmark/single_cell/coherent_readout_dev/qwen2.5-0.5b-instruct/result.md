# Coherent binary readout: Level-0 report

Status: **DEVELOPMENT_READOUT_FORMAT_INVALID**

Mode `development`; 8 donors, 16 items, 64 records.

Global full-vocabulary format adherence: **0.047** (required >= 0.950).

| readout / family | format | O eq | R eq | I eq | item guardrail | result |
|---|---:|---:|---:|---:|---:|---:|
| `cytotoxic_state::unmodified` | 0.000 | FAIL | FAIL | FAIL | FAIL | **FAIL** |
| `lineage::unmodified` | 0.094 | FAIL | FAIL | FAIL | FAIL | **FAIL** |

Design SHA-256: `fd85775771dcb33ef25f7913c4862c610db47b7c8eceb433ac168263762354ab`.

Raw-record canonical SHA-256: `c834c2eb26ed3f20359f0d5317d2b6d073e19ed42d323d202f98ba10174b0035`.

Analyzer code SHA-256: `67ddbc3a6abfd0e5112b1bffeb3d3d8b88a042f12ec32c220608c3b08aed212f`.

Level 0 measures only whether the two-token interface satisfies the registered conditional measurement gates. It does not adjudicate calibration, biological validity, knowledge, integration, gain, activation, or a physical law. Exact sign-flip conclusions remain conditional on the required but unverified donor-effect sign-symmetry assumption.

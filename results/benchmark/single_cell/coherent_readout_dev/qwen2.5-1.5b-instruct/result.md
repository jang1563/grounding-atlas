# Coherent binary readout: Level-0 report

Status: **DEVELOPMENT_READOUT_FORMAT_INVALID**

Mode `development`; 8 donors, 16 items, 64 records.

Global full-vocabulary format adherence: **0.094** (required >= 0.950).

| readout / family | format | O eq | R eq | I eq | item guardrail | result |
|---|---:|---:|---:|---:|---:|---:|
| `cytotoxic_state::unmodified` | 0.062 | FAIL | FAIL | FAIL | FAIL | **FAIL** |
| `lineage::unmodified` | 0.125 | FAIL | FAIL | FAIL | FAIL | **FAIL** |

Design SHA-256: `e4747089d87132cd6068b3d143c6a707bd3e4db772a25a3ba9457afbda36f035`.

Raw-record canonical SHA-256: `c68a052ee103b20c463243b6205d910a0bc4bae287270003d2b848d6343a0842`.

Analyzer code SHA-256: `67ddbc3a6abfd0e5112b1bffeb3d3d8b88a042f12ec32c220608c3b08aed212f`.

Level 0 measures only whether the two-token interface satisfies the registered conditional measurement gates. It does not adjudicate calibration, biological validity, knowledge, integration, gain, activation, or a physical law. Exact sign-flip conclusions remain conditional on the required but unverified donor-effect sign-symmetry assumption.

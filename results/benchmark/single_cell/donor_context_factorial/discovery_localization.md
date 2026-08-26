# GSE96583 post-hoc target-token localization

This artifact recomputes target-token strata from the hash-bound raw
checkpoint. It is descriptive and was not a confirmatory target-gene
test.

## T_plus_cytotoxic / T_TCR_CD8

Donor-level component mean: `+0.114219`.

| target | cells | donors | donor-equal stratum mean | contribution | signed share | negative / zero / positive |
|---|---:|---:|---:|---:|---:|---:|
| CD3D | 25 | 8 | +0.093307 | +0.075547 | 66.1% | 2 / 4 / 19 |
| CD8A | 3 | 3 | +0.347500 | +0.032578 | 28.5% | 0 / 0 / 3 |
| CD8B | 3 | 3 | +0.042500 | +0.003984 | 3.5% | 0 / 0 / 3 |
| CD3E | 1 | 1 | +0.067500 | +0.002109 | 1.8% | 0 / 0 / 1 |

## T_plus_cytotoxic / cytotoxic_effector

Donor-level component mean: `-0.064844`.

| target | cells | donors | donor-equal stratum mean | contribution | signed share | negative / zero / positive |
|---|---:|---:|---:|---:|---:|---:|
| CCL5 | 19 | 8 | -0.005469 | +0.001094 | -1.7% | 5 / 9 / 5 |
| GNLY | 7 | 4 | -0.231667 | -0.050078 | 77.2% | 7 / 0 / 0 |
| GZMB | 4 | 3 | -0.135833 | -0.015859 | 24.5% | 3 / 1 / 0 |
| GZMA | 1 | 1 | +0.000000 | +0.000000 | -0.0% | 0 / 1 / 0 |
| GZMK | 1 | 1 | +0.000000 | +0.000000 | -0.0% | 0 / 1 / 0 |

## NK_receptor_plus_cytotoxic / NK_receptor_identity

Donor-level component mean: `-0.052812`.

| target | cells | donors | donor-equal stratum mean | contribution | signed share | negative / zero / positive |
|---|---:|---:|---:|---:|---:|---:|
| KLRD1 | 10 | 5 | -0.059083 | -0.024271 | 46.0% | 4 / 5 / 1 |
| FCGR3A | 6 | 5 | -0.102000 | -0.025417 | 48.1% | 5 / 1 / 0 |
| KLRC1 | 6 | 5 | -0.010000 | -0.002083 | 3.9% | 1 / 3 / 2 |
| KLRC2 | 1 | 1 | -0.025000 | -0.001042 | 2.0% | 1 / 0 / 0 |
| KLRF1 | 1 | 1 | +0.000000 | +0.000000 | -0.0% | 0 / 1 / 0 |

## NK_receptor_plus_cytotoxic / cytotoxic_effector

Donor-level component mean: `-0.116354`.

| target | cells | donors | donor-equal stratum mean | contribution | signed share | negative / zero / positive |
|---|---:|---:|---:|---:|---:|---:|
| GNLY | 16 | 7 | -0.157024 | -0.109479 | 94.1% | 14 / 2 / 0 |
| CCL5 | 4 | 2 | +0.040000 | +0.006667 | -5.7% | 0 / 1 / 3 |
| NKG7 | 3 | 3 | -0.108333 | -0.013542 | 11.6% | 2 / 1 / 0 |
| GZMB | 1 | 1 | +0.000000 | +0.000000 | -0.0% | 0 / 1 / 0 |

## Quantization and next-test feasibility

- Distinct raw outputs: `7`; the four values 0.15/0.25/0.75/0.85 account for `93.2%` of calls.
- Selected receptor-context cells containing GNLY, NKG7, and CCL5 in the top 50 exist in all eight donors; minimum per donor: `1`.

## Boundary

Target identity was selected from expression rank and inspected after the primary run. These strata motivate prospective single-token hypotheses but do not isolate biological gene effects, homogeneous pathways, latent knowledge, hidden-state activation, or a physical law.

# PBMC68k transfer: marker-program masking versus matched deletion

This experiment moves from prompt auditing to a biological intervention.
A label-matched CD8 or NK marker program frozen from the earlier
PBMC3k builder was masked in an independently annotated PBMC68k Donor A
cell. The control arm
masked the same number of source-neutral genes matched within that cell.
Program and control calls were contemporaneously randomized across all four
order × queried-target prompt forms.

## Primary result

- Cells: `119` (CD8 `88`, NK `31`)
- Model calls: `952`; parse rate: `100.0%`
- Control-adjusted program effect: **+0.294 [+0.256, +0.332]**
- Paired sign-flip p-value: `2e-05`

Positive values mean that deleting the frozen marker program harms the
reference-class probability more than deleting equally many matched
non-program genes.

| class | control P(correct) | program P(correct) | effect (95% CI) | positive cells |
|---|---:|---:|---:|---:|
| CD8 | 0.734 | 0.612 | +0.123 [+0.096, +0.152] | 78/88 |
| NK | 0.729 | 0.264 | +0.466 [+0.394, +0.536] | 30/31 |

Across cells, the paired four-form effect was positive for `108/119`, zero for `10`, and negative for `1`.

## Prompt-factor boundary

| prompt form | effect (95% CI) |
|---|---:|
| `ab_pa` | +0.244 [+0.182, +0.306] |
| `ab_pb` | +0.308 [+0.247, +0.368] |
| `ba_pa` | +0.334 [+0.291, +0.373] |
| `ba_pb` | +0.291 [+0.226, +0.355] |

Order interaction (AB minus BA): -0.036 [-0.073, +0.001]
Queried-target interaction (P(A) minus P(B)): -0.011 [-0.081, +0.061]
Both interactions inside the preregistered ±0.03 equivalence margin: **false**.

## Controls and sensitivities

- Strict matching sensitivity (all rank distances ≤10, expression distances ≤1.0): n=`99` (CD8 `86`, NK `13`), effect +0.287 [+0.225, +0.345]
- Orientation-averaged AUROC, control/program masks: `0.940`/`0.268`
- Orientation-averaged Brier, control/program masks: `0.111`/`0.303`

## Exploratory biological localization

These post-hoc summaries localize the preregistered aggregate; they are
not independent tests.

| reference annotation | n | control P(correct) | program P(correct) | effect |
|---|---:|---:|---:|---:|
| CD56+ NK | 31 | 0.729 | 0.264 | +0.466 |
| CD8+ Cytotoxic T | 46 | 0.616 | 0.466 | +0.150 |
| CD8+/CD45RA+ Naive Cytotoxic | 42 | 0.864 | 0.771 | +0.093 |

- Most frequently masked CD8-program genes: `CD3D` 67/88, `CD3E` 42/88, `GZMK` 24/88.
- Most frequently masked NK-program genes: `GNLY` 31/31, `NKG7` 30/31, `GZMB` 20/31.
- Because genes were co-masked and mask count is confounded with program/class,
  these frequencies do not identify a necessary gene or a dose-response curve.

## Interpretation boundary

This is a causal input-to-output masking contrast for one external cohort.
The operational programs mix lineage-identity and cytotoxic-effector
markers, especially in the NK arm, so this result does not isolate a
pure lineage mechanism from a cytotoxic-state mechanism.
It is not an unassisted annotation benchmark because the reference label
selects which frozen program is masked. It does not establish a hidden-state
activation route, corpus exposure, a physical law, or multi-donor
generalization. Technical barcode suffixes are not treated as donors.

# Asset Reuse Map - Bio_Grounding_Eval

Which existing project feeds which workstream. All paths under `/Users/jak4013/Dropbox/Bioinformatics/Claude/`. Read the source READMEs/HANDOFFs before pulling. Re-anchor everything from safety framing to capability framing.

## WS1 - instrument (MEASURE)
| Source (real path) | What to pull | Reuse vs adapt |
|---|---|---|
| `FRT_Pilot_Execution/` | a **protein-only** annotation instrument (supplies axis A identity-resolution and axis E channel/action-policy); aggregate outcomes; agentic tool-mode harness. Cross-representation is a design goal, not what FRT measured. Start at `README.md`, `OUTCOMES.md`, `HANDOFF.md`, `pilots/`, `public_release/`. | **REUSE** the harness + aggregate numbers. **ADAPT** the framing (safety -> capability). **DO NOT** pull `disclosure/` (raw bypass material; capability project uses aggregate only). |
| `Narrow_Model_Safety_Eval/` (NMSE) | the AUROC-0.9807 separability probe (frozen ESM-2 650M, 5-fold CV, 60+60 sequences; ESM-3 0.942). `results/`, `src/`, `data/`, `huggingface/`. Public on GitHub+HF. | **REUSE** as the B-axis ceiling anchor (the signal is in ESM-2's representation). Note: the LLM side was never measured, so this is the encoding-vs-expression question, not a verbalization-gap finding. |

## WS2 - signal engine (MAKE SIGNAL)
| Source (real path) | What to pull | Reuse vs adapt |
|---|---|---|
| `Negative_result_DB/` | NullAtlas substrate + paired-probe benchmark; `paper_nullatlas_rl/` (NullAtlas RL home). Many HANDOFF/NEURIPS docs; `NAIRR_GPU_AWARD_2026-05-04.md`. | **REUSE** the negative-evidence generation method; **EXTEND** from claim-level to representation-grounding (representation -> verifiable property pairs). |
| `pbs_theory/` | Publication Bias Score theory (the "why positive-only literature has no signal" backbone). | **REUSE** as the motivation/why for WS2. |
| `BioRLHF/` (now NegBioRL) | SFT/DPO/GRPO pipeline (`biorlhf/`, `PLAN_DEFINITIVE.md`, `PROGRESS.md`). | **REUSE** the pipeline; the NegBioRL finding (calibration not bakeable, retrieval wins) is WS3's first data point. |
| `BioProtocolBench/` | LabCraft deterministic agent grader (`labcraft.egg-info` -> the `labcraft` package). ~10K LOC, regex+exact-match, human baseline. | **REUSE** as the non-LLM-judge scoring backbone (used by WS1 + WS2 verifiability checks). |
| `Retracted_concern_DB/` | a second negative-evidence source. | Optional secondary source for WS2. |

## WS3 - decision map (MAP THE LINE)
| Source (real path) | What to pull | Reuse vs adapt |
|---|---|---|
| `BioRLHF/` | the training pipeline for the local PoC (train-the-skill arm). | REUSE. |
| `Evo2/` + ESM-2 (via NMSE) | the SFMs to orchestrate (orchestrate arm). | REUSE as tools. |
| `FRT_Pilot_Execution/` agentic tool-mode | the orchestrate-with-trust harness. | REUSE (aggregate framing). |
| `Calibrated_Permissioning_for_Biological_AI/`, `OverRefusal/`, `AmbiguityCasebook/`, `ConstitutionRules/` | the calibration/over-refusal family (the retrieve-vs-bake calibration data point). | Optional; supports the "retrieve the knowledge" arm. |

## Domain-depth assets (the credible-translator bridge, for the application narrative)
`SpaceOmicsBench/`, `GeneLab_benchmark/`, plus the spaceflight Perturb-seq / single-cell / spatial work (in the Mason-lab project tree) = proof JK can read the SFM outputs because he produces that data. Not code to pull; cite as positioning.

## Hard caveats
- **FRT disclosure boundary:** `FRT_Pilot_Execution/disclosure/` holds raw bypass material held for responsible disclosure. This capability project NEVER ingests it. Aggregate model-behavior numbers only.
- **NegBioRL naming:** the project was "BioRLHF"; current name is **NegBioRL**. Use NegBioRL in writeups.
- **NMSE naming:** call it an "Evaluation Report," not a "system card" (it evaluates third-party models).
- Verify any number against the source before putting it in application text (see `PRELIMINARY_DATA.md` for the vetted set).

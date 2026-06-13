# Perturbation-FM failure: signal/data vs architecture/identifiability (deep-research, 2026-06-12)

Verified archive (deep-research task wyaal7u66, 109 agents, 26 sources, 25 claims verified, 21 confirmed / 4 killed, primary-source). Feeds `ai_scientist_roadmap.md`. No em dashes. Honest framing kept: this leans toward the architecture intuition but it is NOT settled, and it cuts AGAINST the "a better training signal (incl. falsification reward) will build a discoverer" optimism.

## Verdict

GENUINELY OPEN, leaning unresolved. The EMPIRICAL FAILURE is settled (high confidence, 4 independent peer-reviewed benchmarks). The CAUSE (signal/data vs architecture vs metric) is unadjudicated because the three confounds are entangled in all real-data benchmarks. The decisive experiment has not been run.

## Established empirical fact (rock-solid, cite freely)

- No single-cell/perturbation FM (scGPT, scFoundation, scBERT, Geneformer, UCE, GEARS, CPA, GeneCompass) reliably beats deliberately simple baselines (mean/additive/no-change/linear) on unseen single OR double perturbations. Covers the epistatic/combinatorial regime.
  - Ahlmann-Eltze, Huber & Anders, Nat Methods 22:1657-1661 (2025): 5 FMs + GEARS + CPA vs linear baselines, "None outperformed the baselines"; on genuinely synergistic/non-additive interactions, "the no change model simply predicting the unperturbed state outperformed the other methods in all metrics" (Norman 2019 K562, doubles unseen).
  - Kernfeld et al., Genome Biol 26:388 (2025): 11 datasets, "it is uncommon for expression forecasting methods to outperform simple baselines."
  - scPerturBench, Nat Methods s41592-025-02980-0 (Dec 2025): 27 methods x 29 datasets x 6 metrics vs 4 baselines. (Cite for design/roster; its specific per-category superlatives did NOT survive verification, see Refuted.)
  - Wong/Hill/Moccia, Bioinformatics 41(6):btaf317 (2025): CRISPR-informed mean beats GEARS (delta 0.08, p=9.3e-4) and scGPT (delta 0.11, p=8.1e-6) on held-out single-gene CRISPR.
- THE most direct signal-vs-architecture datapoint: FM pretraining provides NO measurable advantage over random init (scGPT fully-fine-tuned vs random-weight-init: Pearson-delta diff 0.004, p=0.89; PDED 0.01, p=0.75; 30 replicate models). Wong et al. 2025. Holds architecture fixed, varies only whether the pretraining signal is present -> the self-supervised signal as currently formulated adds nothing. (Single-source for this exact comparison; robust internally.)
- Zero-shot: scGPT/Geneformer underperform parameter-light baselines (HVG, scVI, Harmony) on clustering, batch integration, and even reconstruction; "scGPT without embeddings underperforms against a naive baseline of predicting the mean" on its OWN pretraining objective. Kedzierska et al. (MSR), Genome Biol 26:101 (2025). (Single-source but canonical/uncontested.)

## Architecture / identifiability camp (the better-supported reading)

- Interventional/counterfactual structure is provably NOT recoverable from i.i.d. observational data without structural assumptions, so observational pretraining cannot in principle learn interventional mechanism.
  - Tejada-Lapuerta et al., Nat Genet 57(4) (2025) / arXiv 2310.14935: "identifying the true interactions from observational data alone is not possible (without specific assumptions)."
  - Lopez et al., CLeaR 2023 / arXiv 2211.03553: disentangled causal latents "infeasible from i.i.d. measurements, without additional structure" (Locatello 2019 impossibility + nonlinear-ICA non-identifiability).
  - Lobentanzer et al., Mol Syst Biol 20:848-858 (2024) / arXiv 2401.09558: Pearl's ladder, "Generating interventional or even counterfactual inferences from observational data is a major challenge, if not impossible."
- Strongest direct mechanism-statement: FM masking objectives mask genes at random or by information-value, NOT because masked genes are downstream of predictor genes, so (unlike language) the data do not implicitly contain causal information. Lobentanzer et al. 2024. CAVEAT: the authors themselves hedge "not explored theoretically or empirically" (2024). 2026 interpretability preprints corroborate the correlation-not-mechanism side (Kendiukhov arXiv 2602.17532: attention captures co-expression not regulatory signal, causal ablation no degradation; SAE atlas arXiv 2603.02952: "minimal regulatory logic") but are NASCENT, unverified, and SAE work has seed-stability problems. Treat as direction-of-travel, not established.

## Signal / data camp (weaker; interested parties; does not cleanly reverse the verdict)

- GenBio AI, bioRxiv 2026.02.18.706454 (Feb 2026): >600 models, "some foundation models fail to outperform simple baselines, others significantly improve"; "with sufficient data, these models approach fundamental performance limits" (closes 77% of train-mean-to-experimental-error gap). CAVEATS: textbook COI (all authors @genbio.ai, Xing/Song/Bar-Joseph; paper titled "Foundation Models Improve..."); non-peer-reviewed; winners are interactome-based FMs NOT scGPT/Geneformer-class; AND in the genuinely UNSEEN-perturbation regression "no embedding convincingly outperformed the negative controls." Near-optimality holds in the SEEN regime only.
- Arc STATE, bioRxiv 2025.06.26.661135 (NeurIPS 2025): claims cross-cell-context extrapolation (same perturbation in NEW cell context) via a learned cell-embedding manifold; reports >30% improved discrimination (Arc PR restates as ~50%). CAVEATS: STATE's axis is unseen-CELL-CONTEXT, NOT unseen-PERTURBATION (the regime that condemned GEARS/scGPT); robustness disputed ("Virtual Cells Need Context" bioRxiv 2026.02.04.703804: generalization "remains questionable"; VCC found most models underperform a mean baseline on MAE); the developer claim that STATE "consistently" beats linear baselines was REFUTED in verification (1-2). An in-house Perturb-seq run confirms: STATE on the strict Arc cell_eval / PDS metric was near-random (PDS ~0.52).

## The crux: why it cannot be cleanly resolved now

The architecture camp's OWN authors say real data cannot separate the two: "if a causal approach does not provide good performance in unseen environments, we do not know whether the assumptions imposed on the model are faulty, or whether the assumptions are accurate but the inference procedure is faulty ... Distinguishing between these two cases would require evaluating ... on data where model assumptions are known to hold, which could be generated through simulations." (Tejada-Lapuerta et al., Nat Genet 2025.) So architecture-vs-signal is structurally non-adjudicable with real noisy data; you need SIMULATION ground truth.

The metric crisis sits on top: Pearson-on-pseudobulk may be the wrong yardstick (Wenkel/Shift bioRxiv 2025.10.20.683304 show DIFFERENT DL models beat the MEAN baseline under rank metrics; "The Metric Picks the Winner" arXiv 2606.12639). BUT the strong metric-rescue claims were REFUTED in verification, and no verified work shows the indicted FMs beating the HARDER additive/no-change baselines under better metrics. So the metric confound is a real open question that has NOT rescued the FMs.

## The adjudicating experiment (the genuine white space)

No published, verified study directly probes whether single-cell/perturbation FM embeddings INTERNALLY encode held-out interventional/counterfactual structure, nor runs the Tejada-Lapuerta simulation test (known causal ground truth). The 2026 probing preprints (Kendiukhov; SAE atlas) are nascent/unverified and lean correlation-not-mechanism. This is the single highest-value open gap.

## What this means for the roadmap AND for the falsification idea (honest)

- For the roadmap: the SETTLED negative (FMs lose to simple baselines; pretraining adds nothing over random init; no-change wins on synergy) is the most trustworthy datapoint in the corpus and sharpens the existing "acceleration vs discovery / no extrapolation under intervention" diagnosis with hard citations. The CAUSE is open and leaning-architecture-but-unprovable-with-real-data. Add the Tejada-Lapuerta non-separability point: it is the cleanest statement of why the field is stuck.
- For the falsification-RLVR idea (skeptical): this CUTS AGAINST the optimistic read. The most direct evidence (pretraining signal adds nothing) and the identifiability theory both suggest the failure is at the representation/architecture level. If so, a falsification REWARD is just another training signal, and you cannot train your way out of an identifiability limit -> a falsification-trained model would not be expected to acquire interventional mechanism either. The "give it the right signal and it becomes a discoverer" bet is betting on the SIGNAL camp, which is the weaker, interested-party (COI), seen-regime-only side. Honest position: the falsification framing's strongest use is as a MEASUREMENT/ADJUDICATION tool (help run the missing experiment, score what a model rules out), NOT as a training recipe that turns a correlation-learner into a mechanism-learner.
- On the "encode-vs-express probe adjudicates signal-vs-architecture" claim, it is now triply undercut. (1) The session's own Bio_Grounding finding: a probe reading a property from embeddings did NOT exceed a no-chemistry substring probe, so "encode" can be interpolation/shortcut not mechanism. (2) Tejada-Lapuerta: real-data cannot separate architecture from inference; you need simulation. (3) The existing probing preprints lean correlation-not-mechanism with seed-stability issues. A real-data probe on STATE is at best SUGGESTIVE, not decisive; the decisive version needs simulation with known causal ground truth.

## Refuted claims (do NOT cite)
- "Wong et al. negative verdict is a one-or-two-gene metric-fragility artifact" (0-3).
- "scPerturBench linearModel achieved highest overall accuracy across all categories on combinatorial perturbations" (0-3) and "FMs outperform on single perturbations only with large fine-tuning data" (0-3): cite scPerturBench for design/roster, not for ranking specifics.
- "STATE consistently beats simple linear baselines" (1-2, developer-origin).

## Key sources
Negative result: Nat Methods s41592-025-02772-6 (Ahlmann-Eltze/Huber); Genome Biol 26:388 (Kernfeld); Nat Methods s41592-025-02980-0 (scPerturBench); Bioinformatics btaf317 (Wong, the pretraining-null); Genome Biol 26:101 (Kedzierska, zero-shot). Architecture: arXiv 2310.14935 (Tejada-Lapuerta, Nat Genet 2025, the non-separability crux); arXiv 2211.03553 (Lopez CLeaR 2023); arXiv 2401.09558 (Lobentanzer MSB 2024). Signal/data: bioRxiv 2026.02.18.706454 (GenBio, COI); bioRxiv 2025.06.26.661135 (Arc STATE). Metric/probing (nascent): 2025.10.20.683304, arXiv 2606.12639, arXiv 2602.17532, arXiv 2603.02952, bioRxiv 2026.02.04.703804.

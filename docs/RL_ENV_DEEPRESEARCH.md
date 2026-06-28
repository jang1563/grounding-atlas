# Deep research: the post-training RL-environment lever for biological FMs

Date 2026-06-28. Method: deep-research workflow (104 agents, 3-vote adversarial verification;
12 verified findings, 2 refuted). Question: does building a post-training RL environment to
elicit/improve emergent capability in scientific/biological FMs OVERTURN or EXTEND our
"route, don't train" read-out finding ([REPORT.md](REPORT.md))?

## Decisive answer
The RL / generative-alignment lever is real, named, and buildable, but as of mid-2026 it does
NOT overturn "route, don't train". The literature frames it as a SEPARATE, not-yet-winning
lever whose decisive head-to-head against external orchestration is EXPLICITLY UNRESOLVED
(Ferruz survey: "it is still not clear which set of approaches lead to better performance under
which conditions and extensive benchmarking is needed"). That unresolved head-to-head IS the
white space, and it is the natural extension of our route-vs-orchestrate-vs-train framing into
the generative/RL regime.

## Three artifact families (the buildable blueprints)

### 1. RLXF / ProtRL (internalized policy-gradient)
- RLXF (Romero Lab, bioRxiv 2025.05.02.651993): policy = generative protein LM (ESM-2-650M; also
  a sequence VAE), RLHF-style SFT-then-PPO (PPO chosen over DPO, "PPO excels at complex generative
  tasks"), action = generate CreiLOV variants (~Hamming-5). Two model copies (frozen reference +
  trainable policy).
- Reward = a TRAINED in-silico discriminative predictor ensemble (~100 MLPs on 6,925 CreiLOV
  deep-mutational-scanning sequences, Spearman 0.93 on held-out 5-mutants) used AS the reward
  model. This is EXACTLY "a cheap discriminative specialist as the reward" (= our read-out head).
- Result: measurably RE-WEIGHTS output toward higher reward (aligned VAE 0.955 win-rate vs
  pretrained ESM-2); produced a REAL wet-lab-validated gain (most fluorescent CreiLOV/FbFP reported
  to date, a ~10-mutant), NOT reward-model overfitting.
- Mechanism (the decisive angle): the authors frame it as ELICITATION / INTEGRATION, not new
  capability: the aligned model FUSES the evolutionary knowledge ALREADY ENCODED in the pretrained
  pLM with experimental observations, surfacing synergistic (epistatic) mutations. This connects
  directly to our encode >> express thesis.
- ProtRL: REINVENT/DPO/GRPO on autoregressive pLMs, closed-loop wet-lab, low-nanomolar EGFR binders.

### 2. VIDD (internalized weight-update for diffusion; arXiv 2507.00445, ICLR 2026, divelab/VIDD)
- Off-policy forward-KL value-weighted iterative distillation; base = FROZEN-checkpoint open
  diffusion model (EvoDiff protein, GDSS molecules, DNA-diffusion).
- Reward = NON-DIFFERENTIABLE scientific oracles (AlphaFold-Multimer ipTM, Enformer HepG2,
  QuickVina2 docking, DSSP).
- Result: internalized fine-tuning BEATS inference-time Best-of-N guidance on in-silico reward:
  protein pLDDT 0.82 vs 0.38, PD-L1 ipTM 0.82 vs 0.27, DNA enhancer Pred-Activity 8.28 vs 1.30,
  BUT the paper itself articulates a train-time-vs-inference-time TAX (a tradeoff, not a free win).

### 3. g-DPO / ProteinGuide (the DPO + the frozen-steering branches)
- g-DPO (arXiv 2510.19474, NeurIPS 2025): DPO on masked pLMs scored via pseudo-log-likelihood;
  attacks DPO's pair-count scalability obstacle.
- ProteinGuide (arXiv 2505.04823) / 2505.15093: COUNTERVAILING evidence. External inference-time
  Bayesian guidance of a FROZEN model BEATS internalized fine-tuning (DPO/RL) in the LOW-DATA
  regime. This is the regime-dependence that makes the head-to-head non-trivial.

## The route-vs-train taxonomy (Ferruz survey, arXiv 2511.21476, Curr Opin Struct Biol 2026)
Partitions all protein-design steering into parameter-UPDATING (RL/SFT, modifies weights to shift
p(x) -> p(x|y)) vs parameter-FIXED (conditional generation, Bayesian/output guidance, RAG,
activation steering, sampling control) = our route-vs-train axis verbatim. Two load-bearing points:
- Reward/verifier reliability is the SINGLE biggest limiting factor for RL/preference post-training
  of protein generative models, NOT the policy-gradient machinery. So the binding risk for us is
  whether our property head is a good enough reward, not whether we can run PPO.
- Whether alignment genuinely uncovers novel functional solutions or merely recombines/re-weights
  training patterns is posed as an OPEN question requiring controlled Parameter-Updating-vs-
  Parameter-Fixed comparison.

## Elicit vs re-weight (angle 4): OPEN, leaning elicitation
- RLXF = explicit elicitation/integration of pre-encoded knowledge (above).
- Ferruz: frozen-model steering "cannot extrapolate beyond representations the base model has not
  learned" (a ceiling argument that applies to the steering branch).
- Refuted (0-3): the strongest form of "RL only sharpens sampling, never expands the frontier at
  large pass@k" (an over-strong reading of arXiv 2504.13837) did NOT survive verification, so
  "only re-weights" is NOT settled. The question is genuinely open, which is what makes a clean
  controlled test worth running.

## The white space (what nobody has cleanly done)
The contamination-safe, MULTI-REGIME HEAD-TO-HEAD: on a generative bio FM, compare
- internalized RL (our property head used AS the reward; PPO/DPO/distillation), vs
- external inference-time guidance of the FROZEN FM (the SAME reward), vs
- the base model,
ACROSS regimes (reward-data size, reward quality), with a TRUE-fitness / novel-held-out evaluation
(a held-out oracle, not the training reward). This is exactly the benchmark the Ferruz survey says
is missing, and it is our route-vs-orchestrate-vs-train question asked in the generative/RL regime.

## Minimal buildable cell
Frozen open SFM (EvoDiff/ESM-2 protein, or GDSS / a molecular generator) + a trained property head
used AS the reward + PPO (RLXF) / DPO (g-DPO) / value-weighted distillation (VIDD). This is exactly
what RLXF and VIDD already stand up, so the BUILD is de-risked; the genuinely under-explored part is
the contamination-safe head-to-head, not the build itself.

## Sources (3-vote verified unless noted)
- RLXF: bioRxiv 2025.05.02.651993 + github.com/RomeroLab/RLXF
- VIDD: arXiv 2507.00445 + github.com/divelab/VIDD
- g-DPO: arXiv 2510.19474 (NeurIPS 2025)
- ProteinGuide: arXiv 2505.04823; 2505.15093
- Ferruz survey: arXiv 2511.21476 (Curr Opin Struct Biol 2026)
- DRAKES: arXiv 2410.13643
- Refuted: "RLXF == ChatGPT-style RLHF" (0-3, over-claim); the strong pass@k frontier claim of
  2504.13837 (0-3).

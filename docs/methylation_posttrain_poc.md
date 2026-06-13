# Methylation to age: does post-training close the web-zero verbalization gap

*Design doc, 2026-06-12 (redesigned). The post-train cell of the decision map. REDESIGN rationale: methylation to age has a clean ground truth (chronological age), so an rBio-style soft-verifier GRPO is over-engineering; SFT is more direct, stable, and light, and it removes the GRPO-induced bottlenecks (RL stack, reward hacking, Qwen spurious-reward, control-arm explosion, multi-cluster). GRPO with a clock soft-verifier is moved to Phase 2, where it answers the genuinely different question (can an SFM teacher close the gap WITHOUT labels). No em dashes.*

## The question

Not "can we build an age predictor" (the epigenetic clock already is one). Whether post-training closes the ENCODING-to-VERBALIZATION gap on a WEB-ZERO numeric mapping, and whether closure is weaker than for a web-documented mapping. Load-bearing prior (our own measurement, `results/methylation_rung.md`): methylation to age is linearly ENCODED in an open model's hidden states (probe AUROC 0.685 vs ceiling 0.701) but NOT verbalized (output 0.487, chance). That ~0.20 gap is what we try to move.

---

## Phase 1: SFT x anchor-swap (the clean, light test)

Because chronological age is a true label, no verifier is needed. Plain supervised fine-tuning on (beta-vector rendering -> true age) directly answers "does post-training close it", and the numbers-paper result (arXiv:2602.07812) says SFT is where most of the closure happens anyway (plain FT did the bulk; the probe-alignment trick added only +3.2%).

### Arms (4 core + 2 reference)

| arm | what | tests |
|---|---|---|
| base (ref) | no training, verbalized output | the 0.487 start |
| probe (ref) | linear probe on hidden states | the 0.685 encoded ceiling |
| SFT | LoRA SFT, beta-text -> age | does ordinary post-training close it |
| SFT + probe-loss | SFT with auxiliary probe alignment (L_LM + beta·L_cls, beta~0.02, arXiv:2602.07812) | does aligning output to the already-present internal rep beat SFT alone |

Each of {SFT, SFT+probe-loss} is run TWICE, the anchor-token swap:
- gene-named (web-rich): CpGs labeled by their gene (ELOVL2, FHL2, KLF14, the canonical web-documented clock age genes).
- anonymized (web-zero): identical beta-values under opaque IDs (feature_0001 ...).

So the matrix is {SFT, SFT+probe} x {gene-named, anon} = 4 training runs, plus the 2 references. The numeric content is IDENTICAL across the swap; only web-anchoring differs.

### Measurement

Continuous (age is continuous, more informative than the binary rung): MAE and Pearson r of verbalized age vs true age. Plus binary AUROC (age > cohort median) for continuity with the 16-rung map. Test on HELD-OUT samples and a novel rendering FORMAT (easy-to-hard split, arXiv:2312.01037) so closure means extraction, not memorization.

### Prediction (falsifiable)

The web-exposure law predicts: closure is LARGER for gene-named than anonymized, with identical numerics. If gene-named >> anon, verbalization rides on a web-rich anchor token, not on the encoded magnitude (the central claim). If gene-named == anon, the bottleneck is the numeric read-out head, not web-anchoring, which is itself a clean publishable negative. Frequency-to-learnability literature (Allen-Zhu Physics 3.1 arXiv:2309.14316: web-absent facts ~0% extraction "regardless of fine-tuning"; Kandpal arXiv:2211.08411; Gekhman arXiv:2405.05904; Hier arXiv:2601.18468) predicts the anon arm closes little and may raise hallucination.

### Compute

LoRA SFT on Qwen2.5-3B, 4 short runs. Cayuga alone (single A100 or A40 per run, a few hours total). No GRPO stack, no rollout/reward server, no multi-cluster. This is the bottleneck-free path.

### Pre-step (training-free, now): in-context anchor-swap

Before any training, render gene-named vs anon and measure FRONTIER output AUROC/MAE (the same way we did single-cell gene-name vs anon). This gives the ceiling of what no-training prompting achieves and a first read on the swap direction, at near-zero GPU cost.

---

## Phase 2: SFM-teacher, label-free (the rBio-narrative version, heavy)

This is where the clock-as-soft-verifier belongs, and it answers a genuinely different and more interesting question: can an SFM teacher close the gap on a cohort with NO chronological-age labels (the situation rBio actually solved for perturbation). Only here do GRPO and its costs apply.

### Setup

GRPO post-train Qwen2.5-3B with a FROZEN epigenetic clock as the soft verifier on a label-free cohort. Reward = exp(-(â - c(x))^2 / 2σ^2), clock c(x) predicts on the TRUE methylome x (not on the policy answer; the â-fed version is methylome editing = Revive-Flow, not this test); σ = clock MAE (Horvath 3.6 yr; AltumAge 2.15/3.19 yr). Small format/mention/Brier-calib shaping (RLCR arXiv:2507.16806). This is the rBio move (predictor likelihood as continuous reward), ported from PerturbQA to a regression.

### Why GRPO's costs only live here

Because there is no true label in this cohort, the clock IS a proxy reward, so all the GRPO-specific controls become necessary: reward hacking (frozen clock, AltumAge robustness, clip, proxy-gold gap reported separately, Gao arXiv:2210.10760; ensemble-of-clocks disagreement, Karwowski arXiv:2407.14503); Qwen spurious-reward (shuffled-reward + wrong-clock control arms + Llama-3.2-3B replication, Shao arXiv:2506.10947). This is the multi-cluster, many-variant workload, and the place H100 max-parallel is genuinely needed.

### What it adds

Phase 1 answers "does post-training close it, and is it web-anchored". Phase 2 answers "can you do it with an SFM teacher and no labels", which is the directly transferable answer for the many biology properties that have no clean ground truth, and the rBio-aligned narrative for the application.

---

## Top citations

rBio1 (bioRxiv 2025.08.18.670981) Phase 2 template; LLMs Know More About Numbers than They Can Say (arXiv:2602.07812) the near-exact Phase 1 analog (probe >90% vs output 50-70%, auxiliary probe loss; plain FT does most); Allen-Zhu and Li Physics 3.1 (arXiv:2309.14316) the web-zero ceiling; Gao arXiv:2210.10760 + Karwowski arXiv:2407.14503 (Phase 2 reward overoptimization); Shao arXiv:2506.10947 (Phase 2 Qwen spurious-reward controls). Clocks: Horvath 2013 (Genome Biology 14:R115, 353 CpGs, MAE 3.6 yr) and AltumAge (npj Aging 2022, ~20k CpGs, MAE 2.15/3.19 yr).

## Connection to the program

Phase 1 fills the post-train cell of the decision map cheaply and answers the web-exposure question; pairs with `results/calibration_routing.md` (orchestrate arm). Phase 2 is the SFM-teacher / label-free extension, and the only part that needs the H100 fleet.

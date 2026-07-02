# Encode ≫ express: where post-training actually helps the LLM × SFM interface

*A short field report from the grounding-atlas / GroundBench project. Not a journal paper — a working
map, meant to be read in ten minutes. CC-BY-SA-4.0.*

## The one idea

Foundation models — language models and scientific foundation models (SFMs) alike — **encode far more
than they express**. A protein language model holds structure and function in its embeddings; an LLM
holds a property in its hidden states. Neither says it by default. The useful question is not "does the
model know it" but **what to do about the gap**: retrain the weights, retrieve, or orchestrate a
specialist — and *when does post-training actually earn its place?*

## What we measured (GroundBench)

Across 23 tasks / 9 modalities / 3 frontier models, with a cheap-specialist ceiling, an open-weight
probe, and the model's verbalized output:

- **The verbalization gap is real and large** — open models encode a property near the specialist
  ceiling while verbalizing it near chance.
- **But its mechanism is not a single "web-exposure law."** A controlled experiment (drop the textbook
  cell-type markers but keep real gene names) decomposes the gap into **token-familiarity / reasoning**
  and **mapping-documentation**, in a **capability-dependent mix**: the token-familiarity share rises
  monotonically with capability (Haiku 0.32 < Sonnet 0.49 < Opus 0.80). Weaker models recall documented
  markers; the frontier reasons over any familiar tokens. The effect is real; the mechanism is the
  correction.
- **An LLM cannot read a raw SFM embedding in-context** (prompt-pasted ESM-2 → chance, even few-shot),
  while a **trained read-out head** reads it. The interface, not the information, is the bottleneck.

## The landscape (what the literature shows)

Three mechanisms recur, across both LLMs and SFMs:

1. **Read-out / elicitation.** "Encode ≫ express" is established on the LLM side (the knowledge
   *acquisition–utilization gap*, Kazemnejad et al. 2305.14775; the *tuned lens*, 2303.08112; *eliciting
   latent knowledge* from quirky models, 2312.01037; *secret elicitation*, 2510.01070; and latent
   knowledge is the *strongest predictor of faster fine-tuning acquisition*, HR 2.6, 2601.18468). On the
   SFM side, instruction-tuning surfaces emergent properties into language (SEPIT 2410.03553; STELLA
   2506.03800; Biology-Instructions 2412.19191).
2. **Generative / preference alignment.** RLHF/DPO-style alignment of generative bio models (ProtRL/RLXF
   biorxiv 2025.05.02.651993; g-DPO 2510.19474; VIDD 2511.21476). The result that matters most:
   **external steering of a *frozen* model beats DPO fine-tuning in low-data** (~100–200 sequences;
   2505.15093). Arc Institute's **Proto** (2026.06.22.733870) is the canonical "orchestrate-a-frozen-FM"
   system — no fine-tuning, just external constraints + an optimizer.
3. **Cross-model bridges (LLM ↔ SFM).** "An LLM can't read a raw embedding; a trained bridge can" is now
   a crowded sub-field: BioVERSE 2510.01428, STELLA 2506.03800, Cell2Text 2509.24840 (Geneformer),
   MutaPLM 2410.22949 (ESM-2), ProteinGPT 2408.11363, ProteinCLIP (biorxiv 2024.05.14.594226). The
   standard recipe is a frozen SFM encoder → a lightweight linear/FF projection (± LoRA) → an LLM.

## The honest read

Our findings are **validated, not novel in isolation.** Encode ≫ express and the trained-bridge are
established; building *another* bridge adds little. Three over-claims were adversarially refuted and we
avoid them: probes are **not** truer than the model's output (the probing-classifier caveat, 2102.12452);
**bigger LLMs are not reliably better SFM-readers**; and no single decoding trick (e.g. prefill) is the
universal best elicitation.

Two clean distinctions fall out:
- **Train a thin interface vs retrain the model.** Training a *bridge / read-out head* (a small
  projection) **wins** for cross-modal access — this is our "orchestrate via a trained head." Fine-tuning
  the *model's weights* (DPO / full FT) **loses** to external steering in low-data. "Train wins nowhere"
  is precise *for in-weight* training; a learned *interface* is a different, winning lever.
- **The effect is settled; the mechanism and the trust are not.**

## The white space (our angle)

Nobody has, in one place:
1. **Fairly compared** external-orchestration (Proto-style) vs a learned bridge vs in-weight fine-tuning
   on the *same* task;
2. tested whether an elicited/bridged capability **transfers** to held-out domains (a general skill) or
   merely **memorizes** one (a contamination-safe train-domain → held-out-domain split);
3. added **calibrated permissioning** to the interface — *when should we trust the bridge's read versus
   defer to the specialist?*

The first two are method hygiene. The third is exactly our strength: GroundBench already shows the
a-priori, input-derived signal beats the model's self-confidence for deciding when to trust. So our
contribution is **not a new bridge — it is the measurement / routing / calibration layer on top of one.**

## What the layer-localization found

We ran the cheap warm-up: where, by layer, do two open-weight 8B co-primaries (Qwen3-8B, Llama-3.1-8B)
encode these properties, and is the read-out calibrated? Pre-registered, with shuffled-label
selectivity, a positive control, nested cross-validation, and a GC-residualization control
([docs/LAYER_LOCALIZATION_PREREG.md](LAYER_LOCALIZATION_PREREG.md)). The controls did the heavy lifting
by **killing two tempting over-claims**:

- **A large "DNA promoter" encode-vs-express gap (encoded 0.85-0.87 vs verbalized ~0.50) was reading GC
  composition, not promoter semantics.** The GC-residualized probe fails to beat the surface floor in
  both models (margin -0.19 / -0.21). Promoters are GC-rich; the probe found the GC.
- **A dramatic single-cell "familiar tokens encoded late, alien tokens early" shift (+0.68 on the raw
  AUROC peak) shrank to +0.17 once measured at the peak-SELECTIVITY layer.** The anonymized form's early
  raw peak was surface readability, not computation; its computed peak is mid-network. Llama shows no
  clear shift.

What survives is therefore trustworthy:

- **The encoded read-out is a far better router than the model's own output**, robustly across both
  models and every task (it ranks its own errors much better than the model's verbalized confidence
  does). This is the firmest result, and the one a calibration layer would rest on.
- **A clean encode-vs-express gap remains on single-cell type** (Qwen encodes the cell type at 0.96-0.98
  and verbalizes it near chance) - the cleanest surviving case once DNA fell.
- **Encoding depth is architecture-specific, not universal**: Qwen computes these properties deep
  (selectivity-peak depth 0.6-1.0), Llama shallow (0.0-0.4). There is no shared "mid-band," so a bridge's
  attach layer must be found per model, not assumed. A methodological note: the "excess over a positive
  control" gap measure only works when the model verbalizes the control - Qwen does (MSA conservation
  0.80), Llama does not (0.56), so weak verbalizers need a different yardstick.

## What the 3-way bridge experiment found (v1)

We then ran the forward experiment. On one shared frozen molecular-FM embedding (ChemBERTa over 7 ADMET
endpoints) and one held-out test set, we put all three placements side by side - a learned LLM x SFM
BRIDGE (the embedding enters the frozen LLM as a soft-prompt and it verbalizes the answer), EXTERNAL
ORCHESTRATION (a trained read-out head, LLM untouched), and IN-WEIGHT LoRA - with refutation paths
pre-committed to overturn "route, don't train" ([docs/BRIDGE_3WAY_PREREG.md](BRIDGE_3WAY_PREREG.md)). v1
(Qwen3-8B, hERG):

- **Within-property: orchestrate (head) 0.89 > bridge 0.85 > in-weight LoRA 0.73.** The decisive control:
  the bridge never beats its own LLM-bypass (the same projection feeding a bare head, no transformer;
  0.85 vs 0.87). Routing the embedding through the frozen LLM in-language adds NOTHING over a head on the
  same projection - the frozen LLM is dead weight in the read. In-weight LoRA lifts the model's output a
  lot (0.48 -> 0.73) but lands well below the head.
- **Held-out-property transfer: no placement transfers.** A read trained on five ADMET properties and
  applied to held-out hERG sits at or below chance for all three (0.44-0.49), at or under the
  cross-property floor. Reading the SFM is a property-specific skill, not a general one - for every
  placement.

So the fair test we built to overturn "route, don't train" came down on confirm: the closed-weight-
friendly thin head on the open SFM is the best placement, the in-language bridge and in-weight
fine-tuning do not earn their extra machinery, and no placement generalizes the read across properties.
(v1 caveats: paired-difference CIs and a pretraining-naive embedding control are the v2 pass; one model,
one endpoint, one fold.)

## What the generative RL experiment found (experiment 3)

The bridge was the discriminative read-out lever; experiment 3 asks the same route-vs-train question in
the GENERATIVE / RL regime - the post-training RL environment the literature leaves as an explicitly
open head-to-head ([docs/RL_ENV_PREREG.md](RL_ENV_PREREG.md), grounded in
[docs/RL_ENV_DEEPRESEARCH.md](RL_ENV_DEEPRESEARCH.md)). We built the whole environment: a FROZEN in-repo
SMILES generator (a 4.18M char-RNN self-trained on disclosed in-corpus molecules with the held-out
scaffolds excluded), our hERG ADMET read-out head as the REWARD, and a scaffold-disjoint RF-on-Morgan
ORACLE as the independent judge (held-out AUROC 0.882). On generated molecules the reward enriches
oracle-actives 14.8x in its top 5% (reward->oracle-pass AUROC 0.953), so guidance and RL both have real
signal to exploit; the contest is genuine, not a foregone negative.

Three arms, judged only on the held-out oracle, at MATCHED reward-query budget (Q=5000), each
delivering 500 designs:

- **Internalized RL (arm A) ties external guidance (arm B): 21 vs 20 oracle-passes; (A-B) = +0.002,
  scaffold-clustered two-sample 95% CI [-0.047, +0.046].** The CI includes 0 and the point sits far
  inside the 0.03 tie band: tuning the generator's weights toward the reward buys nothing over selecting
  top-reward samples from the frozen model at the same budget.
- **The reward drives the gain (drift guard):** the same RL on SHUFFLED rewards collapses to base
  (0/500), so the 21 is reward-driven, not optimization noise. The arm is genuinely reward-internalizing
  (REINVENT augmented likelihood; a first PPO-clipped attempt diverged on the char-RNN and was replaced),
  stable (training reward 0.06 -> 0.62, KL-to-base bounded), and the headline is generative-design
  oracle-success, not a discriminator AUROC.

So the post-training RL environment is buildable and the reward produces real oracle-confirmed design
gains - but internalizing the reward into the weights ties externally selecting it from the frozen
model. "Route, don't train" EXTENDS from the discriminative read-out to the generative/RL lever.

The tie is robust (v2): it holds across three RL seeds (arm A 0.024-0.038, pooled (A-B) = -0.007, CI
[-0.054, +0.031]), in a degraded low-data-reward cell (arm A 0.048 vs guidance 0.028, (A-B) = +0.020,
CI [-0.020, +0.064]), and on a genuinely WEAK endpoint (clearance, run on SDSC Expanse: arm A 0.114 vs
guidance 0.088, (A-B) = +0.026, CI [-0.015, +0.067]). Across all three the point estimate tips toward
RL MORE as the reward weakens (A-B: -0.007 strong -> +0.020 low-data -> +0.026 weak-endpoint) - the
literature's predicted low-data crossover, directionally consistent - but it never reaches significance;
every CI includes 0. And on the weak endpoint the reward barely enriches oracle-actives at all (top-5%
enrichment 14.8x for hERG vs 1.1x for clearance), so the binding constraint there is reward reliability,
not the lever. No cell separates train from route.

**But the budget sweep flags a real high-budget deviation.** Varying the reward-query budget (M=1000,
hERG), the (A-B) gap grows monotonically and at Q=10000 internalized RL beats guidance - pooled over 3
seeds, arm A 0.093 vs guidance 0.029, 95% CI [+0.026, +0.104] (excludes 0) - by shifting the distribution
beyond guidance's frozen-model ceiling, with drug-like, legit-hERG-blocker-chemotype designs (not mode-
collapse or obvious gaming: arm A's output stays as drug-like as real hERG molecules, and its passers
match guidance's passers' profile). But the effect is seed-VARIABLE (0.047-0.170; seed 0's 0.170 was a
high outlier) and its CI lower bound (0.026) just misses the pre-registered 0.03 overturn margin - so it
is INDETERMINATE by the prereg rule, at high drift (KL ~8). So route-don't-train holds at moderate budget
(Q <= 5000, confirmed 6 ways), while the HIGH-budget regime shows a real but modest, sub-threshold RL edge
- the one place in the program where training the weights may pull ahead - with the docking co-primary the
remaining arbiter ([../results/benchmark/rl_env/budget_sweep.md](../results/benchmark/rl_env/budget_sweep.md)).

## What we'd do next

Both experiments are run. The open threads are v2 rigor on the bridge result (paired-difference CIs, a
pretraining-naive embedding control to rule out the molecular FM having already seen the property, and
the cross-architecture / cross-endpoint matrix) and, if the bridge's dead-weight finding holds, leaning
the whole program toward the measurement and routing layer it implies: a calibrated router over a frozen
specialist, since training the read into the model (bridge or LoRA) buys nothing here.

## Reading list

*Elicitation / encode≫express:* 2305.14775 · 2303.08112 · 2312.01037 · 2510.01070 · 2505.14352 ·
2601.18468 · 2102.12452. *Generative alignment & orchestration:* 2505.15093 · biorxiv 2025.05.02.651993 ·
2510.19474 · 2511.21476 · 2507.00445 · Proto biorxiv 2026.06.22.733870. *LLM↔SFM bridges:* 2510.01428 ·
2506.03800 · 2509.24840 · 2410.22949 · 2408.11363 · biorxiv 2024.05.14.594226 · 2410.03553 · 2412.19191.

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

## What we'd do next

The layer-localization above is done; the forward step is the **calibrated LLM x SFM bridge with a
held-out-property transfer eval**, compared head-to-head against (i) external orchestration of the frozen
SFM and (ii) in-weight LoRA. The attach layer is now known to be model-specific (locate it per model, not
at an assumed mid-band), and the read-out's routing edge is the calibration signal the router would use.
Lit-grounded hypothesis: external guidance wins in low-data, the bridge wins with enough data, in-weight
fine-tuning is a strong second - and the new result is the calibration: a frontier router that knows when
its read of the specialist is trustworthy.

## Reading list

*Elicitation / encode≫express:* 2305.14775 · 2303.08112 · 2312.01037 · 2510.01070 · 2505.14352 ·
2601.18468 · 2102.12452. *Generative alignment & orchestration:* 2505.15093 · biorxiv 2025.05.02.651993 ·
2510.19474 · 2511.21476 · 2507.00445 · Proto biorxiv 2026.06.22.733870. *LLM↔SFM bridges:* 2510.01428 ·
2506.03800 · 2509.24840 · 2410.22949 · 2408.11363 · biorxiv 2024.05.14.594226 · 2410.03553 · 2412.19191.

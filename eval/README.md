# WS1 - Grounding and Comprehension Instrument

*Design spec. 2026-06-08. Capability-first. No em dashes. Reference: `../docs/FAILURE_MODES.md`.*

## What this measures

Per representation, does the model ground by **content** or by **name**, and does the understanding a probe can read out of the representation actually surface in the model's output. Organized as a capability ladder:

| Tier | Question | How |
|---|---|---|
| T0 Recognize | does it resolve the identity of a representation | accession/SMILES -> name/family (reuses FRT A-axis) |
| **T1 Comprehend** | does it read a verifiable property out of the content | **axis B head-to-head + content-sensitivity (below); negative-evidence coverage is adjacent, not this instrument** |
| T2 Apply | does it use that understanding downstream | solve / propose / evaluate; **SOLVE + PROPOSE scored** (`../results/t2_apply.md`, `../results/t2_propose.md`), evaluate blocked on the D human-rater pass |

## Axis B: the grounding measurement (and what is out of scope)

This spec is the **axis-B** instrument (`../docs/FAILURE_MODES.md`): does the model read a property that depends *causally on the representation content* (a sequence motif, a chemical substructure)? Measured by the probe-vs-LLM head-to-head plus content-sensitivity. This is the thesis-critical, currently-unmeasured gap. The probe runs on the SFM representation, not the LLM's, so the head-to-head shows *whether* the LLM produces the property, not *why*; separating encoding (never internalized) from expression (internal but unsurfaced) needs an LLM-activation probe. Do not call it a "verbalization gap" until then.

**Out of scope: negative-evidence coverage.** Telling a tested-and-failed relation from a never-tested one is a different, knowledge-coverage capability, already measured by NullAtlas (the WS2 substrate, rho = -0.70). It is not content grounding: scrambling the SMILES does not change whether a pair was tested (DTI L4 shows the SMILES yet LLM MCC = 0.0424, near chance). This spec cites NullAtlas, it does not re-measure it. Holding that line is what keeps Bio_Grounding_Eval's contribution (the grounding instrument) distinct from NullAtlas's.

## Core measurement (axis B): probe vs LLM head-to-head

Same inputs, three arms:

- **structure-probe arm (ceiling):** a probe on the representation's own features (a cheminformatics fingerprint like Morgan for chemistry; an SFM embedding like ESM-2 for protein) -> linear/RF probe -> property. How much of the signal is *in the structure representation*. Only meaningful where the supervised ceiling is genuinely high; a low ceiling means there is nothing to surface. (Note: for the hERG case it is Morgan fingerprint + logistic/RF, a cheminformatics descriptor, not a neural SFM.)
- **LLM-output arm (floor):** the same representation handed to the LLM, asked the same property, scored deterministically. How much it *surfaces in output*.
- **LLM-activation arm (the decider):** a simple linear probe on the LLM's *own hidden states* (every-5th-layer + final, AUROC, disagreement subsets), per the Masked-by-Consensus design (2604.12373). This is what separates **encoding** (signal absent in the LLM) from **expression** (present internally, not surfaced); the two-arm gap alone cannot. Use a linear probe, NOT a second-LLM verbalizer (2509.13316 shows verbalizers leak the verbalizer's own knowledge).
- **Reads:** SFM-probe minus LLM-output = the raw gap (B2, direction TBD). LLM-activation high but LLM-output low = expression gap. LLM-activation also low = encoding gap. This extends Inside-Out (2503.15299, general QA) to biological representations, and unlike Masked-by-Consensus (self-vs-peer LLM, no biology) it probes an SFM-grounded property inside the LLM.

Three confounds the design must control, or the gap is uninterpretable:
1. **Supervision asymmetry.** A trained probe vs a zero-shot LLM is unfair by construction. Give the LLM matched in-context examples (few-shot) and report a *content-sensitivity differential*, not just a raw probe-minus-LLM number.
2. **Input asymmetry.** The probe reads an SFM embedding (already processed by the SFM); the LLM reads raw text. A raw gap could be "SFM-encoded vs raw-text," not "internally-present vs not-surfaced." Also feed the LLM the SFM output (an orchestrate condition) to separate the two.
3. **Property leakage.** The property must not be recoverable from the name alone, or the LLM passes by recall. The content-sensitivity conditions are the guard.

## Content-sensitivity (separates content from name)

Cross four conditions on the LLM arm; the pattern is the content-sensitivity score:

| Condition | content (seq/SMILES) | label (name/accession) | a true content-grounder should |
|---|---|---|---|
| **matched** | real | real (consistent) | be correct |
| **mismatched** | real | wrong entity (= FRT Variant C) | follow content, not the name |
| **scrambled** | corrupted | real | degrade (content destroyed) |
| **content-only** | real | removed (= FRT Variant D) | still solve from content alone |
| **re-notation** | real, alternate VALID notation | real | give the same answer (notation-invariant) |

The **re-notation** condition is the within-entity test of representation invariance: the SAME molecule as a canonical vs a randomized-but-valid SMILES, the same variant as gene+HGVS vs rsID vs raw sequence, the same protein by accession vs sequence. A content-grounder gives the same answer; if the answer moves with the notation while the content is held fixed, the model is grounding the surface form, not the content. This is the general-capability form of the FRT recognition gap (axis A: name ~100% vs accession ~2-28%), and the web-exposure law (`../PROJECT_DESIGN.md` section 7) predicts it within a single entity (a web-frequent notation grounded, a web-rare notation of the same content not). The variant branch measured it (text 0.79 vs seq 0.58 on the same variants); the SMILES randomized condition marks the floor (canonical 0.573 vs randomized 0.553, no notation effect where the output carries no signal to bind), so notation-sensitivity is measurable only where grounding already surfaces.

The scramble must actually destroy the signal, not just permute it. A plain residue shuffle preserves amino-acid composition, so a model reading composition alone would not degrade and the control would falsely read as a name shortcut. Use at least two strengths: a composition-preserving shuffle and a motif-targeted disruption (mutate the catalytic/functional residues the property depends on). The property counts as content-grounded only if it tracks the stronger disruption. FRT already built the mismatched and content-only halves. (GenomeQA 2604.05774 does a dinucleotide-preserving shuffle on one DNA task; our delta is the cross-modality, same-entity real-vs-corrupted matched contrast tied to the activation arm, not a single-task shuffle.)

## Phase 1: NMSE method PoC (does the B-axis machinery work)

NMSE is the one asset with a confirmed high probe ceiling on content (AUROC 0.9807 on frozen ESM-2). It is small and its strong label is dual-use/benign, so Phase 1 is a **method check, not the capability headline**: confirm the head-to-head and the four content-sensitivity conditions actually separate "signal present" from "signal surfaced," before investing in new data.

- **Ceiling, reused:** the 0.9807 separability result stands as the "signal is in ESM-2's representation" anchor (a homolog sequence set; NMSE reports the AUROC on 60+60 arrays while its corrected panel is 71/62, an NMSE-internal discrepancy to reconcile before relying on it, and 60 is not large-N). It is a dual-use/benign label, so it validates the *method*, not a capability-neutral claim, and it is about ESM-2, not the LLM. Flag both.
- **LLM arm:** run the four content-sensitivity conditions on the panel sequences, score deterministically (LabCraft, `BioProtocolBench/labcraft`). Question: does the LLM surface what the probe reads, and does corrupting the sequence move its answer.
- **Capability-flavored mini-case:** the two mechanism-matched pairs (zinc metalloprotease BoNT-A / astacin; RIP ricin / saporin-6) test whether the model reads mechanistic class from content independently of the danger label. Qualitative, n is tiny.
- The earlier multiclass functional-class probe on the panel (16 annotated dual-use proteins + 3 benign controls) is dropped (too few members per class).
- **Output:** evidence that the B-axis method works (or not), and a first probe-minus-LLM gap on a known-high-ceiling task.

## Phase 2: a built content-property task (the real B-axis measurement)

The capability-neutral measurement runs on a **content-property task that has to be built**, not on any existing memorization benchmark. NegBioDB raw data is **one source** for it, not the only one, and the head-to-head gap must not depend on NegBioDB specifically; any established property-prediction dataset of the same shape (content -> deterministic property) is an equally valid source. This is WS2-style work (generating (representation, verifiable-property) pairs).

- **Candidate sources** (property depends causally on content): NegBioDB ADMET (SMILES -> hERG/AMES/CYP endpoint), EPI (promoter sequence -> TF binding), ClinGen/CLNV (sequence + variant -> pathogenicity); plus external content-property benchmarks of the same shape. These expose real content and are mechanistically content-dependent.
- **Widest-open variant:** feed an SFM embedding (ESM / scELMo / BioVERSE-style) as the LLM's input and run the same three arms. The 2026-06 scan flags this as the most open case (no behavioral baseline exists yet); consider prioritizing it once the protein/chem head-to-head works.
- **Gate first, then build.** None of these has a measured supervised ceiling yet. Step 1 is a content-feature probe per candidate (SMILES fingerprint/encoder; DNA/protein embedding) to confirm the ceiling is high (signal exists). Only domains that pass the ceiling gate become head-to-head tasks. This avoids the DTI trap (apparent ceiling from leakage, collapses on cold splits). Caveat: gating on a high probe ceiling conditions the headline gap on SFM-favorable domains, so the reported gap is an upper-bound estimate, not representative across biology; a domain that fails the gate means "no measurable signal under this encoder," not "no signal."
- **Then:** an LLM arm that shows the raw content (SMILES / DNA sequence / variant context), the four content-sensitivity conditions, deterministic scoring, stratified by the NegBioDB PBS difficulty gradient.
- **Modalities added:** chemical, DNA, variant (protein already covered by Phase 1). RNA / metabolite as the roadmap opens (WS4).
- **Output:** the capability-neutral probe-minus-LLM gap and content-sensitivity across modalities and difficulty.

## Why axis B needs a built task (not NegBioDB's results)

Axis B needs a content-grounded property task, and no existing asset provides one off the shelf (NMSE is small and dual-use-labeled; NegBioDB's measured tasks are memorization, not content grounding). So the real B-axis measurement needs a **task to be built** (WS2-style generation of (representation, verifiable-property) pairs), which partially inverts the original "WS1 first" sequence (WS4 doc pass). The dependency is on *building a task*, not on *importing NegBioDB's L4/PBS results*: NegBioDB is one raw-data source, not the measurement, and the instrument stands on axes A/B/E without it.

## Adjacent: negative-evidence coverage (NullAtlas, WS2)

Not part of this instrument. NullAtlas already measures whether a model tells tested-and-failed from never-tested (deterministic MCC, 30 domains, rho = -0.70 across haiku-4-5 / gemini-2.5-flash / gpt-4o-mini). It is NullAtlas's result and the WS2 substrate, cited here, not rebuilt or claimed as a grounding-instrument finding. Keep the L3 (content-reading) vs L4 (recall) boundary intact so the two projects' contributions stay distinct.

## Scoring

Deterministic only: closed-set exact match and rule-based parsing (LabCraft grader). No LLM judge anywhere in WS1.

## Bridge to T2 (designed after T1 returns a first gap)

T1 gives a per-item grounding-fidelity score. T2 puts the same items into a downstream task in three modes:
- **Solve:** predict the property (the T1 task used as a capability, scored vs ground truth).
- **Propose:** generate a candidate (a plausible binder/interactor), scored against ground-truth negatives so proposing a tested-and-failed candidate is penalized deterministically. NullAtlas negatives are one optional source for this check, not a requirement.
- **Evaluate:** judge a presented claim's reliability (reuses the axis-D over-trust setup once it is off exploratory).

The capability test the project rests on: **does higher T1 grounding predict higher T2 performance?** SOLVE is now scored (`../results/t2_apply.md`): not through the solo path (output near chance, flat regardless of internal encoding), but the solo-to-orchestrate headroom decomposes per task into a trainable-read-out (expression) part and an orchestrate-the-specialist (encoding) part, which is the T2 routing rule and the WS3 seed. Propose is scored (`../results/t2_propose.md`: generation competent, but grounded activity undetermined by the in-distribution probe). Evaluate remains blocked on the D human-rater pass (it reuses axis D).

## Guardrails carried in

- Capability-neutral property for the headline (Phase 2); Phase 1 uses the NMSE dual-use label only as a method check, flagged as such.
- FRT `disclosure/` material never enters; aggregate behavioral numbers only.
- No em dashes; verified facts; NMSE is an "Evaluation Report"; substrate is NegBioDB/NullAtlas.
- The mine-to-claim core is the axis-B grounding instrument (with A and E). NullAtlas/NegBioDB is cited and reused (WS2 substrate, one raw-data source), not absorbed as an instrument result.

## Compute
Heavy or GPU work (ESM-2 / Evo2 embeddings, LLM hidden-state extraction for the activation arm, any model fine-tune) runs on **Cayuga** (WCM HPC) or **Expanse** (SDSC, for GPU), not locally. Local is only for light CPU probes. `eval/ceiling_gate.py` (rdkit + sklearn) is the local reference; the probe-vs-LLM head-to-head and the LLM-activation arm move to Cayuga/Expanse.

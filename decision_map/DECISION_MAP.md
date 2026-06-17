# WS3 decision map: first point (hERG)

*2026-06-10, revised 2026-06-11 after an adversarial review added the mandatory no-LLM baseline. The WS3 deliverable (`../PROJECT_DESIGN.md` WS3): per capability, which placement gives the answer, weights vs retrieve vs orchestrate, and whether the LLM is even the right tool. PoC scripts: `../eval/ws3_retrieve.py` (retrieve arm + baseline), `../eval/activation_arm.py` (read-out/weights arm), `../eval/ws3_image.py` (image corner), `../signal/admet/herg` (specialist ceiling). All AUROC at n=120 to 200, so 95 percent CIs are roughly +/-0.08 to +/-0.10; treat sub-0.08 differences as ties. No em dashes.*

## The question

T2-solve showed the solo model verbalizes hERG near chance while a probe reads it off the structure (the expression gap). WS3 asks: given that gap, where do you place the capability, and crucially, is the LLM the right tool at all. The placements, each with the control that decides whether the LLM is contributing:

| placement | hERG AUROC | model | note |
|---|---|---|---|
| solo | 0.633 frontier / 0.453 8B | sonnet-4-6 / Qwen3-8B | the model answers from the SMILES, no help |
| retrieve (dense, LLM) | 0.843 | sonnet-4-6 | k=10 nearest labeled molecules in context |
| **retrieve (dense, NO-LLM neighbor-mean)** | **0.951** | none | the control: average the 10 neighbor labels, never read the query |
| weights (read-out) | 0.787 | Qwen3-8B | linear read-out on hidden states, scaffold GroupKFold |
| orchestrate (Morgan probe) | 0.895 | model-agnostic | the fingerprint specialist, cold scaffold split |

## The clean version: all four placements on ONE scaffold split

The table above mixes models and splits (caveated below). The rigorous single-split version computes every placement on the EXACT scaffold split the LoRA weights PoC used (train 1097 / test 528, `eval/ws3_decision_split.py`, `results/ws3_decision_split.json`):

| placement | hERG AUROC (same split) | uses the LLM? |
|---|---|---|
| retrieve (no-LLM neighbor-mean k-NN) | 0.901 | no |
| orchestrate (Morgan-fingerprint probe) | 0.899 | no |
| weights (LoRA-trained 8B output) | 0.856 | yes (trained) |
| solo (8B output, no training) | 0.575 | yes |

Apples-to-apples, the ranking is unambiguous: the two cheap fingerprint methods (k-NN 0.901, probe 0.899, nearly tied) lead, the trained LLM (0.856) is a strong second, and solo (0.575) is far below. So even AFTER training closes most of the expression gap (solo 0.575 to weights 0.856), the weights-LLM still does not beat a no-LLM fingerprint method on this fingerprint-local property. The decision-map verdict for hERG-SMILES is firm and now rigorously sourced: orchestrate a cheap specialist (or just k-NN); the LLM, even trained, is not the tool here.

## The headline, corrected: for hERG-SMILES the LLM is the wrong tool

The first draft of this doc called retrieve a "surprising win for the LLM." The mandatory control refutes that. The no-LLM neighbor-label-mean (take the 10 Morgan-nearest pool molecules, output the mean of their labels, never show the query to any model) scores **0.951 dense and 0.913 with a scaffold holdout**, while the LLM-in-the-loop retrieve scores **0.843 and 0.831**. The LLM is **0.08 to 0.11 BELOW a trivial arithmetic baseline** (mean neighbor purity 0.87 carries the AUROC). So the retrieve AUROC measures neighbor-label purity, not any model capability, and the LLM, handed the same neighbors, actually degrades the answer.

That reframes the whole hERG-SMILES corner. Every LLM placement is dominated by a cheap fingerprint method:

| how you solve hERG-SMILES | AUROC | uses the LLM? |
|---|---|---|
| no-LLM neighbor-mean k-NN | 0.951 | no |
| orchestrate (Morgan probe) | 0.895 | no |
| LLM + retrieval | 0.843 | yes, and it hurts vs the baseline |
| weights (LLM read-out) | 0.787 | yes (trained probe on hidden states) |
| solo LLM | 0.45 to 0.63 | yes, fails |

So the decision-map verdict for hERG-SMILES is not "retrieve is a cheap LLM placement." It is: **a fingerprint specialist (a trivial k-NN or the Morgan probe) dominates every LLM placement, so this capability should be ORCHESTRATED to a non-LLM specialist, not solved by the model at all.** The instrument's real value here is telling you where the LLM is the wrong tool. The expression gap is real (the 8B read-out 0.787 recovers signal the 8B output 0.453 cannot surface), so training the read-out is a legitimate LLM placement, but it is still beaten by the cheap specialist, so on cost and accuracy you would not.

## The "boundary" was the baseline, not the model

The earlier draft read a sparse-pool result as "retrieve collapses toward solo, routing to weights." The control kills that story too. Capping the pool to 40 with the scaffold holdout drops the LLM to 0.586, but the no-LLM baseline drops to **0.554** at the same time, and solo frontier is 0.633: at n=150 (CI +/-0.08) these three are mutually indistinguishable. There is no separate "retrieve falls back to solo" phenomenon; the neighbor-mean baseline simply degrades as the nearest-neighbor similarity falls (top-10 Tanimoto drops to 0.16 to 0.22, purity 0.72). What the pool-size sweep actually shows is the obvious thing: the fingerprint specialist works when the labeled pool is dense and degrades when it is sparse, independent of any LLM.

| regime | no-LLM k-NN | LLM-retrieve | neighbor purity |
|---|---|---|---|
| dense pool (3813) | 0.951 | 0.843 | 0.87 |
| scaffold holdout | 0.913 | 0.831 | 0.87 |
| sparse pool (40) + holdout | 0.554 | 0.586 | 0.72 |

(`../results/ws3_retrieve_*.json`. The LLM column is always at or below the no-LLM column.)

The adversarial-neighbor arm settles what the LLM is actually doing. Show it the same k neighbors but with their labels FLIPPED (the nearest, most-similar molecules now carry the wrong answer), and the LLM AUROC drops to **0.105** (`../results/ws3_retrieve_random_split_FLIPPED.json`). It follows the wrong labels almost perfectly (0.105 is near 1 minus its 0.843 correct-label score). A model that read the query structure and applied even weak SAR would have held near its solo level (0.5 to 0.63); instead it inverts. So in the retrieve setting the LLM is a near-pure label parrot: it copies the retrieved labels and does not read the query molecule at all. That is the decisive evidence that the retrieve AUROC is the retrieval set's property, not a model capability.

## What this refines (the T2 routing rule), corrected

`../results/t2_apply.md` R2 routed expression-limited capabilities to WEIGHTS (train the read-out). The decision map refines that, but not as "retrieve is the cheap LLM option." The real second axis is **whether a cheap non-LLM specialist plus a labeled pool already solves the property**:

- **a cheap fingerprint specialist solves it** (hERG: Morgan probe 0.895, trivial k-NN 0.951): orchestrate the specialist, do not use the LLM. The encoding/expression state of the LLM is then moot for deployment, it is simply not the tool.
- **no cheap specialist, but the LLM encodes it** (the expression-limited-and-not-fingerprint-trivial case, not instantiated by hERG): train the read-out (weights) is the LLM placement that earns its keep.
- **the LLM cannot encode it AND no cheap specialist reads the input** (molecular image, below): only a dedicated perception specialist works.

Caveat on circularity: orchestrate (Morgan-fingerprint CV, `../signal/admet/herg/verifiability.json` cold 0.895) and the no-LLM neighbor-mean (Morgan-Tanimoto k-NN) are both fingerprint methods on the same `pairs.jsonl`, so they are near-tautologically strong here, and "the specialist dominates" partly reflects that hERG is a fingerprint-friendly property. That is the honest scope: this point shows the LLM is dominated for a fingerprint-local property, not for all properties.

## The second corner: molecular image

hERG-from-SMILES is fingerprint-solvable. The molecular IMAGE of the same molecules is the opposite corner: no cheap specialist reads it, because the perception step itself is hard. Claude is a VLM, so the image arms run directly (`../eval/ws3_image.py`, n=120 rendered hERG molecules, `../results/ws3_image.json`):

| image arm | result | reading |
|---|---|---|
| solo-image (read the property) | AUROC 0.539 | at chance (95 percent CI ~0.44 to 0.64, includes 0.5) |
| OCSR mean Tanimoto to truth | 0.544 | the transcribed structure is about half-right (the defensible metric) |
| OCSR valid-SMILES rate | 0.725 | it emits a valid SMILES most of the time |
| OCSR exact-match rate | 0.158 | exact only 16 percent, but this keeps stereochemistry so it is a lower bound, see caveat |

Solo-image is at chance: it cannot read the property from pixels. The mechanism is the OCSR (optical chemical structure recognition) arm: Claude transcribes only a half-right skeleton (mean Tanimoto 0.54, exact 0.16). Is that a perception limit, or are 320px RDKit renders just hard? A DECIMER baseline (a dedicated OCSR tool) on the SAME 120 images settles it: DECIMER reads them at **mean Tanimoto 0.97, exact 0.85** (`../results/decimer_ocsr.json`, GPU run). So the renders ARE legible, and Claude's 0.54 is a genuine VLM perception limit, not a rendering artifact. The perception-floor claim is therefore confirmed by a calibrated reference (Claude 0.54 vs the legible-render ceiling 0.97), not just asserted.

Placement consequence, CORRECTED 2026-06-11 (`../results/image_rung.md`): the original claim that "the LLM's own OCSR is too noisy to feed a structure probe" is WRONG for hERG. A Morgan probe on Claude's half-right transcriptions (the ocsr_cand SMILES) still scores 0.759, only 0.025 below the true-structure ceiling 0.784, because hERG is a COARSE property (lipophilicity, aromatic rings) whose signal survives a 0.55-Tanimoto transcription. So orchestrating the VLM's OWN OCSR works for hERG (you do not even need DECIMER), and image-hERG is EXPRESSION / orchestration-limited, not encoding-limited: the VLM perceives the hERG-relevant structure (proxy 0.759) but cannot verbalize the property directly (solo-image 0.566). The "only a dedicated perception specialist works" reading holds only for FINE-structure-dependent properties (exact-match, stereochemistry), where the 0.55-Tanimoto floor is fatal; for coarse physicochemistry the cheap LLM-OCSR shortcut is enough. A direct open-VLM hidden-state probe (Qwen2.5-VL-7B, `../eval/activation_arm_image.py`) CONFIRMED it: activation 0.758 (nearly identical to the OCSR proxy 0.759) vs output 0.460 = expression-limited, not encoding-limited.

## The third corner: a non-fingerprint-local endpoint where the LLM WINS

The first two corners both route AWAY from the LLM, which the deep review flagged as foreordained: WS2's gate admits only fingerprint-friendly endpoints, so a cheap structure specialist always exists. This corner is the missing cell. Drug market-WITHDRAWAL (why a drug was pulled: cardiotox, hepatotox, abuse, idiosyncratic ADR) is a clinical/historical fact with little local-substructure basis, so no cheap specialist exists, and the answer is documented for named drugs (`../eval/withdrawn_endpoint.py`, `../results/withdrawn_endpoint.md`, sonnet-4-6, n=500 balanced, WITHDRAWN/DrugBank/ChEMBL/NCATS set):

| placement, withdrawal endpoint | AUROC | 95% CI | uses the LLM? |
|---|---|---|---|
| orchestrate (Morgan probe, scaffold CV) | 0.662 | 0.615, 0.708 | no |
| no-LLM k-NN (Tanimoto) | 0.544 | 0.496, 0.591 | no |
| solo LLM, from SMILES | 0.535 | 0.488, 0.582 | yes |
| **solo LLM, from the drug NAME** | **0.758** | **0.715, 0.795** | **yes (knowledge route)** |
| control: LLM from a shuffled fake name | 0.503 | 0.458, 0.553 | yes |

Here the order INVERTS versus hERG: the structure tools (Morgan 0.662, k-NN 0.544) and the LLM-from-SMILES (0.535) are all weak, and the LLM-from-NAME (0.758) is the best tool, beating Morgan with non-overlapping CIs (0.715 > 0.708). The encoding side is just as striking: an 8B activation probe on the SMILES reads withdrawal at 0.762 (held-out 0.740), far above its own output (0.469) and above Morgan (0.643), and a randomized-SMILES control drops it to the Morgan level (0.662), so the above-structure signal is canonical-string-keyed drug RECOGNITION (the model identifies the known drug from its SMILES and reads a recalled fact but cannot say it), reaching the same level as the frontier name route. This is axis-A entity-recognition surfaced by the probe, distinct from hERG's axis-B content-grounding which the same randomized control leaves intact; the re-notation test is what tells them apart (`../results/withdrawn_endpoint.md`). Three controls make it honest: the win is KNOWLEDGE not structure (name 0.758 vs SMILES 0.535, the LLM cannot read withdrawal from structure either), and it is DRUG-SPECIFIC knowledge not a generic prior (fake-name 0.503 is exactly chance). So this is the first cell that routes TO the LLM, and it routes to the NAME route, not a structure route. Cell: "recognized drug + knowledge-documented, fingerprint-weak endpoint -> LLM (name)." For a novel/unnamed molecule there is no name to recall and the cell collapses back to the weak-structure corner (route to: generate assay data). Caveats in `../results/withdrawn_endpoint.md` (recognition-bounded; possible DrugBank/WITHDRAWN exposure, though the fake-name control localizes the signal to the specific name; single model/endpoint).

## Honest scoping

Three endpoints now (hERG fingerprint-local, image perception-limited, withdrawal knowledge-documented), so the two-axis claim has both a "do not use the LLM" cell AND a "use the LLM" cell and is no longer foreordained by the gate; it is illustrated across three contrasting corners, not established as a law. All AUROCs are n=120 to 200, CI ~+/-0.08 to +/-0.10, so: the within-band ordering of the LLM placements is not resolved; solo-image 0.539 is not distinguishable from solo-text 0.633 (overlapping CIs, do not read "image below text" as real); the pool40 LLM/baseline/solo trio is a three-way tie. The model column is mixed (solo/retrieve sonnet-4-6, weights 8B, orchestrate and baseline model-agnostic), so the clean comparisons are within-model: on sonnet-4-6 the LLM-retrieve (0.83) is below the no-LLM baseline (0.95); on 8B the read-out (0.787) is far above solo (0.453). The cross-model absolute levels are not comparable. Orchestrate/baseline share the fingerprint featurizer with the WS2 gate (circularity above).

The adversarial-neighbor arm is run (flipped labels drop the LLM to 0.105, a near-pure label parrot), the DECIMER OCSR baseline is run (DECIMER 0.97 vs Claude 0.54, perception floor real), and the LoRA weights PoC is run (`../results/ws3_lora.md`): finetuning the 8B read-out lifts its OUTPUT from 0.575 to 0.856 (+0.28), recovering MOST of the same-split structural ceiling (Morgan probe 0.899, k-NN 0.901) but staying ~0.04 below it. So "train into weights" surfaces most of the encoded signal at the output level (P2 at the output level), but on this fingerprint-local property it is a strong SECOND to the cheap specialist, not a winner. A label-shuffle control stays at chance (0.484, rules out label memorization) but was run at the weaker config and the split is near-domain (median test-train Tanimoto ~0.66), so the lift includes local-SAR generalization, not a cold-chemistry result. The endpoint-where-the-k-NN-fails was the explicit gap; the withdrawal corner above now supplies it (Morgan 0.66, k-NN 0.54), and shows that there the winning placement is not weights but the LLM NAME route (0.758), because the signal is documented-knowledge, not encoded-structure. Remaining: more such endpoints, and a fingerprint-weak property where the LLM could win from STRUCTURE rather than name (none found yet, consistent with the surface-decodability result that the LLM has no structural edge).

## Reproduce

Retrieve (LLM arm + the mandatory no-LLM baseline in one run): `python eval/ws3_retrieve.py` (add `WS3_SCAFFOLD_HOLDOUT=1`, `WS3_POOL_CAP=40`); results in `../results/ws3_retrieve_*.json` (`retrieve_auroc` vs `neighbor_mean_baseline_auroc`). Image: `eval/ws3_image.py`. Weights: `sbatch run_activation_cayuga.sh` on a GPU cluster. Orchestrate: `../signal/admet/herg/verifiability.json` cold_auroc.

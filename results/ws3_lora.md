# WS3 weights PoC: does LoRA finetuning close the hERG expression gap in OUTPUT?

*Results section. 2026-06-11. Instrument: `eval/ws3_lora.py`, run on a Cayuga a40. The "train the read-out into weights" placement, tested at the OUTPUT level (not just a frozen probe). No em dashes.*

## The test

Qwen3-8B encodes hERG to AUROC 0.787 (a linear probe on its hidden states) but verbalizes it at chance (output arm 0.453): the expression gap. The activation probe shows the signal is THERE; it does not show the model can be made to SURFACE it. P2 predicts the gap is closable by training the read-out. This LoRA-finetunes the model on the hERG yes/no task and measures the model's own VERBALIZED output AUROC before and after, on a held-out Murcko-scaffold split (same leakage control as the probe). Verbalized score = logP(" yes") vs logP(" no") continuation, identical for base and finetuned.

## Result: scaling the training recovers MOST of the gap

Three runs on the scaffold split (cap n=1625, only 625 hERG positives exist so it is NOT balanced, the strong-run test is 37 percent positive; output AUROC = verbalized logP(yes) vs logP(no), same eval for base and finetuned). The right comparator is the SAME-split structural ceiling (Morgan probe 0.899, k-NN 0.901, `ws3_decision_split.json`), not the cross-split activation probe 0.787:

| run | base output | finetuned output | lift | reading |
|---|---|---|---|---|
| n=1000, r=16, 3 epochs | 0.526 | 0.608 | +0.082 | under-trained |
| **cap-1625, r=32, 5 epochs** | 0.575 | **0.856** | **+0.281** | **recovers most of the gap (ceiling 0.899)** |
| shuffled-label control (n=1000, r=16) | 0.526 | 0.484 | -0.042 | stays at chance |

(Train loss in the strong run fell 0.72 to 0.16 over 5 epochs. The two bases differ because the n=1000 and cap-1625 loads produce different test sets.)

## Reading: P2 confirmed at the output level (most of the gap), with caveats

- **Most of the gap closes at the output level.** With more data and capacity (the cap-1625 set, 37 percent positive, r=32, 5 epochs), LoRA-finetuning lifts the 8B model's own VERBALIZED hERG output from 0.575 to 0.856 (+0.28, 95 percent CI on 0.856 ~ [0.82, 0.89] at n_test=528). On THIS scaffold split the structural ceiling is the same-split Morgan probe 0.899 and the same-split k-NN 0.901, so 0.856 recovers most of the gap but sits ~0.04 BELOW the structural ceiling, it does not reach or exceed it. (An adversarial review corrected an earlier overstatement here: the 0.787 activation probe is a different metric on a different, balanced, GroupKFold set, NOT the comparator on this split, so "exceeds the probe" was a cross-split confound.) The honest claim: the model does not merely encode hERG (the probe showed that), it can be TRAINED to surface MOST of the structural signal in its output. This is P2 at the output level.
- **The earlier +0.08 was under-training.** The first run (n=1000, r=16, 3 epochs) recovered only part; scaling data, rank, and epochs closes most of the rest. So "train the read-out into weights" is a real lever, not a weak one.
- **A negative control (label-shuffle) rules out label memorization, but the split is near-domain.** Finetuning on SHUFFLED labels leaves the output at chance (0.484), so the lift is not label/identity memorization. CAVEAT: that shuffle control was run at the WEAKER config (n=1000, r=16) on a partly-different split, so it validates the +0.08 run, not the headline +0.28 directly (a headline-config shuffle is the pending check). Also, the scaffold split is NEAR-DOMAIN: median test-to-train nearest-neighbor Tanimoto is ~0.66 and 82 percent of test molecules have a train neighbor above 0.4, so disjoint Murcko scaffolds does NOT mean novel chemistry. The lift therefore includes legitimate local-SAR generalization; it is not a cold/out-of-distribution result, and the shuffle control does not isolate "surfacing encoded knowledge" from "learning fingerprint neighborhoods."

## Decision-map implication

On the same scaffold split, the trained weights output (0.856) is a strong SECOND to the two no-LLM fingerprint methods (Morgan probe 0.899, k-NN 0.901) and far above solo (0.575); it does NOT beat the cheap specialist. So training internalizes most of the capability into the model (no specialist call at inference), but for this fingerprint-local property the cheap specialist still wins on accuracy and cost. The weights lever is real and recovers most of the gap, but it is not the best placement here. Where a property is NOT fingerprint-trivial (no cheap specialist), this trained-weights lever is the placement that would deliver it, and demonstrating that needs such an endpoint.

## Reproduce

On a Cayuga GPU (bge venv with torch/transformers/peft): `LORA_N=2000 LORA_R=32 LORA_EPOCHS=5 python eval/ws3_lora.py` (the SLURM wrapper `lora_job.sh` is generated Cayuga-side, not in this repo). Add `LORA_SHUFFLE=1` for the negative control. The same-split fingerprint baselines: `python eval/ws3_decision_split.py` (local, rdkit + sklearn). Results merge by tag into `results/ws3_lora.json`.

# Layer-resolved contrast of the two expression gaps

*Results. 2026-06-11. `eval/layer_profiles.py` (parses the activation logs, no new compute), figure `results/layer_profiles.png`, raw `results/layer_profiles.json`. Supports the content-grounding-vs-recognition separation from `withdrawn_endpoint.md` (axis B vs axis A). No em dashes.*

## The question

hERG and drug-withdrawal both show a large activation-vs-output gap (read-out >> verbalization), but `withdrawn_endpoint.md` argued they are DIFFERENT kinds: hERG encodes surface-decodable STRUCTURE, withdrawal encodes drug RECOGNITION / knowledge. The per-layer activation profiles (canonical vs randomized SMILES, same runs) test that mechanistically.

## Two discriminating facts (strong)

1. **Activation vs the fingerprint.** For hERG the activation probe (0.787) is BELOW the Morgan fingerprint (0.825): the hidden states are a slightly weaker STRUCTURAL reader than ECFP, nothing more. For withdrawal the activation probe (0.762) is ABOVE the Morgan fingerprint (0.643): the hidden states carry something the fingerprint does not.
2. **What re-notation removes.** Averaged across all 37 layers, randomizing the SMILES costs hERG only ~0.05 AUROC but costs withdrawal ~0.11 (about 2x), and for withdrawal it drops the activation all the way back to the Morgan level (0.762 -> 0.662 ~ Morgan 0.643). So withdrawal's above-fingerprint signal is entirely canonical-string-keyed: it is the model recognizing the specific drug from its canonical SMILES, which is exactly the signature of recognition/knowledge, not of a richer structural representation (which would survive re-notation, as hERG's does). The deficit is roughly UNIFORM across depth (early 0.12, late 0.11), so recognition is not a single deep layer, it pervades the canonical-string representation.

## Peak location (suggestive, not strong)

The canonical activation peaks at layer 2 for hERG (very early, surface) and layer 27 for withdrawal (deep, semantic), consistent with structure being read superficially and knowledge being assembled deep. But both profiles are nearly flat (hERG early-band 1to5 = 0.77 vs late-band 25to36 = 0.757; withdrawal 0.741 vs 0.751), so the peak shift is a weak corroboration of the surface-vs-semantic picture, not independent proof. The randomization-deficit contrast above is the load-bearing evidence.

## The picture

| | hERG (axis-B content-grounding) | withdrawal (axis-A recognition) |
|---|---|---|
| activation probe | 0.787 | 0.762 |
| vs Morgan fingerprint | BELOW (0.825) | ABOVE (0.643) |
| output (verbalized) | 0.453 | 0.469 |
| canonical peak layer | 2 (surface) | 27 (deep) |
| AUROC lost to re-notation | ~0.05 (robust = content) | ~0.11, drops to Morgan (canonical-keyed = recognition) |
| reading | property read FROM structure, unspoken | entity recognized, fact recalled, unspoken |

Both show probe >> output, but they are NOT two kinds of the same gap: hERG is axis-B CONTENT-GROUNDING (a notation-invariant structural signal that survives randomization) and withdrawal is axis-A ENTITY-RECOGNITION plus recall (a canonical-string drug-identity signal that COLLAPSES under randomization to the structure level). The re-notation control is precisely what separates them, so only the survivor (hERG) is content-grounding; the withdrawal arithmetic comes from the model recognizing the drug and reading a recalled fact, which the control unmasks. This sharpens the axis-B measurement (a probe-vs-output gap is content-grounding only if it survives re-notation) rather than adding a second gap.

## Honest limits

- The randomized control separates recognition from richer-structure; it does NOT separate "recalls a memorized fact about the recognized drug" from "reasons over recognized pharmacology." Both are knowledge.
- Single open-weight model (Qwen3-8B); the frontier encoding side is unmeasurable via API. The frontier NAME route (0.758) matching the 8B activation (0.762) is suggestive that they tap the same knowledge but is a cross-model, cross-access-path comparison.
- Layer AUROCs are max-over-layer selection-biased; the held-out-layer numbers (hERG 0.760, withdrawal 0.740) and the randomized contrast are the unbiased comparisons.

## Reproduce

`python eval/layer_profiles.py` (reads `/tmp/layer_profiles.txt`, extracted from `act_rand_3038486.log`, `act_wd_3038493.log`, `act_wd_3038495.log` on Cayuga).

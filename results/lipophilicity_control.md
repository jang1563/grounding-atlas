# Two local controls on the "encodes chemistry" soft bedrock

*Results section. 2026-06-11. Instrument: `eval/lipophilicity_control.py` (rdkit + sklearn, CPU, no GPU/API). Answers two of the deep review's foundational objections that do NOT need the LLM hidden states. The third (the LLM ACTIVATION probe on randomized SMILES) needs GPU and is queued. No em dashes.*

## What the deep review worried about

The hERG structure signal (Morgan probe ~0.82, LLM activation probe ~0.79, char-n-gram ~0.80) might be (A) a coarse LIPOPHILICITY axis rather than hERG-specific chemistry, and (B) SURFACE-STRING orthography rather than structure. Both are testable on the structure/orthography probes locally, under the same Murcko-scaffold GroupKFold as the activation arm.

## A. Is the structure signal just lipophilicity? NO (refuted)

| feature set | hERG AUROC (scaffold GroupKFold) |
|---|---|
| descriptors only (logP, MW, TPSA) | 0.675 |
| Morgan fingerprint | 0.825 |
| Morgan + descriptors | 0.829 |
| **Morgan residualized on the 3 descriptors** | **0.768** |

Lipophilicity (logP/MW/TPSA) alone gives 0.675, a real but partial signal. The decisive number is the last row: after linearly regressing every Morgan feature on logP/MW/TPSA and probing the RESIDUALS, hERG AUROC is still 0.768. So most of the structure signal SURVIVES lipophilicity residualization, there is substantial hERG-specific structural signal beyond the lipophilicity axis, and the "0.82 ceiling is just lipophilicity" worry is refuted. (The LLM activation probe at 0.787 sits right at this residual-structure level, consistent with reading real structure, not just lipophilicity.)

## B. Is the surface-string signal a canonical-orthography artifact? NO (the orthography threat is ROBUST)

| char-n-gram probe (analyzer=char, 2 to 5-gram, NO chemistry) | hERG AUROC |
|---|---|
| on canonical SMILES | 0.845 |
| on RANDOMIZED SMILES (same molecules) | 0.812 |

A probe that knows nothing about chemistry, only character substrings, reads hERG at 0.845 from the canonical string and STILL 0.812 from a randomized re-notation (drop only 0.033). So the surface-substring signal is not a canonical-string artifact, it is robust across notations: hERG-relevant substructures show up as character substrings regardless of how the SMILES is written. This CONFIRMS the deep review's orthography point: the property is robustly decodable from surface substrings, so an LLM activation probe reading it at the same ~0.79 level is matched by a no-chemistry substring probe and does NOT, by itself, establish that the model holds a deep CHEMICAL representation.

## Net for the bedrock

These two controls SPLIT the soft-bedrock worry:
- The structure signal is genuine structure, not a coarse lipophilicity proxy (residualized 0.768). So "the probe just reads lipophilicity" is wrong.
- But that structural signal is robustly SURFACE-DECODABLE (a char-n-gram gets 0.81 on randomized SMILES). So "the LLM ENCODES the chemistry" is not supported over "the property is linearly decodable from the (substring statistics of the) string, as it is from the LLM's hidden states." The activation probe does not beat a substring probe, so it is not evidence of representation beyond surface decodability.

The honest framing therefore stands as corrected in `head_to_head.md`: report a readout-vs-verbalization expression GAP (robust), and describe the encoded signal as "linearly decodable from hidden states, as from the raw string," NOT as "the model represents the chemistry." The one test that could still separate the LLM from the char-n-gram is the LLM ACTIVATION probe on randomized SMILES (does the LLM's hidden-state signal survive re-notation like the char-n-gram does, or differ?), which needs GPU and is queued; but given the char-n-gram already survives randomization at 0.81, the most likely outcome is that the LLM signal also survives and remains surface-decodable.

## GPU confirmation: the LLM activation probe on randomized SMILES (added 2026-06-11)

The local controls above are on the structure/orthography probes. The decisive LLM-side test (does the hidden-state signal survive re-notation like the char-n-gram does, or is it canonical-string-specific) was then run on Cayuga (`activation_arm.py ACT_RANDOMIZE`, log `act_rand_3038486.log`, Qwen3-8B, same 1250 hERG molecules, same scaffold split):

| arm | best-layer AUROC | best layer | held-out-layer AUROC |
|---|---|---|---|
| canonical SMILES | 0.787 | 2 | 0.760 |
| randomized SMILES | 0.739 | 20 | 0.732 |

Three readings:
- The activation signal SURVIVES re-notation (0.739 >> chance), so it is NOT pure canonical-string orthography; there is a notation-invariant structural component in the hidden states.
- But the surviving signal (0.739) is BELOW the no-chemistry char-n-gram on randomized SMILES (0.812 above), so the LLM hidden states do NOT exceed a surface substring probe even after randomization. This is the direct confirmation (not the proxy) that "encodes chemistry" is not supported over "linearly decodable structural signal."
- The best layer MOVES from 2 (canonical, very early = surface feature) to 20 (randomized, deeper). The early-layer canonical signal had an orthographic component that dropped on randomization, while a deeper notation-invariant structural component remained. This is the cleanest single piece of evidence that the canonical-string activation signal is part orthographic, part structural.

Companion supervision control (`fewshot_3038487.log`): the FIXED balanced K=10 few-shot output is 0.478 (n=1240, all parsed), near chance and barely above zero-shot 0.453. So given labeled in-context examples the model still cannot verbalize hERG; the probe advantage is not supervision and the expression gap is real.

Bottom line across all four controls: the expression GAP is fully intact and triangulated; what the model encodes is a notation-invariant structural signal that does not exceed a substring probe, so the framing stays "decodable structure," not "represents chemistry."

## Reproduce

`python eval/lipophilicity_control.py` (rdkit + sklearn, CPU). Raw in `results/lipophilicity_control.json`. Activation re-notation: `sbatch run_activation_randomize_cayuga.sh` on Cayuga (logs `act_rand_*.log`); few-shot: `sbatch run_fewshot_cayuga.sh` (`fewshot_*.log`).

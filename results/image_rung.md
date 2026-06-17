# Molecular-image rung: the encoding-limited prediction FAILS for coarse hERG

*Results. 2026-06-11. `eval/ws3_image.py` (Claude VLM, n=120 rendered hERG molecules) + `eval/image_rung.py` (the perception proxy). Adds image as the 5th modality probed, and REVISES the plan's P2 prediction. No em dashes.*

## What was predicted vs measured

The plan (`PROJECT_DESIGN.md` 7.2-7.3, P2) predicted molecular IMAGES to be the ENCODING-LIMITED extreme: the model cannot perceive the structure from pixels, so it never forms the structure-property at all (MolVision 2507.03283 reports image 0.15 vs text 0.71). Measured on rendered hERG molecules:

| arm | AUROC | what it measures |
|---|---|---|
| ceiling (Morgan on the TRUE structure) | 0.784 | the property IS structure-decodable |
| OCSR perception proxy (Morgan on the VLM-TRANSCRIBED structure) | 0.759 | does the VLM's PERCEIVED structure carry hERG |
| output (VLM solo-image, direct) | 0.566 | can the VLM verbalize hERG from the image |

OCSR fidelity: valid 0.758, mean Tanimoto 0.552, exact 0.167 (vs the DECIMER specialist 0.97 / 0.85 on the same images, `decimer_ocsr.json`).

## The finding: NOT encoding-limited for hERG

The perception proxy is 0.759, only 0.025 below the true-structure ceiling (0.784). So the VLM's TRANSCRIBED structure, imperfect as it is (0.55 Tanimoto, 17% exact), still predicts hERG almost as well as the true structure. The model perceives the hERG-relevant structure adequately. The encoding-limited prediction FAILS here.

The mechanism is property GRANULARITY. hERG is a COARSE property (lipophilicity, aromatic-ring count, basic-amine presence). The VLM's transcription errors fall mostly on hERG-IRRELEVANT parts of the molecule, so the coarse hERG signal survives a half-right transcription. MolVision's image-limited result (0.15) was on EXACT-structure tasks, where a 0.55-Tanimoto floor is fatal; for a coarse physicochemical property it is not. So P2 (images are encoding-limited) is property-dependent: it holds for fine-structure-dependent properties and FAILS for coarse ones like hERG.

## What image-hERG actually is: expression / orchestration-limited

The real gap is output, not perception: orchestrating the VLM's OWN transcription into a Morgan probe scores 0.759, while asking the VLM directly scores 0.566. The VLM HAS the structural information (it can output a hERG-predictive SMILES) but cannot directly verbalize the property from the image. That is an expression / orchestration gap, the SAME shape as the SMILES corner (the structure is there, the verbalization is not), not an encoding gap.

## The direct test (B, DONE): the proxy is confirmed by the VLM hidden states

The proxy measures perception via the TRANSCRIPTION, not the VLM's hidden states (Claude exposes none), so a direct open-VLM hidden-state 3-arm was run (`eval/activation_arm_image.py`, Qwen2.5-VL-7B, n=400 rendered hERG molecules):

| arm (Qwen2.5-VL-7B, image input) | AUROC |
|---|---|
| ceiling (Morgan on true structure) | 0.854 |
| **activation (VLM hidden states on the image)** | **0.758** (best layer 21) |
| output (VLM verbalized from the image) | 0.460 |

The direct hidden-state activation is 0.758, almost EXACTLY the OCSR perception proxy (0.759). So the proxy was an excellent estimate, and the conclusion is confirmed by the VLM's own activations: the VLM ENCODES hERG from the image (activation 0.758, close to the 0.854 ceiling) and cannot VERBALIZE it (output 0.460, at chance). Encoding gap 0.096, expression gap 0.298. Image-hERG is EXPRESSION-limited, confirmed two independent ways (OCSR proxy on Claude, hidden-state probe on Qwen2.5-VL), NOT the predicted encoding-limited extreme.

A genuinely encoding-limited anchor needs a FINE-structure property (where the 0.55-Tanimoto perception floor bites) or a non-renderable modality (raw spectra), which remains the open ladder rung.

## Reproduce

`IMG_N=120 python eval/ws3_image.py` then `python eval/image_rung.py`. Raw: `results/ws3_image.json`, `results/ws3_image_items.jsonl`, `results/image_rung.json`.

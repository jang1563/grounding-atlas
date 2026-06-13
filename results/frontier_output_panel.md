# Frontier output panel: scale closes the expression gap in proportion to web-exposure

*Results. 2026-06-12. `eval/frontier_output_panel.py`, n=300 balanced per rung, output arm only (activation is open-weight-only; Claude exposes no hidden states). Tests the two new hard-representation rungs (DNA, spectra) across the three latest Claude models, completing the frontier comparison they lacked. No em dashes.*

## The panel

| representation -> property | Qwen3-8B | haiku-4.5 | sonnet-4.6 | opus-4.8 | specialist ceiling |
|---|---|---|---|---|---|
| DNA sequence -> promoter | 0.396 | 0.622 | 0.798 | 0.815 | 0.889 (6-mer) |
| protein sequence -> thermostability | 0.486 | 0.481 | 0.561 | 0.585 | 0.699 (ESM) |
| protein sequence + ORGANISM -> thermostability | (0.486 seq) | 0.617 | 0.631 | 0.647 | 0.699 |
| MS m/z peaks -> hERG | 0.502 | 0.497 | 0.470 | 0.459 | 0.825 (structure) |

(DNA 8B output is anti-correlated at 0.396; all parses clean except sonnet spectra 14/300 fallback.)

## Protein: the organism-name shortcut, confirmed at frontier scale

The protein rung was the last gap (8B-only output 0.486). Frontier sequence-only grounding is WEAK and only modestly scale-dependent (haiku 0.481 chance, sonnet 0.561, opus 0.585): a raw amino-acid sequence -> thermostability mapping is web-poor, and the frontier picks up only a little (the partly-compositional thermostability signal). Adding the ORGANISM name lifts every model (haiku 0.617, sonnet 0.631, opus 0.647), confirming the plan's protein finding that the model grounds thermostability by the source organism (Thermus thermophilus -> stable), a web-rich prior. The lift is a clean within-entity web-exposure contrast (seq vs seq+organism), parallel to the variant text-vs-seq contrast. Note the scale-by-shortcut interaction: the organism lift is LARGER for the smaller model (haiku +0.136) than the larger (opus +0.062), and opus reads more from the sequence alone (0.585 vs haiku 0.481), so the bigger model leans LESS on the name shortcut.

## Two opposite scale patterns, both predicted by web-exposure

**DNA promoter: the expression gap CLOSES with scale.** A clean monotone ladder, 8B 0.396 (mis-grounded, a TATA heuristic mis-fired) to haiku 0.622 to sonnet 0.798 to opus 0.815, approaching the 0.889 specialist ceiling. So frontier models DO verbalize promoter-ness from a raw DNA sequence; the 8B cannot. The expression gap here is NOT vendor or scale invariant.

**Spectra (MS): the expression gap is FLAT at chance across every model.** 8B 0.502, haiku 0.497, sonnet 0.470, opus 0.459, all at chance, no scale trend. No model, frontier included, can verbalize hERG from a mass spectrum.

The contrast is exactly the web-exposure law expressed along the SCALE axis: how much scale closes the output gap tracks how web-exposed the representation-to-property mapping is.
- DNA sequence -> promoter is web-RICHER (promoter recognition is a textbook genomics task, with motif descriptions and prediction tutorials in pretraining text), so added capacity learns to read it, like variant-seq rising sonnet 0.55 to opus 0.80.
- MS m/z -> hERG is web-ZERO and computation-hard (no text binds fragment-mass lists to hERG, and recovering it needs structure elucidation, a multi-step inversion no forward pass does), so capacity does not help at all.

## The full web-exposure gradient (how much the frontier grounds it)

With protein added, the frontier (opus) output AUROC orders cleanly by the representation-to-property web-exposure, the web-exposure law read off a single panel:

| representation -> property | opus output | web-exposure of the mapping |
|---|---|---|
| DNA sequence -> promoter | 0.815 | high (textbook genomics task, motif descriptions) |
| protein + organism -> thermostability | 0.647 | high via the organism name (Thermus -> stable) |
| protein sequence -> thermostability | 0.585 | low (raw sequence), weak compositional signal |
| MS m/z peaks -> hERG | 0.459 | zero + computation-bound (no text, needs elucidation) |

The output rises monotonically with how often the representation-to-property mapping appears in text, and the within-entity protein contrast (sequence 0.585 vs sequence+organism 0.647) is the same shortcut effect as variant text-vs-seq: a web-rich identity token (organism, gene symbol) grounds the entity where its web-poor raw content does not.

## Why this matters: spectra is the first scale-INVARIANT expression-limit anchor

The 8B ENCODES hERG from the MS peak list (activation 0.729, `spectra_rung.md`) yet NO model (8B plus three frontier scales) can VERBALIZE it (flat ~0.5). This is qualitatively different from SMILES and DNA, where scale lifts the output (SMILES 8B 0.45 to sonnet 0.63; DNA 8B 0.40 to opus 0.82). The MS rung is the first representation where the verbalization is blocked at EVERY scale, because the spectrum-to-property step is not a recall the model can scale into, it is a computation (structure elucidation) the forward pass cannot perform. So the modality ladder now has both a scale-closable expression gap (DNA, web-rich) and a scale-invariant one (spectra, web-zero plus computation-bound), the two ends the web-exposure law predicts.

## Consequence for the earlier characterizations

The DNA rung's "expression-dominant, output anti-correlated" label was the 8B view; at frontier scale the output rises to 0.82 and the gap closes, so DNA is a SCALE-DEPENDENT expression gap (like variant-seq), not a fixed one. The spectra rung's chance output is confirmed scale-invariant (the 8B 0.502 was not a small-model artifact). Activation remains open-weight-only throughout; this panel is the output (verbalization) axis, where the frontier comparison lives.

## Reproduce

`source ~/.api_keys && PANEL_N=300 python eval/frontier_output_panel.py`. Raw: `results/frontier_output_panel.json`. Models: claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5-20251001.

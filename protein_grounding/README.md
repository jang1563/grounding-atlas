# protein_grounding - the protein branch of Bio_Grounding_Eval

A parallel copy of the axis-B grounding instrument (`../eval/`, `../results/head_to_head.md`),
run on **protein sequences** instead of SMILES. Same question, same three arms, one balanced
sample, one leakage-controlled split. The point is a **cross-domain comparison**: does the
shape of the grounding gap depend on how the modality appears in web text?

## The instrument (identical to the SMILES branch)

Does a general LLM ground a specialist model's signal by **content**? Three arms on one set:

- **ceiling (structure-probe analog):** ESM2 (`facebook/esm2_t33_650M_UR50D`, mean-pooled over
  residues) + LogisticRegression. Is the property in the content at all?
- **LLM-activation:** linear probe (LogReg) on the LLM's hidden states, per layer, over the
  raw amino-acid sequence. Does the LLM ENCODE it internally?
- **LLM-output:** the LLM generates a single probability. Does it VERBALIZE it?

Gaps: **encoding = ceiling - activation**; **expression = activation - output**.

**Plus axis A (the original-plan question): does Claude understand and utilize the domain
language?** A frontier-model arm (claude-sonnet-4-5, local API) on the same sequences splits
the "ground by name or by content" question into **understand** (a comprehension probe: can
Claude read surface features, length / cysteine count / composition, from the raw sequence,
scored vs ground truth), **utilize** (output + content-sensitivity: does its property answer
depend on the actual sequence), and **name vs content** (give the source organism instead of /
with the sequence). Result: Claude understands the surface (length r=0.96, Cys r=0.80) but does
not utilize the sequence for the deep property (Tm from sequence at chance, committed 0.523 vs
ESM 0.70); given the source organism it grounds Tm by that NAME instead (name 0.66 vs sequence
0.52), and the organism channel is objectively more predictive than the sequence (oracle 0.81 >
ESM 0.70). A false-organism conflict test is decisive: given a real sequence plus a
wrong-thermal-class organism, Claude follows the false name almost completely (AUROC vs truth
inverts to 0.001); the name-over-content pattern is universal across the Claude line
(sonnet-4-5, opus-4-8, sonnet-4-6). See `results/head_to_head.md`.

## The property

FLIP **meltome** thermostability (melting temperature Tm), binarized at the median (50.26 C)
to a balanced binary label, 1500 proteins (750/750). Leakage control: MMseqs2
sequence-identity clustering + `GroupKFold` (the protein analog of the SMILES scaffold split).
See `eval/README.md` for data and run details.

## The hypothesis (the headline)

SMILES appear in web text as "structure -> property" (chemistry literature), so on hERG the
LLM's internal representation was strong: **activation 0.79, near the structural ceiling
0.825, output at chance 0.45** -> an **expression-dominant** gap (the signal is inside, the
model just does not say it).

A protein melting temperature essentially never appears as "sequence -> Tm" in web text. So
the LLM's internal **encoding** of thermostability from a raw sequence may be **weaker** ->
a **larger encoding gap** than SMILES. If true, the grounding gap's shape depends on how the
modality is represented in web text: chemistry = expression-dominant, protein = possibly
encoding-dominant. That cross-domain contrast is the contribution this branch adds.

This is a directional prediction, registered before the run. The result (`results/`) reports
it honestly either way: a small protein encoding gap would falsify it.

## Layout

| Path | What |
|---|---|
| `eval/` | the three-arm code + Cayuga sbatch (see `eval/README.md`) |
| `data/protein_meltome.csv` | the balanced sample (id, sequence, label, tm, cluster) |
| `results/ceiling_gate.md` | is the signal in the ESM2 content? (the gate) |
| `results/head_to_head.md` | the three-arm result + cross-domain comparison to SMILES |

## Honest scope (carried from the SMILES branch)

- The activation probe measures **linear decodability** of the property from the hidden
  states, not "knowledge". Best-layer activation is **max-over-layers** (selection-biased);
  reported with a bootstrap CI.
- Thermostability correlates with amino-acid composition (charged-residue content, the
  IVYWREL signal), which is a legitimate content feature for ESM2 but also a route by which a
  general LLM could read it without "understanding" structure. A composition-preserving
  shuffle control is the natural next step (the protein analog of the SMILES scramble arm).
- One property, one open-weight model to start (Qwen3-8B); the panel and controls extend it.
- No em dashes; verified numbers only; capability framing (see `../GUARDRAILS.md`).

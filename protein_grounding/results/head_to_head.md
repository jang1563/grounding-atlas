# Head-to-head, protein branch: ESM2 ceiling vs LLM (meltome thermostability, axis B)

*2026-06-09. Qwen3-8B three arms on Cayuga GPU (a40, `eval/activation_arm_protein.py`, non-thinking). Claude domain-language arm via local API (`eval/head_to_head_protein.py` + `eval/comprehension_probe.py`, claude-sonnet-4-5; panel adds opus-4-8 and sonnet-4-6). No em dashes.*

**Question:** the structure-probe reads protein thermostability from the sequence content
(ESM2 cluster AUROC 0.70 LogReg / 0.74 RF, see `ceiling_gate.md`). Does a general LLM ground
it from the raw amino-acid sequence: does it ENCODE the property internally (activation
probe), and does it VERBALIZE it (output)? This is the protein copy of the SMILES instrument
(`../../results/head_to_head.md`), run to test one cross-domain hypothesis (below).

**Method:** three arms on ONE balanced 1500-protein meltome sample (750/750) under ONE
leakage-controlled split. The two trained probes (ESM2 structure-probe, LLM-activation) share
the SAME MMseqs2 sequence-identity cluster GroupKFold (the protein analog of the SMILES
scaffold split). The output arm is instrumented (parsed / percent / fallback) with the SAME
anchored parser as the SMILES branch + a bootstrap CI; best-layer activation is reported as
max-over-layers (selection-biased) with a bootstrap CI. ESM2 = `esm2_t33_650M_UR50D`,
mean-pooled over residues, the analog of the Morgan fingerprint.

## Results (Qwen3-8B, the three arms)

| arm | AUROC | 95% CI | note |
|---|---|---|---|
| structure-probe (ESM2, cluster) | **0.699** | - | = the ceiling gate; the apples-to-apples ceiling |
| LLM-activation (best layer) | **0.609** | [0.581, 0.637] | layer 1 of 37, MAX over layers (selection-biased) |
| LLM-output (generate) | **0.486** | [0.461, 0.513] | parsed 1500 / percent 0 / fallback 0 |

**encoding gap = 0.090** (probe - activation) · **expression gap = 0.123** (activation - output). One 1500-protein set, one cluster split.

- **Both gaps are statistically real.** Activation 0.609 sits below the 0.699 ceiling (CI top
  0.637 < 0.699) AND above chance (CI bottom 0.581 > 0.5), so the LLM encodes *some*
  thermostability but clearly less than the specialist. Output 0.486 covers chance (CI
  includes 0.5): Qwen3 cannot verbalize Tm. The activation and output CIs do not overlap, so
  the expression gap is real too.
- **Clean parse.** 1500/1500 outputs parsed as a number, 0 fallback, so the chance-level
  output is a measured chance, not an unparseable-output artifact (the C3 instrumentation).
- **Layer pattern: weak and shallow.** Chance at layer 0 (0.474), jumps to the peak at layer
  1 (0.609), then sits at a LOW plateau (0.54-0.61) across all 36 transformer layers, never
  approaching the ceiling. Same early-formation shape as SMILES hERG (peak by L1-2), but SMILES
  plateaued at 0.73-0.81 near its ceiling while protein plateaus far below its own.

## Cross-domain comparison: protein (here) vs SMILES (sibling), same model, same instrument

| modality | property | ceiling (LogReg, matched set+split) | activation (best, LLM) | output (LLM) | encoding gap | expression gap |
|---|---|---|---|---|---|---|
| SMILES | hERG block | 0.825 | 0.787 (L2) | 0.453 | 0.038 | **0.334** |
| protein | meltome Tm | 0.699 | 0.609 (L1) | 0.486 | **0.090** | 0.123 |

**Fraction of the specialist's above-chance signal the LLM recovers internally** (activation -
0.5) / (ceiling - 0.5): **SMILES 88%** (0.287/0.325) vs **protein 55%** (0.109/0.199). The
encoding gap as a fraction of the available signal is **12% for SMILES vs 45% for protein**.

*(SMILES row: Qwen3-8B instruct, balanced 1250-molecule hERG set, scaffold GroupKFold, from
`../../results/head_to_head.md`. Protein row: Qwen3-8B, balanced 1500-protein meltome set,
cluster GroupKFold. Same model and same three-arm instrument; the modality and the specialist
encoder differ by design.)*

## The hypothesis under test

SMILES appear in web text as "structure -> property" (chemistry literature), so on hERG the
LLM's internal representation reached near the structural ceiling (activation 0.787 vs ceiling
0.825, encoding gap 0.038) while output stayed at chance (0.453): an **expression-dominant**
gap. A protein melting temperature essentially never appears as "sequence -> Tm" in web text,
so the prediction was that the LLM's internal ENCODING of thermostability would be weaker ->
a **larger encoding gap** than SMILES, possibly an **encoding-dominant** shape. The result
above tests this directly. Because the two ceilings differ (0.825 vs 0.699), the comparison is
made on the gaps and on the fraction of the specialist's above-chance signal the LLM recovers,
not on absolute activation alone.

**Verdict: supported, with a caveat.** The protein encoding gap (0.090) is ~2.4x the SMILES
one (0.038), and relative to each specialist ceiling the LLM recovers only ~55% of the
available protein signal internally versus ~88% for SMILES. The gap's SHAPE shifts: SMILES is
sharply expression-dominant (encodes ~88%, cannot say it), while protein is much more balanced
between encoding and expression (encodes only ~55%, and what little it encodes it also does
not say). This is consistent with the web-exposure account in `../PROJECT_DESIGN.md` (§7.1):
chemistry's "structure -> property" mapping is web-frequent, protein's "sequence -> Tm" is
web-rare, and the internal encoding tracks that. The caveat is that the protein ceiling is
itself lower (0.699), so there is less signal overall and the absolute gaps are smaller and
noisier; the relative-fraction read is the load-bearing comparison, and a higher-ceiling
protein property (localization) is the clean follow-up to separate "web-rare" from "just a
harder task". One model, one property: this is a first data point on the modality ladder, not
the law.

## Does Claude understand and utilize the protein domain language? (original-plan axis A + content-sensitivity)

The arms above use Qwen3 (open weights, needed for hidden states). This arm asks the frontier
model directly, on the same meltome sequences, and splits the original plan's question (does
the model ground by name or by content, `../PROJECT_DESIGN.md` axis A; the measured anchor is
the recognition gap name ~100% vs accession ~2%) into three: **understand** (does Claude parse
the raw sequence at all), **utilize** (does it use the sequence for the property), and **name
vs content** (does it lean on the source organism instead). Model: claude-sonnet-4-5 (matches
the SMILES Claude arm). Scripts: `eval/comprehension_probe.py`, `eval/head_to_head_protein.py`.
Logs: `results/logs/claude_*.log`.

### Understand: comprehension probe (surface features, scored vs ground truth, n=120)

| feature | Pearson r | Spearman ρ | MAE | read |
|---|---|---|---|---|
| sequence length | **+0.961** | **+0.961** | 64 res (underestimates ~20%) | reads relative length almost perfectly |
| cysteine (C) count | **+0.799** | **+0.797** | 1.72 (exact 33%) | reads a specific residue's count |
| charged-residue % (D/E/K/R) | +0.462 | +0.345 | 4.24 pts (mean 22.3 vs truth 24.2) | reads gross composition (weaker) |

Claude is NOT treating the sequence as opaque noise. It parses length, a specific residue
count, and composition from the raw amino-acid string with real fidelity. It underestimates
absolute counts (a known LLM counting weakness), but the RANKING across proteins is strong
(Spearman tracks Pearson for length and cysteine, so it is not a few-outlier artifact). The
weakest read is charged-% (Spearman 0.345 < Pearson 0.462, so that one leans on a few extreme
points). Net: Claude reads the domain language at the compositional/surface level.

### Utilize: output + content-sensitivity (n=240 real; n=120 paired)

| condition | metric | value | read |
|---|---|---|---|
| output (real) | AUROC | 0.545 [0.504, 0.591] | grazes chance, far below ESM ceiling 0.70 |
| scrambled (paired) | mean&#124;Δ&#124; | 0.067 | answer barely moves when residue ORDER is destroyed |
| mismatched (paired) | mean&#124;Δ&#124; | 0.065 | answer barely moves when swapped for an OPPOSITE-label protein |

Claude's thermostability answer sits in a narrow band that hardly moves whether you shuffle
the residues (composition preserved, order destroyed) or replace the whole protein with one of
the opposite class (mean&#124;Δ&#124; ~0.07 in both). So the small output lift (0.545) is a
weak composition-level prior, not a content-driven read: Claude does not utilize the sequence
content for the deep property. (5-8% of calls returned empty and mapped to 0.5, the
decline-to-commit hedge; the SMILES branch saw this on scrambled SMILES, here it appears on
real sequences too.)

### Name vs content (the organism shortcut, the core thesis)

Meltome Tm is strongly organism-determined (thermophiles high, psychrophiles low: in this
sample *Thermus*/*Geobacillus*/*Picrophilus* are pos=1.00, the psychrophile *Oleispira
antarctica* pos=0.08). Giving Claude the source organism instead of / in addition to the
sequence isolates name- vs content-grounding. References: a perfect organism->Tm lookup
(leave-one-out group mean over 23 organisms) scores AUROC 0.81, and the ESM2 sequence ceiling
is 0.70, so the NAME channel objectively carries MORE label signal than the CONTENT channel.

| condition (n=240) | what Claude is given | AUROC [95% CI] | committed-only |
|---|---|---|---|
| content-only | sequence | 0.545 [0.504, 0.591] | 0.523 (n=228) |
| name-only | organism, no sequence | 0.592 [0.527, 0.660] | **0.658** (n=214) |
| matched | organism + sequence | 0.651 [0.583, 0.719] | 0.650 (n=225) |
| **name-conflict** | sequence + a FALSE (opposite-thermal) organism | **0.001 [0.000, 0.002]** | near-total inversion (236/240 committed) |
| *ref:* organism-oracle | perfect organism->Tm lookup | **0.81** | - |
| *ref:* ESM2 ceiling | sequence specialist | 0.70 | - |

- **The conflict test is decisive.** Given the real sequence PLUS a false organism of the
  opposite thermal class, Claude assigns P(thermostable) = **0.92** to truly non-stable proteins
  (tagged with a fake thermophile) and **0.21** to truly stable ones (fake psychrophile): it
  answers by the wrong organism name and ignores the sequence in front of it. Against the true
  label that is AUROC **0.001** (near-total inversion, 236/240 committed); flipping the
  prediction (1-p) recovers 0.999, i.e. Claude is near-perfectly faithful to the false name.
  When name and content conflict, Claude takes the name. This is the taxonomy's true "mismatched"
  (real content + wrong name). Caveat: the false organisms are unambiguous thermophiles /
  psychrophiles that Claude has confident textbook priors about, so the inversion is near-total;
  subtler organisms would move it less. The load-bearing claim is the direction and
  near-completeness, not the exact 0.001.
- **The name carries Claude, the sequence does not.** With the sequence alone Claude grazes
  chance (0.545; committed-only 0.523). The organism name lifts it above chance (name-only and
  matched CIs exclude 0.5), and matched (0.651) is best. When Claude actually commits to an
  organism answer it reaches 0.658 versus 0.523 for a committed sequence answer. So Claude
  grounds thermostability by the organism NAME, not the sequence CONTENT: the property analog
  of the recognition gap (name ~100% vs accession ~2%).
- **Honest twist: the name shortcut is the BETTER channel here.** organism-oracle (0.81) > ESM2
  (0.70) > Claude-matched (0.65), so leaning on the organism is not laziness; the name is
  objectively more predictive than the sequence for this property. The real cost is robustness:
  with no organism label (a novel / engineered / spaceflight sequence) Claude falls back to the
  chance-level content channel (`../PROJECT_DESIGN.md` 7.3, the GeneLab novel-variant case).
- **Claude under-exploits even the name** (matched 0.65 vs oracle 0.81) and declines to commit
  more when given only an organism (26/240 vs 12/240 for sequence), an epistemic-caution signal:
  it knows organism alone is partial.

### Frontier panel: name vs content across Claude models (universality, n=120)

Is the name-over-content grounding specific to sonnet-4-5? Re-run on two more frontier models.
AUROC is committed-only (non-numeric replies excluded); sonnet-4-6 declined or misformatted
heavily on the sequence+organism prompts (matched 66/120, conflict 73/120 fallback, flagged by
the parse instrumentation), so committed-only is the honest read there.

| model | content-only | name-only | matched | name-conflict |
|---|---|---|---|---|
| sonnet-4-5 (n=240) | 0.523 | 0.658 | 0.650 | **0.001** |
| opus-4-8 (n=120) | 0.543 | 0.756 | 0.686 | **0.326** |
| sonnet-4-6 (n=120) | 0.486 | 0.710 | 0.687 | **0.112** |

- **name-only > content-only in all three** (+0.13 to +0.21): the name-over-content lift is
  universal, not a single-model artifact.
- **name-conflict inverts below 0.5 in all three** (the false organism wins): name dominates
  content across the frontier line. opus-4-8 is the least slavish (0.326, it keeps some content
  sensitivity / hedging), sonnet-4-5 and sonnet-4-6 the most (0.001, 0.112).
- **content-only grazes chance in all three** (0.49-0.54): no Claude model reads thermostability
  from the sequence.
- Caveat: n=120 per cell (CIs ~+/-0.09); sonnet-4-6's heavy fallback on sequence+organism is a
  format quirk (it answers in percent and often overruns the 20-token cap before emitting a
  number), not a content signal, and committed-only corrects for it.

### The dissociation (the answer to the original-plan question)

**Claude understands the protein domain language at the surface, does not utilize the sequence
content for the deep property, and grounds that property by the organism NAME instead.** It
reads length (r=0.96), cysteine count (r=0.80), and composition (r=0.46) from the raw sequence,
yet predicts thermostability from the sequence at chance (committed-only 0.523 vs ESM 0.70), is
insensitive to scrambling and opposite-label swaps, does better from the organism name (0.658)
than from the sequence, and follows a FALSE organism over the sequence almost completely
(conflict AUROC 0.001, near-total inversion). This is the protein analog of the
SMILES O2 dissociation (the model detects invalid SMILES but cannot read the property of valid
ones): surface-parsed, deep-property-unread. The name-over-content pattern holds across the
Claude frontier line (sonnet-4-5, opus-4-8, sonnet-4-6: name-only > content-only, and the
false-organism conflict inverts below chance in all three). It also agrees with the Qwen3 arms (weak internal
encoding 0.609, chance output 0.486): across both an open model's internals and a frontier
model's output, the protein "sequence -> deep property" mapping is poorly grounded while the
surface IS read. This refines the "raw sequence is informational noise" claim (CoKE
2510.23127): noise for deep inference, not for surface parsing.

## Caveats

- The activation probe measures **linear decodability** of the property from the hidden
  states, not "knowledge". Best-layer activation is **max-over-layers** (selection-biased),
  reported with a bootstrap CI. A random-label control task would bound the supervised-readout
  advantage (selectivity); not yet run.
- **Moderate ceiling.** Thermostability binarized at the median is harder than hERG even for
  the specialist (0.70 vs 0.825), so there is less above-chance signal to begin with; the
  gaps are read relative to that.
- **Length / composition confound.** Thermostable proteins here are slightly shorter (mean
  283 vs 311 residues) and thermostability correlates with amino-acid composition (the IVYWREL
  signal). Mean-pooled ESM2 is length-invariant, but a general LLM reading the raw sequence
  could exploit composition or length. A composition-preserving shuffle (the protein analog of
  the SMILES scramble arm) and a length-matched control are the natural next steps.
- One open-weight model (Qwen3-8B), one property, zero-shot, non-thinking. The cross-architecture
  panel (Qwen3-32B/Base, Phi-4, gpt-oss) and the thinking arm extend it, as in the SMILES branch.
- **Claude arm scope.** One frontier model (claude-sonnet-4-5), n=240 output / 120 paired / 120
  comprehension, zero-shot. The comprehension correlations are a RANKING read (Claude
  underestimates absolute counts). Because Claude reads charged-% (r=0.46) and thermostability
  tracks composition, the small output lift (0.545) is plausibly that composition prior, which
  is consistent with the near-zero content-sensitivity; a composition-matched control would
  isolate it.
- The SMILES O1 result (LLMs near chance at zero-shot property prediction) is PRIOR ART; what
  is novel here is the cross-domain *shape* of the gap (encoding vs expression) measured on the
  LLM's own activations against a matched specialist ceiling.

## Next

- Cross-architecture panel and the thinking arm (mirror the SMILES `Next`).
- Composition-preserving shuffle + length-matched controls (content-sensitivity for protein),
  and a composition-matched control to isolate whether Claude's 0.545 output lift is just the
  charged-% prior it demonstrably reads.
- Claude name-vs-content panel DONE (opus-4-8, sonnet-4-6 confirm name-only > content-only and
  the false-organism conflict inverts). Still open: the comprehension panel + haiku-4-5; an
  AMBIGUOUS-organism conflict (the current false names are unambiguous extremophiles, so the
  0.001 inversion is an upper bound on name-following); raising sonnet-4-6's token cap to fix
  the matched/conflict fallback.
- A higher-ceiling protein property (e.g. subcellular localization) to check whether the
  cross-domain gap shape holds when the specialist ceiling is high.

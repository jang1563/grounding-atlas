# protein_grounding/eval - the protein branch of the axis-B instrument

The same three-arm grounding instrument as the SMILES branch (`../../eval/`), run on protein
sequences instead of SMILES. One balanced sample, one leakage-controlled split, three arms.

| Arm | SMILES branch | protein branch (here) | Question |
|---|---|---|---|
| ceiling (structure-probe) | Morgan fingerprint + LogReg | **ESM2 650M mean-pooled + LogReg** | is the property in the content at all? |
| LLM-activation | linear probe on LLM hidden states, per layer | same (Qwen3-8B on the raw AA sequence) | does the LLM ENCODE it internally? |
| LLM-output | LLM generates a single probability | same (probability of thermostability) | does the LLM VERBALIZE it? |

Gaps: **encoding = ceiling - activation**; **expression = activation - output**. Both probes
share ONE split so the gaps are comparable.

## Property and data

FLIP **meltome** (protein melting temperature, Tm), `mixed_split.fasta` (27,951 diverse
proteins across species). Binarized at the median Tm (50.26 C) -> balanced binary
"thermostable vs not". Length-filtered to 50-512 standard-AA residues, balanced to 1500
(750/750). Why thermostability: ESM2 reads it from the sequence (a real structural ceiling),
yet a melting temperature essentially never appears as "sequence -> Tm" in web text, so it is
the cleanest test of the cross-domain hypothesis (see `../README.md`).

## Leakage control (the protein analog of the scaffold split)

MMseqs2 sequence-identity clustering (`--min-seq-id 0.3 -c 0.8`) -> a cluster id per
sequence; both trained probes use `GroupKFold` on clusters. A single-protein DMS would
collapse to one cluster (no split possible), which is why this uses a sequence-diverse assay.
The SMILES branch groups by Murcko scaffold; this groups by sequence-identity cluster. Same
intent: no near-neighbor in both train and test.

## Scripts

| File | What | Where |
|---|---|---|
| `prepare_data.py` | download meltome, binarize at median, length-filter, balance, MMseqs2 cluster -> `protein_meltome.csv` (id,sequence,label,tm,cluster) | login node (CPU) |
| `setup_data_cayuga.sh` | installs the MMseqs2 static binary, then runs `prepare_data.py` | login node (CPU) |
| `ceiling_gate_protein.py` | ESM2 embed + LogReg/RF, random vs cluster split (the gate) | GPU |
| `run_ceiling_cayuga.sh` | sbatch for the ceiling gate | a40 |
| `activation_arm_protein.py` | the three arms on one set under one cluster split | GPU |
| `run_activation_cayuga.sh` | sbatch for the activation arm (override `ACT_MODEL` for the panel) | a40 (h100 alt) |
| `head_to_head_protein.py` | Claude output + content-sensitivity (utilize) | local API |
| `comprehension_probe.py` | Claude surface-feature comprehension (understand) | local API |

The last two are the **axis-A "does Claude understand and utilize the domain language" check**
(the original-plan question, see `../results/head_to_head.md`). They run locally against the
Claude API (`set -a; source <your-keys-file>; set +a`), not on Cayuga. `head_to_head_protein.py` is
the protein copy of `../../eval/head_to_head.py`; its `HH_COND` conditions are
`real` (content-only) / `scrambled` / `mismatched` / `name_only` / `matched`. The `name_only`
and `matched` conditions test name-vs-content and need the `organism` column in
`../data/protein_meltome_named.csv` (built by matching each sequence to the FLIP
full_dataset_sequences.fasta header, which carries the source organism). Per-item predictions
are saved to `hh_preds_<cond>.csv` and `comprehension_preds.csv` for reanalysis.

## Run order (Cayuga; run from your branch checkout directory)

```bash
# 0. one-time: data (CPU, login node)
cd <protein branch dir> && bash setup_data_cayuga.sh

# 1. ceiling gate (is the signal in the content?)
sbatch run_ceiling_cayuga.sh

# 2. the three arms (the decisive measurement)
sbatch run_activation_cayuga.sh
# panel: ACT_MODEL=Qwen/Qwen3-32B sbatch run_activation_cayuga.sh   (etc.)
```

Reuses the SMILES branch venv (torch 2.12 / transformers 5.10 / sklearn 1.9,
no `kernels` package) and its HF cache. a100 node g0001 is DRAIN; jobs target
a40 (scu-gpu, qos=normal) with an h100 alternative (preempt_gpu, qos=low) commented in the
sbatch.

## What is reused verbatim from `../../eval/activation_arm.py`

`parse_prob` (anchored last-number parser, percent handling, fallback count), `bootstrap_ci`,
`load_model` (4-bit / gpt-oss dequantize paths), `chat_input`, and the whole main() shape
(per-layer probe, best-layer-as-max-over-layers with selection-bias flag, SUMMARY + gaps
line). Swapped: Morgan FP -> ESM2 embedding; Murcko scaffold -> MMseqs2 cluster; the prompt.

# WS2 - Verifiable signal: the content-property substrate the B-axis needs [MAKE SIGNAL]

*Spec + generator. 2026-06-09. Companion to `../PROJECT_DESIGN.md` (WS2), `../eval/README.md` (the B-axis instrument that consumes this), `../docs/WS1_BACKLOG.md`. No em dashes.*

## What WS2 produces and why it is separate from WS1

The B-axis head-to-head needs a task where the **representation content is the ground truth**, a property that depends causally on the sequence / SMILES / structure, not a contested or memorizable label. No off-the-shelf asset is in that exact shape, so it must be generated. WS2 is the generator plus the check that what it generates is real signal. It partly PRECEDES the WS1-B measurement: you cannot run a probe-vs-LLM head-to-head on a domain until you have confirmed the property is in the content at all.

This generalizes the per-branch ceiling pipelines (`../eval/ceiling_gate.py`, the protein and variant ceiling scripts) into ONE modality-general tool: `generate_signal.py`.

## The verifiable-signal record format

Each emitted record (`admet/<endpoint>/pairs.jsonl`):

```json
{"id": "herg_1589", "modality": "smiles", "property": "herg",
 "condition": "matched", "representation": "<SMILES>", "label": 0, "source": "negbiodb_admet"}
```

Content-sensitivity `condition` values (the conditions the B-axis instrument cross-runs, `../eval/README.md`):

| condition | what changes | a content-grounder should | generated where |
|---|---|---|---|
| **matched** | real content, real label | be correct | data (here) |
| **re_notation** | same molecule, alternate VALID notation (randomized SMILES) | give the same answer (notation-invariant) | data (here) |
| **scrambled** | content corrupted (string shuffled, signal destroyed) | degrade | data (here) |
| **mismatched** | real content, WRONG entity's label asserted | follow content, not the asserted name | LLM-arm prompt |
| **content-only** | content shown, name/label removed | still solve from content | LLM-arm prompt |

`matched / re_notation / scrambled` are content transformations, so the generator emits them. `mismatched / content-only` involve an asserted name and are constructed at LLM-prompt time (head_to_head.py), not in the signal data.

## The verifiability gate (the point of WS2)

A candidate (representation, property) source is admissible ONLY if the property is genuinely in the content and survives a cold split. `gate()` runs, on the content features:

1. **content-feature probe** (SMILES: Morgan r2/2048; sequence: an SFM embedding) -> logistic + RF.
2. **two splits**: a random `StratifiedKFold` and a leakage-controlled **cold split** (Murcko-scaffold `GroupKFold` for SMILES; MMseqs2 cluster for protein; gene `GroupKFold` for variant). A large `random - cold` drop is the **DTI trap** (apparent ceiling from analog leakage that collapses cold).
3. **shuffled-label selectivity** (Hewitt-Liang 1909.03368): a random-label probe must sit at chance; `selectivity = cold_auroc - control_auroc`.

**PASS** iff `cold_auroc >= 0.65` AND `selectivity >= 0.10`. A PASS means the domain is a valid B-axis head-to-head task (there is a content signal for the LLM to fail-or-succeed at surfacing). A FAIL means "no measurable content signal under this featurizer", so it is NOT a head-to-head candidate (it would only measure the featurizer, not the LLM).

Honest caveat carried from `../eval/README.md`: gating on a high probe ceiling conditions the headline gap on SFM-favorable domains, so the reported probe-minus-LLM gap is an upper-bound estimate, not representative across all of biology.

## Modality-general by construction

A Source supplies `(content, label, modality)` + a featurizer + a leakage-control grouper + optional notation variants. Implemented and gated locally (CPU): **SMILES / ADMET** (Morgan FP, scaffold split). The same interface extends to:

| modality | source | featurizer (ceiling) | cold split | gate runs on |
|---|---|---|---|---|
| SMILES (done) | NegBioDB ADMET (7 endpoints) | Morgan r2/2048 | Murcko scaffold | local (CPU) |
| protein | FLIP / meltome, NegBioDB | ESM2-650M embedding | MMseqs2 cluster | Expanse GPU |
| variant | ClinVar + AlphaMissense / DMS | ESM-1v / AM score | gene GroupKFold | Expanse GPU |
| DNA/RNA | promoter / motif sets | NT / Evo2 embedding | chromosome / cluster | Expanse GPU |
| metabolite | HMDB | fingerprint / spectrum | scaffold | local / GPU |

The sequence rungs reuse the activation-arm container env already built on Expanse (`results/expanse_logs/`); adding one is a Source entry + the featurizer, not new machinery.

## Provenance

Borrows NullAtlas's verifiable-signal method (`Negative_result_DB/`, the WS2 substrate). NullAtlas's negative-evidence-coverage result (tested-and-failed vs never-tested, rho = -0.70 across models) is **cited, not re-measured** here; that is a knowledge-coverage capability, distinct from the content-grounding signal WS2 generates (see `../eval/README.md` "Adjacent"). Raw ADMET source: NegBioDB `admet_*` tables (ChEMBL-derived).

## Deliverable status

- `generate_signal.py` - the generator + gate (modality-general; SMILES implemented).
- `admet/<endpoint>/pairs.jsonl` - generated (representation, verifiable-property) pairs + content-sensitivity variants.
- `admet/<endpoint>/verifiability.json` + `verifiability_report.md` - the gate result per endpoint (which are admissible B-axis tasks).

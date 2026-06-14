# The AlphaGenome-era data war, and where grounding-atlas sits

*2026-06-14. Synthesis of the data-production-side reading (AlphaGenome, Perturb-seq,
MPRA/STARR-seq, long-read pangenomes) and two rung extensions it motivated. The
framing is the durable contribution; the two explorations are recorded honestly,
including their negative outcomes. No em dashes.*

## The field shift

A strong DNA-to-function model (AlphaGenome) widens the candidate space but does not
close causality: it says a variant or regulatory element looks important, not that it
is causal in a given disease cell state. So the bottleneck moves from the model to the
production of high-quality perturbation and functional-genomics data in the specific
cell states that matter. Public data concentrates on a handful of immortalized lines
(K562, ENCODE, GTEx); patient, disease, and developmental cell states are
under-measured. The competition after AlphaGenome is who measures the cell states the
model does not know.

## Where grounding-atlas sits (the durable connection)

grounding-atlas is the measurement and calibration layer for exactly this gap. The
web-exposure law is, restated for genomics, a map of where a public-data-trained model
can and cannot ground biological content, which is an a-priori map of where the
expensive new data must be produced. The project does not produce wet-lab data; it
says, before any item is seen, where the model is trustworthy and where a specialist
or a new measurement is required. That is the defensible bridge to the data war:
**grounding-atlas is the ruler that tells the data-production effort where to aim.**

## Two rung extensions, explored and reported honestly

**Regulatory rung with AlphaGenome as the specialist ceiling**
(`docs/ALPHAGENOME_CEILING_DESIGN.md`, `signal/regulatory/`). The pipeline works end
to end (AlphaGenome API, GTEx eQTL fetch, gene- and tissue-matched scoring). A naive
ceiling gate (AlphaGenome predicted LFC vs GTEx `nes` over significant eQTLs) is
inconclusive, because GTEx lead eQTLs are mostly tag SNPs in LD, not the causal variant
AlphaGenome scores, and `nes` (normalized) is not on the LFC scale. Proper validation
needs fine-mapped eQTLs. The LLM arm would predictably be orchestrate-won, the same
shape as the variant flagship (AlphaMissense 0.96), so it is confirmatory rather than
new. Parked at a clean state; AlphaGenome stands as a callable specialist on the
strength of its published validation.

**Data-density rung: web-exposure as a measured covariate**
(`docs/DATA_DENSITY_RUNG_DESIGN.md`, `signal/single_cell/`). The idea was to regress an
LLM's grounding of a cell state on functional-data density D(c) versus naming frequency
N(c). Outcome: NO-GO. Where D and N separate (immortalized lines vs biological cell
types) the entities are incommensurable for a grounding task; where the entities are
commensurable (biological cell types) D(GEO) and N(PubMed) are collinear (Spearman
+0.92, both track how studied a type is); and the marker-grounding task saturates at
the frontier (recall ~1.0). The apparent dissociation in the first probe was a
line-vs-primary artifact.

## The lesson (both negatives are the project's own laws restated)

Neither extension changes the project's conclusions; each fails in a way the project
already characterized. The data-density covariate is confounded exactly as P1 was (the
cross-entity web-exposure regression is mis-specified; only the within-entity contrast
is valid). The regulatory cell is orchestrate-won exactly as the decision map predicts
(call the specialist; do not train or trust the model on the novel case). The data war
does not move the line; it gives the line a sharper, clinically urgent real-world map.

## Reuse

Lead a positioning paragraph with: a frontier model widens the candidate space but does
not close causality, so the post-AlphaGenome contest is data production in disease cell
states, and grounding-atlas is the calibration map that says where a public-data model
fails and therefore where that data is worth producing. The two honest negatives are
evidence of the measurement discipline, not gaps.

# Budget arm: is the web-zero gap a web-exposure effect or a compute-budget artifact?

*A falsification test, not a confirmation. 2026-06-19. `eval/budget_arm.py`.*

## The worry

The leaderboard uses a 16-token SNAP decode. For the NUMERIC web-zero task (methylation betas -> age) and
the SYMBOLIC web-zero task (single-cell with anonymized gene ids), that conflates two explanations of the
near-chance output: the representation-to-property mapping is web-undocumented (web-exposure), versus the
model simply cannot integrate the input in 16 tokens (a compute-budget limit). If the second is true, the
"web-zero -> chance" result is partly a decode artifact, and web-exposure is over-attributed.

## Test

Re-run with a reasoning budget (max_tokens 1024 + "reason step by step, then end with Probability:") and
compare to the snap AUROC. methyl/age and msa/conservation (web-rich control) at n=40; the single-cell
anon result was confirmed at n=200 (opus). Pre-registered: if the snap gap was a compute limit, the
budget AUROC rises; if it is web-exposure, it stays near chance. MSA should stay high (control).

| task | web | model | snap (16-tok) | budget (1024 + reasoning) | delta |
|---|---|---|---|---|---|
| methyl/age | zero | opus | 0.542 | 0.535 | -0.01 |
| methyl/age | zero | gpt-4o | 0.459 | 0.415 | -0.04 |
| single_cell/cd8t_nk:anon | zero | opus | 0.495 | 0.545 (n=200, CI 0.476-0.619) | +0.05 n.s. |
| single_cell/cd8t_nk:anon | zero | gpt-4o | 0.460 | 0.425 (n=40) | -0.04 |
| msa/conservation | rich | opus | 0.962 | 0.994 | +0.03 |
| msa/conservation | rich | gpt-4o | 0.952 | 0.945 | -0.01 |

## Result

The compute-budget rival is REFUTED for both web-zero tasks. A reasoning budget does not recover methyl
(flat in both models) or single-cell anon (0.545 at n=200, CI includes 0.5; the n=40 0.628 was a
small-n fluctuation). The web-rich control is stable. So the web-zero verbalization failures are
budget-invariant, consistent with web-exposure / needing documented-or-trained knowledge, not a 16-token
limit. The core claim was tested and survived on this axis.

## Honest scope (what this does NOT settle)

This refutes ONE rival. It does not address two over-extensions that stand:
1. The "web=zero" leaderboard column conflates distinct failure mechanisms. Only the symbolic controlled
   pairs (gene names vs anon ids, materials formula vs anon) are clean web-exposure. The embedding rows
   (ESM / Nucleotide Transformer) fail because they need a trained read-out, not because of
   documentation; they are an orchestrate result, not web-exposure. The structural rows
   (hERG as graph / 3D / sequence) are representation-parsing limits.
2. Token-familiarity vs mapping-documentation is untested. The clean discriminator is a task with
   FAMILIAR symbols but an UNDOCUMENTED (yet head-learnable) mapping: if the model grounds, familiarity
   suffices; if it stays at chance, the mapping must be documented. This needs a careful task design.

"""GroundBench task registry: the benchmark's task set, decoupled from data format.

Each task is a (representation, operational binary target) classification with:
  - prompt   : a versioned template with a {rep} slot, ending in a numeric anchor;
  - orient   : which label the clause counts as positive ("align" = label 1, "oppose" = label 0);
  - web      : a preregistered descriptive exposure tag ("rich", "zero", or "mixed"), not a
               physical law or a direct corpus-frequency measurement;
  - reference: a named specialist comparator, a float or a key into
               results/benchmark/ceilings.json. ``ceiling`` is retained as a legacy alias;
  - kind     : "pairs" (a pairs.jsonl with matched/scrambled conditions, enabling memo_delta) or
               "twocol" (a CSV with two representation columns, a web-rich vs web-zero contrast).

Adding a task is one entry here; the harness (run_grounding_eval.py) loops this registry. Keeping
the registry separate from the runner is what makes the benchmark extensible by others.
"""
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIGNAL = os.path.join(ROOT, "signal")


def _mol(clause):
    return ("Estimate the probability (a single number between 0 and 1) that this molecule "
            f"{clause}. Judge only from the representation below.\nsmiles: {{rep}}\nProbability:")


def _cell(pos, neg):
    return ("These are the 50 highest-expressed genes retained by the benchmark preprocessing for "
            f"one human immune cell (PBMC), ordered from highest to lower expression:\n{{rep}}\n"
            f"Is it a {pos} or a {neg}? "
            "Reply with ONLY a number between "
            f"0 and 1 = the probability that it is a {pos}.")


_VARIANT_TEXT = (
    "Estimate the probability (a single number between 0 and 1) that this ClinVar-style record is "
    "assigned to the pathogenic/likely-pathogenic class in the benchmark source snapshot. This is a "
    "database-assertion target, not an independent clinical diagnosis. Judge only from the record below."
    "\nvariant record: {rep}\nProbability:"
)
_VARIANT_SEQ = (
    "Estimate the probability (a single number between 0 and 1) that the variant represented below is "
    "assigned to the pathogenic class in the benchmark source snapshot. Judge only from the representation "
    "below.\nvariant representation: {rep}\nProbability:"
)
_METHYL = ("Estimate the probability (a single number between 0 and 1) that this individual is older "
           "than the benchmark cohort median of 33 years, from their blood DNA methylation profile "
           "(CpG site: beta-value pairs). "
           "Judge only from the values below.\nmethylation: {rep}\nProbability:")
_MSA = (
    "Estimate the probability (a single number between 0 and 1) that this protein multiple-sequence-"
    "alignment column belongs to the lower within-family Shannon-entropy tercile (the benchmark's "
    "conserved class), rather than the upper tercile (the variable class). Middle-tercile columns are "
    "excluded. Judge only from the residues below.\ncolumn: {rep}\nProbability:"
)
_METAL = (
    "Estimate the probability (a single number between 0 and 1) that this composition is assigned to "
    "the metallic class, rather than the non-metallic class, in the MatBench experimental-is-metal "
    "benchmark. Judge only from the composition below.\ncomposition: {rep}\nProbability:"
)
_ESM = ("Below is a 640-dimensional protein embedding from the ESM-2 protein language model (a "
        "scientific foundation model). Estimate the probability (a single number between 0 and 1) "
        "that this protein is thermostable (melting temperature above the dataset median). Judge "
        "only from the embedding.\nembedding: {rep}\nProbability:")


def _herg(repline):
    return ("Estimate the probability (a single number between 0 and 1) that this molecule blocks the "
            "hERG potassium channel (cardiotoxicity risk). Judge only from the representation below.\n"
            f"{repline}: {{rep}}\nProbability:")


_RNA = ("Estimate the probability (a single number between 0 and 1) that this nucleotide sequence is "
        "protein-coding (vs non-coding). Judge only from the sequence below.\nsequence: {rep}\nProbability:")
_HISTO = ("This is a 96x96 H&E-stained PatchCamelyon image patch. Estimate the probability (a single "
          "number between 0 and 1) that its central 32x32 region contains at least one tumor pixel "
          "(vs no tumor pixel in that central region).\nProbability:")
_NT = ("Below is a 512-dimensional nucleotide-sequence embedding from the Nucleotide Transformer (a "
       "genomic foundation model). Estimate the probability (a single number between 0 and 1) that the "
       "underlying sequence is protein-coding (vs non-coding). Judge only from the embedding.\n"
       "embedding: {rep}\nProbability:")


TASKS = {
    # ADMET: SMILES -> empirical property (pairs.jsonl, matched/scrambled). web-rich (drug/SMILES
    # tokens are web-documented). orientation per the structural-alert audit (ames = oppose).
    "admet/herg":         dict(kind="pairs", data="admet/herg/pairs.jsonl",
                               prompt=_mol("blocks the hERG potassium channel (cardiotoxicity risk)"),
                               orient="align", web="rich", ceiling="admet/herg",
                               pair_group="herg/common", entity_field="representation"),
    "admet/cyp3a4":       dict(kind="pairs", data="admet/cyp3a4/pairs.jsonl",
                               prompt=_mol("inhibits the CYP3A4 enzyme"),
                               orient="align", web="rich", ceiling="admet/cyp3a4"),
    "admet/cyp2d6":       dict(kind="pairs", data="admet/cyp2d6/pairs.jsonl",
                               prompt=_mol("inhibits the CYP2D6 enzyme"),
                               orient="align", web="rich", ceiling="admet/cyp2d6"),
    "admet/ames":         dict(kind="pairs", data="admet/ames/pairs.jsonl",
                               prompt=_mol("is mutagenic in the Ames test"),
                               orient="oppose", web="rich", ceiling="admet/ames"),
    "admet/solubility":   dict(kind="pairs", data="admet/solubility/pairs.jsonl",
                               prompt=_mol("is highly soluble in water"),
                               orient="oppose", web="rich", ceiling="admet/solubility"),
    "admet/permeability": dict(kind="pairs", data="admet/permeability/pairs.jsonl",
                               prompt=_mol("is highly permeable across a cell membrane"),
                               orient="oppose", web="rich", ceiling="admet/permeability"),
    # Single-cell: expression -> cell type, the SAME cells in a web-rich (gene NAMES) and a
    # anonymized-ID form. The name/anon pair is a controlled representation contrast; attributing
    # any difference specifically to web exposure requires additional corpus evidence.
    "single_cell/cd8t_nk:name": dict(kind="twocol", data="single_cell/cd8t_nk.csv", col="cell_sentence",
                                     prompt=_cell("CD8+ T cell", "NK cell"), orient="align",
                                     web="rich", ceiling=0.992, pair_group="single_cell/cd8t_nk",
                                     entity_field="__row_index__"),
    "single_cell/cd8t_nk:anon": dict(kind="twocol", data="single_cell/cd8t_nk.csv", col="anon",
                                     prompt=_cell("CD8+ T cell", "NK cell"), orient="align",
                                     web="zero", ceiling=0.992, pair_group="single_cell/cd8t_nk",
                                     entity_field="__row_index__"),
    "single_cell/mono:name":    dict(kind="twocol", data="single_cell/mono_cd14_fcgr3a.csv", col="cell_sentence",
                                     prompt=_cell("classical CD14+ monocyte", "non-classical CD16+ monocyte"),
                                     orient="align", web="rich", ceiling=0.989,
                                     pair_group="single_cell/mono", entity_field="__row_index__"),
    "single_cell/mono:anon":    dict(kind="twocol", data="single_cell/mono_cd14_fcgr3a.csv", col="anon",
                                     prompt=_cell("classical CD14+ monocyte", "non-classical CD16+ monocyte"),
                                     orient="align", web="zero", ceiling=0.989,
                                     pair_group="single_cell/mono", entity_field="__row_index__"),
    # Token-familiarity dissociation: the SAME CD8-T/NK cells as real, familiar gene SYMBOLS but with the
    # web-famous markers (GZMB/NKG7/CD8A/...) dropped. Head still reads it (0.979), tokens stay familiar,
    # but the marker->type mapping is less direct. name vs obscure vs anon separates three input
    # conditions; it does not by itself identify the mechanism behind their differences.
    "single_cell/cd8t_nk:obscure": dict(kind="twocol", data="single_cell/cd8t_nk_obscure.csv", col="cell_sentence",
                                        prompt=_cell("CD8+ T cell", "NK cell"), orient="align",
                                        web="mixed", ceiling=0.979, pair_group="single_cell/cd8t_nk",
                                        entity_field="__row_index__"),
    # Variant effect: the SAME variants as web-rich HGVS text vs a web-poor protein sequence
    # (within-entity web-exposure pair). label 1 = pathogenic (align; leakage-audited, 0/2400).
    "variant/text":       dict(kind="twocol", data="clinvar/variant_text.csv", col="text",
                               prompt=_VARIANT_TEXT, orient="align", web="rich", ceiling=None),
    "variant/seq":        dict(kind="pairs", data="variant_seq/pairs.jsonl",
                               prompt=_VARIANT_SEQ, orient="align", web="zero", ceiling=None),
    # Methylation beta vector -> age and MSA column -> conservation are distinct operational tasks,
    # not a matched causal contrast.
    "methyl/age":         dict(kind="twocol", data="methyl/methyl_age.csv", col="beta_text",
                               prompt=_METHYL, orient="align", web="zero", ceiling=0.701),
    "msa/conservation":   dict(kind="twocol", data="msa/msa_conservation.csv", col="column",
                               prompt=_MSA, orient="align", web="rich", ceiling=0.999),
    # Materials (generality beyond biology): metal-vs-not from a formula, web-rich element symbols
    # vs anonymized elements (a third name/anon controlled pair). label 1 = metal.
    "materials/metal:formula": dict(kind="twocol", data="materials/metal.csv", col="formula",
                                    prompt=_METAL, orient="align", web="rich", ceiling=0.927,
                                    pair_group="materials/metal", entity_field="__row_index__"),
    "materials/metal:anon":    dict(kind="twocol", data="materials/metal.csv", col="anon",
                                    prompt=_METAL, orient="align", web="zero", ceiling=0.927,
                                    pair_group="materials/metal", entity_field="__row_index__"),
    # SFM leg (the LLM x SFM interface): can the LLM read a scientific foundation model's OUTPUT
    # embedding? ESM-2 (150M, 640-dim) embedding -> thermostability. The ultimate web-zero form (an
    # abstract float vector). The reference is a supervised readout on that same embedding (LogReg,
    # cluster GroupKFold). Comparing it with native LLM output estimates a supervised
    # decodability-native-output gap, not latent knowledge.
    "protein/esm2_emb": dict(kind="emb", data="sfm_embedding/meltome_esm2.npz",
                             prompt=_ESM, orient="align", web="zero", ceiling=0.633),
    # hERG in three more representations of the SAME molecules (graph / 13C-NMR shifts / 3D coords).
    # These are alternate renderings of the same molecules. A fair representation contrast therefore
    # uses the stable-ID intersection; any relation to web exposure remains a hypothesis.
    "herg/graph":    dict(kind="twocol", data="graph/herg_graph.csv", col="graph",
                          prompt=_herg("molecular graph"), orient="align", web="zero", ceiling=None,
                          pair_group="herg/common", entity_field="smiles"),
    "herg/nmr":      dict(kind="twocol", data="nmr/herg_nmr.csv", col="nmr",
                          prompt=_herg("carbon-13 NMR chemical shifts"), orient="align", web="zero", ceiling=None,
                          pair_group="herg/common", entity_field="smiles"),
    "herg/struct3d": dict(kind="twocol", data="structure3d/herg_xyz.csv", col="xyz",
                          prompt=_herg("3D atomic coordinates (element x y z)"), orient="align", web="zero",
                          ceiling=None, pair_group="herg/common", entity_field="smiles"),
    # RNA coding-vs-noncoding from the nucleotide sequence. web=mixed: ORF/codon structure is a
    # partially documented heuristic the model may use. label 1 = coding. ceiling = 3-mer LR.
    "rna/coding":    dict(kind="twocol", data="rna/coding.csv", col="smiles",
                          prompt=_RNA, orient="align", web="mixed", ceiling=0.856),
    # Image / VLM arm: H&E histopathology patch -> PatchCamelyon central-region label. Historical
    # activation-probe and output results are separate estimands and require patient-level split
    # provenance before confirmatory interpretation.
    "histo/pcam_tumor": dict(kind="image", data="histo/pcam.csv", col="img",
                             prompt=_HISTO, orient="align", web="rich", ceiling=None),
    # 2nd SFM leg (genomic): Nucleotide Transformer (v2-50m) embedding of the RNA coding sequences ->
    # coding-vs-noncoding. This provides an RNA representation contrast. The reference is a trained
    # readout on the embedding; its performance establishes supervised decodability only.
    "rna/nt_emb": dict(kind="emb", data="sfm_embedding/rna_nt.npz",
                       prompt=_NT, orient="align", web="zero", ceiling=0.918,
                       entity_id_scope="snapshot_local_row"),
}

# Registry-level validation status. Quarantined tasks remain readable for historical result
# reproduction but are not admissible benchmark tasks. Exploratory tasks are runnable explicitly
# but excluded from the default CORE until they clear their own signal/readability gate.
TASKS["variant/seq"].update(
    status="quarantined",
    status_reason=(
        "The committed rows omit mutation position and WT-to-ALT identity and originally lacked "
        "stable entity IDs; rebuild from explicit variant records before scoring."
    ),
)
TASKS["protein/esm2_emb"].update(
    status="exploratory",
    status_reason="The committed embedding reference AUROC is 0.633, below the preregistered 0.65 viability gate.",
)
_UNDERSPECIFIED_ASSAY_TASKS = {
    "admet/herg",
    "admet/cyp3a4",
    "admet/cyp2d6",
    "admet/ames",
    "admet/solubility",
    "admet/permeability",
    "herg/graph",
    "herg/nmr",
    "herg/struct3d",
}
for _task_id in _UNDERSPECIFIED_ASSAY_TASKS:
    TASKS[_task_id].update(
        status="exploratory",
        status_reason=(
            "Provisional assay task: the committed source does not yet provide release-grade "
            "assay context, units, binary threshold, and replicate-aggregation provenance."
        ),
    )
for _task_id, _task in TASKS.items():
    _task.setdefault("status", "active")
    _task.setdefault("status_reason", "")
    _task.setdefault("entity_id_scope", (
        "snapshot_local_row"
        if _task.get("entity_field") == "__row_index__"
        else "content_hash_or_source_id"
    ))
    # `ceiling` remains as a deprecated alias for old scripts and historical scorecards.
    _task["reference"] = _task.get("ceiling")
    _task["reference_comparability"] = (
        "context_only_external_cohort_or_split"
        if _task["reference"] is not None
        else "not_available"
    )

TRUTH_LEVEL_CODES = frozenset({"T0", "T1", "T2", "T3", "T4", "T5"})
TRUTH_TAXONOMY_VERSION = "groundbench-t0-t5-v1"

# Machine-facing truth levels classify the TARGET, not the representation. In particular, the
# RDKit-derived graph/NMR/3D inputs retain the empirical hERG target's T2 code, and an embedding
# does not change an empirical melting-temperature target into a software-defined T1 target.
TRUTH_LEVEL_CODE_BY_DOMAIN = {
    "admet": "T2",
    "single_cell": "T3",
    "variant": "T4",
    "methyl": "T3",
    "msa": "T0",
    "materials": "T3",
    "protein": "T2",
    "herg": "T2",
    "rna": "T3",
    "histo": "T3",
}

# Descriptive provenance is intentionally separate from the T0-T5 code. These strings are retained
# exactly from the pre-taxonomy registry so downstream users can distinguish, for example, an
# operational assay aggregation from an expert image annotation within a broader truth level.
TARGET_SOURCE_KIND_BY_DOMAIN = {
    "admet": "operational_assay_aggregation",
    "single_cell": "constructed_annotation",
    "variant": "database_assertion",
    "methyl": "constructed_measurement",
    "msa": "algorithmic_construct",
    "materials": "database_label",
    "protein": "empirical_measurement_via_embedding",
    "herg": "operational_assay_aggregation",
    "rna": "database_annotation",
    "histo": "expert_image_annotation",
}

# Deprecated compatibility alias. Values are T0-T5 codes, never descriptive source-kind strings.
TRUTH_LEVEL = TRUTH_LEVEL_CODE_BY_DOMAIN



def _resolve_truth_metadata(task_id, task):
    """Resolve paired task overrides or fall back to domain-level truth defaults."""

    domain = task_id.split("/", 1)[0]
    has_code_override = "truth_level_code" in task
    has_kind_override = "target_source_kind" in task
    if has_code_override != has_kind_override:
        missing = (
            "target_source_kind"
            if has_code_override
            else "truth_level_code"
        )
        raise RuntimeError(
            f"{task_id}: task-level truth overrides must provide both "
            f"truth_level_code and target_source_kind; missing {missing}"
        )
    if has_code_override:
        truth_level_code = task["truth_level_code"]
        target_source_kind = task["target_source_kind"]
    else:
        if (
            domain not in TRUTH_LEVEL_CODE_BY_DOMAIN
            or domain not in TARGET_SOURCE_KIND_BY_DOMAIN
        ):
            raise RuntimeError(
                f"{task_id}: no domain truth defaults; declare task-level "
                "truth_level_code and target_source_kind"
            )
        truth_level_code = TRUTH_LEVEL_CODE_BY_DOMAIN[domain]
        target_source_kind = TARGET_SOURCE_KIND_BY_DOMAIN[domain]
    if (
        not isinstance(truth_level_code, str)
        or truth_level_code not in TRUTH_LEVEL_CODES
    ):
        raise RuntimeError(
            f"{task_id}: truth_level_code={truth_level_code!r} is outside T0-T5"
        )
    if (
        not isinstance(target_source_kind, str)
        or not target_source_kind.strip()
    ):
        raise RuntimeError(
            f"{task_id}: target_source_kind must be a non-empty string"
        )
    deprecated_alias = task.get("truth_level")
    if deprecated_alias is not None and deprecated_alias != truth_level_code:
        raise RuntimeError(
            f"{task_id}: deprecated truth_level alias must equal truth_level_code"
        )
    task["truth_level_code"] = truth_level_code
    task["target_source_kind"] = target_source_kind
    # Deprecated serialized alias retained for older consumers.
    task["truth_level"] = truth_level_code
    return task


if set(TRUTH_LEVEL_CODE_BY_DOMAIN) != set(TARGET_SOURCE_KIND_BY_DOMAIN):
    raise RuntimeError("Truth-code and target-source-kind defaults must cover the same domains")
for _domain, _default_code in TRUTH_LEVEL_CODE_BY_DOMAIN.items():
    _resolve_truth_metadata(f"{_domain}/__domain_default__", {})
for _task_id, _task in TASKS.items():
    _resolve_truth_metadata(_task_id, _task)

# Phase-0 causal-study metadata. These identifiers describe the benchmark interface and the
# operational question; they do not assert that a model internally represents a biological causal
# variable. `task_family_id` is deliberately broader than `biological_question_id`, so a future
# analysis can hold out an endpoint or an entire family rather than treating alternate renderings as
# independent tasks.
_TASK_METADATA = {
    "admet/herg": ("bq:molecule:herg_blockade", "tf:molecular_admet", "smiles"),
    "admet/cyp3a4": ("bq:molecule:cyp3a4_inhibition", "tf:molecular_admet", "smiles"),
    "admet/cyp2d6": ("bq:molecule:cyp2d6_inhibition", "tf:molecular_admet", "smiles"),
    "admet/ames": ("bq:molecule:ames_mutagenicity", "tf:molecular_admet", "smiles"),
    "admet/solubility": ("bq:molecule:aqueous_solubility", "tf:molecular_admet", "smiles"),
    "admet/permeability": ("bq:molecule:membrane_permeability", "tf:molecular_admet", "smiles"),
    "single_cell/cd8t_nk:name": (
        "bq:pbmc:cd8t_vs_nk",
        "tf:single_cell_annotation",
        "ranked_gene_symbols_top50",
    ),
    "single_cell/cd8t_nk:anon": (
        "bq:pbmc:cd8t_vs_nk",
        "tf:single_cell_annotation",
        "ranked_anonymized_gene_ids_top50",
    ),
    "single_cell/mono:name": (
        "bq:pbmc:classical_vs_nonclassical_monocyte",
        "tf:single_cell_annotation",
        "ranked_gene_symbols_top50",
    ),
    "single_cell/mono:anon": (
        "bq:pbmc:classical_vs_nonclassical_monocyte",
        "tf:single_cell_annotation",
        "ranked_anonymized_gene_ids_top50",
    ),
    "single_cell/cd8t_nk:obscure": (
        "bq:pbmc:cd8t_vs_nk",
        "tf:single_cell_annotation",
        "ranked_marker_depleted_gene_symbols_top50",
    ),
    "variant/text": (
        "bq:variant:clinvar_pathogenic_assertion",
        "tf:variant_assertion",
        "clinvar_style_text",
    ),
    "variant/seq": (
        "bq:variant:clinvar_pathogenic_assertion",
        "tf:variant_assertion",
        "incomplete_protein_sequence_context",
    ),
    "methyl/age": (
        "bq:methylation:age_above_cohort_median",
        "tf:methylation_demographic",
        "cpg_beta_value_pairs",
    ),
    "msa/conservation": (
        "bq:protein:msa_column_conservation",
        "tf:protein_sequence_conservation",
        "msa_column_residues",
    ),
    "materials/metal:formula": (
        "bq:materials:experimental_metallicity",
        "tf:materials_property",
        "chemical_formula",
    ),
    "materials/metal:anon": (
        "bq:materials:experimental_metallicity",
        "tf:materials_property",
        "anonymized_element_formula",
    ),
    "protein/esm2_emb": (
        "bq:protein:thermostability_above_dataset_median",
        "tf:protein_thermostability",
        "esm2_embedding_640d",
    ),
    "herg/graph": (
        "bq:molecule:herg_blockade",
        "tf:molecular_admet",
        "molecular_graph_text",
    ),
    "herg/nmr": (
        "bq:molecule:herg_blockade",
        "tf:molecular_admet",
        "carbon13_nmr_shift_list",
    ),
    "herg/struct3d": (
        "bq:molecule:herg_blockade",
        "tf:molecular_admet",
        "xyz_coordinates",
    ),
    "rna/coding": (
        "bq:rna:protein_coding_status",
        "tf:rna_coding",
        "nucleotide_sequence",
    ),
    "histo/pcam_tumor": (
        "bq:histopathology:pcam_central_tumor",
        "tf:histopathology_annotation",
        "h_and_e_image_96x96",
    ),
    "rna/nt_emb": (
        "bq:rna:protein_coding_status",
        "tf:rna_coding",
        "nucleotide_transformer_embedding_512d",
    ),
}
if set(_TASK_METADATA) != set(TASKS):
    raise RuntimeError("Phase-0 metadata must cover the task registry exactly")

SPLIT_SCOPE_BIOLOGICAL = "protein_family"
SPLIT_SCOPE_ENTITY_PROXY = "exact_entity_proxy_not_biological_dependency_group"
SPLIT_SCOPE_UNAVAILABLE = "unavailable_no_biological_dependency_metadata"
BIOLOGICAL_SPLIT_SCOPES = frozenset(
    {
        "assay_batch",
        "chromosome",
        "compound_scaffold",
        "donor",
        "gene_family",
        "locus",
        "patient",
        "perturbation",
        "protein_family",
        "regulatory_element",
        "slide",
        "study",
    }
)


def _validate_interventional_split_namespace(task_id, task):
    """Require one cross-dataset namespace for biologically grouped interventions."""

    has_intervention_pair = task.get("intervention_pair_field") is not None
    has_biological_group = (
        task.get("split_group_field")
        and task.get("split_group_scope") in BIOLOGICAL_SPLIT_SCOPES
    )
    if has_intervention_pair and has_biological_group:
        namespace = task.get("split_group_namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise RuntimeError(
                f"{task_id}: biologically grouped interventional tasks require "
                "split_group_namespace so source/target overlap cannot be hidden "
                "by task-scoped IDs"
            )


def _validate_truth_intervention_contract(task_id, task):
    """Require row-level intervention identity exactly for T5 targets."""

    truth_level_code = task.get("truth_level_code")
    intervention_pair_field = task.get("intervention_pair_field")
    intervention_pair_id = task.get("intervention_pair_id")
    if intervention_pair_id is not None:
        raise RuntimeError(
            f"{task_id}: task-level intervention_pair_id is forbidden; T5 tasks "
            "must declare a row-level intervention_pair_field"
        )
    if truth_level_code == "T5":
        if (
            not isinstance(intervention_pair_field, str)
            or not intervention_pair_field.strip()
        ):
            raise RuntimeError(
                f"{task_id}: T5 tasks require a non-empty row-level "
                "intervention_pair_field"
            )
    elif intervention_pair_field is not None:
        raise RuntimeError(
            f"{task_id}: intervention_pair_field requires truth_level_code='T5'"
        )


for _task_id, _task in TASKS.items():
    _question_id, _family_id, _representation_level = _TASK_METADATA[_task_id]
    _task["biological_question_id"] = _question_id
    _task["task_family_id"] = _family_id
    # Current tasks have no interventional pair field. Future DMS/MPRA/Perturb-seq
    # registrations can name a row-level source field here.
    _task.setdefault("intervention_pair_field", None)
    _task.setdefault("intervention_pair_id", None)
    _task["factor_levels"] = {
        "representation": _representation_level,
        "web_exposure": _task["web"],
        "causal_status": "descriptive_interface_only",
    }
    if _task_id == "msa/conservation":
        # The source table contains a genuine protein-family field. It is currently the only
        # registry task with a biological dependency group that can support grouped splitting.
        _task.setdefault("split_group_field", "family")
        _task.setdefault("split_group_scope", SPLIT_SCOPE_BIOLOGICAL)
    if _task.get("split_group_field"):
        if _task.get("split_group_scope") not in BIOLOGICAL_SPLIT_SCOPES:
            raise RuntimeError(
                f"{_task_id}: registered split_group_field requires an approved "
                "biological split_group_scope"
            )
        split_group_namespace = _task.get("split_group_namespace")
        if split_group_namespace is not None and (
            not isinstance(split_group_namespace, str)
            or not split_group_namespace.strip()
        ):
            raise RuntimeError(
                f"{_task_id}: split_group_namespace must be a non-empty string"
            )
    elif _task.get("split_group_namespace") is not None:
        raise RuntimeError(
            f"{_task_id}: split_group_namespace requires split_group_field"
        )
    elif _task_id == "variant/seq":
        # The historical rows do not contain sufficient mutation identity; their pair-like file
        # shape cannot justify even an exact-entity proxy. The task remains quarantined.
        _task["split_group_scope"] = SPLIT_SCOPE_UNAVAILABLE
    elif _task.get("pair_group") or _task["kind"] == "pairs":
        # Exact entity grouping keeps alternate renderings and representation controls together.
        # It is not a scaffold, donor, patient, family, slide, or other biological dependency unit.
        _task["split_group_scope"] = SPLIT_SCOPE_ENTITY_PROXY
    else:
        _task["split_group_scope"] = SPLIT_SCOPE_UNAVAILABLE
    _validate_truth_intervention_contract(_task_id, _task)
    _validate_interventional_split_namespace(_task_id, _task)

# Default benchmark set (the empirical output arm). Computable/reasoning tasks, the weak-signal
# exploratory rung, and quarantined tasks are excluded.
CORE = [_task_id for _task_id, _task in TASKS.items() if _task["status"] == "active"]
EXPLORATORY = [_task_id for _task_id, _task in TASKS.items() if _task["status"] == "exploratory"]
QUARANTINED = [_task_id for _task_id, _task in TASKS.items() if _task["status"] == "quarantined"]

# Backward-compat for eval/routing_arm.py and eval/elicit_confidence.py: the ADMET endpoint clause
# text + orientation (mirrors the admet/* tasks above).
CLAUSES = {
    "herg":         ("blocks the hERG potassium channel (cardiotoxicity risk)", "align"),
    "cyp3a4":       ("inhibits the CYP3A4 enzyme", "align"),
    "cyp2d6":       ("inhibits the CYP2D6 enzyme", "align"),
    "ames":         ("is mutagenic in the Ames test", "oppose"),
    "solubility":   ("is highly soluble in water", "oppose"),
    "permeability": ("is highly permeable across a cell membrane", "oppose"),
}


def _stable_id(namespace, value):
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:20]
    return f"{namespace}:{digest}"


def _registered_split_group_id(task_id, task, value):
    """Return a stable biological group ID, optionally shared across task registries.

    The historical fallback is task-scoped. New external-transfer datasets should set a
    globally meaningful namespace such as ``uniref50:2026_01`` so an overlapping protein
    family cannot acquire different IDs merely because it appears in two task entries.
    """

    namespace = task.get("split_group_namespace")
    if namespace is None:
        namespace = f"{task_id}:split:{task['split_group_scope']}"
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError(
            f"{task_id}: split_group_namespace must be a non-empty string"
        )
    return _stable_id(namespace, value)


def _drop_duplicate_representations(items):
    """Keep one entity per identical input and remove inputs carrying conflicting labels.

    A benchmark row is the observable representation, not the hidden source molecule/sample.
    Allowing the exact same representation on both sides of a split inflates trained heads; assigning
    two labels to the same representation makes the question underdetermined.
    """
    by_rep = {}
    for item in items:
        by_rep.setdefault(item["rep"], []).append(item)
    clean = []
    for rep_items in by_rep.values():
        if len({int(item["label"]) for item in rep_items}) != 1:
            continue
        clean.append(sorted(rep_items, key=lambda item: item["entity_key"])[0])
    return clean


def _attach_registered_row_metadata(task_id, task, row, item):
    """Attach registered biological split and intervention-pair values to one source row."""

    for field_key, item_key, label in (
        ("split_group_field", "split_group_value", "split-group"),
        (
            "intervention_pair_field",
            "intervention_pair_value",
            "intervention-pair",
        ),
    ):
        source_field = task.get(field_key)
        if not source_field:
            continue
        value = row.get(source_field)
        if value in (None, ""):
            raise ValueError(
                f"{task_id}: missing registered {label} field {source_field!r}"
            )
        item[item_key] = str(value)
    return item


class GroundBenchSampler:
    """Stateful, seeded sampler that enforces within-entity representation comparisons.

    All members of a `pair_group` are sampled from the intersection of available entity keys.
    The chosen balanced entity set is cached, so name/anonymous and SMILES/graph/NMR/XYZ tasks
    receive exactly the same examples in exactly the same order. Scrambled conditions are joined
    back to matched examples by entity key rather than sliced independently.
    """

    def __init__(self, seed=0):
        self.seed = int(seed)
        self._pool_cache = {}
        self._selection_cache = {}

    def _entity_key(self, task_id, task, row, row_index):
        field = task.get("entity_field")
        if field == "__row_index__":
            return str(row_index)
        if field:
            return str(row[field])
        if row.get("id") not in (None, ""):
            return str(row["id"])
        representation = row.get("representation", row.get(task.get("col", ""), ""))
        return hashlib.sha256(str(representation).encode()).hexdigest()

    def _load_pairs(self, task_id, task):
        rows = [json.loads(line) for line in open(os.path.join(SIGNAL, task["data"])) if line.strip()]
        matched_rows = [row for row in rows if row.get("condition", "matched") == "matched"]
        source_to_entity = {}
        matched = []
        for i, row in enumerate(matched_rows):
            entity_key = self._entity_key(task_id, task, row, i)
            source_id = row.get("id") or f"snapshot_row:{i}"
            if source_id not in (None, ""):
                source_to_entity[str(source_id)] = entity_key
            matched.append(
                _attach_registered_row_metadata(
                    task_id,
                    task,
                    row,
                    {
                        "entity_key": entity_key,
                        "source_id": source_id,
                        "rep": row["representation"],
                        "label": int(row["label"]),
                    },
                )
            )
        matched = _drop_duplicate_representations(matched)
        kept_keys = {item["entity_key"] for item in matched}
        matched_by_key = {item["entity_key"]: item for item in matched}
        conditions = defaultdict(dict)
        for i, row in enumerate(rows):
            condition = row.get("condition")
            if condition in (None, "", "matched"):
                continue
            source_id = row.get("id") or f"snapshot_row:{i}"
            entity_key = source_to_entity.get(str(source_id))
            if entity_key is None:
                entity_key = self._entity_key(task_id, task, row, i)
            if entity_key in kept_keys:
                condition_item = _attach_registered_row_metadata(
                    task_id,
                    task,
                    row,
                    {
                        "entity_key": entity_key,
                        "source_id": source_id,
                        "rep": row["representation"],
                        "label": int(row["label"]),
                    },
                )
                matched_item = matched_by_key[entity_key]
                for metadata_key, label in (
                    ("split_group_value", "split group"),
                    ("intervention_pair_value", "intervention pair"),
                ):
                    if condition_item.get(metadata_key) != matched_item.get(
                        metadata_key
                    ):
                        raise ValueError(
                            f"{task_id}: condition {condition!r} changes {label} "
                            f"for entity {entity_key!r}"
                        )
                conditions[condition][entity_key] = condition_item
        return matched, dict(conditions)

    def _load_embedding(self, task_id, task):
        if task.get("split_group_field") or task.get("intervention_pair_field"):
            raise ValueError(
                f"{task_id}: NPZ embeddings cannot resolve registered row metadata; "
                "use an explicit row manifest"
            )
        data = np.load(os.path.join(SIGNAL, task["data"]))
        emb, labels = data["emb"], data["y"]
        has_ids = "ids" in data.files
        ids = data["ids"] if has_ids else np.arange(len(labels))
        matched = [{
            "entity_key": str(ids[i]),
            "source_id": str(ids[i]) if has_ids else f"snapshot_row:{i}",
            "rep": " ".join(f"{value:.3f}" for value in emb[i]),
            "label": int(labels[i]),
        } for i in range(len(labels))]
        return _drop_duplicate_representations(matched), {}

    def _load_csv(self, task_id, task):
        rows = list(csv.DictReader(open(os.path.join(SIGNAL, task["data"]))))
        matched = []
        for i, row in enumerate(rows):
            entity_key = self._entity_key(task_id, task, row, i)
            source_id = row.get("id") or f"snapshot_row:{i}"
            item = {
                "entity_key": entity_key,
                "source_id": source_id,
                "rep": "" if task["kind"] == "image" else row[task["col"]],
                "label": int(row["label"]),
            }
            _attach_registered_row_metadata(task_id, task, row, item)
            if task["kind"] == "image":
                item["image"] = os.path.join(ROOT, row[task["col"]])
                item["rep"] = os.path.relpath(item["image"], ROOT)
                item["entity_key"] = row[task["col"]]
            matched.append(item)
        return _drop_duplicate_representations(matched), {}

    def _pool(self, task_id):
        if task_id not in self._pool_cache:
            task = TASKS[task_id]
            if task["kind"] == "pairs":
                pool = self._load_pairs(task_id, task)
            elif task["kind"] == "emb":
                pool = self._load_embedding(task_id, task)
            else:
                pool = self._load_csv(task_id, task)
            self._pool_cache[task_id] = pool
        return self._pool_cache[task_id]

    def _group_members(self, pair_group):
        return [task_id for task_id, task in TASKS.items() if task.get("pair_group") == pair_group
                and task["status"] != "quarantined"]

    def _candidate_items(self, task_id):
        task = TASKS[task_id]
        items, _ = self._pool(task_id)
        pair_group = task.get("pair_group")
        if not pair_group:
            return items
        members = self._group_members(pair_group)
        by_task = {member: {item["entity_key"]: item for item in self._pool(member)[0]}
                   for member in members}
        common = set.intersection(*(set(pool) for pool in by_task.values()))
        candidates = [item for item in items if item["entity_key"] in common]
        for entity_key in common:
            labels = {int(by_task[member][entity_key]["label"]) for member in members}
            if len(labels) != 1:
                raise ValueError(f"inconsistent labels in pair_group={pair_group!r}, entity={entity_key!r}")
        return candidates

    def _selected_keys(self, task_id, n):
        task = TASKS[task_id]
        cache_key = (task.get("pair_group") or task_id, int(n))
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]
        candidates = self._candidate_items(task_id)
        group_members = self._group_members(task["pair_group"]) if task.get("pair_group") else [task_id]
        preferred = set()
        for member in group_members:
            preferred.update(self._pool(member)[1].get("scrambled", {}))
        selection_key = f"{self.seed}\0{cache_key[0]}\0{cache_key[1]}".encode()
        selection_seed = int.from_bytes(hashlib.sha256(selection_key).digest()[:8], "big")
        selection_rng = np.random.default_rng(selection_seed)

        def ordered(label):
            items = sorted((item for item in candidates if int(item["label"]) == label),
                           key=lambda item: item["entity_key"])
            with_perturbation = [item for item in items if item["entity_key"] in preferred]
            without_perturbation = [item for item in items if item["entity_key"] not in preferred]
            selection_rng.shuffle(with_perturbation)
            selection_rng.shuffle(without_perturbation)
            return with_perturbation + without_perturbation

        pos, neg = ordered(1), ordered(0)
        k = min(n // 2, len(pos), len(neg))
        keys = [item["entity_key"] for item in pos[:k] + neg[:k]]
        self._selection_cache[cache_key] = keys
        return keys

    def task_condition_items(self, task_id, n):
        """Return the matched rows and every entity-paired registered input condition."""
        if task_id not in TASKS:
            raise KeyError(f"unknown GroundBench task: {task_id}")
        task = TASKS[task_id]
        if task["status"] == "quarantined":
            raise ValueError(f"{task_id} is quarantined: {task['status_reason']}")
        items, condition_maps = self._pool(task_id)
        by_key = {item["entity_key"]: item for item in items}
        keys = self._selected_keys(task_id, n)

        def public(item, condition):
            entity_namespace = task.get("pair_group", task_id)
            entity_id = _stable_id(entity_namespace, item["entity_key"])
            split_group_scope = task["split_group_scope"]
            if split_group_scope in BIOLOGICAL_SPLIT_SCOPES:
                split_group_id = _registered_split_group_id(
                    task_id,
                    task,
                    item["split_group_value"],
                )
            elif split_group_scope == SPLIT_SCOPE_ENTITY_PROXY:
                split_group_id = entity_id
            elif split_group_scope == SPLIT_SCOPE_UNAVAILABLE:
                split_group_id = None
            else:
                raise ValueError(
                    f"{task_id}: unknown split_group_scope={split_group_scope!r}"
                )
            # Registry validation forbids task-level constants. T5 row identity is always
            # resolved from the registered source field.
            intervention_pair_id = None
            if task.get("intervention_pair_field"):
                intervention_pair_value = item.get("intervention_pair_value")
                if intervention_pair_value in (None, ""):
                    raise ValueError(
                        f"{task_id}: sampled row is missing its registered intervention pair"
                    )
                intervention_pair_id = _stable_id(
                    f"{task['biological_question_id']}:intervention_pair",
                    intervention_pair_value,
                )
            factor_levels = dict(task["factor_levels"])
            factor_levels["input_condition"] = condition
            record = {
                "id": _stable_id(f"{task_id}:{condition}", item["entity_key"]),
                "entity_id": entity_id,
                "source_id": item.get("source_id"),
                "entity_id_scope": task["entity_id_scope"],
                "truth_level_code": task["truth_level_code"],
                "target_source_kind": task["target_source_kind"],
                "truth_level": task["truth_level"],
                "biological_question_id": task["biological_question_id"],
                "task_family_id": task["task_family_id"],
                "split_group_id": split_group_id,
                "split_group_scope": split_group_scope,
                "intervention_pair_id": intervention_pair_id,
                "factor_levels": factor_levels,
                "rep": item["rep"],
                "label": int(item["label"]),
                "condition": condition,
            }
            if "image" in item:
                record["image"] = item["image"]
            return record

        selected = {"matched": [public(by_key[key], "matched") for key in keys]}
        for condition, condition_map in sorted(condition_maps.items()):
            selected[condition] = [
                public(condition_map[key], condition)
                for key in keys
                if key in condition_map
            ]
        return selected

    def task_items(self, task_id, n):
        """Backward-compatible matched plus scrambled view of :meth:`task_condition_items`."""
        conditions = self.task_condition_items(task_id, n)
        return conditions["matched"], conditions.get("scrambled", [])


def task_items(task_id, n, rng):
    """Backward-compatible one-task sampler.

    Internal benchmark entry points use one shared `GroundBenchSampler`, which is required for
    paired cross-task comparisons. This wrapper preserves older one-task analysis scripts.
    """
    sampler = GroundBenchSampler(seed=int(rng.integers(0, np.iinfo(np.int64).max)))
    return sampler.task_items(task_id, n)

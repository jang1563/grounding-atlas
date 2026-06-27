"""WS3 weights PoC: does LoRA finetuning close the hERG EXPRESSION gap in OUTPUT?

Qwen3-8B encodes hERG to AUROC 0.787 (activation probe) but verbalizes it at chance
(output 0.453): the expression gap. P2 predicts this is closable by training the read-out.
This finetunes a LoRA on the hERG yes/no task and measures the model's VERBALIZED output
AUROC before vs after, on a held-out Murcko-scaffold split. The right comparator is the
SAME-SPLIT structural ceiling (Morgan probe / k-NN, see eval/ws3_decision_split.py), NOT the
cross-split activation probe 0.787 (a different metric on a different balanced GroupKFold
set). If post-finetune output rises toward the same-split ceiling, P2 holds at the output
level. Note the scaffold split is near-domain (median test-train Tanimoto ~0.66), so the lift
includes local-SAR generalization, and the shuffle-label control rules out only label-memo.

Verbalized score (both base and finetuned, identical eval): logP(" yes") vs logP(" no")
continuation of the prompt -> P(yes), AUROC vs the label. Manual LoRA loop (no trl).
Runs on a Cayuga GPU. Env: LORA_MODEL, LORA_N (balanced, default 1000), LORA_EPOCHS (3).
No em dashes.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cell-parameterized (env), hERG-SMILES defaults preserved. Other WS3 train-placement cells
# (MS->hERG, variant-seq->pathogenic) set LORA_PAIRS + LORA_PROMPT + LORA_CELL; their jsonl
# carries an explicit "group" field (scaffold for MS, gene for variant) for the cold split.
PAIRS = os.environ.get("LORA_PAIRS", os.path.join(ROOT, "signal", "admet", "herg", "pairs.jsonl"))
MODEL = os.environ.get("LORA_MODEL", "Qwen/Qwen3-8B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment-2 (3-way bridge) parity: when LORA_FOLD is set, Arm C reads the SHARED held-out item list
# (signal/admet/folds/<fold>.json) instead of its own GroupShuffleSplit, and uses the SAME
# {property}-templated prompt skeleton as the bridge (only the soft-prompt injection differs).
FOLD = os.environ.get("LORA_FOLD", "")
ENDPOINT = os.environ.get("LORA_ENDPOINT", "")
MODE = os.environ.get("LORA_MODE", "within")              # within | transfer
TRAIN_EP = ["ames", "cyp3a4", "cyp2d6", "solubility", "permeability"]
QUESTION = {"herg": "block the hERG potassium channel (cardiotoxicity risk)",
            "clearance": "have high metabolic clearance", "ames": "test positive for Ames mutagenicity",
            "cyp3a4": "inhibit CYP3A4", "cyp2d6": "inhibit CYP2D6",
            "solubility": "have high aqueous solubility", "permeability": "have high membrane permeability"}
if ENDPOINT in QUESTION:
    PROMPT = os.environ.get("LORA_PROMPT", f"SMILES: {{rep}}\nDoes this molecule {QUESTION[ENDPOINT]}? "
                            "Answer yes or no.\nAnswer:")
else:
    PROMPT = os.environ.get("LORA_PROMPT", "SMILES: {rep}\nDoes this molecule block the hERG potassium "
                            "channel (cardiotoxicity)? Answer yes or no.\nAnswer:")


def scaffold_of(smi):
    try:   # rdkit imported lazily: cells with an explicit "group" (variant=gene) never call this
        from rdkit import RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")
        return MurckoScaffold.MurckoScaffoldSmiles(smi)
    except Exception:
        return smi


def load(n):
    by = defaultdict(list)
    for line in open(PAIRS):
        r = json.loads(line)
        if r.get("condition", "matched") == "matched":   # condition optional (variant/MS jsonl omit it)
            by[int(r["label"])].append(r)
    rng = np.random.RandomState(42)
    smis, ys, groups = [], [], []
    for lab in (0, 1):
        it = by[lab][:]
        rng.shuffle(it)
        for r in it[:n // 2]:
            rep = r["representation"]
            smis.append(rep); ys.append(lab)
            groups.append(r.get("group") or scaffold_of(rep))   # explicit group (gene), else Murcko scaffold
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(smis, ys, groups))
    return ([smis[i] for i in tr], [ys[i] for i in tr],
            [smis[i] for i in te], [ys[i] for i in te])


def _matched(endpoint):
    rows = []
    for line in open(os.path.join(ROOT, "signal", "admet", endpoint, "pairs.jsonl")):
        r = json.loads(line)
        if r.get("condition") == "matched":
            rows.append((r["representation"], int(r["label"]), r["id"]))
    return rows


def load_fold(n):
    """Experiment-2 parity: test = the held-out endpoint's FROZEN shared test_ids; train = within: the
    same endpoint's non-test rows; transfer: the pooled 5 train-endpoints. Balanced + capped to n."""
    fold = json.load(open(os.path.join(ROOT, "signal", "admet", "folds", f"{FOLD}.json")))
    test_ids = set(fold["held_out"][ENDPOINT]["test_ids"])
    ep_rows = _matched(ENDPOINT)
    te = [(s, y) for s, y, i in ep_rows if i in test_ids]
    if MODE == "within":
        train_rows = [(s, y) for s, y, i in ep_rows if i not in test_ids]
    else:
        train_rows = [(s, y) for ep in TRAIN_EP for s, y, i in _matched(ep)]
    rng = np.random.RandomState(42)
    by = defaultdict(list)
    for s, y in train_rows:
        by[y].append(s)
    k = min(n // 2, len(by[0]), len(by[1]))
    smis, ys = [], []
    for lab in (0, 1):
        rng.shuffle(by[lab])
        smis += by[lab][:k]
        ys += [lab] * k
    return smis, ys, [s for s, _ in te], [y for _, y in te]


def yn_ids(tok):
    # first-token ids for " yes" / " no" (leading space, Qwen BPE)
    return tok(" yes", add_special_tokens=False).input_ids, tok(" no", add_special_tokens=False).input_ids


@torch.no_grad()
def eval_output_auroc(model, tok, smis, ys, yes_ids, no_ids):
    model.eval()
    scores = []
    for smi in smis:
        p = PROMPT.format(rep=smi)
        ids = tok(p, return_tensors="pt").to(DEV)
        logits = model(**ids).logits[0, -1]  # next-token distribution after "Answer:"
        lp = torch.log_softmax(logits.float(), -1)
        sy, sn = lp[yes_ids[0]].item(), lp[no_ids[0]].item()
        scores.append(np.exp(sy) / (np.exp(sy) + np.exp(sn)))
    return round(float(roc_auc_score(ys, scores)), 3)


def build_example(tok, smi, lab, maxlen=256):
    p = PROMPT.format(rep=smi)
    tgt = " yes" if lab == 1 else " no"
    pid = tok(p, add_special_tokens=False).input_ids
    tid = tok(tgt, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (pid + tid)[:maxlen]
    labels = ([-100] * len(pid) + tid)[:maxlen]
    return ids, labels


def main():
    n = int(os.environ.get("LORA_N", "1000"))
    epochs = int(os.environ.get("LORA_EPOCHS", "3"))
    R = int(os.environ.get("LORA_R", "16"))
    shuffle = os.environ.get("LORA_SHUFFLE", "0") == "1"  # negative control: train on shuffled labels
    tr_s, tr_y, te_s, te_y = load_fold(n) if FOLD else load(n)
    if FOLD:
        print(f"PARITY fold={FOLD} endpoint={ENDPOINT} mode={MODE} (shared test_ids, prompt={PROMPT[:40]}...)", flush=True)
    if shuffle:
        tr_y = list(np.random.RandomState(1).permutation(tr_y))
        print("NEGATIVE CONTROL: train labels SHUFFLED (output should stay at chance)", flush=True)
    print(f"train={len(tr_s)} test={len(te_s)} (scaffold split) R={R} epochs={epochs}  device={DEV}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV)
    yes_ids, no_ids = yn_ids(tok)

    base_auc = eval_output_auroc(model, tok, te_s, te_y, yes_ids, no_ids)
    print(f"BASE output AUROC = {base_auc}  (output-arm ref 0.453, activation 0.787, ceiling 0.825)", flush=True)

    from peft import LoraConfig, get_peft_model  # lazy: keeps load_fold/load importable without peft
    model = get_peft_model(model, LoraConfig(
        r=R, lora_alpha=2 * R, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    exs = [build_example(tok, s, y) for s, y in zip(tr_s, tr_y)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    rng = np.random.RandomState(0)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(exs))
        tot = 0.0
        for step, i in enumerate(order):
            ids, labels = exs[i]
            inp = torch.tensor([ids]).to(DEV)
            lab = torch.tensor([labels]).to(DEV)
            out = model(input_ids=inp, labels=lab)
            out.loss.backward()
            if (step + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
            tot += out.loss.item()
        opt.step(); opt.zero_grad()
        print(f"  epoch {ep+1}/{epochs} mean_loss={tot/len(exs):.4f}", flush=True)

    ft_auc = eval_output_auroc(model, tok, te_s, te_y, yes_ids, no_ids)
    print(f"FINETUNED output AUROC = {ft_auc}", flush=True)
    cell = os.environ.get("LORA_CELL", ENDPOINT or "herg")
    if FOLD:
        tag = f"{ENDPOINT}_{MODE}_r{R}" + ("_shuffled" if shuffle else "")
    else:
        tag = f"{cell}_shuffled" if shuffle else f"{cell}_n{n}_r{R}_e{epochs}"
    out = {"tag": tag, "model": MODEL, "n_train": len(tr_s), "n_test": len(te_s),
           "epochs": epochs, "lora_r": R, "shuffled_control": shuffle,
           "base_output_auroc": base_auc, "finetuned_output_auroc": ft_auc,
           "lift": round(ft_auc - base_auc, 3),
           "ref_output_arm": 0.453, "ref_activation_probe": 0.787, "ref_structure_ceiling": 0.825}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    p = os.path.join(ROOT, "results", "ws3_lora.json")
    merged = {}
    if os.path.isfile(p):
        d = json.load(open(p))
        for r in (d if isinstance(d, list) else [d]):
            merged[r.get("tag", "n1000_r16_e3")] = r
    merged[tag] = out
    json.dump([merged[k] for k in sorted(merged)], open(p, "w"), indent=2)
    print(f"\nP2 test: base {base_auc} -> finetuned {ft_auc} (lift {out['lift']}); "
          f"closes toward activation 0.787 / ceiling 0.825 if lift is large.", flush=True)


if __name__ == "__main__":
    main()

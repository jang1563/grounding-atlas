"""Validate a GroundBench submission before opening a PR.

Usage:  python eval/validate_submission.py results/benchmark/<model> [--allow-partial]

Checks that a results/benchmark/<model>/ directory is a well-formed, comparable, complete submission:
valid scorecard / manifest / raw.jsonl, the full CORE task set (unless --allow-partial), the current
prompt version, per-task fields present and in range, and raw<->scorecard consistency. Exit 0 = pass,
1 = fail, 2 = usage. See SUBMITTING.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_tasks import CORE, TASKS  # noqa: E402
from run_grounding_eval import PROMPT_VERSION  # noqa: E402

REQUIRED_TASK_FIELDS = ["n", "output_auroc", "output_auroc_ci", "ece", "aurc", "sel_acc_50",
                        "web_exposure", "orientation"]
REQUIRED_MANIFEST = ["model", "prompt_version", "decode", "seed", "data_commit", "tasks"]


def _load_json(path, errs):
    if not os.path.exists(path):
        errs.append(f"missing {os.path.basename(path)}")
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        errs.append(f"invalid JSON {os.path.basename(path)}: {e}")
        return None


def validate(d, allow_partial=False):
    errs, warns = [], []
    sc = _load_json(os.path.join(d, "scorecard.json"), errs)
    man = _load_json(os.path.join(d, "manifest.json"), errs)
    rawp = os.path.join(d, "raw.jsonl")
    if not os.path.exists(rawp):
        errs.append("missing raw.jsonl")
    if sc is None or man is None:
        return errs, warns

    for k in REQUIRED_MANIFEST:
        if k not in man:
            errs.append(f"manifest missing '{k}'")
    if man.get("prompt_version") != PROMPT_VERSION:
        errs.append(f"prompt_version {man.get('prompt_version')!r} != current {PROMPT_VERSION!r} "
                    "(not comparable; re-run on the current harness)")
    if man.get("dry_run"):
        errs.append("manifest dry_run=true (synthetic results, not a real submission)")

    missing = [t for t in CORE if t not in sc]
    if missing:
        msg = f"missing {len(missing)} CORE task(s): {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        (warns if allow_partial else errs).append(msg)
    extra = [t for t in sc if t not in TASKS]
    if extra:
        warns.append(f"{len(extra)} task(s) not in the registry: {', '.join(extra[:5])}")

    for t, rec in sc.items():
        miss = [f for f in REQUIRED_TASK_FIELDS if f not in rec]
        if miss:
            errs.append(f"{t}: missing field(s) {', '.join(miss)}")
            continue
        a = rec["output_auroc"]
        if not (isinstance(a, (int, float)) and 0.0 <= a <= 1.0):
            errs.append(f"{t}: output_auroc {a} out of [0,1]")
        ci = rec["output_auroc_ci"]
        if not (isinstance(ci, list) and len(ci) == 2):
            errs.append(f"{t}: output_auroc_ci must be a [lo, hi] pair, got {ci!r}")
        if rec["web_exposure"] not in ("rich", "zero", "mixed"):
            errs.append(f"{t}: web_exposure {rec['web_exposure']!r} not in rich/zero/mixed")

    if os.path.exists(rawp):
        raw_tasks = set()
        try:
            for line in open(rawp):
                line = line.strip()
                if line:
                    raw_tasks.add(json.loads(line).get("task"))
        except Exception as e:
            errs.append(f"raw.jsonl parse error: {e}")
        sc_only = set(sc) - raw_tasks
        if sc_only:
            warns.append(f"{len(sc_only)} task(s) in scorecard but absent from raw.jsonl: "
                         f"{', '.join(sorted(sc_only)[:5])}")
    return errs, warns


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pos:
        print("usage: python eval/validate_submission.py results/benchmark/<model> [--allow-partial]")
        sys.exit(2)
    d = pos[0]
    errs, warns = validate(d, allow_partial="--allow-partial" in sys.argv)
    for w in warns:
        print(f"WARN  {w}")
    for e in errs:
        print(f"FAIL  {e}")
    if errs:
        print(f"\n{len(errs)} error(s) in {d}. Not a valid submission.")
        sys.exit(1)
    print(f"\nOK: valid submission ({d}). {len(warns)} warning(s).")
    sys.exit(0)


if __name__ == "__main__":
    main()

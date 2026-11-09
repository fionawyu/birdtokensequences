#!/usr/bin/env python3
"""Classification accuracy of all methods on the five tasks (Tables 2, 3, 10).

Four runs of stratified five-fold cross-validation; every method is given the
same splits. Writes:

    <out>/accuracy_long.csv     one row per (task, method): mean, SD, max, run accuracies
    <out>/accuracy_table.csv    tasks x methods, "mean ± SD (max)"
    <out>/accuracy_table.tex
    <out>/predictions.csv       one row per (task, method, run, recording)
    <out>/confusion/<task>__<method>.csv   confusion counts summed over runs

    python scripts/evaluate.py --manifest data/manifest_snr.csv --tokens outputs/tokens \
        --embeddings outputs/perch_embeddings.npz --out outputs/eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq import classify, methods, pipeline  # noqa: E402
from tokenseq.config import load_config, load_tasks  # noqa: E402
from tokenseq.data import read_manifest, task_subset  # noqa: E402


def fmt(s: dict) -> str:
    return f"{s['mean']:.1f} ± {s['sd']:.1f} ({s['max']:.1f})"


def to_latex(table: pd.DataFrame) -> str:
    """Tasks x methods table as a LaTeX tabular; the best mean per task is bold."""
    def mean(cell: str) -> float:
        return float(cell.split(" ")[0])

    lines = [r"\begin{tabular}{l" + "c" * table.shape[1] + "}", r"\toprule",
             "Task & " + " & ".join(table.columns) + r" \\", r"\midrule"]
    for task, row in table.iterrows():
        best = max(mean(c) for c in row)
        cells = [(r"\textbf{%s}" % c if mean(c) == best else c).replace("±", r"$\pm$") for c in row]
        lines.append(f"{task} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokens", required=True, help="directory written by extract_tokens.py")
    ap.add_argument("--embeddings", default=None, help="npz from compute_perch_embeddings.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks-config", default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task keys")
    ap.add_argument("--methods", nargs="*", default=None, choices=methods.METHODS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks_config)
    task_keys = args.tasks or list(tasks)
    method_keys = args.methods or [m for m in methods.METHODS
                                   if args.embeddings or not m.startswith("perch")]
    ev = cfg["evaluation"]
    df = read_manifest(args.manifest)
    emb_all = dict(np.load(args.embeddings)) if args.embeddings else None

    out = Path(args.out)
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    rows, pred_rows = [], []
    for tk in task_keys:
        classes = tasks[tk]["classes"]
        sub, y = task_subset(df, classes)
        records = [pipeline.load(rid, args.tokens) for rid in sub["recording_id"]]
        E = np.stack([emb_all[rid] for rid in sub["recording_id"]]) if emb_all else None
        for mk in method_keys:
            method = methods.build(mk, records, len(classes), cfg, E)
            res = classify.cross_validate(method, y, ev["run_seeds"], ev["n_folds"])
            s = res.summary()
            rows.append({"task": tk, "method": mk, "n": len(y), "n_classes": len(classes),
                         "mean": s["mean"], "sd": s["sd"], "max": s["max"],
                         "runs": ";".join(f"{a:.4f}" for a in s["runs"])})
            for seed, p in zip(ev["run_seeds"], res.predictions):
                pred_rows += [{"task": tk, "method": mk, "run_seed": seed, "recording_id": rid,
                               "true": classes[t], "pred": classes[q]}
                              for rid, t, q in zip(sub["recording_id"], y, p)]
            M = classify.confusion(y, res.predictions, len(classes))
            pd.DataFrame(M, index=classes, columns=classes).to_csv(out / "confusion" / f"{tk}__{mk}.csv")
            print(f"{tasks[tk]['title']:<28} {methods.LABELS[mk]:<40} {fmt(s)}", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(out / "accuracy_long.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)

    wide = long.assign(cell=long.apply(fmt, axis=1)).pivot(index="task", columns="method", values="cell")
    wide = wide.loc[task_keys, method_keys]
    wide.index = [tasks[k]["title"] for k in wide.index]
    wide.columns = [methods.LABELS[m] for m in wide.columns]
    wide.to_csv(out / "accuracy_table.csv")
    (out / "accuracy_table.tex").write_text(to_latex(wide))
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()

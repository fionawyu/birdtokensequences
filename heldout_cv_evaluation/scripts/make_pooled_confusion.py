#!/usr/bin/env python3
"""Pooled held-out confusion matrices for the two hard datasets.

Reads ONLY the saved held-out predictions -- no tokenization, no formula
fitting, no re-clustering, no re-splitting. Every row it consumes is one
held-out test recording whose predicted class came from a cluster-to-class
mapping built on that fold's training recordings.

For each dataset, the predictions from all 20 train/test splits (4 repetitions
x 5 folds) are pooled into one confusion matrix. Each recording contributes
exactly one prediction per repetition, hence four in total; check 10 below
verifies that rather than assuming it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
CONF = OUT / "pooled_confusion"

SOURCE = OUT / "main_datasets_nested" / "heldout_predictions.csv"   # the selected protocol
FALLBACK = OUT / "main_datasets" / "heldout_predictions.csv"

DATASETS = ["AllVar", "YNCCDialect"]
PAPER_NAME = {"AllVar": "Species + BCCH variants",
              "YNCCDialect": "yellow cardinal four dialects"}
# Class order exactly as in manuscript Table tab:class_sample_sizes.
CLASS_ORDER = {
    "AllVar": ["Chickadee", "Tseet", "feebee", "TEWA", "WTSP", "YWAR"],
    "YNCCDialect": ["Co", "RN", "SL", "U"],
}
CLASS_TEX = {
    "Chickadee": r"\textit{chick-a-dee} call", "Tseet": r"\textit{tseet}",
    "feebee": r"\textit{fee-bee} song", "TEWA": "TEWA", "WTSP": "WTSP", "YWAR": "YWAR",
    "Co": "Co", "RN": "RN", "SL": "SL", "U": "U",
}
# Manuscript n from tab:class_sample_sizes, for the reconciliation report.
MANUSCRIPT_N = {"AllVar": 129, "YNCCDialect": 54}
METHOD_ORDER = ["Token + k-medoids", "Shuffled + k-medoids", "Trajectory-DTW + k-medoids",
                "Formula attributes + k-means", "Perch v2 + k-medoids", "Perch v2 + k-means"]


def normalise_class(name: str) -> str:
    """The saved labels derive class names from the dataset id, so the yellow
    cardinal dialects arrive as 'CoCurate', 'RNCurate', ... Strip that suffix so
    they match the manuscript's class names."""
    n = str(name)
    return n[:-len("Curate")] if n.endswith("Curate") else n


def verify_coverage(g: pd.DataFrame, dataset: str, method: str, lines: list[str]) -> bool:
    """Check 10: every recording held out exactly once per repetition."""
    ok = True
    recs = sorted(g["recording_id"].unique())
    for rep, gr in g.groupby("repetition", sort=True):
        counts = gr["recording_id"].value_counts()
        dup = sorted(counts[counts > 1].index)
        missing = sorted(set(recs) - set(counts.index))
        if dup or missing:
            ok = False
            lines.append(f"    rep{rep}: FAIL -- {len(dup)} duplicated, {len(missing)} missing")
            for r in dup[:5]:
                lines.append(f"        duplicated x{counts[r]}: {r}")
            for r in missing[:5]:
                lines.append(f"        missing: {r}")
        else:
            lines.append(f"    rep{rep}: {len(counts)} recordings, each held out exactly once  [OK]")
    n_reps = g["repetition"].nunique()
    per_rec = g["recording_id"].value_counts()
    if not (per_rec == n_reps).all():
        ok = False
        bad = per_rec[per_rec != n_reps]
        lines.append(f"    FAIL -- {len(bad)} recordings do not have exactly "
                     f"{n_reps} predictions (e.g. {bad.index[0]}: {bad.iloc[0]})")
    else:
        lines.append(f"    all {len(per_rec)} recordings have exactly {n_reps} "
                     f"predictions ({len(g)} total)  [OK]")
    return ok


def confusion(g: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    unmapped = np.zeros(len(classes), dtype=int)
    for t, p in zip(g["true_class_short"].map(normalise_class),
                    g["predicted_class_short"].map(normalise_class)):
        if t not in idx:
            continue
        if p in idx:
            cm[idx[t], idx[p]] += 1
        else:
            unmapped[idx[t]] += 1
    return cm, unmapped


def row_pct(cm: np.ndarray) -> np.ndarray:
    tot = cm.sum(axis=1, keepdims=True)
    return 100.0 * np.divide(cm, np.where(tot == 0, 1, tot))


def latex_table(dataset: str, method: str, classes: list[str], cm: np.ndarray,
                pct: np.ndarray, stats: dict) -> str:
    n_cls = len(classes)
    head = [
        r"\begin{table*}[htbp]", r"\centering",
        rf"\caption[Pooled held-out confusion matrix, {PAPER_NAME[dataset]}]{{",
        rf"Pooled held-out confusion matrix for the {PAPER_NAME[dataset]} dataset",
        rf"({method}). Predictions from all $4 \times 5 = 20$ training/test",
        r"divisions are pooled; each recording contributes one held-out prediction",
        r"per repetition and therefore four in total. Rows are true classes,",
        r"columns predicted classes; cells give the count with the row percentage",
        r"in parentheses. Pooled accuracy "
        rf"{stats['pooled_accuracy']:.2f}\% ({stats['n_correct']}/{stats['n_total']});",
        rf"accuracy across the four repetitions "
        rf"{stats['mean']:.2f} $\pm$ {stats['sd']:.2f}\% (best {stats['best']:.2f}\%).",
        r"}",
        rf"\label{{tab:confusion_{dataset.lower().replace('-', '')}}}",
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n_cls + r"c}", r"\toprule",
        r"\textbf{True class}" + "".join(rf" & \textbf{{{CLASS_TEX[c]}}}" for c in classes)
        + r" & \textbf{Recall} \\", r"\midrule",
    ]
    body = []
    for i, c in enumerate(classes):
        cells = []
        for j in range(n_cls):
            txt = rf"{cm[i, j]} ({pct[i, j]:.1f}\%)"
            if i == j:
                txt = rf"\textbf{{{cm[i, j]} ({pct[i, j]:.1f}\%)}}"
            cells.append(txt)
        body.append(CLASS_TEX[c] + " & " + " & ".join(cells)
                    + rf" & {pct[i, i]:.1f}\% \\")
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(head + body + tail) + "\n"


def main() -> None:
    src = SOURCE if SOURCE.exists() else FALLBACK
    preds = pd.read_csv(src)
    CONF.mkdir(parents=True, exist_ok=True)
    lines = [f"POOLED HELD-OUT CONFUSION MATRICES", "=" * 70,
             f"source: {src.relative_to(OUT.parent)}", ""]
    pooled_all = []

    for dataset in DATASETS:
        gd = preds[preds["dataset"] == dataset]
        if gd.empty:
            lines.append(f"[{dataset}] no saved predictions -- SKIPPED")
            continue
        classes = CLASS_ORDER[dataset]
        n_pool = gd["recording_id"].nunique()
        n_reps = gd["repetition"].nunique()
        lines.append("=" * 70)
        lines.append(f"{PAPER_NAME[dataset]}  [{dataset}]")
        lines.append(f"  evaluation pool n = {n_pool} recordings, "
                     f"{n_reps} repetitions -> expected {n_pool * n_reps} predictions")
        lines.append(f"  manuscript Table tab:class_sample_sizes reports n = "
                     f"{MANUSCRIPT_N[dataset]} "
                     f"({'MATCHES' if MANUSCRIPT_N[dataset] == n_pool else 'DOES NOT MATCH'})")
        lines.append("")

        methods = [m for m in METHOD_ORDER if m in set(gd["method"])]
        for method in methods:
            g = gd[gd["method"] == method]
            lines.append(f"-- {method} --")
            lines.append("  coverage check (every recording held out once per repetition):")
            verify_coverage(g, dataset, method, lines)

            cm, unmapped = confusion(g, classes)
            pct = row_pct(cm)
            n_total = int(cm.sum() + unmapped.sum())
            if n_total == 0:
                observed = sorted(set(g["true_class_short"].map(normalise_class)))
                lines.append(f"  ERROR: no prediction matched the expected class order "
                             f"{classes}; observed classes are {observed}. Skipped.")
                continue
            n_correct = int(np.trace(cm))
            rep_acc = [100.0 * gr["correct"].mean()
                       for _, gr in g.groupby("repetition", sort=True)]
            stats = {"pooled_accuracy": 100.0 * n_correct / n_total,
                     "n_correct": n_correct, "n_total": n_total,
                     "mean": float(np.mean(rep_acc)), "sd": float(np.std(rep_acc)),
                     "best": float(np.max(rep_acc))}

            width = max(14, max(len(c) for c in classes) + 2)
            lines.append("")
            lines.append("  " + "true \\ predicted".ljust(width)
                         + "".join(c.ljust(width) for c in classes) + "recall")
            for i, c in enumerate(classes):
                row = "  " + c.ljust(width)
                for j in range(len(classes)):
                    row += f"{cm[i, j]} ({pct[i, j]:.1f}%)".ljust(width)
                row += f"{pct[i, i]:.1f}%"
                lines.append(row)
            if unmapped.sum():
                lines.append(f"  [unmapped cluster: {int(unmapped.sum())} predictions]")
            lines.append("")
            lines.append(f"  pooled accuracy      {stats['pooled_accuracy']:.2f}%  "
                         f"({n_correct}/{n_total})")
            lines.append(f"  across repetitions   mean {stats['mean']:.2f}%  "
                         f"SD {stats['sd']:.2f}  best {stats['best']:.2f}%")
            lines.append(f"  per repetition       "
                         + ", ".join(f"{a:.2f}%" for a in rep_acc))
            lines.append("")

            tag = f"{dataset}_{method.split(' + ')[0].replace(' ', '').replace('-', '')}"
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                CONF / f"counts_{tag}.csv")
            pd.DataFrame(np.round(pct, 2), index=classes, columns=classes).to_csv(
                CONF / f"rowpct_{tag}.csv")
            (CONF / f"table_{tag}.tex").write_text(
                latex_table(dataset, method, classes, cm, pct, stats))

            gg = g.copy()
            gg.insert(0, "paper_dataset", PAPER_NAME[dataset])
            pooled_all.append(gg)

            # figure
            fig, ax = plt.subplots(figsize=(1.15 * len(classes) + 2.6,
                                            1.05 * len(classes) + 2.2))
            im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=9)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center",
                            va="center", fontsize=8,
                            color="white" if pct[i, j] > 55 else "#1a1a1a")
            ax.set_xlabel("Predicted class", fontsize=10)
            ax.set_ylabel("True class", fontsize=10)
            ax.set_title(f"{PAPER_NAME[dataset]} -- {method}\n"
                         f"pooled held-out: {stats['pooled_accuracy']:.2f}% "
                         f"({n_correct}/{n_total}); "
                         f"{stats['mean']:.1f} $\\pm$ {stats['sd']:.1f}% across 4 reps",
                         fontsize=10, pad=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="row percentage (%)")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(CONF / f"confusion_{tag}.{ext}", dpi=240, bbox_inches="tight")
            plt.close(fig)

    if pooled_all:
        pd.concat(pooled_all, ignore_index=True).to_csv(
            CONF / "pooled_heldout_predictions.csv", index=False)

    report = "\n".join(lines)
    (CONF / "pooled_confusion_report.txt").write_text(report + "\n")
    print(report)
    print(f"\nWrote {CONF}")
    print("DONE_MARKER_POOLED_CONFUSION")


if __name__ == "__main__":
    main()

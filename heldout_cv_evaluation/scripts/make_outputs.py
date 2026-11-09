#!/usr/bin/env python3
"""Regenerate the manuscript deliverables from the held-out results:

  * the main accuracy table  (CSV + LaTeX, same shape as tab:quantitative)
  * the main accuracy figure (grouped bars, mean over the 4 repetitions with
    the SD as the error bar and the per-repetition points overlaid)
  * held-out confusion matrices, per dataset x method, pooled over the 4
    repetitions, plus a figure per dataset
  * the controlled-noise table (CSV + LaTeX, same shape as tab:controlled_snr)
  * a side-by-side comparison against the published whole-dataset numbers

Writes only under heldout_cv_evaluation/outputs/.
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
sys.path.insert(0, str(HERE))
import data_sources as ds                                       # noqa: E402

OUT = HERE.parent / "outputs"
MAIN = OUT / "main_datasets"
NOISE = OUT / "noise_conditions"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
CONF = OUT / "confusion_matrices"

DATASET_ORDER = ["SegMix", "BCCH-var", "Species", "AllVar", "YNCCDialect"]
METHOD_ORDER = ["Token + k-medoids", "Shuffled + k-medoids", "Trajectory-DTW + k-medoids",
                "Formula attributes + k-means", "Perch v2 + k-medoids", "Perch v2 + k-means"]
METHOD_TEX = {
    "Token + k-medoids": r"Token + \(k\)-medoids",
    "Shuffled + k-medoids": r"Shuffled + \(k\)-medoids",
    "Trajectory-DTW + k-medoids": r"Trajectory-DTW + \(k\)-medoids",
    "Formula attributes + k-means": r"Formula attr. + \(k\)-means",
    "Perch v2 + k-medoids": r"Perch v2 + \(k\)-medoids",
    "Perch v2 + k-means": r"Perch v2 + \(k\)-means",
}
METHOD_SHORT = {
    "Token + k-medoids": "Token",
    "Shuffled + k-medoids": "Shuffled",
    "Trajectory-DTW + k-medoids": "Traj-DTW",
    "Formula attributes + k-means": "Formula attr.",
    "Perch v2 + k-medoids": "Perch v2 (kmed)",
    "Perch v2 + k-means": "Perch v2 (kmeans)",
}
PALETTE = ["#2c6fb2", "#9aa6b2", "#4f9d69", "#d98c3f", "#8d6bb0", "#c0504d"]
DIFFICULTY_ORDER = ["Easy", "Medium", "Hard"]


# ---------------------------------------------------------------- table
def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dsname in DATASET_ORDER:
        g = summary[summary["dataset"] == dsname]
        if g.empty:
            continue
        row = {
            "difficulty": ds.SETTING_DIFFICULTY[dsname],
            "dataset": dsname,
            "paper_label": ds.SETTING_PAPER_LABEL[dsname],
            "n_classes": int(g["n_classes"].iloc[0]),
            "chance_%": round(float(g["chance_accuracy"].iloc[0]), 2),
        }
        for m in METHOD_ORDER:
            gm = g[g["method"] == m]
            if gm.empty:
                row[f"{m} n"] = np.nan
                for suf in ("mean", "sd", "median", "best"):
                    row[f"{m} {suf}"] = np.nan
                continue
            r = gm.iloc[0]
            row[f"{m} n"] = int(r["n"])
            row[f"{m} mean"] = round(float(r["mean_accuracy"]), 2)
            row[f"{m} sd"] = round(float(r["sd_accuracy"]), 2)
            row[f"{m} median"] = round(float(r["median_accuracy"]), 2)
            row[f"{m} best"] = round(float(r["best_accuracy"]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main_table_latex(summary: pd.DataFrame) -> str:
    head = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption[Held-out classification accuracy across methods]{",
        r"Held-out classification accuracy (\%) across methods. Clusters were built on the",
        r"training recordings of each fold only, mapped to classes with the Hungarian",
        r"algorithm on the training labels only, and used to predict the disjoint held-out",
        r"recordings. Values are mean $\pm$ SD over four repetitions of stratified",
        r"5-fold cross-validation, with the median and the best repetition in parentheses",
        r"as (median / best). Bold marks the highest mean for each evaluation dataset.",
        r"}",
        r"\label{tab:quantitative_heldout}",
        r"",
        r"{\fontsize{8.5pt}{10pt}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X" +
        (r">{\centering\arraybackslash}p{0.135\textwidth}" * len(METHOD_ORDER)) + r"@{}}",
        r"\toprule",
        r"\textbf{Evaluation dataset}",
    ]
    for m in METHOD_ORDER:
        head.append(rf"& \textbf{{{METHOD_TEX[m]}}}")
    head += [r"\\", r"\midrule", ""]

    body = []
    last_diff = None
    for dsname in DATASET_ORDER:
        g = summary[summary["dataset"] == dsname]
        if g.empty:
            continue
        diff = ds.SETTING_DIFFICULTY[dsname]
        if diff != last_diff:
            if last_diff is not None:
                body.append(r"\addlinespace[4pt]")
            body.append(rf"\multicolumn{{{len(METHOD_ORDER)+1}}}{{@{{}}l}}{{\textbf{{{diff}}}}} \\")
            last_diff = diff
        means = {m: (g[g["method"] == m]["mean_accuracy"].iloc[0]
                     if not g[g["method"] == m].empty else np.nan)
                 for m in METHOD_ORDER}
        best_mean = np.nanmax(list(means.values()))
        body.append(ds.SETTING_PAPER_LABEL[dsname].replace("fee-bee", r"\textit{fee-bee}"))
        for m in METHOD_ORDER:
            gm = g[g["method"] == m]
            if gm.empty:
                body += ["&", r"\makecell{--}"]
                continue
            r = gm.iloc[0]
            # mark cells scored on a different pool than the rest of the row
            mark = ""
            row_n = int(g[g["method"] == "Token + k-medoids"]["n"].iloc[0])
            if int(r["n"]) != row_n:
                mark = r"\textsuperscript{a}"
            head_ = rf"{r['mean_accuracy']:.2f} $\pm$ {r['sd_accuracy']:.2f}{mark}"
            if np.isclose(r["mean_accuracy"], best_mean):
                head_ = rf"\textbf{{{r['mean_accuracy']:.2f} $\pm$ {r['sd_accuracy']:.2f}}}{mark}"
            cell = (head_ + rf"\\{{\footnotesize ({r['median_accuracy']:.2f} / "
                    rf"{r['best_accuracy']:.2f})}}")
            body += ["&", rf"\makecell{{{cell}}}"]
        body += [r"\\", ""]

    tail = [
        r"\bottomrule",
        r"\end{tabularx}",
        r"",
        r"\vspace{3pt}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize",
        r"\textit{Note.} Each cell is mean $\pm$ SD (median / best) of the four",
        r"repetition-level held-out accuracies; a repetition-level accuracy pools the",
        r"five disjoint test folds, so every recording contributes once per repetition.",
        r"Training and test recordings never overlap within a fold, and no held-out",
        r"label is used for clustering, model fitting, preprocessing statistics, or the",
        r"cluster-to-class mapping.",
        r"\textsuperscript{a}For the yellow cardinal dialects the cached Perch v2",
        r"embeddings cover a different curated clip set for the RN and SL dialects than",
        r"the recordings the symbolic pipeline tokenizes, so Perch v2 is evaluated on its",
        r"own $n=48$ pool under the identical protocol; the other four methods share one",
        r"$n=47$ pool and identical splits.",
        r"\end{minipage}",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(head + body + tail) + "\n"


# ---------------------------------------------------------------- figure
def make_main_figure(summary: pd.DataFrame, reps: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in set(summary["dataset"])]
    n_m = len(METHOD_ORDER)
    x = np.arange(len(datasets), dtype=float)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    for mi, m in enumerate(METHOD_ORDER):
        offs = x - 0.4 + width * (mi + 0.5)
        means, sds = [], []
        for d in datasets:
            g = summary[(summary["dataset"] == d) & (summary["method"] == m)]
            means.append(float(g["mean_accuracy"].iloc[0]) if not g.empty else np.nan)
            sds.append(float(g["sd_accuracy"].iloc[0]) if not g.empty else np.nan)
        ax.bar(offs, means, width * 0.92, yerr=sds, capsize=2.5, label=METHOD_SHORT[m],
               color=PALETTE[mi], edgecolor="white", linewidth=0.6,
               error_kw=dict(elinewidth=0.9, ecolor="#333333"))
        for di, d in enumerate(datasets):
            pts = reps[(reps["dataset"] == d) & (reps["method"] == m)]["accuracy"].to_numpy()
            if pts.size:
                jitter = (np.linspace(-0.28, 0.28, pts.size)) * width
                ax.scatter(np.full(pts.size, offs[di]) + jitter, pts, s=7,
                           color="#1a1a1a", alpha=0.55, zorder=3, linewidths=0)

    for di, d in enumerate(datasets):
        chance = float(summary[summary["dataset"] == d]["chance_accuracy"].iloc[0])
        ax.plot([x[di] - 0.42, x[di] + 0.42], [chance, chance], ls=(0, (4, 3)),
                color="#b03030", lw=1.2, zorder=4)
    ax.plot([], [], ls=(0, (4, 3)), color="#b03030", lw=1.2, label="chance")

    labels = []
    for d in datasets:
        n_sym = int(summary[(summary["dataset"] == d) &
                            (summary["method"] == "Token + k-medoids")]["n"].iloc[0])
        labels.append(f"{ds.SETTING_PAPER_LABEL[d]}\n({ds.SETTING_DIFFICULTY[d]}, "
                      f"n={n_sym}, k={int(summary[summary['dataset']==d]['n_classes'].iloc[0])})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Held-out classification accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Held-out classification accuracy: 4 x stratified 5-fold cross-validation\n"
                 "(bars = mean over repetitions, error bars = SD, dots = individual repetitions)",
                 fontsize=11.5, pad=12)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"heldout_classification_accuracy.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- confusion matrices
def make_confusions(preds: pd.DataFrame) -> None:
    CONF.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for dsname in DATASET_ORDER:
        g = preds[preds["dataset"] == dsname]
        if g.empty:
            continue
        methods = [m for m in METHOD_ORDER if m in set(g["method"])]
        classes = sorted(set(g["true_class_short"]))
        fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3.5))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, methods):
            gm = g[g["method"] == m]
            cls = sorted(set(gm["true_class_short"]) | set(gm["predicted_class_short"]) - {"UNMAPPED"})
            cls = [c for c in cls if c != "UNMAPPED"]
            idx = {c: i for i, c in enumerate(cls)}
            cm = np.zeros((len(cls), len(cls)), dtype=int)
            for t, p in zip(gm["true_class_short"], gm["predicted_class_short"]):
                if p in idx:
                    cm[idx[t], idx[p]] += 1
            for i, t in enumerate(cls):
                for j, p in enumerate(cls):
                    long_rows.append({"dataset": dsname, "method": m, "true_class": t,
                                      "predicted_class": p, "count": int(cm[i, j])})
            row_tot = cm.sum(axis=1, keepdims=True)
            norm = np.divide(cm, np.where(row_tot == 0, 1, row_tot))
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(cls)))
            ax.set_xticklabels(cls, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(cls)))
            ax.set_yticklabels(cls, fontsize=7.5)
            for i in range(len(cls)):
                for j in range(len(cls)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.55 else "#222222")
            acc = 100.0 * gm["correct"].mean()
            ax.set_title(f"{METHOD_SHORT[m]}\n{acc:.1f}%", fontsize=9)
            ax.set_xlabel("predicted", fontsize=8)
        axes[0].set_ylabel("true", fontsize=8)
        fig.suptitle(f"Held-out confusion matrices -- {ds.SETTING_PAPER_LABEL[dsname]} "
                     f"(pooled over 4 repetitions)", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(CONF / f"confusion_{dsname}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(long_rows).to_csv(CONF / "heldout_confusion_matrices.csv", index=False)


# ---------------------------------------------------------------- noise
def build_noise_outputs() -> None:
    path = NOISE / "heldout_accuracy_summary.csv"
    if not path.exists():
        print("noise summary not present yet -- skipping the noise table")
        return
    s = pd.read_csv(path)
    order = [c for c in ["baseline", "snr_12db", "snr_9db", "snr_6db", "snr_3db", "snr_0db"]
             if c in set(s["condition"])]
    wide = []
    for c in order:
        g = s[s["condition"] == c]
        row = {"condition": c, "manuscript_label": g["manuscript_label"].iloc[0],
               "mean_achieved_db": g["mean_achieved_db"].iloc[0],
               "n": int(g["n"].iloc[0]), "n_classes": int(g["n_classes"].iloc[0])}
        for m in METHOD_ORDER:
            gm = g[g["method"] == m]
            if gm.empty:
                continue
            r = gm.iloc[0]
            row[f"{m} mean"] = round(float(r["mean_accuracy"]), 2)
            row[f"{m} sd"] = round(float(r["sd_accuracy"]), 2)
            row[f"{m} median"] = round(float(r["median_accuracy"]), 2)
            row[f"{m} best"] = round(float(r["best_accuracy"]), 2)
        wide.append(row)
    wide_df = pd.DataFrame(wide)
    wide_df.to_csv(TABLES / "heldout_noise_conditions_table.csv", index=False)

    lines = [
        r"\begin{table*}[htbp]", r"\centering",
        r"\caption{Held-out classification accuracy across controlled SNR conditions.",
        r"Mean $\pm$ SD over four repetitions of stratified 5-fold cross-validation;",
        r"clusters, cluster-to-class mappings and preprocessing statistics come from the",
        r"training recordings of each fold only.}",
        r"\label{tab:controlled_snr_heldout}", r"",
        r"{\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l" + "c" * len(METHOD_ORDER) + r"}", r"\toprule",
        r"\textbf{Condition}",
    ]
    head_tex = {
        "Token + k-medoids": r"& \shortstack{\textbf{Token}\\$+\,k$-medoids}",
        "Shuffled + k-medoids": r"& \shortstack{\textbf{Shuffled}\\$+\,k$-medoids}",
        "Trajectory-DTW + k-medoids": r"& \shortstack{\textbf{Trajectory-DTW}\\$+\,k$-medoids}",
        "Formula attributes + k-means": r"& \shortstack{\textbf{Formula attributes}\\$+\,k$-means}",
        "Perch v2 + k-medoids": r"& \shortstack{\textbf{Perch v2}\\$+\,k$-medoids}",
        "Perch v2 + k-means": r"& \shortstack{\textbf{Perch v2}\\$+\,k$-means}",
    }
    for m in METHOD_ORDER:
        lines.append(head_tex[m])
    lines += [r"\\", r"\midrule", ""]
    for _, r in wide_df.iterrows():
        means = [float(r[f"{m} mean"]) for m in METHOD_ORDER if f"{m} mean" in wide_df.columns]
        best = max(means) if means else float("nan")
        lines.append(str(r["manuscript_label"]).replace("<=", r"$\le$ ").replace("-", "--"))
        for m in METHOD_ORDER:
            if f"{m} mean" not in wide_df.columns or pd.isna(r[f"{m} mean"]):
                lines.append("& --")
                continue
            mu, sd = float(r[f"{m} mean"]), float(r[f"{m} sd"])
            if np.isclose(mu, best):
                lines.append(rf"& $\mathbf{{{mu:.1f} \pm {sd:.1f}\%}}$")
            else:
                lines.append(rf"& ${mu:.1f} \pm {sd:.1f}\%$")
        lines += [r"\\", ""]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (TABLES / "heldout_noise_conditions_table.tex").write_text("\n".join(lines) + "\n")

    # noise figure
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    xs = np.arange(len(wide_df))
    for mi, m in enumerate(METHOD_ORDER):
        col = f"{m} mean"
        if col not in wide_df:
            continue
        ax.errorbar(xs, wide_df[col], yerr=wide_df[f"{m} sd"], marker="o", ms=5,
                    lw=1.6, capsize=3, color=PALETTE[mi], label=METHOD_SHORT[m])
    chance = 100.0 / float(wide_df["n_classes"].iloc[0])
    ax.axhline(chance, ls=(0, (4, 3)), color="#b03030", lw=1.2, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels(wide_df["manuscript_label"], fontsize=9)
    ax.set_ylabel("Held-out classification accuracy (%)", fontsize=11)
    ax.set_xlabel("Controlled noise condition (decreasing SNR $\\rightarrow$)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("Held-out accuracy under controlled Gaussian-noise degradation", fontsize=11.5)
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"heldout_noise_conditions.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- published comparison
# The published YNCCDialect cells for the symbolic methods were computed on the
# full 54-recording token pool; the held-out pool is the 47 recordings for which
# a contour also exists (so all four symbolic methods share one pool). The
# difference column for those cells therefore mixes the pool change with the
# held-out change and is not a clean like-for-like. contour_dtw_baseline
# reproduced Token on the same 47 recordings at 83.51%, which is the fair
# whole-dataset reference for that cell.
NOTES = {
    ("YNCCDialect", "Token + k-medoids"):
        "published n=54; held-out pool n=47. Like-for-like whole-dataset Token on "
        "the same 47 recordings is 83.51% (contour_dtw_baseline/comparison_table.csv), "
        "so the held-out drop against a matched pool is -18.6, not -21.7.",
    ("YNCCDialect", "Shuffled + k-medoids"):
        "published n=54 AND a different control (label permutation, not token-order "
        "shuffling) -- not comparable.",
    ("YNCCDialect", "Formula attributes + k-means"): "published n=54; held-out pool n=47.",
}
for _d in ("SegMix", "BCCH-var", "Species", "AllVar"):
    NOTES[(_d, "Shuffled + k-medoids")] = (
        "different control: the published row permutes class labels, this one shuffles "
        "token order within each recording -- not comparable.")


def build_comparison(summary: pd.DataFrame) -> None:
    pub_path = (ds.REPO_ROOT / "paper_results" / "chapter1" / "method_comparison"
                / "classification_accuracy_summary.csv")
    dtw_path = ds.REPO_ROOT / "contour_dtw_baseline" / "table2_contour_dtw_with_best.csv"
    pub_map = {
        "Token + k-medoids": "Token sequences + edit distance",
        "Shuffled + k-medoids": "Shuffled token sequences + edit distance",
        "Formula attributes + k-means": "Formula features + k-means",
        "Perch v2 + k-medoids": "Perch 2 + k-medoids",
        "Perch v2 + k-means": "Perch 2 + k-means",
    }
    pub = pd.read_csv(pub_path) if pub_path.exists() else pd.DataFrame()
    dtw = pd.read_csv(dtw_path) if dtw_path.exists() else pd.DataFrame()

    rows = []
    for r in summary.itertuples():
        published_mean = published_sd = np.nan
        if r.method == "Trajectory-DTW + k-medoids" and not dtw.empty:
            g = dtw[dtw["dataset"] == r.dataset]
            if not g.empty:
                published_mean, published_sd = float(g["mean"].iloc[0]), float(g["sd"].iloc[0])
        elif not pub.empty and r.method in pub_map:
            g = pub[(pub["dataset"] == r.dataset) & (pub["method"] == pub_map[r.method])]
            if not g.empty:
                published_mean, published_sd = float(g["mean"].iloc[0]), float(g["sd"].iloc[0])
        rows.append({
            "dataset": r.dataset, "method": r.method, "n_heldout_pool": r.n,
            "heldout_mean": round(r.mean_accuracy, 2), "heldout_sd": round(r.sd_accuracy, 2),
            "heldout_median": round(r.median_accuracy, 2), "heldout_best": round(r.best_accuracy, 2),
            "published_whole_dataset_mean": (round(published_mean, 2)
                                             if np.isfinite(published_mean) else np.nan),
            "published_whole_dataset_sd": (round(published_sd, 2)
                                           if np.isfinite(published_sd) else np.nan),
            "difference_heldout_minus_published": (round(r.mean_accuracy - published_mean, 2)
                                                   if np.isfinite(published_mean) else np.nan),
            "note": NOTES.get((r.dataset, r.method), ""),
        })
    pd.DataFrame(rows).to_csv(TABLES / "heldout_vs_published_comparison.csv", index=False)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(MAIN / "heldout_accuracy_summary.csv")
    reps = pd.read_csv(MAIN / "repetition_accuracy.csv")
    preds = pd.read_csv(MAIN / "heldout_predictions.csv")

    build_main_table(summary).to_csv(TABLES / "heldout_accuracy_table.csv", index=False)
    (TABLES / "heldout_accuracy_table.tex").write_text(main_table_latex(summary))
    make_main_figure(summary, reps)
    make_confusions(preds)
    build_comparison(summary)
    build_noise_outputs()

    print("wrote:")
    for p in sorted(list(TABLES.glob("*")) + list(FIGS.glob("*")) + list(CONF.glob("*"))):
        print("  ", p.relative_to(OUT.parent))
    print("DONE_MARKER_MAKE_OUTPUTS")


if __name__ == "__main__":
    main()

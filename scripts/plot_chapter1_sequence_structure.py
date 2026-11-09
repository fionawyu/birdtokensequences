#!/usr/bin/env python3
"""Create the Chapter 1 sequence-structure figure.

Panel a shows row-normalized note-to-note transition probabilities after
treating silence as a separator. Panel b compares note-only observed transition
entropy with shuffled-token controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "diagnosis" / "chapter1_pitch_diagnostic"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_results" / "chapter1" / "sequence_structure"
JOURNAL_DPI = 600
MM_TO_IN = 1 / 25.4

TOKEN_ORDER = ["FLAT", "UP", "DOWN", "WIGGLE"]
TOKEN_ORDER_WITH_SILENCE = ["FLAT", "UP", "DOWN", "WIGGLE", "SILENCE"]
TOKEN_LABELS = {
    "FLAT": "flat",
    "UP": "up",
    "DOWN": "down",
    "WIGGLE": "wiggle",
    "SILENCE": "silence",
}
SILENCE_TOKENS = {"ZERO", "SILENCE"}

DIALECT_DATASETS = [
    "1017_plots47_CoCurate_1017",
    "1017_plots47_UCurate_1017",
    "1017_plots47_RNCurate_1017",
    "1017_plots47_SLCurate_1017",
]

DISPLAY_NAMES = {
    "1017_plots47_Chickadee_1017": "BCCH chickadee",
    "1017_plots47_CoCurate_1017": "YNCC Co",
    "1017_plots47_RNCurate_1017": "YNCC RN",
    "1017_plots47_SLCurate_1017": "YNCC SL",
    "1017_plots47_TEWA_1017": "TEWA",
    "1017_plots47_Tseet_1017": "BCCH tseet",
    "1017_plots47_UCurate_1017": "YNCC U",
    "1017_plots47_WTSP_1017": "WTSP",
    "1017_plots47_YRWA_1017": "YRWA",
    "1017_plots47_YWAR_1017": "YWAR",
    "1017_plots47_feebee_1017": "BCCH feebee",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#222222"
GREY = "#9a9a9a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Chapter 1 sequence-structure figure.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="figure3_sequence_structure")
    parser.add_argument(
        "--sequences-csv",
        type=Path,
        default=None,
        help="defaults to <input-dir>/evaluation_inputs/original_token_sequences.csv",
    )
    parser.add_argument("--n-shuffle", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--heatmap-datasets",
        default=",".join(DIALECT_DATASETS),
        help="comma-separated dataset ids to pool for the transition heatmap",
    )
    parser.add_argument(
        "--include-silence",
        action="store_true",
        help="keep ZERO/SILENCE as SILENCE tokens instead of treating silence as a separator",
    )
    parser.add_argument(
        "--species-transition",
        action="store_true",
        help="plot one transition heatmap per dataset/species instead of the two-panel summary",
    )
    parser.add_argument(
        "--species-datasets",
        default=None,
        help="comma-separated dataset ids for --species-transition; defaults to every dataset in the input",
    )
    return parser.parse_args()


def set_journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.dpi": JOURNAL_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def clean_name(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset.replace("1017_plots47_", "").replace("_1017", ""))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=DARK,
    )


def canonical_token(token: str) -> str:
    upper = token.upper()
    if upper in SILENCE_TOKENS:
        return "SILENCE"
    return upper


def parse_sequence(value: object, include_silence: bool = False) -> list[str]:
    if pd.isna(value):
        return []

    tokens = [canonical_token(token) for token in str(value).split()]
    if not include_silence:
        tokens = [token for token in tokens if token not in SILENCE_TOKENS]
    return tokens


def parse_note_sequence(value: object) -> list[str]:
    return parse_sequence(value, include_silence=False)


def transition_counts_from_sequences(sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for tokens in sequences:
        counts.update(zip(tokens[:-1], tokens[1:]))
    return counts


def entropy_rate_from_transition_counts(counts: Counter[tuple[str, str]]) -> float:
    total_bigrams = sum(counts.values())
    if total_bigrams == 0:
        return float("nan")

    outgoing: defaultdict[str, int] = defaultdict(int)
    for (prev_token, _), count in counts.items():
        outgoing[prev_token] += count

    entropy = 0.0
    for (prev_token, _), count in counts.items():
        p_joint = count / total_bigrams
        p_next_given_prev = count / outgoing[prev_token]
        entropy -= p_joint * np.log2(p_next_given_prev)
    return float(entropy)


def compute_sequence_metrics(
    sequences: pd.DataFrame, n_shuffle: int, random_seed: int, include_silence: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    entropy_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []

    for dataset, group in sequences.groupby("dataset", dropna=False):
        parsed_sequences = [
            parse_sequence(value, include_silence=include_silence)
            for value in group["token_sequence"]
        ]
        valid_sequences = [tokens for tokens in parsed_sequences if len(tokens) >= 2]
        n_sequences = len(valid_sequences)
        n_notes = int(sum(len(tokens) for tokens in valid_sequences))

        if n_sequences == 0:
            entropy_rows.append(
                {
                    "dataset": dataset,
                    "H_obs": np.nan,
                    "H_shuffle_mean": np.nan,
                    "H_shuffle_std": np.nan,
                    "entropy_gap": np.nan,
                    "n_sequences": 0,
                    "n_notes": n_notes,
                    "n_transitions": 0,
                }
            )
            continue

        obs_counts = transition_counts_from_sequences(valid_sequences)
        h_obs = entropy_rate_from_transition_counts(obs_counts)
        n_transitions = int(sum(obs_counts.values()))

        for (prev_token, next_token), count in sorted(obs_counts.items()):
            transition_rows.append(
                {
                    "dataset": dataset,
                    "prev_token": prev_token,
                    "next_token": next_token,
                    "count": count,
                }
            )

        shuffled_entropies: list[float] = []
        for _ in range(n_shuffle):
            shuffled_sequences = []
            for tokens in valid_sequences:
                shuffled = np.array(tokens, dtype=object).copy()
                rng.shuffle(shuffled)
                shuffled_sequences.append(shuffled.tolist())
            shuffled_counts = transition_counts_from_sequences(shuffled_sequences)
            shuffled_entropies.append(entropy_rate_from_transition_counts(shuffled_counts))

        h_shuffle_mean = float(np.nanmean(shuffled_entropies)) if shuffled_entropies else np.nan
        h_shuffle_std = float(np.nanstd(shuffled_entropies)) if shuffled_entropies else np.nan
        entropy_rows.append(
            {
                "dataset": dataset,
                "H_obs": h_obs,
                "H_shuffle_mean": h_shuffle_mean,
                "H_shuffle_std": h_shuffle_std,
                "entropy_gap": h_shuffle_mean - h_obs,
                "n_sequences": n_sequences,
                "n_notes": n_notes,
                "n_transitions": n_transitions,
            }
        )

    entropy = pd.DataFrame(entropy_rows).sort_values("dataset")
    transitions = pd.DataFrame(
        transition_rows,
        columns=["dataset", "prev_token", "next_token", "count"],
    )
    return entropy, transitions


def compute_note_only_sequence_metrics(
    sequences: pd.DataFrame, n_shuffle: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return compute_sequence_metrics(
        sequences, n_shuffle, random_seed, include_silence=False
    )


def build_transition_probability(
    transitions: pd.DataFrame, heatmap_datasets: list[str], token_order: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_order = token_order or TOKEN_ORDER
    pooled = transitions[transitions["dataset"].isin(heatmap_datasets)].copy()
    counts = (
        pooled.groupby(["prev_token", "next_token"], as_index=False)["count"]
        .sum()
        .pivot(index="prev_token", columns="next_token", values="count")
        .reindex(index=token_order, columns=token_order)
        .fillna(0)
    )
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    probabilities = counts.div(row_totals, axis=0).fillna(0)
    return counts, probabilities


def plot_sequence_structure(
    transitions: pd.DataFrame,
    entropy: pd.DataFrame,
    output_stem: Path,
    heatmap_datasets: list[str],
    token_order: list[str] | None = None,
    panel_a_title: str = "YNCC dialect note-to-note transitions",
    panel_b_title: str = "Note-only syntax remains ordered after removing silence",
    transition_label: str = "Note-to-note transition probability",
    x_label: str = "Next note token",
    y_label: str = "Previous note token",
) -> None:
    token_order = token_order or TOKEN_ORDER
    counts, probabilities = build_transition_probability(
        transitions, heatmap_datasets, token_order=token_order
    )

    entropy = entropy.copy()
    entropy["display_name"] = entropy["dataset"].map(clean_name)
    entropy["is_dialect"] = entropy["dataset"].isin(DIALECT_DATASETS)
    entropy = entropy.sort_values("entropy_gap", ascending=True)
    entropy_for_plot = entropy.dropna(subset=["H_obs", "H_shuffle_mean"]).copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM_TO_IN, 88 * MM_TO_IN),
        gridspec_kw={"width_ratios": [1.0, 1.42], "wspace": 0.42},
    )

    ax = axes[0]
    image = ax.imshow(probabilities.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(token_order)))
    ax.set_xticklabels([TOKEN_LABELS[token] for token in token_order], rotation=35, ha="right")
    ax.set_yticks(range(len(token_order)))
    ax.set_yticklabels([TOKEN_LABELS[token] for token in token_order])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(panel_a_title, loc="left")
    add_panel_label(ax, "a")

    for i, prev_token in enumerate(token_order):
        for j, next_token in enumerate(token_order):
            prob = float(probabilities.loc[prev_token, next_token])
            count = int(counts.loc[prev_token, next_token])
            label = f"{prob:.2f}"
            if count == 0:
                color = "#bdbdbd"
            else:
                color = "white" if prob >= 0.55 else DARK
            ax.text(j, i, label, ha="center", va="center", fontsize=5.6, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(transition_label, fontsize=6)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=5.5)

    ax = axes[1]
    y = np.arange(len(entropy_for_plot))
    colors = np.where(entropy_for_plot["is_dialect"], GREEN, BLUE)

    for idx, row in enumerate(entropy_for_plot.itertuples(index=False)):
        ax.plot(
            [row.H_obs, row.H_shuffle_mean],
            [idx, idx],
            color=GREY,
            linewidth=0.75,
            zorder=1,
        )

    ax.scatter(
        entropy_for_plot["H_obs"],
        y,
        s=20,
        color=ORANGE,
        edgecolor=DARK,
        linewidth=0.35,
        label="Observed",
        zorder=3,
    )
    ax.scatter(
        entropy_for_plot["H_shuffle_mean"],
        y,
        s=20,
        color=colors,
        edgecolor=DARK,
        linewidth=0.35,
        label="Shuffled control",
        zorder=3,
    )

    if "H_shuffle_std" in entropy.columns:
        for idx, row in enumerate(entropy_for_plot.itertuples(index=False)):
            ax.errorbar(
                row.H_shuffle_mean,
                idx,
                xerr=row.H_shuffle_std,
                fmt="none",
                ecolor=GREEN if row.is_dialect else BLUE,
                elinewidth=0.6,
                capsize=1.8,
                capthick=0.6,
                alpha=0.75,
                zorder=2,
            )

    ax.axvline(0, color=DARK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(entropy_for_plot["display_name"])
    ax.set_xlabel("Transition entropy")
    ax.set_title(panel_b_title, loc="left")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    add_panel_label(ax, "b")

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="lower right", frameon=False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg", "tif"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=JOURNAL_DPI)
    plt.close(fig)

    counts.to_csv(output_stem.with_name(f"{output_stem.name}_transition_counts.csv"))
    probabilities.to_csv(output_stem.with_name(f"{output_stem.name}_transition_probabilities.csv"))
    entropy[
        [
            "dataset",
            "display_name",
            "H_obs",
            "H_shuffle_mean",
            "H_shuffle_std",
            "entropy_gap",
            "n_sequences",
            "n_notes",
            "n_transitions",
        ]
    ].to_csv(output_stem.with_name(f"{output_stem.name}_entropy_summary.csv"), index=False)


def plot_species_transition_grid(
    transitions: pd.DataFrame,
    output_stem: Path,
    species_datasets: list[str],
    token_order: list[str],
    include_silence: bool,
) -> None:
    n_datasets = len(species_datasets)
    if n_datasets == 0:
        raise ValueError("No datasets available for species transition grid.")

    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(183 * MM_TO_IN, max(70, 43 * n_rows) * MM_TO_IN),
        squeeze=False,
        gridspec_kw={"wspace": 0.28, "hspace": 0.55},
    )

    image = None
    probability_tables: list[pd.DataFrame] = []
    count_tables: list[pd.DataFrame] = []
    for ax, dataset in zip(axes.ravel(), species_datasets):
        counts, probabilities = build_transition_probability(
            transitions, [dataset], token_order=token_order
        )
        image = ax.imshow(probabilities.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(clean_name(dataset), loc="left", fontsize=7)
        ax.set_xticks(range(len(token_order)))
        ax.set_xticklabels(
            [TOKEN_LABELS[token] for token in token_order],
            rotation=35,
            ha="right",
            fontsize=5.3,
        )
        ax.set_yticks(range(len(token_order)))
        ax.set_yticklabels([TOKEN_LABELS[token] for token in token_order], fontsize=5.3)
        ax.tick_params(length=1.8, width=0.45)

        for i, prev_token in enumerate(token_order):
            for j, next_token in enumerate(token_order):
                prob = float(probabilities.loc[prev_token, next_token])
                count = int(counts.loc[prev_token, next_token])
                color = "#bdbdbd" if count == 0 else ("white" if prob >= 0.55 else DARK)
                ax.text(j, i, f"{prob:.2f}", ha="center", va="center", fontsize=4.4, color=color)

        probability_table = probabilities.copy()
        probability_table.insert(0, "prev_token", probability_table.index)
        probability_table.insert(0, "dataset", dataset)
        probability_tables.append(probability_table.reset_index(drop=True))

        count_table = counts.copy()
        count_table.insert(0, "prev_token", count_table.index)
        count_table.insert(0, "dataset", dataset)
        count_tables.append(count_table.reset_index(drop=True))

    for ax in axes.ravel()[n_datasets:]:
        ax.axis("off")

    title = "Species transition probabilities"
    if include_silence:
        title += " including silence"
    else:
        title += " after removing silence"
    fig.suptitle(title, x=0.03, y=0.995, ha="left", fontsize=8.5, fontweight="bold", color=DARK)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
        cbar.set_label("Token-to-token transition probability", fontsize=6)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=5.5)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg", "tif"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=JOURNAL_DPI)
    plt.close(fig)

    pd.concat(probability_tables, ignore_index=True).to_csv(
        output_stem.with_name(f"{output_stem.name}_transition_probabilities.csv"),
        index=False,
    )
    pd.concat(count_tables, ignore_index=True).to_csv(
        output_stem.with_name(f"{output_stem.name}_transition_counts.csv"),
        index=False,
    )


def main() -> None:
    args = parse_args()
    set_journal_style()
    heatmap_datasets = [dataset.strip() for dataset in args.heatmap_datasets.split(",") if dataset.strip()]
    sequences_csv = args.sequences_csv or args.input_dir / "evaluation_inputs" / "original_token_sequences.csv"
    sequences = pd.read_csv(sequences_csv)
    entropy, transitions = compute_sequence_metrics(
        sequences,
        args.n_shuffle,
        args.random_seed,
        include_silence=args.include_silence,
    )
    output_stem = args.output_dir / args.output_stem
    if args.species_transition:
        if args.species_datasets:
            species_datasets = [
                dataset.strip() for dataset in args.species_datasets.split(",") if dataset.strip()
            ]
        else:
            species_datasets = sorted(sequences["dataset"].dropna().astype(str).unique(), key=clean_name)
        token_order = TOKEN_ORDER_WITH_SILENCE if args.include_silence else TOKEN_ORDER
        plot_species_transition_grid(
            transitions,
            output_stem,
            species_datasets,
            token_order=token_order,
            include_silence=args.include_silence,
        )
        print(f"Species transition datasets: {', '.join(species_datasets)}")
        print(f"Wrote {output_stem.with_suffix('.png')}")
        print(f"Wrote {output_stem.with_suffix('.pdf')}")
        return

    if args.include_silence:
        plot_sequence_structure(
            transitions,
            entropy,
            output_stem,
            heatmap_datasets,
            token_order=TOKEN_ORDER_WITH_SILENCE,
            panel_a_title="YNCC dialect transitions including silence",
            panel_b_title="Syntax with silence tokens retained",
            transition_label="Token-to-token transition probability",
            x_label="Next token",
            y_label="Previous token",
        )
    else:
        plot_sequence_structure(transitions, entropy, output_stem, heatmap_datasets)
    print(f"Heatmap datasets: {', '.join(heatmap_datasets)}")
    sequence_label = "Silence-retained" if args.include_silence else "Note-only"
    print(f"{sequence_label} sequences: {sequences_csv}")
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

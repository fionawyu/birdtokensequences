"""Build manifest.csv for classiifier.py (Perch 2.0 baseline) from the same
six evaluation settings used by the sequence/attribute method
(combined_model/common.py: DATASETS), so results are directly comparable to
results/table2.csv.

One row per recording (song_id) per setting it belongs to: setting,path,label
where label is the per-setting class name (mirrors what the existing method
clusters against).

USAGE
    python build_perch_manifest.py
"""

from pathlib import Path

import pandas as pd

from combined_model.common import DATASETS, REPO_ROOT, SEGMENTS_CSV

CLASS_LABELS = {
    "1017_plots47_WTSP_1017": "WTSP",
    "1017_plots47_feebee_1017": "Feebee",
    "1017_plots47_YWAR_1017": "YWAR",
    "1017_plots47_YRWA_1017": "YRWA",
    "1017_plots47_Chickadee_1017": "Chickadee",
    "1017_plots47_Tseet_1017": "Tseet",
    "1017_plots47_TEWA_1017": "TEWA",
    "1017_plots47_CoCurate_1017": "Co",
    "1017_plots47_UCurate_1017": "U",
    "1017_plots47_RNCurate_1017": "RN",
    "1017_plots47_SLCurate_1017": "SL",
}


# YNCCDialect recordings only exist in this repo as precomputed .mat
# spectrograms, not raw audio, so a raw-audio model like Perch can't embed
# them. Excluded until the original wav files are located.
SKIP_SETTINGS = {"YNCCDialect"}


def main():
    df = pd.read_csv(SEGMENTS_CSV, usecols=["dataset", "song_id", "file_path"])
    songs = df.drop_duplicates(["dataset", "song_id"])[["dataset", "song_id", "file_path"]]

    rows = []
    missing = []
    for setting, dataset_names in DATASETS.items():
        if setting in SKIP_SETTINGS:
            continue
        sub = songs[songs["dataset"].isin(dataset_names)]
        for _, r in sub.iterrows():
            p = REPO_ROOT / r["file_path"]
            if not p.exists():
                missing.append(str(p))
                continue
            rows.append(
                {
                    "setting": setting,
                    "path": str(p),
                    "label": CLASS_LABELS[r["dataset"]],
                }
            )

    out = pd.DataFrame(rows).sort_values(["setting", "label", "path"])
    out.to_csv(REPO_ROOT / "manifest.csv", index=False)

    print(f"[manifest] wrote {len(out)} rows to manifest.csv")
    print(out.groupby("setting")["label"].value_counts())
    if missing:
        print(f"\n[warn] {len(missing)} file(s) referenced in segments.csv were not found, e.g.:")
        for m in missing[:5]:
            print(" ", m)


if __name__ == "__main__":
    main()

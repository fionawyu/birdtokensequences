#!/usr/bin/env python3
"""Sensitivity of Token + k-medoids accuracy to the token-mapping thresholds
(Appendix C.3, Table 9). Each threshold is varied one at a time, the others
kept at their defaults; tokens are recomputed from the fitted formulas.

    python scripts/mapping_sensitivity.py --manifest data/manifest_snr.csv \
        --tokens outputs/tokens --out outputs/mapping_sensitivity
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenseq import classify, methods, pipeline  # noqa: E402
from tokenseq.config import load_config, load_tasks, parse_time_points  # noqa: E402
from tokenseq.data import read_manifest, task_subset  # noqa: E402

GRID = {
    "slope_sign_fraction": [0.80, 0.85, 0.90, 0.95],
    "change_fraction": [0.70, 0.75, 0.80, 0.85, 0.90],
    "flat_range": [0.00005, 0.0001, 0.0002, 0.0005, 0.001],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tasks-config", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    base["tokens"]["time_points"] = parse_time_points(base["tokens"]["time_points"])
    tasks, ev = load_tasks(args.tasks_config), base["evaluation"]
    df = read_manifest(args.manifest)

    rows = []
    for tk, task in tasks.items():
        sub, y = task_subset(df, task["classes"])
        stored = [pipeline.load(rid, args.tokens) for rid in sub["recording_id"]]
        for param, values in GRID.items():
            for v in values:
                cfg = copy.deepcopy(base)
                cfg["tokens"][param] = v
                records = [pipeline.retokenize(r, cfg) for r in stored]
                res = classify.cross_validate(methods.token_method(records, len(task["classes"])),
                                              y, ev["run_seeds"], ev["n_folds"])
                rows.append({"task": tk, "parameter": param, "value": v,
                             "default": base["tokens"][param], **res.summary()})
                print(tk, param, v, f"{rows[-1]['mean']:.1f}", flush=True)

    long = pd.DataFrame(rows).drop(columns="runs")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "sensitivity_long.csv", index=False)

    # largest absolute change in mean accuracy from the default, per task and parameter
    cells = []
    for (task, param), g in long.groupby(["task", "parameter"], sort=False):
        ref = g.loc[np.isclose(g["value"], g["default"]), "mean"].iloc[0]
        cells.append({"task": task, "parameter": param,
                      "max_abs_change": float((g["mean"] - ref).abs().max())})
    table = pd.DataFrame(cells).pivot(index="task", columns="parameter", values="max_abs_change")
    table.to_csv(out / "sensitivity_table.csv")
    print(table.round(1))


if __name__ == "__main__":
    main()

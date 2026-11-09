#!/usr/bin/env python3
"""Build vocabulary-diagnosis input CSVs from decomp YAML files."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


TOKEN_FEATURES = ("ZERO", "FLAT", "WIGGLE", "UP", "DOWN")
FEATURE_COLUMNS = [f"feature_{idx}" for idx in range(9)]
SEQUENCE_COLUMNS = [
    "dataset",
    "song_id",
    "label",
    "region",
    "mode",
    "file_path",
    "token_sequence",
]
SEGMENT_COLUMNS = [
    "dataset",
    "song_id",
    "segment_id",
    "token",
    "label",
    "region",
    "mode",
    "file_path",
    "start",
    "end",
    "duration",
    "subformula",
    "formula_length",
    *FEATURE_COLUMNS,
]


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build segments.csv and sequences.csv for vocabulary sufficiency "
            "diagnosis from *_decomp.yml files."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode-filter", default="silent")
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        warn(f"Skipping {path}: could not parse YAML ({exc}).")
        return None
    except OSError as exc:
        warn(f"Skipping {path}: could not read file ({exc}).")
        return None

    if not isinstance(data, dict):
        warn(f"Skipping {path}: YAML root is not a mapping.")
        return None
    return data


def infer_mode_from_path(path: Path) -> str:
    parts = path.parts
    if "labels" in parts:
        labels_idx = parts.index("labels")
        if labels_idx + 1 < len(parts):
            return parts[labels_idx + 1]
    return ""


def song_id_from_paths(yaml_path: Path, file_path: object) -> str:
    if isinstance(file_path, str) and file_path.strip():
        return Path(file_path).stem

    stem = yaml_path.stem
    if stem.endswith("_decomp"):
        return stem[: -len("_decomp")]
    return stem


def as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_boundary(boundary: object, yaml_path: Path, segment_id: int) -> tuple[float, float] | None:
    if not isinstance(boundary, (list, tuple)) or len(boundary) < 2:
        warn(f"Skipping segment {segment_id} in {yaml_path}: invalid segment boundary.")
        return None

    try:
        start = float(boundary[0])
        end = float(boundary[1])
    except (TypeError, ValueError):
        warn(f"Skipping segment {segment_id} in {yaml_path}: non-numeric segment boundary.")
        return None

    return start, end


def formula_for_segment(freq_subfmls: list[Any], segment_id: int) -> str:
    if segment_id >= len(freq_subfmls):
        return ""
    value = freq_subfmls[segment_id]
    if value is None:
        return ""
    return str(value)


def feature_vector(token: str, start: float, end: float, formula_length: int) -> list[float]:
    duration = end - start
    upper_token = token.upper()
    indicator_values = [1.0 if upper_token == name else 0.0 for name in TOKEN_FEATURES]
    return [
        duration,
        start,
        end,
        float(formula_length),
        *indicator_values,
    ]


def process_yaml_file(
    yaml_path: Path,
    dataset_name: str | None,
    mode_filter: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    data = read_yaml(yaml_path)
    if data is None:
        return None, []

    mode = str(data.get("mode") or infer_mode_from_path(yaml_path))
    if mode_filter and mode != mode_filter:
        return None, []

    freq_labels = as_list(data.get("freq_labels"))
    segments_freq = as_list(data.get("segments_freq"))
    if not freq_labels or not segments_freq:
        warn(f"Skipping {yaml_path}: missing freq_labels or segments_freq.")
        return None, []

    usable_len = min(len(freq_labels), len(segments_freq))
    if len(freq_labels) != len(segments_freq):
        warn(
            f"{yaml_path}: freq_labels length {len(freq_labels)} does not match "
            f"segments_freq length {len(segments_freq)}; using first {usable_len} entries."
        )

    if usable_len == 0:
        warn(f"Skipping {yaml_path}: no usable labels/segments.")
        return None, []

    species = str(data.get("species") or "unknown")
    dataset = dataset_name or species
    label = species
    region = str(data.get("region") or "")
    file_path = str(data.get("file_path") or "")
    song_id = song_id_from_paths(yaml_path, file_path)
    tokens = [str(token) for token in freq_labels[:usable_len]]
    freq_subfmls = as_list(data.get("freq_subfmls"))

    sequence_row = {
        "dataset": dataset,
        "song_id": song_id,
        "label": label,
        "region": region,
        "mode": mode,
        "file_path": file_path,
        "token_sequence": " ".join(tokens),
    }

    segment_rows: list[dict[str, Any]] = []
    for segment_id, (token, boundary) in enumerate(
        zip(tokens, segments_freq[:usable_len], strict=False)
    ):
        parsed_boundary = parse_boundary(boundary, yaml_path, segment_id)
        if parsed_boundary is None:
            continue

        start, end = parsed_boundary
        duration = end - start
        subformula = formula_for_segment(freq_subfmls, segment_id)
        formula_length = len(subformula)
        features = feature_vector(token, start, end, formula_length)

        row = {
            "dataset": dataset,
            "song_id": song_id,
            "segment_id": segment_id,
            "token": token,
            "label": label,
            "region": region,
            "mode": mode,
            "file_path": file_path,
            "start": start,
            "end": end,
            "duration": duration,
            "subformula": subformula,
            "formula_length": formula_length,
        }
        row.update({column: value for column, value in zip(FEATURE_COLUMNS, features)})
        segment_rows.append(row)

    if not segment_rows:
        warn(f"Skipping sequence for {yaml_path}: no valid segment rows.")
        return None, []

    return sequence_row, segment_rows


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"{title}:")
    if not counter:
        print("  <none>")
        return
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"ERROR: --root does not exist: {args.root}")
    if not args.root.is_dir():
        raise SystemExit(f"ERROR: --root is not a directory: {args.root}")

    yaml_paths = sorted(args.root.rglob("*_decomp.yml"))
    sequence_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    token_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    yaml_files_processed = 0

    for yaml_path in yaml_paths:
        sequence_row, rows = process_yaml_file(
            yaml_path=yaml_path,
            dataset_name=args.dataset_name,
            mode_filter=args.mode_filter,
        )
        if sequence_row is None:
            continue

        yaml_files_processed += 1
        sequence_rows.append(sequence_row)
        segment_rows.extend(rows)
        token_counts.update(row["token"] for row in rows)
        species_counts.update([sequence_row["label"]])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = args.out_dir / "sequences.csv"
    segments_path = args.out_dir / "segments.csv"

    sequences_df = pd.DataFrame(sequence_rows, columns=SEQUENCE_COLUMNS)
    segments_df = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    sequences_df.to_csv(sequences_path, index=False)
    segments_df.to_csv(segments_path, index=False)

    print(f"Found YAML files: {len(yaml_paths)}")
    print(f"YAML files processed: {yaml_files_processed}")
    print(f"Sequences written: {len(sequences_df)}")
    print(f"Segments written: {len(segments_df)}")
    print(f"Wrote {sequences_path}")
    print(f"Wrote {segments_path}")
    print_counter("Token counts", token_counts)
    print_counter("Species counts", species_counts)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build diagnosis CSVs with real spectrogram patches from MATLAB files."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from PIL import Image

try:
    from skimage.transform import resize as skimage_resize
except Exception:  # pragma: no cover - fallback is exercised only without skimage
    skimage_resize = None


BASE_COLUMNS = [
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
    "mat_path",
    "mat_variable",
    "crop_x0",
    "crop_x1",
]
SEQUENCE_COLUMNS = [
    "dataset",
    "song_id",
    "label",
    "region",
    "mode",
    "file_path",
    "token_sequence",
]
PREFERRED_VARIABLE_HINTS = ("spectrogram", "denoised_spec", "spec", "S", "data")


@dataclass(frozen=True)
class MatCandidate:
    path: Path
    path_lower: str
    name_lower: str


@dataclass
class SpectrogramSelection:
    spec: np.ndarray
    variable_name: str
    original_shape: tuple[int, ...]


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sequences.csv and MAT-spectrogram-feature segments.csv."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode-filter", default="silent")
    parser.add_argument("--mat-root", type=Path, default=None)
    parser.add_argument("--image-height", default=32, type=int)
    parser.add_argument("--image-width", default=32, type=int)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--transpose-mat", action="store_true")
    parser.add_argument(
        "--normalize-patch",
        choices=("minmax", "zscore", "none"),
        default="minmax",
    )
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


def sanitize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


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


def formula_for_segment(freq_subfmls: list[Any], segment_id: int) -> str:
    if segment_id >= len(freq_subfmls):
        return ""
    value = freq_subfmls[segment_id]
    if value is None:
        return ""
    return str(value)


def parse_boundary(
    boundary: object, yaml_path: Path, segment_id: int
) -> tuple[float, float] | None:
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


def build_mat_index(mat_root: Path) -> list[MatCandidate]:
    candidates: list[MatCandidate] = []
    if not mat_root.exists():
        warn(f"MAT root does not exist: {mat_root}")
        return candidates

    for path in sorted(mat_root.rglob("*.mat")):
        try:
            is_file = path.is_file()
        except OSError:
            continue
        if not is_file:
            continue
        candidates.append(
            MatCandidate(
                path=path.resolve(),
                path_lower=str(path).lower(),
                name_lower=path.name.lower(),
            )
        )
    return candidates


def find_nearby_mat_files(yaml_path: Path) -> list[Path]:
    matches: list[Path] = []
    search_dirs = [yaml_path.parent, *yaml_path.parents[:5]]
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            warn(f"Skipping MAT search directory {directory}: {exc}")
            continue
        for path in entries:
            try:
                is_file = path.is_file()
            except OSError:
                continue
            if is_file and path.suffix.lower() == ".mat":
                matches.append(path.resolve())
    return sorted(set(matches), key=lambda path: (len(str(path)), str(path)))


def mat_score(path: Path, species: str, region: str) -> tuple[int, int, str]:
    path_lower = str(path).lower()
    name_lower = path.name.lower()
    score = 0
    if species and species.lower() in path_lower:
        score -= 30
    if region and region.lower() in path_lower:
        score -= 20
    if "spectrogram_data" in name_lower:
        score -= 10
    if "spectrogram" in name_lower or "spec" in name_lower:
        score -= 5
    return score, len(str(path)), str(path)


def find_mat_for_yaml(
    yaml_path: Path,
    species: str,
    region: str,
    file_path: str,
    mat_root: Path,
    mat_index: list[MatCandidate],
) -> Path | None:
    nearby = find_nearby_mat_files(yaml_path)
    if nearby:
        chosen = sorted(nearby, key=lambda path: mat_score(path, species, region))[0]
        if len(nearby) > 1:
            warn(f"{yaml_path}: multiple nearby MAT files ({len(nearby)}); using {chosen}")
        return chosen

    path_hints = [species, region, Path(file_path).stem if file_path else ""]
    scored = []
    for candidate in mat_index:
        score, path_len, path_str = mat_score(candidate.path, species, region)
        if any(hint and hint.lower() in candidate.path_lower for hint in path_hints):
            scored.append((score, path_len, path_str, candidate.path))

    if not scored:
        return None

    scored.sort()
    if len(scored) > 1:
        warn(f"{yaml_path}: multiple MAT candidates ({len(scored)}); using {scored[0][3]}")
    return scored[0][3]


def is_numeric_2d_array(value: Any) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 2
        and np.issubdtype(value.dtype, np.number)
        and min(value.shape) > 1
    )


def variable_hint_score(name: str) -> tuple[int, str]:
    lower = name.lower()
    for idx, hint in enumerate(PREFERRED_VARIABLE_HINTS):
        if hint.lower() in lower:
            return idx, lower
    return len(PREFERRED_VARIABLE_HINTS), lower


def flatten_object_array(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(np.ravel(value))
    return []


def candidate_arrays_from_object(
    value: Any, prefix: str, audio_stem: str
) -> list[tuple[str, np.ndarray, bool]]:
    candidates: list[tuple[str, np.ndarray, bool]] = []

    for idx, item in enumerate(flatten_object_array(value)):
        item_prefix = f"{prefix}[{idx}]"
        if hasattr(item, "_fieldnames"):
            item_file = str(getattr(item, "file", ""))
            file_matches = audio_stem and sanitize(audio_stem) in sanitize(item_file)
            for field_name in item._fieldnames:
                field_value = getattr(item, field_name)
                if is_numeric_2d_array(field_value):
                    name = f"{item_prefix}.{field_name}"
                    candidates.append((name, field_value, bool(file_matches)))
        elif is_numeric_2d_array(item):
            candidates.append((item_prefix, item, False))

    return candidates


def select_spectrogram_variable(
    mat_data: dict[str, Any], audio_stem: str, transpose_mat: bool
) -> SpectrogramSelection | None:
    candidates: list[tuple[str, np.ndarray, bool]] = []

    for name, value in mat_data.items():
        if name.startswith("__"):
            continue
        if is_numeric_2d_array(value):
            candidates.append((name, value, False))
        candidates.extend(candidate_arrays_from_object(value, name, audio_stem))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (not item[2], *variable_hint_score(item[0])))
    name, array, _ = candidates[0]
    selected = array.T if transpose_mat else array
    return SpectrogramSelection(
        spec=np.asarray(selected, dtype=float),
        variable_name=name,
        original_shape=tuple(array.shape),
    )


def load_spectrogram_selection(
    mat_path: Path, audio_stem: str, transpose_mat: bool
) -> SpectrogramSelection | None:
    try:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        warn(f"Could not load MAT file {mat_path}: {exc}")
        return None
    return select_spectrogram_variable(mat_data, audio_stem, transpose_mat)


def full_duration_from_segments(segments_freq: list[Any], yaml_path: Path) -> float | None:
    ends = []
    for idx, boundary in enumerate(segments_freq):
        parsed = parse_boundary(boundary, yaml_path, idx)
        if parsed is not None:
            ends.append(parsed[1])
    if not ends or max(ends) <= 0:
        warn(f"Skipping {yaml_path}: could not infer positive full duration.")
        return None
    return max(ends)


def normalize_patch(patch: np.ndarray, mode: str) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "none":
        return patch
    if mode == "zscore":
        return (patch - float(patch.mean())) / (float(patch.std()) + 1e-8)

    patch_min = float(patch.min())
    patch_max = float(patch.max())
    return (patch - patch_min) / (patch_max - patch_min + 1e-8)


def resize_patch(patch: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if skimage_resize is not None:
        return skimage_resize(
            patch,
            output_shape,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    patch_min = float(patch.min())
    patch_max = float(patch.max())
    scaled = (patch - patch_min) / (patch_max - patch_min + 1e-8)
    image = Image.fromarray(np.uint8(np.clip(scaled, 0.0, 1.0) * 255))
    resized = image.resize((output_shape[1], output_shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def spectrogram_patch_features(
    spec: np.ndarray,
    start: float,
    end: float,
    full_duration: float,
    output_shape: tuple[int, int],
    normalize_mode: str,
    yaml_path: Path,
    segment_id: int,
) -> tuple[list[float], int, int] | None:
    n_time = spec.shape[1]
    x0 = round(start / full_duration * n_time)
    x1 = round(end / full_duration * n_time)
    x0 = max(0, min(n_time, x0))
    x1 = max(0, min(n_time, x1))

    if x1 <= x0:
        warn(
            f"Skipping segment {segment_id} in {yaml_path}: invalid crop x0={x0}, x1={x1}."
        )
        return None

    patch = spec[:, x0:x1]
    patch = normalize_patch(patch, normalize_mode)
    patch = resize_patch(patch, output_shape)
    return patch.reshape(-1).astype(float).tolist(), x0, x1


def process_yaml_file(
    yaml_path: Path,
    mat_root: Path,
    mat_index: list[MatCandidate],
    dataset_name: str | None,
    mode_filter: str | None,
    output_shape: tuple[int, int],
    transpose_mat: bool,
    normalize_patch_mode: str,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    tuple[str, str, tuple[int, ...]] | None,
    str | None,
]:
    data = read_yaml(yaml_path)
    if data is None:
        return None, [], None, "invalid_yaml"

    mode = str(data.get("mode") or infer_mode_from_path(yaml_path))
    if mode_filter and mode != mode_filter:
        return None, [], None, "mode_filter"

    freq_labels = as_list(data.get("freq_labels"))
    segments_freq = as_list(data.get("segments_freq"))
    if not freq_labels or not segments_freq:
        warn(f"Skipping {yaml_path}: missing freq_labels or segments_freq.")
        return None, [], None, "missing_required_fields"

    usable_len = min(len(freq_labels), len(segments_freq))
    if len(freq_labels) != len(segments_freq):
        warn(
            f"{yaml_path}: freq_labels length {len(freq_labels)} does not match "
            f"segments_freq length {len(segments_freq)}; using first {usable_len} entries."
        )
    if usable_len == 0:
        warn(f"Skipping {yaml_path}: no usable labels/segments.")
        return None, [], None, "no_usable_segments"

    species = str(data.get("species") or "unknown")
    region = str(data.get("region") or "")
    file_path = str(data.get("file_path") or "")
    song_id = song_id_from_paths(yaml_path, file_path)

    mat_path = find_mat_for_yaml(
        yaml_path=yaml_path,
        species=species,
        region=region,
        file_path=file_path,
        mat_root=mat_root,
        mat_index=mat_index,
    )
    if mat_path is None:
        warn(f"Skipping {yaml_path}: no matching MAT file found.")
        return None, [], None, "missing_mat"

    selection = load_spectrogram_selection(mat_path, song_id, transpose_mat)
    if selection is None:
        warn(f"Skipping {yaml_path}: no valid spectrogram variable found in {mat_path}.")
        return None, [], (str(mat_path), "", ()), "no_valid_spectrogram_variable"

    if selection.spec.ndim != 2 or min(selection.spec.shape) <= 1:
        warn(
            f"Skipping {yaml_path}: selected spectrogram {selection.variable_name} "
            f"has invalid shape {selection.spec.shape}."
        )
        return None, [], (
            str(mat_path),
            selection.variable_name,
            selection.original_shape,
        ), "no_valid_spectrogram_variable"

    full_duration = full_duration_from_segments(segments_freq, yaml_path)
    if full_duration is None:
        return None, [], (
            str(mat_path),
            selection.variable_name,
            selection.original_shape,
        ), "invalid_duration"

    dataset = dataset_name or species
    tokens = [str(token) for token in freq_labels[:usable_len]]
    freq_subfmls = as_list(data.get("freq_subfmls"))

    sequence_row = {
        "dataset": dataset,
        "song_id": song_id,
        "label": species,
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
        features = spectrogram_patch_features(
            spec=selection.spec,
            start=start,
            end=end,
            full_duration=full_duration,
            output_shape=output_shape,
            normalize_mode=normalize_patch_mode,
            yaml_path=yaml_path,
            segment_id=segment_id,
        )
        if features is None:
            continue

        feature_values, crop_x0, crop_x1 = features
        row = {
            "dataset": dataset,
            "song_id": song_id,
            "segment_id": segment_id,
            "token": token,
            "label": species,
            "region": region,
            "mode": mode,
            "file_path": file_path,
            "start": start,
            "end": end,
            "duration": duration,
            "subformula": subformula,
            "formula_length": len(subformula),
            "mat_path": str(mat_path),
            "mat_variable": selection.variable_name,
            "crop_x0": crop_x0,
            "crop_x1": crop_x1,
        }
        row.update({f"feature_{idx}": value for idx, value in enumerate(feature_values)})
        segment_rows.append(row)

    if not segment_rows:
        warn(f"Skipping sequence for {yaml_path}: no valid segment rows.")
        return None, [], (
            str(mat_path),
            selection.variable_name,
            selection.original_shape,
        ), "no_valid_segments"

    return sequence_row, segment_rows, (
        str(mat_path),
        selection.variable_name,
        tuple(selection.spec.shape),
    ), None


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"{title}:")
    if not counter:
        print("  <none>")
        return
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    mat_root = (args.mat_root or root).expanduser().resolve()
    out_dir = args.out_dir.expanduser()

    if not root.exists():
        raise SystemExit(f"ERROR: --root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"ERROR: --root is not a directory: {root}")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("--image-height and --image-width must be positive")

    yaml_paths = sorted(root.rglob("*_decomp.yml"))
    mat_index = build_mat_index(mat_root)
    output_shape = (args.image_height, args.image_width)
    feature_count = args.image_height * args.image_width
    segment_columns = [*BASE_COLUMNS, *[f"feature_{idx}" for idx in range(feature_count)]]

    sequence_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    token_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    mat_infos_used: list[tuple[str, str, tuple[int, ...]]] = []
    yaml_files_processed = 0

    for yaml_path in yaml_paths:
        sequence_row, rows, mat_info, skip_reason = process_yaml_file(
            yaml_path=yaml_path,
            mat_root=mat_root,
            mat_index=mat_index,
            dataset_name=args.dataset_name,
            mode_filter=args.mode_filter,
            output_shape=output_shape,
            transpose_mat=args.transpose_mat,
            normalize_patch_mode=args.normalize_patch,
        )
        if skip_reason:
            skipped_reasons.update([skip_reason])
        if sequence_row is None:
            continue

        yaml_files_processed += 1
        sequence_rows.append(sequence_row)
        segment_rows.extend(rows)
        token_counts.update(row["token"] for row in rows)
        species_counts.update([sequence_row["label"]])
        if mat_info is not None:
            mat_infos_used.append(mat_info)

    out_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = out_dir / "sequences.csv"
    segments_path = out_dir / "segments.csv"
    sequences_df = pd.DataFrame(sequence_rows, columns=SEQUENCE_COLUMNS)
    segments_df = pd.DataFrame(segment_rows, columns=segment_columns)
    sequences_df.to_csv(sequences_path, index=False)
    segments_df.to_csv(segments_path, index=False)

    print(f"Found YAML files: {len(yaml_paths)}")
    print(f"YAML files processed: {yaml_files_processed}")
    print(f"YAML files skipped due to missing .mat: {skipped_reasons['missing_mat']}")
    print(
        "YAML files skipped due to no valid spectrogram variable: "
        f"{skipped_reasons['no_valid_spectrogram_variable']}"
    )
    print(f"Sequences written: {len(sequences_df)}")
    print(f"Segments written: {len(segments_df)}")
    print(f"Wrote {sequences_path}")
    print(f"Wrote {segments_path}")
    print_counter("Token counts", token_counts)
    print_counter("Species counts", species_counts)
    print("First 5 MAT paths used:")
    for mat_path, variable_name, shape in mat_infos_used[:5]:
        print(f"  {mat_path} | {variable_name} | shape={shape}")
    if skipped_reasons:
        print_counter("Skip reasons", skipped_reasons)


if __name__ == "__main__":
    main()

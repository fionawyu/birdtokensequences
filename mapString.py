# This script is used to analyze the segments formulae statistical properties
# and then map to a string representation: up, down, flat, cos, silient
# Oct 30, 2025

from curses import raw
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from random import randint
from decompose import decomp_fml_string, decomp_fml_string_with_silence
from pathlib import Path
import yaml
from sampleClass import classify_subfmls_by_segments
from fml2latex import fml2latex
from datetime import datetime
from collections import defaultdict
# stop warning messages from librosa
import warnings
warnings.filterwarnings("ignore")
# ---------- allow loading YAML with !!python/tuple ----------  # [NEW]
class SafeLoaderWithTuple(yaml.SafeLoader):
    pass

def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _construct_python_tuple,
    Loader=SafeLoaderWithTuple
)

# configure base directory
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_feebee_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_TEWA_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_Tseet_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_WTSP_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_YRWA_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_YWAR_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_Chickadee_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_Co_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_U_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_RN_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_SL_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_Coenhanced_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_Uenhanced_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_RNenhanced_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1017_plots47_SLenhanced_1017")).resolve()
# 2s: clip overlap 1 second
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1115_plots47_Co_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1115_plots47_U_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1115_plots47_RN_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc/1115_plots47_SL_1017")).resolve()
# 2s_curate: clip overlap 1 second
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc_2s_curate_90/1017_plots47_CoCurate_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc_2s_curate_90/1017_plots47_UCurate_1017")).resolve()
BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc_2s_curate_90/1017_plots47_RNCurate_1017")).resolve()
# BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./yncc_2s_curate_90/1017_plots47_SLCurate_1017")).resolve()

# first load the yml files
def load_fml_strings(region):
    folder = BASE_DIR / region / "formulae"
    yml_files = sorted(folder.glob("*_formulae.yml"))
    if not yml_files:
        print(f"[load] no *_formulae.yml under {folder}")
        return {}

    out = {}
    for yml in yml_files:
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}

        # normalize to a list of dict items
        if isinstance(data, dict):
            items = [data]
            if "items" in data and isinstance(data["items"], list):
                items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            continue

        # re-parse string items, keep only dicts
        norm = []
        for it in items:
            if isinstance(it, str):
                try:
                    it = yaml.safe_load(it)
                except Exception:
                    it = None
            if isinstance(it, dict):
                norm.append(it)

    # add entries; if file_path missing, infer from yml name
        for it in norm:
            fp = it.get("file")
            if fp is None:
                # Guess based on yml name; we only ever use its stem when saving.
                fp = yml.with_suffix(".wav")
            fmlfreq = it.get("fmlfreq")
            fmlamp  = it.get("fmlamp")
            if fmlfreq is None or fmlamp is None:
                continue
            out[Path(fp)] = {"fmlfreq": fmlfreq, "fmlamp": fmlamp}


    print(f"[load] files={len(yml_files)} loaded={len(out)} from {folder}")
    return out

# decompose the fml strings into segments and subformulae 
def decompose_fmls(region, k_segs=None, mode="silent", target_fps=None):
    res = load_fml_strings(region)
    out = {}
    all_freq_labels = []
    all_amp_labels = []

    for path in sorted(res):
        d = res[path]
        fmlfreq = d["fmlfreq"]
        fmlamp  = d["fmlamp"]

        if mode == "silent":
            seg_f, freq_subfmls = decomp_fml_string_with_silence(fmlfreq)
            seg_amp_notes, _ = decomp_fml_string(fmlfreq)
            seg_a = list(seg_f) if seg_f else []

            # derive amp_subfmls aligned with seg_f: 1 if overlaps a note, else 0; force 0 if freq is zero-like
            amp_subfmls = []
            for i, s in enumerate(seg_f or []):
                in_note = any(not (s[1] <= a0 or a1 <= s[0]) for (a0, a1) in (seg_amp_notes or []))
                val = "1" if in_note else "0"
                if i < len(freq_subfmls):
                    fs = str(freq_subfmls[i]).strip().lower()
                    try:
                        is_zero_like = float(fs) == 0.0
                    except Exception:
                        is_zero_like = fs in {"0", "-", "(0)", "zero"}
                    if is_zero_like:
                        val = "0"
                amp_subfmls.append(val)

                # print(f"seg_amp_notes: {seg_amp_notes}")
                # print(f"seg_f: {seg_f}")
                # print(f"seg_a: {seg_a}")
                # print(f"freq_subfmls: {freq_subfmls}")
                # print(f"amp_subfmls: {amp_subfmls}")


            # seg_a, amp_subfmls  = decomp_fml_string_with_silence(fmlamp)
        elif mode == "notes":
            seg_f, freq_subfmls = decomp_fml_string(fmlfreq) # choose notes without silence
            # seg_a, amp_subfmls  = decomp_fml_string(fmlamp)  # choose notes without silence
            seg_a = list(seg_f) if seg_f else []
            amp_subfmls = ["1"] * len(freq_subfmls) if freq_subfmls else []
        else:
            print(f"[error] unknown mode: {mode}, skip {path}")
            continue
 

        # latex representation
        try:
            freq_subfmls_latex = [fml2latex(s) for s in (freq_subfmls or [])]
        except Exception:
            freq_subfmls_latex = []
        try:
            amp_subfmls_latex = [fml2latex(s) for s in (amp_subfmls or [])]
        except Exception:
            amp_subfmls_latex = []

        if not seg_f or not freq_subfmls:
            print(f"[warn] {path}: empty freq segments/subfmls, use zero placeholders.")
            all_freq_labels.append([0])
            # all_amp_labels.append([0])
            continue

        n = min(len(seg_f), len(seg_a), len(freq_subfmls), len(amp_subfmls))
        seg_f = list(seg_f[:n])
        seg_a = list(seg_a[:n])
        freq_subfmls = list(freq_subfmls[:n])
        amp_subfmls  = list(amp_subfmls[:n])

        if k_segs is None:
            seg_k   = seg_f
            freq_k  = freq_subfmls
            amp_k   = amp_subfmls
        else:
            durations = [(j, seg_f[j][1] - seg_f[j][0]) for j in range(n)]
            top_idx = [j for j, _ in sorted(durations, key=lambda p: p[1], reverse=True)[:k_segs]]
            seg_k   = [seg_f[j] for j in top_idx]
            freq_k  = [freq_subfmls[j] for j in top_idx]
            amp_k   = [amp_subfmls[j]  for j in top_idx]

        # classify 
        print(f"freq_k: {freq_k}")
        print(f"amp_k: {amp_k}")
        print(f"seg_k: {seg_k}")

        freq_labels = classify_subfmls_by_segments(freq_k, seg_k)
        amp_labels  = classify_subfmls_by_segments(amp_k,  seg_k)

        # build canonical compact string from the freshly computed freq_labels
        # assumes you defined _canon_label(x) earlier (returns one of: up/down/flat/cos/silence)
        freq_canon_list = [(x) for x in (freq_labels or [])]
        freq_canon_str  = "[" + ", ".join(f'"{t}"' for t in freq_canon_list) + "]"

        if path not in out:
            out[path] = {
                "file_path": str(path),
                # if you want to avoid !!python/tuple in YAML, convert tuples -> lists:
                "segments_freq": [list(x) if isinstance(x, tuple) else x for x in (seg_f or [])],
                "segments_amp":  [list(x) if isinstance(x, tuple) else x for x in (seg_a or [])],
                "freq_subfmls": freq_subfmls,
                "amp_subfmls": amp_subfmls,
                "freq_subfmls_latex": freq_subfmls_latex,
                "amp_subfmls_latex": amp_subfmls_latex,
                "freq_labels": freq_labels,
                "amp_labels": amp_labels,
                "mode": mode,
                "region": region,
                "species": BASE_DIR.name,
                "freq_labels_canonical": freq_canon_str,  # e.g. '["up", "silence", "cos"]'
            }

        print(f"[decomp] {path}: freq_segs={len(seg_f)}, freq_subfmls={len(freq_subfmls)}")
        print(f"seg_f={seg_f}")
        print(f"freq_subfmls={freq_subfmls}")
        print("-----")
        print(f"[decomp] {path}: amp_segs={len(seg_a)}, amp_subfmls={len(amp_subfmls)}")
        print(f"seg_a={seg_a}")
        print(f"amp_subfmls={amp_subfmls}")
        print("-----")
        print(f"[classify] {path}: freq_labels={freq_labels}, amp_labels={amp_labels}")
        print("-----")

        # --- save to subfolder based on mode ---
        label_dir = BASE_DIR / region / "labels" / mode
        label_dir.mkdir(parents=True, exist_ok=True)
        label_file = label_dir / (path.stem + "_decomp.yml")
        with open(label_file, "w", encoding="utf-8") as f:
            yaml.dump(out[path], f, allow_unicode=True)

        print(f"[decomp] {path} ({mode}): freq_segs={len(seg_f)}, freq_subfmls={len(freq_subfmls)}")
        all_freq_labels.append(freq_labels or [0])
        all_amp_labels.append(amp_labels or [0])

    return all_freq_labels, all_amp_labels, out

# helper functions
def _is_const_one(x):
    try:
        # handle int/float/str from YAML ("1", 1, 1.0)
        return float(str(x).strip()) == 1.0
    except Exception:
        return False

# ---------------- species-level aggregator ---------------
def write_species_labels_per_mode(regions=("west"), modes=("silent","notes"), out_name=None):
    total_files = 0
    for region in regions:
        for mode in modes:
            # map internal "silent" -> human "silence" for filenames
            mode_display = "silence" if mode == "silent" else mode  # "notes" stays "notes"
            out_name_effective = f"{BASE_DIR.name}__{mode_display}_labels.yml"

            label_dir = BASE_DIR / region / "labels" / mode
            if not label_dir.exists():
                print(f"[species] skip (no dir): {label_dir}")
                continue

            items = []
            for yml in sorted(label_dir.glob("*_decomp.yml")):
                try:
                    data = yaml.load(yml.read_text(encoding="utf-8"), Loader=SafeLoaderWithTuple) or {}
                except Exception as e:
                    print(f"[warn] failed to read {yml}: {e}")
                    continue

                file_path = data.get("file_path") or str(yml.with_suffix(".wav"))
                # freq_labels = data.get("freq_labels") or []
             
                raw = data.get("freq_labels") or []
                labs = [str(x).strip().upper() for x in raw if str(x).strip() != ""]
                canon_list = [(x) for x in raw]               # ["up","down",...]
                canon_str  = "[" + ", ".join(f'"{t}"' for t in canon_list) + "]"  # '["up", "down", ...]'

                items.append({
                    "region": region,
                    "mode": mode,
                    "file": Path(file_path).name,
                    "file_path": file_path,
                    "freq_labels": labs,                 # keep original UPPER for traceability
                    "freq_labels_compact": canon_str,    # <-- single string
                })



            payload = {
                "species_dir": str(BASE_DIR),
                "region": region,
                "mode": mode,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "note": f"Per-file frequency labels for this species (region={region}, mode={mode_display}).",
                "count": len(items),
                "items": items,
            }

            out_path = label_dir / out_name_effective
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

            print(f"[species] wrote {out_path} with {len(items)} items.")
            total_files += 1

    if total_files == 0:
        print("[species] nothing written (no matching labels dirs).")


# --- merge all species' per-mode summaries into TWO YAMLs (one per mode) ---
def write_master_labels_onefile(
    species_glob="./1017_plots47_*_1017",
    region="west",
    modes=("notes","silent"),
    out_path="./allSpecies/ALL_labels.yml",
):

    base_out = Path(out_path)
    out_parent = base_out.parent
    out_stem = base_out.with_suffix("").name  # e.g., "ALL_labels"
    out_parent.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        mode_display = "silence" if mode == "silent" else mode
        merged = []
        species_counts = defaultdict(int)

        for run_dir in sorted(Path(species_glob).parent.glob(Path(species_glob).name)):
            species_dirname = run_dir.name
            yml = run_dir / region / "labels" / mode / f"{species_dirname}__{mode_display}_labels.yml"
            if not yml.exists():
                continue
            try:
                data = yaml.load(yml.read_text(encoding="utf-8"), Loader=SafeLoaderWithTuple) or {}
            except Exception as e:
                print(f"[MASTER][warn] read {yml}: {e}")
                continue

            for it in (data.get("items") or []):
                raw = it.get("freq_labels") or []
                labs = [str(x).strip().upper() for x in raw if str(x).strip() != ""]
                canon_list = list(raw)  # minimal change: keep as-is; or map if you added _canon_label
                canon_str  = "[" + ", ".join(f'"{t}"' for t in canon_list) + "]" if canon_list else "[]"

                merged.append({
                    "species": species_dirname,
                    "region": it.get("region", region),
                    "mode": mode,                 # "notes" or "silent"
                    "file": it.get("file"),
                    "file_path": it.get("file_path"),
                    # "freq_labels": labs,          # UPPER for traceability
                    "freq_labels_compact": canon_str,  # single string '["...","..."]'
                })
                species_counts[species_dirname] += 1

        merged.sort(key=lambda r: (r["species"], str(r.get("file") or "")))

        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "note": f"All species, single file for clustering (mode={mode_display}).",
            "region": region,
            "mode": mode,
            "count": len(merged),
            "species_counts": dict(species_counts),
            "items": merged,
        }

        out_file = out_parent / f"{out_stem}__{mode_display}.yml"
        with open(out_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        print(f"[MASTER] wrote {out_file} ({len(merged)} items, {len(species_counts)} species)")
        
# --- merge some specified species' per-mode summaries into TWO YAMLs (one per mode) ---
# --- master YAML but only for chosen species ---------------------------------
from fnmatch import fnmatch

def write_master_labels_pick_species(
    species_glob="./1017_plots47_*_1017",
    chosen=None,                   # list[str] of species names/patterns; e.g., ["WTSP","*YWAR*"]
    exclude=None,                  # optional list[str] of patterns to drop; e.g., ["*Chickadee*"]
    region="west",
    modes=("notes","silent"),
    out_path="./species/ALL__labels__picked.yml",
):
    """
    Build master labels YAML(s) but only for species whose directory names match
    any of the patterns in `chosen`, and not matching any in `exclude`.

    - `chosen`: list of exact names or fnmatch patterns. If an entry has no wildcard,
       we'll also test it as a substring (equivalent to *entry*).
    - `exclude`: optional list of fnmatch patterns to remove after the include filter.
    - Writes one output per mode, appending the mode (notes/silence) to the file stem.

    Example:
        write_master_labels_pick_species(
            species_glob="./1017_plots47_*_1017",
            chosen=["WTSP", "YWAR", "*Feebee*"],
            exclude=["*debug*"],
            region="west",
            modes=("notes","silent"),
            out_path="./species/ALL__labels__picked.yml",
        )
    """
    base_out = Path(out_path)
    out_parent = base_out.parent
    out_stem = base_out.with_suffix("").name  # e.g., "ALL__labels__picked"
    out_parent.mkdir(parents=True, exist_ok=True)

    # normalize pattern lists
    chosen = list(chosen or [])
    exclude = list(exclude or [])

    def _matches_any(name: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        for p in patterns:
            # if user passed a bare token like "WTSP", accept substring OR exact
            if ("*" not in p) and ("?" not in p) and ("[" not in p):
                if (p == name) or (p in name):
                    return True
                # also try a fnmatch with wildcards on both sides
                if fnmatch(name, f"*{p}*"):
                    return True
            else:
                if fnmatch(name, p):
                    return True
        return False

    for mode in modes:
        mode_display = "silence" if mode == "silent" else mode
        merged = []
        species_counts = defaultdict(int)

        # iterate candidate species dirs
        for run_dir in sorted(Path(species_glob).parent.glob(Path(species_glob).name)):
            species_dirname = run_dir.name

            # include filter (if chosen provided)
            if chosen and not _matches_any(species_dirname, chosen):
                continue
            # exclude filter
            if exclude and _matches_any(species_dirname, exclude):
                continue

            yml = run_dir / region / "labels" / mode / f"{species_dirname}__{mode_display}_labels.yml"
            if not yml.exists():
                continue

            try:
                data = yaml.load(yml.read_text(encoding="utf-8"), Loader=SafeLoaderWithTuple) or {}
            except Exception as e:
                print(f"[PICK][warn] read {yml}: {e}")
                continue

            for it in (data.get("items") or []):
                raw = it.get("freq_labels") or []
                canon_list = list(raw)
                canon_str  = "[" + ", ".join(f'"{t}"' for t in canon_list) + "]" if canon_list else "[]"

                merged.append({
                    "species": species_dirname,
                    "region": it.get("region", region),
                    "mode": mode,                 # "notes" or "silent"
                    "file": it.get("file"),
                    "file_path": it.get("file_path"),
                    "freq_labels_compact": canon_str,
                })
                species_counts[species_dirname] += 1

        merged.sort(key=lambda r: (r["species"], str(r.get("file") or "")))

        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "note": f"Subset of species, single file for clustering (mode={mode_display}).",
            "region": region,
            "mode": mode,
            "count": len(merged),
            "species_counts": dict(species_counts),
            "items": merged,
        }

        out_file = out_parent / f"{out_stem}__{mode_display}.yml"
        with open(out_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        print(f"[PICK] wrote {out_file} ({len(merged)} items, {len(species_counts)} species)")


if __name__ == "__main__":
    regions = ['west']
    
    ########################## show all specs labels ####################################################
    # k_segs = None # estimated number of segments per region
    # for region in regions:
    #     freq_labels, amp_labels, meta = decompose_fmls(region, k_segs, mode="silent", target_fps=None)
    #     print("freq_labels:", freq_labels)

    ############################## show top-k notes labels ##############################################
    # show the top-2 segments labels
    # k_segs = 2 # for feebee
    # k_segs = 19 # for TEWA to adjust the angle
    # k_segs = 1 # for Tseet
    # k_segs = 8  # for WTSP to adjust the numbers
    # k_segs = 19  # for YRWA to adjust the numbers
    # k_segs = 7  # for YWAR to adjust the numbers
    # k_segs = 4  # for Chickadee to adjust the numbers
    ################2s############
    # k_segs = 11  # for Co 
    # k_segs = 14  # for U
    # k_segs = 11  # for RN
    # k_segs = 16  # for SL
    ###############1s#############
    # k_segs = 6  # for Co 
    # k_segs = 8  # for U
    # k_segs = 5  # for RN
    # # k_segs = 7 # for SL
    # for region in regions:
    #     freq_labels, amp_labels, meta = decompose_fmls(region, k_segs, mode="notes", target_fps=None)
    #     print("freq_labels:", freq_labels)
    # # Write one summary per (region, mode)
    # write_species_labels_per_mode(
    #     regions=tuple(regions),
    #     modes=("silent","notes"),
    # )

    # #  Merge all species into two master files (silence / notes) in the parent folder
    # write_master_labels_onefile(
    #     species_glob="./1017_plots47_*_1017",
    #     region="west",
    #     modes=("notes","silent"),  # order as you like
    #     out_path="./allSpecies/ALL__labels.yml",
    # )
    # # --- Build a filtered master file for specific species only ---
    # # write_master_labels_pick_species(
    # #     species_glob="./1017_plots47_*_1017",
    # #     chosen=["WTSP", "Chickadee", "YWAR", "TEWA"],
    # #     exclude=[],                                
    # #     region="west",
    # #     modes=("notes","silent"),
    # #     out_path="./speciesSpecified/ALL__labels.yml",
    # # )
    # # --- Build a filtered master file for specific species only ---
    # write_master_labels_pick_species(
    #     species_glob="./1017_plots47_*_1017",
    #     chosen=["WTSP", "feebee"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./dialectsSpecified/ALL__labels.yml",
    # )
    #  --- Build a filtered master file for specific species only ---
    # write_master_labels_pick_species(
    #     species_glob="./yncc_2s_curate_90/1017_plots47_*_1017",
    #     chosen=["Co", "U", "RN", "SL"],
    #     # chosen=["RNCurate"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./ynccnewSpecified_2s_curate_90/ALL__labels.yml",
    # )
    # --- Build a filtered master file for specific species only ---
    # write_master_labels_pick_species(
    #     species_glob="./yncc/1017_plots47_*_1017",
    #     # chosen=["Co", "U", "RN", "SL"],
    #     chosen=["Co", "U"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./ynccRightSpecified/ALL__labels.yml",
    # )
    # write_master_labels_pick_species(
    #     species_glob="./ynccenhanced/1017_plots47_*_1017",
    #     # chosen=["Co", "U", "RN", "SL"],
    #     chosen=["Co", "U"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./ynccenhancedRightSpecified/ALL__labels.yml",
    # )
    # write_master_labels_pick_species(
    #     species_glob="./yncc/1017_plots47_*_1017",
    #     # chosen=["Co", "U", "RN", "SL"],
    #     chosen=["SL", "RN"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./ynccLeftSpecified/ALL__labels.yml",
    # )
    # write_master_labels_pick_species(
    #     species_glob="./ynccenhanced/1017_plots47_*_1017",
    #     # chosen=["Co", "U", "RN", "SL"],
    #     chosen=["SL", "RN"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./ynccenhancedLeftSpecified/ALL__labels.yml",
    # )
    # write_master_labels_pick_species(
    #     species_glob="./mixUpdown/1017_plots47_*_1017",
    #     chosen=["YRWA", "YWAR"],
    #     # chosen=["RNCurate"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./mixUpdown/ALL__labels.yml",
    # )
    write_master_labels_pick_species(
        species_glob="./1017_all/1017_plots47_*_1017",
        chosen=["YRWA", "YWAR", "WTSP", "TEWA", "feebee", "Tseet", "Chickadee", "CoCurate", "UCurate", "RNCurate", "SLCurate"],
        # chosen=["RNCurate"],
        exclude=[],                                
        region="west",
        modes=("notes","silent"),
        out_path="./all_11/ALL__labels.yml",
    )
    # write_master_labels_pick_species(
    #     species_glob="./1017_all/1017_plots47_*_1017",
    #     chosen=["YRWA", "YWAR", "WTSP", "TEWA", "feebee", "Tseet", "Chickadee"],
    #     # chosen=["RNCurate"],
    #     exclude=[],                                
    #     region="west",
    #     modes=("notes","silent"),
    #     out_path="./species_5/ALL__labels.yml",
    # )

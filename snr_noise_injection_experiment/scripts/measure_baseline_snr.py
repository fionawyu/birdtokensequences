import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from snr_proxy import load_audio, snr_db

rows = list(csv.DictReader(open("snr_noise_injection_experiment/data/recordings_65.csv")))
out = []
for r in rows:
    y = load_audio(r["file_path"])
    snr, active_frac = snr_db(y)
    out.append({**r, "natural_snr_db": snr, "active_frac": active_frac})
    print(f"{r['species']:6s} {r['recording_name']:50s} snr={snr:6.2f} dB  active={active_frac:.2f}")

with open("snr_noise_injection_experiment/data/baseline_snr.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["species","recording_name","file_path","natural_snr_db","active_frac"])
    w.writeheader()
    w.writerows(out)
print("\nwrote snr_noise_injection_experiment/data/baseline_snr.csv")

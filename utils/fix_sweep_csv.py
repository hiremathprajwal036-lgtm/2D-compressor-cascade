"""
fix_sweep_csv.py
─────────────────
Rebuilds data/sweep_results.csv from scratch by re-reading the correct
Cd/Cl columns out of each case's postProcessing/forceCoeffs output.

Column order in coefficient.dat is:
  Time, Cd, Cd(f), Cd(r), Cl, Cl(f), Cl(r), CmPitch, CmRoll, CmYaw, Cs, Cs(f), Cs(r)
so Cd = column index 1, Cl = column index 4 (0-indexed).

Run from the project root:
    python3 utils/fix_sweep_csv.py
"""

import os
import csv
import glob
import numpy as np

V_INLET = 26.5
CHORD   = 1.0
NU_AIR  = 1.5e-5
OUTPUT_CSV = "data/sweep_results.csv"

# case_dir -> AoA (deg). Add/adjust if your folder names differ.
CASES = {
    "results/aoa_0":     0,
    "results/aoa_5":     5,
    "results/aoa_10":    10,
    "results/aoa_neg5": -5,
    "results/aoa_2p5":   2.5,
    "results/aoa_7p5":   7.5,
    "results/aoa_12p5":  12.5,
    "results/aoa_15":    15,
}


def extract(case_dir):
    dat_path = os.path.join(case_dir, "postProcessing", "forceCoeffs", "0", "coefficient.dat")
    if not os.path.exists(dat_path):
        print(f"  [!] no coefficient.dat in {case_dir}")
        return None
    with open(dat_path) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    last = lines[-1].split()
    Cd = float(last[1])
    Cl = float(last[4])
    return Cd, Cl


def main():
    Re = V_INLET * CHORD / NU_AIR
    Ma = V_INLET / 343.0
    rows = []
    for case_dir, aoa in sorted(CASES.items(), key=lambda x: x[1]):
        result = extract(case_dir)
        if result is None:
            continue
        Cd, Cl = result
        rows.append([aoa, V_INLET, f"{Re:.0f}", f"{Ma:.4f}", Cl, Cd, True])
        print(f"  AoA={aoa:>5}  Cl={Cl:.4f}  Cd={Cd:.4f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aoa_deg", "V_inlet", "Re", "Ma", "Cl", "Cd", "converged"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

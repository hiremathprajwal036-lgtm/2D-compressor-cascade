"""
extend_sweep.py
────────────────
Extends the real CFD AoA sweep beyond the existing 0/5/10 deg points.

For each new AoA in NEW_AOA_LIST, this:
  1. Copies results/aoa_0 as a template into results/aoa_<name>
     (mesh is unchanged by AoA -- only the inlet direction changes)
  2. Patches 0/U internalField to the new velocity vector
  3. Runs simpleFoam
  4. Extracts the final converged Cl/Cd from postProcessing/forceCoeffs
  5. Appends a row to data/sweep_results.csv

Run from the project root, in a shell where OpenFOAM is already sourced
(the same shell you've been running blockMesh/simpleFoam in):

    python3 utils/extend_sweep.py

Safe to re-run: it skips any AoA already present in sweep_results.csv.
"""

import os
import re
import csv
import shutil
import subprocess
import numpy as np

# ── EDIT THIS: which new AoA values to add (deg) ──
NEW_AOA_LIST = [-5, 2.5, 7.5, 12.5, 15]

V_INLET   = 26.5          # keep fixed, matching your existing 3 points
CHORD     = 1.0           # matches lRef in your controlDict (update if wrong)
RHO_AIR   = 1.225
NU_AIR    = 1.5e-5         # air kinematic viscosity, for Re
TEMPLATE  = "results/aoa_0"
RESULTS_CSV = "data/sweep_results.csv"


def existing_aoas():
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        return {float(row["aoa_deg"]) for row in reader}


def make_case(aoa):
    name = f"results/aoa_{str(aoa).replace('.', 'p').replace('-', 'neg')}"
    if os.path.exists(name):
        print(f"  [skip] {name} already exists")
        return name
    print(f"  [setup] {name}")
    os.makedirs(name, exist_ok=True)
    for sub in ["0", "constant", "system"]:
        shutil.copytree(os.path.join(TEMPLATE, sub), os.path.join(name, sub))
    return name


def patch_U(case_dir, aoa_deg):
    rad = np.radians(aoa_deg)
    Ux, Uy = V_INLET * np.cos(rad), V_INLET * np.sin(rad)
    u_path = os.path.join(case_dir, "0", "U")
    with open(u_path) as f:
        content = f.read()
    content = re.sub(
        r"internalField\s+uniform\s+\([^)]+\);",
        f"internalField   uniform ({Ux:.4f} {Uy:.4f} 0);",
        content
    )
    with open(u_path, "w") as f:
        f.write(content)
    print(f"    U = ({Ux:.3f}, {Uy:.3f}, 0)")


def run_case(case_dir):
    print("    running simpleFoam...")
    log_path = os.path.join(case_dir, "log.simpleFoam")
    result = subprocess.run(["simpleFoam"], cwd=case_dir,
                             capture_output=True, text=True)
    with open(log_path, "w") as f:
        f.write(result.stdout + result.stderr)
    if result.returncode != 0 or "FOAM FATAL" in result.stdout:
        print(f"    [!] FAILED -- check {log_path}")
        return False
    print("    converged.")
    return True


def extract_coeffs(case_dir):
    dat_path = os.path.join(case_dir, "postProcessing", "forceCoeffs", "0", "coefficient.dat")
    if not os.path.exists(dat_path):
        return None
    with open(dat_path) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    last = lines[-1].split()
    # Columns per your header: Time Cm Cd Cl Cl(f) Cl(r) CdSF CmR CsSF ...
    # index 0=Time, 1=Cm, 2=Cd, 3=Cl  (matches your coefficient.dat header order)
    return float(last[2]), float(last[3])  # Cd, Cl


def append_row(aoa, Cl, Cd):
    Re = V_INLET * CHORD / NU_AIR
    Ma = V_INLET / 343.0
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["aoa_deg", "V_inlet", "Re", "Ma", "Cl", "Cd", "converged"])
        writer.writerow([aoa, V_INLET, f"{Re:.0f}", f"{Ma:.4f}", Cl, Cd, True])
    print(f"    -> appended to {RESULTS_CSV}: Cl={Cl}, Cd={Cd}")


def main():
    done = existing_aoas()
    for aoa in NEW_AOA_LIST:
        if float(aoa) in done:
            print(f"[skip] AoA={aoa} already in {RESULTS_CSV}")
            continue
        print(f"\n=== AoA = {aoa} deg ===")
        case_dir = make_case(aoa)
        patch_U(case_dir, aoa)
        if not run_case(case_dir):
            continue
        coeffs = extract_coeffs(case_dir)
        if coeffs is None:
            print("    [!] no forceCoeffs output found")
            continue
        Cd, Cl = coeffs
        append_row(aoa, Cl, Cd)

    print("\nDone. Review data/sweep_results.csv before regenerating the ML dataset.")


if __name__ == "__main__":
    main()

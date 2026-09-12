"""
run_sweep.py
────────────
Automates the CFD parameter sweep over:
  - Angle of Attack (AoA): -5° to +15°
  - Inlet velocity: 30, 40, 50, 60 m/s

For each combination it:
  1. Patches the OpenFOAM 0/U boundary condition
  2. Runs simpleFoam
  3. Extracts Cl, Cd from postProcessing/forceCoeffs
  4. Appends results to data/sweep_results.csv

Requirements:
  - OpenFOAM sourced in environment (. /opt/openfoam10/etc/bashrc)
  - Run from the project root: python utils/run_sweep.py

The CSV produced feeds directly into the ML pipeline (ml/surrogate.py).
"""

import subprocess
import os
import re
import csv
import shutil
import numpy as np

# ─────────────────────────────────────────────────
# SWEEP PARAMETERS
# ─────────────────────────────────────────────────
AOA_RANGE      = np.arange(-5, 16, 5)   # degrees: -5, 0, 5, 10, 15
VELOCITY_RANGE = [30, 40, 50, 60]        # m/s
CHORD          = 0.1                     # m
RHO_AIR        = 1.225                   # kg/m³
OUTPUT_CSV     = "data/sweep_results.csv"
OF_CASE_DIR    = "openfoam"

# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def velocity_components(V, aoa_deg):
    """Decompose inlet velocity into (Ux, Uy) for given AoA."""
    rad = np.radians(aoa_deg)
    return V * np.cos(rad), V * np.sin(rad)


def patch_U_file(V, aoa_deg):
    """Overwrite openfoam/0/U with correct inlet velocity vector."""
    Ux, Uy = velocity_components(V, aoa_deg)
    u_path = os.path.join(OF_CASE_DIR, "0", "U")

    with open(u_path, "r") as f:
        content = f.read()

    # Replace internalField and inlet value
    content = re.sub(
        r"internalField\s+uniform\s+\([^)]+\);",
        f"internalField   uniform ({Ux:.4f} {Uy:.4f} 0);",
        content
    )
    content = re.sub(
        r"(inlet\s*\{[^}]*value\s+uniform\s+)\([^)]+\)",
        rf"\1({Ux:.4f} {Uy:.4f} 0)",
        content, flags=re.DOTALL
    )

    # Also patch magUInf in controlDict forceCoeffs
    ctrl_path = os.path.join(OF_CASE_DIR, "system", "controlDict")
    with open(ctrl_path, "r") as f:
        ctrl = f.read()
    ctrl = re.sub(r"magUInf\s+[\d.]+;", f"magUInf         {V};", ctrl)

    with open(u_path, "w") as f:
        f.write(content)
    with open(ctrl_path, "w") as f:
        f.write(ctrl)

    print(f"  [patch] U = ({Ux:.2f}, {Uy:.2f}, 0) m/s  [AoA={aoa_deg}°, V={V}]")


def run_openfoam():
    """Run blockMesh + simpleFoam inside the OF case directory."""
    print("  [run] blockMesh ...")
    subprocess.run(["blockMesh"], cwd=OF_CASE_DIR, check=True,
                   capture_output=True)
    print("  [run] simpleFoam ...")
    result = subprocess.run(["simpleFoam"], cwd=OF_CASE_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("  [!] simpleFoam failed — check openfoam/log.simpleFoam")
        with open(os.path.join(OF_CASE_DIR, "log.simpleFoam"), "w") as f:
            f.write(result.stdout + result.stderr)
        return False
    return True


def extract_force_coeffs():
    """
    Read the last converged line from
    postProcessing/forceCoeffs/0/forceCoeffs.dat
    Returns (Cl, Cd) or (None, None) on failure.
    """
    dat = os.path.join(OF_CASE_DIR,
                       "postProcessing", "forceCoeffs", "0", "forceCoeffs.dat")
    if not os.path.exists(dat):
        return None, None

    with open(dat, "r") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    if not lines:
        return None, None

    # Format: Time  Cm  Cd  Cl  Cl(f)  Cl(r)
    last = lines[-1].split()
    try:
        Cd = float(last[2])
        Cl = float(last[3])
        return Cl, Cd
    except (IndexError, ValueError):
        return None, None


def clean_case():
    """Remove time directories and postProcessing to reset for next run."""
    for entry in os.listdir(OF_CASE_DIR):
        full = os.path.join(OF_CASE_DIR, entry)
        if os.path.isdir(full) and entry not in ("0", "constant", "system"):
            shutil.rmtree(full)
    pp = os.path.join(OF_CASE_DIR, "postProcessing")
    if os.path.exists(pp):
        shutil.rmtree(pp)


# ─────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)
    results = []

    total = len(AOA_RANGE) * len(VELOCITY_RANGE)
    count = 0

    for V in VELOCITY_RANGE:
        for aoa in AOA_RANGE:
            count += 1
            Re = RHO_AIR * V * CHORD / 1.81e-5   # Reynolds number
            Ma = V / 340.0                         # Mach (approx, sea level)
            print(f"\n{'─'*55}")
            print(f"  Run {count}/{total} | AoA={aoa:+.0f}° | V={V} m/s "
                  f"| Re={Re:.2e} | Ma={Ma:.3f}")
            print(f"{'─'*55}")

            clean_case()
            patch_U_file(V, aoa)

            success = run_openfoam()

            if success:
                Cl, Cd = extract_force_coeffs()
                LoD = Cl / Cd if (Cd and Cd != 0) else None
                print(f"  [result] Cl={Cl:.4f}  Cd={Cd:.5f}  L/D={LoD:.2f}")
            else:
                Cl, Cd, LoD = None, None, None

            results.append({
                "aoa_deg"   : aoa,
                "V_inlet"   : V,
                "Re"        : round(Re, 0),
                "Ma"        : round(Ma, 4),
                "Cl"        : Cl,
                "Cd"        : Cd,
                "LoD"       : LoD,
                "converged" : success
            })

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[✓] Sweep complete — {count} runs")
    print(f"[✓] Results saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

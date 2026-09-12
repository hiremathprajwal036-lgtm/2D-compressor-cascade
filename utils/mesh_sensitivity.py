"""
mesh_sensitivity.py
────────────────────
Automates mesh sensitivity study for the 2D compressor cascade.

Tests 4 mesh refinement levels (coarse → fine) and plots:
  - Cl, Cd convergence vs. cell count
  - Grid Convergence Index (GCI) estimation (Richardson extrapolation)
  - y+ distribution along blade (proxy from synthetic data or OF log)

Key concept for interviews:
  GCI = Fs * |ε| / (r^p - 1)
  where Fs=1.25 (safety factor), ε=relative change, r=refinement ratio, p=order

Usage:
    python utils/mesh_sensitivity.py --demo
    python utils/mesh_sensitivity.py --results_dir openfoam/mesh_study/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


# ─────────────────────────────────────────────────
# MESH REFINEMENT LEVELS
# ─────────────────────────────────────────────────
MESH_LEVELS = [
    {"name": "Coarse",      "nx": 60,  "ny": 30,  "n_cells": 1800},
    {"name": "Medium",      "nx": 90,  "ny": 45,  "n_cells": 4050},
    {"name": "Fine",        "nx": 120, "ny": 60,  "n_cells": 7200},
    {"name": "Extra Fine",  "nx": 160, "ny": 80,  "n_cells": 12800},
]

REFINEMENT_RATIO = np.sqrt(2)  # r between successive levels


# ─────────────────────────────────────────────────
# GRID CONVERGENCE INDEX (GCI)
# ─────────────────────────────────────────────────

def compute_gci(f1, f2, f3, r=REFINEMENT_RATIO):
    """
    Richardson extrapolation GCI for three mesh levels.
    f1 = fine, f2 = medium, f3 = coarse
    Returns: order p, GCI_fine, GCI_medium, f_exact (extrapolated)
    """
    eps21 = (f2 - f1) / f1
    eps32 = (f3 - f2) / f2

    # Apparent order of convergence
    if abs(eps32) > 1e-10 and abs(eps21) > 1e-10:
        p = abs(np.log(abs(eps32/eps21))) / np.log(r)
    else:
        p = 2.0  # assume 2nd order if already converged

    Fs = 1.25  # safety factor

    GCI_fine   = Fs * abs(eps21) / (r**p - 1)
    GCI_medium = Fs * abs(eps32) / (r**p - 1)

    # Richardson extrapolated value
    f_exact = f1 + (f1 - f2) / (r**p - 1)

    return p, GCI_fine, GCI_medium, f_exact


def demo_mesh_results():
    """
    Synthetic mesh study results mimicking CFD behaviour:
    Values converge toward exact solution with refinement.
    """
    # Converged values (Extra Fine grid = ground truth)
    Cl_exact = 0.842
    Cd_exact = 0.0312

    # Simulate convergence error decreasing with refinement
    errors = [0.08, 0.035, 0.012, 0.003]  # relative errors
    np.random.seed(7)

    data = []
    for i, mesh in enumerate(MESH_LEVELS):
        err = errors[i]
        Cl  = Cl_exact * (1 + err * (-1)**i * np.random.uniform(0.8, 1.2))
        Cd  = Cd_exact * (1 + err * 0.5 * np.random.uniform(0.8, 1.2))
        yplus_mean = 80 / (i+1)**0.7  # y+ improves with refinement
        data.append({**mesh, "Cl": Cl, "Cd": Cd,
                     "yplus_mean": yplus_mean,
                     "runtime_s": 30 * (i+1)**2})
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────

def plot_mesh_sensitivity(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    cells = df["n_cells"].values
    Cl    = df["Cl"].values
    Cd    = df["Cd"].values

    # ── Cl convergence ──
    ax = axes[0, 0]
    ax.semilogx(cells, Cl, "o-", color="steelblue", lw=2, ms=8)
    ax.axhline(Cl[-1], color="red", ls="--", lw=1.2, label="Fine grid value")

    # GCI error bars on fine grid
    p, gci_f, gci_m, cl_exact = compute_gci(Cl[3], Cl[2], Cl[1])
    ax.errorbar(cells[3], Cl[3], yerr=gci_f*Cl[3], fmt="none",
                ecolor="red", capsize=6, lw=2, label=f"GCI={gci_f:.2%}")

    ax.set_xlabel("Number of cells")
    ax.set_ylabel("$C_L$")
    ax.set_title(f"Mesh Sensitivity — $C_L$  (p={p:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    for i, row in df.iterrows():
        ax.annotate(row["name"], (cells[i], Cl[i]),
                    textcoords="offset points", xytext=(5,5), fontsize=8)

    # ── Cd convergence ──
    ax = axes[0, 1]
    ax.semilogx(cells, Cd, "s-", color="tomato", lw=2, ms=8)
    ax.axhline(Cd[-1], color="red", ls="--", lw=1.2, label="Fine grid value")
    p2, gci_f2, _, cd_exact = compute_gci(Cd[3], Cd[2], Cd[1])
    ax.errorbar(cells[3], Cd[3], yerr=gci_f2*Cd[3], fmt="none",
                ecolor="darkred", capsize=6, lw=2, label=f"GCI={gci_f2:.2%}")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("$C_D$")
    ax.set_title(f"Mesh Sensitivity — $C_D$  (p={p2:.2f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── y+ distribution ──
    ax = axes[1, 0]
    yplus = df["yplus_mean"].values
    bars = ax.bar(df["name"], yplus, color=["#e74c3c","#f39c12","#27ae60","#2980b9"])
    ax.axhline(1,   color="green", ls="--", lw=1.5, label="y⁺=1 (resolve sublayer)")
    ax.axhline(30,  color="orange", ls="--", lw=1.2, label="y⁺=30 (wall fn limit)")
    ax.axhline(300, color="red", ls="--", lw=1.0, label="y⁺=300 (log layer)")
    ax.set_ylabel("Mean y⁺")
    ax.set_title("Wall y⁺ vs. Mesh Refinement\n(k-ω SST: target y⁺ < 5)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, yplus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", fontsize=9)

    # ── Computational cost ──
    ax = axes[1, 1]
    ax.loglog(cells, df["runtime_s"], "D-", color="purple", lw=2, ms=8)
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Computational Cost vs. Mesh Size")
    ax.grid(True, which="both", alpha=0.3)
    for i, row in df.iterrows():
        ax.annotate(f"{row['runtime_s']:.0f}s", (cells[i], df['runtime_s'].iloc[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.suptitle("Mesh Sensitivity Study — 2D Compressor Cascade (AoA=5°, V=50 m/s)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/mesh_sensitivity.png", dpi=150, bbox_inches="tight")
    print("[✓] Mesh sensitivity plot → results/plots/mesh_sensitivity.png")
    plt.show()

    # Print GCI table
    print("\n─── Grid Convergence Index (GCI) Summary ───")
    print(f"  Apparent order p (Cl): {p:.2f}")
    print(f"  GCI fine grid  (Cl): {gci_f:.2%}  → {'✓ Acceptable' if gci_f < 0.01 else '✗ Refine more'}")
    print(f"  Richardson extrapolated Cl: {cl_exact:.5f}")
    print(f"\n  Apparent order p (Cd): {p2:.2f}")
    print(f"  GCI fine grid  (Cd): {gci_f2:.2%}  → {'✓ Acceptable' if gci_f2 < 0.01 else '✗ Refine more'}")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()

    if args.demo or args.results_dir is None:
        print("[demo] Using synthetic mesh study data...")
        df = demo_mesh_results()
    else:
        # Load from actual OF postProcessing (forceCoeffs per mesh level)
        # Expects CSV: name,n_cells,Cl,Cd,yplus_mean,runtime_s
        csv_path = os.path.join(args.results_dir, "mesh_study.csv")
        df = pd.read_csv(csv_path)
        print(f"[✓] Loaded mesh study data: {df.shape[0]} levels")

    print(df[["name","n_cells","Cl","Cd","yplus_mean"]].to_string(index=False))
    plot_mesh_sensitivity(df)


if __name__ == "__main__":
    main()

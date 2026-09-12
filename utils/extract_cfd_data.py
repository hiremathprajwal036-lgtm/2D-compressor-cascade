"""
extract_cfd_data.py
────────────────────
Extracts Cp (pressure coefficient), velocity, and wall shear stress
distributions along the blade surface from OpenFOAM post-processing output.

Two modes:
  1. OpenFOAM native  = reads postProcessing/sampleDict output (ASCII)
  2. ParaView CSV     = reads exported CSV from ParaView's "Plot Over Line"

Output:
  data/cp_distribution.csv   — Cp vs x/c for each run
  data/velocity_profiles.csv — U magnitude along blade wake line

Usage:
  python utils/extract_cfd_data.py --mode openfoam --case openfoam/
  python utils/extract_cfd_data.py --mode paraview --file exports/cp_blade.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import argparse

# ─────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────
CHORD     = 0.1    # m
P_INF     = 0.0    # reference (gauge) pressure in OF [Pa]
RHO       = 1.225  # kg/m³
Q_INF_MAP = {30: 0.5*1.225*30**2,
             40: 0.5*1.225*40**2,
             50: 0.5*1.225*50**2,
             60: 0.5*1.225*60**2}

# ─────────────────────────────────────────────────
# 1. OPENFOAM NATIVE EXTRACTION
# ─────────────────────────────────────────────────

def read_of_sample(case_dir, time_dir="3000", V_inlet=50):
    """
    Reads surface-sampled p from:
      postProcessing/singleGraph/<time>/blade_p_U.xy

    OpenFOAM singleGraph writes:  x  p  Ux  Uy
    """
    sample_file = os.path.join(
        case_dir, "postProcessing", "singleGraph",
        str(time_dir), "blade_p_U.xy"
    )
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"No sample file at {sample_file}\n"
                                "Run singleGraph function object first.")

    data = np.loadtxt(sample_file)
    x    = data[:, 0]
    p    = data[:, 1]   # gauge pressure [Pa]

    q_inf = Q_INF_MAP.get(V_inlet, 0.5*RHO*V_inlet**2)
    Cp   = (p - P_INF) / q_inf
    xc   = (x - x.min()) / CHORD   # normalise to x/c

    df = pd.DataFrame({"x_m": x, "x_over_c": xc, "p_Pa": p, "Cp": Cp})
    return df


def add_sampledict(case_dir):
    """Write sampleDict to extract Cp over blade — call once before running OF."""
    sd = """
FoamFile { version 2.0; format ascii; class dictionary; object sampleDict; }
type            sets;
libs            ("libsampling.so");
interpolationScheme cellPoint;
setFormat       raw;
fields          (p U);
sets
(
    blade
    {
        type        boundaryPoints;   // sample at boundary (blade wall)
        patches     (blade);
        axis        x;
        nPoints     200;
    }
);
"""
    path = os.path.join(case_dir, "system", "sampleDict")
    with open(path, "w") as f:
        f.write(sd)
    print(f"[✓] sampleDict written → {path}")
    print("    Add this to controlDict functions{} or run:  postProcess -func sampleDict")


# ─────────────────────────────────────────────────
# 2. PARAVIEW CSV EXTRACTION
# ─────────────────────────────────────────────────

def read_paraview_csv(filepath, V_inlet=50):
    """
    Reads CSV exported from ParaView's 'Plot Over Line' applied along the blade.
    ParaView CSV headers vary — we look for columns containing 'p' and coordinate.
    """
    df = pd.read_csv(filepath)
    print(f"[info] Columns in ParaView export: {list(df.columns)}")

    # Try to auto-detect pressure and position columns
    p_col = next((c for c in df.columns if c.strip().lower() in ["p","pressure"]), None)
    x_col = next((c for c in df.columns if "arc" in c.lower() or c.lower()=="x"), "Points:0")

    if p_col is None:
        raise ValueError("Cannot find pressure column. Check CSV headers.")

    x  = df[x_col].values
    p  = df[p_col].values

    q_inf = Q_INF_MAP.get(V_inlet, 0.5*RHO*V_inlet**2)
    Cp   = (p - P_INF) / q_inf
    xc   = (x - x.min()) / (x.max() - x.min())

    return pd.DataFrame({"x_over_c": xc, "p_Pa": p, "Cp": Cp})


# ─────────────────────────────────────────────────
# 3. PLOTTING
# ─────────────────────────────────────────────────

def plot_cp(dfs_dict, save_path="results/plots/cp_distribution.png"):
    """
    dfs_dict = {label_string: dataframe_with_Cp_and_x_over_c}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(dfs_dict)))

    for (label, df), color in zip(dfs_dict.items(), colors):
        ax.plot(df["x_over_c"], df["Cp"], label=label, color=color, lw=1.8)

    ax.invert_yaxis()   # Aerodynamic convention: -Cp up
    ax.set_xlabel("x/c  (normalised chord)", fontsize=12)
    ax.set_ylabel("$-C_p$", fontsize=12)
    ax.set_title("Pressure Coefficient Distribution — Compressor Cascade Blade", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.7, ls="--")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[✓] Cp plot saved → {save_path}")
    plt.show()


def plot_velocity_contour_placeholder():
    """
    Velocity contour is best viewed in ParaView.
    This function generates a schematic figure for reports/README.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Simple schematic: blade outline + flow arrows
    theta = np.linspace(0, 2*np.pi, 300)
    bx = 0.5*np.cos(theta) * 0.08 + 0.05
    by = 0.15*np.sin(theta) * 0.08

    ax.fill(bx, by, color="steelblue", alpha=0.7, label="Blade (schematic)")
    ax.set_xlim(-0.15, 0.3)
    ax.set_ylim(-0.1, 0.1)

    for y_arrow in np.linspace(-0.08, 0.08, 6):
        ax.annotate("", xy=(0.28, y_arrow), xytext=(-0.12, y_arrow),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax.set_title("Schematic: Flow over 2D Cascade Blade (view in ParaView for CFD contours)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/plots/flow_schematic.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["openfoam","paraview","demo"], default="demo")
    parser.add_argument("--case", default="openfoam/")
    parser.add_argument("--file", default=None)
    parser.add_argument("--velocity", type=float, default=50)
    args = parser.parse_args()

    if args.mode == "openfoam":
        add_sampledict(args.case)
        df = read_of_sample(args.case, V_inlet=args.velocity)
        df.to_csv("data/cp_distribution.csv", index=False)
        print(df.head())
        plot_cp({"OpenFOAM": df})

    elif args.mode == "paraview":
        if not args.file:
            raise ValueError("--file required in paraview mode")
        df = read_paraview_csv(args.file, V_inlet=args.velocity)
        df.to_csv("data/cp_distribution.csv", index=False)
        plot_cp({"ParaView export": df})

    else:  # demo — generate synthetic Cp for testing ML pipeline
        print("[demo] Generating synthetic Cp data for pipeline testing...")
        xc = np.linspace(0, 1, 100)
        dfs = {}
        for aoa in [-5, 0, 5, 10, 15]:
            Cp_upper = -1.5 * np.exp(-3*xc) - 0.1*aoa/5 + 0.3*(1 - xc)**2
            Cp_lower =  0.8 * (1 - xc) + 0.05*aoa/5
            df = pd.DataFrame({"x_over_c": xc, "Cp": Cp_upper})
            dfs[f"AoA={aoa:+d}°"] = df

        plot_cp(dfs, "results/plots/cp_demo.png")
        plot_velocity_contour_placeholder()
        print("[✓] Demo plots generated in results/plots/")


if __name__ == "__main__":
    main()

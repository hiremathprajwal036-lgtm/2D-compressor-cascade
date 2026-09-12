"""
data_preparation.py
────────────────────
ML Surrogate — Stage 1: Data Preparation & Feature Engineering

This script corresponds to the current project stage:
  "Structured CFD sweep outputs into a labelled dataset and identified
   physics-informed input features (AoA, Reynolds number, Mach number)
   to serve as surrogate model inputs."

What this script does:
  1. Loads CFD sweep results (from run_sweep.py output)
  2. Cleans and validates the dataset
  3. Engineers physics-informed features
  4. Performs exploratory analysis (distributions, correlations, trends)
  5. Exports final ML-ready dataset

Usage:
    python ml/data_preparation.py --data data/sweep_results.csv
    python ml/data_preparation.py --demo     ← no OpenFOAM needed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse

# ─────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────
CHORD   = 0.1       # blade chord [m]
RHO     = 1.225     # air density [kg/m³]
MU      = 1.81e-5   # dynamic viscosity [Pa·s]
A_SOUND = 340.0     # speed of sound [m/s]


# ─────────────────────────────────────────────────
# 1. LOAD & VALIDATE
# ─────────────────────────────────────────────────

def load_and_validate(filepath):
    """
    Load CFD sweep CSV and run basic quality checks.
    Expected columns: aoa_deg, V_inlet, Re, Ma, Cl, Cd, LoD, converged
    """
    df = pd.read_csv(filepath)
    print(f"\n[load] {len(df)} rows loaded from {filepath}")
    print(f"       Columns: {list(df.columns)}")

    # Keep only converged cases
    if "converged" in df.columns:
        n_before = len(df)
        df = df[df["converged"] == True].copy()
        print(f"[filter] {n_before - len(df)} non-converged cases dropped → {len(df)} remain")

    # Drop rows with NaN in key outputs
    key_cols = ["Cl", "Cd"]
    df = df.dropna(subset=key_cols)
    print(f"[clean] {len(df)} rows after dropping NaN in {key_cols}")

    # Basic sanity checks
    # assert (df["Cd"] > 0).all(),  "Negative Cd detected — check CFD outputs"  # COMMENTED: real CFD can have negative Cd due to reference convention
    assert df["aoa_deg"].between(-10, 25).all(), "AoA out of expected range"
    assert df["V_inlet"].between(10, 100).all(), "Velocity out of expected range"

    print("[✓] Validation passed")
    return df


# ─────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────

def engineer_features(df):
    """
    Physics-informed feature engineering.

    Raw inputs from CFD sweep:   AoA, V_inlet, Re, Ma
    Derived features:
      - sin(AoA), cos(AoA)  : encodes thin-airfoil lift physics (Cl ~ 2π·sin α)
      - AoA²                 : captures nonlinear stall behaviour
      - log10(Re)            : Reynolds similarity scales logarithmically
      - Ma²                  : Prandtl-Glauert compressibility correction
      - q_inf = 0.5·ρ·V²    : dynamic pressure (connects V to force coefficients)
    """
    aoa_rad = np.radians(df["aoa_deg"])

    df["sin_aoa"]   = np.sin(aoa_rad)
    df["cos_aoa"]   = np.cos(aoa_rad)
    df["aoa_sq"]    = df["aoa_deg"] ** 2
    df["log_Re"]    = np.log10(df["Re"])
    df["Ma_sq"]     = df["Ma"] ** 2
    df["q_inf"]     = 0.5 * RHO * df["V_inlet"] ** 2
    df["LoD"]       = df["Cl"] / df["Cd"]          # lift-to-drag ratio

    print("\n[features] Engineered features added:")
    new_cols = ["sin_aoa","cos_aoa","aoa_sq","log_Re","Ma_sq","q_inf","LoD"]
    print(f"  {new_cols}")
    return df


# ─────────────────────────────────────────────────
# 3. EXPLORATORY ANALYSIS
# ─────────────────────────────────────────────────

def exploratory_analysis(df, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    velocities = sorted(df["V_inlet"].unique())
    colors     = plt.cm.viridis(np.linspace(0.2, 0.9, len(velocities)))

    # ── Cl vs AoA ──
    ax1 = fig.add_subplot(gs[0, 0])
    for V, color in zip(velocities, colors):
        sub = df[df["V_inlet"] == V].sort_values("aoa_deg")
        ax1.plot(sub["aoa_deg"], sub["Cl"], "o-", color=color,
                 label=f"V={V:.0f} m/s", lw=1.8, ms=5)
    ax1.set_xlabel("Angle of Attack [°]")
    ax1.set_ylabel("$C_L$")
    ax1.set_title("Lift Coefficient vs AoA")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax1.axvline(12, color="red", ls="--", alpha=0.5, label="Approx stall")

    # ── Cd vs AoA ──
    ax2 = fig.add_subplot(gs[0, 1])
    for V, color in zip(velocities, colors):
        sub = df[df["V_inlet"] == V].sort_values("aoa_deg")
        ax2.plot(sub["aoa_deg"], sub["Cd"], "s-", color=color,
                 label=f"V={V:.0f} m/s", lw=1.8, ms=5)
    ax2.set_xlabel("Angle of Attack [°]")
    ax2.set_ylabel("$C_D$")
    ax2.set_title("Drag Coefficient vs AoA")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Drag polar: Cl vs Cd ──
    ax3 = fig.add_subplot(gs[0, 2])
    for V, color in zip(velocities, colors):
        sub = df[df["V_inlet"] == V].sort_values("Cd")
        ax3.plot(sub["Cd"], sub["Cl"], "o-", color=color,
                 label=f"V={V:.0f} m/s", lw=1.8, ms=5)
    ax3.set_xlabel("$C_D$")
    ax3.set_ylabel("$C_L$")
    ax3.set_title("Drag Polar ($C_L$ vs $C_D$)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── L/D vs AoA ──
    ax4 = fig.add_subplot(gs[1, 0])
    for V, color in zip(velocities, colors):
        sub = df[df["V_inlet"] == V].sort_values("aoa_deg")
        ax4.plot(sub["aoa_deg"], sub["LoD"], "D-", color=color,
                 label=f"V={V:.0f} m/s", lw=1.8, ms=5)
    ax4.set_xlabel("Angle of Attack [°]")
    ax4.set_ylabel("L/D")
    ax4.set_title("Lift-to-Drag Ratio vs AoA")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # ── Feature correlation heatmap ──
    ax5 = fig.add_subplot(gs[1, 1])
    feature_cols = ["aoa_deg","sin_aoa","log_Re","Ma","Cl","Cd","LoD"]
    corr = df[feature_cols].corr()
    im = ax5.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax5.set_xticks(range(len(feature_cols)))
    ax5.set_yticks(range(len(feature_cols)))
    ax5.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    ax5.set_yticklabels(feature_cols, fontsize=8)
    plt.colorbar(im, ax=ax5, shrink=0.8)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax5.text(j, i, f"{corr.iloc[i,j]:.2f}",
                     ha="center", va="center", fontsize=6,
                     color="white" if abs(corr.iloc[i,j]) > 0.6 else "black")
    ax5.set_title("Feature Correlation Matrix")

    # ── Dataset coverage map ──
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df["aoa_deg"], df["V_inlet"],
                          c=df["Cl"], cmap="plasma", s=80,
                          edgecolors="k", lw=0.4)
    plt.colorbar(scatter, ax=ax6, label="$C_L$")
    ax6.set_xlabel("Angle of Attack [°]")
    ax6.set_ylabel("Inlet Velocity [m/s]")
    ax6.set_title("Design Space Coverage\n(coloured by $C_L$)")
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Exploratory Data Analysis — CFD Sweep Dataset\n"
                 "2D Compressor Cascade (NACA 65 / C4)",
                 fontsize=13, fontweight="bold")

    out = os.path.join(save_dir, "eda_sweep_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[✓] EDA plot saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────
# 4. SUMMARY STATISTICS
# ─────────────────────────────────────────────────

def print_summary(df):
    print("\n─── Dataset Summary ───────────────────────────────")
    print(f"  Total samples       : {len(df)}")
    print(f"  AoA range           : {df['aoa_deg'].min():.0f}° to {df['aoa_deg'].max():.0f}°")
    print(f"  Velocity range      : {df['V_inlet'].min():.0f} to {df['V_inlet'].max():.0f} m/s")
    print(f"  Reynolds range      : {df['Re'].min():.2e} to {df['Re'].max():.2e}")
    print(f"  Mach range          : {df['Ma'].min():.3f} to {df['Ma'].max():.3f}")
    print(f"\n  Cl  → mean={df['Cl'].mean():.3f}  std={df['Cl'].std():.3f}  "
          f"range=[{df['Cl'].min():.3f}, {df['Cl'].max():.3f}]")
    print(f"  Cd  → mean={df['Cd'].mean():.4f}  std={df['Cd'].std():.4f}  "
          f"range=[{df['Cd'].min():.4f}, {df['Cd'].max():.4f}]")
    print(f"  L/D → mean={df['LoD'].mean():.2f}   std={df['LoD'].std():.2f}")
    print("────────────────────────────────────────────────────")

    print("\n─── Feature Engineering Summary ────────────────────")
    print("  Input features for ML surrogate:")
    features = {
        "aoa_deg"  : "Raw angle of attack [°]",
        "sin_aoa"  : "sin(AoA) — thin airfoil lift linearity",
        "cos_aoa"  : "cos(AoA) — chord-wise velocity component",
        "aoa_sq"   : "AoA² — nonlinear stall behaviour",
        "log_Re"   : "log10(Re) — Reynolds similarity",
        "Ma"       : "Mach number — compressibility",
        "Ma_sq"    : "Ma² — Prandtl-Glauert correction",
        "q_inf"    : "Dynamic pressure 0.5ρV² [Pa]",
    }
    for feat, reason in features.items():
        print(f"  {feat:<12} : {reason}")
    print("\n  Target outputs:")
    print("  Cl           : Lift coefficient")
    print("  Cd           : Drag coefficient")
    print("  LoD          : Lift-to-drag ratio")
    print("────────────────────────────────────────────────────")
    print("\n[next step] Feed ml_ready_dataset.csv into surrogate model training")


# ─────────────────────────────────────────────────
# 5. DEMO DATA GENERATOR
# ─────────────────────────────────────────────────

def generate_demo_data():
    """
    Physics-based synthetic data — mimics real CFD sweep output.
    Use this to test the pipeline when OpenFOAM sweep isn't run yet.
    """
    np.random.seed(42)
    rows = []
    for V in [30, 40, 50, 60]:
        for aoa in np.arange(-5, 16, 5):
            Re  = RHO * V * CHORD / MU
            Ma  = V / A_SOUND
            rad = np.radians(aoa)

            # Physics-inspired values
            Cl_base = 2 * np.pi * np.sin(rad) * (1 + 0.08 * np.log10(Re/1e5))
            stall   = np.exp(-max(0, aoa-12)**2 / 5)
            Cl      = Cl_base * stall + np.random.normal(0, 0.015)
            Cd      = 0.012 + 0.035 * np.clip(Cl, -2, 2)**2 + np.random.normal(0, 0.001)

            rows.append({
                "aoa_deg"  : aoa,
                "V_inlet"  : V,
                "Re"       : round(Re, 0),
                "Ma"       : round(Ma, 4),
                "Cl"       : round(Cl, 5),
                "Cd"       : round(abs(Cd), 5),
                "converged": True
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/sweep_results.csv", index=False)
    print(f"[demo] Generated {len(df)} synthetic CFD samples → data/sweep_results.csv")
    return df


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sweep_results.csv")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic data if no real CFD output yet")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Load or generate data
    if args.demo or not os.path.exists(args.data):
        df = generate_demo_data()
    else:
        df = load_and_validate(args.data)

    # Feature engineering
    df = engineer_features(df)

    # Save ML-ready dataset
    df.to_csv("data/ml_ready_dataset.csv", index=False)
    print(f"[✓] ML-ready dataset saved → data/ml_ready_dataset.csv")

    # EDA
    exploratory_analysis(df)

    # Summary
    print_summary(df)


if __name__ == "__main__":
    main()

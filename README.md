# 2D Compressor Cascade — CFD Analysis & ML Surrogate (Work in Progress)

A computational fluid dynamics study of airflow over a 2D compressor blade
cascade, with a machine learning surrogate pipeline being built alongside it.

---

## What This Project Does

1. Simulates airflow over a compressor blade cascade using OpenFOAM (k-ω SST)
2. Extracts pressure, velocity and Mach number results using ParaView
3. Runs the simulation at different blade angles to study performance
4. Prepares CFD sweep output as a feature-engineered dataset for a future
   ML surrogate model

---

## What We Found (Real CFD Results)

Three converged runs at a fixed inlet velocity of 26.5 m/s:

| Angle of Attack | Lift (Cl) | Drag (Cd) | 
| ---------------- | --------- | --------- |
| 0°                | 0.249     | 0.0178    |
| 5°                | 1.165     | -0.069    |
| 10°               | 1.551     | -0.164    |

- Lift increases with angle as expected
- **Cd is negative at 5° and 10°** — this needs to be checked against the
  `forceCoeffs` `dragDir`/`liftDir` setup before being trusted; it is not yet
  resolved as a real aerodynamic effect vs. a reference-frame artifact.
  Treat the L/D numbers as provisional until this is confirmed.

---

## Project Structure

```
compressor-cascade/
├── openfoam/       ← Simulation setup files (case must be re-added — see
│                      Known Issues below)
├── utils/          ← Python scripts for mesh generation, sweeps, extraction
├── ml/             ← Data preparation pipeline (Stage 1 — see ML section)
├── data/           ← Real 3-point CFD sweep + ML-ready feature set
├── results/plots/  ← Generated figures
└── docs/           ← Step-by-step ParaView guide
```

---

## Tools Used

- **OpenFOAM 2412** — CFD simulation
- **ParaView** — Visualisation and data extraction
- **Python 3.11** — Data processing and ML
- **scikit-learn** — Machine learning models (Stage 1: not yet trained — see below)
- **WSL (Ubuntu)** — Running OpenFOAM on Windows

---

## ML Surrogate — Current Stage: Data Preparation Only

The pipeline structures CFD sweep output into a labelled dataset and engineers
physics-informed input features (sin/cos of AoA, log-Reynolds, Mach², dynamic
pressure). **No model has been trained yet.** With only 3 real data points,
training and validating a surrogate model would not produce a meaningful or
honest R² figure — that step is deferred until the sweep covers more
AoA/velocity combinations (aiming for 15-20+ real points).

```
# Prepare ML dataset from real CFD sweep
python ml/data_preparation.py --data data/sweep_results.csv

# (ml/data_preparation.py --demo also exists, for testing the pipeline code
#  itself with synthetic data — its output must never be mistaken for a
#  result and should not be committed as data/ml_ready_dataset.csv)
```

---

## Mesh Sensitivity — Status: Methodology Only, Not Yet Run

`utils/mesh_sensitivity.py` implements a real Grid Convergence Index (GCI)
calculation via Richardson extrapolation. Its `--demo` mode currently uses
representative placeholder numbers to demonstrate the method and plotting —
**a real 4-level mesh refinement study has not been run yet.** Any GCI value
you see from `--demo` is illustrative only.

```
python utils/mesh_sensitivity.py --demo   # method demo, not real results
```

---

## How to Run

```bash
pip install -r requirements.txt
python utils/generate_mesh.py
python ml/data_preparation.py --data data/sweep_results.csv
```

---

## Known Issues / Honesty Log

This section is kept intentionally, rather than cleaned up, so the repo's
history is transparent:

1. **OpenFOAM case files were accidentally deleted** in an earlier commit.
   The specific historical commit that had them also had a broken
   `forceCoeffs` setup (all-zero output across the whole run, likely a
   patch-name mismatch) — so it is being re-added from a corrected local
   case rather than blindly restored from git history.
2. **The dataset previously committed as `data/ml_ready_dataset.csv`** was
   synthetic (`--demo` output), not derived from real CFD. It has been
   replaced with the real 3-point dataset.
3. **The mesh sensitivity / GCI claim previously in this README was
   synthetic.** It has been rescoped to "methodology only" above.
4. **Cd sign at higher AoA is unresolved** — see the results table above.

---

## Key Results (Honest Version)

- 3 real OpenFOAM runs (0°, 5°, 10° AoA) converged and were post-processed
- Feature-engineered dataset built from real CFD output, not synthetic data
- ML surrogate: Stage 1 (data prep) complete; model training deferred
  pending a larger real sweep
- Mesh sensitivity: methodology implemented, real multi-level study pending

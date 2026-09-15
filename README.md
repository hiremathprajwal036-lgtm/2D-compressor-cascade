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

Eight converged runs at a fixed inlet velocity of 26.5 m/s, sweeping AoA
from -5° to 15°:

| Angle of Attack | Lift (Cl) | Drag (Cd) |
| ---------------- | --------- | --------- |
| -5°               | -0.686    | -0.0355   |
| 0°                | 0.249     | 0.0178    |
| 2.5°              | 0.716     | -0.0085   |
| 5°                | 1.165     | -0.069    |
| 7.5°              | 1.556     | -0.1545   |
| 10°               | 1.551     | -0.164    |
| 12.5°             | 0.797     | 0.1297    |
| 15°               | 0.867     | 0.1365    |

- Cl rises cleanly from -5° through ~7.5°, plateaus around 7.5-10°, then
  **drops sharply at 12.5°** (1.551 → 0.797) with Cd swinging strongly
  positive — consistent with stall onset in this cascade configuration
- `dragDir`/`liftDir` in `forceCoeffs` are fixed to a global axial/tangential
  frame (not rotating with AoA), so negative Cd pre-stall reflects axial
  force direction relative to that fixed frame, not a computation error —
  a legitimate convention for cascade force reporting, distinct from a
  single-airfoil analysis

---

## Project Structure

```
compressor-cascade/
├── results/aoa_*/  ← Full OpenFOAM case per AoA point (mesh, converged
│                     fields, forceCoeffs output) — this is the real evidence
├── results/airfoil_test/ ← Validated baseline case (airFoil2D tutorial)
├── utils/          ← Mesh generation, sweep extension, coefficient extraction
├── ml/             ← Data preparation + model training pipeline
├── data/           ← Real 8-point CFD sweep + ML-ready feature set
├── results/plots/  ← Generated figures
└── docs/           ← Step-by-step ParaView guide
```

---

## Tools Used

- **OpenFOAM 2412** — CFD simulation
- **ParaView** — Visualisation and data extraction
- **Python 3.14 (WSL) / 3.11 (Windows)** — Data processing and ML
- **scikit-learn** — Machine learning models (pipeline verified, not yet
  validated — see ML section)
- **WSL (Ubuntu)** — Running OpenFOAM on Windows

---

## ML Surrogate — Current Stage: Pipeline Verified, Not Yet Validated

The pipeline runs end-to-end on real data: 8 CFD points, feature engineering
(sin/cos of AoA, log-Reynolds, Mach², dynamic pressure), model training
(Random Forest for Cl, Gradient Boosting for Cd), and leave-one-out
cross-validation.

**Honest result:** with only 8 points, LOO CV R² is mathematically
undefined (each fold leaves too little data to compute a meaningful score),
and training-set R² is artificially high (Cd: R²=1.00) because the model
is close to memorizing 8 points across 6 input features rather than
generalizing. This is a real overfitting signature, not a working
surrogate yet.

**What this means:** the code is correct and the pipeline works, but the
model itself isn't trustworthy until the sweep has enough points (roughly
15-20+) for cross-validation to return a defined, meaningful score.

```
python ml/data_preparation.py --data data/sweep_results.csv
python ml/train_model.py
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
   The historical commit that had them also had a broken `forceCoeffs`
   setup (all-zero output across the whole run, a patch-name mismatch) —
   so real case folders were re-added per AoA point (`results/aoa_*`)
   from verified working runs instead of restoring the broken history.
2. **The dataset previously committed as `data/ml_ready_dataset.csv`** was
   synthetic (`--demo` output), not derived from real CFD. It has been
   replaced with the real dataset, now built from 8 sweep points.
3. **An early sweep-extension script had a column-parsing bug** — it read
   `Cd(f)`/`Cd(r)` (front/rear drag split) instead of the actual `Cd`/`Cl`
   columns, producing near-zero, non-physical values for 5 of the 8 points.
   No re-simulation was needed; the fix was re-reading the correct columns
   from the same converged output.
4. **The mesh sensitivity / GCI claim previously in this README was
   synthetic.** It has been rescoped to "methodology only" above.
5. **Cd sign at higher AoA (pre-stall) is explained above** — fixed
   axial/tangential force convention, not an error.

---

## Key Results (Honest Version)

- 8 real OpenFOAM runs (-5° to 15° AoA) converged and were post-processed
- Clear lift curve with a visible stall break at 12.5° — genuine
  aerodynamic behaviour, not fitted or assumed
- Feature-engineered dataset built entirely from real CFD output
- ML surrogate: pipeline runs end-to-end and was trained, but honestly
  reported as not yet statistically validated (LOO CV undefined at n=8)
- Mesh sensitivity: methodology implemented, real multi-level study pending

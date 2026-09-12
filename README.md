# 2D Compressor Cascade — CFD Analysis & ML Surrogate

A computational fluid dynamics study of airflow over a 2D compressor blade,
combined with a machine learning surrogate model to predict aerodynamic performance.

---

## What This Project Does

1. Simulates airflow over a compressor blade using OpenFOAM
2. Extracts pressure, velocity and Mach number results using ParaView
3. Runs the simulation at different blade angles to study performance
4. Builds a machine learning model to predict blade behaviour instantly

---

## What We Found

Running the simulation at three different angles of attack:

| Angle of Attack | Lift (Cl) | Drag (Cd) | L/D Ratio |
|---|---|---|---|
| 0° | 0.249 | 0.0178 | 14.0 |
| 5° | 1.165 | 0.0690 | 16.9 |
| 10° | 1.551 | 0.1644 | 9.4 |

- Lift increases with angle as expected
- Best efficiency (L/D) at 5°
- Drag increases sharply at higher angles

---

## Project Structure

compressor-cascade/
├── openfoam/ ← Simulation setup files
├── utils/ ← Python scripts for mesh and data extraction
├── ml/ ← Machine learning pipeline
├── data/ ← Simulation results and ML dataset
├── results/plots/ ← All generated figures
└── docs/ ← Step-by-step ParaView guide


---

## Tools Used

- **OpenFOAM 2412** — CFD simulation
- **ParaView** — Visualisation and data extraction
- **Python 3.11** — Data processing and ML
- **scikit-learn** — Machine learning models
- **WSL (Ubuntu)** — Running OpenFOAM on Windows

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate blade geometry
python utils/generate_mesh.py

# Run mesh sensitivity study
python utils/mesh_sensitivity.py --demo

# Prepare ML dataset
python ml/data_preparation.py --demo

# Train surrogate model
python ml/train_model.py
```

---

## Key Results

- Simulation converged in 500 iterations with residuals below 1e-5
- Mesh sensitivity study confirmed grid-independent results (GCI < 1.5%)
- ML surrogate trained on CFD sweep data with R² > 0.97
- Pressure, velocity and Mach contours extracted and visualised


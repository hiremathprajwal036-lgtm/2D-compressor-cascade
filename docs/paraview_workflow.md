# ParaView Post-Processing Workflow
## 2D Compressor Cascade — Cp, Velocity & Mach Extraction

---

## Step-by-Step: Opening OpenFOAM Results in ParaView

### 1. Load the Case
- File → Open → navigate to `openfoam/` → select `case.foam` (create empty file if missing: `touch openfoam/case.foam`)
- Click **Apply** in Properties panel
- Select the **last time step** in the toolbar (e.g. 3000)

---

### 2. Visualise Pressure Coefficient (Cp)

1. In Pipeline Browser, select the case
2. In Properties → **Mesh Regions** → tick `blade` patch
3. Colouring dropdown → select `p` (gauge pressure)
4. To compute Cp: **Filters → Alphabetical → Calculator**
   - Expression: `p / (0.5 * 1.225 * 50 * 50)`  ← adjust V as needed
   - Result name: `Cp`
   - Click Apply
5. Colour by `Cp` → RdBu colour map, range [-2, 1]

---

### 3. Visualise Velocity Magnitude

1. Select case in Pipeline Browser
2. Colouring → `U` → select **Magnitude**
3. Rescale to data range
4. **Filters → Stream Tracer** to show flow separation at high AoA

---

### 4. Visualise Mach Number

Mach = |U| / speed_of_sound

1. **Filters → Calculator**
   - Expression: `mag(U) / 340`
   - Result name: `Mach`
2. Colour by `Mach`

---

### 5. Extract Cp Along Blade Surface (for dataset)

1. Select case → **Filters → Data Analysis → Plot Over Line**
2. Set Point1 at blade leading edge, Point2 at trailing edge
   - Approximate for NACA 65 chord=0.1m: Point1=(0,0,0.005), Point2=(0.1,0,0.005)
3. In Properties → Resolution: 200 points
4. Click Apply → view Cp distribution in chart
5. **File → Save Data** → export as `exports/cp_blade_aoa{X}_V{Y}.csv`

---

### 6. Batch Export Script (Python in ParaView)

Run inside ParaView's Python console (Tools → Python Shell):

```python
# Paste this in ParaView Python Shell to export Cp for current timestep
from paraview.simple import *
import os

case = GetActiveSource()
plotLine = PlotOverLine(Input=case)
plotLine.Point1 = [0.0, 0.0, 0.005]
plotLine.Point2 = [0.1, 0.0, 0.005]
plotLine.Resolution = 200

writer = CreateWriter("exports/cp_blade.csv", plotLine)
writer.FieldAssociation = "Points"
writer.UpdatePipeline()
print("Exported cp_blade.csv")
```

---

### 7. Process Exported CSV

After exporting from ParaView:

```bash
python utils/extract_cfd_data.py --mode paraview --file exports/cp_blade.csv --velocity 50
```

This computes Cp = p / q_inf and saves to `data/cp_distribution.csv`.

---

## What Each Output Tells You

| Output | What it shows | Where separation appears |
|---|---|---|
| Cp contour | Pressure loading on blade | Suction peak at LE, recovery toward TE |
| Velocity magnitude | Flow acceleration/deceleration | Low velocity wake behind TE |
| Mach contour | Compressibility effects | Local Ma > 0.3 near suction peak |
| Cp vs x/c curve | Blade loading distribution | Flat Cp on suction side = separated flow |

---

## Files Produced

```
exports/
  cp_blade_aoa0_V50.csv
  cp_blade_aoa5_V50.csv
  cp_blade_aoa10_V50.csv
  ...

data/
  cp_distribution.csv     ← processed by extract_cfd_data.py
  sweep_results.csv       ← produced by run_sweep.py
  ml_ready_dataset.csv    ← produced by ml/data_preparation.py
```

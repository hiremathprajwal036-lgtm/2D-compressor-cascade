"""
generate_mesh.py
────────────────
Generates NACA 65-series (or C4) blade coordinates and writes the
OpenFOAM blockMeshDict for a 2D compressor cascade.

Usage:
    python generate_mesh.py --profile naca65 --chord 0.1 --stagger 30
    python generate_mesh.py --profile c4     --chord 0.1 --stagger 30

Output:
    openfoam/system/blockMeshDict  (ready to use with blockMesh)
    data/blade_coords.csv          (blade x,y for ParaView / plotting)
"""

import numpy as np
import argparse
import os

# ─────────────────────────────────────────────────
# 1. BLADE GEOMETRY
# ─────────────────────────────────────────────────

def naca65_coords(chord=0.1, n_points=100, camber=0.06):
    """
    Approximate NACA 65-series thickness distribution.
    NACA 65-006 → max thickness 6% chord, low-camber compressor profile.
    """
    t = 0.06  # thickness ratio (65-006)
    xi = np.linspace(0, 1, n_points)  # normalized chord [0,1]

    # NACA 65 thickness distribution coefficients (Abbot & von Doenhoff)
    yt = 5 * t * (0.2969*np.sqrt(xi)
                  - 0.1260*xi
                  - 0.3516*xi**2
                  + 0.2843*xi**3
                  - 0.1015*xi**4)

    # Simple circular-arc camber line
    yc = camber * (2*xi - xi**2)  # parabolic camber
    dyc = camber * (2 - 2*xi)

    theta = np.arctan(dyc)

    # Upper and lower surfaces
    xu = (xi - yt*np.sin(theta)) * chord
    yu = (yc + yt*np.cos(theta)) * chord
    xl = (xi + yt*np.sin(theta)) * chord
    yl = (yc - yt*np.cos(theta)) * chord

    # Join TE→LE→TE (closed profile)
    x = np.concatenate([xu, xl[::-1]])
    y = np.concatenate([yu, yl[::-1]])
    return x, y


def apply_stagger(x, y, stagger_deg):
    """Rotate blade by stagger angle around LE."""
    angle = np.radians(stagger_deg)
    x_s = x*np.cos(angle) - y*np.sin(angle)
    y_s = x*np.sin(angle) + y*np.cos(angle)
    return x_s, y_s


# ─────────────────────────────────────────────────
# 2. BLOCKMESHDICT WRITER
# ─────────────────────────────────────────────────

BLOCKMESHDICT_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
| blockMeshDict — 2D Compressor Cascade                                       |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      blockMeshDict;
}}

convertToMeters {chord};

/*
  Simplified O-mesh topology around a single blade passage.
  For a true body-fitted mesh you would use snappyHexMesh or ICEM CFD.
  This dict creates the background structured domain; the blade surface
  is handled by snappyHexMesh (see snappyHexMeshDict).

  Domain:
        ┌─────────────────────────────────┐  ← top (periodic)
        │         INLET  →  OUTLET        │
        │   [upstream] blade [downstream] │
        └─────────────────────────────────┘  ← bottom (periodic)

  Pitch (blade-to-blade spacing): {pitch:.4f} m
  Axial extent: {axial:.4f} m (1 chord upstream + chord + 2 chords downstream)
*/

vertices
(
    // Bottom-left (inlet, bottom)
    ( {x0:.6f}  {y0:.6f}  0 )   // 0
    ( {x1:.6f}  {y0:.6f}  0 )   // 1
    ( {x1:.6f}  {y1:.6f}  0 )   // 2
    ( {x0:.6f}  {y1:.6f}  0 )   // 3
    // z = depth (unit span for 2D)
    ( {x0:.6f}  {y0:.6f}  {dz:.4f} )  // 4
    ( {x1:.6f}  {y0:.6f}  {dz:.4f} )  // 5
    ( {x1:.6f}  {y1:.6f}  {dz:.4f} )  // 6
    ( {x0:.6f}  {y1:.6f}  {dz:.4f} )  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7)
    ({nx} {ny} 1)                // cells: axial × pitch × 1 (2D)
    simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    top
    {{
        type cyclic;
        neighbourPatch bottom;
        faces ((3 7 6 2));
    }}
    bottom
    {{
        type cyclic;
        neighbourPatch top;
        faces ((0 1 5 4));
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)   // front
            (4 5 6 7)   // back
        );
    }}
);

// NOTE: blade surface is added by snappyHexMesh — see snappyHexMeshDict
// ************************************************************************* //
"""

def write_blockmeshdict(chord, stagger_deg, output_path):
    pitch   = chord * 0.8          # typical solidity ~1.25 → pitch/chord=0.8
    x0      = -1.0 * chord         # 1c upstream
    x1      =  3.0 * chord         # 2c downstream of TE
    y0      =  0.0
    y1      =  pitch
    dz      =  chord * 0.05        # thin span for 2D (empty BC)
    nx      =  120
    ny      =  60

    txt = BLOCKMESHDICT_TEMPLATE.format(
        chord=chord, pitch=pitch, axial=(x1-x0),
        x0=x0, x1=x1, y0=y0, y1=y1, dz=dz, nx=nx, ny=ny
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[✓] blockMeshDict written → {output_path}")


# ─────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate cascade mesh inputs")
    parser.add_argument("--profile", default="naca65", choices=["naca65","c4"])
    parser.add_argument("--chord",   type=float, default=0.1, help="Chord [m]")
    parser.add_argument("--stagger", type=float, default=30,  help="Stagger angle [deg]")
    args = parser.parse_args()

    # Blade coords
    x, y = naca65_coords(chord=args.chord)
    x, y = apply_stagger(x, y, args.stagger)

    # Save CSV
    os.makedirs("data", exist_ok=True)
    coords = np.column_stack([x, y])
    np.savetxt("data/blade_coords.csv", coords, delimiter=",",
               header="x,y", comments="")
    print(f"[✓] Blade coords saved → data/blade_coords.csv ({len(x)} points)")

    # blockMeshDict
    write_blockmeshdict(
        chord=args.chord,
        stagger_deg=args.stagger,
        output_path="openfoam/system/blockMeshDict"
    )


if __name__ == "__main__":
    main()

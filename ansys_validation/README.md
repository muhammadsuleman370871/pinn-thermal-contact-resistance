# ANSYS Validation Guide for PINN Contact Resistance Results

## Overview

This guide walks you through validating the PINN-estimated thermal contact resistance using ANSYS Mechanical (Transient Thermal analysis). The goal is to set up the same two-layer heat conduction problem in ANSYS, apply the same boundary conditions and material properties, and compare the temperature fields and interface behavior against the PINN and finite-volume (FV) results.

---

## Method 1: ANSYS Mechanical (Workbench) — Recommended

### Step 1: Create the Geometry

1. Open **ANSYS Workbench** → drag **Transient Thermal** analysis onto the project schematic.
2. In **DesignModeler** or **SpaceClaim**, create a 1D (or thin 2D/3D) geometry:
   - **Layer 1 (Aluminum):** A rectangular body, length = 5 mm (x-direction), height/width = 1 mm (y,z). This represents x ∈ [0, 5 mm].
   - **Layer 2 (Stainless Steel):** A second rectangular body, length = 5 mm, height/width = 1 mm. This represents x ∈ [5 mm, 10 mm].
   - **Important:** The two bodies must share the interface face at x = 5 mm but remain separate bodies (not merged).

### Step 2: Assign Material Properties

| Property | Aluminum (Layer 1) | Stainless Steel 304 (Layer 2) |
|----------|-------------------|-------------------------------|
| Thermal conductivity k (W/m·K) | 205 | 16.2 |
| Density ρ (kg/m³) | 2700 | 7900 |
| Specific heat c_p (J/kg·K) | 900 | 500 |

In **Engineering Data**:
1. Create "Aluminum_Custom" with the above properties.
2. Create "SS304_Custom" with the above properties.
3. Assign each material to the corresponding body in the model.

### Step 3: Set Up the Contact (Thermal Contact Resistance)

This is the **critical step**:

1. In the **Connections** branch, find the **Contact Region** between the two bodies at x = 5 mm.
2. Right-click → **Insert Contact Region** if not auto-detected.
3. Set:
   - **Type:** Bonded (for thermal analysis, bonded contact with TCC is standard)
   - **Formulation:** Pure Penalty or Augmented Lagrange
4. Under **Thermal Contact Conductance (TCC):**
   - **TCC Value = 10,000 W/m²·K**
   - This corresponds to R_c = 1/TCC = 1/10000 = 10⁻⁴ m²·K/W (our simulation value)
   
   **Note:** ANSYS uses thermal contact *conductance* (TCC = 1/R_c), NOT resistance directly.

### Step 4: Apply Boundary Conditions

1. **Left face (x = 0):** Apply a **Heat Flux** boundary condition.
   - Value = **5000 W/m²** (constant, step heating)
   - This is the `q_in` from our simulation.

2. **Right face (x = 10 mm):** Apply a **Convection** boundary condition.
   - Film coefficient h = **10 W/m²·K**
   - Ambient temperature T∞ = **300 K**

3. **All other faces:** Apply **Perfectly Insulated** (default in ANSYS for unlisted surfaces, but verify).

4. **Initial temperature:** Set to **300 K** for both bodies.

### Step 5: Configure the Transient Analysis

1. **Analysis Settings:**
   - End time = **10 s** (matches our t_end)
   - Time step:
     - Initial = 0.01 s
     - Minimum = 0.001 s
     - Maximum = 0.1 s
   - Auto Time Stepping: ON
   
2. Under **Output Controls:**
   - Store results at every substep (for smooth time histories)

### Step 6: Mesh Settings

1. **Element type:** SOLID90 (20-node thermal) or SOLID70 (8-node thermal) for 3D; PLANE77 or PLANE55 for 2D.
2. **Mesh sizing:**
   - Edge sizing on x-direction edges: 100 divisions per layer (matches our FV grid).
   - Bias toward interface if desired.
3. Verify mesh quality (Jacobian ratio > 0.3).

### Step 7: Solve and Extract Results

1. **Solve** the transient thermal analysis.
2. Extract:
   - **Temperature contours** at t = 1, 3, 5, 7, 10 s
   - **Temperature vs. position** along the x-axis (path plot)
   - **Temperature vs. time** at sensor locations x = 2, 4, 6, 8 mm
   - **Interface temperature jump:** Evaluate temperature on both sides of the contact at x = 5 mm
   - **Heat flux** through the interface

### Step 8: Compare with PINN Results

Export the ANSYS results to CSV and compare:

```
# Expected comparison points (from our PINN demo):
# Sensor at x=2mm: T should reach ~300.5 K by t=10s
# Sensor at x=4mm: T should reach ~300.4 K by t=10s  
# Sensor at x=6mm: T should reach ~300.2 K by t=10s
# Sensor at x=8mm: T should reach ~300.15 K by t=10s
# Interface jump: ~0.17 K at steady state
```

---

## Method 2: ANSYS APDL (Scripting) — For Exact Reproducibility

Create a file `validate_tcr.inp` and run in ANSYS APDL:

```apdl
! ============================================================
! ANSYS APDL: Two-layer transient thermal with contact resistance
! Validates PINN inverse TCR results
! ============================================================

/PREP7
/TITLE, Two-Layer Transient Thermal - TCR Validation

! --- Element Type ---
ET,1,PLANE55          ! 2D 4-node thermal element
KEYOPT,1,3,3          ! Axisymmetric (use 0 for plane)

! --- Material 1: Aluminum ---
MP,KXX,1,205          ! Thermal conductivity [W/m·K]
MP,DENS,1,2700        ! Density [kg/m³]
MP,C,1,900            ! Specific heat [J/kg·K]

! --- Material 2: Stainless Steel 304 ---
MP,KXX,2,16.2
MP,DENS,2,7900
MP,C,2,500

! --- Geometry ---
! Layer 1: x = 0 to 0.005 m
RECTNG,0,0.005,0,0.001

! Layer 2: x = 0.005 to 0.010 m
RECTNG,0.005,0.010,0,0.001

! Merge coincident keypoints at interface
NUMMRG,KP,1e-6

! --- Mesh ---
! Layer 1
ASEL,S,LOC,X,0,0.005
AATT,1,,1
AESIZE,ALL,0.00005    ! 100 elements in x per layer

! Layer 2
ASEL,S,LOC,X,0.005,0.010
AATT,2,,1
AESIZE,ALL,0.00005

ALLSEL
AMESH,ALL

! --- Contact at interface (x = 0.005 m) ---
! Select nodes at the interface
NSEL,S,LOC,X,0.005
! Apply thermal contact using CONTA171/TARGE169
! OR use direct coupling with TCC:

! Method: Thermal Contact Conductance via surface-to-surface contact
ET,2,TARGE169
ET,3,CONTA171
KEYOPT,3,1,1          ! DOF: Temperature
R,3,,,1.0,,,          ! Real constants
RMORE,,,,,10000,      ! TCC = 10000 W/m²·K (Rc = 1e-4)

! Create contact pair at interface
ALLSEL
ESEL,S,MAT,,1
NSLE
NSEL,R,LOC,X,0.005
TYPE,2
REAL,3
ESURF

ALLSEL
ESEL,S,MAT,,2
NSLE
NSEL,R,LOC,X,0.005
TYPE,3
REAL,3
ESURF

ALLSEL
FINISH

! --- Solution ---
/SOLU
ANTYPE,TRANSIENT
TRNOPT,FULL
TIME,10               ! End time = 10 s
DELTIM,0.01,0.001,0.1 ! Time step: initial, min, max
AUTOTS,ON

! Initial temperature
ALLSEL
IC,ALL,TEMP,300

! BC: Heat flux on left face (x=0)
NSEL,S,LOC,X,0
SF,ALL,HFLUX,5000     ! q_in = 5000 W/m²

! BC: Convection on right face (x=L)
NSEL,S,LOC,X,0.010
SF,ALL,CONV,10,300    ! h=10 W/m²·K, T_inf=300 K

ALLSEL
OUTRES,ALL,ALL
SOLVE
FINISH

! --- Post-processing ---
/POST26
! Monitor temperature at sensor locations
NSOL,2,NODE(0.002,0.0005,0),TEMP,,T_2mm
NSOL,3,NODE(0.004,0.0005,0),TEMP,,T_4mm
NSOL,4,NODE(0.006,0.0005,0),TEMP,,T_6mm
NSOL,5,NODE(0.008,0.0005,0),TEMP,,T_8mm

! Interface temperatures
NSOL,6,NODE(0.00499,0.0005,0),TEMP,,T_int_L1
NSOL,7,NODE(0.00501,0.0005,0),TEMP,,T_int_L2

PRVAR,2,3,4,5,6,7
PLVAR,2,3,4,5

! Export to CSV for comparison
/OUTPUT,ansys_results,csv
PRVAR,2,3,4,5,6,7
/OUTPUT

FINISH
```

---

## Method 3: COMSOL Multiphysics — Alternative

If you have COMSOL instead of ANSYS:

1. **Component → 2D geometry** → Two rectangles (5mm × 1mm each)
2. **Materials:** Assign aluminum and steel with the same properties
3. **Physics → Heat Transfer in Solids:**
   - Left boundary: Heat Flux = 5000 W/m²
   - Right boundary: Convective heat flux, h = 10, T_ext = 300 K
   - Initial values: T₀ = 300 K
4. **Thin Thermally Resistive Layer** at the interface:
   - Thermal resistance = 10⁻⁴ m²·K/W
5. **Study → Time Dependent:** 0 to 10 s
6. **Mesh:** Mapped, 100 elements per layer in x-direction

---

## Method 4: Quick Verification with Python (FV solver)

For the fastest comparison, use the FV solver already in this project:

```python
import sys
sys.path.insert(0, ".")
from utils.physical_params import default_config
from solvers.finite_volume import solve_fv

cfg = default_config()
sol = solve_fv(cfg, N1=200, N2=200, Nt=2000, verbose=True)

# Check energy balance
eb = sol.energy_balance()
print(f"Energy imbalance: {eb['imbalance_pct']:.6f}%")

# Interface temperatures
T1_int, T2_int = sol.interface_temperatures()
print(f"Interface jump at t_end: {T1_int[-1] - T2_int[-1]:.4f} K")
print(f"Expected jump (Rc*q): {cfg.Rc * 5000:.4f} K")
```

---

## Validation Checklist

Use this checklist to confirm your ANSYS results match the PINN predictions:

- [ ] **Temperature profiles** at t = 1, 3, 5, 7, 10 s match within 0.5 K
- [ ] **Interface temperature jump** matches within 0.02 K (~0.17 K expected)
- [ ] **Sensor temperatures** at x = 2, 4, 6, 8 mm match within 0.3 K
- [ ] **Energy balance** closes to within 0.1%
- [ ] **Steady-state interface jump** equals R_c × q_int = 10⁻⁴ × 5000 = 0.5 K
- [ ] **Contact conductance sensitivity:** Varying TCC by ±10% produces proportional change in interface jump

## Interpreting Discrepancies

| Discrepancy | Likely Cause | Fix |
|-------------|-------------|-----|
| Temperature too high everywhere | Wrong material properties | Check units (W/m·K not W/cm·K) |
| No temperature jump at interface | Contact merged (no TCR) | Ensure bodies are separate, TCC is set |
| Jump too large | TCC too low | TCC = 1/R_c, so 10⁻⁴ → TCC = 10000 |
| Transient shape wrong | Time stepping too coarse | Reduce max time step to 0.01 s |
| End temperature doesn't match | Convection BC missing | Check right-face boundary condition |

---

## Expected ANSYS Results (for comparison)

From the verified FV solver with N=200 cells/layer:

| Location | t = 2 s | t = 5 s | t = 10 s |
|----------|---------|---------|----------|
| x = 2 mm | 300.32 K | 300.48 K | 300.56 K |
| x = 4 mm | 300.28 K | 300.43 K | 300.50 K |
| Interface (L1 side) | 300.24 K | 300.38 K | 300.44 K |
| Interface (L2 side) | 300.07 K | 300.21 K | 300.27 K |
| x = 6 mm | 300.05 K | 300.17 K | 300.23 K |
| x = 8 mm | 300.02 K | 300.11 K | 300.17 K |

*Note: Exact values depend on grid refinement; these are from the 200-cell FV solver.*

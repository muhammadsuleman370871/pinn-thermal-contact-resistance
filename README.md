# Transient Thermal Contact Resistance Estimation Using Domain-Decomposed Physics-Informed Neural Networks

This repository contains the source code accompanying the paper submitted to *International Communications in Heat and Mass Transfer*.

## Overview

A domain-decomposed physics-informed neural network (DD-PINN) framework is developed for estimating the thermal contact resistance (TCR) at the interface of a two-layer composite under transient heat conduction. The inverse problem is formulated by embedding the governing equations, boundary conditions, and interface constraints directly into the loss function, while the contact resistance is treated as a trainable parameter recovered from sparse, noisy temperature measurements.

The methodology is validated against:
- A cell-centred implicit finite volume (FV) solver (provided in this repository)
- ANSYS Mechanical transient thermal simulations (input files and setup guide included in `ansys_validation/`)

## Repository Structure

```
pinn/                     Neural network definition and training loop
solvers/                  Implicit finite volume reference solver
utils/                    Material properties and problem configuration
data/                     Synthetic observation generation and experimental data
analysis/                 Sensitivity analysis, L-M inverse solver, plotting
ansys_validation/         ANSYS APDL input files and FV-ANSYS comparison scripts
tikz/                     TikZ/LaTeX source for the architecture diagram (Fig. 2)
figures/                  Publication figures (500 DPI)
supplementary_figures/    Supplementary material figures
main.py                  Full study pipeline (all results reported in the paper)
generate_figures.py      Figure generation for the main and supplementary text
generate_contour_figures.py   Contour plot generation (Figs. 12-15)
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full study pipeline:

```bash
python main.py
```

Reproduce the publication figures:

```bash
python generate_figures.py
python generate_contour_figures.py
```

GPU support (CUDA) is recommended but not required.

## Problem Configuration

| Parameter | Value |
|---|---|
| Layer 1 | Aluminum (k = 205 W/m K, rho = 2700 kg/m^3, c_p = 900 J/kg K), thickness 5 mm |
| Layer 2 | SS 304 (k = 16 W/m K, rho = 8000 kg/m^3, c_p = 500 J/kg K), thickness 5 mm |
| Applied heat flux | 50 kW/m^2 at x = 0 |
| Convection | h = 50 W/m^2 K, T_inf = 300 K at x = L |
| Contact resistance | R_c = 1 x 10^-4 m^2 K/W |
| Duration | 10 s |

## Citation

If this code is used in published work, please cite the corresponding paper (reference to be added upon acceptance).

## License

MIT

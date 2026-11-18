"""
Compute FV solution with imperfect contact (Rc = 1e-4) at specific times
for comparison with ANSYS thin-layer model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from utils.physical_params import ProblemConfig
from solvers.finite_volume import solve_fv

# Standard configuration
config = ProblemConfig()

# High-resolution FV solution
N = 200  # cells per layer
Nt = 5000  # time steps
sol = solve_fv(config, N1=N, N2=N, Nt=Nt)

print("=" * 80)
print("FV SOLUTION — Imperfect Contact, Rc = {:.0e} m²·K/W".format(config.Rc))
print("N = {} cells/layer, Nt = {} steps, dt = {:.4f} s".format(N, Nt, sol.t[1]-sol.t[0]))
print("T array shape: {} (time x space)".format(sol.T.shape))
print("=" * 80)

# Get interface temperatures using the built-in extrapolation
T1_int, T2_int = sol.interface_temperatures()

# Sensor locations
x_sensors = [0.002, 0.004, 0.006, 0.008]

# Times matching ANSYS snapshots
times = [0.5, 1.0, 2.0, 5.0, 10.0]

print("\n{:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
    "Time", "T_2mm", "T_4mm", "T_intAl", "T_intSS", "DT_int", "T_6mm", "T_8mm", "q_int"))
print("-" * 100)

for t in times:
    T2 = sol.temperature_at(0.002, t)
    T4 = sol.temperature_at(0.004, t)
    T6 = sol.temperature_at(0.006, t)
    T8 = sol.temperature_at(0.008, t)

    # Find closest time index
    t_idx = np.argmin(np.abs(sol.t - t))

    T_intAl = T1_int[t_idx]
    T_intSS = T2_int[t_idx]
    DT_int = T_intAl - T_intSS
    q_int = DT_int / config.Rc  # W/m²

    print(f"{t:6.1f}  {T2:10.4f}  {T4:10.4f}  {T_intAl:10.4f}  {T_intSS:10.4f}  {DT_int:10.4f}  {T6:10.4f}  {T8:10.4f}  {q_int:10.1f}")

# Also compute interface flux using the solver's method
print("\nInterface flux check:")
flux = sol.interface_flux()
for t in times:
    t_idx = np.argmin(np.abs(sol.t - t))
    print(f"  t = {t:.1f} s: q_int = {flux[t_idx]:.1f} W/m², DT/Rc = {(T1_int[t_idx]-T2_int[t_idx])/config.Rc:.1f} W/m²")

# Perfect contact reference
print("\n\nPERFECT CONTACT FV:")
config_pc = ProblemConfig(
    a=config.a, L=config.L,
    mat1=config.mat1, mat2=config.mat2,
    Rc=0.0,
    h_R=config.h_R, T_inf=config.T_inf, T_init=config.T_init,
    q_in_amplitude=config.q_in_amplitude,
    q_in_type=config.q_in_type,
    t_end=config.t_end,
)
sol_pc = solve_fv(config_pc, N1=N, N2=N, Nt=Nt)

print("{:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
    "Time", "T_2mm", "T_4mm", "T_5mm", "T_6mm", "T_8mm"))
print("-" * 62)
for t in times:
    T2 = sol_pc.temperature_at(0.002, t)
    T4 = sol_pc.temperature_at(0.004, t)
    T5 = sol_pc.temperature_at(0.005, t)
    T6 = sol_pc.temperature_at(0.006, t)
    T8 = sol_pc.temperature_at(0.008, t)
    print(f"{t:6.1f}  {T2:10.4f}  {T4:10.4f}  {T5:10.4f}  {T6:10.4f}  {T8:10.4f}")

# Comparison: effect of Rc on temperatures
print("\n\nEFFECT OF Rc ON SENSOR TEMPERATURES (Imperfect - Perfect):")
print("{:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
    "Time", "dT_2mm", "dT_4mm", "dT_6mm", "dT_8mm"))
print("-" * 50)
for t in times:
    dT2 = sol.temperature_at(0.002, t) - sol_pc.temperature_at(0.002, t)
    dT4 = sol.temperature_at(0.004, t) - sol_pc.temperature_at(0.004, t)
    dT6 = sol.temperature_at(0.006, t) - sol_pc.temperature_at(0.006, t)
    dT8 = sol.temperature_at(0.008, t) - sol_pc.temperature_at(0.008, t)
    print(f"{t:6.1f}  {dT2:10.4f}  {dT4:10.4f}  {dT6:10.4f}  {dT8:10.4f}")

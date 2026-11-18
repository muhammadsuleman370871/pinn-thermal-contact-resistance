#!/usr/bin/env python3
"""
Publication figures for the main paper and supplementary material.

Produces:
  Main paper:   Fig_01.png … Fig_16.png
  Supplementary: Fig_S1.png … Fig_S3.png

Font sizes: base 12 pt, axis labels 14 pt, legends 11 pt, ticks 12 pt.
All figures at 500 DPI following Elsevier artwork guidelines.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.patches as mpatches
import torch
import time

from utils.physical_params import ProblemConfig, default_config
from solvers.finite_volume import solve_fv
from pinn.networks import ContactResistancePINN
from pinn.trainer import (TrainingConfig, TrainingHistory, train_forward,
                           train_inverse, sample_collocation_points)
from data.synthetic import generate_observations, SensorConfig
from analysis.sensitivity import compute_sensitivity, heating_waveform_comparison
from analysis.conventional_inverse import inverse_lm, multi_start_inverse

# ---- Output directories ----
_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_MAIN = os.path.join(_ROOT, "figures")
OUT_SUPP = os.path.join(_ROOT, "supplementary_figures")
os.makedirs(OUT_MAIN, exist_ok=True)
os.makedirs(OUT_SUPP, exist_ok=True)

# ---- Global style: 500 DPI, 12pt base, bold axes ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.grid': False,
    'grid.alpha': 0.0,
    'lines.linewidth': 1.8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

DPI = 500

# Known ANSYS values at t=10s (from v5 run with KBC,1)
ANSYS_DATA = {
    'T_2mm': 317.881, 'T_4mm': 317.503, 'T_intL': 317.341,
    'T_6mm': 315.579, 'T_8mm': 313.209,
}


# ========================================================================
# Fig. 1: Problem schematic
# ========================================================================
def fig01_problem_schematic(config):
    print("  Fig. 1: Problem schematic...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-2.2, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layer 1 (Aluminum)
    rect1 = FancyBboxPatch((0, 0), 5, 4, boxstyle="round,pad=0.05",
                            facecolor='#4ECDC4', edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(rect1)
    ax.text(2.5, 2, 'Layer 1\nAluminum\n$k_1$ = 205 W/(m K)\n$\\rho_1 c_1$ = 2.43 MJ/(m$^3$ K)',
            ha='center', va='center', fontsize=12, weight='bold')

    # Layer 2 (SS304)
    rect2 = FancyBboxPatch((5.3, 0), 5, 4, boxstyle="round,pad=0.05",
                            facecolor='#FF6B6B', edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(rect2)
    ax.text(7.8, 2, 'Layer 2\nSS 304\n$k_2$ = 16 W/(m K)\n$\\rho_2 c_2$ = 4.00 MJ/(m$^3$ K)',
            ha='center', va='center', fontsize=12, weight='bold')

    # Interface with TCR — moved up to avoid sensor-label collision
    ax.plot([5.15, 5.15], [-0.3, 4.3], 'k--', linewidth=2)
    ax.annotate('Interface\n$R_c$ = 10$^{-4}$ m$^2$ K/W',
                xy=(5.15, 4.3), xytext=(5.15, 5.6), ha='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Left BC: heat flux
    for y in [1, 2, 3]:
        ax.annotate('', xy=(0, y), xytext=(-0.8, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(-0.9, 2, '$q_{in}$\n= 50 kW/m$^2$', ha='right', va='center',
            fontsize=12, color='red', weight='bold')

    # Right BC: convection
    for y in [1, 2, 3]:
        ax.annotate('', xy=(11.1, y), xytext=(10.3, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(11.3, 2, '$h_R$ = 50 W/(m$^2$ K)\n$T_\\infty$ = 300 K',
            ha='left', va='center', fontsize=12, color='blue', weight='bold')

    # Dimension labels — stacked with more vertical clearance
    ax.annotate('', xy=(0, -0.5), xytext=(5.15, -0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(2.575, -0.85, '$a$ = 5 mm', ha='center', fontsize=12)

    ax.annotate('', xy=(0, -1.5), xytext=(10.3, -1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(5.15, -1.85, '$L$ = 10 mm', ha='center', fontsize=12)

    # x-axis labels
    ax.text(0, -0.15, '$x = 0$', ha='center', va='top', fontsize=12)
    ax.text(5.15, -0.15, '$x = a$', ha='center', va='top', fontsize=12)
    ax.text(10.3, -0.15, '$x = L$', ha='center', va='top', fontsize=12)

    # Sensor markers — labels below markers to avoid interface box
    sensor_positions = {2: 0.002, 4: 0.004, 6: 0.006, 8: 0.008}
    for x_mm, x_m in sensor_positions.items():
        x_plot = x_mm / 10 * 10.3
        ax.plot(x_plot, 4.0, 'v', color='darkgreen', markersize=12)
        ax.text(x_plot, 4.45, f'{x_mm} mm', ha='center', fontsize=12, color='darkgreen')

    fig.savefig(f'{OUT_MAIN}/Fig_01.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 2: PINN Architecture Diagram
# ========================================================================
def fig02_pinn_architecture():
    print("  Fig. 2: PINN architecture diagram...")
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ---- Section 1: Neural Network Architecture (left) ----
    ax.text(3.5, 8.5, 'DOMAIN-DECOMPOSED NEURAL NETWORK', fontsize=13,
            ha='center', weight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#2c3e50'))

    # Input layer
    ax.text(0.5, 7.7, 'Input\nLayer', ha='center', fontsize=12, weight='bold')
    for i, (label, y) in enumerate(zip(['$x$', '$t$'], [7.2, 6.5])):
        circle = Circle((0.5, y), 0.2, facecolor='#3498db', edgecolor='black', lw=1.5)
        ax.add_patch(circle)
        ax.text(0.5, y, label, ha='center', va='center', fontsize=12, color='white', weight='bold')

    # Hidden layers for T1_net
    ax.text(3.0, 8.0, '$T_1$ Network (Layer 1: $0 < x < a$)', ha='center',
            fontsize=12, style='italic', color='#2980b9')
    colors_h = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085']
    for h, (x_pos, col) in enumerate(zip([1.5, 2.5, 3.5, 4.5], colors_h)):
        ax.text(x_pos, 7.85, f'{64}', ha='center', fontsize=9, color='gray')
        for j in range(5):
            y_pos = 6.3 + j * 0.35
            circle = Circle((x_pos, y_pos), 0.12, facecolor=col, edgecolor='black', lw=0.5, alpha=0.8)
            ax.add_patch(circle)
        ax.text(x_pos, 6.0, '...', ha='center', fontsize=12, color='gray')

    # Draw connections (simplified)
    for y_in in [7.2, 6.5]:
        for y_h in [6.3 + j * 0.35 for j in range(5)]:
            ax.plot([0.7, 1.38], [y_in, y_h], 'gray', alpha=0.15, lw=0.5)

    # Output T1
    circle_out1 = Circle((5.5, 6.85), 0.25, facecolor='#e74c3c', edgecolor='black', lw=1.5)
    ax.add_patch(circle_out1)
    ax.text(5.5, 6.85, '$T_1$', ha='center', va='center', fontsize=12, color='white', weight='bold')

    # T2 network (below)
    ax.text(3.0, 5.3, '$T_2$ Network (Layer 2: $a < x < L$)', ha='center',
            fontsize=12, style='italic', color='#c0392b')
    colors_h2 = ['#e67e22', '#d35400', '#f39c12', '#e74c3c']
    for h, (x_pos, col) in enumerate(zip([1.5, 2.5, 3.5, 4.5], colors_h2)):
        ax.text(x_pos, 5.15, f'{64}', ha='center', fontsize=9, color='gray')
        for j in range(5):
            y_pos = 3.6 + j * 0.35
            circle = Circle((x_pos, y_pos), 0.12, facecolor=col, edgecolor='black', lw=0.5, alpha=0.8)
            ax.add_patch(circle)
        ax.text(x_pos, 3.3, '...', ha='center', fontsize=12, color='gray')

    # Output T2
    circle_out2 = Circle((5.5, 4.3), 0.25, facecolor='#e74c3c', edgecolor='black', lw=1.5)
    ax.add_patch(circle_out2)
    ax.text(5.5, 4.3, '$T_2$', ha='center', va='center', fontsize=12, color='white', weight='bold')

    # Trainable Rc parameter
    rect_rc = FancyBboxPatch((4.8, 2.2), 1.4, 0.7, boxstyle="round,pad=0.1",
                              facecolor='#f1c40f', edgecolor='black', lw=1.5)
    ax.add_patch(rect_rc)
    ax.text(5.5, 2.55, '$\\theta = \\log(R_c)$\n(trainable)', ha='center', va='center',
            fontsize=12, weight='bold')

    # Hidden layer label
    ax.text(3.0, 5.55, '4 Hidden Layers x 64 neurons (tanh)', ha='center',
            fontsize=12, color='gray',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='gray', alpha=0.7))

    # ---- Section 2: Automatic Differentiation (middle) ----
    rect_ad = FancyBboxPatch((6.3, 4.5), 2.4, 3.5, boxstyle="round,pad=0.1",
                              facecolor='#dfe6e9', edgecolor='#2c3e50', lw=1.5)
    ax.add_patch(rect_ad)
    ax.text(7.5, 7.7, 'AUTOMATIC\nDIFFERENTIATION', ha='center', fontsize=12,
            weight='bold', color='#2c3e50')

    derivs = [
        ('$\\partial T_i / \\partial t$', 6.9),
        ('$\\partial T_i / \\partial x$', 6.3),
        ('$\\partial^2 T_i / \\partial x^2$', 5.7),
    ]
    for txt, y in derivs:
        ax.text(7.5, y, txt, ha='center', fontsize=12, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#95a5a6', alpha=0.9))

    # Arrow from NN to AD
    ax.annotate('', xy=(6.3, 6.85), xytext=(5.75, 6.85),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(6.3, 5.0), xytext=(5.75, 4.3),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # ---- Section 3: PDE Residual / Physics (right-top) ----
    rect_pde = FancyBboxPatch((9.2, 5.5), 3.5, 2.8, boxstyle="round,pad=0.1",
                               facecolor='#e8f8f5', edgecolor='#1abc9c', lw=2)
    ax.add_patch(rect_pde)
    ax.text(10.95, 8.0, 'PHYSICS-INFORMED\nRESIDUAL', ha='center', fontsize=12,
            weight='bold', color='#1abc9c')

    pde_lines = [
        'Governing Equation:',
        '$\\rho_i c_i \\frac{\\partial T_i}{\\partial t} = k_i \\frac{\\partial^2 T_i}{\\partial x^2}$',
        '',
        'Interface Conditions:',
        '$-k_1 \\partial T_1/\\partial x|_a = -k_2 \\partial T_2/\\partial x|_a$',
        '$T_1(a) - T_2(a) = R_c \\cdot q_{int}$',
    ]
    y_start = 7.6
    for i, line in enumerate(pde_lines):
        if line:
            fs = 12 if 'Equation' in line or 'Conditions' in line else 10
            ax.text(10.95, y_start - i * 0.33, line, ha='center', fontsize=fs,
                    color='#2c3e50')

    # Arrow from AD to PDE
    ax.annotate('', xy=(9.2, 6.5), xytext=(8.7, 6.5),
                arrowprops=dict(arrowstyle='->', color='#1abc9c', lw=2))

    # ---- Section 4: Loss Function (bottom-center) ----
    rect_loss = FancyBboxPatch((6.0, 0.3), 5.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#fef9e7', edgecolor='#f39c12', lw=2)
    ax.add_patch(rect_loss)
    ax.text(8.75, 2.55, 'COMPOSITE LOSS FUNCTION', ha='center', fontsize=12,
            weight='bold', color='#e67e22')

    loss_text = (
        '$\\mathcal{L}_{total} = \\lambda_{PDE} \\mathcal{L}_{PDE} + '
        '\\lambda_{BC} \\mathcal{L}_{BC} + \\lambda_{IC} \\mathcal{L}_{IC}$\n'
        '$+ \\lambda_{flux} \\mathcal{L}_{flux} + '
        '\\lambda_{jump} \\mathcal{L}_{jump} + '
        '\\lambda_{data} \\mathcal{L}_{data}$'
    )
    ax.text(8.75, 1.55, loss_text, ha='center', va='center', fontsize=11, color='#2c3e50')

    ax.text(8.75, 0.65, '(6 loss terms: PDE, BC, IC, flux continuity, temperature jump, data fidelity)',
            ha='center', fontsize=9, color='gray', style='italic')

    # Arrow from PDE to Loss
    ax.annotate('', xy=(10.95, 3.0), xytext=(10.95, 5.5),
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))

    # ---- Section 5: Optimization (right) ----
    rect_opt = FancyBboxPatch((12.5, 0.3), 3.0, 2.5, boxstyle="round,pad=0.1",
                               facecolor='#fadbd8', edgecolor='#e74c3c', lw=2)
    ax.add_patch(rect_opt)
    ax.text(14.0, 2.55, 'OPTIMIZATION', ha='center', fontsize=12,
            weight='bold', color='#c0392b')
    ax.text(14.0, 1.75, 'Adam Optimizer\n+ StepLR Scheduler', ha='center',
            fontsize=12, color='#2c3e50')
    ax.text(14.0, 0.95, 'lr = 10$^{-3}$, decay = 0.5\nevery 3000 epochs',
            ha='center', fontsize=10, color='gray')

    # Arrow from Loss to Optimizer
    ax.annotate('', xy=(12.5, 1.55), xytext=(11.5, 1.55),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    # Backpropagation arrow
    ax.annotate('', xy=(0.5, 2.5), xytext=(12.5, 0.3),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2.5,
                               connectionstyle='arc3,rad=0.2', linestyle='--'))
    ax.text(6.5, 0.0, 'Backpropagation of Total Loss',
            ha='center', fontsize=12, weight='bold', color='#9b59b6')

    fig.savefig(f'{OUT_MAIN}/Fig_02.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 3: FV mesh convergence
# ========================================================================
def fig03_fv_mesh_convergence(config):
    print("  Fig. 3: FV mesh convergence...")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    N_list = [25, 50, 100, 200]
    results = {'N': [], 'T_2mm': [], 'T_8mm': [], 'jump': []}

    for N in N_list:
        sol = solve_fv(config, N1=N, N2=N, Nt=N*10)
        T1_int, T2_int = sol.interface_temperatures()
        results['N'].append(N)
        results['T_2mm'].append(sol.temperature_at(0.002, config.t_end))
        results['T_8mm'].append(sol.temperature_at(0.008, config.t_end))
        results['jump'].append(T1_int[-1] - T2_int[-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) Temperature at x = 2 mm
    axes[0].plot(results['N'], results['T_2mm'], 'bo-', lw=2, markersize=9)
    axes[0].set_xlabel('Cells per layer, $N$')
    axes[0].set_ylabel('$T$ at $x$ = 2 mm, $t$ = 10 s (K)')
    axes[0].set_title('(a) Sensor at 2 mm', fontsize=13)

    # (b) Temperature at x = 8 mm
    axes[1].plot(results['N'], results['T_8mm'], 'rs-', lw=2, markersize=9)
    axes[1].set_xlabel('Cells per layer, $N$')
    axes[1].set_ylabel('$T$ at $x$ = 8 mm, $t$ = 10 s (K)')
    axes[1].set_title('(b) Sensor at 8 mm', fontsize=13)

    # (c) Interface temperature jump
    axes[2].plot(results['N'], results['jump'], 'g^-', lw=2, markersize=9)
    axes[2].set_xlabel('Cells per layer, $N$')
    axes[2].set_ylabel('$\\Delta T$ at interface (K)')
    axes[2].set_title('(c) Interface jump', fontsize=13)

    # Set explicit x-ticks on all subplots — no log scale, just the 4 data points
    for ax in axes:
        ax.set_xticks(N_list)
        ax.set_xticklabels([str(n) for n in N_list])
        ax.set_xlim([15, 215])
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_03.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 4: FV reference temperature profiles
# ========================================================================
def fig04_fv_reference_profiles(sol_ref, config):
    print("  Fig. 4: FV reference temperature profiles...")
    times = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(times)))

    for i, t_val in enumerate(times):
        it = np.argmin(np.abs(sol_ref.t - t_val))
        ax.plot(sol_ref.x * 1e3, sol_ref.T[it, :], '-', color=colors[i],
                linewidth=2, label=f'$t$ = {t_val:.0f} s')

    ax.axvline(x=config.a * 1e3, color='gray', linestyle='--', alpha=0.7, label='Interface')
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Temperature $T$ [K]')
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlim([0, config.L * 1e3])
    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_04.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 5: ANSYS vs FV comparison (perfect contact)
# ========================================================================
def fig05_ansys_fv_comparison(config):
    print("  Fig. 5: ANSYS vs FV comparison (perfect contact)...")
    cfg_pc = ProblemConfig(Rc=1e-12)
    sol_pc = solve_fv(cfg_pc, N1=100, N2=100, Nt=1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Spatial profiles at t=10s
    ax = axes[0]
    ax.plot(sol_pc.x * 1e3, sol_pc.T[-1, :], 'b-', lw=2, label='FV ($N$=200)')
    ansys_x = [2, 4, 5, 6, 8]
    ansys_T = [ANSYS_DATA['T_2mm'], ANSYS_DATA['T_4mm'], ANSYS_DATA['T_intL'],
               ANSYS_DATA['T_6mm'], ANSYS_DATA['T_8mm']]
    ax.plot(ansys_x, ansys_T, 'rs', markersize=10, label='ANSYS APDL', zorder=5)
    ax.axvline(x=cfg_pc.a * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Temperature $T$ [K]')
    ax.legend()

    # (b) Difference bar chart
    ax = axes[1]
    fv_T = [sol_pc.temperature_at(x*1e-3, cfg_pc.t_end) for x in [2, 4, 6, 8]]
    ansys_T_pts = [ANSYS_DATA['T_2mm'], ANSYS_DATA['T_4mm'],
                   ANSYS_DATA['T_6mm'], ANSYS_DATA['T_8mm']]
    diff = [a - f for a, f in zip(ansys_T_pts, fv_T)]

    x_pos = np.arange(4)
    labels = ['2 mm', '4 mm', '6 mm', '8 mm']
    bars = ax.bar(x_pos, diff, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], width=0.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sensor position')
    ax.set_ylabel('$T_{ANSYS} - T_{FV}$ [K]')
    ax.axhline(y=0, color='black', lw=0.5)
    for i, d in enumerate(diff):
        # Place labels just inside the bar near the zero line
        ax.text(i, d * 0.08, f'{d:.4f} K', ha='center', va='top',
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.0))

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_05.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 6: ANSYS FV imperfect contact profiles (was fig07b)
# ========================================================================
def fig06_imperfect_profiles(config):
    print("  Fig. 6: ANSYS FV imperfect contact profiles...")
    sol_100 = solve_fv(config, N1=100, N2=100, Nt=2000, verbose=False)
    sol_400 = solve_fv(config, N1=400, N2=400, Nt=8000, verbose=False)

    snap_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    x_ansys_mm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.9, 5.1, 6.0, 7.0, 8.0, 9.0, 10.0])
    x_ansys_m = x_ansys_mm * 1e-3

    fig = plt.figure(figsize=(15, 9))
    ax_positions = [
        fig.add_subplot(2, 3, 1),
        fig.add_subplot(2, 3, 2),
        fig.add_subplot(2, 3, 3),
    ]
    ax_positions.append(fig.add_axes([0.125, 0.06, 0.35, 0.38]))
    ax_positions.append(fig.add_axes([0.55, 0.06, 0.35, 0.38]))

    for idx, (ax, t_snap) in enumerate(zip(ax_positions, snap_times)):
        t_idx_100 = np.argmin(np.abs(sol_100.t - t_snap))
        x_mm_100 = sol_100.x * 1000
        T_100 = sol_100.T[t_idx_100, :]
        T_ansys = np.array([sol_400.temperature_at(xm, t_snap) for xm in x_ansys_m])

        ax.plot(x_mm_100, T_100, 'b-', linewidth=1.8,
                label='FV (100 cells/layer)' if idx == 0 else None)
        ax.plot(x_ansys_mm, T_ansys, 'ro', markersize=7, markeredgecolor='red',
                label='ANSYS (FEM)' if idx == 0 else None)
        ax.axvline(x=5.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        # Place layer labels ABOVE the plot frame so they never touch any data
        ax.text(0.22, 1.03, 'Al', fontsize=13, color='blue', ha='center', va='bottom',
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.78, 1.03, 'SS304', fontsize=13, color='red', ha='center', va='bottom',
                fontweight='bold', transform=ax.transAxes)

        ax.set_xlabel('$x$ (mm)')
        ax.set_ylabel('$T$ (K)')

    ax_positions[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.subplots_adjust(top=0.97, hspace=0.35, wspace=0.35)
    fig.savefig(f'{OUT_MAIN}/Fig_06.png', dpi=DPI, bbox_inches='tight')
    plt.close()


# ========================================================================
# Fig. 7: Interface jump comparison (was fig07e)
# ========================================================================
def fig07_interface_jump(config):
    print("  Fig. 7: Interface jump comparison...")
    sol_100 = solve_fv(config, N1=100, N2=100, Nt=2000, verbose=False)
    sol_400 = solve_fv(config, N1=400, N2=400, Nt=8000, verbose=False)

    T1_int_100, T2_int_100 = sol_100.interface_temperatures()
    DT_100 = T1_int_100 - T2_int_100
    T1_int_400, T2_int_400 = sol_400.interface_temperatures()
    DT_400 = T1_int_400 - T2_int_400
    DT_400_interp = np.interp(sol_100.t, sol_400.t, DT_400)
    jump_diff = DT_400_interp - DT_100

    q_ss = config.q_in_amplitude
    DT_ss = q_ss * config.Rc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(sol_100.t, DT_100, 'b-', linewidth=1.8, label='FV (100 cells/layer)')
    ax1.plot(sol_400.t[::20], DT_400[::20], 'r--', linewidth=1.5, alpha=0.8,
             label='ANSYS (thin layer)')
    ax1.axhline(y=DT_ss, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
    ax1.text(6.5, DT_ss - 0.15, f'Steady-state\n$q \\cdot R_c = {DT_ss:.1f}$ K',
             fontsize=12, color='gray', ha='center', va='top')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'$\Delta T_{\mathrm{int}}$ (K)')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, DT_ss + 0.8])

    ax2.plot(sol_100.t, jump_diff, 'g-', linewidth=1.5)
    ax2.fill_between(sol_100.t, jump_diff, alpha=0.15, color='green')
    max_diff = np.max(np.abs(jump_diff))
    ax2.annotate(f'max |diff| = {max_diff:.3f} K',
                 xy=(0.97, 0.97), xycoords='axes fraction',
                 ha='right', va='top', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='gray', alpha=0.9))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'$\Delta T_{\mathrm{int}}^{\mathrm{ANSYS}} - \Delta T_{\mathrm{int}}^{\mathrm{FV}}$ (K)')
    ax2.set_xlim([0, 10])

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_07.png', dpi=DPI)
    plt.close()


# ========================================================================
# Fig. 8: TCR effect on temperature field
# ========================================================================
def fig08_tcr_effect(config):
    print("  Fig. 8: TCR effect on temperature field...")
    cfg_pc = ProblemConfig(Rc=1e-12)
    sol_pc = solve_fv(cfg_pc, N1=100, N2=100, Nt=1000)
    sol_tcr = solve_fv(config, N1=100, N2=100, Nt=1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(sol_pc.x * 1e3, sol_pc.T[-1, :], 'b-', lw=2, label='Perfect contact')
    ax.plot(sol_tcr.x * 1e3, sol_tcr.T[-1, :], 'r-', lw=2,
            label=f'$R_c$ = {config.Rc:.0e} m$^2$ K/W')
    ax.axvline(x=config.a * 1e3, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Temperature $T$ [K]')
    ax.legend()

    ax = axes[1]
    ax.grid(False)
    X_tcr, T_grid_tcr = np.meshgrid(sol_tcr.x * 1e3, sol_tcr.t)
    T_diff = sol_tcr.T - sol_pc.T
    cs = ax.contourf(X_tcr, T_grid_tcr, T_diff, levels=20, cmap='RdBu_r')
    fig.colorbar(cs, ax=ax, label='$\\Delta T$ (TCR $-$ perfect) [K]')
    ax.axvline(x=config.a * 1e3, color='black', linestyle='--', lw=1.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Time $t$ [s]')

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_08.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 9: Forward PINN vs FV comparison (was fig11)
# ========================================================================
def fig09_forward_comparison(fwd_model, sol_ref, config):
    print("  Fig. 9: Forward PINN vs FV comparison...")
    times = [2.0, 5.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_plot = np.linspace(0, config.L, 300)

    for j, t_val in enumerate(times):
        ax = axes[j]
        it = np.argmin(np.abs(sol_ref.t - t_val))
        ax.plot(sol_ref.x * 1e3, sol_ref.T[it, :], 'b-', lw=2, label='FV reference')
        t_arr = np.full_like(x_plot, t_val)
        T_pinn = fwd_model.predict(x_plot, t_arr)
        ax.plot(x_plot * 1e3, T_pinn, 'r--', lw=1.8, label='PINN')
        ax.axvline(x=config.a * 1e3, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Position $x$ [mm]')
        ax.set_ylabel('Temperature $T$ [K]')
        ax.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_09.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 10: Forward PINN error field (was fig12)
# ========================================================================
def fig10_forward_error(fwd_model, sol_ref, config):
    print("  Fig. 10: Forward PINN error field...")
    x_plot = np.linspace(0, config.L, 300)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Error at t=10s
    ax = axes[0]
    t_val = config.t_end
    t_arr = np.full_like(x_plot, t_val)
    T_pinn = fwd_model.predict(x_plot, t_arr)
    T_fv = np.array([sol_ref.temperature_at(xi, t_val) for xi in x_plot])
    error = np.abs(T_pinn - T_fv)

    ax.plot(x_plot * 1e3, error, 'k-', lw=1.8)
    ax.axvline(x=config.a * 1e3, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('$|T_{PINN} - T_{FV}|$ [K]', labelpad=10)
    ax.text(0.95, 0.95, f'Max error: {error.max():.4f} K\nMean error: {error.mean():.4f} K',
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) Error contour
    ax = axes[1]
    ax.grid(False)
    n_t = 50
    t_grid = np.linspace(0.5, config.t_end, n_t)
    error_field = np.zeros((n_t, len(x_plot)))
    for i, t_val in enumerate(t_grid):
        T_pinn_i = fwd_model.predict(x_plot, np.full_like(x_plot, t_val))
        T_fv_i = np.array([sol_ref.temperature_at(xi, t_val) for xi in x_plot])
        error_field[i, :] = np.abs(T_pinn_i - T_fv_i)

    X_err, T_err = np.meshgrid(x_plot * 1e3, t_grid)
    cs = ax.contourf(X_err, T_err, error_field, levels=20, cmap='YlOrRd')
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label('$|T_{PINN} - T_{FV}|$ [K]', labelpad=15)
    ax.axvline(x=config.a * 1e3, color='cyan', ls='--', lw=1.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Time $t$ [s]')

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_10.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 11: Inverse PINN recovered field (was fig14)
# ========================================================================
def fig11_inverse_comparison(inv_model, sol_ref, obs_data, config):
    print("  Fig. 11: Inverse PINN recovered field...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_plot = np.linspace(0, config.L, 300)

    # (a) Spatial profile at t=10s
    ax = axes[0]
    T_pinn = inv_model.predict(x_plot, np.full_like(x_plot, config.t_end))
    it = np.argmin(np.abs(sol_ref.t - config.t_end))
    ax.plot(sol_ref.x * 1e3, sol_ref.T[it, :], 'b-', lw=2, label='FV reference')
    ax.plot(x_plot * 1e3, T_pinn, 'r--', lw=1.8, label='Inverse PINN')
    ax.axvline(x=config.a * 1e3, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Position $x$ [mm]')
    ax.set_ylabel('Temperature $T$ [K]')
    ax.legend()

    # (b) Sensor time histories
    ax = axes[1]
    x_sensors = obs_data['x_sensors']
    n_times = len(obs_data['t']) // len(x_sensors)
    colors_s = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, xs in enumerate(x_sensors):
        t_i = obs_data['t'][i*n_times:(i+1)*n_times].numpy().flatten()
        T_obs_i = obs_data['T'][i*n_times:(i+1)*n_times].numpy().flatten()
        T_pinn_i = inv_model.predict(np.full(len(t_i), xs), t_i)

        layer = 'L1' if xs < config.a else 'L2'
        ax.plot(t_i, T_obs_i, 'o', color=colors_s[i], markersize=4, alpha=0.4)
        ax.plot(t_i, T_pinn_i, '-', color=colors_s[i], lw=1.8,
                label=f'$x$ = {xs*1e3:.0f} mm ({layer})')

    ax.set_xlabel('Time $t$ [s]')
    ax.set_ylabel('Temperature $T$ [K]')
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_11.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 12: Spatiotemporal contour (was fig17)
# ========================================================================
def fig12_spatiotemporal_contour(sol_ref, config):
    print("  Fig. 12: Spatiotemporal temperature contour...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid(False)
    X, T_grid = np.meshgrid(sol_ref.x * 1e3, sol_ref.t)
    levels = np.linspace(config.T_inf, np.max(sol_ref.T), 30)
    cf = ax.contourf(X, T_grid, sol_ref.T, levels=levels, cmap='hot')
    plt.colorbar(cf, ax=ax, label='Temperature (K)')
    ax.axvline(x=config.a * 1e3, color='white', linestyle='--', linewidth=2,
               label='Interface ($x = a$)')
    ax.set_xlabel('Position $x$ (mm)')
    ax.set_ylabel('Time $t$ (s)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_12.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 13: Layer contours (was fig18)
# ========================================================================
def fig13_layer_contours(sol_ref, config):
    print("  Fig. 13: Layer-specific contours...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    i_int = sol_ref.N1

    # Layer 1 (Aluminum)
    X1, T1_time = np.meshgrid(sol_ref.x[:i_int+1] * 1e3, sol_ref.t)
    T_layer1 = sol_ref.T[:, :i_int+1]
    levels1 = np.linspace(config.T_inf, np.max(T_layer1), 25)
    cf1 = ax1.contourf(X1, T1_time, T_layer1, levels=levels1, cmap='YlOrRd')
    plt.colorbar(cf1, ax=ax1, label='$T$ (K)')
    ax1.set_xlabel('Position $x$ (mm)')
    ax1.set_ylabel('Time $t$ (s)')

    # Layer 2 (Steel)
    X2, T2_time = np.meshgrid(sol_ref.x[i_int:] * 1e3, sol_ref.t)
    T_layer2 = sol_ref.T[:, i_int:]
    levels2 = np.linspace(config.T_inf, np.max(T_layer2), 25)
    cf2 = ax2.contourf(X2, T2_time, T_layer2, levels=levels2, cmap='YlGnBu')
    plt.colorbar(cf2, ax=ax2, label='$T$ (K)')
    ax2.set_xlabel('Position $x$ (mm)')
    ax2.set_ylabel('Time $t$ (s)')

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_13.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 14: Heat flux contour (was fig19)
# ========================================================================
def fig14_heatflux_contour(sol_ref, config):
    print("  Fig. 14: Heat flux contour...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid(False)

    N1 = sol_ref.N1
    N2 = sol_ref.N2
    dx1 = config.a / N1
    dx2 = (config.L - config.a) / N2

    q_field = np.zeros_like(sol_ref.T)
    for ti in range(len(sol_ref.t)):
        # Left boundary: applied heat flux
        q_field[ti, 0] = config.q_in_amplitude

        # Layer 1 interior (indices 1 .. N1-2)
        for xi in range(1, N1 - 1):
            q_field[ti, xi] = -config.mat1.k * (sol_ref.T[ti, xi+1] - sol_ref.T[ti, xi-1]) / (2 * dx1)

        # Layer 1 last cell (one-sided toward interface)
        q_field[ti, N1-1] = -config.mat1.k * (sol_ref.T[ti, N1-1] - sol_ref.T[ti, N1-2]) / dx1

        # Layer 2 first cell (one-sided away from interface)
        q_field[ti, N1] = -config.mat2.k * (sol_ref.T[ti, N1+1] - sol_ref.T[ti, N1]) / dx2

        # Layer 2 interior (indices N1+1 .. N1+N2-2)
        for xi in range(N1 + 1, N1 + N2 - 1):
            q_field[ti, xi] = -config.mat2.k * (sol_ref.T[ti, xi+1] - sol_ref.T[ti, xi-1]) / (2 * dx2)

        # Right boundary: convection
        q_field[ti, -1] = config.h_R * (sol_ref.T[ti, -1] - config.T_inf)

    q_kw = q_field / 1000
    X, T_grid = np.meshgrid(sol_ref.x * 1e3, sol_ref.t)
    levels_q = np.linspace(np.min(q_kw), np.max(q_kw), 30)
    cf = ax.contourf(X, T_grid, q_kw, levels=levels_q, cmap='coolwarm')
    plt.colorbar(cf, ax=ax, label='Heat Flux (kW/m$^2$)')
    ax.axvline(x=config.a * 1e3, color='black', linestyle='--', linewidth=2, label='Interface')
    ax.set_xlabel('Position $x$ (mm)')
    ax.set_ylabel('Time $t$ (s)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_14.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 15: Temperature gradient contour (was fig20)
# ========================================================================
def fig15_gradient_contour(sol_ref, config):
    print("  Fig. 15: Temperature gradient contour...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    dx = sol_ref.x[1] - sol_ref.x[0]
    grad_T = np.zeros_like(sol_ref.T)
    for ti in range(len(sol_ref.t)):
        for xi in range(1, len(sol_ref.x) - 1):
            grad_T[ti, xi] = abs(sol_ref.T[ti, xi+1] - sol_ref.T[ti, xi-1]) / (2 * dx)
        grad_T[ti, 0] = abs(sol_ref.T[ti, 1] - sol_ref.T[ti, 0]) / dx
        grad_T[ti, -1] = abs(sol_ref.T[ti, -1] - sol_ref.T[ti, -2]) / dx

    grad_km = grad_T / 1000
    X, T_grid = np.meshgrid(sol_ref.x * 1e3, sol_ref.t)
    levels_g = np.linspace(0, np.percentile(grad_km, 99), 25)
    cf = ax.contourf(X, T_grid, grad_km, levels=levels_g, cmap='inferno')
    plt.colorbar(cf, ax=ax, label='$|dT/dx|$ (K/mm)')
    ax.axvline(x=config.a * 1e3, color='white', linestyle='--', linewidth=2, label='Interface')
    ax.set_xlabel('Position $x$ (mm)')
    ax.set_ylabel('Time $t$ (s)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_15.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. 16: Energy balance verification
# ========================================================================
def fig16_energy_balance(sol_ref, config):
    print("  Fig. 16: Energy balance verification...")
    dx1 = config.a / sol_ref.N1
    dx2 = (config.L - config.a) / sol_ref.N2

    Q_in_cum, Q_out_cum, E_stored_cum = [], [], []
    for it in range(len(sol_ref.t)):
        q_in_arr = config.q_in_array(sol_ref.t[:it+1])
        Q_in = np.trapezoid(q_in_arr, sol_ref.t[:it+1]) if it > 0 else 0
        T_right = sol_ref.T[:it+1, -1]
        q_out_arr = config.h_R * (T_right - config.T_inf)
        Q_out = np.trapezoid(q_out_arr, sol_ref.t[:it+1]) if it > 0 else 0
        E1 = config.mat1.rho * config.mat1.cp * dx1 * np.sum(sol_ref.T[it, :sol_ref.N1] - sol_ref.T[0, :sol_ref.N1])
        E2 = config.mat2.rho * config.mat2.cp * dx2 * np.sum(sol_ref.T[it, sol_ref.N1:] - sol_ref.T[0, sol_ref.N1:])
        Q_in_cum.append(Q_in)
        Q_out_cum.append(Q_out)
        E_stored_cum.append(E1 + E2)

    Q_in_cum = np.array(Q_in_cum)
    Q_out_cum = np.array(Q_out_cum)
    E_stored_cum = np.array(E_stored_cum)
    imbalance = Q_in_cum - Q_out_cum - E_stored_cum

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(sol_ref.t, Q_in_cum, 'r-', lw=2, label='$Q_{in}$ (cumulative)')
    ax1.plot(sol_ref.t, Q_out_cum, 'b-', lw=2, label='$Q_{out}$ (cumulative)')
    ax1.plot(sol_ref.t, E_stored_cum, 'g-', lw=2, label='$E_{stored}$')
    ax1.set_ylabel('Energy [J/m$^2$]')
    ax1.legend()

    ax2.plot(sol_ref.t, imbalance, 'k-', lw=1.8)
    ax2.set_xlabel('Time $t$ [s]')
    ax2.set_ylabel('Imbalance [J/m$^2$]')
    final_imb_pct = abs(imbalance[-1]) / max(abs(Q_in_cum[-1]), 1e-30) * 100
    ax2.text(0.95, 0.95, f'Final imbalance: {final_imb_pct:.4f}%',
             transform=ax2.transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.tight_layout()
    fig.savefig(f'{OUT_MAIN}/Fig_16.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. S1: Forward PINN training convergence (was fig10)
# ========================================================================
def figS1_forward_training(fwd_model, fwd_history, config):
    print("  Fig. S1: Forward PINN training convergence...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.semilogy(fwd_history.epochs, fwd_history.total_loss, 'k-', lw=1.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')

    ax = axes[0, 1]
    ax.semilogy(fwd_history.epochs, fwd_history.pde_loss, label='PDE', lw=1.4)
    ax.semilogy(fwd_history.epochs, fwd_history.bc_loss, label='BC', lw=1.4)
    ax.semilogy(fwd_history.epochs, fwd_history.ic_loss, label='IC', lw=1.4)
    ax.semilogy(fwd_history.epochs, fwd_history.flux_loss, label='Flux', lw=1.4)
    ax.semilogy(fwd_history.epochs, fwd_history.jump_loss, label='Jump', lw=1.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(fwd_history.epochs, fwd_history.wall_time, 'g-', lw=1.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wall time [s]')

    ax = axes[1, 1]
    ax.plot(fwd_history.epochs, fwd_history.Rc_history, 'b-', lw=1.8)
    ax.axhline(y=config.Rc, color='r', ls='--', label=f'True $R_c$ = {config.Rc:.1e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$R_c$ [m$^2$ K/W]')
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT_SUPP}/Fig_S1.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. S2: Inverse PINN training convergence (was fig13)
# ========================================================================
def figS2_inverse_training(inv_history, config):
    print("  Fig. S2: Inverse PINN training convergence...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.semilogy(inv_history.epochs, inv_history.total_loss, 'k-', lw=1.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')

    ax = axes[0, 1]
    ax.semilogy(inv_history.epochs, inv_history.pde_loss, label='PDE')
    ax.semilogy(inv_history.epochs, inv_history.bc_loss, label='BC')
    ax.semilogy(inv_history.epochs, inv_history.ic_loss, label='IC')
    ax.semilogy(inv_history.epochs, inv_history.flux_loss, label='Flux')
    ax.semilogy(inv_history.epochs, inv_history.jump_loss, label='Jump')
    ax.semilogy(inv_history.epochs, inv_history.data_loss, label='Data')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(inv_history.epochs, inv_history.Rc_history, 'b-', lw=2, label='Estimated $R_c$')
    ax.axhline(y=config.Rc, color='r', ls='--', lw=2, label=f'True $R_c$ = {config.Rc:.1e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$R_c$ [m$^2$ K/W]')
    ax.legend()

    Rc_final = inv_history.Rc_history[-1]
    Rc_err = abs(Rc_final - config.Rc) / config.Rc * 100
    ax.text(0.95, 0.5, f'$R_c^{{est}}$ = {Rc_final:.4e}\nError: {Rc_err:.2f}%',
            transform=ax.transAxes, ha='right', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax = axes[1, 1]
    ax.plot(inv_history.epochs, inv_history.wall_time, 'g-', lw=1.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wall time [s]')

    fig.tight_layout()
    fig.savefig(f'{OUT_SUPP}/Fig_S2.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# Fig. S3: Heating waveform comparison (was fig15)
# ========================================================================
def figS3_waveform(config):
    print("  Fig. S3: Heating waveform comparison...")
    x_sensors = [0.002, 0.004, 0.006, 0.008]
    wave_results = heating_waveform_comparison(config, x_sensors, N=100, Nt=500)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t_obs = np.linspace(0.5, config.t_end, 30)
    labels = {'step': 'Step', 'pulse': 'Pulse (30%)', 'multi_pulse': 'Multi-pulse'}
    colors_w = {'step': '#e74c3c', 'pulse': '#3498db', 'multi_pulse': '#2ecc71'}

    for j, xs in enumerate([0.002, 0.006, 0.008]):
        ax = axes[j]
        layer = 'L1' if xs < config.a else 'L2'
        for q_type in ['step', 'pulse', 'multi_pulse']:
            s = wave_results[q_type]['sensitivity_per_sensor'][xs]
            ax.plot(t_obs, s, '-o', color=colors_w[q_type], markersize=4,
                    lw=1.8, label=labels[q_type])
        ax.set_xlabel('Time $t$ [s]')
        ax.set_ylabel('$|\\partial T / \\partial (\\log_{10} R_c)|$ [K]')
        ax.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT_SUPP}/Fig_S3.png', dpi=DPI)
    plt.close(fig)


# ========================================================================
# MAIN
# ========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("REGENERATING ALL FIGURES AT 400 DPI WITH 12pt FONTS")
    print("=" * 70)
    t_total = time.time()

    config = default_config()
    print(f"\n{config.summary()}\n")

    # ---- FV reference solution ----
    print("Solving FV reference (N=200, Nt=2000)...")
    sol_ref = solve_fv(config, N1=200, N2=200, Nt=2000, verbose=True)
    eb = sol_ref.energy_balance()
    print(f"  Energy imbalance: {eb['imbalance_pct']:.4f}%\n")

    # ---- Main paper figures 1-8 (no PINN training needed) ----
    print("--- MAIN PAPER FIGURES ---")
    fig01_problem_schematic(config)
    fig02_pinn_architecture()
    fig03_fv_mesh_convergence(config)
    fig04_fv_reference_profiles(sol_ref, config)
    fig05_ansys_fv_comparison(config)
    fig06_imperfect_profiles(config)
    fig07_interface_jump(config)
    fig08_tcr_effect(config)

    # ---- Train Forward PINN ----
    print("\nTraining FORWARD PINN (5000 epochs)...")
    fwd_cfg = TrainingConfig(epochs=5000, log_every=500, lr=1e-3)
    fwd_model, fwd_history = train_forward(config, fwd_cfg, hidden_layers=4,
                                            hidden_units=64, verbose=True)

    fig09_forward_comparison(fwd_model, sol_ref, config)
    fig10_forward_error(fwd_model, sol_ref, config)

    # ---- Generate synthetic observations & Train Inverse PINN ----
    print("\nGenerating synthetic sensor data...")
    sensor_cfg = SensorConfig(x_sensors=[0.002, 0.004, 0.006, 0.008],
                               n_time_samples=50, noise_level=0.01)
    obs_data, _ = generate_observations(config, sensor_cfg, seed=42)
    print(f"  {len(obs_data['x'])} observations at {len(obs_data['x_sensors'])} sensors")

    print("\nTraining INVERSE PINN (10000 epochs)...")
    inv_cfg = TrainingConfig(epochs=10000, log_every=500, lr=1e-3, w_data=100.0)
    inv_model, inv_history = train_inverse(config, obs_data, inv_cfg,
                                            hidden_layers=4, hidden_units=64,
                                            Rc_init=config.Rc * 2.0, verbose=True)

    Rc_final = inv_history.Rc_history[-1]
    Rc_err = abs(Rc_final - config.Rc) / config.Rc * 100
    print(f"\n  Final Rc = {Rc_final:.6e} (error: {Rc_err:.2f}%)")

    fig11_inverse_comparison(inv_model, sol_ref, obs_data, config)

    # ---- Contour plots (from FV reference) ----
    fig12_spatiotemporal_contour(sol_ref, config)
    fig13_layer_contours(sol_ref, config)
    fig14_heatflux_contour(sol_ref, config)
    fig15_gradient_contour(sol_ref, config)
    fig16_energy_balance(sol_ref, config)

    # ---- Supplementary figures ----
    print("\n--- SUPPLEMENTARY FIGURES ---")
    figS1_forward_training(fwd_model, fwd_history, config)
    figS2_inverse_training(inv_history, config)
    figS3_waveform(config)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"All 19 figures generated in {elapsed:.1f} s")
    print(f"Main paper:    {OUT_MAIN}/Fig_01.png ... Fig_16.png")
    print(f"Supplementary: {OUT_SUPP}/Fig_S1.png ... Fig_S3.png")
    print(f"DPI: {DPI}, Base font: 12pt")
    print(f"{'=' * 70}")

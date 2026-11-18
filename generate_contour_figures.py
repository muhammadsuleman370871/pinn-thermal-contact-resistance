#!/usr/bin/env python3
"""High-resolution (500 DPI) contour figures for the main paper, Figs. 12-15.

Two sets are written:

  figures/Fig_12..15.png
      Same data as the figures embedded in the document. The field comes from
      the stand-alone explicit solver, reproduced
      here step for step (vectorised, identical arithmetic).

  figures/FV_consistent/Fig_12..15.png
      Same four figures computed from the validated finite-volume solver
      (solve_fv), i.e. the solver behind Figs. 4-11 and 16.

Style: 500 DPI, 12 pt text, bold axis labels,
bold spines, no grid.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.physical_params import default_config
from solvers.finite_volume import solve_fv

_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DOC = os.path.join(_ROOT, "figures")
OUT_FV = os.path.join(_ROOT, "figures", "FV_consistent")
os.makedirs(OUT_FV, exist_ok=True)
DPI = 500

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
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.grid': False,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

# ---- physical parameters (identical to generate_contour_plots.py) ----
a, L = 0.005, 0.010
k1, k2 = 205.0, 16.0
rho1, rho2 = 2700.0, 8000.0
cp1, cp2 = 900.0, 500.0
alpha1, alpha2 = k1 / (rho1 * cp1), k2 / (rho2 * cp2)
Rc, q_in, h_R, T_inf, t_end = 1e-4, 5e4, 50.0, 300.0, 10.0


# ======================================================================
#  Data set A: the document's field (generate_contour_plots.analytical_approx)
# ======================================================================
def document_field(Nx=200, Nt=150):
    t = t_end
    dx = L / (Nx - 1)
    dt_stable = 0.4 * dx**2 / max(alpha1, alpha2)
    x_grid = np.linspace(0, L, Nx)
    T = np.ones(Nx) * T_inf
    i_int = int(a / dx)
    t_current = 0
    T_hist, t_hist = [T.copy()], [0.0]
    dt_use = min(dt_stable, 0.001)
    while t_current < t:
        T_new = T.copy()
        T_new[1:i_int] = T[1:i_int] + dt_use * alpha1 * (
            T[2:i_int + 1] - 2 * T[1:i_int] + T[0:i_int - 1]) / dx**2
        T_new[i_int + 1:Nx - 1] = T[i_int + 1:Nx - 1] + dt_use * alpha2 * (
            T[i_int + 2:Nx] - 2 * T[i_int + 1:Nx - 1] + T[i_int:Nx - 2]) / dx**2
        T_new[0] = T_new[1] + q_in * dx / k1
        T_new[-1] = (k2 * T_new[-2] / dx + h_R * T_inf) / (k2 / dx + h_R)
        q_int = (T[i_int] - T[i_int + 1]) / (Rc + dx / (2 * k1) + dx / (2 * k2))
        T_new[i_int] = T_new[i_int - 1] + q_int * dx / (2 * k1)
        T_new[i_int + 1] = T_new[i_int] - q_int * Rc
        T = T_new
        t_current += dt_use
        if len(t_hist) < Nt and t_current >= len(t_hist) * t / Nt:
            T_hist.append(T.copy())
            t_hist.append(t_current)
    T_hist.append(T.copy())
    t_hist.append(t_current)
    return x_grid, np.array(t_hist), np.array(T_hist), i_int


def document_derived(x_grid, t_arr, T_field):
    """Heat flux and gradient exactly as computed for the document figures."""
    dx = x_grid[1] - x_grid[0]
    kx = np.where(x_grid < a, k1, k2)
    central = (T_field[:, 2:] - T_field[:, :-2]) / (2 * dx)
    q = np.zeros_like(T_field)
    q[:, 1:-1] = -kx[1:-1] * central
    q[:, 0] = q_in
    q[:, -1] = -k2 * (T_field[:, -1] - T_field[:, -2]) / dx
    g = np.zeros_like(T_field)
    g[:, 1:-1] = np.abs(central)
    g[:, 0] = np.abs(T_field[:, 1] - T_field[:, 0]) / dx
    g[:, -1] = np.abs(T_field[:, -1] - T_field[:, -2]) / dx
    return q / 1000.0, g / 1000.0


# ======================================================================
#  Data set B: validated finite-volume solver, with face values added so the
#  field spans exactly 0..a and a..L and the two layers are never
#  interpolated across the contact interface.
# ======================================================================
def fv_fields():
    cfg = default_config()
    sol = solve_fv(cfg, N1=200, N2=200, Nt=2000, verbose=False)
    N1, N2 = sol.N1, sol.N2
    dx1, dx2 = cfg.a / N1, (cfg.L - cfg.a) / N2
    K1, K2 = cfg.mat1.k, cfg.mat2.k
    T1, T2 = sol.T[:, :N1], sol.T[:, N1:]
    qin = cfg.q_in_array(sol.t)
    q_int = (T1[:, -1] - T2[:, 0]) / (dx1 / (2 * K1) + cfg.Rc + dx2 / (2 * K2))
    q_L = cfg.h_R * (T2[:, -1] - cfg.T_inf)

    # temperatures: cell centres plus the two faces of each layer
    x1 = np.concatenate([[0.0], sol.x1, [cfg.a]])
    x2 = np.concatenate([[cfg.a], sol.x2, [cfg.L]])
    T1e = np.column_stack([T1[:, 0] + qin * dx1 / (2 * K1), T1,
                           T1[:, -1] - q_int * dx1 / (2 * K1)])
    T2e = np.column_stack([T2[:, 0] + q_int * dx2 / (2 * K2), T2,
                           T2[:, -1] - q_L * dx2 / (2 * K2)])

    # conservative face fluxes of the scheme
    xf1 = np.linspace(0.0, cfg.a, N1 + 1)
    xf2 = np.linspace(cfg.a, cfg.L, N2 + 1)
    q1 = np.column_stack([qin, -K1 * np.diff(T1, axis=1) / dx1, q_int])
    q2 = np.column_stack([q_int, -K2 * np.diff(T2, axis=1) / dx2, q_L])
    return dict(cfg=cfg, sol=sol, t=sol.t, x1=x1, x2=x2, T1=T1e, T2=T2e,
                xf1=xf1, xf2=xf2, q1=q1, q2=q2, K1=K1, K2=K2, q_int=q_int)


# ======================================================================
#  Plot helpers
# ======================================================================
def finish_axes(ax, x0=0, x1=10):
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, t_end)
    ax.set_xlabel('Position $x$ (mm)')
    ax.set_ylabel('Time $t$ (s)')
    ax.grid(False)
    ax.tick_params(length=5)


def add_cbar(fig, cf, ax, label, nbins=8):
    cb = fig.colorbar(cf, ax=ax, ticks=ticker.MaxNLocator(nbins=nbins))
    cb.set_label(label, fontweight='bold')
    cb.outline.set_linewidth(1.5)
    cb.ax.tick_params(width=1.2, length=4)
    return cb


def single_panel(parts, levels, cmap, cbar_label, line_color, line_label, path,
                 extend='neither'):
    """parts: list of (x_mm, t, field) blocks drawn on one axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for xm, tt, F in parts:
        X, TT = np.meshgrid(xm, tt)
        cf = ax.contourf(X, TT, F, levels=levels, cmap=cmap, extend=extend)
    add_cbar(fig, cf, ax, cbar_label)
    ax.axvline(a * 1e3, color=line_color, ls='--', lw=2, label=line_label)
    finish_axes(ax)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    if line_color == 'white':
        ax.legend(loc='upper right', facecolor='0.3', edgecolor='white', labelcolor='white',
                  framealpha=0.85)
    else:
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def layer_panels(x1mm, x2mm, tt, T1, T2, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, xm, F, cmap, tag in ((ax1, x1mm, T1, 'YlOrRd', '(a) Aluminum'),
                                 (ax2, x2mm, T2, 'YlGnBu', '(b) SS304')):
        X, TT = np.meshgrid(xm, tt)
        cf = ax.contourf(X, TT, F, levels=np.linspace(T_inf, F.max(), 25), cmap=cmap)
        add_cbar(fig, cf, ax, '$T$ (K)')
        finish_axes(ax, xm[0], xm[-1])
        ax.set_title(tag)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticks([5, 6, 7, 8, 9, 10])
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ======================================================================
def main():
    # ------------------------- set A: document data -------------------------
    print("Set A: field used in the document figures ...")
    xg, tA, TA, i_int = document_field()
    xmm = xg * 1e3
    qA, gA = document_derived(xg, tA, TA)
    print(f"  T max = {TA.max():.2f} K | layer maxima {TA[:, :i_int+1].max():.2f} / "
          f"{TA[:, i_int:].max():.2f} K | q range {qA.min():.2f}..{qA.max():.2f} kW/m2 | "
          f"grad p99 = {np.percentile(gA, 99):.3f} K/mm")

    single_panel([(xmm, tA, TA)], np.linspace(T_inf, TA.max(), 30), 'hot',
                 'Temperature (K)', 'white', 'Interface ($x = a$)', f'{OUT_DOC}/Fig_12.png')
    layer_panels(xmm[:i_int + 1], xmm[i_int:], tA, TA[:, :i_int + 1], TA[:, i_int:],
                 f'{OUT_DOC}/Fig_13.png')
    single_panel([(xmm, tA, qA)], np.linspace(qA.min(), qA.max(), 25), 'coolwarm',
                 'Heat Flux (kW/m$^2$)', 'black', 'Interface', f'{OUT_DOC}/Fig_14.png')
    single_panel([(xmm, tA, gA)], np.linspace(0, np.percentile(gA, 99), 25), 'inferno',
                 '$|dT/dx|$ (K/mm)', 'white', 'Interface', f'{OUT_DOC}/Fig_15.png',
                 extend='max')

    # energy check of the document field at t = 10 s
    dTA = TA[-1] - T_inf
    rc = np.where(xg < a, rho1 * cp1, rho2 * cp2)
    stored_A = np.trapezoid(rc * dTA, xg)
    print(f"  energy stored at 10 s = {stored_A/1e3:.1f} kJ/m2 "
          f"(supplied q_in*t = {q_in*t_end/1e3:.1f} kJ/m2)")

    # ------------------------- set B: validated FV --------------------------
    print("Set B: validated finite-volume solver ...")
    B = fv_fields()
    tB = B['t']
    x1mm, x2mm = B['x1'] * 1e3, B['x2'] * 1e3
    xf1mm, xf2mm = B['xf1'] * 1e3, B['xf2'] * 1e3
    q1, q2 = np.clip(B['q1'], 0, None) / 1e3, np.clip(B['q2'], 0, None) / 1e3
    g1, g2 = np.abs(B['q1']) / B['K1'] / 1e3, np.abs(B['q2']) / B['K2'] / 1e3
    Tmax = max(B['T1'].max(), B['T2'].max())
    eb = B['sol'].energy_balance() if hasattr(B['sol'], 'energy_balance') else None
    print(f"  T max = {Tmax:.2f} K | interface jump at 10 s = "
          f"{B['T1'][-1, -1] - B['T2'][-1, 0]:.2f} K | q_int(10 s) = {B['q_int'][-1]/1e3:.2f} kW/m2")
    print(f"  |dT/dx| at interface, t = 10 s: Al {g1[-1, -1]*1e3:.0f} K/m, "
          f"SS304 {g2[-1, 0]*1e3:.0f} K/m, ratio {g2[-1, 0]/g1[-1, -1]:.2f}")
    if eb is not None:
        print("  energy balance:", {k: (float(v) if np.ndim(v) == 0 else '...') for k, v in eb.items()})

    single_panel([(x1mm, tB, B['T1']), (x2mm, tB, B['T2'])],
                 np.linspace(T_inf, Tmax, 30), 'hot', 'Temperature (K)', 'white',
                 'Interface ($x = a$)', f'{OUT_FV}/Fig_12_FV.png')
    layer_panels(x1mm, x2mm, tB, B['T1'], B['T2'], f'{OUT_FV}/Fig_13_FV.png')
    qmax = max(q1.max(), q2.max())
    single_panel([(xf1mm, tB, q1), (xf2mm, tB, q2)], np.linspace(0, qmax, 26), 'coolwarm',
                 'Heat Flux (kW/m$^2$)', 'black', 'Interface', f'{OUT_FV}/Fig_14_FV.png')
    gmax = max(g1.max(), g2.max())
    single_panel([(xf1mm, tB, g1), (xf2mm, tB, g2)], np.linspace(0, gmax, 26), 'inferno',
                 '$|dT/dx|$ (K/mm)', 'white', 'Interface', f'{OUT_FV}/Fig_15_FV.png')
    print("Done.")


if __name__ == '__main__':
    main()

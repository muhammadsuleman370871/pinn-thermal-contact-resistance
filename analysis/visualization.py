"""
Visualization functions for the PINN contact resistance project.

Generates publication-quality plots for:
- Temperature fields and comparisons
- Training convergence
- Rc recovery convergence
- Sensitivity maps
- Identifiability analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig
from solvers.finite_volume import FVSolution


# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def plot_temperature_field(ref_sol: FVSolution, config: ProblemConfig,
                            pinn_model=None, times: List[float] = None,
                            save_path: str = None):
    """
    Plot temperature profiles at selected times, comparing FV and PINN.
    """
    if times is None:
        times = [0.1 * config.t_end, 0.3 * config.t_end,
                 0.5 * config.t_end, 0.7 * config.t_end, config.t_end]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for i, t_val in enumerate(times):
        # FV reference
        it = np.argmin(np.abs(ref_sol.t - t_val))
        ax.plot(ref_sol.x * 1e3, ref_sol.T[it, :], '-',
                color=colors[i], linewidth=2,
                label=f't = {t_val:.1f}s (FV)')

        # PINN prediction
        if pinn_model is not None:
            x_plot = np.linspace(0, config.L, 200)
            t_plot = np.full_like(x_plot, t_val)
            T_pinn = pinn_model.predict(x_plot, t_plot)
            ax.plot(x_plot * 1e3, T_pinn, '--',
                    color=colors[i], linewidth=1.5,
                    label=f't = {t_val:.1f}s (PINN)')

    # Interface line
    ax.axvline(x=config.a * 1e3, color='gray', linestyle=':', alpha=0.7,
               label='Interface')

    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Temperature T [K]')
    ax.set_title('Temperature Profiles')
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_interface_jump(ref_sol: FVSolution, config: ProblemConfig,
                         save_path: str = None):
    """
    Plot temperature jump at the interface over time.
    """
    T1_int, T2_int = ref_sol.interface_temperatures()
    jump = T1_int - T2_int
    q_int = ref_sol.interface_flux()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(ref_sol.t, jump, 'b-', linewidth=2)
    ax1.set_ylabel('T₁(a⁻) - T₂(a⁺) [K]')
    ax1.set_title(f'Interface Temperature Jump (Rc = {config.Rc:.2e} m²·K/W)')
    ax1.grid(True, alpha=0.3)

    # Expected jump = Rc * q_int
    expected_jump = config.Rc * q_int
    ax1.plot(ref_sol.t[1:], expected_jump[1:], 'r--', linewidth=1.5,
             label='Rc × q_int')
    ax1.legend()

    ax2.plot(ref_sol.t[1:], q_int[1:], 'r-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Interface heat flux [W/m²]')
    ax2.set_title('Interface Heat Flux')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_training_convergence(history, config: ProblemConfig = None,
                               save_path: str = None):
    """
    Plot training loss convergence.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history.epochs, history.total_loss, 'k-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)

    # Individual losses
    ax = axes[0, 1]
    ax.semilogy(history.epochs, history.pde_loss, label='PDE')
    ax.semilogy(history.epochs, history.bc_loss, label='BC')
    ax.semilogy(history.epochs, history.ic_loss, label='IC')
    ax.semilogy(history.epochs, history.flux_loss, label='Flux')
    ax.semilogy(history.epochs, history.jump_loss, label='Jump')
    if any(d > 0 for d in history.data_loss):
        ax.semilogy(history.epochs, history.data_loss, label='Data')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Individual Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rc convergence (inverse problem)
    ax = axes[1, 0]
    if config is not None:
        ax.plot(history.epochs, history.Rc_history, 'b-', linewidth=1.5,
                label='Estimated Rc')
        ax.axhline(y=config.Rc, color='r', linestyle='--', linewidth=1.5,
                   label=f'True Rc = {config.Rc:.2e}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rc [m²·K/W]')
        ax.set_title('Contact Resistance Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Forward problem\n(Rc fixed)',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

    # Wall time
    ax = axes[1, 1]
    ax.plot(history.epochs, history.wall_time, 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wall Time [s]')
    ax.set_title('Computation Time')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_sensor_observations(obs_data: Dict, config: ProblemConfig,
                              ref_sol: FVSolution, pinn_model=None,
                              save_path: str = None):
    """
    Plot observed data vs reference and PINN prediction at sensor locations.
    """
    x_sensors = obs_data['x_sensors']
    n_sensors = len(x_sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(4 * n_sensors, 4),
                              sharey=True)
    if n_sensors == 1:
        axes = [axes]

    n_times = len(obs_data['t']) // n_sensors

    for i, xs in enumerate(x_sensors):
        ax = axes[i]
        # Observations
        t_i = obs_data['t'][i * n_times:(i + 1) * n_times].numpy().flatten()
        T_obs_i = obs_data['T'][i * n_times:(i + 1) * n_times].numpy().flatten()
        T_true_i = obs_data['T_true'][i * n_times:(i + 1) * n_times].numpy().flatten()

        ax.plot(t_i, T_obs_i, 'ko', markersize=3, alpha=0.5, label='Observed')
        ax.plot(t_i, T_true_i, 'b-', linewidth=1.5, label='True')

        if pinn_model is not None:
            T_pinn = pinn_model.predict(np.full(len(t_i), xs), t_i)
            ax.plot(t_i, T_pinn, 'r--', linewidth=1.5, label='PINN')

        layer = "Layer 1" if xs < config.a else "Layer 2"
        ax.set_title(f'Sensor at x={xs*1e3:.1f}mm ({layer})')
        ax.set_xlabel('Time [s]')
        if i == 0:
            ax.set_ylabel('Temperature [K]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_sensitivity(sens_results: Dict, save_path: str = None):
    """
    Plot sensitivity of observations to Rc at each sensor location.
    """
    t_obs = sens_results['t_obs']
    x_sensors = sens_results['x_sensors']

    fig, ax = plt.subplots(figsize=(8, 5))

    for xs in x_sensors:
        sens = sens_results['sensitivity_per_sensor'][xs]
        layer = "L1" if xs < 0.005 else "L2"
        ax.plot(t_obs, sens, '-o', markersize=3,
                label=f'x={xs*1e3:.1f}mm ({layer})')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|∂T/∂(log₁₀Rc)| [K]')
    ax.set_title('Sensitivity to Contact Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_error_field(ref_sol: FVSolution, pinn_model, config: ProblemConfig,
                      t_val: float = None, save_path: str = None):
    """
    Plot absolute error between FV and PINN at a given time.
    """
    if t_val is None:
        t_val = config.t_end

    x_plot = np.linspace(0, config.L, 200)
    t_plot = np.full_like(x_plot, t_val)

    T_pinn = pinn_model.predict(x_plot, t_plot)
    T_ref = np.array([ref_sol.temperature_at(xi, t_val) for xi in x_plot])
    error = np.abs(T_pinn - T_ref)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(x_plot * 1e3, T_ref, 'b-', linewidth=2, label='FV Reference')
    ax1.plot(x_plot * 1e3, T_pinn, 'r--', linewidth=1.5, label='PINN')
    ax1.axvline(x=config.a * 1e3, color='gray', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title(f'Temperature Comparison at t = {t_val:.1f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_plot * 1e3, error, 'k-', linewidth=1.5)
    ax2.axvline(x=config.a * 1e3, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Position x [mm]')
    ax2.set_ylabel('|T_PINN - T_FV| [K]')
    ax2.set_title('Absolute Error')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_energy_balance(ref_sol: FVSolution, save_path: str = None):
    """Plot cumulative energy balance over time."""
    config = ref_sol.config
    fig, ax = plt.subplots(figsize=(8, 5))

    dx1 = config.a / ref_sol.N1
    dx2 = (config.L - config.a) / ref_sol.N2

    Q_in_cum = []
    Q_out_cum = []
    E_stored_cum = []

    for it in range(len(ref_sol.t)):
        q_in_arr = config.q_in_array(ref_sol.t[:it+1])
        Q_in = np.trapezoid(q_in_arr, ref_sol.t[:it+1]) if it > 0 else 0

        T_right = ref_sol.T[:it+1, -1]
        q_out_arr = config.h_R * (T_right - config.T_inf)
        Q_out = np.trapezoid(q_out_arr, ref_sol.t[:it+1]) if it > 0 else 0

        E1 = config.mat1.rho * config.mat1.cp * dx1 * np.sum(
            ref_sol.T[it, :ref_sol.N1] - ref_sol.T[0, :ref_sol.N1])
        E2 = config.mat2.rho * config.mat2.cp * dx2 * np.sum(
            ref_sol.T[it, ref_sol.N1:] - ref_sol.T[0, ref_sol.N1:])

        Q_in_cum.append(Q_in)
        Q_out_cum.append(Q_out)
        E_stored_cum.append(E1 + E2)

    Q_in_cum = np.array(Q_in_cum)
    Q_out_cum = np.array(Q_out_cum)
    E_stored_cum = np.array(E_stored_cum)
    imbalance = Q_in_cum - Q_out_cum - E_stored_cum

    ax.plot(ref_sol.t, Q_in_cum, 'r-', label='Q_in (cumulative)')
    ax.plot(ref_sol.t, Q_out_cum, 'b-', label='Q_out (cumulative)')
    ax.plot(ref_sol.t, E_stored_cum, 'g-', label='E_stored')
    ax.plot(ref_sol.t, imbalance, 'k--', label='Imbalance')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J/m²]')
    ax.set_title('Cumulative Energy Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig

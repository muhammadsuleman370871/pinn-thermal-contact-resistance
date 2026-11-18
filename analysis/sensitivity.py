"""
Sensitivity and identifiability analysis for the inverse contact resistance problem.

Computes:
- Sensitivity of observations to Rc (Jacobian)
- Condition number of the sensitivity matrix
- Fisher information for identifiability assessment
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig
from solvers.finite_volume import solve_fv


def compute_sensitivity(config: ProblemConfig,
                        x_sensors: List[float],
                        t_obs: np.ndarray,
                        delta_log_Rc: float = 0.01,
                        N: int = 100, Nt: int = 500) -> Dict:
    """
    Compute sensitivity of temperature observations to Rc using finite differences.

    dT/d(log10 Rc) at each sensor location and time.

    Parameters
    ----------
    config : ProblemConfig
    x_sensors : list of sensor locations
    t_obs : observation times
    delta_log_Rc : finite-difference step in log10 space

    Returns
    -------
    dict with:
        'J': Jacobian matrix (n_obs x 1)
        'sensitivity': absolute sensitivity at each point
        'max_sensitivity_time': time of maximum sensitivity per sensor
        'condition_number': condition number of J^T J
        'fisher_info': Fisher information scalar
    """
    log_Rc = np.log10(config.Rc)

    # Forward solutions at Rc - delta and Rc + delta
    def solve_at_logRc(lr):
        cfg = ProblemConfig(
            a=config.a, L=config.L,
            mat1=config.mat1, mat2=config.mat2,
            Rc=10**lr,
            h_R=config.h_R, T_inf=config.T_inf, T_init=config.T_init,
            q_in_amplitude=config.q_in_amplitude,
            q_in_type=config.q_in_type,
            pulse_duration=config.pulse_duration,
            t_end=config.t_end,
        )
        sol = solve_fv(cfg, N1=N, N2=N, Nt=Nt)
        T_vals = []
        for xs in x_sensors:
            for ti in t_obs:
                T_vals.append(sol.temperature_at(xs, ti))
        return np.array(T_vals)

    T_plus = solve_at_logRc(log_Rc + delta_log_Rc)
    T_minus = solve_at_logRc(log_Rc - delta_log_Rc)

    # Central difference
    J = (T_plus - T_minus) / (2 * delta_log_Rc)

    n_times = len(t_obs)
    sensitivity_per_sensor = {}
    max_sens_time = {}

    for i, xs in enumerate(x_sensors):
        sens = np.abs(J[i * n_times:(i + 1) * n_times])
        sensitivity_per_sensor[xs] = sens
        max_idx = np.argmax(sens)
        max_sens_time[xs] = t_obs[max_idx]

    # Fisher information: F = J^T Sigma^{-1} J (scalar for single parameter)
    # With proportional noise sigma_i = eta * (T_i - T_init), eta = 0.01 (1% noise):
    # Sigma^{-1}_{ii} = 1 / sigma_i^2
    # For unit noise (eta = 1), F_unit = sum(J_i^2 / (T_i - T_init)^2)
    # Actual F = F_unit / eta^2
    # We report the unscaled form (eta factored out) so F is noise-level-independent.
    # Compute reference temperatures at baseline Rc for weighting
    T_ref = solve_at_logRc(log_Rc)
    T_rise = T_ref - config.T_init  # local temperature rise at each observation
    T_rise = np.maximum(np.abs(T_rise), 1e-6)  # avoid division by zero at early times
    # Unweighted (legacy): fisher_unweighted = np.dot(J, J)
    # Weighted with proportional noise covariance:
    fisher_info = np.sum((J / T_rise) ** 2)  # = J^T Sigma^{-1} J with sigma_i = T_rise_i

    # Condition number
    JtJ = np.array([[fisher_info]])
    cond = np.linalg.cond(JtJ)

    return {
        'J': J.reshape(-1, 1),
        'sensitivity_per_sensor': sensitivity_per_sensor,
        'max_sensitivity_time': max_sens_time,
        'fisher_info': fisher_info,
        'condition_number': cond,
        't_obs': t_obs,
        'x_sensors': x_sensors,
    }


def identifiability_map(config: ProblemConfig,
                        Rc_range: np.ndarray,
                        sensor_sets: Dict[str, List[float]],
                        noise_levels: List[float] = [0.0, 0.01, 0.03],
                        N: int = 100, Nt: int = 500) -> Dict:
    """
    Build an identifiability map across Rc values, sensor placements, and noise.

    For each combination, compute Fisher information and estimate
    expected parameter uncertainty.

    Parameters
    ----------
    config : ProblemConfig
    Rc_range : array of Rc values to test
    sensor_sets : dict mapping name to list of sensor locations
    noise_levels : list of noise fractions

    Returns
    -------
    dict with identifiability metrics.
    """
    t_obs = np.linspace(0.05 * config.t_end, config.t_end, 30)

    results = {}
    for name, sensors in sensor_sets.items():
        for Rc_val in Rc_range:
            cfg_test = ProblemConfig(
                a=config.a, L=config.L,
                mat1=config.mat1, mat2=config.mat2,
                Rc=Rc_val,
                h_R=config.h_R, T_inf=config.T_inf, T_init=config.T_init,
                q_in_amplitude=config.q_in_amplitude,
                q_in_type=config.q_in_type,
                pulse_duration=config.pulse_duration,
                t_end=config.t_end,
            )

            sens = compute_sensitivity(cfg_test, sensors, t_obs, N=N, Nt=Nt)
            fisher = sens['fisher_info']

            for noise in noise_levels:
                # Expected parameter uncertainty: sigma_Rc ~ sigma_noise / sqrt(Fisher)
                # In log10 space: sigma_log_Rc ~ noise_std / sqrt(Fisher)
                T_scale = max(cfg_test.T_scale, 1.0)
                sigma_noise = noise * T_scale  # approximate noise scale
                if fisher > 1e-20:
                    sigma_log_Rc = sigma_noise / np.sqrt(fisher) if fisher > 0 else np.inf
                    # Convert to relative Rc uncertainty
                    sigma_Rc_rel = np.log(10) * sigma_log_Rc  # d(Rc)/Rc ~ ln(10) * d(log10 Rc)
                else:
                    sigma_Rc_rel = np.inf

                results[(name, Rc_val, noise)] = {
                    'fisher_info': fisher,
                    'sigma_Rc_rel': sigma_Rc_rel,
                    'identifiable': sigma_Rc_rel < 0.5,  # < 50% relative uncertainty
                    'max_sensitivity': np.max(np.abs(sens['J'])),
                }

    return results


def heating_waveform_comparison(config: ProblemConfig,
                                 x_sensors: List[float],
                                 N: int = 100, Nt: int = 500) -> Dict:
    """
    Compare identifiability across heating waveforms.

    Tests step, pulse, and multi-pulse heating.
    """
    t_obs = np.linspace(0.05 * config.t_end, config.t_end, 30)
    results = {}

    for q_type in ["step", "pulse", "multi_pulse"]:
        cfg = ProblemConfig(
            a=config.a, L=config.L,
            mat1=config.mat1, mat2=config.mat2,
            Rc=config.Rc,
            h_R=config.h_R, T_inf=config.T_inf, T_init=config.T_init,
            q_in_amplitude=config.q_in_amplitude,
            q_in_type=q_type,
            pulse_duration=0.3,
            t_end=config.t_end,
        )

        sens = compute_sensitivity(cfg, x_sensors, t_obs, N=N, Nt=Nt)
        results[q_type] = {
            'fisher_info': sens['fisher_info'],
            'max_sensitivity': np.max(np.abs(sens['J'])),
            'sensitivity_per_sensor': sens['sensitivity_per_sensor'],
            'max_sensitivity_time': sens['max_sensitivity_time'],
        }

    return results

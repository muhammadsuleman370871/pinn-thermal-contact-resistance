"""
Conventional inverse solver for thermal contact resistance estimation.

Uses Levenberg-Marquardt (scipy.optimize.least_squares) wrapped around
the finite-volume forward solver. This provides the baseline comparison
for the PINN inverse approach.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Optional, Tuple, List
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig
from solvers.finite_volume import solve_fv


def forward_model(log_Rc: float, config: ProblemConfig,
                  x_sensors: List[float], t_obs: np.ndarray,
                  N: int = 100, Nt: int = 500) -> np.ndarray:
    """
    Run forward model for given Rc and return temperatures at sensor locations.

    Parameters
    ----------
    log_Rc : float
        log10(Rc) value.
    config : ProblemConfig
        Base configuration (Rc will be overridden).
    x_sensors : list of float
        Sensor x-coordinates.
    t_obs : np.ndarray
        Observation times.

    Returns
    -------
    T_pred : np.ndarray
        Predicted temperatures, shape (n_sensors * n_times,).
    """
    # Create config with trial Rc
    cfg_trial = ProblemConfig(
        a=config.a, L=config.L,
        mat1=config.mat1, mat2=config.mat2,
        Rc=10**log_Rc,
        h_R=config.h_R, T_inf=config.T_inf, T_init=config.T_init,
        q_in_amplitude=config.q_in_amplitude,
        q_in_type=config.q_in_type,
        pulse_duration=config.pulse_duration,
        t_end=config.t_end,
    )

    sol = solve_fv(cfg_trial, N1=N, N2=N, Nt=Nt)

    T_pred = []
    for x_s in x_sensors:
        for ti in t_obs:
            T_pred.append(sol.temperature_at(x_s, ti))

    return np.array(T_pred)


def inverse_lm(config: ProblemConfig, obs_data: Dict,
               log_Rc_init: float = None,
               N: int = 100, Nt: int = 500,
               verbose: bool = True) -> Dict:
    """
    Estimate Rc using Levenberg-Marquardt least-squares.

    Parameters
    ----------
    config : ProblemConfig
        With TRUE Rc for reference.
    obs_data : dict
        Observation data from synthetic.generate_observations().
    log_Rc_init : float
        Initial guess for log10(Rc).
    N, Nt : int
        FV solver resolution.

    Returns
    -------
    result : dict
        'Rc_est': estimated Rc
        'Rc_true': true Rc
        'Rc_error_pct': relative error %
        'log_Rc_est': log10 of estimated Rc
        'n_fev': number of function evaluations
        'cost': final cost
        'wall_time': computation time
        'T_pred': predicted temperatures at observations
    """
    if log_Rc_init is None:
        log_Rc_init = np.log10(config.Rc * 2.0)  # start 2x off

    x_sensors = obs_data['x_sensors']
    x_obs = obs_data['x'].numpy().flatten()
    t_obs_all = obs_data['t'].numpy().flatten()
    T_obs = obs_data['T'].numpy().flatten()

    # Get unique observation times
    t_obs_unique = np.unique(t_obs_all)

    def residuals(params):
        log_Rc = params[0]
        T_pred = forward_model(log_Rc, config, x_sensors, t_obs_unique, N=N, Nt=Nt)
        return T_pred - T_obs

    t0 = time.time()
    result = least_squares(
        residuals, [log_Rc_init],
        method='lm',
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=50,
        verbose=2 if verbose else 0,
    )
    wall_time = time.time() - t0

    Rc_est = 10**result.x[0]
    Rc_true = config.Rc
    Rc_error = abs(Rc_est - Rc_true) / Rc_true * 100

    # Final prediction
    T_pred_final = forward_model(result.x[0], config, x_sensors, t_obs_unique, N=N, Nt=Nt)

    return {
        'Rc_est': Rc_est,
        'Rc_true': Rc_true,
        'Rc_error_pct': Rc_error,
        'log_Rc_est': result.x[0],
        'n_fev': result.nfev,
        'cost': result.cost,
        'wall_time': wall_time,
        'T_pred': T_pred_final,
        'success': result.success,
    }


def multi_start_inverse(config: ProblemConfig, obs_data: Dict,
                        n_starts: int = 5, N: int = 100, Nt: int = 500,
                        verbose: bool = False) -> Dict:
    """
    Run inverse with multiple starting guesses to assess uniqueness.

    Parameters
    ----------
    config : ProblemConfig
    obs_data : dict
    n_starts : int
        Number of different starting guesses.

    Returns
    -------
    results : dict
        'all_Rc': list of estimated Rc values
        'best_Rc': best estimate (lowest cost)
        'spread': max - min of estimates
        'Rc_true': true value
        'best_error_pct': error of best estimate
    """
    log_Rc_true = np.log10(config.Rc)

    # Starting guesses spanning orders of magnitude
    log_Rc_starts = np.linspace(log_Rc_true - 2, log_Rc_true + 2, n_starts)

    all_results = []
    for i, lr in enumerate(log_Rc_starts):
        if verbose:
            print(f"\n  Start {i+1}/{n_starts}: log10(Rc_init) = {lr:.2f}")
        res = inverse_lm(config, obs_data, log_Rc_init=lr, N=N, Nt=Nt, verbose=False)
        all_results.append(res)

    all_Rc = [r['Rc_est'] for r in all_results]
    costs = [r['cost'] for r in all_results]
    best_idx = np.argmin(costs)

    return {
        'all_Rc': all_Rc,
        'all_costs': costs,
        'best_Rc': all_Rc[best_idx],
        'best_error_pct': all_results[best_idx]['Rc_error_pct'],
        'spread': max(all_Rc) - min(all_Rc),
        'Rc_true': config.Rc,
        'all_results': all_results,
    }

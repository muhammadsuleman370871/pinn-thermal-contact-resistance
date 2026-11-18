"""
Synthetic observation data generation from the finite-volume reference solver.

Generates sensor temperature histories at specified locations,
optionally with added Gaussian noise, for training the inverse PINN.
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig
from solvers.finite_volume import solve_fv, FVSolution


@dataclass
class SensorConfig:
    """Configuration for synthetic sensor observations."""
    # Sensor locations [m] (distance from x=0)
    x_sensors: List[float] = None

    # Time sampling
    n_time_samples: int = 50     # number of time observations per sensor
    t_start_frac: float = 0.05   # start time as fraction of t_end
    t_end_frac: float = 1.0      # end time as fraction of t_end

    # Noise
    noise_level: float = 0.01    # relative noise std (0.01 = 1%)
    noise_absolute: float = 0.0  # absolute noise std [K] (added on top)

    # Reference solver resolution
    N_ref: int = 200             # cells per layer for reference
    Nt_ref: int = 2000           # time steps for reference

    def __post_init__(self):
        if self.x_sensors is None:
            # Default: 3 sensors, one in each layer + near interface
            self.x_sensors = [0.002, 0.004, 0.006, 0.008]


def generate_observations(config: ProblemConfig,
                          sensor_cfg: Optional[SensorConfig] = None,
                          seed: int = 42) -> Tuple[Dict, FVSolution]:
    """
    Generate synthetic observation data from the FV reference solver.

    Parameters
    ----------
    config : ProblemConfig
        Problem configuration with TRUE Rc.
    sensor_cfg : SensorConfig
        Sensor placement and noise configuration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    obs_data : dict
        'x': torch.Tensor (N_obs, 1)
        't': torch.Tensor (N_obs, 1)
        'T': torch.Tensor (N_obs, 1)  (noisy observations)
        'T_true': torch.Tensor (N_obs, 1)  (noise-free)
        'x_sensors': list of sensor locations
        'noise_level': actual noise level used
    ref_sol : FVSolution
        Reference solution for comparison.
    """
    if sensor_cfg is None:
        sensor_cfg = SensorConfig()

    np.random.seed(seed)

    # Solve reference problem
    ref_sol = solve_fv(config, N1=sensor_cfg.N_ref, N2=sensor_cfg.N_ref,
                       Nt=sensor_cfg.Nt_ref)

    # Generate observation times
    t_start = sensor_cfg.t_start_frac * config.t_end
    t_end = sensor_cfg.t_end_frac * config.t_end
    t_obs = np.linspace(t_start, t_end, sensor_cfg.n_time_samples)

    # Collect observations at each sensor
    x_list, t_list, T_true_list = [], [], []

    for x_s in sensor_cfg.x_sensors:
        T_history = np.array([ref_sol.temperature_at(x_s, ti) for ti in t_obs])
        x_list.append(np.full(len(t_obs), x_s))
        t_list.append(t_obs.copy())
        T_true_list.append(T_history)

    x_all = np.concatenate(x_list)
    t_all = np.concatenate(t_list)
    T_true_all = np.concatenate(T_true_list)

    # Add noise
    noise_std = sensor_cfg.noise_level * np.abs(T_true_all - config.T_init).clip(min=1.0)
    noise_std += sensor_cfg.noise_absolute
    noise = np.random.randn(len(T_true_all)) * noise_std
    T_noisy = T_true_all + noise

    obs_data = {
        'x': torch.tensor(x_all, dtype=torch.float32).reshape(-1, 1),
        't': torch.tensor(t_all, dtype=torch.float32).reshape(-1, 1),
        'T': torch.tensor(T_noisy, dtype=torch.float32).reshape(-1, 1),
        'T_true': torch.tensor(T_true_all, dtype=torch.float32).reshape(-1, 1),
        'x_sensors': sensor_cfg.x_sensors,
        'noise_level': sensor_cfg.noise_level,
    }

    return obs_data, ref_sol


def generate_sensor_sweep_data(config: ProblemConfig,
                               sensor_locations: List[List[float]],
                               noise_levels: List[float] = [0.0, 0.01, 0.03],
                               n_seeds: int = 5) -> Dict:
    """
    Generate data for sensor placement and noise sweep.

    Parameters
    ----------
    config : ProblemConfig
    sensor_locations : list of lists
        Each inner list is a set of sensor x-coordinates.
    noise_levels : list of float
    n_seeds : int

    Returns
    -------
    dict with all generated data indexed by (sensors, noise, seed).
    """
    sweep_data = {}

    for sensors in sensor_locations:
        for noise in noise_levels:
            for seed in range(n_seeds):
                scfg = SensorConfig(
                    x_sensors=sensors,
                    noise_level=noise,
                )
                obs, ref = generate_observations(config, scfg, seed=seed + 100)
                sweep_data[(tuple(sensors), noise, seed)] = {
                    'obs': obs,
                    'ref': ref,
                }

    return sweep_data


def withheld_data(config: ProblemConfig, ref_sol: FVSolution,
                  n_points: int = 100) -> Dict:
    """
    Generate withheld validation data at random (x, t) locations.
    """
    np.random.seed(999)
    x_val = np.random.uniform(0, config.L, n_points)
    t_val = np.random.uniform(0.05 * config.t_end, config.t_end, n_points)
    T_val = np.array([ref_sol.temperature_at(xi, ti) for xi, ti in zip(x_val, t_val)])

    return {
        'x': torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1),
        't': torch.tensor(t_val, dtype=torch.float32).reshape(-1, 1),
        'T': torch.tensor(T_val, dtype=torch.float32).reshape(-1, 1),
    }

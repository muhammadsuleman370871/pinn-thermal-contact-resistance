"""
Training loop for the domain-decomposed PINN.

Supports both forward and inverse problems.
Implements adaptive loss weighting (optional) and learning rate scheduling.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig
from pinn.networks import ContactResistancePINN


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Collocation points
    n_pde_1: int = 2000       # PDE collocation points in layer 1
    n_pde_2: int = 2000       # PDE collocation points in layer 2
    n_bc: int = 200           # boundary condition points
    n_ic: int = 200           # initial condition points per layer
    n_interface: int = 200    # interface points

    # Training
    epochs: int = 10000
    lr: float = 1e-3
    lr_decay: float = 0.5
    lr_step: int = 3000
    min_lr: float = 1e-5

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 10.0
    w_flux: float = 10.0
    w_jump: float = 10.0
    w_data: float = 100.0     # for inverse problems

    # Logging
    log_every: int = 500
    seed: int = 42


@dataclass
class TrainingHistory:
    """Training history container."""
    epochs: List[int] = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)
    pde_loss: List[float] = field(default_factory=list)
    bc_loss: List[float] = field(default_factory=list)
    ic_loss: List[float] = field(default_factory=list)
    flux_loss: List[float] = field(default_factory=list)
    jump_loss: List[float] = field(default_factory=list)
    data_loss: List[float] = field(default_factory=list)
    Rc_history: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)


def sample_collocation_points(config: ProblemConfig, train_cfg: TrainingConfig,
                               device: torch.device = torch.device('cpu')) -> dict:
    """
    Generate collocation points for all loss terms.

    Returns dict of torch tensors on the specified device.
    """
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    a = config.a
    L = config.L
    t_end = config.t_end

    # PDE points in layer 1: x in (0, a), t in (0, t_end)
    x_pde1 = torch.rand(train_cfg.n_pde_1, 1) * a
    t_pde1 = torch.rand(train_cfg.n_pde_1, 1) * t_end

    # PDE points in layer 2: x in (a, L), t in (0, t_end)
    x_pde2 = a + torch.rand(train_cfg.n_pde_2, 1) * (L - a)
    t_pde2 = torch.rand(train_cfg.n_pde_2, 1) * t_end

    # Boundary points: t in (0, t_end)
    t_bc = torch.rand(train_cfg.n_bc, 1) * t_end

    # Initial condition points
    x_ic1 = torch.rand(train_cfg.n_ic, 1) * a
    x_ic2 = a + torch.rand(train_cfg.n_ic, 1) * (L - a)

    # Interface points: t in (0, t_end)
    t_int = torch.rand(train_cfg.n_interface, 1) * t_end

    # Compute q_in for boundary points
    q_in_vals = config.q_in_array(t_bc.numpy().flatten())
    q_in = torch.tensor(q_in_vals, dtype=torch.float32).reshape(-1, 1)

    points = {
        'x_pde1': x_pde1.to(device), 't_pde1': t_pde1.to(device),
        'x_pde2': x_pde2.to(device), 't_pde2': t_pde2.to(device),
        't_bc': t_bc.to(device), 'q_in': q_in.to(device),
        'x_ic1': x_ic1.to(device), 'x_ic2': x_ic2.to(device),
        't_int': t_int.to(device),
    }
    return points


def train_pinn(model: ContactResistancePINN, config: ProblemConfig,
               train_cfg: TrainingConfig,
               obs_data: Optional[Dict] = None,
               device: torch.device = torch.device('cpu'),
               verbose: bool = True) -> TrainingHistory:
    """
    Train the PINN model.

    Parameters
    ----------
    model : ContactResistancePINN
    config : ProblemConfig
    train_cfg : TrainingConfig
    obs_data : dict with 'x', 't', 'T' tensors for inverse problems
    device : torch.device
    verbose : bool

    Returns
    -------
    TrainingHistory
    """
    model = model.to(device)
    model.train()

    # Optimizer setup
    if model.inverse:
        params = list(model.parameters())
    else:
        # Exclude log_Rc from optimization in forward mode
        params = [p for name, p in model.named_parameters() if 'log_Rc' not in name]

    optimizer = optim.Adam(params, lr=train_cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=train_cfg.lr_step, gamma=train_cfg.lr_decay
    )

    # Sample collocation points
    points = sample_collocation_points(config, train_cfg, device)

    history = TrainingHistory()
    t_start = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        optimizer.zero_grad()

        # PDE residuals
        r_pde1 = model.compute_pde_residual(points['x_pde1'], points['t_pde1'], layer=1)
        r_pde2 = model.compute_pde_residual(points['x_pde2'], points['t_pde2'], layer=2)
        loss_pde = torch.mean(r_pde1**2) + torch.mean(r_pde2**2)

        # Boundary conditions
        bc_left, bc_right = model.compute_bc_residuals(points['t_bc'], points['q_in'])
        loss_bc = torch.mean(bc_left**2) + torch.mean(bc_right**2)

        # Initial conditions
        ic1, ic2 = model.compute_ic_residual(points['x_ic1'], points['x_ic2'])
        loss_ic = torch.mean(ic1**2) + torch.mean(ic2**2)

        # Interface conditions
        flux_res, jump_res = model.compute_interface_residuals(points['t_int'])
        loss_flux = torch.mean(flux_res**2)
        loss_jump = torch.mean(jump_res**2)

        # Total physics loss
        total_loss = (
            train_cfg.w_pde * loss_pde +
            train_cfg.w_bc * loss_bc +
            train_cfg.w_ic * loss_ic +
            train_cfg.w_flux * loss_flux +
            train_cfg.w_jump * loss_jump
        )

        # Data loss for inverse problems
        loss_data = torch.tensor(0.0)
        if model.inverse and obs_data is not None:
            loss_data = model.compute_data_loss(
                obs_data['x'].to(device),
                obs_data['t'].to(device),
                obs_data['T'].to(device)
            )
            total_loss = total_loss + train_cfg.w_data * loss_data

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = max(pg['lr'], train_cfg.min_lr)

        # Logging
        if epoch % train_cfg.log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            history.epochs.append(epoch)
            history.total_loss.append(total_loss.item())
            history.pde_loss.append(loss_pde.item())
            history.bc_loss.append(loss_bc.item())
            history.ic_loss.append(loss_ic.item())
            history.flux_loss.append(loss_flux.item())
            history.jump_loss.append(loss_jump.item())
            history.data_loss.append(loss_data.item())
            history.Rc_history.append(model.Rc.item())
            history.wall_time.append(elapsed)

            if verbose:
                rc_val = model.Rc.item()
                rc_true = config.Rc
                rc_err = abs(rc_val - rc_true) / rc_true * 100 if model.inverse else 0
                print(f"Epoch {epoch:6d} | Loss: {total_loss.item():.2e} | "
                      f"PDE: {loss_pde.item():.2e} | BC: {loss_bc.item():.2e} | "
                      f"IC: {loss_ic.item():.2e} | Flux: {loss_flux.item():.2e} | "
                      f"Jump: {loss_jump.item():.2e}"
                      + (f" | Data: {loss_data.item():.2e} | "
                         f"Rc: {rc_val:.2e} ({rc_err:.1f}% err)"
                         if model.inverse else "")
                      + f" | t: {elapsed:.1f}s")

    return history


def train_forward(config: ProblemConfig, train_cfg: Optional[TrainingConfig] = None,
                  hidden_layers: int = 4, hidden_units: int = 64,
                  verbose: bool = True) -> Tuple[ContactResistancePINN, TrainingHistory]:
    """
    Convenience function for forward problem training.
    """
    if train_cfg is None:
        train_cfg = TrainingConfig()

    torch.manual_seed(train_cfg.seed)
    model = ContactResistancePINN(
        config, hidden_layers=hidden_layers, hidden_units=hidden_units,
        inverse=False
    )

    history = train_pinn(model, config, train_cfg, verbose=verbose)
    return model, history


def train_inverse(config: ProblemConfig, obs_data: Dict,
                  train_cfg: Optional[TrainingConfig] = None,
                  hidden_layers: int = 4, hidden_units: int = 64,
                  Rc_init: Optional[float] = None,
                  verbose: bool = True) -> Tuple[ContactResistancePINN, TrainingHistory]:
    """
    Convenience function for inverse problem training.

    Parameters
    ----------
    config : ProblemConfig
        Problem with TRUE Rc (for reference/comparison).
    obs_data : dict
        Observation data with 'x', 't', 'T' tensors.
    Rc_init : float, optional
        Initial guess for Rc.
    """
    if train_cfg is None:
        train_cfg = TrainingConfig()

    torch.manual_seed(train_cfg.seed)
    model = ContactResistancePINN(
        config, hidden_layers=hidden_layers, hidden_units=hidden_units,
        inverse=True, Rc_init=Rc_init
    )

    history = train_pinn(model, config, train_cfg, obs_data=obs_data, verbose=verbose)
    return model, history

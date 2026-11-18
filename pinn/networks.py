"""
Neural network architectures for domain-decomposed PINN.

Architecture:
    - T1_net(x, t) -> T1: temperature in layer 1 (0 < x < a)
    - T2_net(x, t) -> T2: temperature in layer 2 (a < x < L)
    - log_Rc: shared trainable parameter (ensures Rc > 0)

The domain decomposition allows a true temperature discontinuity
at the interface x = a, which a single network cannot represent.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TemperatureNet(nn.Module):
    """
    Fully connected network for temperature prediction in one layer.

    Inputs: (x, t) normalized to [0, 1]
    Output: T (dimensional, in Kelvin)

    Uses tanh activation and Xavier initialization for smooth PDE residuals.
    Optional Fourier feature embedding for better high-frequency capture.
    """

    def __init__(self, hidden_layers: int = 4, hidden_units: int = 64,
                 use_fourier: bool = False, n_fourier: int = 32,
                 T_ref: float = 300.0, T_scale: float = 50.0):
        super().__init__()
        self.T_ref = T_ref
        self.T_scale = T_scale
        self.use_fourier = use_fourier

        if use_fourier:
            # Random Fourier features
            self.B = nn.Parameter(
                torch.randn(2, n_fourier) * 2.0, requires_grad=False
            )
            input_dim = 2 * n_fourier
        else:
            input_dim = 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))

        self.net = nn.Sequential(*layers)

        # Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x, t : torch.Tensor, shape (N, 1) each, normalized to [0, 1].

        Returns
        -------
        T : torch.Tensor, shape (N, 1), dimensional temperature.
        """
        xt = torch.cat([x, t], dim=1)

        if self.use_fourier:
            proj = xt @ self.B  # (N, n_fourier)
            xt = torch.cat([torch.sin(np.pi * proj),
                            torch.cos(np.pi * proj)], dim=1)

        out = self.net(xt)
        # Scale output: T = T_ref + T_scale * network_output
        return self.T_ref + self.T_scale * out


class ContactResistancePINN(nn.Module):
    """
    Domain-decomposed PINN for two-layer heat conduction
    with thermal contact resistance.

    Components:
        - T1_net: temperature network for layer 1
        - T2_net: temperature network for layer 2
        - log_Rc: trainable log(Rc/Rc_scale) for inverse problems
        - Physical parameters stored for loss computation

    For forward problems: Rc is fixed (log_Rc not optimized).
    For inverse problems: log_Rc is included in optimizer.
    """

    def __init__(self, config, hidden_layers: int = 4, hidden_units: int = 64,
                 use_fourier: bool = False, inverse: bool = False,
                 Rc_init: Optional[float] = None):
        super().__init__()

        # Store physical parameters
        self.a = config.a
        self.L = config.L
        self.k1 = config.mat1.k
        self.k2 = config.mat2.k
        self.rhocp1 = config.mat1.rho * config.mat1.cp
        self.rhocp2 = config.mat2.rho * config.mat2.cp
        self.h_R = config.h_R
        self.T_inf = config.T_inf
        self.T_init = config.T_init
        self.inverse = inverse

        # Temperature scale
        T_scale = max(config.T_scale, 10.0)

        # Temperature networks
        self.T1_net = TemperatureNet(
            hidden_layers=hidden_layers, hidden_units=hidden_units,
            use_fourier=use_fourier, T_ref=config.T_init, T_scale=T_scale
        )
        self.T2_net = TemperatureNet(
            hidden_layers=hidden_layers, hidden_units=hidden_units,
            use_fourier=use_fourier, T_ref=config.T_init, T_scale=T_scale
        )

        # Contact resistance parameter
        Rc_true = config.Rc
        Rc_scale = max(Rc_true, 1e-6)
        self.Rc_scale = Rc_scale

        if inverse:
            # Trainable: initialize with a guess (possibly wrong)
            if Rc_init is None:
                Rc_init = Rc_true * 2.0  # start 2x off
            self.log_Rc = nn.Parameter(
                torch.tensor(np.log(Rc_init / Rc_scale), dtype=torch.float32)
            )
        else:
            # Fixed: use true value
            self.register_buffer(
                'log_Rc',
                torch.tensor(np.log(Rc_true / Rc_scale), dtype=torch.float32)
            )

        # Normalization constants
        self.x_scale = config.L
        self.t_scale = config.t_end

    @property
    def Rc(self) -> torch.Tensor:
        """Current contact resistance (always positive)."""
        return self.Rc_scale * torch.exp(self.log_Rc)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.x_scale

    def normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.t_scale

    def T1(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Temperature in layer 1 (dimensional x, t)."""
        x_n = self.normalize_x(x)
        t_n = self.normalize_t(t)
        return self.T1_net(x_n, t_n)

    def T2(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Temperature in layer 2 (dimensional x, t)."""
        x_n = self.normalize_x(x)
        t_n = self.normalize_t(t)
        return self.T2_net(x_n, t_n)

    def compute_pde_residual(self, x: torch.Tensor, t: torch.Tensor,
                              layer: int) -> torch.Tensor:
        """
        Compute PDE residual: rhocp * dT/dt - d/dx(k * dT/dx) = 0

        Parameters
        ----------
        x, t : torch.Tensor, shape (N, 1), dimensional
        layer : int, 1 or 2

        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
        """
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        if layer == 1:
            T = self.T1(x, t)
            k = self.k1
            rhocp = self.rhocp1
        else:
            T = self.T2(x, t)
            k = self.k2
            rhocp = self.rhocp2

        # First derivatives
        dT = torch.autograd.grad(T, [x, t], grad_outputs=torch.ones_like(T),
                                  create_graph=True)
        dT_dx = dT[0]
        dT_dt = dT[1]

        # Second derivative w.r.t. x
        d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx),
                                       create_graph=True)[0]

        # Residual: rhocp * dT/dt - k * d2T/dx2 = 0
        residual = rhocp * dT_dt - k * d2T_dx2

        # Normalize by characteristic scale
        scale = rhocp * (self.T1_net.T_scale) / self.t_scale
        return residual / max(scale, 1e-10)

    def compute_interface_residuals(self, t: torch.Tensor):
        """
        Compute interface conditions at x = a.

        Returns
        -------
        flux_residual : flux continuity violation
        jump_residual : temperature-jump law violation
        """
        x_a = torch.full_like(t, self.a).requires_grad_(True)
        t = t.requires_grad_(True)

        # T1 at x = a (from left)
        T1_a = self.T1(x_a, t)
        dT1_dx = torch.autograd.grad(T1_a, x_a, grad_outputs=torch.ones_like(T1_a),
                                      create_graph=True)[0]
        q1 = -self.k1 * dT1_dx  # flux from layer 1

        # T2 at x = a (from right)
        x_a2 = torch.full_like(t, self.a).requires_grad_(True)
        T2_a = self.T2(x_a2, t)
        dT2_dx = torch.autograd.grad(T2_a, x_a2, grad_outputs=torch.ones_like(T2_a),
                                      create_graph=True)[0]
        q2 = -self.k2 * dT2_dx  # flux into layer 2

        # Flux continuity: q1 = q2
        q_scale = self.k1 * self.T1_net.T_scale / self.x_scale
        flux_residual = (q1 - q2) / max(q_scale, 1e-10)

        # Temperature jump: T1(a) - T2(a) = Rc * q_int
        # Use average flux for q_int
        q_avg = 0.5 * (q1 + q2)
        T_jump = T1_a - T2_a
        jump_target = self.Rc * q_avg
        T_scale = max(self.T1_net.T_scale, 1.0)
        jump_residual = (T_jump - jump_target) / T_scale

        return flux_residual, jump_residual

    def compute_bc_residuals(self, t: torch.Tensor, q_in: torch.Tensor):
        """
        Compute boundary condition residuals.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
        q_in : torch.Tensor, shape (N, 1), heat flux at x=0

        Returns
        -------
        bc_left_residual : at x = 0
        bc_right_residual : at x = L
        """
        # Left BC: -k1 * dT1/dx(0) = q_in
        x_0 = torch.zeros_like(t).requires_grad_(True)
        t_bc = t.requires_grad_(True)
        T1_0 = self.T1(x_0, t_bc)
        dT1_dx_0 = torch.autograd.grad(T1_0, x_0, grad_outputs=torch.ones_like(T1_0),
                                         create_graph=True)[0]
        q_scale = self.k1 * self.T1_net.T_scale / self.x_scale
        bc_left = (-self.k1 * dT1_dx_0 - q_in) / max(q_scale, 1e-10)

        # Right BC: -k2 * dT2/dx(L) = h_R * (T2(L) - T_inf)
        x_L = torch.full_like(t, self.L).requires_grad_(True)
        T2_L = self.T2(x_L, t)
        dT2_dx_L = torch.autograd.grad(T2_L, x_L, grad_outputs=torch.ones_like(T2_L),
                                         create_graph=True)[0]
        q_scale2 = self.k2 * self.T2_net.T_scale / self.x_scale
        bc_right = (-self.k2 * dT2_dx_L - self.h_R * (T2_L - self.T_inf)) / max(q_scale2, 1e-10)

        return bc_left, bc_right

    def compute_ic_residual(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute initial condition residual at t = 0.

        Returns
        -------
        ic1_residual, ic2_residual
        """
        t_0 = torch.zeros_like(x1)
        T1_0 = self.T1(x1, t_0)
        T2_0 = self.T2(x2, t_0)

        T_scale = max(self.T1_net.T_scale, 1.0)
        ic1 = (T1_0 - self.T_init) / T_scale
        ic2 = (T2_0 - self.T_init) / T_scale

        return ic1, ic2

    def compute_data_loss(self, x_data: torch.Tensor, t_data: torch.Tensor,
                          T_data: torch.Tensor) -> torch.Tensor:
        """
        Compute observation data loss for inverse problems.

        Parameters
        ----------
        x_data : sensor x-coordinates
        t_data : observation times
        T_data : measured temperatures
        """
        # Determine which layer each point belongs to
        mask1 = (x_data < self.a).squeeze()
        mask2 = (~mask1)

        loss = torch.tensor(0.0)
        T_scale = max(self.T1_net.T_scale, 1.0)

        if mask1.any():
            T_pred1 = self.T1(x_data[mask1], t_data[mask1])
            loss = loss + torch.mean(((T_pred1 - T_data[mask1]) / T_scale) ** 2)

        if mask2.any():
            T_pred2 = self.T2(x_data[mask2], t_data[mask2])
            loss = loss + torch.mean(((T_pred2 - T_data[mask2]) / T_scale) ** 2)

        return loss

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict temperature at arbitrary (x, t) points (numpy interface).

        Automatically routes to T1 or T2 based on x position.
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            t_t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

            T_pred = torch.zeros_like(x_t)
            mask1 = (x_t < self.a).squeeze()
            mask2 = ~mask1

            if mask1.any():
                T_pred[mask1] = self.T1_net(
                    self.normalize_x(x_t[mask1]),
                    self.normalize_t(t_t[mask1])
                )
            if mask2.any():
                T_pred[mask2] = self.T2_net(
                    self.normalize_x(x_t[mask2]),
                    self.normalize_t(t_t[mask2])
                )

        return T_pred.numpy().flatten()

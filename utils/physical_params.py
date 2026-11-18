"""
Physical parameters and nondimensionalization for two-layer heat conduction
with thermal contact resistance.

Geometry:
    Layer 1: 0 < x < a
    Layer 2: a < x < L

Governing equations:
    rho_i * c_i * dT_i/dt = d/dx(k_i * dT_i/dx),  i = 1, 2

Interface conditions at x = a:
    q_int = -k1 * dT1/dx(a-) = -k2 * dT2/dx(a+)   (flux continuity)
    T1(a-) - T2(a+) = Rc * q_int                     (temperature jump)

Boundary conditions:
    x = 0: -k1 * dT1/dx(0) = q_in(t)                (prescribed heat flux)
    x = L: -k2 * dT2/dx(L) = h_R * (T2(L) - T_inf)  (convection)

Dimensionless groups:
    Fo_i = alpha_i * t / L^2       (Fourier number)
    Bi_R = h_R * L / k2            (Biot number at right boundary)
    R*   = Rc * k_ref / L          (dimensionless contact resistance)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class MaterialProperties:
    """Properties for a single layer."""
    rho: float       # density [kg/m^3]
    cp: float        # specific heat [J/(kg·K)]
    k: float         # thermal conductivity [W/(m·K)]

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)


@dataclass
class ProblemConfig:
    """Complete problem configuration."""
    # Geometry
    a: float = 0.005          # interface location [m] (layer 1 thickness)
    L: float = 0.010          # total thickness [m]

    # Material properties
    mat1: MaterialProperties = field(
        default_factory=lambda: MaterialProperties(rho=2700.0, cp=900.0, k=205.0)  # Aluminum
    )
    mat2: MaterialProperties = field(
        default_factory=lambda: MaterialProperties(rho=8000.0, cp=500.0, k=16.0)   # Stainless steel
    )

    # Contact resistance
    Rc: float = 1e-4          # thermal contact resistance [m^2·K/W]

    # Boundary conditions
    h_R: float = 50.0         # convection coefficient at x=L [W/(m^2·K)]
    T_inf: float = 300.0      # ambient temperature [K]
    T_init: float = 300.0     # initial temperature [K]

    # Heat flux input (default: step function)
    q_in_amplitude: float = 5e4  # heat flux amplitude [W/m^2]
    q_in_type: str = "step"      # "step", "pulse", "multi_pulse"
    pulse_duration: float = 0.5  # fraction of total time for pulse

    # Time
    t_end: float = 10.0       # simulation end time [s]

    def q_in(self, t: float) -> float:
        """Heat flux boundary condition at x=0."""
        if self.q_in_type == "step":
            return self.q_in_amplitude
        elif self.q_in_type == "pulse":
            t_pulse = self.pulse_duration * self.t_end
            return self.q_in_amplitude if t <= t_pulse else 0.0
        elif self.q_in_type == "multi_pulse":
            # Two pulses: on for 25%, off for 25%, on for 25%, off for 25%
            period = self.t_end / 2.0
            t_mod = t % period
            return self.q_in_amplitude if t_mod < period / 2.0 else 0.0
        else:
            return self.q_in_amplitude

    def q_in_array(self, t_arr: np.ndarray) -> np.ndarray:
        """Vectorized heat flux."""
        return np.array([self.q_in(ti) for ti in t_arr])

    @property
    def k_ref(self) -> float:
        """Reference conductivity for nondimensionalization."""
        return self.mat1.k

    @property
    def Fo1(self) -> float:
        """Fourier number for layer 1 at t_end."""
        return self.mat1.alpha * self.t_end / self.L**2

    @property
    def Fo2(self) -> float:
        """Fourier number for layer 2 at t_end."""
        return self.mat2.alpha * self.t_end / self.L**2

    @property
    def Bi_R(self) -> float:
        """Biot number at right boundary."""
        return self.h_R * self.L / self.mat2.k

    @property
    def R_star(self) -> float:
        """Dimensionless contact resistance."""
        return self.Rc * self.k_ref / self.L

    @property
    def T_scale(self) -> float:
        """Temperature scale for nondimensionalization."""
        return self.q_in_amplitude * self.L / self.k_ref

    def summary(self) -> str:
        """Print problem summary with dimensionless groups."""
        lines = [
            "=" * 60,
            "Problem Configuration",
            "=" * 60,
            f"Geometry: a = {self.a*1e3:.2f} mm, L = {self.L*1e3:.2f} mm",
            f"Layer 1: k={self.mat1.k:.1f} W/(m·K), alpha={self.mat1.alpha:.2e} m^2/s",
            f"Layer 2: k={self.mat2.k:.1f} W/(m·K), alpha={self.mat2.alpha:.2e} m^2/s",
            f"Contact resistance: Rc = {self.Rc:.2e} m^2·K/W",
            f"Heating: q_in = {self.q_in_amplitude:.1e} W/m^2 ({self.q_in_type})",
            f"Convection: h_R = {self.h_R:.1f} W/(m^2·K), T_inf = {self.T_inf:.1f} K",
            "-" * 40,
            "Dimensionless groups:",
            f"  Fo1 (at t_end) = {self.Fo1:.4f}",
            f"  Fo2 (at t_end) = {self.Fo2:.4f}",
            f"  Bi_R           = {self.Bi_R:.4f}",
            f"  R*             = {self.R_star:.4f}",
            f"  T_scale        = {self.T_scale:.2f} K",
            "=" * 60,
        ]
        return "\n".join(lines)


def default_config() -> ProblemConfig:
    """Standard test configuration."""
    return ProblemConfig()


def config_low_resistance() -> ProblemConfig:
    """Near-perfect contact for limit verification."""
    cfg = ProblemConfig()
    cfg.Rc = 1e-8
    return cfg


def config_high_resistance() -> ProblemConfig:
    """High contact resistance for strong temperature jump."""
    cfg = ProblemConfig()
    cfg.Rc = 1e-2
    return cfg


def config_pulse_heating() -> ProblemConfig:
    """Pulse heating for identifiability study."""
    cfg = ProblemConfig()
    cfg.q_in_type = "pulse"
    cfg.pulse_duration = 0.3
    return cfg


def config_multi_pulse() -> ProblemConfig:
    """Multi-pulse heating for richer transient."""
    cfg = ProblemConfig()
    cfg.q_in_type = "multi_pulse"
    return cfg

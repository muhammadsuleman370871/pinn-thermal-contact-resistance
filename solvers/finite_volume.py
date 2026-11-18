"""
Finite-volume reference solver for two-layer heat conduction with
thermal contact resistance at the interface.

Uses implicit (backward Euler) time stepping for unconditional stability.
The interface is handled by enforcing flux continuity and temperature-jump
conditions directly in the system matrix.

Grid layout:
    Layer 1: N1 cells in [0, a], cell centres x1_j = (j+0.5)*dx1
    Layer 2: N2 cells in [a, L], cell centres x2_j = a + (j+0.5)*dx2

Interface between last cell of layer 1 and first cell of layer 2 is
handled through a harmonic-mean + contact-resistance coupling.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.physical_params import ProblemConfig


@dataclass
class FVSolution:
    """Container for finite-volume solution."""
    x: np.ndarray          # cell-centre coordinates [m], shape (N,)
    t: np.ndarray          # time array [s], shape (Nt,)
    T: np.ndarray          # temperature field [K], shape (Nt, N)
    x1: np.ndarray         # layer 1 cell centres
    x2: np.ndarray         # layer 2 cell centres
    N1: int                # cells in layer 1
    N2: int                # cells in layer 2
    config: ProblemConfig  # problem configuration

    def temperature_at(self, x_query: float, t_query: float) -> float:
        """Interpolate temperature at arbitrary (x, t)."""
        # Find time bracket
        it = np.searchsorted(self.t, t_query, side='right') - 1
        it = max(0, min(it, len(self.t) - 2))
        wt = (t_query - self.t[it]) / (self.t[it+1] - self.t[it]) if it < len(self.t)-1 else 0.0

        # Find spatial bracket
        ix = np.searchsorted(self.x, x_query, side='right') - 1
        ix = max(0, min(ix, len(self.x) - 2))
        wx = (x_query - self.x[ix]) / (self.x[ix+1] - self.x[ix]) if ix < len(self.x)-1 else 0.0

        # Bilinear interpolation
        T00 = self.T[it, ix]
        T01 = self.T[it, ix+1] if ix+1 < len(self.x) else T00
        T10 = self.T[it+1, ix] if it+1 < len(self.t) else T00
        T11 = self.T[it+1, ix+1] if (it+1 < len(self.t) and ix+1 < len(self.x)) else T00

        return (1-wt)*((1-wx)*T00 + wx*T01) + wt*((1-wx)*T10 + wx*T11)

    def sensor_history(self, x_sensor: float) -> np.ndarray:
        """Get temperature history at a sensor location."""
        return np.array([self.temperature_at(x_sensor, ti) for ti in self.t])

    def interface_temperatures(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get T1(a-) and T2(a+) over time (extrapolated from cell centres)."""
        # Linear extrapolation from last two cells of layer 1
        T1_interface = self.T[:, self.N1-1] + (
            (self.T[:, self.N1-1] - self.T[:, self.N1-2])
            * 0.5  # half cell to face
        ) if self.N1 >= 2 else self.T[:, self.N1-1]

        # Linear extrapolation from first two cells of layer 2
        T2_interface = self.T[:, self.N1] - (
            (self.T[:, self.N1+1] - self.T[:, self.N1])
            * 0.5
        ) if self.N2 >= 2 else self.T[:, self.N1]

        return T1_interface, T2_interface

    def interface_flux(self) -> np.ndarray:
        """Compute interface heat flux over time."""
        cfg = self.config
        dx1 = cfg.a / self.N1
        # Flux from layer 1 side: -k1 * (T_face - T_last_cell) / (dx1/2)
        # Use cell values and interface condition
        T1_last = self.T[:, self.N1-1]
        T2_first = self.T[:, self.N1]
        dx2 = (cfg.L - cfg.a) / self.N2

        # From the coupling: q = (T1_last - T2_first) / (dx1/(2*k1) + Rc + dx2/(2*k2))
        R_total = dx1 / (2 * cfg.mat1.k) + cfg.Rc + dx2 / (2 * cfg.mat2.k)
        q_int = (T1_last - T2_first) / R_total
        return q_int

    def energy_balance(self) -> dict:
        """Check global energy conservation."""
        cfg = self.config
        dt_arr = np.diff(self.t)
        N = len(self.t)

        # Cumulative input heat at x=0
        q_in_arr = cfg.q_in_array(self.t)
        Q_in = np.trapezoid(q_in_arr, self.t)

        # Cumulative output heat at x=L (convection)
        T_right = self.T[:, -1]
        q_out_arr = cfg.h_R * (T_right - cfg.T_inf)
        Q_out = np.trapezoid(q_out_arr, self.t)

        # Stored energy change
        dx1 = cfg.a / self.N1
        dx2 = (cfg.L - cfg.a) / self.N2
        E_stored_1 = cfg.mat1.rho * cfg.mat1.cp * dx1 * np.sum(
            self.T[-1, :self.N1] - self.T[0, :self.N1]
        )
        E_stored_2 = cfg.mat2.rho * cfg.mat2.cp * dx2 * np.sum(
            self.T[-1, self.N1:] - self.T[0, self.N1:]
        )
        E_stored = E_stored_1 + E_stored_2

        imbalance = Q_in - Q_out - E_stored
        imbalance_pct = abs(imbalance) / max(abs(Q_in), 1e-30) * 100

        return {
            "Q_in": Q_in,
            "Q_out": Q_out,
            "E_stored": E_stored,
            "E_stored_1": E_stored_1,
            "E_stored_2": E_stored_2,
            "imbalance": imbalance,
            "imbalance_pct": imbalance_pct,
        }


def solve_fv(config: ProblemConfig, N1: int = 50, N2: int = 50,
             Nt: int = 500, verbose: bool = False) -> FVSolution:
    """
    Solve the two-layer heat conduction problem using implicit finite volumes.

    Parameters
    ----------
    config : ProblemConfig
        Physical and geometric parameters.
    N1 : int
        Number of cells in layer 1.
    N2 : int
        Number of cells in layer 2.
    Nt : int
        Number of time steps.
    verbose : bool
        Print progress.

    Returns
    -------
    FVSolution
        Solution container with temperature field.
    """
    a = config.a
    L = config.L
    k1 = config.mat1.k
    k2 = config.mat2.k
    rhocp1 = config.mat1.rho * config.mat1.cp
    rhocp2 = config.mat2.rho * config.mat2.cp
    Rc = config.Rc
    h_R = config.h_R
    T_inf = config.T_inf

    dx1 = a / N1
    dx2 = (L - a) / N2
    dt = config.t_end / Nt
    N = N1 + N2

    # Cell centres
    x1 = np.linspace(dx1/2, a - dx1/2, N1)
    x2 = np.linspace(a + dx2/2, L - dx2/2, N2)
    x = np.concatenate([x1, x2])

    # Time array (store every step)
    t_arr = np.linspace(0, config.t_end, Nt + 1)

    # Initialize temperature
    T = np.full(N, config.T_init)
    T_history = np.zeros((Nt + 1, N))
    T_history[0, :] = T.copy()

    # Build coefficient matrix (implicit Euler)
    # For each interior cell in layer i:
    #   rhocp_i * V * (T^{n+1} - T^n) / dt = sum of face fluxes
    # Interior face flux: k * (T_neighbor - T_cell) / dx

    for n in range(Nt):
        t_now = t_arr[n + 1]
        q_in = config.q_in(t_now)

        # Build sparse system A * T^{n+1} = b
        rows, cols, vals = [], [], []
        b = np.zeros(N)

        for i in range(N):
            if i < N1:
                # Layer 1
                rhocp = rhocp1
                dx = dx1
                k = k1
            else:
                # Layer 2
                rhocp = rhocp2
                dx = dx2
                k = k2

            # Diagonal: transient term
            a_diag = rhocp * dx / dt

            # Left face
            if i == 0:
                # x=0: prescribed flux q_in (positive into domain)
                # Flux = q_in (already added to RHS)
                b[i] += q_in
            elif i == N1:
                # First cell of layer 2: left face is the interface
                # Interface coupling: q = (T_{N1-1} - T_{N1}) / R_int
                # R_int = dx1/(2*k1) + Rc + dx2/(2*k2)
                R_int = dx1 / (2*k1) + Rc + dx2 / (2*k2)
                coeff_int = 1.0 / R_int
                a_diag += coeff_int
                rows.append(i); cols.append(N1-1); vals.append(-coeff_int)
            else:
                # Interior left face
                if i < N1:
                    coeff_L = k1 / dx1
                else:
                    coeff_L = k2 / dx2
                a_diag += coeff_L
                rows.append(i); cols.append(i-1); vals.append(-coeff_L)

            # Right face
            if i == N - 1:
                # x=L: convection BC
                # q_out = h_R * (T_N - T_inf) outward
                # Energy: -h_R * (T_N - T_inf)  (loss)
                a_diag += h_R
                b[i] += h_R * T_inf
            elif i == N1 - 1:
                # Last cell of layer 1: right face is the interface
                R_int = dx1 / (2*k1) + Rc + dx2 / (2*k2)
                coeff_int = 1.0 / R_int
                a_diag += coeff_int
                rows.append(i); cols.append(N1); vals.append(-coeff_int)
            else:
                # Interior right face
                if i < N1:
                    coeff_R = k1 / dx1
                else:
                    coeff_R = k2 / dx2
                a_diag += coeff_R
                rows.append(i); cols.append(i+1); vals.append(-coeff_R)

            # Diagonal entry
            rows.append(i); cols.append(i); vals.append(a_diag)

            # RHS: transient term
            b[i] += rhocp * dx / dt * T[i]

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        T = spsolve(A, b)
        T_history[n + 1, :] = T.copy()

        if verbose and (n + 1) % (Nt // 10) == 0:
            print(f"  Step {n+1}/{Nt}, t={t_now:.4f}s, "
                  f"T_max={T.max():.2f}K, T_min={T.min():.2f}K")

    return FVSolution(
        x=x, t=t_arr, T=T_history,
        x1=x1, x2=x2, N1=N1, N2=N2,
        config=config
    )


def convergence_study(config: ProblemConfig,
                      N_list: List[int] = [25, 50, 100, 200],
                      Nt_factor: int = 10) -> dict:
    """
    Run grid convergence study.

    Parameters
    ----------
    config : ProblemConfig
    N_list : list of int
        Number of cells per layer to test.
    Nt_factor : int
        Nt = N * Nt_factor.

    Returns
    -------
    dict with grid sizes and key output metrics.
    """
    results = {"N": [], "T_max": [], "T_interface_jump": [],
               "Q_in": [], "imbalance_pct": []}

    for N in N_list:
        sol = solve_fv(config, N1=N, N2=N, Nt=N*Nt_factor)
        T1_int, T2_int = sol.interface_temperatures()
        eb = sol.energy_balance()

        results["N"].append(N)
        results["T_max"].append(sol.T[-1].max())
        results["T_interface_jump"].append(np.mean(T1_int[-10:] - T2_int[-10:]))
        results["Q_in"].append(eb["Q_in"])
        results["imbalance_pct"].append(eb["imbalance_pct"])

    return results


if __name__ == "__main__":
    from utils.physical_params import default_config

    cfg = default_config()
    print(cfg.summary())

    print("\nSolving with default configuration...")
    sol = solve_fv(cfg, N1=100, N2=100, Nt=1000, verbose=True)

    eb = sol.energy_balance()
    print(f"\nEnergy balance:")
    print(f"  Q_in       = {eb['Q_in']:.4f} J/m^2")
    print(f"  Q_out      = {eb['Q_out']:.4f} J/m^2")
    print(f"  E_stored   = {eb['E_stored']:.4f} J/m^2")
    print(f"  Imbalance  = {eb['imbalance_pct']:.4f}%")

    T1_int, T2_int = sol.interface_temperatures()
    print(f"\nInterface jump at t_end: {T1_int[-1] - T2_int[-1]:.4f} K")


# dcn_core.py
"""
Dynamic Chaos Net (DCN)
=======================

A 9-dimensional ODE-based architecture for adaptive classification and control.
Implements Trinomial Repression Dynamics with Rigor-Dyn Learning.

Author: Hari Hardiyan
License: MIT
"""

import numpy as np
from scipy.integrate import odeint
import logging
from typing import Tuple, List

# -----------------------------------------------------------------------------
# Global Parameters (configurable)
# -----------------------------------------------------------------------------
MU_LEARNING_RATE: float = 0.001
TARGET_HIGH: float = 10.0
TARGET_LOW: float = 0.1
LLE_THRESHOLD: float = 0.0600
D_RATE: float = 0.005  # De-Excitation Rate

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------------------------
# ODE System
# -----------------------------------------------------------------------------
def rigor_dyn_ode_9d(
    Y: np.ndarray,
    t: float,
    alpha: float,
    gamma: float,
    eta: float,
    beta: float,
    beta_noise: float,
    D_rate: float,
    R_rec: float,
    S_filtered: float,
    Noise_rate: float,
    C_target: np.ndarray
) -> np.ndarray:
    """
    9D ODE system for Dynamic Chaos Net (DCN).

    Parameters
    ----------
    Y : np.ndarray
        State vector [C1, C2, C3, K12, K13, K21, K23, K31, K32].
    t : float
        Time variable.
    alpha, gamma, eta, beta, beta_noise, D_rate, R_rec, S_filtered, Noise_rate : float
        System parameters.
    C_target : np.ndarray
        Target vector for states [C1_target, C2_target, C3_target].

    Returns
    -------
    np.ndarray
        Derivatives [dC1dt, dC2dt, dC3dt, dK12dt, dK13dt, dK21dt, dK23dt, dK31dt, dK32dt].
    """
    if Y.shape[0] != 9:
        raise ValueError("State vector Y must have length 9.")

    C1, C2, C3 = Y[0:3]
    K12, K13, K21, K23, K31, K32 = Y[3:9]

    # Error terms
    E_C1 = C1 - C_target[0]
    E_C2 = C2 - C_target[1]
    E_C3 = C3 - C_target[2]

    # State dynamics
    dC1dt = alpha / (1 + (C2 / K12) ** eta + (C3 / K13) ** eta) + beta * S_filtered - gamma * C1
    dC2dt = alpha / (1 + (C1 / K21) ** eta + (C3 / K23) ** eta) - gamma * C2 + R_rec
    dC3dt = alpha / (1 + (C1 / K31) ** eta + (C2 / K32) ** eta) + beta_noise * Noise_rate - gamma * C3 - D_rate * C3

    # Learning dynamics
    dK12dt = -MU_LEARNING_RATE * C1 * E_C2
    dK13dt = -MU_LEARNING_RATE * C1 * E_C3
    dK21dt = -MU_LEARNING_RATE * C2 * E_C1
    dK23dt = -MU_LEARNING_RATE * C2 * E_C3
    dK31dt = -MU_LEARNING_RATE * C3 * E_C1
    dK32dt = -MU_LEARNING_RATE * C3 * E_C2

    return np.array([dC1dt, dC2dt, dC3dt, dK12dt, dK13dt, dK21dt, dK23dt, dK31dt, dK32dt])

# -----------------------------------------------------------------------------
# Dynamic Chaos Net Class
# -----------------------------------------------------------------------------
class DynamicChaosNet:
    """
    Dynamic Chaos Net (DCN): Adaptive control system based on 9D ODEs.

    Attributes
    ----------
    alpha, gamma, eta, beta, beta_noise, D_rate : float
        System parameters.
    t : np.ndarray
        Time vector for integration.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        eta: float,
        beta: float,
        beta_noise: float,
        D_rate: float = D_RATE,
        t_max: float = 100.0,
        n_points: int = 2000
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        self.beta_noise = beta_noise
        self.D_rate = D_rate
        self.t = np.linspace(0, t_max, n_points)

    def train_step(
        self,
        Y0_start: List[float],
        S_input: float,
        Noise_input: float,
        R_rec_input: float,
        target_index: int,
        epoch_name: str = "TEST"
    ) -> Tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        """
        Run one training step of DCN.

        Parameters
        ----------
        Y0_start : list of float
            Initial state vector (length 9).
        S_input : float
            Filtered signal input.
        Noise_input : float
            Noise input.
        R_rec_input : float
            Recovery boost input.
        target_index : int
            Index of target state (0=C1, 1=C2, 2=C3).
        epoch_name : str
            Label for logging.

        Returns
        -------
        Y_final : np.ndarray
            Final state vector.
        success : bool
            Whether classification matched target.
        C_final : np.ndarray
            Final C states [C1, C2, C3].
        K_final : np.ndarray
            Final K thresholds [K12, K13, K21, K23, K31, K32].
        """
        if len(Y0_start) != 9:
            raise ValueError("Initial state Y0_start must have length 9.")

        C_target = np.full(3, TARGET_LOW)
        C_target[target_index] = TARGET_HIGH
        Y0 = np.array(Y0_start)

        ode_params = (
            self.alpha, self.gamma, self.eta, self.beta, self.beta_noise,
            self.D_rate, R_rec_input, S_input, Noise_input, C_target
        )

        solution = odeint(rigor_dyn_ode_9d, Y0, self.t, args=ode_params)

        Y_final = solution[-1, :]
        C_final = Y_final[0:3]
        K_final = Y_final[3:9]

        winner_actual = int(np.argmax(C_final))
        success = (winner_actual == target_index)

        logging.info(
            "| EPOCH: %-10s | Target: C%d | R_Rec: %.3f | C_FINAL: [%.4f, %.4f, %.4f] | SUCCESS: %s",
            epoch_name, target_index + 1, R_rec_input, C_final[0], C_final[1], C_final[2], success
        )

        return Y_final, success, C_final, K_final

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example initial state
    Y0 = [1.0, 0.5, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    dcn = DynamicChaosNet(alpha=2.0, gamma=0.5, eta=2.0, beta=0.1, beta_noise=0.05)

    Y_final, success, C_final, K_final = dcn.train_step(
        Y0_start=Y0,
        S_input=0.2,
        Noise_input=0.1,
        R_rec_input=0.5,
        target_index=1,
        epoch_name="DEMO"
    )

    print("Final C states:", C_final)
    print("Final K thresholds:", K_final)

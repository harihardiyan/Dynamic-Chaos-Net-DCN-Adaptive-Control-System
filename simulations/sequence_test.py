# sequence_test.py
"""
Dynamic Chaos Net (DCN) Sequence Test
=====================================

Self-contained script: defines DynamicChaosNet class and runs
a sequence of scenarios (Normal, Chaos, Anomaly, Recovery),
then plots the evolution of C1, C2, C3 states.
"""

import numpy as np
from scipy.integrate import odeint
import logging
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global Parameters
# -----------------------------------------------------------------------------
MU_LEARNING_RATE = 0.001
TARGET_HIGH = 10.0
TARGET_LOW = 0.1
D_RATE = 0.005  # De-excitation rate

# -----------------------------------------------------------------------------
# ODE System
# -----------------------------------------------------------------------------
def rigor_dyn_ode_9d(Y, t, alpha, gamma, eta, beta, beta_noise,
                     D_rate, R_rec, S_filtered, Noise_rate, C_target):
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

    return np.array([dC1dt, dC2dt, dC3dt,
                     dK12dt, dK13dt, dK21dt, dK23dt, dK31dt, dK32dt])

# -----------------------------------------------------------------------------
# Dynamic Chaos Net Class
# -----------------------------------------------------------------------------
class DynamicChaosNet:
    def __init__(self, alpha, gamma, eta, beta, beta_noise,
                 D_rate=D_RATE, t_max=100.0, n_points=2000):
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        self.beta_noise = beta_noise
        self.D_rate = D_rate
        self.t = np.linspace(0, t_max, n_points)

    def train_step(self, Y0_start, S_input, Noise_input,
                   R_rec_input, target_index, epoch_name="TEST"):
        if len(Y0_start) != 9:
            raise ValueError("Initial state must have length 9.")

        C_target = np.full(3, TARGET_LOW)
        C_target[target_index] = TARGET_HIGH
        Y0 = np.array(Y0_start)

        ode_params = (self.alpha, self.gamma, self.eta, self.beta,
                      self.beta_noise, self.D_rate, R_rec_input,
                      S_input, Noise_input, C_target)

        solution = odeint(rigor_dyn_ode_9d, Y0, self.t, args=ode_params)

        Y_final = solution[-1, :]
        C_final = Y_final[0:3]
        K_final = Y_final[3:9]

        winner_actual = int(np.argmax(C_final))
        success = (winner_actual == target_index)

        logging.info(
            "| EPOCH: %-10s | Target: C%d | R_Rec: %.3f | "
            "C_FINAL: [%.4f, %.4f, %.4f] | SUCCESS: %s",
            epoch_name, target_index + 1, R_rec_input,
            C_final[0], C_final[1], C_final[2], success
        )

        return Y_final, success, C_final, K_final

# -----------------------------------------------------------------------------
# Sequence Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parameters
    ALPHA = 10.0; GAMMA = 1.0; ETA = 4.0
    BETA = 120.0; BETA_NOISE = 50.0; D_RATE = 0.005
    R_REC_BOOST = 5.0; LLE_THRESHOLD = 0.0600
    Y0_INIT = [4.0, 6.0, 4.0] + [5.0] * 6

    dcn = DynamicChaosNet(ALPHA, GAMMA, ETA, BETA, BETA_NOISE, D_RATE)

    print("===========================================================")
    print(f"## 🚀 DCN: FINAL CONTROL TEST (R_Rec={R_REC_BOOST}) ##")
    print("===========================================================")

    Y_current = Y0_INIT
    C_history = []; labels = []

    # Test 1: NORMAL
    Y_current, _, C_final, _ = dcn.train_step(Y_current,
        S_input=max(0.0, 0.0500 - LLE_THRESHOLD),
        Noise_input=0.0001, R_rec_input=0.0,
        target_index=1, epoch_name="NORMAL")
    C_history.append(C_final); labels.append("NORMAL")

    # Test 2: CHAOS
    Y_current, _, C_final, _ = dcn.train_step(Y_current,
        S_input=max(0.0, 0.0800 - LLE_THRESHOLD),
        Noise_input=0.0005, R_rec_input=0.0,
        target_index=0, epoch_name="CHAOS")
    C_history.append(C_final); labels.append("CHAOS")

    # Test 3: ANOMALY
    Y_current, _, C_final, _ = dcn.train_step(Y_current,
        S_input=max(0.0, 0.0400 - LLE_THRESHOLD),
        Noise_input=0.1500, R_rec_input=0.0,
        target_index=2, epoch_name="ANOMALY")
    C_history.append(C_final); labels.append("ANOMALY")

    # Test 4: RECOVERY
    Y_current, success, C_final, K_final = dcn.train_step(Y_current,
        S_input=max(0.0, 0.0550 - LLE_THRESHOLD),
        Noise_input=0.0001, R_rec_input=R_REC_BOOST,
        target_index=1, epoch_name="RECOVERY")
    C_history.append(C_final); labels.append("RECOVERY")

    print("-----------------------------------------------------------")
    print(f"| FINAL K (Weights) after Sequence: {K_final}")
    print(f"| **FORCED RECOVERY SUCCESS (C2 Win): {success}**")
    print("===========================================================")

    # Plotting
    C_history = np.array(C_history)
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.plot(x, C_history[:, 0], marker='o', label="C1 (Chaos)")
    plt.plot(x, C_history[:, 1], marker='o', label="C2 (Normal)")
    plt.plot(x, C_history[:, 2], marker='o', label="C3 (Anomaly)")
    plt.xticks(x, labels)
    plt.xlabel("Test Sequence"); plt.ylabel("Final C State Value")
    plt.title("DCN Sequence Test: Evolution of C States")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

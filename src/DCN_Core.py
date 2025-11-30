# DCN_Core.py

import numpy as np
from scipy.integrate import odeint

# ==============================================================================
# RIGOR-DYN LEARNING & GLOBAL PARAMETERS
# ==============================================================================
MU_LEARNING_RATE = 0.001 
TARGET_HIGH = 10.0 
TARGET_LOW = 0.1   
LLE_THRESHOLD = 0.0600
D_RATE = 0.005 # C3 De-Excitation Rate
# R_REC_BOOST is defined in the execution script (sequence_test.py)

# ==============================================================================
# 9D ODE SYSTEM (The Core Dynamics with Recovery D and R)
# ==============================================================================

def rigor_dyn_ode_9d_final_control(Y, t, alpha, gamma, eta, beta, beta_noise, D_rate, R_rec, S_filtered, Noise_rate, C_target):
    """
    The 9D ODE System for the Dynamic Chaos Net (DCN), including learning and control terms.
    Y = [C1, C2, C3, K12, K13, K21, K23, K31, K32]
    """
    C1, C2, C3 = Y[0:3]
    K12, K13, K21, K23, K31, K32 = Y[3:9]
    
    # Error Calculation: E = Actual - Target
    E_C1 = C1 - C_target[0]
    E_C2 = C2 - C_target[1]
    E_C3 = C3 - C_target[2]
    
    # 2. ODEs for States (dC/dt)
    dC1dt = alpha / (1 + (C2 / K12)**eta + (C3 / K13)**eta) + beta * S_filtered - gamma * C1
    dC2dt = alpha / (1 + (C1 / K21)**eta + (C3 / K23)**eta) - gamma * C2 + R_rec # R_rec boost
    dC3dt = alpha / (1 + (C1 / K31)**eta + (C2 / K32)**eta) + beta_noise * Noise_rate - gamma * C3 - D_rate * C3 # D_rate decay
    
    # 3. ODEs for Learning/Weights (dK_ij/dt) - RIGOR-DYN LEARNING Rule
    dK12dt = -MU_LEARNING_RATE * C1 * E_C2 
    dK13dt = -MU_LEARNING_RATE * C1 * E_C3 
    dK21dt = -MU_LEARNING_RATE * C2 * E_C1 
    dK23dt = -MU_LEARNING_RATE * C2 * E_C3
    dK31dt = -MU_LEARNING_RATE * C3 * E_C1
    dK32dt = -MU_LEARNING_RATE * C3 * E_C2

    return np.array([dC1dt, dC2dt, dC3dt, dK12dt, dK13dt, dK21dt, dK23dt, dK31dt, dK32dt])

# ==============================================================================
# DCN LEARNING CORE CLASS
# ==============================================================================

class DynamicChaosNet:
    def __init__(self, alpha, gamma, eta, beta, beta_noise, D_rate):
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        self.beta_noise = beta_noise
        self.D_rate = D_rate  
        self.t = np.linspace(0, 100, 2000)

    def train_step(self, Y0_start, S_input, Noise_input, R_rec_input, target_index, epoch_name="TEST"):
        
        C_target = np.full(3, TARGET_LOW)
        C_target[target_index] = TARGET_HIGH
        Y0 = np.array(Y0_start) 
        
        ode_params = (self.alpha, self.gamma, self.eta, self.beta, self.beta_noise, self.D_rate, R_rec_input, S_input, Noise_input, C_target)
        solution = odeint(rigor_dyn_ode_9d_final_control, Y0, self.t, args=ode_params)
        
        Y_final = solution[-1, :]
        C_final = Y_final[0:3]
        K_final = Y_final[3:9]
        
        winner_actual = np.argmax(C_final)
        success = (winner_actual == target_index)
        
        print(f"| EPOCH: {epoch_name:<10} (Target C{target_index+1}) | R_Rec: {R_rec_input:.1f} | C_FINAL: [{C_final[0]:.4f}, {C_final[1]:.4f}, {C_final[2]:.4f}] | SUCCESS: {success}")
        
        return Y_final, success, C_final, K_final

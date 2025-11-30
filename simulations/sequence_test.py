# sequence_test.py
# This script executes the final DCN test suite using DCN_Core.py

import sys
import os
import numpy as np

# Add the src directory to the system path to import DCN_Core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from DCN_Core import DynamicChaosNet # Import the core class

# ==============================================================================
# TEST PARAMETERS
# ==============================================================================
# DCN Dynamics Parameters
ALPHA = 10.0
GAMMA = 1.0
ETA = 4.0
BETA = 120.0
BETA_NOISE = 50.0
D_RATE = 0.005 # C3 De-excitation rate

# Test Control Parameters
R_REC_BOOST = 5.0 # External boost applied to C2 during recovery
LLE_THRESHOLD = 0.0600
Y0_INIT = [4.0, 6.0, 4.0] + [5.0] * 6 # Initial C and K values

# ==============================================================================
# DYNAMIC SEQUENCE TEST EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # Initialize DCN
    DCN = DynamicChaosNet(ALPHA, GAMMA, ETA, BETA, BETA_NOISE, D_RATE)

    print("===========================================================")
    print(f"## 🚀 DCN: FINAL CONTROL TEST (R_Rec={R_REC_BOOST}) ##")
    print("===========================================================")

    Y_current = Y0_INIT

    # --- TEST 1: NORMAL OPERATION ---
    S_NORMAL = max(0.0, 0.0500 - LLE_THRESHOLD) 
    NOISE_NORMAL = 0.0001
    R_REC_NORMAL = 0.0
    
    Y_current, _, _, _ = DCN.train_step(
        Y_current, S_input=S_NORMAL, Noise_input=NOISE_NORMAL, R_rec_input=R_REC_NORMAL, target_index=1, epoch_name="NORMAL"
    )

    # --- TEST 2: CHAOS SPIKE ---
    S_CHAOS = max(0.0, 0.0800 - LLE_THRESHOLD) 
    NOISE_CHAOS = 0.0005
    R_REC_CHAOS = 0.0
    
    Y_current, _, _, _ = DCN.train_step(
        Y_current, S_input=S_CHAOS, Noise_input=NOISE_CHAOS, R_rec_input=R_REC_CHAOS, target_index=0, epoch_name="CHAOS"
    )

    # --- TEST 3: ANOMALY SPIKE (Creating C3 bias) ---
    S_ANOMALY = max(0.0, 0.0400 - LLE_THRESHOLD)
    NOISE_ANOMALY = 0.1500 
    R_REC_ANOMALY = 0.0
    
    Y_current, _, _, _ = DCN.train_step(
        Y_current, S_input=S_ANOMALY, Noise_input=NOISE_ANOMALY, R_rec_input=R_REC_ANOMALY, target_index=2, epoch_name="ANOMALY"
    )

    # --- TEST 4: FORCED RECOVERY (Applying R_REC_BOOST) ---
    S_RECOVERY = max(0.0, 0.0550 - LLE_THRESHOLD) 
    NOISE_RECOVERY = 0.0001
    R_REC_RECOVERY = R_REC_BOOST 
    
    Y_current, success, C_final, K_final = DCN.train_step(
        Y_current, S_input=S_RECOVERY, Noise_input=NOISE_RECOVERY, R_rec_input=R_REC_RECOVERY, target_index=1, epoch_name="RECOVERY"
    )

    print("-----------------------------------------------------------")
    print(f"| FINAL K (Weights) after Sequence: {K_final}")
    print(f"| **FORCED RECOVERY SUCCESS (C2 Win): {success}**")
    print("===========================================================")

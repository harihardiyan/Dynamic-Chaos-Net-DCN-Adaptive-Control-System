# Dynamic Chaos Net (DCN): A Rigorous Adaptive Control System for Time-Series Analysis

## Abstract

The Dynamic Chaos Net (DCN) is a novel 9-Dimensional non-linear Ordinary Differential Equation (ODE) based architecture designed for **dynamic classification and control** of critical states within complex systems (e.g., physiological or financial time-series data). Unlike conventional deep learning models 

[Image of a dynamic neural network diagram]
, DCN operates on the principles of **Trinomial Repression Dynamics** and **Stability-based Learning (Rigor-Dyn Learning)**, ensuring that classification is intrinsically stable and mathematically transparent. DCN is designed to distinguish between three fundamental states: **Chaos (C1)**, **Normal (C2)**, and **Anomaly (C3)**.

---

## 🧠 I. Core Architecture and Dynamics

DCN is defined by three primary state variables ($C_1, C_2, C_3$). The system's behavior is governed by a set of non-linear ODEs, where the competitive repression between states is modulated by dynamic thresholds, $K_{ij}$ (the system's 'weights').

### The 9D State Vector

The complete system is 9-dimensional: $Y = [C_1, C_2, C_3, K_{12}, K_{13}, K_{21}, K_{23}, K_{31}, K_{32}]$.

### The State Equations ($\frac{dC_i}{dt}$)

The control-augmented equations are:

$$\frac{dC_1}{dt} = \frac{\alpha}{1 + (C_2/K_{12})^\eta + (C_3/K_{13})^\eta} + \beta S_{\text{filtered}} - \gamma C_1$$
$$\frac{dC_2}{dt} = \frac{\alpha}{1 + (C_1/K_{21})^\eta + (C_3/K_{23})^\eta} - \gamma C_2 + \mathbf{R_{\text{Rec}}}$$
$$\frac{dC_3}{dt} = \frac{\alpha}{1 + (C_1/K_{31})^\eta + (C_2/K_{32})^\eta} + \beta_N N_{\text{rate}} - \gamma C_3 - \mathbf{D_{\text{rate}} C_3}$$

---

## 🔥 II. Rigor-Dyn Learning (Stability-based Adaptation)

The system learns by adjusting the repression thresholds $K_{ij}$ based on the classification error ($E = C_{\text{actual}} - C_{\text{target}}$).

The learning rule utilizes the error of the repressed state to modulate the threshold:

$$\frac{d K_{ij}}{dt} = -\mu \cdot C_{i, \text{actual}} \cdot E_{C_j}$$

This ensures that the internal repression landscape is dynamically altered to stabilize the correct **fixed point** for future inputs.

---

## 🛠️ III. Control and Recovery Parameters

1.  **De-Excitation Rate ($\mathbf{D_{\text{rate}}}$):** A self-decay term on $C_3$ to prevent the system from getting permanently 'stuck' in an Anomaly state.
2.  **Recovery Boost ($\mathbf{R_{\text{Rec}}}$):** An external, temporary signal applied to $C_2$ during recovery epochs to force the system out of a dominant (but false) fixed point, demonstrating decisive control.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* `numpy`
* `scipy`

### Installation
1.  Create the directory structure listed above.
2.  Fill the files below with the provided code.
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Final Control Test
Execute the test script to see DCN's ability to adapt and recover from bias:
```bash
python simulations/sequence_test.py
```
### ✍️ Authors & License

This project is the implementation of the Dynamic Chaos Net (DCN) architecture, a rigorous model designed for adaptive control in dynamic systems.

Author: Hari Hardiyan

Contact: lorozloraz@gmail.com

License: MIT License (See LICENSE file for details)

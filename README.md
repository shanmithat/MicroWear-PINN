# MicroWear-PINN

Physics-Informed Neural Network (PINN) framework designed to model micro-scale abrasive/sliding wear evolution, surface profile degradation, and subsurface contact stress fields over sliding cycles.

## Physical & Mathematical Formulation

MicroWear-PINN solves the coupled contact mechanics and wear problem on an elastic half-space $\Omega = [-L, L] \times [-D, 0]$ over sliding time $t \in [0, T]$ using three key physical laws:

1. **Kinematic Wear Law (Archard's Differential Extension)**:
   $$\frac{\partial h(x, t)}{\partial t} = -k_w(x) \cdot p(x, t) \cdot v_{rel}(t)$$
   where $h(x, t)$ is the surface height profile, $k_w(x)$ is the local wear coefficient field, $p(x, t)$ is the contact pressure distribution, and $v_{rel}$ is relative sliding velocity.

2. **2D Elastic Half-Space Subsurface Stress Field**:
   Governed by the biharmonic equation of the Airy stress function $\Phi(x, y, t)$:
   $$\nabla^4 \Phi = 0$$
   where stresses are given by:
   $$\sigma_{xx} = \frac{\partial^2 \Phi}{\partial y^2}, \quad \sigma_{yy} = \frac{\partial^2 \Phi}{\partial x^2}, \quad \tau_{xy} = -\frac{\partial^2 \Phi}{\partial x \partial y}$$

3. **Boundary Conditions (Contact Interface $y=0$)**:
   - Normal contact: $\sigma_{yy}(x, 0, t) = -p(x, t)$
   - Tangential shear: $\tau_{xy}(x, 0, t) = -\mu \cdot p(x, t) \cdot \text{sgn}(v_{rel})$

---

## Key Features

- **Decoupled Physical Architecture**: Employs three independent sub-networks ($k_w(x)$, $h(x, t)$, $\Phi(x, y, t)$) to structurally enforce spatial-temporal properties by construction rather than relying solely on loss components.
- **Random Fourier Feature (RFF) Projection**: Captures micro-scale contact roughness grooves and overcomes neural network spectral bias by mapping coordinates to high-frequency harmonic features.
- **Efficient Biharmonic Operator**: Uses the Laplacian-of-Laplacian method ($\nabla^2 (\nabla^2 \Phi)$) reducing backpropagation graph traversals by 20% compared to expanded fourth-order partial derivative formulations.
- **Adaptive Weight Balancing (SoftAdapt)**: Dynamically weights multi-objective losses (wear PDE, biharmonic PDE, boundary conditions, initial conditions, and sparse data) depending on their relative convergence rates.
- **Quasi-Monte Carlo (QMC) Sampler**: Supports Sobol sequences and Latin Hypercube Sampling (LHS) for space-time collocation grids.
- **Dual-Stage Optimization**: Employs **AdamW** for global initial alignment followed by **L-BFGS-B** for high-precision local PDE residual minimization.

---

## Repository Structure

```text
microwear_pinn/
├── configs/
│   ├── default_config.yaml   # Default training/model/physics hyper-parameters
│   └── fast_config.yaml      # Light configuration for rapid verification
├── src/
│   ├── __init__.py
│   ├── model.py              # MLP with Random Fourier Feature (RFF) embeddings
│   ├── physics.py            # Autograd PDE residual operators (Biharmonic, Archard wear)
│   ├── dataset.py            # Collocation samplers (Sobol/LHS) & metrology loaders
│   ├── trainer.py            # Dual-stage training loop & SoftAdapt weight updates
│   └── utils.py              # Profilometry & stress visualization plotting utilities
├── tests/
│   └── test_residuals.py     # Unit tests verifying PDE residual gradients
├── requirements.txt
└── main.py                   # CLI entrypoint for training, evaluation, and testing
```

---

## Installation

Ensure you have Python 3.8+ and PyTorch installed. Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run Verification Unit Tests
To verify the autograd derivatives against exact symbolic solutions:
```bash
python main.py test
```

### 2. Train the Model (Benchmark Scenario)
Train the network on the exact sinusoidal benchmark profile:
```bash
python main.py train --save-dir results
```
This will train the model, save weights to `results/microwear_pinn_model.pt`, and generate plots for:
- Loss convergence history & SoftAdapt weight adjustments
- Wear profile degradation over time
- Subsurface stress field contour maps ($\sigma_{xx}, \sigma_{yy}, \tau_{xy}$, and Von Mises stress)
- Identified spatial wear coefficient field $k_w(x)$

### 3. Evaluate a Trained Checkpoint
Run inference and visualize results for a saved model:
```bash
python main.py evaluate --checkpoint results/microwear_pinn_model.pt --save-dir eval_results
```

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
- **NASA Milling Data Calibration**: Ingests spindle current RMS and cutting parameters from the UC Berkeley/NASA milling dataset to estimate cutting forces, couples contact area growth to continuous contact pressure feedback, and distributes pressure across a dynamic Hertzian contact patch:
  $$p_{ext}(x, t) = p_{peak}(t) \sqrt{\max\left(0, 1 - \left(\frac{x}{a(t)}\right)^2\right)}$$
  where $a(t) = VB(t)/2 + 0.05\text{ mm}$.
- **ONNX Web Assembly Engine**: Exports PyTorch weights to ONNX format with dynamic batch sizes and evaluates model predictions in real-time in the browser using WebAssembly.
- **Zero-Backend SPA Web App**: Real-time interactive UI utilizing Tailwind CSS, Plotly.js, and ONNX Runtime Web. Estimates subsurface stresses from the ONNX Airy potential output via central finite differences on the fly.

---

## Repository Structure

```text
microwear_pinn/
├── configs/
│   ├── default_config.yaml   # Default training/model/physics hyper-parameters
│   ├── fast_config.yaml      # Light configuration for rapid verification
│   └── nasa_config.yaml      # Configuration for NASA milling calibration
├── src/
│   ├── __init__.py
│   ├── model.py              # MLP with Random Fourier Feature (RFF) embeddings
│   ├── physics.py            # Autograd PDE residual operators (Biharmonic, Archard wear)
│   ├── dataset.py            # Collocation samplers (Sobol/LHS) & metrology loaders
│   ├── trainer.py            # Dual-stage training loop & SoftAdapt weight updates
│   ├── nasa_milling_loader.py# Ingests UC Berkeley/NASA milling MAT files
│   ├── export_onnx.py        # Compiles and validates PyTorch model to ONNX
│   └── utils.py              # Profilometry & stress visualization plotting utilities
├── tests/
│   └── test_residuals.py     # Unit tests verifying PDE residual gradients
├── .github/workflows/
│   └── deploy.yml            # CI/CD Page deployment pipeline
├── index.html                # Zero-backend interactive web application
├── model/
│   ├── microwear_pinn.onnx   # Compiled ONNX model
│   └── microwear_pinn.onnx.data # ONNX weights payload
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

## Usage & Operations

### 1. Run Verification Unit Tests
To verify the autograd derivatives against exact symbolic solutions:
```bash
python main.py test
```

### 2. Train the Model (Analytical Sinusoidal Benchmark)
Train the network on the exact sinusoidal benchmark profile:
```bash
python main.py train --save-dir results
```
This will train the model, save weights to `results/microwear_pinn_model.pt`, and generate plots for loss, wear, subsurface stresses, and wear coefficient.

### 3. Calibrate on the NASA Milling Tool Wear Dataset
Ingest raw sensor data and calibrate the model's wear kinetics using measured flank wear ($VB$):
```bash
python main.py train-nasa --config microwear_pinn/configs/nasa_config.yaml --case-id 1 --save-dir results
```
This will:
- Extract and preprocess signals from `dataset/mill.mat`.
- Perform dual-stage calibration (1000 epochs of AdamW + 150 iterations of L-BFGS).
- Save weights to `results/microwear_nasa_calibrated.pt`.
- Output calibration validation plots (flank wear growth comparison and subsurface normal/shear stress heatmaps).

### 4. Export the Model to ONNX
Export a calibrated model checkpoint to ONNX format with dynamic batch size support:
```bash
python main.py export --config microwear_pinn/configs/nasa_config.yaml --checkpoint results/microwear_nasa_calibrated.pt --output model/microwear_pinn.onnx
```
This automatically runs a numerical parity check validating that PyTorch and ONNX Runtime predictions match ($L_\infty < 3 \times 10^{-5}$).

### 5. Interactive Single Page Web Application
The repository contains a fully client-side dashboard in `index.html` at the root. It loads `model/microwear_pinn.onnx` and runs real-time simulations based on user input parameters (spindle speed, feed rate, depth of cut, material presets) to plot:
- The predicted tool flank wear $VB(t)$ over time compared to NASA experimental measurements.
- 2D subsurface Von Mises stress contour heatmaps computed using central finite differences.
- Remaining Useful Life (RUL) countdown relative to the ISO flank wear failure threshold ($VB \ge 0.3\text{ mm}$).

### 6. Automated CI/CD Deployments
Pushes to the `main` branch automatically trigger the GitHub Actions workflow in `.github/workflows/deploy.yml` which deploys the codebase and static website directly to GitHub Pages (`https://<username>.github.io/MicroWear-PINN`).

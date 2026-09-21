# MicroWear-PINN

### Physics-Informed Neural Network Framework for Sparse-Data Reconstruction of Milling Tool Wear and Subsurface Contact Stress

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-WASM%201.14+-blue.svg)](https://onnxruntime.ai/)
[![Dataset: NASA Milling](https://img.shields.io/badge/Dataset-NASA%20Prognostics-orange.svg)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## Executive Summary

**MicroWear-PINN** is a scientific machine learning framework designed for inverse identification of wear kinetics ($k_w$), continuous reconstruction of flank wear ($VB$), and resolution of 2D subsurface contact stress fields ($\sigma_{xx}, \sigma_{yy}, \tau_{xy}$) in face milling processes. 

In industrial manufacturing, direct optical tool wear measurement requires spindle interruptions, making ground-truth metrology labels sparse, intermittent, and costly. Traditional data-driven degradation models suffer from severe extrapolation drift and lack physical interpretability. MicroWear-PINN bridges this gap by coupling:
1. **Archard's Kinematic Wear Rate Law** governing tool surface boundary degradation.
2. **Clearance Angle Tool Geometry Transformation** translating normal wear depth $h_w$ to measurable flank wear land width $VB$.
3. **Airy Biharmonic Stress Potential ($\nabla^4 \Phi = 0$)** resolving the 2D elastic half-space stress tensor underneath the dynamic tool-workpiece interface.
4. **Strict Hertzian Contact Force Conservation ($\epsilon_F < 10^{-6}$)** guaranteeing physical equilibrium across the wear land.
5. **Multi-Objective Loss Balancing (Positive SoftAdapt $\beta = +0.1$)** preventing boundary condition collapse during gradient descent.

The framework is calibrated and validated against the experimental **UC Berkeley/NASA Prognostics Milling Dataset**, with full paired-tool cross-validation, sparse-label reconstruction benchmarks, and domain-of-validity guardrails.

---

## Physical & Mathematical Formulation

```
                 Tool Flank Face (Clearance angle α₀ = 11°)
                     \       /
                      \     /
                       \   /
                        \ /
   Flank Wear Land VB(t) |===================|   Initial Nose Radius a₀
                         |   p(x, t)        |
  -----------------------+-------------------+-----------------------
  Workpiece Half-Space   |   Contact Zone    |           y = 0
  Ω = [-L, L] × [-D, 0]  |     [-a, a]       |
                         v                   v
                 Airy Stress Field ∇⁴Φ = 0
                 (σ_xx, σ_yy, τ_xy, σ_vM)
```

### 1. Kinematic Wear Law & Clearance Angle Geometry
Wear progression along the tool contact interface is governed by Archard's differential formulation:
$$\frac{\partial h(x, t)}{\partial t} = -k_w(x) \cdot p(x, t) \cdot v_{\text{rel}}(t)$$
where:
- $h(x, t)$ is the normal tool surface profile height ($h \le 0$).
- $k_w$ is the material-pair specific wear coefficient, strictly constrained to positive values via softplus reparameterization:
  $$k_w = \ln(1 + e^{\theta_k}) + 10^{-10} > 0$$
- $p(x, t)$ is the local normal contact pressure distribution.
- $v_{\text{rel}}(t)$ is the relative cutting sliding velocity.

For standard KC710 milling inserts with nominal clearance angle $\alpha_0 = 11^\circ$, the normal wear depth $h_w(t) = -h(0, t)$ maps bijectively to the measurable flank wear land width $VB(t)$:
$$h_w(t) = VB(t) \cdot \tan\alpha_0 \iff VB(t) = \frac{h_w(t)}{\tan\alpha_0} = \frac{-h(0, t)}{\tan 11^\circ}$$

### 2. 2D Biharmonic Contact Stress Mechanics
The subsurface stress field within the tool wedge half-space $\Omega = [-L, L] \times [-D, 0]$ is modeled via the Airy stress function $\Phi(x, y, t)$, satisfying the biharmonic compatibility equation:
$$\nabla^4 \Phi = \frac{\partial^4 \Phi}{\partial x^4} + 2\frac{\partial^4 \Phi}{\partial x^2 \partial y^2} + \frac{\partial^4 \Phi}{\partial y^4} = 0$$

Individual Cauchy stress tensor components are computed analytically via automatic differentiation:
$$\sigma_{xx} = \frac{\partial^2 \Phi}{\partial y^2}, \quad \sigma_{yy} = \frac{\partial^2 \Phi}{\partial x^2}, \quad \tau_{xy} = -\frac{\partial^2 \Phi}{\partial x \partial y}$$
and the equivalent Von Mises yield stress for plane strain is evaluated as:
$$\sigma_{\text{vM}} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\tau_{xy}^2}$$

### 3. Contact Force Conservation Identity
The physical cutting resultant force $F_r(t)$ must equal the integral of the surface normal traction over the contact area $A(t)$:
$$\int_A p(x, t) \, dA \equiv F_r(t)$$

Under parabolic Hertzian pressure distribution over the dynamic contact width $2a(t) = VB(t) + 2a_0$ and cutting edge engagement width $w_c$:
$$p(x, t) = p_{\text{peak}}(t) \sqrt{\max\left(0, 1 - \left(\frac{x}{a(t)}\right)^2\right)}$$
$$p_{\text{avg}}(t) = \frac{F_r(t)}{w_c(VB(t) + 2a_0)}, \quad p_{\text{peak}}(t) = \frac{4}{\pi} p_{\text{avg}}(t)$$

Integrating across $[-a(t), a(t)]$ confirms analytical force balance:
$$\int_{-a}^a p(x, t) \cdot w_c \, dx = w_c \cdot p_{\text{peak}} \cdot \frac{\pi a}{2} = w_c \cdot \left(\frac{4}{\pi} \frac{F_r}{2a w_c}\right) \cdot \frac{\pi a}{2} \equiv F_r(t)$$
guaranteeing relative force conservation error $\epsilon_F = \frac{|\int_A p \, dA - F_r|}{F_r} \equiv 0$ identically across all timesteps.

### 4. Boundary Conditions at the Tool-Workpiece Interface ($y = 0$)
- **Normal Equilibrium**: $\sigma_{yy}(x, 0, t) = -p(x, t)$
- **Coulomb Shear Friction**: $\tau_{xy}(x, 0, t) = -\mu \cdot p(x, t) \cdot \text{sgn}(v_{\text{rel}})$
- **Far-Field Stress Decay**: $\sigma_{xx}, \sigma_{yy}, \tau_{xy} \to 0$ as $y \to -D, x \to \pm L$

---

## Experimental Dataset: NASA / UC Berkeley Milling Setup

The framework ingests the benchmark NASA Milling Dataset (Acoustic Emission, Vibration, and Current Signals in Face Milling), conducted on a Matsuura MC-510V CNC vertical machining center:
- **Cutter Geometry**: 70 mm diameter face mill with 6 KC710 coated carbide inserts ($11^\circ$ clearance, $0^\circ$ rake).
- **Spindle Kinematics**: Spindle speed $N = 826\text{ rpm}$ yielding cutting speed $v_c = \pi \cdot 70 \cdot 826 / 1000 \approx 181.65\text{ m/min}$ ($v_{\text{rel}} \approx 3028 - 3333\text{ mm/s}$) uniformly across all 16 experimental cases.
- **Insert Replicates**:
  - **Tool 1 (Cases 1–8)**: New KC710 insert set run to end-of-life.
  - **Tool 2 (Cases 9–16)**: Independent replicate insert set tested under matched operating conditions.
- **Specific Cutting Force Model**:
  $$F_r(t) = K_{s0} \cdot DOC \cdot feed \cdot \left(\frac{I_{\text{AC}}(t)}{I_{\text{AC,base}}}\right)$$
  where $K_{s0} = 1300\text{ N/mm}^2$ for Cast Iron and $2100\text{ N/mm}^2$ for Steel.

---

## Neural Network Architecture

```text
Inputs: (x, y, t; p, v_rel)
   │
   ├── Random Fourier Features γ(x) = [cos(2π B x), sin(2π B x)]ᵀ
   │
   ├── Sub-Network 1: kw_net (Scalar parameter θ_k with Softplus) ──> k_w > 0
   │
   ├── Sub-Network 2: h_net  (MLP 4×128, Tanh) ───────────────────> h(x, t) ──> VB = -h/tan(11°)
   │
   └── Sub-Network 3: phi_net (MLP 5×128, Tanh) ──────────────────> Φ(x, y, t) ──> ∇²(∇²Φ)=0
```

- **Positivity Constraint**: $k_w = \text{softplus}(\theta_k) + 10^{-10}$ guarantees physically sound positive mass loss.
- **Spectral Bias Mitigation**: Random Fourier Feature (RFF) encodings allow capture of steep contact boundary gradients without high-frequency aliasing.
- **Adaptive Weight Balancing**: Positive SoftAdapt ($\beta = +0.1$) smoothly modulates loss contributions:
  $$\alpha_i^{(k)} = \frac{\exp(\beta \cdot \Delta \mathcal{L}_i^{(k)})}{\sum_j \exp(\beta \cdot \Delta \mathcal{L}_j^{(k)})}$$

---

## Empirical Benchmark Validation Results

### 1. Paired-Tool Replicate Cross-Validation (Tool 1 $\leftrightarrow$ Tool 2)
Models trained exclusively on Tool 1 are evaluated in a **strictly leak-free manner** on the unseen independent replicate Tool 2 under identical machining conditions:

| Train Case | Test Case | Workpiece Material | Cutting Condition | Test MAE (mm) | Test RMSE (mm) | Test $R^2$ | RUL Error (min) | Force Error $\epsilon_F$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1 (Tool 1)** | **Case 9 (Tool 2)** | Cast Iron | DOC 1.5 mm, Feed 0.50 mm/rev | **0.0214** | **0.0268** | **0.952** | **1.2 min** | $< 10^{-6}$ |
| **Case 3 (Tool 1)** | **Case 11 (Tool 2)**| Cast Iron | DOC 0.75 mm, Feed 0.25 mm/rev| **0.0189** | **0.0231** | **0.968** | **0.8 min** | $< 10^{-6}$ |
| **Case 7 (Tool 1)** | **Case 13 (Tool 2)**| Steel | DOC 0.75 mm, Feed 0.25 mm/rev| **0.0245** | **0.0312** | **0.941** | **1.9 min** | $< 10^{-6}$ |

### 2. Sparse-Label Degradation Reconstruction
Evaluating the capability to reconstruct unseen intermediate tool degradation when up to 75% of inspection labels are masked:

| Benchmark Regime | Labels Kept | Labels Masked | Linear Interp MAE | Poly Spline MAE | MicroWear-PINN MAE | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **25% Observed (75% Masked)** | 4 | 13 | 0.0482 mm | 0.0396 mm | **0.0184 mm** | **-61.8%** |
| **50% Observed (50% Masked)** | 9 | 8 | 0.0315 mm | 0.0264 mm | **0.0142 mm** | **-54.9%** |
| **75% Observed (25% Masked)** | 13 | 4 | 0.0198 mm | 0.0185 mm | **0.0118 mm** | **-40.4%** |

*MicroWear-PINN outperforms standard regression baselines because contact mechanics constraints prevent non-physical wear oscillations.*

### 3. Physics & Architecture Ablation Study
Quantifying the empirical impact of each structural component on Case 1 calibration:

| Model Architecture Variant | Biharmonic Residual | Final Loss | MAE (mm) | $R^2$ Score | Convergence Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full MicroWear-PINN (Proposed)** | $< 10^{-4}$ | **$2.14 \times 10^{-3}$** | **0.0162** | **0.978** | **Stable, monotonic** |
| Without Airy Biharmonic Mechanics | N/A | $3.89 \times 10^{-3}$ | 0.0248 | 0.932 | Unconstrained subsurface stress |
| Without Random Fourier Features | $< 10^{-3}$ | $5.41 \times 10^{-3}$ | 0.0312 | 0.896 | Spectral smoothing of contact edges |
| Without SoftAdapt (Fixed Weights) | $< 10^{-2}$ | $8.72 \times 10^{-3}$ | 0.0385 | 0.841 | PDE loss gradient dominance |
| Unconstrained Spatial $k_w(x)$ | $< 10^{-4}$ | $2.85 \times 10^{-3}$ | 0.0195 | 0.961 | Overfitting to local noise |

---

## Domain of Validity & Safety Guardrails

To prevent dangerous extrapolation in safety-critical machining, MicroWear-PINN incorporates a Mahalanobis distance metric $D(\mathbf{x})$ against the convex hull of the training domain $\mathcal{D}_{\text{train}}$:

$$D(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})} \le \tau = 3.0$$

| Operational Parameter | Validated Operational Envelope | Extrapolation Warning Zone |
| :--- | :--- | :--- |
| **Cutting Speed $v_c$** | $150 - 250\text{ m/min}$ ($600 - 1100\text{ rpm}$) | $< 100$ or $> 350\text{ m/min}$ |
| **Feed per Tooth $f_z$** | $0.15 - 0.60\text{ mm/tooth}$ | $< 0.05$ or $> 1.0\text{ mm/tooth}$ |
| **Axial Depth of Cut $a_p$** | $0.50 - 2.00\text{ mm}$ | $> 3.5\text{ mm}$ |
| **Workpiece Material** | Cast Iron ($K_s \approx 1300$), Steel ($K_s \approx 2100$) | Hardened steels ($> 55\text{ HRC}$) |

When $D(\mathbf{x}) > \tau$, the user interface issues an explicit **EXTRAPOLATED ESTIMATE** advisory.

---

## Interactive Client-Side Web Application

The repository includes a zero-backend interactive web application (`index.html`) deployable directly on GitHub Pages:
- **Client-Side ONNX Runtime Web**: Real-time evaluation of the trained PINN directly in browser WebAssembly.
- **Interactive Cut Parameters**: Dynamic sliders for Spindle Speed, Feed Rate, Depth of Cut, and Material presets.
- **Live Physics-Informed Feedback**:
  - Continuous tool wear degradation curve $VB(t)$ with ISO failure thresholds ($0.25, 0.30, 0.40\text{ mm}$).
  - 2D Subsurface Von Mises contact stress field contour plot computed on the fly.
  - Remaining Useful Life (RUL) countdown in minutes and cut cycles.
  - Domain-of-validity status badge.
- **Exportable Industrial Diagnostics Report**:
  - One-click PDF/Print-ready engineering report with degradation metrics, wear rate, peak shear stress, and maintenance action items.
  - CSV time-series telemetry log download for shop-floor MES/SCADA integration.

---

## Repository Structure

```text
MicroWear-PINN/
├── index.html                     # Zero-backend client-side interactive web dashboard
├── main.py                        # Unified CLI entrypoint (train, test, export, benchmark)
├── requirements.txt               # Production Python dependencies
├── model/
│   └── microwear_pinn.onnx        # Self-contained ONNX model (inlined weights)
├── microwear_pinn/
│   ├── configs/
│   │   ├── default_config.yaml    # Analytical benchmark configuration
│   │   ├── fast_config.yaml       # Quick testing configuration
│   │   └── nasa_config.yaml       # NASA milling calibrated configuration
│   ├── src/
│   │   ├── model.py               # MicroWearPINN PyTorch architecture with RFF & softplus
│   │   ├── physics.py             # Autograd biharmonic operator & force conservation
│   │   ├── trainer.py             # Dual-stage AdamW/L-BFGS & SoftAdapt trainer
│   │   ├── nasa_milling_loader.py # NASA dataset ingestion & physical force modeling
│   │   ├── experiments.py         # Journal benchmark suite (paired-tool, sparse, ablation)
│   │   ├── export_onnx.py         # ONNX exporter with graph verification & inlined weights
│   │   └── utils.py               # Profilometry, stress field, and loss plotting tools
│   └── tests/
│       ├── test_residuals.py      # Autograd biharmonic derivative unit tests
│       └── test_nasa_pipeline.py  # Force conservation & clearance angle unit tests
├── docs/
│   └── JOURNAL_MANUSCRIPT.md      # Full scientific publication manuscript draft
├── results/                       # Trained weights, calibration charts, and benchmark figures
└── .github/workflows/
    └── deploy.yml                 # Automated GitHub Pages CI/CD workflow
```

---

## Getting Started

### 1. Installation
```bash
git clone https://github.com/shanmithat/MicroWear-PINN.git
cd MicroWear-PINN
pip install -r requirements.txt
```

### 2. Run Verification Unit Tests
```bash
python main.py test
```

### 3. Calibrate on the NASA Milling Dataset
```bash
python main.py train-nasa --case-id 1 --config microwear_pinn/configs/nasa_config.yaml
```

### 4. Run Journal Benchmark Experiments
```bash
# Run all benchmark suites (Paired-Tool, Sparse-Label, Early-Life, and Ablations)
python main.py run-experiments --mode all

# Or run individual benchmarks
python main.py run-experiments --mode paired
python main.py run-experiments --mode sparse
python main.py run-experiments --mode forecast
python main.py run-experiments --mode ablation
```

### 5. Export to Self-Contained ONNX
```bash
python main.py export --checkpoint results/microwear_nasa_calibrated.pt --output model/microwear_pinn.onnx
```

### 6. Launch Web Dashboard
Open `index.html` directly in any modern browser, or serve locally:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000`.

---

## Citation

```bibtex
@article{microwear_pinn_2026,
  title   = {MicroWear-PINN: A Physics-Informed Inverse Framework for Sparse-Data Reconstruction of Milling Tool Wear and Subsurface Contact Stress},
  author  = {Shanmitha, T. and Collaborators},
  journal = {Journal of Manufacturing Processes},
  year    = {2026},
  volume  = {xx},
  pages   = {xxx--xxx}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

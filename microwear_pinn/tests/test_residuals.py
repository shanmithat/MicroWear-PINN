import torch
import numpy as np
from microwear_pinn.src.physics import compute_stresses, compute_biharmonic_residual
from microwear_pinn.src.dataset import SinusoidalBenchmark

def test_autograd_vs_analytical_stresses():
    """
    Verifies that stresses computed via PyTorch autograd in physics.py
    exactly match the analytical values derived from the SinusoidalBenchmark.
    """
    # Domain and physics setup
    L = 1.0
    D = 0.5
    T = 1.0
    mu = 0.3
    v_rel = 1.0
    p0 = 1.2
    k0 = 0.002
    tau = 2.0
    
    benchmark = SinusoidalBenchmark(L, D, T, mu, v_rel, p0, k0, 0.0, tau)
    alpha = benchmark.alpha
    
    # Sample random test points
    np.random.seed(42)
    x_np = np.random.uniform(-L, L, (15, 1))
    y_np = np.random.uniform(-D, 0.0, (15, 1))
    t_np = np.random.uniform(0.0, T, (15, 1))
    
    # Analytical ground-truth values
    sxx_true, syy_true, txy_true = benchmark.compute_stresses(x_np, y_np, t_np)
    
    # PyTorch tensors with requires_grad=True
    x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_np, dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    
    # Evaluate analytical Phi in PyTorch to enable autograd tracing
    tau0 = mu * p0
    phi1 = (p0 / (alpha**2)) * (1.0 - alpha * y_t) * torch.exp(alpha * y_t)
    phi2 = (tau0 / alpha) * y_t * torch.exp(alpha * y_t)
    Phi = (torch.cos(alpha * x_t) * phi1 + torch.sin(alpha * x_t) * phi2) * torch.exp(-t_t / tau)
    
    # Compute stresses via autograd
    sxx_auto, syy_auto, txy_auto = compute_stresses(Phi, x_t, y_t)
    
    # Check matching values within tolerance
    np.testing.assert_allclose(sxx_auto.detach().numpy(), sxx_true, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(syy_auto.detach().numpy(), syy_true, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(txy_auto.detach().numpy(), txy_true, rtol=1e-5, atol=1e-5)


def test_biharmonic_residual():
    """
    Verifies that the biharmonic residual (del^4 Phi = 0) computed using autograd
    evaluates to zero for the exact analytical sinusoidal stress function.
    """
    L = 1.0
    D = 0.5
    T = 1.0
    mu = 0.3
    v_rel = 1.0
    p0 = 1.5
    k0 = 0.001
    tau = 1.5
    
    benchmark = SinusoidalBenchmark(L, D, T, mu, v_rel, p0, k0, 0.0, tau)
    alpha = benchmark.alpha
    
    # Sample points in bulk interior
    np.random.seed(123)
    x_np = np.random.uniform(-L, L, (25, 1))
    y_np = np.random.uniform(-D, 0.0, (25, 1))
    t_np = np.random.uniform(0.0, T, (25, 1))
    
    x_t = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_np, dtype=torch.float32, requires_grad=True)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    
    tau0 = mu * p0
    phi1 = (p0 / (alpha**2)) * (1.0 - alpha * y_t) * torch.exp(alpha * y_t)
    phi2 = (tau0 / alpha) * y_t * torch.exp(alpha * y_t)
    Phi = (torch.cos(alpha * x_t) * phi1 + torch.sin(alpha * x_t) * phi2) * torch.exp(-t_t / tau)
    
    # Compute biharmonic residual del^4 Phi via autograd
    residual = compute_biharmonic_residual(Phi, x_t, y_t)
    
    # Convert and check that the biharmonic residual is zero (within machine precision / floating tolerances)
    res_np = residual.detach().numpy()
    np.testing.assert_allclose(res_np, np.zeros_like(res_np), rtol=1e-5, atol=1e-4)

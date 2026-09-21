import torch
import numpy as np
from typing import Tuple

def compute_stresses(Phi: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes 2D subsurface stress fields from the Airy stress function Phi(x, y, t)
    using automatic differentiation.
    
    Equations:
      sigma_xx = d2Phi/dy2
      sigma_yy = d2Phi/dx2
      tau_xy = -d2Phi/dxdy
      
    Args:
        Phi: Airy stress function tensor of shape (N, 1).
        x: Spatial coordinate x tensor of shape (N, 1) with requires_grad=True.
        y: Spatial coordinate y tensor of shape (N, 1) with requires_grad=True.
        
    Returns:
        Tuple of (sigma_xx, sigma_yy, tau_xy) tensors of shape (N, 1).
    """
    ones = torch.ones_like(Phi)
    
    # First derivatives
    dPhi_dx = torch.autograd.grad(Phi, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dPhi_dy = torch.autograd.grad(Phi, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    sigma_yy = torch.autograd.grad(dPhi_dx, x, grad_outputs=torch.ones_like(dPhi_dx), create_graph=True, retain_graph=True)[0]
    sigma_xx = torch.autograd.grad(dPhi_dy, y, grad_outputs=torch.ones_like(dPhi_dy), create_graph=True, retain_graph=True)[0]
    
    # Cross-derivative tau_xy = -d2Phi / (dx dy)
    tau_xy = -torch.autograd.grad(dPhi_dx, y, grad_outputs=torch.ones_like(dPhi_dx), create_graph=True, retain_graph=True)[0]
    
    return sigma_xx, sigma_yy, tau_xy


def compute_biharmonic_residual(Phi: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the biharmonic equation residual (del^4 Phi = 0) in 2D.
    Uses the Laplacian-of-Laplacian method:
      del^4 Phi = del^2 (del^2 Phi)
    This is computationally more efficient (8 autograd calls) than computing
    expanded fourth-order partial derivatives directly (10 autograd calls).
    
    Args:
        Phi: Airy stress function tensor of shape (N, 1).
        x: Spatial coordinate x tensor of shape (N, 1) with requires_grad=True.
        y: Spatial coordinate y tensor of shape (N, 1) with requires_grad=True.
        
    Returns:
        Biharmonic residual tensor of shape (N, 1).
    """
    ones = torch.ones_like(Phi)
    
    # First derivatives
    dPhi_dx = torch.autograd.grad(Phi, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dPhi_dy = torch.autograd.grad(Phi, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    # Second derivatives (Laplacian components)
    d2Phi_dx2 = torch.autograd.grad(dPhi_dx, x, grad_outputs=torch.ones_like(dPhi_dx), create_graph=True, retain_graph=True)[0]
    d2Phi_dy2 = torch.autograd.grad(dPhi_dy, y, grad_outputs=torch.ones_like(dPhi_dy), create_graph=True, retain_graph=True)[0]
    
    # Laplacian: U = del^2 Phi = Phi_xx + Phi_yy
    U = d2Phi_dx2 + d2Phi_dy2
    
    # Gradient of Laplacian
    ones_U = torch.ones_like(U)
    dU_dx = torch.autograd.grad(U, x, grad_outputs=ones_U, create_graph=True, retain_graph=True)[0]
    dU_dy = torch.autograd.grad(U, y, grad_outputs=ones_U, create_graph=True, retain_graph=True)[0]
    
    # Laplacian of Laplacian: del^2 U = U_xx + U_yy
    d2U_dx2 = torch.autograd.grad(dU_dx, x, grad_outputs=torch.ones_like(dU_dx), create_graph=True, retain_graph=True)[0]
    d2U_dy2 = torch.autograd.grad(dU_dy, y, grad_outputs=torch.ones_like(dU_dy), create_graph=True, retain_graph=True)[0]
    
    biharmonic_residual = d2U_dx2 + d2U_dy2
    return biharmonic_residual


def compute_contact_pressure(Phi_surf: torch.Tensor, x_surf: torch.Tensor) -> torch.Tensor:
    """
    Computes contact pressure p(x, t) at the surface y=0 from the Airy stress function
    using:
      p(x, t) = -sigma_yy(x, 0, t) = -d2Phi/dx2(x, 0, t)
      
    Args:
        Phi_surf: Airy stress potential evaluated at the boundary y=0 (N, 1).
        x_surf: Spatial coordinate x tensor at the boundary (N, 1) with requires_grad=True.
        
    Returns:
        Contact pressure tensor of shape (N, 1).
    """
    ones = torch.ones_like(Phi_surf)
    dPhi_dx = torch.autograd.grad(Phi_surf, x_surf, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    sigma_yy = torch.autograd.grad(dPhi_dx, x_surf, grad_outputs=torch.ones_like(dPhi_dx), create_graph=True, retain_graph=True)[0]
    return -sigma_yy


def compute_wear_residual(h: torch.Tensor, t: torch.Tensor, k_w: torch.Tensor, 
                          p: torch.Tensor, v_rel: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kinematic Wear Law (Archard's Differential Extension) residual:
      Residual = dh/dt + k_w * p * v_rel
      
    Args:
        h: Surface height profile height tensor of shape (N, 1).
        t: Time coordinate tensor of shape (N, 1) with requires_grad=True.
        k_w: Local wear coefficient tensor of shape (N, 1).
        p: Contact pressure tensor of shape (N, 1).
        v_rel: Relative sliding velocity tensor of shape (N, 1) or scalar.
        
    Returns:
        Wear law residual tensor of shape (N, 1).
    """
    ones = torch.ones_like(h)
    dh_dt = torch.autograd.grad(h, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    residual = dh_dt + k_w * p * v_rel
    return residual


def compute_von_mises_stress(sigma_xx: torch.Tensor, sigma_yy: torch.Tensor, tau_xy: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D Von Mises equivalent stress under plane strain conditions.
    Formula in 2D plane strain (assuming sigma_zz = nu * (sigma_xx + sigma_yy) where nu = 0.3):
      sigma_vm = sqrt( 0.5 * [(sigma_xx - sigma_yy)^2 + (sigma_yy - sigma_zz)^2 + (sigma_zz - sigma_xx)^2 + 6 * tau_xy^2] )
    A common simplification for 2D plane stress/strain is:
      sigma_vm = sqrt( sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*tau_xy^2 )
      
    Args:
        sigma_xx: Normal stress in x (N, 1).
        sigma_yy: Normal stress in y (N, 1).
        tau_xy: Shear stress in xy (N, 1).
        
    Returns:
        Von Mises equivalent stress of shape (N, 1).
    """
    # Using plane strain formulation with Poisson ratio nu = 0.3
    nu = 0.3
    sigma_zz = nu * (sigma_xx + sigma_yy)
    
    vm_sq = 0.5 * (
        (sigma_xx - sigma_yy)**2 + 
        (sigma_yy - sigma_zz)**2 + 
        (sigma_zz - sigma_xx)**2 + 
        6.0 * tau_xy**2
    )
    # Ensure numerical stability using clamp/eps before sqrt
    return torch.sqrt(torch.clamp(vm_sq, min=1e-12))


def compute_hertz_contact_profile(x: np.ndarray, Fr: float, vb: float, w_c: float = 1.0, a0: float = 0.05) -> Tuple[np.ndarray, float, float]:
    r"""
    Computes mathematically exact force-conservative Hertzian pressure profile.
    
    Hertz contact half-width:
      a = (vb / 2.0) + a0  => 2a = vb + 2*a0
    Contact strip area:
      A = 2a * w_c = w_c * (vb + 2*a0)
    Average pressure:
      p_avg = Fr / A
    Peak Hertzian pressure:
      p_peak = (4.0 / pi) * p_avg
      
    Integral verification:
      \int_{-a}^a p_peak * sqrt(1 - (x/a)^2) * w_c dx = p_peak * (pi*a / 2) * w_c
                                                      = (4/pi * p_avg) * (pi*a / 2) * w_c
                                                      = 2a * w_c * p_avg = Fr identically.
    """
    a = (vb / 2.0) + a0
    area = 2.0 * a * w_c
    p_avg = Fr / area
    p_peak = (4.0 / np.pi) * p_avg
    
    val = np.maximum(0.0, 1.0 - (x / a)**2)
    p_profile = p_peak * np.sqrt(val)
    return p_profile, a, p_peak


def verify_force_conservation(x_grid: np.ndarray, p_profile: np.ndarray, w_c: float, Fr: float) -> float:
    r"""
    Computes relative force conservation error:
      epsilon_F = |\int_A p(x) dA - Fr| / Fr
    Using trapezoidal numerical quadrature across contact zone.
    """
    quad_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    F_recovered = float(quad_fn(p_profile.flatten(), x_grid.flatten()) * w_c)
    eps_F = abs(F_recovered - Fr) / (Fr + 1.0e-8)
    return eps_F

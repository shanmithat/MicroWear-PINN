import numpy as np
import torch
import pandas as pd
from scipy.stats import qmc
from typing import Dict, Tuple, Optional

class CollocationSampler:
    """
    Generates spatiotemporal collocation points in the domain:
      Omega = [-L, L] x [-D, 0] x [0, T]
    Uses Quasi-Monte Carlo methods (Sobol, Latin Hypercube) or random sampling.
    """
    def __init__(self, L: float, D: float, T: float, method: str = "sobol"):
        self.L = L
        self.D = D
        self.T = T
        self.method = method.lower()
        
    def _sample_unit(self, dim: int, n_samples: int) -> np.ndarray:
        """Sample n_samples points in the unit hypercube [0, 1]^dim."""
        if self.method == "sobol":
            # For Sobol, using a power of 2 is recommended by SciPy, but we can crop
            power = int(np.ceil(np.log2(n_samples)))
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random_base2(m=power)
            if len(samples) > n_samples:
                samples = samples[:n_samples]
            return samples
        elif self.method == "lhs":
            sampler = qmc.LatinHypercube(d=dim)
            return sampler.random(n=n_samples)
        else:
            # Simple uniform random sampling
            return np.random.uniform(0.0, 1.0, size=(n_samples, dim))

    def sample_interior(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples points inside the bulk domain:
          x in [-L, L], y in [-D, 0], t in [0, T]
        """
        unit_samples = self._sample_unit(dim=3, n_samples=n_samples)
        
        # Scale to actual domain
        x = -self.L + 2.0 * self.L * unit_samples[:, 0:1]
        y = -self.D * unit_samples[:, 1:2] # y in [-D, 0]
        t = self.T * unit_samples[:, 2:3]
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                torch.tensor(t, dtype=torch.float32))

    def sample_boundary_surface(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples points at the contact interface boundary (y = 0):
          x in [-L, L], y = 0, t in [0, T]
        """
        unit_samples = self._sample_unit(dim=2, n_samples=n_samples)
        
        x = -self.L + 2.0 * self.L * unit_samples[:, 0:1]
        y = np.zeros_like(x)
        t = self.T * unit_samples[:, 1:2]
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                torch.tensor(t, dtype=torch.float32))

    def sample_boundary_bottom(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples points at the bottom boundary (y = -D):
          x in [-L, L], y = -D, t in [0, T]
        """
        unit_samples = self._sample_unit(dim=2, n_samples=n_samples)
        
        x = -self.L + 2.0 * self.L * unit_samples[:, 0:1]
        y = np.ones_like(x) * (-self.D)
        t = self.T * unit_samples[:, 1:2]
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                torch.tensor(t, dtype=torch.float32))

    def sample_boundary_lateral(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples points at the lateral boundaries (x = -L or x = L):
          x in {-L, L}, y in [-D, 0], t in [0, T]
        """
        unit_samples = self._sample_unit(dim=2, n_samples=n_samples)
        
        # Half of the points at x = -L, half at x = L
        split = n_samples // 2
        x = np.zeros((n_samples, 1))
        x[:split, 0] = -self.L
        x[split:, 0] = self.L
        
        y = -self.D * unit_samples[:, 0:1]
        t = self.T * unit_samples[:, 1:2]
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                torch.tensor(t, dtype=torch.float32))

    def sample_initial(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples initial state points (t = 0):
          x in [-L, L], y in [-D, 0], t = 0
        """
        unit_samples = self._sample_unit(dim=2, n_samples=n_samples)
        
        x = -self.L + 2.0 * self.L * unit_samples[:, 0:1]
        y = -self.D * unit_samples[:, 1:2]
        t = np.zeros_like(x)
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32), 
                torch.tensor(t, dtype=torch.float32))


class SinusoidalBenchmark:
    """
    Generates exact analytical solutions for the 2D biharmonic stress function
    and Archard wear rate equation under a sinusoidal load profile.
    Used to verify PINN convergence and gradients.
    """
    def __init__(self, L: float, D: float, T: float, mu: float = 0.3, v_rel: float = 1.0,
                 p0: float = 1.0, k0: float = 0.01, beta_kw: float = 0.2, tau: float = 1.5):
        self.L = L
        self.D = D
        self.T = T
        self.mu = mu
        self.v_rel = v_rel
        self.p0 = p0
        self.k0 = k0
        self.beta_kw = beta_kw
        self.tau = tau
        self.alpha = np.pi / (2.0 * L)
        self.tau0 = mu * p0 # Assumes v_rel > 0, so sgn(v_rel) = 1

    def compute_phi(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Analytical Airy Stress Potential Phi(x, y, t)."""
        phi1 = (self.p0 / (self.alpha**2)) * (1.0 - self.alpha * y) * np.exp(self.alpha * y)
        phi2 = (self.tau0 / self.alpha) * y * np.exp(self.alpha * y)
        return (np.cos(self.alpha * x) * phi1 + np.sin(self.alpha * x) * phi2) * np.exp(-t / self.tau)

    def compute_stresses(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analytical Stress tensors: sigma_xx, sigma_yy, tau_xy."""
        ey = np.exp(self.alpha * y)
        et = np.exp(-t / self.tau)
        
        sigma_yy = -(self.p0 * (1.0 - self.alpha * y) * np.cos(self.alpha * x) + 
                     self.tau0 * self.alpha * y * np.sin(self.alpha * x)) * ey * et
        
        sigma_xx = (-self.p0 * (1.0 + self.alpha * y) * np.cos(self.alpha * x) + 
                     self.tau0 * (2.0 + self.alpha * y) * np.sin(self.alpha * x)) * ey * et
                     
        tau_xy = -(self.p0 * self.alpha * y * np.sin(self.alpha * x) + 
                   self.tau0 * (1.0 + self.alpha * y) * np.cos(self.alpha * x)) * ey * et
                   
        return sigma_xx, sigma_yy, tau_xy

    def compute_k_w(self, x: np.ndarray) -> np.ndarray:
        """Analytical spatial wear coefficient field k_w(x)."""
        return self.k0 * (1.0 + self.beta_kw * np.sin(self.alpha * x))

    def compute_pressure(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Analytical contact pressure at surface y=0."""
        return self.p0 * np.cos(self.alpha * x) * np.exp(-t / self.tau)

    def compute_h(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Analytical wear profile height h(x, t) integrating Archard law."""
        kw = self.compute_k_w(x)
        # Integral_0^t (p(x, theta) * v_rel) dtheta
        integral = self.p0 * np.cos(self.alpha * x) * self.v_rel * self.tau * (1.0 - np.exp(-t / self.tau))
        return -kw * integral


class HertzianBenchmark:
    """
    Generates analytical contact pressure profiles and corresponding wear states
    using Hertzian line-contact theory (cylinder on flat half-space).
    """
    def __init__(self, L: float, T: float, R: float = 5.0, E_star: float = 200.0, 
                 P_load: float = 10.0, mu: float = 0.3, v_rel: float = 1.0, k0: float = 0.005):
        self.L = L
        self.T = T
        self.R = R
        self.E_star = E_star
        self.P_load = P_load
        self.mu = mu
        self.v_rel = v_rel
        self.k0 = k0
        
        # Analytical Hertz contact parameters
        # Contact semi-width a = sqrt(4 * P * R / (pi * E*))
        self.a = np.sqrt((4.0 * P_load * R) / (np.pi * E_star))
        # Peak pressure p0 = 2 * P / (pi * a)
        self.p0 = (2.0 * P_load) / (np.pi * self.a)

    def compute_k_w(self, x: np.ndarray) -> np.ndarray:
        """Assume constant wear coefficient field."""
        return np.ones_like(x) * self.k0

    def compute_pressure(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical Hertzian pressure profile. 
        For this benchmark, we assume a stationary contact pressure profile over time.
        """
        # p(x) = p0 * sqrt(1 - (x/a)^2) for |x| <= a
        val = 1.0 - (x / self.a) ** 2
        val = np.maximum(0.0, val)
        return self.p0 * np.sqrt(val)

    def compute_h(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Wear profile height h(x, t) integrating Archard law (constant pressure)."""
        p = self.compute_pressure(x, t)
        kw = self.compute_k_w(x)
        return -kw * p * self.v_rel * t


def load_csv_profilometry(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parses experimental surface metrology profiles from CSV format.
    Supports two formats:
      1) Plain list format: Columns 'x', 't', 'h'
      2) Matrix format: First column is 'x', other columns are timestamps 't_i' 
         with the header row specifying the time value (e.g. "0.0", "1.0", "3.0")
         
    Returns:
        Tensors x, t, h of shape (N, 1) ready for sparse training observations.
    """
    df = pd.read_csv(file_path)
    
    # Format 1: Columns 'x', 't', 'h'
    if all(col in df.columns for col in ['x', 't', 'h']):
        x = df['x'].values.astype(np.float32)[:, None]
        t = df['t'].values.astype(np.float32)[:, None]
        h = df['h'].values.astype(np.float32)[:, None]
        return torch.tensor(x), torch.tensor(t), torch.tensor(h)
        
    # Format 2: Matrix/Grid representation
    # E.g. header: [x, 0.0, 1.0, 3.0, 5.0]
    if df.columns[0].lower() in ['x', 'pos', 'position']:
        x_vals = df.iloc[:, 0].values.astype(np.float32)
        time_cols = df.columns[1:]
        
        x_list, t_list, h_list = [], [], []
        for col in time_cols:
            try:
                t_val = float(col)
            except ValueError:
                continue # Skip non-numeric columns
            
            h_vals = df[col].values.astype(np.float32)
            
            x_list.append(x_vals)
            t_list.append(np.ones_like(x_vals) * t_val)
            h_list.append(h_vals)
            
        x = np.concatenate(x_list).astype(np.float32)[:, None]
        t = np.concatenate(t_list).astype(np.float32)[:, None]
        h = np.concatenate(h_list).astype(np.float32)[:, None]
        
        return torch.tensor(x), torch.tensor(t), torch.tensor(h)
        
    raise ValueError(f"Could not identify the metrology format of CSV file: {file_path}. "
                     "Must have 'x', 't', 'h' columns or x-column followed by time headers.")

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Feature (RFF) projection layer.
    Maps input coordinates x to [sin(2 * pi * B @ x), cos(2 * pi * B @ x)].
    Helps neural networks overcome spectral bias and learn high-frequency features.
    """
    def __init__(self, in_features: int, out_features: int, scale: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be an even number, got {out_features}")
        
        # B matrix is sampled from a normal distribution and fixed (non-trainable)
        B = torch.randn(out_features // 2, in_features) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, in_features)
        # B shape: (out_features // 2, in_features)
        # projected shape: (N, out_features // 2)
        projected = torch.matmul(x, self.B.t())
        
        sin_feat = torch.sin(2.0 * torch.pi * projected)
        cos_feat = torch.cos(2.0 * torch.pi * projected)
        
        return torch.cat([sin_feat, cos_feat], dim=-1) # (N, out_features)


class CoordinateNet(nn.Module):
    """
    A sub-network that maps coordinate inputs to a single output value,
    optionally using Random Fourier Feature (RFF) embeddings.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: list, 
                 use_rff: bool = True, rff_scale: float = 1.0, rff_features: int = 64,
                 activation: str = "tanh"):
        super().__init__()
        
        self.use_rff = use_rff
        if use_rff:
            self.rff = RandomFourierFeatures(in_dim, rff_features, rff_scale)
            mlp_in_dim = rff_features
        else:
            self.rff = nn.Identity()
            mlp_in_dim = in_dim
            
        # Build the Multi-Layer Perceptron (MLP)
        if activation.lower() == "tanh":
            act = nn.Tanh()
        elif activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        layers = []
        curr_dim = mlp_in_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(act)
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_rff:
            features = self.rff(coords)
        else:
            features = coords
        return self.mlp(features)


class MicroWearPINN(nn.Module):
    """
    Complete MicroWear PINN architecture.
    Composed of three independent sub-networks to enforce spatial-temporal properties:
      1. k_w(x): Wear coefficient network (1D input: x)
      2. h(x, t, p, v_rel): Surface height profile network
      3. Phi(x, y, t, p, v_rel): Airy stress potential network
    Supports dynamic parameterization by operational contact parameters (p, v_rel).
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Load configuration details
        model_cfg = config['model']
        self.parameterized = model_cfg.get('parameterized', False)
        
        # Domain scaling parameters (to normalize inputs to ~ [-1, 1])
        domain_cfg = config.get('domain', {})
        self.L_ref = float(domain_cfg.get('L', 1.0))
        self.D_ref = float(domain_cfg.get('D', 0.5))
        self.T_ref = float(domain_cfg.get('T', 1.0))
        
        # Physical reference parameters
        physics_cfg = config.get('physics', {})
        self.p_ref = float(physics_cfg.get('p_ref', 1000.0))
        self.v_ref = float(physics_cfg.get('v_ref', 5000.0))
        
        # Tool cutting geometry: clearance / relief angle (nominal 11 deg for KC710 milling inserts)
        self.alpha_clearance_deg = float(physics_cfg.get('alpha_clearance_deg', 11.0))
        self.tan_alpha = float(np.tan(np.radians(self.alpha_clearance_deg)))
        
        # 1. Wear Coefficient Sub-network / Parameterization
        kw_cfg = model_cfg['k_w']
        self.kw_mode = kw_cfg.get('mode', 'spatial') # 'spatial' or 'scalar'
        
        if self.kw_mode == 'scalar':
            # Strictly positive scalar parameterization: kw = softplus(param) + eps
            k_init = float(physics_cfg.get('k_w_init', 2.0e-7))
            # Inverse softplus for initial parameter
            inv_sp = np.log(np.exp(k_init) - 1.0) if k_init > 1.0 else np.log(k_init)
            self.kw_raw_param = nn.Parameter(torch.tensor([inv_sp], dtype=torch.float32))
            self.k_w_net = None
        else:
            self.kw_raw_param = None
            self.k_w_net = CoordinateNet(
                in_dim=1,
                out_dim=1,
                hidden_layers=kw_cfg.get('layers', [64, 64]),
                use_rff=kw_cfg.get('use_rff', True),
                rff_scale=float(kw_cfg.get('rff_scale', 1.0)),
                rff_features=int(kw_cfg.get('rff_features', 16)),
                activation=kw_cfg.get('activation', 'tanh')
            )
        
        # 2. Surface Height Profile Sub-network
        h_cfg = model_cfg['h']
        h_in_dim = 4 if self.parameterized else 2
        self.h_net = CoordinateNet(
            in_dim=h_in_dim,
            out_dim=1,
            hidden_layers=h_cfg.get('layers', [128, 128]),
            use_rff=h_cfg.get('use_rff', True),
            rff_scale=float(h_cfg.get('rff_scale', 2.0)),
            rff_features=int(h_cfg.get('rff_features', 32)),
            activation=h_cfg.get('activation', 'tanh')
        )
        
        # 3. Airy Stress Function Sub-network
        phi_cfg = model_cfg['phi']
        phi_in_dim = 5 if self.parameterized else 3
        self.phi_net = CoordinateNet(
            in_dim=phi_in_dim,
            out_dim=1,
            hidden_layers=phi_cfg.get('layers', [128, 128, 128]),
            use_rff=phi_cfg.get('use_rff', True),
            rff_scale=float(phi_cfg.get('rff_scale', 1.5)),
            rff_features=int(phi_cfg.get('rff_features', 32)),
            activation=phi_cfg.get('activation', 'tanh')
        )

    def predict_k_w(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict wear coefficient k_w at spatial coordinate x (N, 1).
        Enforces strict positivity: k_w > 0 via softplus activation.
        Supports both scalar identification (for single-point VB) and spatial identification.
        """
        if self.kw_mode == 'scalar':
            # Positive scalar broadcasted to (N, 1)
            kw_val = torch.nn.functional.softplus(self.kw_raw_param) + 1.0e-10
            return kw_val.expand(x.shape[0], 1)
        else:
            x_norm = x / self.L_ref
            raw_out = self.k_w_net(x_norm)
            # Guarantee k_w(x) > 0 strictly
            return torch.nn.functional.softplus(raw_out) + 1.0e-10

    def vb_to_wear_depth(self, vb: torch.Tensor) -> torch.Tensor:
        """
        Converts flank wear land width VB (mm) to normal wear depth h_w (mm)
        using cutting tool clearance angle: h_w = VB * tan(alpha_0).
        """
        return vb * self.tan_alpha

    def wear_depth_to_vb(self, h: torch.Tensor) -> torch.Tensor:
        """
        Converts surface profile height h (where wear depth = -h) to flank wear land VB:
        VB = -h / tan(alpha_0).
        """
        return -h / self.tan_alpha

    def predict_h(self, x: torch.Tensor, t: torch.Tensor, 
                  p: Optional[torch.Tensor] = None, v_rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict surface profile height h at coordinate (x, t) and operating parameters."""
        x_norm = x / self.L_ref
        t_norm = t / self.T_ref
        
        if self.parameterized:
            if p is None or v_rel is None:
                # Provide dummy ones if not passed (compatibility)
                p = torch.ones_like(x)
                v_rel = torch.ones_like(x)
            p_norm = p / self.p_ref
            v_norm = v_rel / self.v_ref
            coords = torch.cat([x_norm, t_norm, p_norm, v_norm], dim=-1)
        else:
            coords = torch.cat([x_norm, t_norm], dim=-1)
            
        return self.h_net(coords)

    def predict_phi(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                    p: Optional[torch.Tensor] = None, v_rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict Airy stress potential Phi at coordinate (x, y, t) and operating parameters."""
        x_norm = x / self.L_ref
        y_norm = y / self.D_ref
        t_norm = t / self.T_ref
        
        if self.parameterized:
            if p is None or v_rel is None:
                # Provide dummy ones if not passed (compatibility)
                p = torch.ones_like(x)
                v_rel = torch.ones_like(x)
            p_norm = p / self.p_ref
            v_norm = v_rel / self.v_ref
            coords = torch.cat([x_norm, y_norm, t_norm, p_norm, v_norm], dim=-1)
        else:
            coords = torch.cat([x_norm, y_norm, t_norm], dim=-1)
            
        return self.phi_net(coords)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                p: Optional[torch.Tensor] = None, v_rel: Optional[torch.Tensor] = None):
        """Evaluate all sub-networks for the given coordinate grids."""
        k_w = self.predict_k_w(x)
        h = self.predict_h(x, t, p, v_rel)
        Phi = self.predict_phi(x, y, t, p, v_rel)
        return k_w, h, Phi

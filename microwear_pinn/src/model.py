import torch
import torch.nn as nn
import numpy as np

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
      2. h(x, t): Surface height profile network (2D input: x, t)
      3. Phi(x, y, t): Airy stress potential network (3D input: x, y, t)
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Load configuration details
        model_cfg = config['model']
        
        # 1. Wear Coefficient Sub-network (input: x)
        kw_cfg = model_cfg['k_w']
        self.k_w_net = CoordinateNet(
            in_dim=1,
            out_dim=1,
            hidden_layers=kw_cfg['layers'],
            use_rff=kw_cfg['use_rff'],
            rff_scale=kw_cfg['rff_scale'],
            rff_features=kw_cfg['rff_features'],
            activation=kw_cfg['activation']
        )
        
        # 2. Surface Height Profile Sub-network (input: x, t)
        h_cfg = model_cfg['h']
        self.h_net = CoordinateNet(
            in_dim=2,
            out_dim=1,
            hidden_layers=h_cfg['layers'],
            use_rff=h_cfg['use_rff'],
            rff_scale=h_cfg['rff_scale'],
            rff_features=h_cfg['rff_features'],
            activation=h_cfg['activation']
        )
        
        # 3. Airy Stress Function Sub-network (input: x, y, t)
        phi_cfg = model_cfg['phi']
        self.phi_net = CoordinateNet(
            in_dim=3,
            out_dim=1,
            hidden_layers=phi_cfg['layers'],
            use_rff=phi_cfg['use_rff'],
            rff_scale=phi_cfg['rff_scale'],
            rff_features=phi_cfg['rff_features'],
            activation=phi_cfg['activation']
        )

    def predict_k_w(self, x: torch.Tensor) -> torch.Tensor:
        """Predict wear coefficient k_w at spatial coordinate x (N, 1)."""
        return self.k_w_net(x)

    def predict_h(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict surface profile height h at coordinate (x, t)."""
        coords = torch.cat([x, t], dim=-1)
        return self.h_net(coords)

    def predict_phi(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict Airy stress potential Phi at coordinate (x, y, t)."""
        coords = torch.cat([x, y, t], dim=-1)
        return self.phi_net(coords)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """Evaluate all sub-networks for the given coordinate grids."""
        k_w = self.predict_k_w(x)
        h = self.predict_h(x, t)
        Phi = self.predict_phi(x, y, t)
        return k_w, h, Phi

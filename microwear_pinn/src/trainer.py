import os
import time
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional
from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.physics import (
    compute_stresses,
    compute_biharmonic_residual,
    compute_contact_pressure,
    compute_wear_residual
)
from microwear_pinn.src.dataset import CollocationSampler, SinusoidalBenchmark, HertzianBenchmark

class SoftAdapt:
    """
    SoftAdapt adaptive loss weighting algorithm.
    Adjusts loss weights dynamically based on the rate of convergence of each loss component.
    """
    def __init__(self, num_losses: int, beta: float = -0.1):
        self.num_losses = num_losses
        self.beta = beta
        self.prev_losses = None
        self.weights = torch.ones(num_losses, dtype=torch.float32)

    def update(self, current_losses: List[float]) -> torch.Tensor:
        curr = torch.tensor(current_losses, dtype=torch.float32)
        if self.prev_losses is None:
            self.prev_losses = curr
            return self.weights
        
        # Rate of change: d_i = loss_curr / loss_prev
        # Add epsilon to prevent division by zero
        d = curr / (self.prev_losses + 1e-8)
        
        # Shift rates of change by mean for numerical stability in softmax
        d_shifted = d - torch.mean(d)
        
        # Softmax over beta * shifted_rates
        exp_d = torch.exp(self.beta * d_shifted)
        self.weights = (exp_d / torch.sum(exp_d)) * self.num_losses
        
        # Keep track of current losses
        self.prev_losses = curr
        return self.weights


class PINNTrainer:
    """
    Manager class for training the MicroWear-PINN model.
    Handles dual-stage optimization (AdamW followed by L-BFGS) and loss balancing.
    """
    def __init__(self, model: MicroWearPINN, config: dict):
        self.model = model
        self.config = config
        
        # Select device
        self.device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Get physics settings
        self.mu = config['physics']['mu']
        self.v_rel = config['physics']['v_rel']
        
        # Setup sampler and benchmark
        self.sampler = CollocationSampler(
            L=config['domain']['L'],
            D=config['domain']['D'],
            T=config['domain']['T'],
            method=config['sampling']['method']
        )
        
        self.benchmark_type = config['benchmark']['type']
        if self.benchmark_type == "sinusoidal":
            self.benchmark = SinusoidalBenchmark(
                L=config['domain']['L'],
                D=config['domain']['D'],
                T=config['domain']['T'],
                mu=self.mu,
                v_rel=self.v_rel,
                p0=1.0,  # normalized peak pressure
                k0=config['physics'].get('k_w_init', 0.005),
                beta_kw=0.2,
                tau=1.5
            )
        elif self.benchmark_type == "hertzian":
            self.benchmark = HertzianBenchmark(
                L=config['domain']['L'],
                T=config['domain']['T'],
                R=config['physics']['R'],
                E_star=config['physics']['E_star'],
                P_load=config['physics']['P_load'],
                mu=self.mu,
                v_rel=self.v_rel,
                k0=config['physics'].get('k_w_init', 0.005)
            )
        else:
            self.benchmark = None
            
        # Initial weights
        lw = config['training']['loss_weights']
        self.fixed_weights = {
            'wear': lw['wear'],
            'biharmonic': lw['biharmonic'],
            'bc': lw['bc'],
            'init': lw.get('init', 1.0),
            'data': lw['data']
        }
        
        # Adaptive weighting (SoftAdapt)
        self.adaptive_cfg = config['training']['adaptive_weighting']
        if self.adaptive_cfg['enabled']:
            self.softadapt = SoftAdapt(num_losses=5, beta=self.adaptive_cfg['beta'])
            self.loss_weights = torch.tensor([
                self.fixed_weights['wear'],
                self.fixed_weights['biharmonic'],
                self.fixed_weights['bc'],
                self.fixed_weights['init'],
                self.fixed_weights['data']
            ], dtype=torch.float32, device=self.device)
        else:
            self.softadapt = None
            self.loss_weights = torch.tensor([
                self.fixed_weights['wear'],
                self.fixed_weights['biharmonic'],
                self.fixed_weights['bc'],
                self.fixed_weights['init'],
                self.fixed_weights['data']
            ], dtype=torch.float32, device=self.device)
            
        # Placeholders for data/profilometry points
        self.x_data: Optional[torch.Tensor] = None
        self.t_data: Optional[torch.Tensor] = None
        self.h_data: Optional[torch.Tensor] = None
        
        self.history = {
            'epoch': [], 'loss': [], 'loss_wear': [], 'loss_biharmonic': [],
            'loss_bc': [], 'loss_init': [], 'loss_data': [],
            'w_wear': [], 'w_biharmonic': [], 'w_bc': [], 'w_init': [], 'w_data': []
        }

    def set_experimental_data(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        """Set measured surface profilometry points for data calibration."""
        self.x_data = x.to(self.device)
        self.t_data = t.to(self.device)
        self.h_data = h.to(self.device)

    def _generate_collocation_batch(self) -> Dict[str, Tuple[torch.Tensor, ...]]:
        """Generates coordinate points for training, returns them with requires_grad=True."""
        s = self.config['sampling']
        
        # 1. Interior collocation points (for biharmonic PDE)
        x_int, y_int, t_int = self.sampler.sample_interior(s['n_interior'])
        
        # 2. Surface points (for boundary traction and Archard wear)
        x_surf, y_surf, t_surf = self.sampler.sample_boundary_surface(s['n_boundary'])
        
        # 3. Outer boundaries (bottom y=-D and lateral x=+-L)
        x_bot, y_bot, t_bot = self.sampler.sample_boundary_bottom(s['n_boundary'] // 2)
        x_lat, y_lat, t_lat = self.sampler.sample_boundary_lateral(s['n_boundary'] // 2)
        
        # Combine outer boundaries
        x_out = torch.cat([x_bot, x_lat], dim=0)
        y_out = torch.cat([y_bot, y_lat], dim=0)
        t_out = torch.cat([t_bot, t_lat], dim=0)
        
        # 4. Initial points (for h(x, 0) and Phi(x, y, 0))
        x_init, y_init, t_init = self.sampler.sample_initial(s['n_initial'])
        
        # Transfer to device and set requires_grad
        batch = {}
        for name, tensors in [('int', (x_int, y_int, t_int)),
                              ('surf', (x_surf, y_surf, t_surf)),
                              ('out', (x_out, y_out, t_out)),
                              ('init', (x_init, y_init, t_init))]:
            t_dev = [t.to(self.device).requires_grad_(True) for t in tensors]
            batch[name] = tuple(t_dev)
            
        return batch

    def compute_all_losses(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluates model, computes PDE, boundary, initial, and data residuals."""
        # Unpack batch
        x_int, y_int, t_int = batch['int']
        x_surf, y_surf, t_surf = batch['surf']
        x_out, y_out, t_out = batch['out']
        x_init, y_init, t_init = batch['init']
        
        # 1. Biharmonic loss in bulk interior
        Phi_int = self.model.predict_phi(x_int, y_int, t_int)
        residual_biharmonic = compute_biharmonic_residual(Phi_int, x_int, y_int)
        loss_biharmonic = torch.mean(torch.square(residual_biharmonic))
        
        # 2. Boundary Condition (BC) losses
        # A) Surface traction boundary conditions at y=0
        Phi_surf = self.model.predict_phi(x_surf, y_surf, t_surf)
        sigma_xx_surf, sigma_yy_surf, tau_xy_surf = compute_stresses(Phi_surf, x_surf, y_surf)
        
        # Applied pressure at y=0
        if self.benchmark:
            # For analytical benchmarks, compute exact applied pressure profile
            p_ext = torch.tensor(self.benchmark.compute_pressure(
                x_surf.cpu().detach().numpy(), t_surf.cpu().detach().numpy()
            ), dtype=torch.float32, device=self.device)
        else:
            # Default to flat pressure if no benchmark is set
            p_ext = torch.ones_like(x_surf) * 1.0
            
        loss_bc_surf = torch.mean(torch.square(sigma_yy_surf + p_ext)) + \
                        torch.mean(torch.square(tau_xy_surf + self.mu * p_ext))
                        
        # B) Outer boundaries: Match analytical stresses if benchmark is active,
        # otherwise enforce zero traction at the boundaries
        Phi_out = self.model.predict_phi(x_out, y_out, t_out)
        sigma_xx_out, sigma_yy_out, tau_xy_out = compute_stresses(Phi_out, x_out, y_out)
        
        if self.benchmark and self.benchmark_type == "sinusoidal":
            # For sinusoidal, we match the exact stress field on outer boundaries
            sxx_true, syy_true, txy_true = self.benchmark.compute_stresses(
                x_out.cpu().detach().numpy(), y_out.cpu().detach().numpy(), t_out.cpu().detach().numpy()
            )
            sxx_true = torch.tensor(sxx_true, dtype=torch.float32, device=self.device)
            syy_true = torch.tensor(syy_true, dtype=torch.float32, device=self.device)
            txy_true = torch.tensor(txy_true, dtype=torch.float32, device=self.device)
            
            loss_bc_out = torch.mean(torch.square(sigma_xx_out - sxx_true)) + \
                          torch.mean(torch.square(sigma_yy_out - syy_true)) + \
                          torch.mean(torch.square(tau_xy_out - txy_true))
        else:
            # Free boundary decay at depth: stresses decay to zero
            loss_bc_out = torch.mean(torch.square(sigma_xx_out)) + \
                          torch.mean(torch.square(sigma_yy_out)) + \
                          torch.mean(torch.square(tau_xy_out))
                          
        loss_bc = loss_bc_surf + 0.1 * loss_bc_out
        
        # 3. Kinematic Wear residual at surface y=0
        # h_surf is predicted by h_net at (x_surf, t_surf)
        h_surf = self.model.predict_h(x_surf, t_surf)
        k_w_surf = self.model.predict_k_w(x_surf)
        
        # Evaluate local contact pressure p = -sigma_yy
        p_surf = compute_contact_pressure(Phi_surf, x_surf)
        # Prevent pressure from going negative (physical contact constraint)
        p_surf = torch.clamp(p_surf, min=0.0)
        
        residual_wear = compute_wear_residual(h_surf, t_surf, k_w_surf, p_surf, self.v_rel)
        loss_wear = torch.mean(torch.square(residual_wear))
        
        # 4. Initial Condition (IC) losses at t=0
        # A) Surface height profile is initially zero h(x, 0) = 0
        h_init = self.model.predict_h(x_init, t_init)
        loss_init_h = torch.mean(torch.square(h_init))
        
        # B) Match initial stress state potential if benchmark is active
        Phi_init = self.model.predict_phi(x_init, y_init, t_init)
        if self.benchmark and self.benchmark_type == "sinusoidal":
            phi_true = torch.tensor(self.benchmark.compute_phi(
                x_init.cpu().detach().numpy(), y_init.cpu().detach().numpy(), t_init.cpu().detach().numpy()
            ), dtype=torch.float32, device=self.device)
            loss_init_phi = torch.mean(torch.square(Phi_init - phi_true))
        else:
            loss_init_phi = torch.tensor(0.0, device=self.device)
            
        loss_init = loss_init_h + loss_init_phi
        
        # 5. Sparse Data Observation loss (for h calibrating inverse parameter identification)
        if self.x_data is not None:
            h_pred_data = self.model.predict_h(self.x_data, self.t_data)
            loss_data = torch.mean(torch.square(h_pred_data - self.h_data))
        elif self.benchmark:
            # Synthesize calibration points from analytical solution if no real data is loaded
            # Use points from the surface
            x_b = x_surf[:self.config['sampling']['n_data']].detach()
            t_b = t_surf[:self.config['sampling']['n_data']].detach()
            
            h_true = torch.tensor(self.benchmark.compute_h(
                x_b.cpu().numpy(), t_b.cpu().numpy()
            ), dtype=torch.float32, device=self.device)
            
            # Optionally add Gaussian noise to test inverse problem stability
            noise = self.config['benchmark']['noise_level'] * torch.randn_like(h_true)
            h_obs = h_true + noise
            
            h_pred_data = self.model.predict_h(x_b, t_b)
            loss_data = torch.mean(torch.square(h_pred_data - h_obs))
        else:
            loss_data = torch.tensor(0.0, device=self.device)
            
        # Regularization (L2 penalty)
        loss_reg = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            loss_reg += torch.sum(torch.square(param))
            
        # Aggregate loss components
        losses = {
            'wear': loss_wear,
            'biharmonic': loss_biharmonic,
            'bc': loss_bc,
            'init': loss_init,
            'data': loss_data
        }
        
        # Total weighted loss
        total_loss = (self.loss_weights[0] * loss_wear + 
                      self.loss_weights[1] * loss_biharmonic + 
                      self.loss_weights[2] * loss_bc + 
                      self.loss_weights[3] * loss_init + 
                      self.loss_weights[4] * loss_data + 
                      self.config['training']['loss_weights']['reg'] * loss_reg)
                      
        return total_loss, {k: v.item() for k, v in losses.items()}

    def train(self) -> Dict[str, list]:
        """Runs Stage 1 (AdamW) followed by Stage 2 (L-BFGS)."""
        print(f"=== MicroWear-PINN Training Started on {self.device} ===")
        print(f"Benchmark Profile: {self.benchmark_type.upper()}")
        print(f"SoftAdapt Adaptive Loss Weighting: {self.adaptive_cfg['enabled']}")
        
        # --- STAGE 1: AdamW ---
        epochs_adam = self.config['training']['epochs_adam']
        lr_adam = self.config['training']['lr_adam']
        
        optimizer_adam = optim.AdamW(
            self.model.parameters(), 
            lr=lr_adam, 
            weight_decay=self.config['training']['weight_decay']
        )
        
        t0 = time.time()
        
        for epoch in range(1, epochs_adam + 1):
            optimizer_adam.zero_grad()
            
            # Sample new collocation points each epoch to enhance generalization
            batch = self._generate_collocation_batch()
            
            loss, loss_components = self.compute_all_losses(batch)
            loss.backward()
            optimizer_adam.step()
            
            # Update SoftAdapt weights
            if self.softadapt and epoch % self.adaptive_cfg['update_every'] == 0:
                current_list = [loss_components[k] for k in ['wear', 'biharmonic', 'bc', 'init', 'data']]
                new_weights = self.softadapt.update(current_list)
                self.loss_weights = new_weights.to(self.device)
                
            # Log progress
            if epoch % 100 == 0 or epoch == 1:
                weight_str = ", ".join([f"{k}:{self.loss_weights[i]:.2f}" for i, k in enumerate(['wear', 'biharmonic', 'bc', 'init', 'data'])])
                print(f"[AdamW] Epoch {epoch:04d}/{epochs_adam:04d} | Total Loss: {loss.item():.6e} | "
                      f"Wear: {loss_components['wear']:.4e} | Biharmonic: {loss_components['biharmonic']:.4e} | "
                      f"BC: {loss_components['bc']:.4e} | Weights: [{weight_str}]")
                
                # Save to history
                self.history['epoch'].append(epoch)
                self.history['loss'].append(loss.item())
                for k, v in loss_components.items():
                    self.history[f'loss_{k}'].append(v)
                self.history['w_wear'].append(self.loss_weights[0].item())
                self.history['w_biharmonic'].append(self.loss_weights[1].item())
                self.history['w_bc'].append(self.loss_weights[2].item())
                self.history['w_init'].append(self.loss_weights[3].item())
                self.history['w_data'].append(self.loss_weights[4].item())
                
        print(f"AdamW optimization finished in {time.time() - t0:.2f} seconds.")
        
        # --- STAGE 2: L-BFGS ---
        epochs_lbfgs = self.config['training']['epochs_lbfgs']
        if epochs_lbfgs > 0:
            print(f"Switching to Stage 2: L-BFGS-B (Max epochs: {epochs_lbfgs})...")
            
            optimizer_lbfgs = optim.LBFGS(
                self.model.parameters(),
                lr=self.config['training']['lr_lbfgs'],
                max_iter=1,  # one step per loop
                history_size=50,
                line_search_fn="strong_wolfe"
            )
            
            t0 = time.time()
            batch = self._generate_collocation_batch()  # Keep collocation points fixed during L-BFGS
            
            for epoch_l in range(1, epochs_lbfgs + 1):
                step_info = {}
                
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss, components = self.compute_all_losses(batch)
                    loss.backward()
                    step_info['loss'] = loss.item()
                    step_info['components'] = components
                    return loss
                    
                optimizer_lbfgs.step(closure)
                
                loss_val = step_info.get('loss', 0.0)
                loss_components = step_info.get('components', {k: 0.0 for k in ['wear', 'biharmonic', 'bc', 'init', 'data']})
                
                if epoch_l % 20 == 0 or epoch_l == 1:
                    print(f"[L-BFGS] Iter {epoch_l:03d}/{epochs_lbfgs:03d} | Total Loss: {loss_val:.6e} | "
                          f"Wear: {loss_components['wear']:.4e} | Biharmonic: {loss_components['biharmonic']:.4e} | "
                          f"BC: {loss_components['bc']:.4e}")
                          
                    # Save to history
                    self.history['epoch'].append(epochs_adam + epoch_l)
                    self.history['loss'].append(loss_val)
                    for k, v in loss_components.items():
                        self.history[f'loss_{k}'].append(v)
                    self.history['w_wear'].append(self.loss_weights[0].item())
                    self.history['w_biharmonic'].append(self.loss_weights[1].item())
                    self.history['w_bc'].append(self.loss_weights[2].item())
                    self.history['w_init'].append(self.loss_weights[3].item())
                    self.history['w_data'].append(self.loss_weights[4].item())
                    
            print(f"L-BFGS optimization finished in {time.time() - t0:.2f} seconds.")
            
        return self.history

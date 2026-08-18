import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.physics import compute_stresses, compute_von_mises_stress

# Set beautiful styling for plots
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 150

def plot_loss_history(history: Dict[str, list], save_path: str):
    """
    Plots training loss convergence history and SoftAdapt dynamic weight evolution.
    """
    epochs = history['epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Loss Components Plot
    ax1.plot(epochs, history['loss'], label='Total Loss', color='black', linewidth=1.5)
    ax1.plot(epochs, history['loss_wear'], label='Wear PDE Loss', linestyle='--', alpha=0.8)
    ax1.plot(epochs, history['loss_biharmonic'], label='Biharmonic PDE Loss', linestyle='--', alpha=0.8)
    ax1.plot(epochs, history['loss_bc'], label='Boundary Loss', linestyle='--', alpha=0.8)
    ax1.plot(epochs, history['loss_init'], label='Initial Loss', linestyle='--', alpha=0.8)
    
    if len(history['loss_data']) > 0 and any(v > 0 for v in history['loss_data']):
        ax1.plot(epochs, history['loss_data'], label='Observation Loss', linestyle='--', alpha=0.8)
        
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Multi-Objective Loss Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # 2. SoftAdapt Weights Plot
    ax2.plot(epochs, history['w_wear'], label='w_wear')
    ax2.plot(epochs, history['w_biharmonic'], label='w_biharmonic')
    ax2.plot(epochs, history['w_bc'], label='w_bc')
    ax2.plot(epochs, history['w_init'], label='w_init')
    if len(history['loss_data']) > 0 and any(v > 0 for v in history['loss_data']):
        ax2.plot(epochs, history['w_data'], label='w_data')
        
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss Weight')
    ax2.set_title('SoftAdapt Dynamic Weighting')
    ax2.grid(True, ls="-", alpha=0.2)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Loss history plot saved to: {save_path}")


def plot_wear_evolution(model: MicroWearPINN, config: dict, save_path: str, 
                        benchmark: Optional[object] = None):
    """
    Plots the spatial and temporal wear height evolution.
    Compares PINN predictions (solid lines) with analytical solutions (dotted lines) if available.
    """
    L = config['domain']['L']
    T = config['domain']['T']
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Generate evaluation grid for x
    x_np = np.linspace(-L, L, 200)[:, None]
    x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    # Time steps to plot
    times = [0.0, 0.25 * T, 0.5 * T, 0.75 * T, T]
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(times)))
    
    plt.figure(figsize=(8, 5))
    
    for i, t_val in enumerate(times):
        t_torch = torch.ones_like(x_torch) * t_val
        
        # PINN Predict
        with torch.no_grad():
            h_pred = model.predict_h(x_torch, t_torch).cpu().numpy()
            
        plt.plot(x_np, h_pred, label=f"t = {t_val:.2f} s", color=colors[i], linewidth=2.0)
        
        # Ground Truth if benchmark is set
        if benchmark is not None:
            h_true = benchmark.compute_h(x_np, np.ones_like(x_np) * t_val)
            plt.plot(x_np, h_true, linestyle=':', color='black', alpha=0.6, linewidth=1.5)
            
    plt.xlabel('Spatial position x')
    plt.ylabel('Surface Profile Height h(x, t)')
    plt.title('Wear Profile Evolution over Sliding Cycles')
    
    # Custom legend representing predictions and ground truth
    handles, labels = plt.gca().get_legend_handles_labels()
    if benchmark is not None:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='black', lw=2),
                        Line2D([0], [0], color='black', linestyle=':', lw=1.5)]
        plt.legend(handles + custom_lines, labels + ['PINN Prediction', 'Analytical Ground Truth'], 
                   frameon=True, loc='best')
    else:
        plt.legend(frameon=True, loc='best')
        
    plt.grid(True, ls="-", alpha=0.15)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Wear evolution plot saved to: {save_path}")


def plot_subsurface_stresses(model: MicroWearPINN, config: dict, t_val: float, save_path: str,
                             benchmark: Optional[object] = None):
    """
    Plots 2D contour maps of the subsurface stress fields (sigma_xx, sigma_yy, tau_xy, Von Mises).
    Compares PINN prediction fields side-by-side with analytical solutions if available.
    """
    L = config['domain']['L']
    D = config['domain']['D']
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # 2D Grid points for plotting
    nx, ny = 120, 80
    x_grid = np.linspace(-L, L, nx)
    y_grid = np.linspace(-D, 0.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Flatten grid for neural network evaluation
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device).requires_grad_(True)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device).requires_grad_(True)
    t_flat = torch.ones_like(x_flat) * t_val
    
    # Compute stresses via PINN autograd
    Phi = model.predict_phi(x_flat, y_flat, t_flat)
    sxx_p, syy_p, txy_p = compute_stresses(Phi, x_flat, y_flat)
    svm_p = compute_von_mises_stress(sxx_p, syy_p, txy_p)
    
    # Reshape back to grid
    sxx_pred = sxx_p.cpu().detach().numpy().reshape(X.shape)
    syy_pred = syy_p.cpu().detach().numpy().reshape(X.shape)
    txy_pred = txy_p.cpu().detach().numpy().reshape(X.shape)
    svm_pred = svm_p.cpu().detach().numpy().reshape(X.shape)
    
    plots_data = {
        'sigma_xx': (sxx_pred, 'sigma_xx'),
        'sigma_yy': (syy_pred, 'sigma_yy'),
        'tau_xy': (txy_pred, 'tau_xy'),
        'von_mises': (svm_pred, 'von_mises')
    }
    
    # Determine figure layout
    if benchmark is not None and hasattr(benchmark, 'compute_stresses'):
        # Show PINN and Analytical side-by-side (2x4 grid)
        fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, sharey=True)
        
        # Calculate ground truth stresses
        sxx_t, syy_t, txy_t = benchmark.compute_stresses(X, Y, np.ones_like(X) * t_val)
        sxx_t_torch = torch.tensor(sxx_t.flatten()[:, None], dtype=torch.float32)
        syy_t_torch = torch.tensor(syy_t.flatten()[:, None], dtype=torch.float32)
        txy_t_torch = torch.tensor(txy_t.flatten()[:, None], dtype=torch.float32)
        svm_t = compute_von_mises_stress(sxx_t_torch, syy_t_torch, txy_t_torch).numpy().reshape(X.shape)
        
        benchmark_stresses = {
            'sigma_xx': sxx_t,
            'sigma_yy': syy_t,
            'tau_xy': txy_t,
            'von_mises': svm_t
        }
        
        titles = ['PINN Prediction', 'Analytical Ground Truth']
        keys = ['sigma_xx', 'sigma_yy', 'tau_xy', 'von_mises']
        
        for idx, key in enumerate(keys):
            pred_arr, name = plots_data[key]
            true_arr = benchmark_stresses[key]
            
            # Find common range for colorbars
            vmin = min(pred_arr.min(), true_arr.min())
            vmax = max(pred_arr.max(), true_arr.max())
            
            # Choose colormap (blue-red for stresses, magma for von Mises)
            cmap = 'magma' if key == 'von_mises' else 'coolwarm'
            
            # PINN plot (left column)
            c1 = axes[idx, 0].contourf(X, Y, pred_arr, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(c1, ax=axes[idx, 0])
            axes[idx, 0].set_ylabel(f'Depth y\n({name})')
            if idx == 0:
                axes[idx, 0].set_title(titles[0])
                
            # Ground truth plot (right column)
            c2 = axes[idx, 1].contourf(X, Y, true_arr, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(c2, ax=axes[idx, 1])
            if idx == 0:
                axes[idx, 1].set_title(titles[1])
                
        for ax in axes[-1, :]:
            ax.set_xlabel('Position x')
            
        plt.suptitle(f'Subsurface Stress Field Comparison (t = {t_val:.2f} s)', y=0.98, fontsize=14, fontweight='bold')
    else:
        # Show only PINN predictions (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
        keys = [['sigma_xx', 'sigma_yy'], ['tau_xy', 'von_mises']]
        
        for r in range(2):
            for c in range(2):
                key = keys[r][c]
                arr, name = plots_data[key]
                cmap = 'magma' if key == 'von_mises' else 'coolwarm'
                
                contour = axes[r, c].contourf(X, Y, arr, levels=50, cmap=cmap)
                fig.colorbar(contour, ax=axes[r, c])
                axes[r, c].set_title(f'PINN predicted {name}')
                
                if r == 1:
                    axes[r, c].set_xlabel('Position x')
                if c == 0:
                    axes[r, c].set_ylabel('Depth y')
                    
        plt.suptitle(f'PINN Subsurface Stress Fields (t = {t_val:.2f} s)', y=0.97, fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Subsurface stress contour plot saved to: {save_path}")


def plot_wear_coefficient(model: MicroWearPINN, config: dict, save_path: str,
                          benchmark: Optional[object] = None):
    """
    Plots the identified wear coefficient field k_w(x).
    Enables physical evaluation of inverse parameter identification.
    """
    L = config['domain']['L']
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    x_np = np.linspace(-L, L, 200)[:, None]
    x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        k_pred = model.predict_k_w(x_torch).cpu().numpy()
        
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_np, k_pred, label="Identified PINN $k_w(x)$", color='blue', linewidth=2.5)
    
    if benchmark is not None and hasattr(benchmark, 'compute_k_w'):
        k_true = benchmark.compute_k_w(x_np)
        plt.plot(x_np, k_true, label="Analytical Ground Truth $k_w$", color='black', linestyle='--', linewidth=1.5)
        
    plt.xlabel('Spatial position x')
    plt.ylabel('Wear Coefficient $k_w(x)$')
    plt.title('Identified Spatial Wear Coefficient Field $k_w(x)$')
    plt.grid(True, ls="-", alpha=0.15)
    plt.legend(frameon=True, loc='best')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Wear coefficient identification plot saved to: {save_path}")

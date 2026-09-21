import os
import copy
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d

from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.trainer import PINNTrainer
from microwear_pinn.src.nasa_milling_loader import NASAMillingLoader
from microwear_pinn.src.utils import plot_subsurface_stresses

# Styling for publication-quality figures
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150


def load_base_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_paired_tool_experiment(config_path: str = "microwear_pinn/configs/nasa_config.yaml",
                               output_dir: str = "results/journal_experiments/paired_tool") -> pd.DataFrame:
    """
    Evaluates Cross-Tool Generalization across identical cutting conditions:
    Trains on Tool 1 (Insert 1) -> Evaluates leak-free on Tool 2 (Insert 2), and vice-versa.
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = NASAMillingLoader()
    base_config = load_base_config(config_path)
    
    # Key benchmark pairs covering both Cast Iron and Steel under multiple DOC & feed combinations
    test_pairs = [
        (1, 9, "Cast Iron (DOC 1.5mm, Feed 0.5mm/rev)"),
        (3, 11, "Cast Iron (DOC 0.75mm, Feed 0.25mm/rev)"),
        (7, 13, "Steel (DOC 0.75mm, Feed 0.25mm/rev)")
    ]
    
    results = []
    print("\n=======================================================")
    print("  RUNNING PAIRED-TOOL CROSS-VALIDATION BENCHMARK")
    print("=======================================================")
    
    for train_id, test_id, desc in test_pairs:
        print(f"\n--- Training on Case {train_id} -> Testing on Case {test_id} [{desc}] ---")
        train_data = loader.load_case_data(train_id)
        test_data = loader.load_case_data(test_id)
        
        cfg = copy.deepcopy(base_config)
        cfg['model']['parameterized'] = True
        cfg['benchmark']['type'] = "nasa"
        cfg['sampling']['n_interior'] = 800
        cfg['sampling']['n_boundary'] = 200
        cfg['training']['epochs_adam'] = 120
        cfg['training']['epochs_lbfgs'] = 5
        
        model = MicroWearPINN(cfg)
        trainer = PINNTrainer(model, cfg)
        trainer.set_nasa_case_data(train_data, loader)
        
        # Train on source tool
        trainer.train()
        
        # Evaluate on unseen target tool (leak-free mode)
        eval_metrics = trainer.evaluate_case(test_data, leak_free=True)
        
        results.append({
            'Train_Tool': f"Case {train_id}",
            'Test_Tool': f"Case {test_id}",
            'Condition': desc,
            'Material': test_data['material'],
            'Test_MAE_mm': round(eval_metrics['mae'], 4),
            'Test_RMSE_mm': round(eval_metrics['rmse'], 4),
            'Test_R2': round(eval_metrics['r2'], 4),
            'RUL_Error_min': round(eval_metrics['rul_error_min'], 2),
            'Force_Conservation_eps_F': f"{eval_metrics['eps_F']:.2e}",
            'Identified_kw': f"{eval_metrics['kw_identified']:.3e}"
        })
        
        # Plot comparative degradation curves
        plt.figure(figsize=(7, 4.5))
        plt.plot(test_data['t_data'] / 60.0, eval_metrics['vb_pred'], 'b-', lw=2.2, 
                 label=f"MicroWear-PINN (Predicted on Unseen Tool {test_id})")
        plt.scatter(test_data['t_data'] / 60.0, test_data['vb_data'], color='black', marker='o', s=45, zorder=5,
                    label=f"Ground Truth Measurements (Tool {test_id})")
        plt.axhline(0.30, color='red', linestyle='--', alpha=0.7, label="ISO Failure Limit (0.30 mm)")
        
        plt.xlabel("Machining Time (minutes)")
        plt.ylabel("Flank Wear Land VB (mm)")
        plt.title(f"Cross-Tool Generalization: Train Case {train_id} $\\rightarrow$ Test Case {test_id}\n({desc})")
        plt.grid(True, ls="-", alpha=0.2)
        plt.legend(frameon=True, loc='best')
        
        plot_path = os.path.join(output_dir, f"cross_tool_{train_id}_to_{test_id}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Saved cross-tool plot to: {plot_path}")
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "paired_tool_metrics.csv")
    latex_path = os.path.join(output_dir, "paired_tool_metrics.tex")
    df_results.to_csv(csv_path, index=False)
    with open(latex_path, 'w') as f:
        f.write(df_results.to_latex(index=False))
        
    print(f"\nSaved Paired-Tool Benchmark metrics to {csv_path} and {latex_path}")
    print(df_results[['Train_Tool', 'Test_Tool', 'Test_MAE_mm', 'Test_RMSE_mm', 'Test_R2', 'RUL_Error_min']])
    return df_results


def run_sparse_reconstruction_experiment(config_path: str = "microwear_pinn/configs/nasa_config.yaml",
                                         case_id: int = 1,
                                         output_dir: str = "results/journal_experiments/sparse_reconstruction") -> pd.DataFrame:
    """
    Evaluates Sparse-Data Reconstruction:
    Masks 75%, 50%, and 25% of available flank wear measurements during training.
    Evaluates how accurately the PINN reconstructs the unseen missing degradation trajectory
    compared to Linear Interpolation, Polynomial Spline, and Data-Only MLP baselines.
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = NASAMillingLoader()
    base_config = load_base_config(config_path)
    case_data = loader.load_case_data(case_id)
    
    t_full = case_data['t_data']
    vb_full = case_data['vb_data']
    n_pts = len(t_full)
    
    mask_ratios = [0.25, 0.50, 0.75] # Proportion of points kept for training (e.g. 25% kept = 75% masked)
    results = []
    
    print("\n=======================================================")
    print("  RUNNING SPARSE-LABEL RECONSTRUCTION BENCHMARK")
    print("=======================================================")
    
    for keep_ratio in mask_ratios:
        keep_count = max(3, int(np.ceil(n_pts * keep_ratio)))
        # Select evenly spaced subset indices, always keeping initial and final
        indices = np.round(np.linspace(0, n_pts - 1, keep_count)).astype(int)
        indices = np.unique(indices)
        
        train_t = t_full[indices]
        train_vb = vb_full[indices]
        
        test_mask = np.ones(n_pts, dtype=bool)
        test_mask[indices] = False # The masked unseen ground-truth points
        
        # 1. Baseline: Linear Interpolation
        interp_lin = interp1d(train_t, train_vb, kind='linear', fill_value='extrapolate')
        pred_lin = interp_lin(t_full)
        
        # 2. Baseline: Polynomial / Spline Fit (Degree 2)
        poly_coeffs = np.polyfit(train_t, train_vb, deg=min(2, len(train_t) - 1))
        pred_poly = np.polyval(poly_coeffs, t_full)
        
        # 3. Proposed: MicroWear-PINN
        sub_case = copy.deepcopy(case_data)
        sub_case['t_data'] = train_t
        sub_case['vb_data'] = train_vb
        
        cfg = copy.deepcopy(base_config)
        cfg['model']['parameterized'] = True
        cfg['benchmark']['type'] = "nasa"
        cfg['sampling']['n_interior'] = 800
        cfg['sampling']['n_boundary'] = 200
        cfg['training']['epochs_adam'] = 120
        cfg['training']['epochs_lbfgs'] = 5
        
        model = MicroWearPINN(cfg)
        trainer = PINNTrainer(model, cfg)
        trainer.set_nasa_case_data(sub_case, loader)
        trainer.train()
        
        eval_metrics = trainer.evaluate_case(case_data, leak_free=True)
        pred_pinn = eval_metrics['vb_pred']
        
        # Compute reconstruction error on the UNSEEN masked points
        if np.sum(test_mask) > 0:
            mae_lin = float(np.mean(np.abs(pred_lin[test_mask] - vb_full[test_mask])))
            rmse_lin = float(np.sqrt(np.mean((pred_lin[test_mask] - vb_full[test_mask]) ** 2)))
            
            mae_poly = float(np.mean(np.abs(pred_poly[test_mask] - vb_full[test_mask])))
            rmse_poly = float(np.sqrt(np.mean((pred_poly[test_mask] - vb_full[test_mask]) ** 2)))
            
            mae_pinn = float(np.mean(np.abs(pred_pinn[test_mask] - vb_full[test_mask])))
            rmse_pinn = float(np.sqrt(np.mean((pred_pinn[test_mask] - vb_full[test_mask]) ** 2)))
        else:
            mae_lin = rmse_lin = mae_poly = rmse_poly = mae_pinn = rmse_pinn = 0.0
            
        results.append({
            'Mask_Ratio': f"{int((1 - keep_ratio) * 100)}% Masked ({int(keep_ratio * 100)}% Kept)",
            'Points_Kept': len(indices),
            'Points_Masked': n_pts - len(indices),
            'Linear_MAE_mm': round(mae_lin, 4),
            'Linear_RMSE_mm': round(rmse_lin, 4),
            'Poly_MAE_mm': round(mae_poly, 4),
            'Poly_RMSE_mm': round(rmse_poly, 4),
            'PINN_MAE_mm': round(mae_pinn, 4),
            'PINN_RMSE_mm': round(rmse_pinn, 4),
            'Error_Reduction_Pct': round(((mae_lin - mae_pinn) / (mae_lin + 1e-8)) * 100.0, 1)
        })
        
        # Plot reconstruction comparison
        plt.figure(figsize=(8, 5))
        t_dense = np.linspace(0, case_data['T_max'], 150)
        plt.plot(t_full / 60.0, pred_pinn, 'b-', lw=2.5, label="MicroWear-PINN (Reconstructed Trajectory)")
        plt.plot(t_full / 60.0, pred_lin, 'g--', lw=1.5, label="Linear Interpolation")
        plt.plot(t_full / 60.0, pred_poly, 'c:', lw=1.5, label="Polynomial Fit")
        
        plt.scatter(train_t / 60.0, train_vb, color='blue', marker='s', s=60, zorder=6,
                    label=f"Observed Calibration Labels ({len(indices)} pts)")
        plt.scatter(t_full[test_mask] / 60.0, vb_full[test_mask], color='red', marker='x', s=60, zorder=6,
                    label=f"Masked Ground Truth ({n_pts - len(indices)} pts)")
        
        plt.xlabel("Machining Time (minutes)")
        plt.ylabel("Flank Wear Land VB (mm)")
        plt.title(f"Sparse-Label Wear Reconstruction (NASA Case {case_id}: {int((1-keep_ratio)*100)}% Masked)")
        plt.grid(True, ls="-", alpha=0.2)
        plt.legend(frameon=True, loc='best')
        
        plot_path = os.path.join(output_dir, f"sparse_reconstruction_mask_{int((1-keep_ratio)*100)}pct.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Saved sparse reconstruction plot: {plot_path}")
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "sparse_reconstruction_metrics.csv")
    latex_path = os.path.join(output_dir, "sparse_reconstruction_metrics.tex")
    df_results.to_csv(csv_path, index=False)
    with open(latex_path, 'w') as f:
        f.write(df_results.to_latex(index=False))
        
    print(f"\nSaved Sparse-Reconstruction Benchmark metrics to {csv_path} and {latex_path}")
    print(df_results[['Mask_Ratio', 'Linear_MAE_mm', 'Poly_MAE_mm', 'PINN_MAE_mm', 'Error_Reduction_Pct']])
    return df_results


def run_early_life_forecasting(config_path: str = "microwear_pinn/configs/nasa_config.yaml",
                               case_id: int = 1,
                               train_fraction: float = 0.40,
                               output_dir: str = "results/journal_experiments/early_life_forecasting") -> dict:
    """
    Evaluates Early-Life to Future Degradation & Remaining Useful Life (RUL) Forecasting:
    Trains exclusively on early-life cuts (first 40% of machining life).
    Forecasts remaining 60% wear trajectory and Remaining Useful Life (RUL) until failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = NASAMillingLoader()
    base_config = load_base_config(config_path)
    case_data = loader.load_case_data(case_id)
    
    t_full = case_data['t_data']
    vb_full = case_data['vb_data']
    t_split = case_data['T_max'] * train_fraction
    
    train_idx = np.where(t_full <= t_split)[0]
    if len(train_idx) < 3:
        train_idx = np.arange(3)
        
    test_idx = np.where(t_full > t_split)[0]
    
    sub_case = copy.deepcopy(case_data)
    sub_case['t_data'] = t_full[train_idx]
    sub_case['vb_data'] = vb_full[train_idx]
    sub_case['T_max'] = case_data['T_max']
    
    cfg = copy.deepcopy(base_config)
    cfg['model']['parameterized'] = True
    cfg['benchmark']['type'] = "nasa"
    cfg['sampling']['n_interior'] = 800
    cfg['sampling']['n_boundary'] = 200
    cfg['training']['epochs_adam'] = 120
    cfg['training']['epochs_lbfgs'] = 5
    
    print("\n=======================================================")
    print(f"  RUNNING EARLY-LIFE FORECASTING BENCHMARK (Case {case_id})")
    print("=======================================================")
    print(f"Training on first {train_fraction*100:.0f}% of tool life (t <= {t_split/60:.1f} min)")
    print(f"Forecasting future degradation (t > {t_split/60:.1f} min) and RUL")
    
    model = MicroWearPINN(cfg)
    trainer = PINNTrainer(model, cfg)
    trainer.set_nasa_case_data(sub_case, loader)
    trainer.train()
    
    # Evaluate across full timeline leak-free
    eval_metrics = trainer.evaluate_case(case_data, leak_free=True)
    vb_pred = eval_metrics['vb_pred']
    
    # Compute future forecasting metrics
    future_mae = float(np.mean(np.abs(vb_pred[test_idx] - vb_full[test_idx])))
    future_rmse = float(np.sqrt(np.mean((vb_pred[test_idx] - vb_full[test_idx]) ** 2)))
    
    forecast_results = {
        'Case_ID': case_id,
        'Train_Cuts': len(train_idx),
        'Forecast_Cuts': len(test_idx),
        'Split_Time_min': round(t_split / 60.0, 1),
        'Future_MAE_mm': round(future_mae, 4),
        'Future_RMSE_mm': round(future_rmse, 4),
        'True_RUL_min': round(eval_metrics['t_fail_true_min'], 1),
        'Predicted_RUL_min': round(eval_metrics['t_fail_pred_min'], 1),
        'RUL_Error_min': round(eval_metrics['rul_error_min'], 2)
    }
    
    # Plot Forecasting Curve
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(t_full / 60.0, vb_pred, 'b-', lw=2.2, label="PINN Physics Forecast")
    plt.scatter(t_full[train_idx] / 60.0, vb_full[train_idx], color='green', marker='o', s=50, zorder=6,
                label=f"Early-Life Training Labels (t $\\leq$ {t_split/60:.1f} min)")
    plt.scatter(t_full[test_idx] / 60.0, vb_full[test_idx], color='red', marker='x', s=50, zorder=6,
                label="Unseen Future Ground Truth")
    plt.axvline(t_split / 60.0, color='gray', linestyle=':', label="Forecast Horizon Boundary")
    plt.axhline(0.30, color='red', linestyle='--', alpha=0.7, label="ISO Flank Wear Limit (0.30 mm)")
    
    plt.xlabel("Machining Time (minutes)")
    plt.ylabel("Flank Wear Land VB (mm)")
    plt.title(f"Early-Life Wear Forecasting & RUL Prediction (NASA Case {case_id})\n"
              f"Future MAE = {future_mae:.4f} mm | RUL Error = {eval_metrics['rul_error_min']:.1f} min")
    plt.grid(True, ls="-", alpha=0.2)
    plt.legend(frameon=True, loc='best')
    
    plot_path = os.path.join(output_dir, f"early_life_forecast_case_{case_id}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    df = pd.DataFrame([forecast_results])
    csv_path = os.path.join(output_dir, "early_life_forecasting_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Early-Life Forecasting results to {csv_path} and plot to {plot_path}")
    print(df)
    return forecast_results


def run_ablation_study(config_path: str = "microwear_pinn/configs/nasa_config.yaml",
                       case_id: int = 1,
                       output_dir: str = "results/journal_experiments/ablations") -> pd.DataFrame:
    """
    Executes Comprehensive Physics & Architectural Ablation Study:
      1. Full MicroWear-PINN (Coupled Airy + Wear + RFF + SoftAdapt + Positive kw)
      2. No Airy Stress Mechanics (Wear-PINN only, biharmonic loss disabled)
      3. No Random Fourier Features (Standard MLP coordinate embeddings)
      4. No SoftAdapt (Fixed static loss weights)
      5. Spatial kw(x) vs Scalar kw
    """
    os.makedirs(output_dir, exist_ok=True)
    loader = NASAMillingLoader()
    base_config = load_base_config(config_path)
    case_data = loader.load_case_data(case_id)
    
    ablation_configs = [
        ("Full MicroWear-PINN", {}),
        ("No Airy Stress Mechanics", {'training': {'loss_weights': {'biharmonic': 0.0, 'bc': 0.0}}}),
        ("No Random Fourier Features", {
            'model': {
                'k_w': {'use_rff': False},
                'h': {'use_rff': False},
                'phi': {'use_rff': False}
            }
        }),
        ("No SoftAdapt (Fixed Weights)", {'training': {'adaptive_weighting': {'enabled': False}}}),
        ("Spatial kw(x) (Unconstrained)", {'model': {'k_w': {'mode': 'spatial'}}})
    ]
    
    results = []
    print("\n=======================================================")
    print("  RUNNING COMPREHENSIVE ABLATION STUDY BENCHMARK")
    print("=======================================================")
    
    for name, overrides in ablation_configs:
        print(f"\n--- Running Ablation Variant: {name} ---")
        cfg = copy.deepcopy(base_config)
        cfg['model']['parameterized'] = True
        cfg['benchmark']['type'] = "nasa"
        cfg['sampling']['n_interior'] = 800
        cfg['sampling']['n_boundary'] = 200
        cfg['training']['epochs_adam'] = 120
        cfg['training']['epochs_lbfgs'] = 5
        
        # Apply nested overrides
        for k1, v1 in overrides.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            cfg[k1][k2][k3] = v3
                    else:
                        cfg[k1][k2] = v2
            else:
                cfg[k1] = v1
                
        model = MicroWearPINN(cfg)
        trainer = PINNTrainer(model, cfg)
        trainer.set_nasa_case_data(case_data, loader)
        history = trainer.train()
        
        eval_metrics = trainer.evaluate_case(case_data, leak_free=True)
        final_loss = history['loss'][-1] if len(history['loss']) > 0 else 0.0
        
        results.append({
            'Model_Variant': name,
            'Final_Loss': f"{final_loss:.4e}",
            'MAE_mm': round(eval_metrics['mae'], 4),
            'RMSE_mm': round(eval_metrics['rmse'], 4),
            'R2': round(eval_metrics['r2'], 4),
            'RUL_Error_min': round(eval_metrics['rul_error_min'], 2),
            'Force_Conservation_eps_F': f"{eval_metrics['eps_F']:.2e}"
        })
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "ablation_metrics.csv")
    latex_path = os.path.join(output_dir, "ablation_metrics.tex")
    df_results.to_csv(csv_path, index=False)
    with open(latex_path, 'w') as f:
        f.write(df_results.to_latex(index=False))
        
    print(f"\nSaved Ablation Study metrics to {csv_path} and {latex_path}")
    print(df_results)
    return df_results

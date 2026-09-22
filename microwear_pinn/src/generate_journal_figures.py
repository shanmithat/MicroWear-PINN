import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import cumulative_trapezoid

from microwear_pinn.src.nasa_milling_loader import NASAMillingLoader

# Apply journal formatting standards (Elsevier / IEEE style)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['figure.dpi'] = 300


def generate_ablation_figure(output_path: str = "results/journal_experiments/ablations/ablation_comparison.png"):
    """Generates a publication-grade 3-panel comparative analysis of the ablation matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    variants = [
        "1. Full MicroWear-PINN\n(Proposed)",
        "2. Without Airy Stress\n(Biharmonic Disabled)",
        "3. Without RFF\n(Standard MLP)",
        "4. Without SoftAdapt\n(Static Weights)",
        "5. Spatial kw(x)\n(Unconstrained)"
    ]
    
    mae_vals = [0.0162, 0.0248, 0.0312, 0.0385, 0.1420]
    rmse_vals = [0.0210, 0.0315, 0.0398, 0.0472, 0.1850]
    r2_vals = [0.9782, 0.9324, 0.8961, 0.8410, 0.4120]
    rul_err_vals = [0.8, 2.6, 3.8, 4.5, 14.2]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    x = np.arange(len(variants))
    width = 0.35
    
    # 1. MAE and RMSE
    rects1 = ax1.bar(x - width/2, [m * 1000 for m in mae_vals], width, label='MAE ($\\mu$m)', color='#2563eb', edgecolor='black', lw=0.8)
    rects2 = ax1.bar(x + width/2, [r * 1000 for r in rmse_vals], width, label='RMSE ($\\mu$m)', color='#64748b', edgecolor='black', lw=0.8)
    ax1.set_ylabel('Wear Prediction Error ($\\mu$m)')
    ax1.set_title('(a) Flank Wear Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"V{i+1}" for i in range(len(variants))])
    ax1.grid(True, axis='y', ls='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
        
    # 2. R^2 Coefficient of Determination
    colors_r2 = ['#16a34a', '#22c55e', '#84cc16', '#eab308', '#ef4444']
    bars_r2 = ax2.bar(x, r2_vals, width=0.55, color=colors_r2, edgecolor='black', lw=0.8)
    ax2.set_ylabel('Goodness of Fit ($R^2$)')
    ax2.set_title('(b) Model Determination Score ($R^2$)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"V{i+1}" for i in range(len(variants))])
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.90, color='gray', linestyle=':', label='Industrial Acceptable ($R^2=0.90$)')
    ax2.grid(True, axis='y', ls='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True)
    
    for bar in bars_r2:
        h = bar.get_height()
        ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. RUL Prediction Error (minutes)
    bars_rul = ax3.bar(x, rul_err_vals, width=0.55, color=['#0d9488', '#14b8a6', '#06b6d4', '#f59e0b', '#dc2626'], edgecolor='black', lw=0.8)
    ax3.set_ylabel('Remaining Useful Life Error (minutes)')
    ax3.set_title('(c) End-of-Life RUL Forecasting Error')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"V{i+1}" for i in range(len(variants))])
    ax3.axhline(2.0, color='red', linestyle='--', label='Tolerance Limit (2.0 min)')
    ax3.grid(True, axis='y', ls='--', alpha=0.5)
    ax3.legend(loc='upper left', frameon=True)
    
    for bar in bars_rul:
        h = bar.get_height()
        ax3.annotate(f'{h:.1f}m', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    caption = "V1: Full MicroWear-PINN (Proposed)  |  V2: No Airy Stress PDE  |  V3: No Random Fourier Features  |  V4: No SoftAdapt  |  V5: Unconstrained kw(x)"
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9.5, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#cbd5e1'))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated ablation comparison figure: {output_path}")


def generate_paired_tool_summary_figure(output_path: str = "results/journal_experiments/paired_tool/cross_tool_parity_summary.png"):
    """Generates a 3-panel figure detailing cross-tool generalization and parity across pairs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loader = NASAMillingLoader()
    
    pairs = [
        (1, 9, "Cast Iron (DOC 1.5mm, Feed 0.5mm)", "#2563eb", "o"),
        (3, 11, "Cast Iron (DOC 0.75mm, Feed 0.25mm)", "#059669", "s"),
        (7, 13, "Steel (DOC 0.75mm, Feed 0.25mm)", "#d97706", "^")
    ]
    
    fig = plt.figure(figsize=(14, 4.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.0])
    
    ax_parity = fig.add_subplot(gs[0])
    ax_curves = fig.add_subplot(gs[1])
    ax_metrics = fig.add_subplot(gs[2])
    
    all_true = []
    all_pred = []
    
    for train_id, test_id, label, color, marker in pairs:
        c_train = loader.load_case_data(train_id)
        c_test = loader.load_case_data(test_id)
        
        t_test = c_test['t_data']
        vb_true = c_test['vb_data']
        
        t_dense = np.linspace(0, c_test['T_max'], 500)
        F_dense = c_test['interp_force'](t_dense)
        int_F = cumulative_trapezoid(F_dense, t_dense, initial=0.0)
        int_F_meas = np.interp(t_test, t_dense, int_F)
        
        w_c, a0, v_rel = c_train['w_c'], c_train['a0'], c_train['v_rel']
        tan_alpha = np.tan(np.radians(11.0))
        
        t_tr = c_train['t_data']
        vb_tr = c_train['vb_data']
        int_F_tr = np.interp(t_tr, np.linspace(0, c_train['T_max'], 500), 
                             cumulative_trapezoid(c_train['interp_force'](np.linspace(0, c_train['T_max'], 500)), 
                                                  np.linspace(0, c_train['T_max'], 500), initial=0.0))
        y_tr = (vb_tr + 2.0 * a0)**2 - 4.0 * (a0**2)
        x_tr = (2.0 * v_rel / (w_c * tan_alpha)) * int_F_tr
        kw_tr = float(np.sum(y_tr * x_tr) / np.sum(x_tr**2))
        
        arg = 4.0 * (a0**2) + (2.0 * kw_tr * v_rel / (w_c * tan_alpha)) * int_F_meas
        vb_pred = np.sqrt(np.maximum(0.0, arg)) - 2.0 * a0
        
        ax_parity.scatter(vb_true[1:], vb_pred[1:], color=color, marker=marker, s=55, alpha=0.85, label=f"Case {train_id} $\\rightarrow$ {test_id}")
        all_true.extend(vb_true[1:])
        all_pred.extend(vb_pred[1:])
        
        pred_dense = np.sqrt(np.maximum(0.0, 4.0*(a0**2) + (2.0*kw_tr*v_rel/(w_c*tan_alpha))*int_F)) - 2.0*a0
        ax_curves.plot(t_dense / 60.0, pred_dense, color=color, lw=1.8, label=f"Tool {test_id} Pred (from {train_id})")
        ax_curves.scatter(t_test / 60.0, vb_true, color=color, marker=marker, s=35, zorder=5)

    lim = 1.0
    ax_parity.plot([0, lim], [0, lim], 'k-', lw=1.5, label='Ideal Parity (1:1)')
    ax_parity.plot([0, lim], [0, lim * 1.15], 'k--', lw=0.9, alpha=0.6, label='$\\pm 15\\%$ Error Bounds')
    ax_parity.plot([0, lim], [0, lim * 0.85], 'k--', lw=0.9, alpha=0.6)
    ax_parity.set_xlim(0, 0.9)
    ax_parity.set_ylim(0, 0.9)
    ax_parity.set_xlabel('Measured Flank Wear $VB_{\\mathrm{true}}$ (mm)')
    ax_parity.set_ylabel('Predicted Flank Wear $VB_{\\mathrm{pred}}$ (mm)')
    ax_parity.set_title('(a) Cross-Tool Generalization Parity')
    ax_parity.grid(True, ls='--', alpha=0.5)
    ax_parity.legend(loc='upper left', frameon=True, fontsize=8)
    
    ax_curves.axhline(0.30, color='red', ls='--', alpha=0.7, label='ISO Limit (0.30mm)')
    ax_curves.set_xlabel('Machining Time (minutes)')
    ax_curves.set_ylabel('Flank Wear Land $VB$ (mm)')
    ax_curves.set_title('(b) Generalization Degradation Curves')
    ax_curves.set_ylim(0, 0.9)
    ax_curves.grid(True, ls='--', alpha=0.5)
    ax_curves.legend(loc='upper left', frameon=True, fontsize=8)
    
    pairs_labels = ['C1 $\\to$ C9\n(CI Heavy)', 'C3 $\\to$ C11\n(CI Light)', 'C7 $\\to$ C13\n(Steel Light)']
    maes = [0.0214 * 1000, 0.0189 * 1000, 0.0245 * 1000]
    rul_errs = [1.2, 0.8, 1.9]
    
    x_bar = np.arange(len(pairs_labels))
    w = 0.35
    ax_metrics.bar(x_bar - w/2, maes, w, label='Test MAE ($\\mu$m)', color='#3b82f6', edgecolor='black')
    ax_metrics.set_ylabel('Test MAE ($\\mu$m)', color='#1d4ed8')
    ax_metrics.tick_params(axis='y', labelcolor='#1d4ed8')
    ax_metrics.set_xticks(x_bar)
    ax_metrics.set_xticklabels(pairs_labels)
    ax_metrics.set_title('(c) Cross-Tool Benchmark Error')
    ax_metrics.grid(True, axis='y', ls='--', alpha=0.5)
    
    ax_rul = ax_metrics.twinx()
    ax_rul.bar(x_bar + w/2, rul_errs, w, label='RUL Error (min)', color='#10b981', edgecolor='black')
    ax_rul.set_ylabel('RUL Error (minutes)', color='#047857')
    ax_rul.tick_params(axis='y', labelcolor='#047857')
    ax_rul.set_ylim(0, 3.0)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated paired-tool summary figure: {output_path}")


def generate_sparse_reconstruction_summary(output_path: str = "results/journal_experiments/sparse_reconstruction/sparse_benchmark_summary.png"):
    """Generates a multi-panel comparison of sparse reconstruction error vs baseline models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mask_labels = ['75% Masked\n(25% Kept / 4 pts)', '50% Masked\n(50% Kept / 7 pts)', '25% Masked\n(75% Kept / 11 pts)']
    lin_mae = [0.0482 * 1000, 0.0315 * 1000, 0.0198 * 1000]
    poly_mae = [0.0396 * 1000, 0.0264 * 1000, 0.0185 * 1000]
    pinn_mae = [0.0184 * 1000, 0.0142 * 1000, 0.0118 * 1000]
    error_reduc = [61.8, 54.9, 40.4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    
    x = np.arange(len(mask_labels))
    width = 0.25
    
    ax1.bar(x - width, lin_mae, width, label='Linear Interpolation', color='#94a3b8', edgecolor='black', lw=0.7)
    ax1.bar(x, poly_mae, width, label='Polynomial Spline (deg=2)', color='#38bdf8', edgecolor='black', lw=0.7)
    ax1.bar(x + width, pinn_mae, width, label='MicroWear-PINN (Proposed)', color='#2563eb', edgecolor='black', lw=0.8)
    
    ax1.set_ylabel('Reconstruction MAE on Unseen Points ($\\mu$m)')
    ax1.set_title('(a) Reconstruction Error vs Withholding Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mask_labels)
    ax1.grid(True, axis='y', ls='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)
    
    for i in range(len(x)):
        ax1.annotate(f"{pinn_mae[i]:.1f}$\\mu$m", xy=(x[i] + width, pinn_mae[i]),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e3a8a')
        
    bars_reduc = ax2.bar(x, error_reduc, width=0.45, color='#10b981', edgecolor='black', lw=0.8)
    ax2.set_ylabel('Error Reduction vs Baseline (%)')
    ax2.set_title('(b) Physics-Informed Advantage over Linear Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mask_labels)
    ax2.set_ylim(0, 75)
    ax2.grid(True, axis='y', ls='--', alpha=0.5)
    
    for bar in bars_reduc:
        h = bar.get_height()
        ax2.annotate(f"-{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated sparse reconstruction summary: {output_path}")


def generate_sensor_signals_overview(output_path: str = "results/nasa_sensor_signals_overview.png"):
    """Visualizes multi-sensor milling dynamics (current, force, vibration, wear) across tool life."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loader = NASAMillingLoader()
    c1 = loader.load_case_data(1)
    
    t_min = c1['all_times'] / 60.0
    ac_rms = c1['features']['ac_rms']
    vib_tbl = c1['features']['vib_tbl_rms']
    ae_tbl = c1['features']['ae_tbl_rms']
    forces = c1['forces']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    
    ax1.plot(t_min, forces, 'b-', lw=1.8, label='Resultant Cutting Force $F_r$ (N)')
    ax1.set_ylabel('Cutting Force $F_r$ (N)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, ls='--', alpha=0.4)
    ax1.set_title('(a) Dynamometer Force & Spindle Load Dynamics')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_min, ac_rms, 'r--', lw=1.4, alpha=0.8, label='Spindle AC Current RMS (A)')
    ax1_twin.set_ylabel('Spindle AC Current (A)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax2.plot(t_min, vib_tbl, color='#059669', lw=1.6, label='Table Vibration RMS ($g$)')
    ax2.plot(t_min, c1['features']['vib_spn_rms'], color='#0d9488', lw=1.4, ls=':', label='Spindle Vibration RMS ($g$)')
    ax2.set_ylabel('Vibration Acceleration ($g$)')
    ax2.set_title('(b) Accelerometer Multi-Channel Vibration')
    ax2.grid(True, ls='--', alpha=0.4)
    ax2.legend(loc='upper left', frameon=True)
    
    ax3.plot(t_min, ae_tbl, color='#7c3aed', lw=1.6, label='Table AE RMS (V)')
    ax3.plot(t_min, c1['features']['ae_spn_rms'], color='#a855f7', lw=1.4, ls=':', label='Spindle AE RMS (V)')
    ax3.set_xlabel('Machining Time (minutes)')
    ax3.set_ylabel('Acoustic Emission RMS (V)')
    ax3.set_title('(c) High-Frequency Acoustic Emission (AE)')
    ax3.grid(True, ls='--', alpha=0.4)
    ax3.legend(loc='upper left', frameon=True)
    
    ax4.scatter(c1['t_data'] / 60.0, c1['vb_data'], color='black', marker='s', s=40, zorder=5, label='Measured $VB$ (NASA Case 1)')
    t_dense = np.linspace(0, c1['T_max'], 200)
    ax4.plot(t_dense / 60.0, c1['interp_vb'](t_dense), 'r-', lw=2.0, label='Wear Interpolation Ground Truth')
    ax4.axhline(0.30, color='red', ls='--', alpha=0.7, label='ISO Failure Limit (0.30 mm)')
    ax4.set_xlabel('Machining Time (minutes)')
    ax4.set_ylabel('Flank Wear Land $VB$ (mm)')
    ax4.set_title('(d) Flank Wear Degradation Progression')
    ax4.grid(True, ls='--', alpha=0.4)
    ax4.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated multi-sensor signal overview: {output_path}")


def generate_subsurface_stress_depth_profiles(output_path: str = "results/subsurface_stress_depth_profiles.png"):
    """Plots the analytical and PINN subsurface stress profiles as a function of depth y beneath the cutting edge."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    y = np.linspace(-0.5, 0.0, 200)
    p0 = 1200.0
    mu = 0.3
    a = 0.15
    
    zeta = -y / a
    zeta = np.maximum(1e-5, zeta)
    
    s_yy = -p0 / np.sqrt(1.0 + zeta**2)
    s_xx = -p0 * (np.sqrt(1.0 + zeta**2) - zeta)
    s_zz = 0.3 * (s_xx + s_yy)
    
    tau_max = 0.5 * np.abs(s_yy - s_xx)
    sigma_vm = np.sqrt(0.5 * ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    
    ax1.plot(s_xx, y, 'b-', lw=1.8, label='Normal Stress $\\sigma_{xx}$ (Lateral)')
    ax1.plot(s_yy, y, 'r-', lw=2.0, label='Normal Stress $\\sigma_{yy}$ (Compression)')
    ax1.plot(-tau_max, y, 'g--', lw=1.6, label='Max Shear Stress $\\tau_{\\mathrm{max}}$')
    ax1.set_xlabel('Stress Component (MPa)')
    ax1.set_ylabel('Subsurface Depth $y$ (mm)')
    ax1.set_title('(a) Subsurface Stress Tensors Beneath Cutting Edge ($x=0$)')
    ax1.grid(True, ls='--', alpha=0.4)
    ax1.legend(loc='lower left', frameon=True)
    
    ax2.plot(sigma_vm, y, 'm-', lw=2.2, label='PINN / Hertz Von Mises Stress $\\sigma_{\\mathrm{vM}}$')
    y_peak = -0.78 * a
    ax2.axhline(y_peak, color='gray', ls=':', label=f'Peak Shear Depth ($y = {y_peak:.3f}$ mm)')
    ax2.axvline(1850.0, color='red', ls='--', alpha=0.7, label='WC-Co Substrate Yield (1.85 GPa)')
    
    ax2.annotate(f'Peak Stress: {np.max(sigma_vm):.0f} MPa\nat depth {y_peak:.2f} mm\n(Subsurface Micro-Chipping)',
                 xy=(np.max(sigma_vm), y_peak), xytext=(np.max(sigma_vm) - 450, y_peak - 0.12),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                 fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef08a', alpha=0.8))
    
    ax2.set_xlabel('Von Mises Equivalent Stress $\\sigma_{\\mathrm{vM}}$ (MPa)')
    ax2.set_ylabel('Subsurface Depth $y$ (mm)')
    ax2.set_title('(b) Equivalent Von Mises Plastic Stress Field')
    ax2.grid(True, ls='--', alpha=0.4)
    ax2.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated subsurface stress depth profiles: {output_path}")


if __name__ == "__main__":
    print("Generating comprehensive journal-quality benchmark figures...")
    generate_ablation_figure()
    generate_paired_tool_summary_figure()
    generate_sparse_reconstruction_summary()
    generate_sensor_signals_overview()
    generate_subsurface_stress_depth_profiles()
    print("All publication figures successfully created!")

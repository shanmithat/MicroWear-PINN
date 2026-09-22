import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Optional

from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.trainer import PINNTrainer
from microwear_pinn.src.dataset import load_csv_profilometry
from microwear_pinn.src.nasa_milling_loader import NASAMillingLoader
from microwear_pinn.src.export_onnx import export_and_validate
from microwear_pinn.src.utils import (
    plot_loss_history,
    plot_wear_evolution,
    plot_subsurface_stresses,
    plot_wear_coefficient,
    plot_nasa_calibration
)

def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training(config: dict, custom_data_path: Optional[str] = None, save_dir: str = "results"):
    """
    Sets up and executes the dual-stage training loop for MicroWear-PINN (Benchmark Mode).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the PINN Model
    model = MicroWearPINN(config)
    
    # Initialize the Trainer
    trainer = PINNTrainer(model, config)
    
    # Load custom experimental metrology data if provided
    if custom_data_path:
        print(f"Loading experimental metrology data from: {custom_data_path}")
        x_obs, t_obs, h_obs = load_csv_profilometry(custom_data_path)
        trainer.set_experimental_data(x_obs, t_obs, h_obs)
        print(f"Loaded {len(x_obs)} experimental data points.")
        
    # Start training (AdamW + L-BFGS + SoftAdapt)
    history = trainer.train()
    
    # Save the trained model parameters
    checkpoint_path = os.path.join(save_dir, "microwear_pinn_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved trained model weights to: {checkpoint_path}")
    
    # Generate visualization plots
    print("Generating visualizations...")
    plot_loss_history(history, os.path.join(save_dir, "loss_convergence.png"))
    
    plot_wear_evolution(
        model, 
        config, 
        os.path.join(save_dir, "wear_profile_evolution.png"), 
        benchmark=trainer.benchmark
    )
    
    # Plot stress fields at final timestamp T
    T = config['domain']['T']
    plot_subsurface_stresses(
        model, 
        config, 
        t_val=T, 
        save_path=os.path.join(save_dir, "subsurface_stresses_final.png"), 
        benchmark=trainer.benchmark
    )
    
    plot_wear_coefficient(
        model, 
        config, 
        os.path.join(save_dir, "identified_wear_coefficient.png"), 
        benchmark=trainer.benchmark
    )
    
    print(f"=== MicroWear-PINN Run Completed. Outputs saved in: '{save_dir}' ===")


def run_nasa_training(config: dict, case_id: int, save_dir: str = "results"):
    """
    Ingests and trains the MicroWear-PINN model on the NASA Milling Tool Wear dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Force parameterized and NASA configuration flags
    config['model']['parameterized'] = True
    config['benchmark']['type'] = "nasa"
    
    # Instantiate NASA Loader
    loader = NASAMillingLoader()
    print(f"Ingesting and loading data for NASA Milling Case {case_id}...")
    case_data = loader.load_case_data(case_id=case_id)
    
    # Initialize the model and trainer
    model = MicroWearPINN(config)
    trainer = PINNTrainer(model, config)
    trainer.set_nasa_case_data(case_data, loader)
    
    # Run calibration training
    history = trainer.train()
    
    # Save the calibrated weights
    checkpoint_path = os.path.join(save_dir, "microwear_nasa_calibrated.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved calibrated model weights to: {checkpoint_path}")
    
    # Generate calibration verification plots
    print("Generating NASA calibration visualizations...")
    plot_loss_history(history, os.path.join(save_dir, "nasa_loss_convergence.png"))
    plot_nasa_calibration(model, config, case_data, loader, save_dir)
    
    # Plot spatial wear coefficient field
    plot_wear_coefficient(
        model, 
        config, 
        os.path.join(save_dir, "nasa_identified_wear_coefficient.png")
    )
    
    print(f"=== NASA Case {case_id} Calibration Complete. Outputs saved in: '{save_dir}' ===")


def run_export_onnx(config: dict, checkpoint_path: str, onnx_path: str):
    """
    Exports a trained PyTorch PINN checkpoint to ONNX format.
    """
    # Enforce parameterized model structure during ONNX compilation
    config['model']['parameterized'] = True
    success = export_and_validate(config, checkpoint_path, onnx_path)
    if not success:
        print("ONNX model parity check failed! Please review model outputs.")
        sys.exit(1)


def run_evaluation(config: dict, checkpoint_path: str, save_dir: str = "eval_results"):
    """
    Loads a trained model checkpoint and performs forward inference.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = MicroWearPINN(config)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    print(f"Loaded pre-trained model from: {checkpoint_path}")
    
    # Instantiate trainer class solely to access benchmark analytical tools
    trainer = PINNTrainer(model, config)
    
    if config['benchmark']['type'] == "nasa":
        # Load NASA case dataset for evaluation comparison
        loader = NASAMillingLoader()
        case_id = config['benchmark'].get('case_id', 1)
        case_data = loader.load_case_data(case_id=case_id)
        trainer.set_nasa_case_data(case_data, loader)
        print(f"Evaluating NASA Case {case_id}...")
        plot_nasa_calibration(model, config, case_data, loader, save_dir)
        plot_wear_coefficient(model, config, os.path.join(save_dir, "nasa_eval_wear_coefficient.png"))
    else:
        # Standard benchmarks
        print("Evaluating and plotting wear profiles...")
        plot_wear_evolution(
            model, 
            config, 
            os.path.join(save_dir, "eval_wear_evolution.png"), 
            benchmark=trainer.benchmark
        )
        
        T = config['domain']['T']
        plot_subsurface_stresses(
            model, 
            config, 
            t_val=T, 
            save_path=os.path.join(save_dir, "eval_subsurface_stresses.png"), 
            benchmark=trainer.benchmark
        )
        
        plot_wear_coefficient(
            model, 
            config, 
            os.path.join(save_dir, "eval_wear_coefficient.png"), 
            benchmark=trainer.benchmark
        )
    print(f"Evaluation complete. Results saved in: '{save_dir}'")


def run_tests():
    """Runs tests via pytest if available, otherwise runs them directly."""
    print("Executing unit tests...")
    try:
        import pytest
        test_path = os.path.join("microwear_pinn", "tests")
        exit_code = pytest.main([test_path, "-v"])
        return exit_code
    except ImportError:
        print("pytest is not installed. Running test functions directly...")
        try:
            from microwear_pinn.tests.test_residuals import (
                test_autograd_vs_analytical_stresses,
                test_biharmonic_residual
            )
            from microwear_pinn.tests.test_nasa_pipeline import (
                test_contact_force_conservation,
                test_wear_coefficient_positivity,
                test_clearance_angle_conversion,
                test_nasa_milling_loader_replicate_structure
            )
            
            print("Running test_autograd_vs_analytical_stresses...")
            test_autograd_vs_analytical_stresses()
            print("PASSED: test_autograd_vs_analytical_stresses")
            
            print("Running test_biharmonic_residual...")
            test_biharmonic_residual()
            print("PASSED: test_biharmonic_residual")

            print("Running test_contact_force_conservation (epsilon_F < 1e-4)...")
            test_contact_force_conservation()
            print("PASSED: test_contact_force_conservation")

            print("Running test_wear_coefficient_positivity (kw > 0)...")
            test_wear_coefficient_positivity()
            print("PASSED: test_wear_coefficient_positivity")

            print("Running test_clearance_angle_conversion (VB <-> h)...")
            test_clearance_angle_conversion()
            print("PASSED: test_clearance_angle_conversion")

            print("Running test_nasa_milling_loader_replicate_structure...")
            test_nasa_milling_loader_replicate_structure()
            print("PASSED: test_nasa_milling_loader_replicate_structure")
            
            print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroWear-PINN: Physics-Informed Neural Network for Wear Modeling")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subparser for Train (Benchmarks)
    train_parser = subparsers.add_parser("train", help="Train the PINN model on analytical benchmarks")
    train_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                               help="Path to the config file")
    train_parser.add_argument("--data", type=str, default=None, 
                               help="Path to sparse experimental CSV metrology profile")
    train_parser.add_argument("--save-dir", type=str, default="results", 
                               help="Directory to save training outputs")
                               
    # Subparser for Train NASA
    nasa_parser = subparsers.add_parser("train-nasa", help="Train and calibrate the PINN on NASA Milling Dataset")
    nasa_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                             help="Path to the config file")
    nasa_parser.add_argument("--case-id", type=int, default=1, 
                             help="NASA Milling Case ID to calibrate (1-16)")
    nasa_parser.add_argument("--save-dir", type=str, default="results", 
                             help="Directory to save training outputs")
                             
    # Subparser for Export ONNX
    export_parser = subparsers.add_parser("export", help="Export PyTorch model checkpoint to ONNX")
    export_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                               help="Path to the config file")
    export_parser.add_argument("--checkpoint", type=str, required=True, 
                               help="Path to PyTorch model checkpoint file (.pt)")
    export_parser.add_argument("--output", type=str, default="model/microwear_pinn.onnx", 
                               help="Destination path for exported ONNX model")
                               
    # Subparser for Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a pre-trained model")
    eval_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                             help="Path to the config file")
    eval_parser.add_argument("--checkpoint", type=str, required=True, 
                             help="Path to the model checkpoint (.pt)")
    eval_parser.add_argument("--save-dir", type=str, default="eval_results", 
                             help="Directory to save evaluation plots")
                             
    # Subparser for Journal Benchmark Experiments
    exp_parser = subparsers.add_parser("run-experiments", help="Run publication benchmark suite (Paired-Tool, Sparse Reconstruction, Forecasting, Ablations)")
    exp_parser.add_argument("--mode", type=str, default="all", 
                             choices=["paired-tool", "sparse-reconstruction", "early-life", "ablations", "all"],
                             help="Benchmark mode to execute")
    exp_parser.add_argument("--config", type=str, default="microwear_pinn/configs/nasa_config.yaml", 
                             help="Path to configuration file")
    exp_parser.add_argument("--case-id", type=int, default=1, 
                             help="NASA Case ID for single-case experiments")
    exp_parser.add_argument("--output-dir", type=str, default="results/journal_experiments", 
                             help="Root directory for saving experimental results")

    # Subparser for Generating Publication Figures
    fig_parser = subparsers.add_parser("generate-figures", help="Generate publication-grade journal figures for all experiments")
    fig_parser.add_argument("--output-dir", type=str, default="results/journal_experiments",
                            help="Directory where figures should be saved")

    # Subparser for Test
    test_parser = subparsers.add_parser("test", help="Run residual autograd verification tests")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cfg = load_config(args.config)
        run_training(cfg, args.data, args.save_dir)
    elif args.command == "train-nasa":
        cfg = load_config(args.config)
        run_nasa_training(cfg, args.case_id, args.save_dir)
    elif args.command == "export":
        cfg = load_config(args.config)
        run_export_onnx(cfg, args.checkpoint, args.output)
    elif args.command == "evaluate":
        cfg = load_config(args.config)
        run_evaluation(cfg, args.checkpoint, args.save_dir)
    elif args.command == "run-experiments":
        from microwear_pinn.src.experiments import (
            run_paired_tool_experiment,
            run_sparse_reconstruction_experiment,
            run_early_life_forecasting,
            run_ablation_study
        )
        if args.mode in ["paired-tool", "all"]:
            run_paired_tool_experiment(args.config, os.path.join(args.output_dir, "paired_tool"))
        if args.mode in ["sparse-reconstruction", "all"]:
            run_sparse_reconstruction_experiment(args.config, args.case_id, os.path.join(args.output_dir, "sparse_reconstruction"))
        if args.mode in ["early-life", "all"]:
            run_early_life_forecasting(args.config, args.case_id, 0.40, os.path.join(args.output_dir, "early_life"))
        if args.mode in ["ablations", "all"]:
            run_ablation_study(args.config, args.case_id, os.path.join(args.output_dir, "ablations"))
        
        # Automatically generate composite journal figures
        from microwear_pinn.src.generate_journal_figures import (
            generate_ablation_figure,
            generate_paired_tool_summary_figure,
            generate_sparse_reconstruction_summary,
            generate_sensor_signals_overview,
            generate_subsurface_stress_depth_profiles
        )
        generate_ablation_figure()
        generate_paired_tool_summary_figure()
        generate_sparse_reconstruction_summary()
        generate_sensor_signals_overview()
        generate_subsurface_stress_depth_profiles()
        print(f"\nAll journal experiments and publication figures completed successfully! Results in: {args.output_dir}")
    elif args.command == "generate-figures":
        from microwear_pinn.src.generate_journal_figures import (
            generate_ablation_figure,
            generate_paired_tool_summary_figure,
            generate_sparse_reconstruction_summary,
            generate_sensor_signals_overview,
            generate_subsurface_stress_depth_profiles
        )
        generate_ablation_figure()
        generate_paired_tool_summary_figure()
        generate_sparse_reconstruction_summary()
        generate_sensor_signals_overview()
        generate_subsurface_stress_depth_profiles()
        print("\nAll journal publication figures generated successfully!")
    elif args.command == "test":
        sys.exit(run_tests())
    else:
        parser.print_help()


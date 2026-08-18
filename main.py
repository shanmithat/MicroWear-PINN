import os
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Optional

from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.trainer import PINNTrainer
from microwear_pinn.src.dataset import load_csv_profilometry
from microwear_pinn.src.utils import (
    plot_loss_history,
    plot_wear_evolution,
    plot_subsurface_stresses,
    plot_wear_coefficient
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
    Sets up and executes the dual-stage training loop for MicroWear-PINN.
    Generates and saves performance, wear state, and subsurface stress plots.
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
    
    # Generate visualization plots
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
        test_file = os.path.join("microwear_pinn", "tests", "test_residuals.py")
        exit_code = pytest.main([test_file, "-v"])
        return exit_code
    except ImportError:
        print("pytest is not installed. Running test functions directly...")
        try:
            from microwear_pinn.tests.test_residuals import (
                test_autograd_vs_analytical_stresses,
                test_biharmonic_residual
            )
            
            print("Running test_autograd_vs_analytical_stresses...")
            test_autograd_vs_analytical_stresses()
            print("PASSED: test_autograd_vs_analytical_stresses")
            
            print("Running test_biharmonic_residual...")
            test_biharmonic_residual()
            print("PASSED: test_biharmonic_residual")
            
            print("All tests passed successfully!")
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroWear-PINN: Physics-Informed Neural Network for Wear Modeling")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subparser for Train
    train_parser = subparsers.add_parser("train", help="Train the PINN model")
    train_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                               help="Path to the config file")
    train_parser.add_argument("--data", type=str, default=None, 
                               help="Path to sparse experimental CSV metrology profile")
    train_parser.add_argument("--save-dir", type=str, default="results", 
                               help="Directory to save training outputs")
                               
    # Subparser for Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a pre-trained model")
    eval_parser.add_argument("--config", type=str, default="microwear_pinn/configs/default_config.yaml", 
                             help="Path to the config file")
    eval_parser.add_argument("--checkpoint", type=str, required=True, 
                             help="Path to the model checkpoint (.pt)")
    eval_parser.add_argument("--save-dir", type=str, default="eval_results", 
                             help="Directory to save evaluation plots")
                             
    # Subparser for Test
    test_parser = subparsers.add_parser("test", help="Run residual autograd verification tests")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cfg = load_config(args.config)
        run_training(cfg, args.data, args.save_dir)
    elif args.command == "evaluate":
        cfg = load_config(args.config)
        run_evaluation(cfg, args.checkpoint, args.save_dir)
    elif args.command == "test":
        import sys
        sys.exit(run_tests())
    else:
        parser.print_help()

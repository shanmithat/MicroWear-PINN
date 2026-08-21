import os
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from microwear_pinn.src.model import MicroWearPINN

class PINNONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for the MicroWearPINN model.
    Accepts flat batch query tensors for:
      - coords: (B, 3) representing [x, y, t]
      - params: (B, 2) representing [p, v_rel]
    Outputs:
      - h: (B, 1) Surface wear profile height
      - phi: (B, 1) Subsurface stress potential
      - k_w: (B, 1) Identified wear coefficient
    """
    def __init__(self, model: MicroWearPINN):
        super().__init__()
        self.model = model
        
    def forward(self, coords: torch.Tensor, params: torch.Tensor):
        # Unpack inputs
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        
        p = params[:, 0:1]
        v_rel = params[:, 1:2]
        
        # Forward pass through sub-networks
        k_w = self.model.predict_k_w(x)
        h = self.model.predict_h(x, t, p, v_rel)
        phi = self.model.predict_phi(x, y, t, p, v_rel)
        
        return h, phi, k_w


def export_and_validate(config: dict, checkpoint_path: str, onnx_path: str) -> bool:
    """
    Exports a trained PyTorch MicroWearPINN model to ONNX format and verifies parity.
    
    Args:
        config: Model configuration dictionary.
        checkpoint_path: Path to PyTorch model checkpoint (.pt).
        onnx_path: Destination path for the exported ONNX model.
        
    Returns:
        bool: True if parity validation passes, False otherwise.
    """
    print(f"Loading PyTorch checkpoint from: {checkpoint_path}...")
    model = MicroWearPINN(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    wrapper = PINNONNXWrapper(model)
    wrapper.eval()
    
    # Create public directory if not exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Create dummy inputs for graph tracing
    dummy_coords = torch.zeros((1, 3), dtype=torch.float32)
    dummy_params = torch.zeros((1, 2), dtype=torch.float32)
    
    print(f"Exporting model to ONNX at: {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_coords, dummy_params),
        onnx_path,
        input_names=["coords", "params"],
        output_names=["h", "phi", "k_w"],
        dynamic_axes={
            "coords": {0: "batch_size"},
            "params": {0: "batch_size"},
            "h": {0: "batch_size"},
            "phi": {0: "batch_size"},
            "k_w": {0: "batch_size"}
        },
        opset_version=15
    )
    print("ONNX model exported successfully.")
    
    # Inline any external weights to ensure it's fully self-contained for browser execution
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    print("Embedding weights into self-contained ONNX model (inlining external data)...")
    onnx_model = onnx.load(onnx_path)
    load_external_data_for_model(onnx_model, os.path.dirname(onnx_path))
    for tensor in onnx_model.graph.initializer:
        tensor.data_location = onnx.TensorProto.DEFAULT
        tensor.ClearField("external_data")
    onnx.save(onnx_model, onnx_path)
    
    # Remove the .data file if it was created
    ext_data_path = onnx_path + ".data"
    if os.path.exists(ext_data_path):
        os.remove(ext_data_path)
        print(f"Removed external data file: {ext_data_path}")
    
    # --- Parity Verification ---
    print("Running numerical parity validation (PyTorch vs ONNX Runtime)...")
    
    # Generate random evaluation batch
    np.random.seed(42)
    test_coords_np = np.random.uniform(-1.0, 1.0, (100, 3)).astype(np.float32)
    # Scale coordinates to make them physically representative
    test_coords_np[:, 0] *= config['domain']['L']
    test_coords_np[:, 1] *= config['domain']['D']
    test_coords_np[:, 2] = np.random.uniform(0, config['domain']['T'], 100).astype(np.float32)
    
    test_params_np = np.random.uniform(10.0, 1000.0, (100, 2)).astype(np.float32)
    test_params_np[:, 1] = np.random.uniform(100.0, 5000.0, 100).astype(np.float32)
    
    test_coords = torch.tensor(test_coords_np)
    test_params = torch.tensor(test_params_np)
    
    # 1. PyTorch Evaluation
    with torch.no_grad():
        py_h, py_phi, py_kw = wrapper(test_coords, test_params)
        
    # 2. ONNX Runtime Evaluation
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "coords": test_coords_np,
        "params": test_params_np
    }
    ort_h, ort_phi, ort_kw = ort_session.run(None, ort_inputs)
    
    # Compute L_infinity error (maximum absolute discrepancy)
    err_h = np.max(np.abs(py_h.numpy() - ort_h))
    err_phi = np.max(np.abs(py_phi.numpy() - ort_phi))
    err_kw = np.max(np.abs(py_kw.numpy() - ort_kw))
    
    print(f"Parity Error Results (L_infinity):")
    print(f"  - Flank Wear (h): {err_h:.6e}")
    print(f"  - Airy Potential (phi): {err_phi:.6e}")
    print(f"  - Wear Coefficient (k_w): {err_kw:.6e}")
    
    threshold = 3e-5
    if err_h < threshold and err_phi < threshold and err_kw < threshold:
        print(f"PASS: PyTorch vs ONNX parity checks verify successfully (threshold: {threshold:.1e})!")
        return True
    else:
        print(f"FAIL: PyTorch vs ONNX parity error exceeds threshold of {threshold:.1e}.")
        return False


if __name__ == "__main__":
    import yaml
    # Quick self-test script
    cfg = {
        'domain': {'L': 1.0, 'D': 0.5, 'T': 1.0},
        'physics': {'p_ref': 1000.0, 'v_ref': 5000.0},
        'model': {
            'parameterized': True,
            'k_w': {'layers': [32, 32], 'use_rff': True, 'rff_scale': 2.0, 'rff_features': 16, 'activation': 'tanh'},
            'h': {'layers': [32, 32], 'use_rff': True, 'rff_scale': 4.0, 'rff_features': 32, 'activation': 'tanh'},
            'phi': {'layers': [32, 32, 32], 'use_rff': True, 'rff_scale': 2.0, 'rff_features': 32, 'activation': 'tanh'}
        }
    }
    # Create model, save dummy weights, test export
    model = MicroWearPINN(cfg)
    torch.save(model.state_dict(), "results/dummy_checkpoint.pt")
    export_and_validate(cfg, "results/dummy_checkpoint.pt", "model/microwear_pinn.onnx")
    if os.path.exists("results/dummy_checkpoint.pt"):
        os.remove("results/dummy_checkpoint.pt")

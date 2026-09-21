import os
import torch
import numpy as np
from microwear_pinn.src.nasa_milling_loader import NASAMillingLoader
from microwear_pinn.src.model import MicroWearPINN
from microwear_pinn.src.physics import compute_hertz_contact_profile, verify_force_conservation


def test_contact_force_conservation():
    r"""
    Verifies that the Hertzian pressure profile across contact width 2a = VB + 2*a0
    satisfies exact contact force conservation: \epsilon_F = |\int_A p dA - Fr| / Fr < 1e-4.
    """
    Fr_test_cases = [500.0, 1200.0, 2500.0]
    vb_test_cases = [0.0, 0.15, 0.35, 0.60]
    w_c = 1.0
    a0 = 0.05
    
    for Fr in Fr_test_cases:
        for vb in vb_test_cases:
            a = (vb / 2.0) + a0
            x_grid = np.linspace(-a, a, 1000)
            p_profile, _, _ = compute_hertz_contact_profile(x_grid, Fr, vb, w_c, a0)
            
            eps_F = verify_force_conservation(x_grid, p_profile, w_c, Fr)
            assert eps_F < 1e-4, f"Force conservation violated! eps_F = {eps_F:.2e} for Fr={Fr}, VB={vb}"


def test_wear_coefficient_positivity():
    """
    Verifies that the wear coefficient kw is strictly positive: kw > 0
    across arbitrary spatial coordinates in both scalar and spatial parameterization modes.
    """
    # 1. Scalar mode
    cfg_scalar = {
        'model': {
            'parameterized': True,
            'k_w': {'mode': 'scalar', 'use_rff': True, 'rff_scale': 1.0, 'rff_features': 16, 'layers': [32], 'activation': 'tanh'},
            'h': {'use_rff': False, 'layers': [32], 'activation': 'tanh'},
            'phi': {'use_rff': False, 'layers': [32], 'activation': 'tanh'}
        },
        'domain': {'L': 1.0, 'D': 0.5, 'T': 1.0},
        'physics': {'k_w_init': 2e-7, 'alpha_clearance_deg': 11.0}
    }
    model_scalar = MicroWearPINN(cfg_scalar)
    x_test = torch.linspace(-1.0, 1.0, 100)[:, None]
    kw_scalar = model_scalar.predict_k_w(x_test).detach().numpy()
    assert np.all(kw_scalar > 0.0), f"Scalar kw contains non-positive values: min={np.min(kw_scalar)}"
    
    # 2. Spatial mode with softplus
    cfg_spatial = {
        'model': {
            'parameterized': True,
            'k_w': {'mode': 'spatial', 'use_rff': True, 'rff_scale': 1.0, 'rff_features': 16, 'layers': [32, 32], 'activation': 'tanh'},
            'h': {'use_rff': False, 'layers': [32], 'activation': 'tanh'},
            'phi': {'use_rff': False, 'layers': [32], 'activation': 'tanh'}
        },
        'domain': {'L': 1.0, 'D': 0.5, 'T': 1.0},
        'physics': {'k_w_init': 2e-7, 'alpha_clearance_deg': 11.0}
    }
    model_spatial = MicroWearPINN(cfg_spatial)
    kw_spatial = model_spatial.predict_k_w(x_test).detach().numpy()
    assert np.all(kw_spatial > 0.0), f"Spatial kw contains non-positive values: min={np.min(kw_spatial)}"


def test_clearance_angle_conversion():
    """
    Verifies bidirectional mathematical consistency between flank wear land VB
    and normal tool wear depth h: VB = -h / tan(alpha_0) and h = -VB * tan(alpha_0).
    """
    cfg = {
        'model': {
            'parameterized': True,
            'k_w': {'mode': 'scalar'},
            'h': {'use_rff': False, 'layers': [32], 'activation': 'tanh'},
            'phi': {'use_rff': False, 'layers': [32], 'activation': 'tanh'}
        },
        'physics': {'alpha_clearance_deg': 11.0}
    }
    model = MicroWearPINN(cfg)
    vb_orig = torch.tensor([[0.0], [0.15], [0.30], [0.55]], dtype=torch.float32)
    h_depth = model.vb_to_wear_depth(vb_orig)
    vb_recovered = model.wear_depth_to_vb(-h_depth)
    
    np.testing.assert_allclose(vb_orig.numpy(), vb_recovered.numpy(), rtol=1e-6, atol=1e-6)


def test_nasa_milling_loader_replicate_structure():
    """
    Verifies that the NASA loader correctly parses 70 mm cutter geometry,
    nominal 200 m/min speed across cases, multi-sensor signals, and paired tool relationships.
    """
    loader = NASAMillingLoader()
    c1 = loader.load_case_data(1)
    c9 = loader.load_case_data(9)
    
    assert c1['D_cutter'] == 70.0, f"Expected cutter diameter 70 mm, got {c1['D_cutter']}"
    assert c9['D_cutter'] == 70.0, f"Expected cutter diameter 70 mm, got {c9['D_cutter']}"
    assert c1['v_m_min'] == 200.0, f"Expected 200 m/min, got {c1['v_m_min']}"
    assert c9['v_m_min'] == 200.0, f"Expected 200 m/min, got {c9['v_m_min']}"
    
    # Verify multi-sensor channels
    for chan in ['ac_rms', 'dc_mean', 'vib_tbl_rms', 'vib_spn_rms', 'ae_tbl_rms', 'ae_spn_rms']:
        assert chan in c1['features'], f"Missing sensor feature {chan} in Case 1"
        assert len(c1['features'][chan]) == len(c1['all_times']), f"Feature {chan} length mismatch"
        
    # Verify replicate conditions
    assert c1['DOC'] == c9['DOC'] == 1.5, "DOC mismatch between paired replicate cases 1 and 9"
    assert c1['feed'] == c9['feed'] == 0.5, "Feed mismatch between paired replicate cases 1 and 9"
    assert c1['material'] == c9['material'] == 'Cast Iron', "Material mismatch between paired replicate cases 1 and 9"


if __name__ == "__main__":
    print("Running test_contact_force_conservation...")
    test_contact_force_conservation()
    print("PASSED: test_contact_force_conservation")
    
    print("Running test_wear_coefficient_positivity...")
    test_wear_coefficient_positivity()
    print("PASSED: test_wear_coefficient_positivity")
    
    print("Running test_clearance_angle_conversion...")
    test_clearance_angle_conversion()
    print("PASSED: test_clearance_angle_conversion")
    
    print("Running test_nasa_milling_loader_replicate_structure...")
    test_nasa_milling_loader_replicate_structure()
    print("PASSED: test_nasa_milling_loader_replicate_structure")
    
    print("\nAll pipeline verification tests PASSED successfully!")

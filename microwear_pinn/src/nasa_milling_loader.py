import os
import zipfile
import scipy.io
import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List, Optional

class NASAMillingLoader:
    """
    Data processor for the NASA Ames Prognostics Milling Tool Wear Dataset.
    Matsuura MC-510V CNC Machining Center with a 70 mm diameter face milling cutter
    fitted with 6 KC710 carbide inserts.
    
    Extracts multi-sensor time-series (spindle AC/DC current, table/spindle vibration,
    table/spindle acoustic emission), cutting conditions, and flank wear (VB).
    Calibrates Archard wear kinetics and force-conservative Airy contact stress fields.
    """
    
    # Paired replicate test cases under identical conditions: Tool 1 (Cases 1-8) <-> Tool 2 (Cases 9-16)
    PAIRED_CASES = {
        1: 9,   # Cast Iron, DOC 1.50 mm, feed 0.50 mm/rev
        2: 12,  # Cast Iron, DOC 0.75 mm, feed 0.50 mm/rev
        3: 11,  # Cast Iron, DOC 0.75 mm, feed 0.25 mm/rev
        4: 10,  # Cast Iron, DOC 1.50 mm, feed 0.25 mm/rev
        5: 16,  # Steel,     DOC 1.50 mm, feed 0.50 mm/rev
        7: 13,  # Steel,     DOC 0.75 mm, feed 0.25 mm/rev
        8: 14,  # Steel,     DOC 0.75 mm, feed 0.50 mm/rev
    }
    
    def __init__(self, mat_path: Optional[str] = None):
        self.mat_path = self._resolve_dataset_path(mat_path)
        self.raw_data = scipy.io.loadmat(self.mat_path)
        self.mill = self.raw_data['mill']
        self.num_runs = self.mill.shape[1]
        
    def _resolve_dataset_path(self, user_path: Optional[str]) -> str:
        """Dynamically locates mill.mat or automatically unpacks mill.zip if necessary."""
        candidates = []
        if user_path:
            candidates.append(user_path)
        if 'NASA_MILL_PATH' in os.environ:
            candidates.append(os.environ['NASA_MILL_PATH'])
            
        candidates.extend([
            os.path.join(os.getcwd(), 'dataset', 'mill.mat'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'mill.mat'),
            os.path.join(os.getcwd(), 'mill.mat'),
            r"e:\Downloads\MicroWear-PINN\dataset\mill.mat"
        ])
        
        for p in candidates:
            if p and os.path.exists(p):
                return os.path.abspath(p)
                
        # If mill.mat not found, try to unpack mill.zip
        zip_candidates = [
            os.path.join(os.getcwd(), 'mill.zip'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'mill.zip')
        ]
        for zp in zip_candidates:
            if os.path.exists(zp):
                extract_dir = os.path.join(os.path.dirname(zp), 'dataset')
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zp, 'r') as zf:
                    zf.extractall(extract_dir)
                target_mat = os.path.join(extract_dir, 'mill.mat')
                if os.path.exists(target_mat):
                    return os.path.abspath(target_mat)
                    
        raise FileNotFoundError(
            f"Could not locate NASA Milling dataset ('mill.mat'). Checked paths:\n" + 
            "\n".join([str(c) for c in candidates])
        )
        
    def load_case_data(self, case_id: int, w_c: float = 1.0, a0: float = 0.05) -> dict:
        """
        Extracts and pre-processes operational parameters, multi-sensor signals, 
        calibrated cutting forces, and flank wear observations for a specific NASA Case.
        
        Args:
            case_id: The NASA case ID (1 to 16).
            w_c: Tool contact width in mm.
            a0: Tool edge rounding radius / half-contact offset in mm (default 0.05 mm = 50 um).
            
        Returns:
            Dictionary containing experimental time-series, multi-sensor statistics,
            and continuous interpolation functions.
        """
        # Filter runs for this case
        runs = []
        for i in range(self.num_runs):
            run_struct = self.mill[0, i]
            if int(run_struct['case'][0, 0]) == case_id:
                runs.append(run_struct)
                
        if len(runs) == 0:
            raise ValueError(f"No runs found for Case ID {case_id}")
            
        times = []       # Machining time in seconds
        ac_rms = []      # Spindle AC current RMS (Amps)
        dc_mean = []     # Spindle DC current mean (Amps)
        vib_tbl_rms = [] # Table vibration RMS (g)
        vib_spn_rms = [] # Spindle vibration RMS (g)
        ae_tbl_rms = []  # Table AE RMS (V)
        ae_spn_rms = []  # Spindle AE RMS (V)
        
        docs = []        # Depth of Cut (mm)
        feeds = []       # Feed rate (mm/rev)
        materials = []   # Material ID (1=Cast Iron, 2=Steel)
        vbs = []         # Measured flank wear VB (mm)
        vb_times = []    # Timestamps where VB is valid (not nan)
        
        # Matsuura MC-510V: 70 mm face milling cutter with 6 KC710 inserts
        # Spindle speed N = 826 rpm. Nominal cutting speed vc = 200 m/min across ALL 16 cases
        # vc = (200 m/min * 1000 mm/m) / 60 s/min = 3333.33 mm/s
        v_m_min = 200.0
        v_rel_val = (v_m_min * 1000.0) / 60.0
        D_cutter = 70.0  # mm
        N_spindle = 826.0 # rpm
        
        for run in runs:
            t_sec = float(run['time'][0, 0]) * 60.0
            times.append(t_sec)
            
            # Multi-sensor signal feature extraction
            smcAC = run['smcAC'].squeeze()
            smcDC = run['smcDC'].squeeze()
            vib_tbl = run['vib_table'].squeeze()
            vib_spn = run['vib_spindle'].squeeze()
            ae_tbl = run['AE_table'].squeeze()
            ae_spn = run['AE_spindle'].squeeze()
            
            ac_rms.append(float(np.sqrt(np.mean(smcAC ** 2))))
            dc_mean.append(float(np.mean(np.abs(smcDC))))
            vib_tbl_rms.append(float(np.sqrt(np.mean(vib_tbl ** 2))))
            vib_spn_rms.append(float(np.sqrt(np.mean(vib_spn ** 2))))
            ae_tbl_rms.append(float(np.sqrt(np.mean(ae_tbl ** 2))))
            ae_spn_rms.append(float(np.sqrt(np.mean(ae_spn ** 2))))
            
            docs.append(float(run['DOC'][0, 0]))
            feeds.append(float(run['feed'][0, 0]))
            materials.append(int(run['material'][0, 0]))
            
            vb = run['VB']
            if vb.shape[1] > 0 and not np.isnan(vb[0, 0]):
                vbs.append(float(vb[0, 0]))
                vb_times.append(t_sec)
                
        times = np.array(times)
        ac_rms = np.array(ac_rms)
        dc_mean = np.array(dc_mean)
        vib_tbl_rms = np.array(vib_tbl_rms)
        vib_spn_rms = np.array(vib_spn_rms)
        ae_tbl_rms = np.array(ae_tbl_rms)
        ae_spn_rms = np.array(ae_spn_rms)
        
        docs = np.array(docs)
        feeds = np.array(feeds)
        materials = np.array(materials)
        vbs = np.array(vbs)
        vb_times = np.array(vb_times)
        
        mat_name = 'Cast Iron' if materials[0] == 1 else 'Steel'
        # Specific cutting force Ks0: ~1300 N/mm2 for Cast Iron, ~2100 N/mm2 for J45 Steel
        Ks0 = 1300.0 if materials[0] == 1 else 2100.0
        
        # Calibrated cutting force model: Fr = Ks0 * DOC * feed * (I_AC(t) / I_AC_base)
        # Normalizes current dynamics to physical force units (Newtons)
        ac_base = max(ac_rms[0], 0.1)
        current_factor = ac_rms / ac_base
        forces = Ks0 * docs * feeds * current_factor # Resultant force Fr in Newtons
        
        # Initial boundary condition at t=0
        if 0.0 not in vb_times:
            vb_times = np.insert(vb_times, 0, 0.0)
            vbs = np.insert(vbs, 0, 0.0)
            
        t_max = float(np.max(times))
        
        # Continuous interpolation functions
        interp_vb = interp1d(vb_times, vbs, kind='linear', bounds_error=False, fill_value=(0.0, vbs[-1]))
        interp_force = interp1d(times, forces, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        return {
            'case_id': case_id,
            'material': mat_name,
            'material_id': int(materials[0]),
            'DOC': float(docs[0]),
            'feed': float(feeds[0]),
            'v_rel': v_rel_val,       # mm/s
            'v_m_min': v_m_min,       # m/min
            'D_cutter': D_cutter,     # mm
            'N_spindle': N_spindle,   # rpm
            'T_max': t_max,           # seconds
            'w_c': w_c,               # tool contact width (mm)
            'a0': a0,                 # initial contact half-width (mm)
            'Ks0': Ks0,               # base specific cutting force (N/mm2)
            # Continuous interpolation
            'interp_vb': interp_vb,
            'interp_force': interp_force,
            # Discrete experimental observations
            't_data': vb_times,
            'vb_data': vbs,
            'all_times': times,
            'forces': forces,
            # Multi-sensor feature matrix
            'features': {
                'ac_rms': ac_rms,
                'dc_mean': dc_mean,
                'vib_tbl_rms': vib_tbl_rms,
                'vib_spn_rms': vib_spn_rms,
                'ae_tbl_rms': ae_tbl_rms,
                'ae_spn_rms': ae_spn_rms
            }
        }
        
    def sample_operational_pressure(self, case_data: dict, t: np.ndarray, 
                                    vb_override: Optional[np.ndarray] = None,
                                    leak_free: bool = False) -> np.ndarray:
        """
        Evaluates interface contact pressure p(t) across the contact strip.
        
        Exact Force Conservation Formulation:
          Hertz half-width: a(t) = VB(t) / 2 + a0  => 2a(t) = VB(t) + 2*a0
          Hertz contact strip area: A(t) = 2a(t) * w_c = w_c * (VB(t) + 2*a0)
          Average pressure: p_avg(t) = Fr(t) / A(t)
          Peak Hertz pressure: p_peak(t) = (4 / pi) * p_avg(t)
          
        Args:
            case_data: Dictionary returned by load_case_data.
            t: Query timestamps (seconds).
            vb_override: Predicted/estimated wear array for leak-free forward evaluation.
            leak_free: If True and vb_override is None, uses sharp edge contact (VB=0)
                       to prevent ground-truth VB leakage into prediction features.
                       
        Returns:
            Average contact pressure p_avg(t) in MPa (N/mm^2).
        """
        forces = case_data['interp_force'](t)
        w_c = case_data['w_c']
        a0 = case_data['a0']
        
        if vb_override is not None:
            vbs = vb_override
        elif leak_free:
            # Leak-free evaluation: do not use ground-truth VB to form input features
            vbs = np.zeros_like(forces)
        else:
            # Inverse calibration mode: uses observed interpolated VB
            vbs = case_data['interp_vb'](t)
            
        # Mathematically consistent full Hertz width: 2a = VB + 2*a0
        # When integrated across [-a, a] with p_peak = (4/pi)*p_avg, \int_A p dA = Fr identically
        area = w_c * (vbs + 2.0 * a0)
        return forces / area


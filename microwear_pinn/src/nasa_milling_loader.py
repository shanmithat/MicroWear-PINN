import os
import scipy.io
import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List, Optional

class NASAMillingLoader:
    """
    Data processor for the NASA Prognostics Milling Tool Wear Dataset.
    Loads and processes MATLAB files to extract spindle currents, cutting parameters, 
    and flank wear (VB) measurements to calibrate Archard's wear kinetics.
    """
    def __init__(self, mat_path: str = r"e:\Downloads\MicroWear-PINN\dataset\mill.mat"):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"NASA Milling mat file not found at: {mat_path}")
        
        self.mat_path = mat_path
        self.raw_data = scipy.io.loadmat(mat_path)
        self.mill = self.raw_data['mill']
        self.num_runs = self.mill.shape[1]
        
    def load_case_data(self, case_id: int, w_c: float = 1.0, vb0: float = 0.05) -> dict:
        """
        Extracts and pre-processes operational parameters, forces, pressures,
        and tool wear signals for a specific milling Case.
        
        Args:
            case_id: The NASA case ID (1 to 16).
            w_c: Tool contact width in mm.
            vb0: Flank wear tool-tip rounding offset in mm (prevents pressure singularity).
            
        Returns:
            Dictionary containing interpolated time-series functions for continuous sampling
            and discrete arrays of experimental observations.
        """
        # Filter runs for this case
        runs = []
        for i in range(self.num_runs):
            run_struct = self.mill[0, i]
            if int(run_struct['case'][0, 0]) == case_id:
                runs.append(run_struct)
                
        if len(runs) == 0:
            raise ValueError(f"No runs found for Case ID {case_id}")
            
        # Parse cut signals and operational values
        times = []       # cumulative machining time in seconds
        ac_rms = []      # AC spindle current RMS (Amps)
        docs = []        # Depth of Cut (mm)
        feeds = []       # Feed rate (mm/rev)
        materials = []   # Material ID (1=Cast Iron, 2=Steel)
        vbs = []         # Measured flank wear VB (mm)
        vb_times = []    # Timestamps where VB is valid (not nan)
        
        # Determine velocity (speed) based on literature for the Berkeley face milling setup
        # Cases 1-8: 200 m/min. Cases 9-16: 250 m/min.
        # Cutter diameter is 19.05 mm (0.75 in)
        v_m_min = 200.0 if case_id <= 8 else 250.0
        v_rel_val = (v_m_min * 1000.0) / 60.0 # convert to mm/s (e.g. 3333.3 mm/s or 4166.7 mm/s)
        
        for run in runs:
            # Time is in minutes, convert to seconds
            t_sec = float(run['time'][0, 0]) * 60.0
            times.append(t_sec)
            
            # AC current RMS
            smcAC = run['smcAC']
            rms_val = float(np.sqrt(np.mean(smcAC ** 2)))
            ac_rms.append(rms_val)
            
            # Constants per run
            docs.append(float(run['DOC'][0, 0]))
            feeds.append(float(run['feed'][0, 0]))
            materials.append(int(run['material'][0, 0]))
            
            # Flank wear VB
            vb = run['VB']
            if vb.shape[1] > 0 and not np.isnan(vb[0, 0]):
                vbs.append(float(vb[0, 0]))
                vb_times.append(t_sec)
                
        times = np.array(times)
        ac_rms = np.array(ac_rms)
        docs = np.array(docs)
        feeds = np.array(feeds)
        materials = np.array(materials)
        vbs = np.array(vbs)
        vb_times = np.array(vb_times)
        
        # 1. Estimate cutting force: Fr = 1000.0 * DOC * feed * ac_rms
        # Scaled specific cutting force (approx 1000 to 2000 N depending on material)
        forces = 1000.0 * docs * feeds * ac_rms # in Newtons
        
        # 2. Build continuous interpolation functions over [0, T]
        # First timestamp is t=0, flank wear is 0
        if 0.0 not in vb_times:
            vb_times = np.insert(vb_times, 0, 0.0)
            vbs = np.insert(vbs, 0, 0.0)
            
        t_max = float(np.max(times))
        
        # Interpolate wear VB(t) continuously
        interp_vb = interp1d(vb_times, vbs, kind='linear', bounds_error=False, fill_value=(0.0, vbs[-1]))
        
        # Interpolate resultant force Fr(t) continuously
        interp_force = interp1d(times, forces, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Return processed data package
        return {
            'case_id': case_id,
            'material': 'Cast Iron' if materials[0] == 1 else 'Steel',
            'DOC': float(docs[0]),
            'feed': float(feeds[0]),
            'v_rel': v_rel_val, # mm/s
            'T_max': t_max,      # seconds
            # Continuous interpolation functions
            'interp_vb': interp_vb,
            'interp_force': interp_force,
            # Discrete experimental data points for loss optimization
            't_data': vb_times,
            'vb_data': vbs,
            'w_c': w_c,
            'vb0': vb0
        }
        
    def sample_operational_pressure(self, case_data: dict, t: np.ndarray) -> np.ndarray:
        """
        Evaluates the interface pressure p(t) at any continuous coordinates t.
        Formula:
          p(t) = Fr(t) / (w_c * (VB(t) + vb0))
        """
        # Interpolate VB and Fr at query times t
        vbs = case_data['interp_vb'](t)
        forces = case_data['interp_force'](t)
        
        # Contact area increases with flank wear
        area = case_data['w_c'] * (vbs + case_data['vb0'])
        
        # Return average contact pressure (in MPa)
        return forces / area

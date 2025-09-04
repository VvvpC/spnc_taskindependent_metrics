"""
Softgate utility functions for reservoir parameter evaluation.

This module provides fast softgate evaluation using triangular wave input
to quickly assess reservoir parameter viability before expensive CQ/MC computation.

Created on 2025-09-04
@author: Chen
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required dependencies
from spnc import spnc_anisotropy
from formal_Parameter_Dynamics_Preformance import RunSpnc
from deterministic_mask import fixed_seed_mask


def generate_triangular_wave(T=1000, cycles=4):
    """
    Generate triangular wave signal for softgate evaluation.
    
    Parameters:
    T (int): Total number of time points (default: 1000)
    cycles (int): Number of triangular wave cycles (default: 4)
    
    Returns:
    numpy.ndarray: Triangular wave signal
    """
    t = np.linspace(0, 1, T)
    tri = 2 * np.abs(2 * (t * cycles % 1) - 1) - 1
    return tri


def filter(u, y, period):
    """
    Filter function to evaluate reservoir output quality based on multiple metrics.
    
    Parameters:
    u (array): Input signal
    y (array): Output signal  
    period (float): Signal period for lag calculation
    
    Returns:
    tuple: (pass_flag, score, info_dict)
        - pass_flag (bool): Whether the signal passes quality thresholds
        - score (float): Weighted quality score (0-1)
        - info_dict (dict): Detailed metrics (Sat, Drift, LagRatio, THD)
    """
    y = np.ravel(np.asarray(y))
    u = np.ravel(np.asarray(u))
    assert len(y) == len(u)

    # Saturation
    ymin, ymax = y.min(), y.max()
    R = max(1e-12, ymax - ymin)
    Sat = np.mean((y <= ymin + 0.1*R) | (y >= ymax - 0.1*R))
    
    # Drift
    t = np.arange(len(y))
    Drift = 0.0 if y.std() == 0 else float(abs(np.corrcoef(t, y)[0,1]))
    
    # Lag
    y0 = y - y.mean()
    u0 = u - u.mean()
    L = np.argmax(np.correlate(y0, u0, mode='full')) - (len(y)-1)
    LagRatio = abs(L) / float(period)
    
    # THD (Total Harmonic Distortion - simplified version)
    Yf = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(len(y), d=1)
    f0 = 1.0/float(period)
    k1 = np.argmin(abs(freqs - f0))
    A1 = abs(Yf[k1])
    Ak = np.abs(Yf)
    Ak[[0, k1]] = 0
    THD = float(np.linalg.norm(Ak) / (A1 + 1e-12))

    # Hard threshold criteria
    hard_pass = (Drift <= 0.60) and (THD <= 0.80)
    
    # Weighted score components
    z_sat   = max(0.0, 1.0 - Sat)
    z_drift = max(0.0, 1.0 - Drift)
    z_lag   = max(0.0, 1.0 - abs(LagRatio - 0.2)/0.2)
    z_thd   = max(0.0, 1.0 - THD/0.8)

    # Combined weighted score
    score = 0.35*z_sat + 0.25*z_drift + 0.20*z_lag + 0.20*z_thd
    
    # Final pass/fail decision
    pass_flag = hard_pass and (score >= 0.55)

    info = {
        'Sat': Sat, 
        'Drift': Drift, 
        'LagRatio': LagRatio, 
        'THD': THD,
        'z_sat': z_sat, 
        'z_drift': z_drift, 
        'z_lag': z_lag, 
        'z_thd': z_thd
    }
    
    return pass_flag, score, info


def softgate(tri_signal, rparams):
    """
    Fast softgate evaluation using triangular wave input.
    
    This function performs a lightweight reservoir evaluation using a single
    virtual node to quickly assess parameter viability before expensive
    CQ/MC computation.
    
    Parameters:
    tri_signal (array): Triangular wave input signal
    rparams (ReservoirParams): Reservoir parameters object
    
    Returns:
    dict: Result dictionary containing:
        - 'pass_flag' (bool): Whether parameters pass quality threshold
        - 'score' (float): Quality score (0-1)
        - 'info' (dict): Detailed quality metrics
    """
    try:
        # Create SPNC object with reservoir parameters
        spn = spnc_anisotropy(0.4, 90, 0, 45, rparams.beta_prime, restart=True)
        
        # Generate reservoir response using single virtual node for speed
        transform = spn.gen_signal_slow_delayed_feedback
        S = RunSpnc(tri_signal.reshape(-1, 1), 1, 1, 1, rparams.m0, transform, 
                    rparams.params, fixed_mask=True, seed_mask=1234)
        
        # Evaluate output quality using filter function
        pass_flag, score, info = filter(tri_signal, S.ravel(), 250)
        
        return {
            'pass_flag': pass_flag,
            'score': score,
            'info': info
        }
        
    except Exception as e:
        # Return failed result if evaluation encounters error
        return {
            'pass_flag': False,
            'score': 0.0,
            'info': {'error': str(e)}
        }
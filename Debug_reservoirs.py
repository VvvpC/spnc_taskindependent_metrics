# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 2025

@author: Chen

Debug reservoirs for comparing computational results between different reservoir implementations.
Contains Standard_reservoir and Hetero_reservoir classes for comparative analysis.
"""

import numpy as np
from spnc import spnc_anisotropy
from formal_Parameter_Dynamics_Preformance import (
    generate_signal, linear_MC, gen_KR_GR_input, Evaluate_KR_GR, RunSpnc, ReservoirParams
)


class Standard_reservoir:
    """
    Standard reservoir implementation using gen_signal_slow_delayed_feedback transform.
    Based on test_one_reservoir function from formal_Parameter_Dynamics_Preformance.py
    """
    
    def __init__(self, reservoir_params):
        """
        Initialize standard reservoir with given parameters.
        
        Parameters:
        -----------
        reservoir_params : ReservoirParams
            Reservoir configuration parameters
        """
        self.reservoir_params = reservoir_params
        
    def evaluate_MC(self, signal_len=550, **kwargs):
        """
        Evaluate Memory Capacity using standard transform function.
        """
        signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
        
        spn = spnc_anisotropy(
            self.reservoir_params.h,
            self.reservoir_params.theta_H,
            self.reservoir_params.k_s_0,
            self.reservoir_params.phi,
            self.reservoir_params.beta_prime,
            restart=True
        )
        
        transform = spn.gen_signal_slow_delayed_feedback
        
        Output = RunSpnc(
            signal,
            1,                 
            len(signal),       
            self.reservoir_params.Nvirt,
            self.reservoir_params.m0,
            transform,
            self.reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
        
        MC = linear_MC(signal, Output, splits=[0.2, 0.6], delays=10)
        
        return {'MC': MC}
    
    def evaluate_KR_GR(self, Nreadouts=50, Nwash=7, **kwargs):
        """
        Evaluate KR and GR using standard transform function.
        """
        Nreadouts = self.reservoir_params.Nvirt
        
        inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=1234)
        outputs = []
        
        for input_row in inputs:
            input_row = input_row.reshape(-1, 1)
            spn = spnc_anisotropy(
                self.reservoir_params.h, 
                self.reservoir_params.theta_H,
                self.reservoir_params.k_s_0, 
                self.reservoir_params.phi,
                self.reservoir_params.beta_prime, 
                restart=True
            )
            transforms = spn.gen_signal_slow_delayed_feedback
            output = RunSpnc(
                input_row, 1, len(input_row), self.reservoir_params.Nvirt,
                self.reservoir_params.m0, transforms, self.reservoir_params.params,
                fixed_mask=True,
                seed_mask=1234
            )
            outputs.append(output)
            
        States = np.stack(outputs, axis=0)
        KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=0.1)
        
        return {'KR': KR, 'GR': GR}
    
    def evaluate_all(self, **kwargs):
        """
        Evaluate both MC and KR/GR metrics.
        """
        mc_result = self.evaluate_MC(**kwargs)
        krgr_result = self.evaluate_KR_GR(**kwargs)
        
        return {
            'MC': mc_result['MC'],
            'KR': krgr_result['KR'],
            'GR': krgr_result['GR']
        }


class Hetero_reservoir:
    """
    Heterogeneous reservoir implementation using heteroRes_sameinput transform.
    Based on fixed gamma calculations from Reservoirs_diameter_CQ_MC.py
    """
    
    def __init__(self, reservoir_params, ref_beta_prime=None):
        """
        Initialize heterogeneous reservoir with given parameters.
        
        Parameters:
        -----------
        reservoir_params : ReservoirParams
            Reservoir configuration parameters
        ref_beta_prime : float, optional
            Reference beta_prime for constant input rate calculation
        """
        self.reservoir_params = reservoir_params
        self.ref_beta_prime = ref_beta_prime or reservoir_params.beta_prime
        
    def evaluate_MC(self, signal_len=550, **kwargs):
        """
        Evaluate Memory Capacity using heteroRes_sameinput transform.
        """
        signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
        
        spn = spnc_anisotropy(
            self.reservoir_params.h,
            self.reservoir_params.theta_H,
            self.reservoir_params.k_s_0,
            self.reservoir_params.phi,
            self.reservoir_params.beta_prime,
            restart=True
        )
        
        def transform_with_constant_rate(K_s, params, *args, **kwargs):
            return spn.get_signal_slow_delayed_feedback_heteroRes_sameinput(
                K_s, params, self.ref_beta_prime, self.reservoir_params.h
            )
        
        Output = RunSpnc(
            signal,
            1,                 
            len(signal),       
            self.reservoir_params.Nvirt,
            self.reservoir_params.m0,
            transform_with_constant_rate,
            self.reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
        
        MC = linear_MC(signal, Output, splits=[0.2, 0.6], delays=10)
        
        return {'MC': MC}
    
    def evaluate_KR_GR(self, Nreadouts=50, Nwash=7, **kwargs):
        """
        Evaluate KR and GR using heteroRes_sameinput transform.
        """
        Nreadouts = self.reservoir_params.Nvirt
        
        inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=1234)
        outputs = []
        
        for input_row in inputs:
            input_row = input_row.reshape(-1, 1)
            
            spn = spnc_anisotropy(
                self.reservoir_params.h, 
                self.reservoir_params.theta_H,
                self.reservoir_params.k_s_0, 
                self.reservoir_params.phi,
                self.reservoir_params.beta_prime, 
                restart=True
            )
            
            def transform_with_constant_rate(K_s, params, *args, **kwargs):
                return spn.get_signal_slow_delayed_feedback_heteroRes_sameinput(
                    K_s, params, self.ref_beta_prime, self.reservoir_params.h
                )
            
            output = RunSpnc(
                input_row, 1, len(input_row), self.reservoir_params.Nvirt,
                self.reservoir_params.m0, transform_with_constant_rate, 
                self.reservoir_params.params,
                fixed_mask=True,
                seed_mask=1234
            )
            outputs.append(output)
        
        States = np.stack(outputs, axis=0)
        KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=0.1)
        
        return {'KR': KR, 'GR': GR}
    
    def evaluate_all(self, **kwargs):
        """
        Evaluate both MC and KR/GR metrics.
        """
        mc_result = self.evaluate_MC(**kwargs)
        krgr_result = self.evaluate_KR_GR(**kwargs)
        
        return {
            'MC': mc_result['MC'],
            'KR': krgr_result['KR'],
            'GR': krgr_result['GR']
        }


def print_reservoir_inputs(reservoir_instance, reservoir_name):
    """
    Print detailed input parameters used in MC and KRandGR calculations.
    
    Parameters:
    -----------
    reservoir_instance : Standard_reservoir or Hetero_reservoir
        The reservoir instance to debug
    reservoir_name : str
        Name of the reservoir for display purposes
    """
    print(f"\n{'='*60}")
    print(f"DEBUG: {reservoir_name} Input Parameters")
    print(f"{'='*60}")
    
    # Reservoir parameters
    params = reservoir_instance.reservoir_params
    print(f"\nReservoir Parameters:")
    print(f"  h: {params.h}")
    print(f"  theta_H: {params.theta_H}")
    print(f"  k_s_0: {params.k_s_0}")
    print(f"  phi: {params.phi}")
    print(f"  beta_prime: {params.beta_prime}")
    print(f"  m0: {params.m0}")
    print(f"  Nvirt: {params.Nvirt}")
    print(f"  params dict: {params.params}")
    
    # Additional parameters for hetero reservoir
    if hasattr(reservoir_instance, 'ref_beta_prime'):
        print(f"  ref_beta_prime: {reservoir_instance.ref_beta_prime}")
    
    # MC calculation inputs
    print(f"\nMC Calculation Inputs:")
    signal_len = 550
    seed = 1234
    print(f"  signal_len: {signal_len}")
    print(f"  signal generation seed: {seed}")
    print(f"  MC splits: [0.2, 0.6]")
    print(f"  MC delays: 10")
    
    # Generate and display first few signal values
    signal = generate_signal(signal_len, seed=seed)
    print(f"  signal shape: {signal.shape}")
    print(f"  signal first 10 values: {signal[:10].flatten()}")
    
    # KR/GR calculation inputs
    print(f"\nKR/GR Calculation Inputs:")
    Nreadouts = params.Nvirt
    Nwash = 7
    seed_krgr = 1234
    print(f"  Nreadouts: {Nreadouts}")
    print(f"  Nwash: {Nwash}")
    print(f"  KR/GR seed: {seed_krgr}")
    print(f"  threshold: 0.1")
    
    # Generate and display KR/GR input info
    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=seed_krgr)
    print(f"  KR/GR inputs shape: {inputs.shape}")
    print(f"  First input sequence (first 10 values): {inputs[0][:10]}")
    
    # Transform function information
    print(f"\nTransform Function:")
    if isinstance(reservoir_instance, Standard_reservoir):
        print(f"  Transform: gen_signal_slow_delayed_feedback")
        print(f"  RunSpnc parameters:")
        print(f"    fixed_mask: False (default)")
        print(f"    seed_mask: None (default)")
    else:  # Hetero_reservoir
        print(f"  Transform: get_signal_slow_delayed_feedback_heteroRes_sameinput")
        print(f"  Transform uses ref_beta_prime: {reservoir_instance.ref_beta_prime}")
        print(f"  Transform uses h: {params.h}")
        print(f"  RunSpnc parameters:")
        print(f"    fixed_mask: True")
        print(f"    seed_mask: 1234")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage and comparison
    reservoir_params = ReservoirParams(
        h=0.4055105807072985, 
        m0=0.004305768634622887, 
        Nvirt=125, 
        beta_prime=41.7965657362074, 
        params={
            'gamma': 0.06707779187420466, 
            'theta': 0.09581885346062773, 
            'Nvirt': 125
        }
    )
    
    # Create reservoir instances
    standard_res = Standard_reservoir(reservoir_params)
    hetero_res = Hetero_reservoir(reservoir_params, ref_beta_prime=41.7965657362074)
    
    # Print detailed input parameters for debugging
    print_reservoir_inputs(standard_res, "Standard Reservoir")
    print_reservoir_inputs(hetero_res, "Hetero Reservoir")
    
    # Run evaluations
    print("\n" + "="*60)
    print("COMPUTATION RESULTS")
    print("="*60)
    
    print("\nStandard Reservoir Results:")
    standard_results = standard_res.evaluate_all()
    print(standard_results)
    
    print("\nHetero Reservoir Results:")
    hetero_results = hetero_res.evaluate_all()
    print(hetero_results)
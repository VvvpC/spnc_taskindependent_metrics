#!/usr/bin/env python3
"""
TI46 Framework Testing Script
Converted from test_TI46_framework.ipynb
"""

import os
import torch 
import torch.nn as nn
from spnc import spnc_anisotropy
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import pickle
import spnc_ml as ml
import contextlib
import io
import itertools
import copy
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from pathlib import Path


CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
    Path(r"/Users/vvvp./Desktop"),
]
searchpaths = [p for p in CANDIDATES if p.exists()]

# Tuple of repos
repos = ('machine_learning_library',)

# Import repository tools and modules
from deterministic_mask import fixed_seed_mask, max_sequences_mask
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from sklearn.metrics import classification_report


class ReservoirParams:
    """
    Reservoir parameters configuration class
    """
    def __init__(self, **kwargs):
        # Reservoir parameters 
        self.h = 0.4
        self.theta_H = 90
        self.k_s_0 = 0
        self.phi = 45
        self.beta_prime = 20

        # Network parameters 
        self.Nvirt = 50
        self.m0 = 0.13
        self.bias = True
        self.Nwarmup = 0
        self.verbose_repr = False

        self.params = {
            'theta': 0.046,
            'gamma': 0.13738441393289658,
            'delay_feedback': 0,
            'Nvirt': self.Nvirt,
            'length_warmup': self.Nwarmup,
            'warmup_sample': self.Nwarmup * self.Nvirt,
            'voltage_noise': False,
            'seed_voltage_noise': 1234,
            'delta_V': 0.1,
            'johnson_noise': False,
            'seed_johnson_noise': 1234,
            'mean_johnson_noise': 0.0000,
            'std_johnson_noise': 0.00001,
            'thermal_noise': False,
            'seed_thermal_noise': 1234,
            'lambda_ou': 1.0,
            'sigma_ou': 0.1
        }

        # Update parameters from kwargs
        for key in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'Nvirt', 'm0', 'bias', 'Nwarmup']:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        if 'params' in kwargs and isinstance(kwargs['params'], dict):
            self.params.update(kwargs['params'])

    def update_params(self, **kwargs):
        """Update parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.params:
                self.params[key] = value
            if not hasattr(self, key) and key not in self.params:
                raise AttributeError(f"ReservoirParams has no attribute or param key '{key}'")
            
    def print_params(self, verbose=False):
        """Print parameter information"""
        if not verbose:
            print(f"ReservoirParams(h={self.h}, beta_prime={self.beta_prime}, Nvirt={self.Nvirt})")
        else:
            print(f"ReservoirParams detailed info:")
            print(f"  h = {self.h}")
            print(f"  theta_H = {self.theta_H}")
            print(f"  k_s_0 = {self.k_s_0}")
            print(f"  phi = {self.phi}")
            print(f"  beta_prime = {self.beta_prime}")
            print(f"  Nvirt = {self.Nvirt}")
            print(f"  m0 = {self.m0}")
            print(f"  bias = {self.bias}")
            print("  params dictionary:")
            for k, v in self.params.items():
                print(f"    {k}: {v}")


def run_parameter_sweep(params_configs, speakers, base_params, transform, nfft=2048, save_results=True, filename_prefix="parameter_sweep"):
    """
    Run parameter sweep for all parameter combinations
    
    Parameters:
    -----------
    params_configs : dict
        Parameter configuration dictionary, format like {'param_name': [values]}
    speakers : list
        List of speakers
    base_params : ReservoirParams
        Base parameter object
    transform : function
        Signal transformation function
    nfft : int, optional
        Number of FFT points, default 2048
    save_results : bool, optional
        Whether to save results, default True
    filename_prefix : str, optional
        Prefix for saved file, default "parameter_sweep"
    
    Returns:
    --------
    results : dict
        Dictionary containing all results
    """
    
    # Get all parameter combinations
    param_names = list(params_configs.keys())
    param_values = list(params_configs.values())
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total {len(param_combinations)} parameter combinations to test")
    print(f"Parameters: {param_names}")
    
    results = {
        'param_names': param_names,
        'param_values': param_values,
        'combinations': [],
        'accuracies': [],
        'configurations': []
    }
    
    # Iterate through all parameter combinations
    for i, combination in enumerate(tqdm(param_combinations, desc="Parameter sweep progress")):
        # Create new parameter object (by copying base_params)
        test_params = copy.deepcopy(base_params)
        
        # Update parameters for current combination
        param_dict = dict(zip(param_names, combination))
        test_params.update_params(**param_dict)
        
        try:
            # Execute spnc_spoken_digits function
            with contextlib.redirect_stdout(io.StringIO()):
                acc = ml.spnc_spoken_digits(
                    speakers, 
                    test_params.Nvirt, 
                    test_params.m0, 
                    test_params.bias, 
                    transform, 
                    test_params.params,
                    nfft=nfft, 
                    return_accuracy=True
                )
            
            # Save results
            results['combinations'].append(combination)
            results['accuracies'].append(acc)
            results['configurations'].append(param_dict)
            
        except Exception as e:
            print(f"Error (combination {i+1}): {e}")
            results['combinations'].append(combination)
            results['accuracies'].append(None)
            results['configurations'].append(param_dict)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {filename}")
    
    return results


def plot_simple_results(results):
    """
    Create simple matplotlib plots for parameter sweep results
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_parameter_sweep function
    """
    
    # Prepare data
    param_names = results['param_names']
    combinations = results['combinations']
    accuracies = results['accuracies']
    
    # Filter out None values
    valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
    valid_combinations = [combinations[i] for i in valid_indices]
    valid_accuracies = [accuracies[i] for i in valid_indices]
    
    if not valid_accuracies:
        print("No valid accuracy results")
        return
    
    # Create subplots
    n_params = len(param_names)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot relationship between each parameter and accuracy
    for i, param_name in enumerate(param_names):
        param_values = [combo[i] for combo in valid_combinations]
        
        if i < len(axes):
            scatter = axes[i].scatter(param_values, valid_accuracies, 
                                    c=valid_accuracies, cmap='viridis', 
                                    alpha=0.7, s=50)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(f'{param_name} vs Accuracy')
            axes[i].grid(True, alpha=0.3)
            
            # Add colorbar
            if i == 0:
                plt.colorbar(scatter, ax=axes[i], label='Accuracy')
    
    # Hide extra subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(valid_accuracies, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.max(valid_accuracies), color='red', linestyle='--', 
                label=f'Best accuracy: {np.max(valid_accuracies):.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def load_and_plot_results(filename):
    """
    Load pkl file and plot results
    
    Parameters:
    -----------
    filename : str
        Path to pkl file
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results['accuracies'])} experimental results")
    
    # Use matplotlib for plotting
    plot_simple_results(results)
    
    # Print statistics
    valid_accuracies = [acc for acc in results['accuracies'] if acc is not None]
    if valid_accuracies:
        print(f"\n=== Statistics Summary ===")
        print(f"Total experiments: {len(valid_accuracies)}")
        print(f"Best accuracy: {np.max(valid_accuracies):.4f}")
        print(f"Average accuracy: {np.mean(valid_accuracies):.4f}")
        print(f"Accuracy std: {np.std(valid_accuracies):.4f}")
        
        # Find best configuration
        best_idx = results['accuracies'].index(np.max(valid_accuracies))
        best_config = results['configurations'][best_idx]
        print(f"\nBest parameter configuration:")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"  Accuracy: {np.max(valid_accuracies):.4f}")


def main():
    """Main execution function"""
    print("Starting TI46 Framework Test")
    
    # Initialize parameters
    params = ReservoirParams()
    
    # Create SPNC transform
    spn = spnc_anisotropy(
        params.h,
        params.theta_H,
        params.k_s_0,
        params.phi,
        params.beta_prime,
        restart=True
    )
    
    transform = spn.gen_signal_slow_delayed_feedback
    
    # Define speakers
    speakers = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    # Test basic functionality
    print("\nTesting basic spoken digits recognition...")
    acc = ml.spnc_spoken_digits(speakers, params.Nvirt, params.m0, params.bias, transform, params.params, nfft=512, return_accuracy=True)
    print(f"Basic test accuracy: {acc:.4f}")
    
    # Define parameter configurations for sweep
    params_configs = {
        'gamma': np.linspace(0.01, 0.5, 10),
    }
    
    # Run parameter sweep
    print("\nRunning parameter sweep...")
    params.Nvirt = 50
    results = run_parameter_sweep(
        params_configs=params_configs,
        speakers=speakers,
        base_params=params,
        transform=transform,
        nfft=512,
        save_results=True,
        filename_prefix="TI46_parameter_sweep"
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_simple_results(results)


if __name__ == "__main__":
    # 绘制文件中的数据
    load_and_plot_results("TI46_parameter_sweep_20250708_154405.pkl")
    # main()
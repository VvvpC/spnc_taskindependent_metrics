# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15:24:04 2025

@author: Chen

This script contains a framework for evaluating the MC, KRandGR and computational performance of a superparamagnetic nanodot system (spn) with varying parameters.

"""

# import necessary libraries
import os
# import torch 
# import torch.nn as nn
from spnc import spnc_anisotropy
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import pickle
import spnc_ml as ml


from pathlib import Path

CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
    Path(r"/Users/vvvp/Desktop/machine_learning_library"),
]
searchpaths = [p for p in CANDIDATES if p.exists()]

#tuple of repos
repos = ('machine_learning_library',)

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

# General functions

# -----------------MC----------------------------

def generate_signal(I,washout = 50,seed=1234):
    '''
    Generate a i.i.d. signal sequence with a [-1,1] range
    '''
    if seed is not None:
        np.random.seed(seed)
        
    signal_sequence = np.random.uniform(-1,1,size = I)

    washed_signal = signal_sequence[washout:]

    # Convert to 2D array
    washed_signal = washed_signal.reshape(-1,1)

    # print(np.shape(washed_signal))

    return washed_signal

def RidgeRegression(states, target, l, bias=True):
    # Ensure numpy
    if torch.is_tensor(states):
        states = states.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if bias==True:
        # Add bias to states
        bias = np.ones((len(states), 1))
        states = np.concatenate((bias, states), axis=1)
    # Setup matrices from inputs
    M1 = np.matmul(states.transpose(), target) 
    M2 = np.matmul(states.transpose(), states)
    # Perform ridge regression
    weights = np.matmul(np.linalg.pinv(M2+l*np.identity(len(M2))), M1)
    return weights

def linear_MC(signal, states, splits=[0.2, 0.8], delays=50):
    # ensure flat input signal
    signal = np.asarray(signal).flatten()
    # generate target signal from delayed input signal
    shift = np.zeros((len(signal), delays))
    for i in range(len(signal)-delays):
        i += delays
        shift[i, :] = signal[i-delays:i]
    # split data
    wash, Ytrain, Ytest = np.split(shift, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    wash, Xtrain, Xtest = np.split(states, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    # sweep over range of hyperparameters gamma to find optimal MC
    bestMC = 0
    gammas = np.logspace(-10, 0, 11)
    for gamma in gammas:
        # Calculate weights
        weights = RidgeRegression(Xtrain, Ytrain, gamma, bias=False)
        # Predict test states
        prediction = np.matmul(Xtest, weights)
        # Loop over all delays k and evaluate MC_k
        MC_k = np.zeros(delays)
        for k in range(delays):
            # Take prediction and target for each delay
            pred = prediction[:, k]
            targ = Ytest[:, k]
            # Set up matrix to calculate covariance
            M = pred, targ
            # Calculate covariance
            coVarM = np.cov(M)
            # Take cov(xy) 
            coVar = coVarM[0,1]
            # Measure the variance of the signals
            outVar = np.var(pred)
            targVar = np.var(targ)
            # Calculate the total variance of the raw target and the specific
            # target
            totVar = outVar*targVar
            # If the covariance coefficient is greater than 0.1, treat as better
            # than random guessing and add to MC_k outputs
            if coVar**2/totVar > 0.1:
                MC_k[k] = coVar**2/totVar
        # Account for floating point errors in MC
        MC_k[MC_k>1] = 1
        # Sum memory capacity over all delays
        MC = sum(MC_k)
        # If best reported MC, save data
        if MC > bestMC:
            bestMC = MC
    return bestMC

# ------------------------ KRandGR ----------------------------
'''
23/06/25 Chen


Generate input with more equal figures for KRandGR, last 7  columns are GR inputs, the rest are KR inputs.

Here Nwash = 7 for KR, rest of 7 columns are GR

'''
def gen_KR_GR_input(Nreadouts, Nwash=7, seed=1234):
    rng = np.random.default_rng(seed) 
    KR_inputs = rng.random((Nreadouts, Nwash))
    GR_inputs = np.tile(rng.random(7), (Nreadouts, 1)) 
    all_inputs = np.concatenate((KR_inputs, GR_inputs), axis=1)
    return all_inputs

def Evaluate_KR_GR(states, Nreadouts, threshold=0.1):
    GR_states = states[:,-1,:]
    '''
    Change the last 7 columns to GR states, the rest are KR states
    '''
    KR_states = states[:,-8,:]
    uGR, sGR, vGR = np.linalg.svd(GR_states)
    uKR, sKR, vKR = np.linalg.svd(KR_states)
    KR = 0
    GR = 0
    for i in range(Nreadouts):
        if sGR[i]>threshold:
            GR += 1
        if sKR[i]>threshold:
            KR += 1
    return KR, GR

# ------------------------ Reservoir ----------------------------
def RunSpnc(signal,Nin,Nout,Nvirt,m0,transform, params,**kwargs):
    '''
    Run a reservoir computer with the signal sequence
    '''
    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask==True:
        # print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            # print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            # print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    
    # Run
    S,_ = snr.transform(signal,params)
    
    return S

# ------------------------ Reservoir Parameters Dictionary --------------------------

class  ReservoirParams:
    def __init__(self, **kwargs):
            # Reservoir parameters 
            self.h = 0.4473502275692851
            self.theta_H = 90
            self.k_s_0 = 0
            self.phi = 45
            self.beta_prime = 20

            # Network parameters 
            self.Nvirt = 30
            self.m0 = 0.007586422893538462
            self.bias = True
            self.Nwarmup = 0
            self.verbose_repr = False

            self.params = {
                'theta': 0.5540233436467944,
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

            for key in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'Nvirt', 'm0', 'bias', 'Nwarmup']:
                if key in kwargs:
                    setattr(self, key, kwargs[key])

            
            if 'params' in kwargs and isinstance(kwargs['params'], dict):
                self.params.update(kwargs['params'])

    
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.params:
                self.params[key] = value
            if not hasattr(self, key) and key not in self.params:
                raise AttributeError(f"ReservoirParams has no attribute or param key '{key}'")
            
    def print_params(self, verbose=False):
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

# ##########
# A General Framwork for evaluating the MC, KRandGR and computational performance of a superparamagnetic nanodot system (spn) with varying parameters.
# ##########

class ReservoirPerformanceEvaluator:
    def __init__(self, task, param_name, param_range, result_keys, result_labels, reservoir_params = None, extra_args = None):
        self.task = task
        self.param_name = param_name
        self.param_range = param_range
        self.result_keys = result_keys
        self.result_labels = result_labels
        self.reservoir_params = reservoir_params 
        self.extra_args = extra_args or {}

    def evaluate(self, save_dir='./Results', plot = False, verbose = False, filename_prefix = None):

        result_dict = {self.param_name: self.param_range}
        for key in self.result_keys:
            result_dict[key] = []

        for v in tqdm.tqdm(self.param_range):
            self.reservoir_params.update_params(**{self.param_name: v})
            if verbose:
                self.reservoir_params.print_params(verbose=True)

            try:
                task_result = self.task(self.reservoir_params, **self.extra_args)
                for key in self.result_keys:
                    result_dict[key].append(task_result[key])

            except Exception as e:
                                print(f"Error evaluating {self.param_name}={v}: {e}")
                                for key in self.result_keys:
                                    result_dict[key].append(np.nan)

        # Save results
        if filename_prefix is None:
            filename_prefix = f"{self.task.__name__}_{self.param_name}"
        filename = f"{filename_prefix}_evaluate_{self.param_name}_{self.param_range[0]}to{self.param_range[-1]}_step{self.param_range[1]-self.param_range[0]}.pkl"         
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Results saved to {save_path}")

        # Plot results 
        fig, ax = plt.subplots(figsize=(8,5))
        if len(self.result_keys) == 1:
            ax.plot(self.param_range, result_dict[self.result_keys[0]], marker='o')
            ax.set_xlabel(self.param_name)
            ax.set_ylabel(self.result_labels[0])
            ax.set_title(f"{self.result_labels[0]} vs {self.param_name}")
            
        elif len(self.result_keys) == 2:
            color1, color2 = 'tab:blue', 'tab:orange'
            ax1 = ax
            ax1.set_xlabel(self.param_name)
            ax1.set_ylabel(self.result_labels[0], color=color1)
            ax1.plot(self.param_range, result_dict[self.result_keys[0]], marker='o', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax2 = ax1.twinx()
            ax2.set_ylabel(self.result_labels[1], color=color2)
            ax2.plot(self.param_range, result_dict[self.result_keys[1]], marker='x', linestyle='--', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            fig.suptitle(f"{self.result_labels[0]} & {self.result_labels[1]} vs {self.param_name}")
            
        fig.tight_layout()

        if filename_prefix is None:
            filename_prefix = f"{self.task.__name__}_{self.param_name}"
        plot_filename = f"{filename_prefix}_evaluate_{self.param_name}_{self.param_range[0]}to{self.param_range[-1]}_step{self.param_range[1]-self.param_range[0]}.png"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, plot_filename))
        print(f"Figure saved to {os.path.join(save_dir, plot_filename)}")


        if plot:
            plt.show()

        plt.close(fig)


        return result_dict
    

# ##########
# MC task function
# ##########
def evaluate_MC(reservoir_params, signal_len = 550, **kwargs):

    signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
    # 打印signal的前10个元素


    spn = spnc_anisotropy(
        reservoir_params.h,
        reservoir_params.theta_H,
        reservoir_params.k_s_0,
        reservoir_params.phi,
        reservoir_params.beta_prime,
        restart=True
    )

    transform = spn.gen_signal_slow_delayed_feedback

    Output = RunSpnc(
        signal,
        1,                 
        len(signal),       
        reservoir_params.Nvirt,
        reservoir_params.m0,
        transform,
        reservoir_params.params,
    )


    MC = linear_MC(signal, Output, splits=[0.2,0.6], delays=10)

    return {'MC': MC}

# ##########
# KRandGR task function

# ##########


def evaluate_KRandGR(reservoir_params, Nreadouts=50, Nwash=7, **kwargs):
    
    Nreadouts= reservoir_params.Nvirt

    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=1234)   # <--- 用Nreadouts
    outputs = []
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                              reservoir_params.k_s_0, reservoir_params.phi,
                              reservoir_params.beta_prime, restart=True)
        transforms = spn.gen_signal_slow_delayed_feedback
        output = RunSpnc(input_row, 1, len(input_row), reservoir_params.Nvirt,
                         reservoir_params.m0, transforms, reservoir_params.params)
        outputs.append(output)
    States = np.stack(outputs, axis=0)
    KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=0.1)  # <--- 用Nreadouts
    return {'KR': KR, 'GR': GR}


# ##########
# NARMA10 task function
# ##########

def evaluate_NARMA10(reservoir_params, Ntrain=2000, Ntest=1000, **kwargs):
    spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                          reservoir_params.k_s_0, reservoir_params.phi,
                          reservoir_params.beta_prime, restart=True)
    transform = spn.gen_signal_slow_delayed_feedback
    NRMSE = ml.spnc_narma10(Ntrain, Ntest, reservoir_params.Nvirt,
                            reservoir_params.m0, reservoir_params.bias,
                            transform, reservoir_params.params,
                            seed_NARMA=1234, fixed_mask=True, return_NRMSE=True)
    return {'NRMSE': NRMSE}

# ##########
# test one reservoir with given parameters
# ##########

def test_one_reservoir(reservoir_params, **kwargs):

# 执行MC任务
    MC = evaluate_MC(reservoir_params, signal_len=550, **kwargs)
# 执行KRandGR任务

    krgr_result = evaluate_KRandGR(reservoir_params, Nreadouts=reservoir_params.Nvirt, Nwash=7, **kwargs)
    KR = krgr_result['KR']
    GR = krgr_result['GR']
    print(KR, GR)   

# 返回MC和KR,GR
    return {'MC': MC, 'KR': KR, 'GR': GR}
    
# 执行test_one_reservoir任务

if __name__ == "__main__":
    # 设定reservoir_params
    reservoir_params = ReservoirParams(h=0.4055105807072985, m0=0.004305768634622887, Nvirt=20, beta_prime= 41.7965657362074, params={'gamma': 0.06707779187420466, 'theta': 0.09581885346062773, 'Nvirt': 20})

    # 执行test_one_reservoir任务
    result = test_one_reservoir(reservoir_params)
    print(result)








# ##########
# run the evaluation
# ##########

def run_evaluation(
    task_type,
    param_name,
    param_range,
    reservoir_params=None,
    result_dir="./results",
    plot=True,
    verbose=False,
    extra_args=None,
    filename_prefix=None
):

    if task_type.upper() == 'MC':
        task = evaluate_MC
        result_keys = ['MC']
        result_labels = ['Memory Capacity']
    elif task_type.upper() in ['KR_GR', 'KRGR', 'KR&GR','KRandGR']:
        task = evaluate_KRandGR
        result_keys = ['KR','GR']
        result_labels = ['KR','GR']
    elif task_type.upper() == 'NARMA10':
        task = evaluate_NARMA10
        result_keys = ['NRMSE']
        result_labels = ['NRMSE']
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    scanner = ReservoirPerformanceEvaluator(
        task=task,
        param_name=param_name,
        param_range=param_range,
        result_keys=result_keys,
        result_labels=result_labels,
        reservoir_params=reservoir_params or ReservoirParams(),
        extra_args=extra_args
    )
    return scanner.evaluate(save_dir=result_dir, plot=plot, verbose=verbose, filename_prefix=filename_prefix)


# Single Parameter Evaluation Example

# if __name__ == "__main__":
#     task_types = ['MC', 'KR_GR', 'NARMA10']
#     param_name = 'theta'
#     param_range = np.linspace(0.01, 0.8, 21)  # Example range for theta
#     reservoir_params = ReservoirParams(h=0.4, m0=0.003, Nvirt=200, params={'gamma': 0.113, 'Nvirt': 200})

#     all_results = {}

#     for task_type in task_types:
#         print(f"\n=== Evaluating Task: {task_type} ===")
        
#         result = run_evaluation(
#             task_type=task_type,
#             param_name=param_name,
#             param_range=param_range,
#             reservoir_params=reservoir_params,
#             result_dir="./Results",
#             plot=False,
#             verbose=False
#         )
#         all_results[task_type] = result


# Multiple Parameter Evaluation Example
params_configs = {
    # 'beta_prime': np.arange(20, 25, 2),
    'beta_prime': np.arange(20, 51, 2),
    'theta': np.linspace(0.01, 0.8, 21),
    'gamma': np.linspace(0.01, 0.3, 21),
    'm0': np.linspace(0.001, 0.006, 21),
    'h': np.linspace(0.3, 0.5, 21)
}


# if __name__ == "__main__":
#     task_types = ['KR_GR']
#     base_params = dict(h=0.4, m0=0.003, Nvirt=200, params={'gamma': 0.113, 'theta': 0.3, 'Nvirt': 200})
#     results_all = {}

#     for param_name, param_range in params_configs.items():
#         print(f"\n== Scanning parameter: {param_name} ==")
        
#         for task_type in task_types:
#             print(f"\n--- Evaluating Task: {task_type} ---")

#             reservoir_params = ReservoirParams(**base_params)
#             result = run_evaluation(
#                 task_type=task_type,
#                 param_name=param_name,
#                 param_range=param_range,
#                 reservoir_params=reservoir_params,
#                 result_dir=f"./Results/{param_name}",
#                 plot=False,
#                 verbose=False,
#                 filename_prefix=f"Nwash=7"
#             )

#             results_all[(param_name, task_type)] = result


import numpy as np
import matplotlib.pyplot as plt
import os

def run_2d_heatmap_scan(
    task_type, param1_name, param1_range, param2_name, param2_range,
    reservoir_params_base=None, result_dir='./Results', plot=True, verbose=False, extra_args=None, filename_prefix=None
):

    heatmap_data = np.zeros((len(param1_range), len(param2_range)))

    all_heatmaps = {}

    for i, v1 in enumerate(param1_range):
        for j, v2 in enumerate(param2_range):
            params = reservoir_params_base or ReservoirParams()

            params = ReservoirParams(**vars(params))

            params.update_params(**{param1_name: v1, param2_name: v2})
            if verbose:
                print(f"Running {task_type} with {param1_name}={v1}, {param2_name}={v2}")
                params.print_params(verbose=True)

            if task_type.upper() == 'MC':
                result = evaluate_MC(params, **(extra_args or {}))
                key = 'MC'
            elif task_type.upper() in ['KR_GR', 'KRGR', 'KR&GR','KRandGR']:
                result = evaluate_KRandGR(params, **(extra_args or {}))
                key = ['KR', 'GR']
            elif task_type.upper() == 'NARMA10':
                result = evaluate_NARMA10(params, **(extra_args or {}))
                key = 'NRMSE'
            else:
                raise ValueError(f"Unknown task_type: {task_type}")


            if isinstance(key, list):  
                for k in key:
                    if k not in all_heatmaps:
                        all_heatmaps[k] = np.zeros((len(param1_range), len(param2_range)))
                    all_heatmaps[k][i, j] = result[k]
            else:  
                if key not in all_heatmaps:
                    all_heatmaps[key] = np.zeros((len(param1_range), len(param2_range)))
                all_heatmaps[key][i, j] = result[key]


    os.makedirs(result_dir, exist_ok=True)
    for k, data in all_heatmaps.items():
        if filename_prefix is None:
            base = f"heatmap_{task_type}_{k}"
        else:
            base = f"{filename_prefix}_{task_type}_{k}"
        npy_path = os.path.join(
            result_dir,
            f"{base}_{param1_name}_{param1_range[0]}to{param1_range[-1]}_{param2_name}_{param2_range[0]}to{param2_range[-1]}.npy"
        )
        np.save(npy_path, data)
        print(f"Saved heatmap data for {k} to {npy_path}")

        if plot:
            plt.figure(figsize=(8,6))
            plt.imshow(data, aspect='auto', origin='lower',
                    extent=[param2_range[0], param2_range[-1], param1_range[0], param1_range[-1]])
            plt.colorbar(label=k)
            plt.xlabel(param2_name)
            plt.ylabel(param1_name)
            plt.title(f"{task_type} {k} heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(
                result_dir,
                f"{base}_{param1_name}_{param1_range[0]}to{param1_range[-1]}_{param2_name}_{param2_range[0]}to{param2_range[-1]}.png"
            ))
            plt.show()
    return all_heatmaps


# if __name__ == "__main__":
#     param1_name = 'gamma'
#     param1_range = np.linspace(0.01, 0.3, 11)  # Example range for gamma
#     param2_name = 'beta_prime'
#     param2_range = np.arange(20, 51, 3)

#     base_params = ReservoirParams(h=0.4, m0=0.003, Nvirt=200, params={'theta': 0.3, 'Nvirt': 200})


#     for task_type in ['NARMA10']:
#         print(f"\n=== HEATMAP for Task: {task_type} ({param1_name} vs {param2_name}) ===")
#         all_heatmaps = run_2d_heatmap_scan(
#             task_type=task_type,
#             param1_name=param1_name, param1_range=param1_range,
#             param2_name=param2_name, param2_range=param2_range,
#             reservoir_params_base=base_params,
#             result_dir='./Results/heatmaps',
#             plot=False, verbose=False,
#             filename_prefix=None
#         )

from scipy.ndimage import zoom

def plot_heatmap(filepath, 
                          param1_range, param2_range,
                          param1_name='param1', param2_name='param2', 
                          metric_name='Metric', 
                          cmap='viridis', 
                          vmin=None, vmax=None, 
                          save_path=None):

    data = np.load(filepath)
    data_hr = zoom(data,(2,2), order=3)  # Upsample by a factor of 2 for better resolution

    plt.figure(figsize=(8,6))
    plt.imshow(data, interpolation='bicubic',aspect='auto', origin='lower',
            extent=[param2_range[0], param2_range[-1], param1_range[0], param1_range[-1]],
            cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{metric_name} ")
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.title(f"{metric_name} Heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()


# plt.figure(figsize=(8,6))
# # pick up data from the KR
# filepath = './Results/heatmaps/heatmap_KR_GR_KR_gamma_0.01to0.3_beta_prime_20to50.npy'
# data_KR = np.load(filepath)
# # pick up data from the GR
# filepath = './Results/heatmaps/heatmap_KR_GR_GR_gamma_0.01to0.3_beta_prime_20to50.npy'
# data_GR = np.load(filepath)

# # Calculate the CQ = KR - GR
# data_CQ = data_KR - data_GR

# plt.imshow(data_CQ, interpolation='bicubic', aspect='auto',
#            extent=[20, 50, 0.01, 0.3],)
# plt.colorbar(label='CQ (KR - GR)')
# plt.xlabel('Beta Prime')
# plt.ylabel('Theta')
# plt.title('CQ Heatmap (KR - GR)')
# plt.tight_layout()
# plt.show()           




# Example usage of plot_heatmap
# if __name__ == "__main__":

#     filepath = './Results/heatmaps/Nwash=7_KR_GR_KR_gamma_0.01to0.3_beta_prime_20to50.npy'

   
#     param1_range = np.linspace(0.01, 0.3, 11)
#     param2_range = np.arange(20, 51, 3)
#     plot_heatmap(
#         filepath,
#         param1_range=param1_range, param2_range=param2_range,
#         param1_name='Gamma', param2_name='Beta Prime',
#         metric_name='KR',
#         cmap='viridis',
#         save_path='./Results/heatmaps/Nwash=7_KR_GR_KR_gamma_0.01to0.3_beta_prime_20to50.png'
#     )

# #######
# plot several curves of various parameters with MC in one figure
# #######

def plot_multiple_curves(file_infos, save_path=True):

    plt.figure(figsize=(10, 6))

    for info in file_infos:
        if isinstance(info, dict):
            filename = info['filename']
            param_name = info['param_name']
            metric_name = info['metric_name']
            label = info.get('label', f"{param_name}_{metric_name}")

        with open(filename, 'rb') as f:
            d = pickle.load(f)
        x = d[param_name]
        # normalize x 
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = d[metric_name]

        plt.plot(x, y, marker='o', label=label)

    plt.xlabel('Parameter')
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs {param_name}")
    plt.legend()    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.show()

# Example usage of plot_multiple_curves
# if __name__ == "__main__":

#     file_infos = [
#         {'filename': './Results/beta_prime/evaluate_KRandGR_beta_prime_evaluate_beta_prime_20to50_step2.pkl', 'param_name': 'beta_prime', 'metric_name': 'GR', },
#         {'filename': './Results/theta/evaluate_KRandGR_theta_evaluate_theta_0.01to0.8_step0.0395.pkl', 'param_name': 'theta', 'metric_name': 'GR', },
#         {'filename': './Results/gamma/evaluate_KRandGR_gamma_evaluate_gamma_0.01to0.3_step0.0145.pkl', 'param_name': 'gamma', 'metric_name': 'GR', },
#         {'filename': './Results/h/evaluate_KRandGR_h_evaluate_h_0.3to0.5_step0.010000000000000009.pkl', 'param_name': 'h', 'metric_name': 'GR', },
#         {'filename': './Results/m0/evaluate_KRandGR_m0_evaluate_m0_0.001to0.006_step0.00025.pkl', 'param_name': 'm0', 'metric_name': 'GR', },
#     ]

#     plot_multiple_curves(
#         file_infos,
#         save_path='./Results/multiple_curves_GR.png'
#     )


# Single Test for a given parameter
def evaluate_and_save_single_reservoir(params, save_dir="./Results/SingleTests"):
    
    result_mc = evaluate_MC(params)
    result_krgr = evaluate_KRandGR(params)
    result_narma10 = evaluate_NARMA10(params)

    CQ = result_krgr['KR'] - result_krgr['GR']

    # 打包所有参数和结果
    record = {
        'params_dict': {
            'h': params.h,
            'beta_prime': params.beta_prime,
            'Nvirt': params.Nvirt,
            'theta': params.params['theta'],
            'm0': params.m0,
        },
        'results': {
            'MC': result_mc['MC'],
            'KR': result_krgr['KR'],
            'GR': result_krgr['GR'],
            'CQ': CQ,
            'NRMSE': result_narma10['NRMSE']
        }
    }



    filename = f"SingleTest_Nvirt{params.Nvirt}_beta_prime{params.beta_prime:.2f}.pkl"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(record, f)
    print(f"Single test results saved to {filepath}")

    return record

# Example usage of evaluate_and_save_single_reservoir
# if __name__ == "__main__":
#     params = ReservoirParams(
#         h=0.4431531552026543, m0=0.005641983615625242, Nvirt=315, beta_prime=40.66463236801917,
#         params={'theta': 0.4198538148599367, 'gamma': 0.017005257078706242, 'Nvirt': 315}
#     )
#     res = evaluate_and_save_single_reservoir(params)
#     print(res)
# import necessary libraries
import os
import torch 
import torch.nn as nn
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
    Path(r"/Users/vvvp./Desktop"),
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

from spnc_ml import spnc_TI46, spnc_narma10, spnc_narma10_Lyapunov


class ReservoirParams:
    def __init__(self, **kwargs):
            # Reservoir parameters 
            self.h = 0.4
            self.theta_H = 90
            self.k_s_0 = 0
            self.phi = 45
            self.beta_prime = 27.251620432439488

            # Network parameters 
            self.Nvirt = 30
            self.m0 = 0.006937322149792008
            self.bias = True
            self.Nwarmup = 0
            self.verbose_repr = False

            self.params = {
                'theta': 0.01,
                'gamma': 0.3663969812988086,
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


class LyapunovCalculatorTimeMux:

    def __init__(self, reservoir_params: ReservoirParams, perturbation_scale=1e-5, seed=42):

        self.reservoir_params = reservoir_params
        self.perturbation_scale = perturbation_scale
        self.seed = seed
        np.random.seed(seed)

    def _create_reservoir(self):
        return spnc_anisotropy(
            h=self.reservoir_params.h,
            theta_H=self.reservoir_params.theta_H,
            k_s=self.reservoir_params.k_s_0,
            phi=self.reservoir_params.phi,
            beta_prime=self.reservoir_params.beta_prime,
            restart=True
        )

    def compute_narma10_lyapunov(self, Ntrain=2000, Ntest=1000, **kwargs):
        """
        使用双轨迹法计算储层在NARMA10任务上的Lyapunov指数
        
        Parameters
        ----------
        Ntrain : int, default=1000
            训练样本数量
        Ntest : int, default=1000
            测试样本数量
        **kwargs : dict
            传递给spnc_narma10_Lyapunov的其他参数
            
        Lyapunov相关参数:
        ---------------
        epsilon : float, default=1e-8
            扰动幅度，使用类初始化的perturbation_scale作为默认值
        perturbation_seed : int, default=42
            扰动随机种子，使用类初始化的seed作为默认值
        lyapunov_window : int, default=100
            Lyapunov指数计算窗口大小
        return_trajectories : bool, default=False
            是否返回轨迹数据
        return_lyapunov : bool, default=True
            是否计算并返回Lyapunov指数
            
        Returns
        -------
        result : float or dict
            如果只返回Lyapunov指数，返回float
            如果返回多个值，返回包含各种结果的dict
        """
        
        # 设置Lyapunov计算的默认参数
        lyapunov_kwargs = {
            'epsilon': self.perturbation_scale,
            'perturbation_seed': self.seed,
            'return_lyapunov': True,
            'lyapunov_window': 50,
            't_start': 10,
            'min_dist': 1e-300,
        }
        
        # 用用户提供的参数覆盖默认值
        lyapunov_kwargs.update(kwargs)

        spn = self._create_reservoir()

        transform = spn.gen_signal_slow_delayed_feedback


        
        # 调用spnc_narma10_Lyapunov函数
        result = spnc_narma10_Lyapunov(
            Ntrain=Ntrain,
            Ntest=Ntest, 
            Nvirt=self.reservoir_params.Nvirt,
            m0=self.reservoir_params.m0,
            bias=self.reservoir_params.bias,
            transform=transform,
            params=self.reservoir_params.params,
            seed_NARMA=1234,
            fixed_mask=True,
            seed_mask=1234,
            **lyapunov_kwargs,
        )
        
        return result


bestCQ = ReservoirParams(
    h=0.4, m0=0.006937322149792008, Nvirt=30, beta_prime=27.251620432439488,
    params={'theta': 0.01, 'gamma': 0.3663969812988086, 'Nvirt': 30}
)
# bestMC = ReservoirParams(
#         h=0.4, m0=0.008, Nvirt=30, beta_prime=	50.0,
#         params={'theta': 0.1564938388583194, 'gamma': 0.04608425844940916, 'Nvirt': 30}
#     )
# newest = ReservoirParams(
#     h=0.4, m0=0.008, Nvirt=30, beta_prime=	38.1018718224039,
#     params={'theta': 0.6, 'gamma': 0.07387649701842956, 'Nvirt': 30}
# )
# lyapunov_calculator = LyapunovCalculatorTimeMux(bestCQ)
lyapunov_calculator = LyapunovCalculatorTimeMux(bestCQ)

result = lyapunov_calculator.compute_narma10_lyapunov(Ntrain=2000, Ntest=1000)
print('Lyapunov exponent: ', result['lyapunov_exponent'])

# 可视化outputs
# trajectory
plt.figure(figsize=(10, 5))
plt.plot(result['original_trajectory'][100:110], label='original')
plt.plot(result['perturbed_trajectory'][100:110], label='perturbed')
plt.legend()
plt.show()

# trajectory difference
plt.figure(figsize=(10, 5))
plt.plot(result['original_trajectory'] - result['perturbed_trajectory'], label='difference')
plt.legend()
plt.show()








    
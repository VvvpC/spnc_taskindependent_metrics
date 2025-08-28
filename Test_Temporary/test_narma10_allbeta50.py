'''
这个代码实现单独评估'特定Pareto前沿'的结果在NARMA10任务中的表现。
'''
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
import pandas as pd
import ast

# 构建储层对象
class ReservoirParams:
    def __init__(self, **kwargs):
            # Reservoir parameters 
            self.h = 0.4
            self.theta_H = 90
            self.k_s_0 = 0
            self.phi = 45
            self.beta_prime = 35.13826524755751

            # Network parameters 
            self.Nvirt = 50
            self.m0 = 0.005288612874870094
            self.bias = True
            self.Nwarmup = 0
            self.verbose_repr = False

            self.params = {
                'theta': 0.34142235979698393,
                'gamma': 0.069274461903986,
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

def MSE(pred, desired):
    return np.mean(np.square(np.subtract(pred, desired)))

def NRMSE(pred, y_test, spacer=0.001):
    return np.sqrt(MSE(pred, y_test) / np.var(y_test))      

def collect_data(file_path, x_col, y_col, z_col=None):
    # load data
    df = pd.read_csv(file_path)

    # extract x and y columns
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Column {x_col} or {y_col} not found in the dataframe")

    x = df[x_col].values
    y = df[y_col].values

    # extract z column if provided
    if z_col is not None:
        if z_col not in df.columns:
            raise ValueError(f"Column {z_col} not found in the dataframe")
        # 解析字符串为 list，再转成二维 numpy 数组
        z = df[z_col].apply(ast.literal_eval).to_list()
        z = np.array(z, dtype=float)
    else:
        z = None

    return x, y, z

import pickle
# if __name__ == "__main__":

#     # define the Ntrain and Ntest
#     Ntrain = 2000
#     Ntest = 1000

#     # load the data
#     file_path = r"C:\Users\Chen\Desktop\Repository\spnc_taskindependent_metrics\saved_studies\CQ_MC_Pareto_beta50_20250825_121711_pareto.csv"
#     x_col = "param_gamma"
#     y_col = "param_m0"
#     z_col = "values"

#     x, y, z = collect_data(file_path, x_col, y_col, z_col)

#     gamma_list = x
#     m0_list = y
#     CQ_list = z[:,0]
#     MC_list = z[:,1]


#     # create a dictionary to store the results
#     result_dict = {}

#     # loop through the list of gamma and m0
#     for idx, (gamma, m0) in enumerate(zip(gamma_list, m0_list)):

#         print("--------------------------------")
#         print(f"Current trial: {idx}, gamma: {gamma}, m0: {m0}")
#         print("--------------------------------")
#         # create a new ReservoirParams
#         params = ReservoirParams(
#             h=0.4, m0=m0, Nvirt=200, beta_prime=50.0,
#             params={'theta': 0.2, 'gamma': gamma, 'Nvirt': 200}
#         )
#         # create a new spnc_anisotropy
#         spn = spnc_anisotropy(
#             params.h,
#             params.theta_H,
#             params.k_s_0,
#             params.phi,
#             params.beta_prime,
#             restart=True
#         )
#         # set the transform
#         transform = spn.gen_signal_slow_delayed_feedback
#         # perform the NARMA10 task
#         (y_test, pred) = ml.spnc_narma10(
#             Ntrain,
#             Ntest,
#             params.Nvirt,
#             params.m0,
#             params.bias,
#             transform,
#             params.params,
#             seed_NARMA=1234,
#             fixed_mask=True,
#             seed_mask=1234,
#             return_outputs=True,
#         )
#         # calculate the NRMSE
#         nrmse = NRMSE(pred, y_test)
#         # store the result
#         result_dict[idx] = {
#             'gamma': gamma,
#             'm0': m0,
#             'CQ': CQ_list[idx],
#             'MC': MC_list[idx],
#             'NRMSE': nrmse,
#             'y_test': y_test,
#             'pred': pred
#         }

#     with open("result_dict_beta50.pkl", "wb") as f:
#         pickle.dump(result_dict, f)

# -------------------
# 对特定m0和gamma范围内进行网格扫描

# 定义m0和gamma的范围
gamma_range = np.linspace(0.046, 0.053, 10)
m0_range = np.linspace(0.035, 0.055, 10)

Ntrain = 2000
Ntest = 1000

# 创建一个dict来存储结果
results = {}

# 遍历m0和gamma的网格
# 使用tqdm来显示进度
from tqdm import tqdm
for i, gamma in enumerate(tqdm(gamma_range)):
    for j, m0 in enumerate(tqdm(m0_range)):
        print(f"Current trial: gamma={gamma}, m0={m0}")
        # 创建一个新 ReservoirParams
        params = ReservoirParams(
            h=0.4, m0=m0, Nvirt=200, beta_prime=50.0,
            params={'theta': 0.2, 'gamma': gamma, 'Nvirt': 200}
        )

        # 创建一个新 spnc_anisotropy
        spn = spnc_anisotropy(
            params.h,
            params.theta_H,
            params.k_s_0,
            params.phi,
            params.beta_prime,
            restart=True
        )

        # 设置transform
        transform = spn.gen_signal_slow_delayed_feedback
        (y_test, pred) = ml.spnc_narma10(
            Ntrain,
            Ntest,
            params.Nvirt,
            params.m0,
            params.bias,
            transform,
            params.params,
            seed_NARMA=1234,
            fixed_mask=True,
            seed_mask=1234,
            return_outputs=True,
        )

        # 计算NRMSE
        nrmse = NRMSE(pred, y_test)

        # 创建另一个spn,
        spn_MC = spnc_anisotropy(
            params.h,
            params.theta_H,
            params.k_s_0,
            params.phi,
            params.beta_prime,
            restart=True
        )

        # 设置transform
        transform_MC = spn_MC.gen_signal_slow_delayed_feedback


        # 存储结果
        results[i, j] = {
            'gamma': gamma,
            'm0': m0,
            'NRMSE': nrmse,
            'y_test': y_test,
            'pred': pred
        }

# 将结果保存为pickle文件
with open("results_beta50_grid_scan.pkl", "wb") as f:
    pickle.dump(results, f)


# load the result from the pickle file
# with open("result_dict_beta50.pkl", "rb") as f:
#     result_dict = pickle.load(f)


# # plot all the pred and y_test
# plt.plot(result_dict[0]['pred'][100:400],'r',label='pred')
# plt.plot(result_dict[0]['y_test'][100:400],'b',label='y_test')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.legend()
# plt.show()

# plt.figure(figsize=(7, 5))
# nrmse_list = [result_dict[idx]['NRMSE'] for idx in result_dict]
# plt.scatter(range(len(nrmse_list)), nrmse_list)
# # 每个点的上方显示该点的idx
# for idx, nrmse in enumerate(nrmse_list):
#     plt.text(idx, nrmse, str(idx), ha='center', va='bottom')
# plt.xlabel('trial')
# plt.ylabel('NRMSE')
# plt.title('NRMSE vs trial')
# plt.show()

# print('28th nrmse', result_dict[28]['NRMSE'])
# print('35th nrmse', result_dict[35]['NRMSE'])
# print('41th nrmse', result_dict[41]['NRMSE'])

# plt.figure(figsize=(7, 5))
# plt.plot(result_dict[28]['pred'][100:200],'r',label='pred')
# plt.plot(result_dict[28]['y_test'][100:200],'b',label='y_test')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.legend()
# plt.show()








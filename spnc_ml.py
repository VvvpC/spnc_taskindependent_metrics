"""
Machine learning with SPNC resevoirs

Local Dependancies
------------------
machine_learning_library  : v0.1.4.dev
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required

Functions
---------
spnc_narma10(Ntrain,Ntest,Nvirt,m0, bias, transform,params,*args,**kwargs)
    Perform the NARMA10 task with a given resevoir

spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params,*args,**kwargs)
    Perform the TI46 spoken digit task with a given resevoir
"""


# libraries
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir as shr
from joblib import Parallel, delayed
from contextmanager import no_print

CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
    Path(r"/Users/vvvp./Desktop"),
   

]
searchpaths = [p for p in CANDIDATES if p.exists()]
#tuple of repos
repos = ('machine_learning_library',)



# Add local modules and paths to local repos
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
from audio_preprocess import mfcc
from sklearn.metrics import classification_report

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def spnc_narma10(Ntrain,Ntest,Nvirt,m0, bias,
                       transform,params,*args,**kwargs):
    """
    perform the NARMA10 task with a given resevoir

    Parameters
    ----------
    Ntrain : int
        Number of samples to train
    Ntest : int
        Number of sampels to test
    Nvirt : int
        Number of virtual nodes for the resevoir
    m0 : float
        input scaling, no scaling for value of 1
    bias : bool
        True - use bias, False - don't
    transform : function or class method
        transforms a 1D numpy array through the resevoir
    params : dict
        parameters for the resevoir
    """

    seed_NARMA = kwargs.get('seed_NARMA', None)
    # print("seed NARMA: "+str(seed_NARMA))
    u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)

    x_train = u[:Ntrain]
    y_train = d[:Ntrain]
    x_test = u[Ntrain:]
    y_test = d[Ntrain:]
    
    # print first 10 elements of x_train and y_train
    # print("first 10 elements of x_train: ", x_train[:10])


    # print("Samples for training: ", len(x_train))
    # print("Samples for test: ", len(x_test))

    # Net setup
    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(y_train))

    # print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transform)
    net = linear(Nin, Nout, bias = bias)

    fixed_mask = kwargs.get('fixed_mask', True)
    if fixed_mask==True:
        # print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            # print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            # print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)



    # Training
    S_train, J_train = snr.transform(x_train,params)
    np.size(S_train)
    seed_training = kwargs.get('seed_training', 1234)
    RR.Kfold_train(net,S_train,y_train,10, quiet = True, seed_training=seed_training)


    # Testing
    S_test, J_test = snr.transform(x_test,params)

    spacer = kwargs.get('spacer_NRMSE', 0) # avoid the problem of dividing by zero
    # print("Spacer NRMSE:"+str(spacer))
    pred = net.forward(S_test)
    np.size(pred)
    error = MSE(pred, y_test)
    predNRMSE = NRMSE(pred, y_test, spacer=spacer)
    print(error, predNRMSE)

    plot = kwargs.get('plot', False)
    if plot:

        plt.figure(figsize=(7,5))
        plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
        plt.plot(y_test, pred, 'o')
        # plt.show()

    return_outputs = kwargs.get('return_outputs', False)
    if return_outputs:
        return(y_test,pred)

    return_NRMSE = kwargs.get('return_NRMSE', False)
    if return_NRMSE:
        return(predNRMSE)


def spnc_narma10_Lyapunov(Ntrain, Ntest, Nvirt, m0, bias,
                          transform, params, *args, **kwargs):
    """
    Perform the NARMA10 task with (simplified, no-renormalization) CLE using a dual-trajectory method.

    Changes vs your original:
    - Use the SAME test input for both trajectories (no input perturbation).
    - Create TWO independent reservoir kernels even if `transform` is a bound method.
    - Add an infinitesimal perturbation to the INTERNAL STATE of the second reservoir at t0=0 (if accessible).
    - Compute distance per macro-step as L2/Frobenius between state snapshots, and call
      `calculate_lyapunov_exponent` with new args: `t_start` and `min_dist`.

    Additional kwargs for CLE:
    --------------------------
    epsilon : float, default=1e-8
        Magnitude of the initial state perturbation (used at t0=0 on the perturbed reservoir).
    lyapunov_window : int, default=100
        Window size used inside calculate_lyapunov_exponent (unchanged behavior).
    t_start : int, default=0
        Number of initial macro-steps to skip when computing log-divergence (burn-in for CLE fit).
    min_dist : float, default=1e-300
        Lower bound to avoid log(0) when two trajectories coincide numerically.
    return_trajectories : bool, default=False
        Whether to return both state trajectories and the applied perturbation info.
    return_lyapunov : bool, default=True
        Whether to compute and return the Lyapunov exponent.
    seed_NARMA, fixed_mask, seed_mask, seed_training, plot, spacer_NRMSE, return_outputs, return_NRMSE
        Behave as in your original pipeline.
    """

    # -----------------------------
    # 0) Read CLE-related arguments
    # -----------------------------
    epsilon = kwargs.get('epsilon', 1e-8)            # 微小的“状态扰动”幅度（作用在 t0=0 的内部状态上）
    lyapunov_window = kwargs.get('lyapunov_window', 100)
    t_start = kwargs.get('t_start', 100)               # 新增：丢弃 CLE 拟合前的若干宏步
    min_dist = kwargs.get('min_dist', 1e-300)        # 新增：避免 log(0) 的下限
    return_lyapunov = kwargs.get('return_lyapunov', True)


    # -----------------------------
    # 1) 生成 NARMA10 序列
    # -----------------------------
    seed_NARMA = kwargs.get('seed_NARMA', None)
    u, d = NARMA10(Ntrain + Ntest, seed=seed_NARMA)

    x_train = u[:Ntrain]
    y_train = d[:Ntrain]
    x_test  = u[Ntrain:]
    y_test  = d[Ntrain:]



    # ---------------------------------------
    # 2) 构建 SNR（单节点时分复用储层）与读出
    # ---------------------------------------
    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(y_train))   # 保持你的原逻辑；若用于纯回归，推荐固定为 1（此处不改动）

    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)
    net = linear(Nin, Nout, bias=bias)

    # -----------------------------
    # 3) 固定掩膜（与原始一致）
    # -----------------------------
    fixed_mask = kwargs.get('fixed_mask', True)
    if fixed_mask:
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask >= 0:
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            snr.M = max_sequences_mask(Nin, Nvirt, m0)

    # ----------------------------------------------
    # 4) 训练读出（teacher-forced, 开环；与原始一致）
    # ----------------------------------------------
    S_train, J_train = snr.transform(x_train, params)
    seed_training = kwargs.get('seed_training', 1234)
    RR.Kfold_train(net, S_train, y_train, 10, quiet=True, seed_training=seed_training)

    # ====================================================================================
    # 5) 构建“两条轨迹”的两个【完全独立】SNR 实例（关键改动）
    # ====================================================================================
    # 目标：即使 transform 是绑定到某个“物理核对象”的方法（bound method），
    # 也要为两条轨迹分别创建独立的“核实例”，从而避免内部状态串扰。
    import copy, types

    # 5.1 识别 transform 是否为绑定方法；若是，则对其 __self__ 做深拷贝并重绑定
    if hasattr(transform, "__self__") and hasattr(transform, "__func__"):
        # 原始核实例（可能包含内部态，如 p1/p2、k_s 等）
        _spn0 = transform.__self__
        _func = transform.__func__

        # 为两条轨迹各自深拷贝一份独立的“物理核”
        _spn1 = copy.deepcopy(_spn0)
        _spn2 = copy.deepcopy(_spn0)

        # 将同一方法函数 __func__ 重新绑定到不同实例上，得到两个独立的 transform
        transform1 = types.MethodType(_func, _spn1)
        transform2 = types.MethodType(_func, _spn2)
    else:
        # 非绑定方法（纯函数）或无法识别 → 两条轨迹共用同一 callable（通常是纯函数，安全）
        transform1 = transform
        transform2 = transform

    # 5.2 用两份 transform 构建两个互不共享状态的 SNR
    snr_original  = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform1)
    snr_perturbed = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform2)

    # 5.3 确保两条轨迹使用**同一** mask（同构）
    snr_original.M  = snr.M
    snr_perturbed.M = snr.M

    # ====================================================================================
    # 6) 在 t0=0 对“第二条轨迹”的【内部状态】施加极小扰动（关键改动）
    # ====================================================================================
    # 简化 CLE 里，我们在测试开始前对内部态加扰动，然后用同一 x_test 同步驱动两条轨迹。
    # 若 transform 是绑定方法且其实例有可访问的状态（如 p1/p2），就在此处轻微调整；
    # 若不可访问，则跳过（不报错），CLE 仍可计算，但两条轨迹可能在最初完全重合（距离从极小噪声或数值误差中“起跳”）。
    # 注意：这里仅“尝试”对常见字段(p1/p2)扰动，保持保守且不假设具体内核细节。
    log_state = kwargs.get('log_state', True)  # 是否打印/记录 t0 的 p1/p2

    _state_perturb_applied = False
    p1_orig = p2_orig = p1_after = p2_after = None
    eps_state = None  # 状态空间的初始距离 ||[p1',p2'] - [p1,p2]||

    if hasattr(transform2, "__self__"):
        _kern1 = transform1.__self__   # 原轨迹的核实例
        _kern2 = transform2.__self__   # 扰动轨迹的核实例

        try:
            if hasattr(_kern2, "p1") and hasattr(_kern2, "p2") and \
            hasattr(_kern1, "p1") and hasattr(_kern1, "p2"):

                # 读取 t0 之前两条轨迹的 p1/p2（此时两者应相同）
                p1_orig = float(getattr(_kern1, "p1"))
                p2_orig = float(getattr(_kern1, "p2"))

                if log_state:
                    print(f"[CLE:t0] BEFORE perturbation | p1=p2(orig)={p1_orig:.9f}, {p2_orig:.9f},  p1+p2={p1_orig+p2_orig:.9f}")

                # 按 (+epsilon, -epsilon) 施加微扰；保持 0<=p1<=1 与 p1+p2=1
                p1_after = min(max(p1_orig + float(epsilon), 0.0), 1.0)
                p2_after = 1.0 - p1_after
                setattr(_kern2, "p1", p1_after)
                setattr(_kern2, "p2", p2_after)
                _state_perturb_applied = True

                # 计算状态空间的初始距离（理论上 ~ sqrt(2)*epsilon）
                dp1 = p1_after - p1_orig
                dp2 = p2_after - p2_orig
                eps_state = float(np.sqrt(dp1*dp1 + dp2*dp2))

                if log_state:
                    print(f"[CLE:t0]  AFTER perturbation | p1'={p1_after:.9f}, p2'={p2_after:.9f},  p1'+p2'={p1_after+p2_after:.9f}")
                    print(f"[CLE:t0]  Initial state distance ||Δr|| = {eps_state:.3e} (expected ~ {np.sqrt(2)*float(epsilon):.3e})")

            else:
                if log_state:
                    print("[CLE:t0] Kernel does not expose p1/p2; state perturbation skipped.")
                _state_perturb_applied = False

        except Exception as e:
            if log_state:
                print(f"[CLE:t0] Exception during state perturbation: {e}. Skipping perturbation.")
            _state_perturb_applied = False
    else:
        if log_state:
            print("[CLE:t0] transform2 is not a bound method; cannot access internal state to perturb.")
        _state_perturb_applied = False

    # ====================================================================================
    # 7) 以相同 x_test 驱动两条轨迹，得到状态序列 S_test_*（关键改动：不再扰动输入）
    # ====================================================================================
    # transform 内部会从初始内部态出发完成整段时分复用与延迟反馈，因此不能切断为多次调用；
    # CLE 的“热身期”由下游的 t_start 去截断，不改变这里的一次性调用方式。
    S_test_original,  J_test_original  = snr_original.transform(x_test,  params)
    S_test_perturbed, J_test_perturbed = snr_perturbed.transform(x_test, params)

    # ====================================================================================
    # 8) 计算（简化版）CLE：逐宏步距离 → log → 滑窗拟合
    #     注意：epsilon 建议用“初始状态距离”而不是“输入噪声幅度”（本实现已无输入噪声）
    # ====================================================================================
    lyapunov_exponent = None
    if return_lyapunov:
        # 以 t0=0 处的状态距离作为参考 epsilon0；若两轨迹完全重合，则用 min_dist 兜底
        # （量纲自洽，避免把“输入噪声幅度”当作状态距离的归一化）
        if getattr(S_test_original[0], 'ndim', 1) > 1:
            _eps0 = np.linalg.norm(S_test_original[0].flatten() - S_test_perturbed[0].flatten())
        else:
            _eps0 = np.linalg.norm(S_test_original[0] - S_test_perturbed[0])
        _eps0 = float(max(_eps0, min_dist))

        # 传入你已修改的 calculate_lyapunov_exponent（含 t_start 与 min_dist）
        lyapunov_exponent = calculate_lyapunov_exponent(
            S_test_original, S_test_perturbed,
            epsilon=_eps0,                 # 用初始状态距离归一化
            window_size=lyapunov_window,
            t_start=t_start,               # 新增：丢弃前若干步，避免预热影响
            min_dist=min_dist              # 新增：避免 log(0)
        )

    # -----------------------------------------------
    # 9) 标准的预测与误差（沿用你原来的评估方式）
    # -----------------------------------------------

    pred = net.forward(S_test_original)
    error = MSE(pred, y_test)
    predNRMSE = NRMSE(pred, y_test, spacer=0)

    print('Original error: ', error, 'Original NRMSE: ', predNRMSE)

    pred_perturbed = net.forward(S_test_perturbed)
    error_perturbed = MSE(pred_perturbed, y_test)
    predNRMSE_perturbed = NRMSE(pred_perturbed, y_test, spacer=0)
    print('Perturbed error: ', error_perturbed, 'Perturbed NRMSE: ', predNRMSE_perturbed)

    # -----------------------------
    # 10) 可选绘图（原样保留）
    # -----------------------------
    plot = kwargs.get('plot', False)
    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), 'k--')
        plt.plot(y_test, pred, 'o')

    # -----------------------------
    # 11) 组织返回结果（兼容原接口）
    # -----------------------------
    results = {}

    return_outputs = kwargs.get('return_outputs', False)
    if return_outputs:
        results['outputs'] = (y_test, pred)

    return_NRMSE = kwargs.get('return_NRMSE', False)
    if return_NRMSE:
        results['NRMSE'] = predNRMSE

    if return_lyapunov:
        results['lyapunov'] = {
            'original_p1': p1_orig,
            'original_p2': p2_orig,
            'perturbed_p1': p1_after,
            'perturbed_p2': p2_after,
            'lyapunov_exponent': lyapunov_exponent,
            'original_state':   S_test_original,
            'perturbed_state':  S_test_perturbed,
            'original_trajectory': pred,
            'perturbed_trajectory': pred_perturbed,
            'target': y_test,
            'original_nrmse': predNRMSE,
            'perturbed_nrmse': predNRMSE_perturbed,
        }

    if len(results) == 0:
        return predNRMSE
    elif len(results) == 1:
        return list(results.values())[0]
    else:
        return results

    

# create a new function for the narma10 task that use heterogenous reservoirs
# def spnc_narma10_heterogenous(Ntrain,Ntest,Nvirt,gamma, beta_prime, beta_ref,deltabeta_list,h,theta,m0,step,beta_left,beta_right,*weights,bias
#                        ,params,**kwargs):

'''
27/12/24 by chen, try to use 'joblib' to accelerate the temp-test
'''

import contextlib
import io

def temp_test_single(Nin, Nout, i, temp_params, res_params, snr, net, x_test, y_test, c, params, weights, spacer):

    copy_temp_params = temp_params.copy()
    copy_temp_params['beta_prime'] = i

    snr_test = shr(Nin, snr.Nvirt, Nout, copy_temp_params, res_params)
    snr_test.M = snr.M

    # 禁用 transform 内部 print
    with contextlib.redirect_stdout(io.StringIO()):
        S_warmup, J_warmup, instance_info_warmup, res_info_warmup = snr_test.transform(c, params, *weights)
        S_test, J_test, instance_info_test, res_info_test = snr_test.transform(x_test, params, *weights)
    
    pred = net.forward(S_test)
    predNRMSE = NRMSE(pred, y_test, spacer=spacer)

    return (i, predNRMSE, y_test, pred, instance_info_test, res_info_test)


def spnc_narma10_heterogenous(Ntrain,Ntest,Nvirt,bias,temp_params,params,res_params,*weights,**kwargs):
    """
    perform the NARMA10 task with a given heterogenous resevoir

    Parameters
    ----------
    Ntrain : int
        Number of samples to train
    Ntest : int
        Number of sampels to test
    Nvirt : int
        Number of virtual nodes for the heterogenous resevoir
    m0 : float
        input scaling, no scaling for value of 1
    bias : bool
        True - use bias, False - don't
    transform : function or class method
        transforms a 1D numpy array through the heterogenous resevoir
    params : dict
        parameters for the heterogenous resevoir

    adjust it by comparing with the wide_temperature_testing.py
    """

    seed_NARMA = kwargs.get('seed_NARMA', None)
    # print("seed NARMA: "+str(seed_NARMA))
    Nwarmup = kwargs.get('Nwarmup', 0)
    print("Nwarmup: "+str(Nwarmup))
    u, d = NARMA10(Nwarmup + Ntrain + Ntest,seed=seed_NARMA)

    x_train = u[Nwarmup:Nwarmup+Ntrain]
    y_train = d[Nwarmup:Nwarmup+Ntrain]
    x_test = u[Nwarmup+Ntrain:]
    y_test = d[Nwarmup+Ntrain:]
    c = u[:Nwarmup]
    l = d[:Nwarmup]

    # print('x_train', x_train[:10])
    # print("Samples for training: ", len(x_train))
    # print("Samples for test: ", len(x_test))

    # Consider DAC noise

    voltage_noise = params.get('voltage_noise', False)
    seed_voltage_noise = params.get('seed_voltage_noise', None) 
    delta_V = params.get('delta_V', 0.0)

    rng_DAC = np.random.default_rng(seed_voltage_noise)
    DAC_noise_train = rng_DAC.normal(-delta_V/2, delta_V/2, len(x_train[0]))
    DAC_noise_test = rng_DAC.normal(-delta_V/2, delta_V/2, len(x_test[0]))


    if voltage_noise == True:
        x_train = x_train + DAC_noise_train
        x_test = x_test + DAC_noise_test
        print("Voltage noise was added")
        print("Delta_V: "+str(delta_V))
    else:
        print("No voltage noise will be added")

    # Net setup
    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(y_train))

    # not sure if the shape of input is suitable for the heterogenous resevoir

    # print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    # create a heterogenous reservoir
    snr = shr(Nin, Nvirt, Nout, temp_params, res_params)
    net = linear(Nin, Nout, bias = bias)

    m0 = res_params['m0']
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

    # Warmup before training
    S_warmup, J_warmup, instance_info_warmup, res_info_warmup  = snr.transform(c,params,*weights)

    # Training
    S_train, J_train, instance_info_train, res_info_train = snr.transform(x_train,params, *weights)
    np.size(S_train)
    seed_training = kwargs.get('seed_training', 1234)
    RR.Kfold_train(net,S_train,y_train,10, quiet = True, seed_training=seed_training)
   

    # set the testing temperature range
    start_beta_prime = temp_params['beta_ref']
    step_beta_prime =  temp_params['step']
    beta_left =  temp_params['beta_left']
    beta_right =  temp_params['beta_right']

    left_sequence = np.arange(start_beta_prime, beta_left, -step_beta_prime)[1:]  
    right_sequence = np.arange(start_beta_prime, beta_right + step_beta_prime, step_beta_prime) 
    beta_prime_list = np.sort(np.concatenate((left_sequence, right_sequence)))

    beta_primes_temp = []
    nrmse_temp = []

    spacer = kwargs.get('spacer_NRMSE', 0)

    # Parallel testing
    results = Parallel(n_jobs=-1, verbose=10)(
    delayed(temp_test_single)(Nin, Nout, i, temp_params, res_params, snr, net, x_test, y_test, c, params, weights, spacer)
    for i in beta_prime_list)
    results.sort(key=lambda x: x[0])

    beta_primes_temp = [r[0] for r in results]
    nrmse_temp       = [r[1] for r in results]
    y_tests           = [r[2] for r in results]
    preds             = [r[3] for r in results]
    instance_info_tests    = [r[4] for r in results]
    res_info_tests         = [r[5] for r in results]

    return beta_primes_temp, nrmse_temp, y_tests, preds, instance_info_train, instance_info_tests, res_info_train, res_info_tests


def crop_or_pad(array, target_length):
    '''
    array: shape = (N_frams,features)
    target_length: int, the desired length of the array

    return: (target_length, features)

    '''
    n = array.shape[0]
    if n >= target_length:
        # If the array is longer than the target length, crop it
        return array[:target_length]
    else:
        pad = np.zeros((target_length - n, array.shape[1]), dtype=array.dtype)
        return np.vstack([array, pad])
    

# 修改后的 spnc_spoken_digits 函数（添加verbose控制）

def spnc_spoken_digits(speakers, Nvirt, m0, bias, transform, params, *args, verbose=False, **kwargs):
    """
    perfoms the spoken digit task with a given resevoirs

    This code is in draft, it is subject to change and error!

    Parameters
    ----------
    speakers : list of str
        Leave as None for all, otherwise list, e.g: speakers = ['f1', 'f2',...]
    Nvirt : int
        Number of virtual nodes for the resevoir
    m0 : float
        input scaling, no scaling for value of 1
    bias : bool
        True - use bias, False - don't
    transform : function or class method
        transforms a 1D numpy array through the resevoir
    params : dict
        parameters for the resevoir
    verbose : bool, optional
        If True, print progress information. If False, suppress all output.
        Default is True.
    """
    
    # 定义一个条件打印函数
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    # 定义一个条件显示函数
    def vshow():
        if verbose:
            plt.show()
        else:
            plt.close()

    # Specifying digits_only=True and train=True returns only
    # the spoken digits part of TI20 training set
    # It returns the signal, label, sampling rate and speaker of the data
    train_signal, train_label, train_rate, train_speaker = TI46.load_TI20(
        speakers, digits_only=True, train=True)

    def stratified_split(labels, N, seed=1234):
        '''
        keys = tuple
        N = int, number of each key for the first split
        seed = int, seed for RNG
        '''

        indexes = tuple(np.unique(x) for x in labels)
        sizes = tuple(len(x) for x in indexes)
        label_ids = list(np.array([], dtype=int) for i in range(np.prod(sizes)))

        for i, label in enumerate(zip(*labels)):
            ids = np.array([np.where(l == index)[0][0] for l, index in zip(label, indexes)])
            idx = np.ravel_multi_index(ids, sizes)
            label_ids[idx] = np.append(label_ids[idx], i)

        rng = np.random.default_rng(seed)
        split1 = np.array([], dtype=int)
        split2 = np.array([], dtype=int)
        for idxs in label_ids:
            rng.shuffle(idxs)
            split1 = np.append(split1, idxs[:N])
            split2 = np.append(split2, idxs[N:])
        rng.shuffle(split1)
        rng.shuffle(split2)
        return split1, split2

    split1, split2 = stratified_split((train_speaker, train_label), 9)

    # To load the test data, specify train=False
    test_signal, test_label, test_rate, test_speaker = TI46.load_TI20(
        speakers, digits_only=True, train=False)

    vprint("Samples for training: ", len(train_signal))
    vprint("Samples for test: ", len(test_signal))

    # Pre-processing
    from audio_preprocess import mfcc_func
    pre_process = mfcc_func

    nf = kwargs.get('nfft', 512)
    x_train = pre_process(train_signal, train_rate, nfft=nf)

    vprint(x_train[0].shape)

    # Normalise the input into the range 0 - 1
    xn = normalise(x_train)

    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(train_label))

    vprint('Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    SNR = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask:
        vprint("Deterministic mask will be used")
        SNR.M = fixed_seed_mask(Nin, Nvirt, m0)

    S_train, J_train = SNR.transform(xn, params)

    post_process = block_process
    post_process = lambda S, *args, **kwargs: np.copy(S)

    Nblocks = 4
    z_train = post_process(S_train, Nblocks, plot=False)

    y_train_1h = create_1hot_like(Nout, z_train, train_label)

    # Instantiate a linear output layer
    # act and inv_act are the activation function and it's inverse
    # either leave blank or set to linear to not apply activation fn
    net = linear(Nin, Nout, bias=bias)

    z_train_flat = np.vstack(z_train[split1])
    y_train_1h_flat = np.vstack(y_train_1h[split1])
    # Use the ridge regression training routine

    # Set quiet parameter for ridge regression based on verbose
    quiet_mode = not verbose
    
    RR.Kfold_train(net, z_train_flat, y_train_1h_flat, 5, quiet=quiet_mode)

    vprint('Weight matrix size = ', net.W.shape)

    conf_mat = np.zeros((Nout, Nout))
    pred_labels = np.zeros(len(split2), dtype=int)
    Ncorrect = 0
    for i, (zi, li) in enumerate(zip(z_train[split2], train_label[split2])):
        pi = net.forward(zi)
        pl = np.argmax(np.mean(pi, axis=0))
        pred_labels[i] = pl
        conf_mat[li, pl] += 1.0
        if pl == li:
            Ncorrect += 1

    # Only print classification report if verbose
    if verbose:
        vprint(classification_report(train_label[split2], pred_labels))
        valid_report = classification_report(train_label[split2], pred_labels, output_dict=True)
        vprint(valid_report.keys())
        vprint(Ncorrect/len(split2), Ncorrect, len(split2))

    # Only show plots if verbose
    if verbose:
        plt.imshow(conf_mat)
        plt.title('Training Confusion Matrix')
        # vshow()

        plt.imshow(net.W)
        plt.title('Weight Matrix')
        # vshow()

    x_test = pre_process(test_signal, test_rate, nfft=nf)
    xn_test = normalise(x_test)
    S_test, J_test = SNR.transform(xn_test, params)
    z_test = post_process(S_test, Nblocks, plot=False)

    conf_mat = np.zeros((Nout, Nout))

    Ncorrect = 0
    for i in range(len(z_test)):
        pi = net.forward(z_test[i])
        pl = np.argmax(np.mean(pi, axis=0))
        conf_mat[test_label[i], pl] += 1.0
        if pl == test_label[i]:
            Ncorrect += 1

    # Print final accuracy
    final_accuracy = Ncorrect/len(S_test)
    vprint(f"Test accuracy: {final_accuracy:.4f} ({Ncorrect}/{len(z_test)})")

    # Only show plots if verbose
    if verbose:
        plt.imshow(conf_mat)
        plt.title('Test Confusion Matrix')
        vshow()

        plt.imshow(net.W)
        plt.title('Final Weight Matrix')
        vshow()

    return_accuracy = kwargs.get('return_accuracy', False)
    if return_accuracy:
        return final_accuracy

    # np.savetxt('W', net.W)
    # np.savetxt('M', SNR.M.M)
    # np.savetxt('V', np.matmul(net.W.T[:,:-1], SNR.M.M).T)

    # np.savetxt('W', net.W)
    # np.savetxt('M', SNR.M.M)
    # np.savetxt('V', np.matmul(net.W.T[:,:-1], SNR.M.M).T)


# ################# THIS LINE IS LEFT INTENTIONALLY COMMENTED ###############

'''
Add new arguments:
    -Nvirt: number of virtual nodes for the resevoir
    -m0: input scaling, no scaling for value of 1
    -bias: True - use bias, False - don't
old version: def spnc_TI46(speakers, params, res_transform = None, prepro = "mfcc",  *args, **kwargs):

new version: def spnc_TI46(speakers, Nvirt, m0, bias, res_transform = None, prepro = "mfcc", *args, **kwargs):
'''
def spnc_TI46(speakers, Nvirt, m0, bias=True, res_transform = None, params = None, prepro = "mfcc", *args, **kwargs):

    # Select whether to run or load reservoir transformation from file
    force_compute = True

    # Specifying digits_only=True and train=True returns only the spoken digits part of TI20 training set
    # It returns the signal, label, sampling rate and speaker of the data
    train_signal, train_label, train_rate, train_speaker = TI46.load_TI20(speakers, digits_only=True, train=True)


    # To load the test data, specify train=False
    test_signal, test_label, test_rate, test_speaker = TI46.load_TI20(speakers, digits_only=True, train=False)

    print("Samples for training: ", len(train_signal))
    print("Samples for test: ", len(test_signal))



    # Pre-processing, input is a list of utterances, and output is a list containg the mfcc features of each utterance
    if prepro == "mfcc":
        print('Using MFCC preprocessing')
        pre_process = mfcc(rate=train_rate[0], nfft=1024)

    x_train = pre_process.fit_transform(train_signal)
    # print('the shape of x_train is: ', x_train.shape)
    # print('the feature of the first utterance is: ', x_train[0])
    # print('the shape of the first utterance is: ', x_train[0].shape)


    #Normalise the input into the range 0 - 1
    # becuse the mfcc features might have different value range between different dimensions, example: the value range is [0,1] for the first dimension, and [0,100] for the second dimension
    # so we need to normalise the input into the range 0 - 1
    prescaler = normaliser()
    xn = prescaler.fit_transform(x_train)
    # print('the type of xn is: ', type(xn))
    # print('the shape of xn is: ', xn.shape)
    # print('after normalise, the first feature value is: ', xn[0][0])


    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(train_label))

    print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)



    SNR = single_node_reservoir(Nin, Nout, Nvirt, m0, dilution=1.0, res=res_transform)

    fixed_mask = kwargs.get('fixed_mask', True)
    if fixed_mask: # 这里有个问题：没有注明seed。
        print("Deterministic mask will be used")
        SNR.M = fixed_seed_mask(Nin, Nvirt, m0)
        print('the shape of the mask is: ', SNR.M.M.shape)

    S_train, J_train = SNR.transform(xn, params)

    # print('J_train.shape, J_train[0].shape  ', J_train.shape, J_train[0].shape)
    # print('S_train.shape, S_train[0].shape', S_train.shape, S_train[0].shape)

    Nblocks = 8

    res_scaler = scaler()

    res_scaler.fit(S_train)

    #post_process = lambda S, *args, **kwargs : np.copy(S)
    #post_process = lambda S, *args, **kwargs : res_scaler.transform(block_process(S, Nblocks, plot=True))
    post_process = lambda S, *args, **kwargs : res_scaler.transform(S)

    z_train = post_process(S_train, Nblocks, plot=True)


    # Instantiate a linear output layer
    # act and inv_act are the activation function and it's inverse
    # either leave blank or set to linear to not apply activation fn
    net = linear(Nvirt, Nout, bias=bias)
    # print('net',net)


    # Select how many utterances to train on
    Ntrain_utter = 8
    split1, split2 = stratified_split((train_speaker, train_label), Ntrain_utter)
    # print('split1',split1)
    # print('split2',split2)


    # Create desired 1 hot output for the training
    y_train_1h = create_1hot_like(Nout, z_train, train_label)

    # Since TI46 is stored as a list of np arrays stack these into a flat array
    z_train_flat = np.vstack(z_train[split1])
    y_train_1h_flat = np.vstack(y_train_1h[split1])
    # print('the shape of z_train_flat is: ', z_train_flat.shape)
    # print('the shape of y_train_1h_flat is: ', y_train_1h_flat.shape)
    # print('the first element of z_train_flat is: ', z_train_flat[0])
    # print('the first element of y_train_1h_flat is: ', y_train_1h_flat[0])

    # Use the ridge regression training routine
    alpha = RR.Kfold_train(net, z_train_flat, y_train_1h_flat, 5, quiet=True)
    print('Optimal regression parameter = ', alpha)

    #Save the weights if needed
    # np.savetxt('Weights', net.W)


    # Calculated the predicted labels on the training set and print information
    print('Train report')
    pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_train[split1] ])
    print(classification_report(train_label[split1], pred_labels, digits=3))

    # If some utterances were held back for validation now calculate predicted labels
    if Ntrain_utter < 10:
        print('Valid report')
        pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_train[split2] ])
        print(classification_report(train_label[split2], pred_labels, digits=3))

        valid_report = classification_report(train_label[split2], pred_labels, output_dict=True)
        conf_mat = confusion_matrix(train_label[split2], pred_labels)


    # Now perform calculation on the test part of the data set

    params["name"] = "test"

    # Use the fitted/trained parts of the model to transform the test data
    x_test = pre_process.transform(test_signal)
    xn_test = prescaler.transform(x_test)
    S_test, J_test = SNR.transform(xn_test, params, force_compute)
    z_test = post_process(S_test, Nblocks, plot=False)

    # Predict the labels from the predicted output
    pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_test ])

    print(classification_report(test_label, pred_labels, digits=3))

    test_report = classification_report(test_label, pred_labels, output_dict=True)
    conf_mat = confusion_matrix(test_label, pred_labels)
    


    return accuracy_score(test_label, pred_labels)





def calculate_lyapunov_exponent(S_original, S_perturbed, epsilon, 
                                window_size=100, 
                                t_start=0, 
                                min_dist=1e-300):
    """
    Calculate the (simplified, no renormalization) conditional Lyapunov exponent
    from dual trajectories.

    Parameters
    ----------
    S_original : array_like
        Original reservoir state trajectory (sequence of state snapshots)
    S_perturbed : array_like
        Perturbed reservoir state trajectory (same length/shape as S_original)
    epsilon : float
        Initial perturbation magnitude (will be clamped by min_dist)
    window_size : int
        Window size for sliding linear fit over log-divergence
    t_start : int, optional
        Number of initial macro-steps to skip (burn-in) before computing distances
    min_dist : float, optional
        Lower bound to avoid log(0) when two trajectories coincide

    Returns
    -------
    float
        Estimated Lyapunov exponent (slope of log-distance vs time)
    """
    if len(S_original) != len(S_perturbed):
        raise ValueError("Trajectory lengths must match")

    # Clamp epsilon to avoid log(0)
    eps = float(epsilon)
    if not np.isfinite(eps):
        eps = min_dist
    eps = max(eps, min_dist)

    log_divergence = []

    # Skip the first t_start steps
    start = int(max(0, t_start))
    for i in range(start, len(S_original)):
        # Take the state snapshot at step i and flatten non-time axes
        if getattr(S_original[i], 'ndim', 1) > 1:
            state_orig = S_original[i].flatten()
            state_pert = S_perturbed[i].flatten()
        else:
            state_orig = S_original[i]
            state_pert = S_perturbed[i]

        # Euclidean distance at step i, with lower bound to avoid log(0)
        distance = np.linalg.norm(state_orig - state_pert)
        if not np.isfinite(distance):
            # guard against NaN/Inf
            distance = min_dist
        distance = max(distance, min_dist)

        # Logarithmic divergence relative to initial epsilon (clamped)
        log_div = np.log(distance / eps)
        log_divergence.append(log_div)

    # Estimate slope (Lyapunov exponent) by sliding window linear fit,
    # preserving your original behavior
    if len(log_divergence) >= window_size:
        windowed_lyapunov = []
        for j in range(window_size, len(log_divergence)):
            window_data = log_divergence[j - window_size:j]
            time_steps = np.arange(len(window_data))
            # Avoid degenerate fits (all equal values)
            if len(np.unique(window_data)) > 1:
                slope = np.polyfit(time_steps, window_data, 1)[0]
                windowed_lyapunov.append(slope)

        if windowed_lyapunov:
            lyapunov_exp = np.mean(windowed_lyapunov)
        else:
            lyapunov_exp = 0.0
    else:
        # Fallback: single fit over all available points
        time_steps = np.arange(len(log_divergence))
        if len(log_divergence) > 1 and len(np.unique(log_divergence)) > 1:
            lyapunov_exp = np.polyfit(time_steps, log_divergence, 1)[0]
        else:
            lyapunov_exp = 0.0

    return lyapunov_exp



def spnc_TI46_test(speakers, Nvirt, m0, bias=True, res_transform = None, params = None, prepro = "mfcc", *args, **kwargs):

    # Select whether to run or load reservoir transformation from file
    force_compute = True

    # Specifying digits_only=True and train=True returns only the spoken digits part of TI20 training set
    # It returns the signal, label, sampling rate and speaker of the data
    train_signal, train_label, train_rate, train_speaker = TI46.load_TI20(speakers, digits_only=True, train=True)


    # To load the test data, specify train=False
    test_signal, test_label, test_rate, test_speaker = TI46.load_TI20(speakers, digits_only=True, train=False)

    print("Samples for training: ", len(train_signal))
    print("Samples for test: ", len(test_signal))



    # Pre-processing, input is a list of utterances, and output is a list containg the mfcc features of each utterance
    if prepro == "mfcc":
        print('Using MFCC preprocessing')
        pre_process = mfcc(rate=train_rate[0], nfft=1024)

    x_train = pre_process.fit_transform(train_signal)
    # print('the shape of x_train is: ', x_train.shape)
    # print('the feature of the first utterance is: ', x_train[0])
    # print('the shape of the first utterance is: ', x_train[0].shape)


    #Normalise the input into the range 0 - 1
    # becuse the mfcc features might have different value range between different dimensions, example: the value range is [0,1] for the first dimension, and [0,100] for the second dimension
    # so we need to normalise the input into the range 0 - 1
    prescaler = normaliser()
    xn = prescaler.fit_transform(x_train)
    print('the type of xn is: ', type(xn))
    print('the shape of xn is: ', xn.shape)
    print('after normalise, the first feature value is: ', xn[0][0])


    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(train_label))

    print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    dataset_size = xn.shape[0]
    S_train = np.empty(dataset_size, dtype=object)
    J_train = np.empty(dataset_size, dtype=object)

    # 将xn拆分依次进入reservoir，得到S_train
    for i in range(xn.shape[0]):
        single_xn = xn[i]
        print('the type of single_xn is: ', type(single_xn))
        print('the shape of single_xn is: ', single_xn.shape)
        print('the first element of single_xn is: ', single_xn[0])

        SNR = single_node_reservoir(Nin, Nout, Nvirt, m0, dilution=1.0, res=res_transform)

        fixed_mask = kwargs.get('fixed_mask', True)
        if fixed_mask: # 这里有个问题：没有注明seed。
            print("Deterministic mask will be used")
            SNR.M = fixed_seed_mask(Nin, Nvirt, m0)
            # print('the shape of the mask is: ', SNR.M.M.shape)

        S, J = SNR.transform(single_xn, params)
        S_train[i] = S
        J_train[i] = J

    print('the type of S_train is: ', type(S_train))
    print('J_train.shape, J_train[0].shape  ', J_train.shape, J_train[0].shape)
    print('S_train.shape, S_train[0].shape', S_train.shape, S_train[0].shape)

    Nblocks = 8

    res_scaler = scaler()

    res_scaler.fit(S_train)

    #post_process = lambda S, *args, **kwargs : np.copy(S)
    #post_process = lambda S, *args, **kwargs : res_scaler.transform(block_process(S, Nblocks, plot=True))
    post_process = lambda S, *args, **kwargs : res_scaler.transform(S)

    z_train = post_process(S_train, Nblocks, plot=True)


    # Instantiate a linear output layer
    # act and inv_act are the activation function and it's inverse
    # either leave blank or set to linear to not apply activation fn
    net = linear(Nvirt, Nout, bias=bias)
    # print('net',net)


    # Select how many utterances to train on
    Ntrain_utter = 8
    split1, split2 = stratified_split((train_speaker, train_label), Ntrain_utter)
    # print('split1',split1)
    # print('split2',split2)


    # Create desired 1 hot output for the training
    y_train_1h = create_1hot_like(Nout, z_train, train_label)

    # Since TI46 is stored as a list of np arrays stack these into a flat array
    z_train_flat = np.vstack(z_train[split1])
    y_train_1h_flat = np.vstack(y_train_1h[split1])
    # print('the shape of z_train_flat is: ', z_train_flat.shape)
    # print('the shape of y_train_1h_flat is: ', y_train_1h_flat.shape)
    # print('the first element of z_train_flat is: ', z_train_flat[0])
    # print('the first element of y_train_1h_flat is: ', y_train_1h_flat[0])

    # Use the ridge regression training routine
    alpha = RR.Kfold_train(net, z_train_flat, y_train_1h_flat, 5, quiet=True)
    print('Optimal regression parameter = ', alpha)

    #Save the weights if needed
    # np.savetxt('Weights', net.W)


    # Calculated the predicted labels on the training set and print information
    print('Train report')
    pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_train[split1] ])
    print(classification_report(train_label[split1], pred_labels, digits=3))

    # If some utterances were held back for validation now calculate predicted labels
    if Ntrain_utter < 10:
        print('Valid report')
        pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_train[split2] ])
        print(classification_report(train_label[split2], pred_labels, digits=3))

        valid_report = classification_report(train_label[split2], pred_labels, output_dict=True)
        conf_mat = confusion_matrix(train_label[split2], pred_labels)


    # Now perform calculation on the test part of the data set

    params["name"] = "test"

    # Use the fitted/trained parts of the model to transform the test data
    x_test = pre_process.transform(test_signal)
    xn_test = prescaler.transform(x_test)

    # 将xn_test拆分依次进入reservoir，得到S_test
    dataset_size = xn_test.shape[0]
    S_test = np.empty(dataset_size, dtype=object)
    J_test = np.empty(dataset_size, dtype=object)

    for i in range(xn_test.shape[0]):
        single_xn_test = xn_test[i]
        SNR = single_node_reservoir(Nin, Nout, Nvirt, m0, dilution=1.0, res=res_transform)

        fixed_mask = kwargs.get('fixed_mask', True)
        if fixed_mask: # 这里有个问题：没有注明seed。
            print("Deterministic mask will be used")
            SNR.M = fixed_seed_mask(Nin, Nvirt, m0)
            # print('the shape of the mask is: ', SNR.M.M.shape)

        S, J = SNR.transform(single_xn_test, params)
        S_test[i] = S
        J_test[i] = J

    z_test = post_process(S_test, Nblocks, plot=False)

    # Predict the labels from the predicted output
    pred_labels = np.array([ np.argmax(np.mean(net.forward(zi), axis=0)) for zi in z_test ])

    print(classification_report(test_label, pred_labels, digits=3))

    test_report = classification_report(test_label, pred_labels, output_dict=True)
    conf_mat = confusion_matrix(test_label, pred_labels)
    


    return accuracy_score(test_label, pred_labels)

"""
储层评估模块 (Reservoir Evaluation)
==================================

这个模块实现了对均质和异质储层的CQ和MC评估：
1. evaluate_heterogeneous_MC - 评估异质储层内存容量
2. evaluate_heterogeneous_KRandGR - 评估异质储层KR和GR

Author: Chen
Date: 2025-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# 导入储层创造模块
from .Reservoirs_morphology_creator import MorphologyConfig, ReservoirMorphologyManager

# 导入参数和评估函数
from formal_Parameter_Dynamics_Preformance import (
    ReservoirParams, 
    generate_signal, 
    linear_MC,
    gen_KR_GR_input,
    Evaluate_KR_GR,
    evaluate_MC,
    evaluate_KRandGR,
    RunSpnc,
    fixed_seed_mask,
    max_sequences_mask
)
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir
from spnc import spnc_anisotropy

def RunSpnc_heterogenous(signal, Nin, Nvirt, Nout, temp_params, res_params, params, *weights, **kwargs):

    snr = single_node_heterogenous_reservoir(
        Nin, Nvirt, Nout, temp_params, res_params)
    m0 = res_params['m0']
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
    # Run
    S,_,_,_ = snr.transform(signal,params, *weights)
    
    return S




def evaluate_heterogeneous_MC(reservoir_params: ReservoirParams, config: MorphologyConfig, weights: List[float], signal_len: int = 550, **kwargs):
    """
    评估异质储层的内存容量 (Memory Capacity)
    
    Parameters:
    -----------
    reservoir_params : ReservoirParams
        储层参数
    config : MorphologyConfig
        形貌配置
    signal_len : int
        信号长度
    **kwargs : dict
        额外参数（如种子等）
        
    Returns:
    --------
    dict: {'MC': float}
    """
    print(f"\n=== 开始评估储层MC ===")
    print(f"储层配置类型: {config.morph_type}")
    print(f"储层参数 - h: {reservoir_params.h}, m0: {reservoir_params.m0}, Nvirt: {reservoir_params.Nvirt}")
    print(f"beta_prime: {reservoir_params.beta_prime}")
    print(f"信号长度: {signal_len}")
    
    # 生成测试信号
    signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
    print(f"测试信号种子: {kwargs.get('seed', 1234)}")
    
    # 判断储层类型
    if config.morph_type == 'uniform':
        # 均质储层：使用 RunSpnc
        print("使用均质储层计算")
        spn = spnc_anisotropy(
            reservoir_params.h,
            reservoir_params.theta_H,
            reservoir_params.k_s_0,
            reservoir_params.phi,
            reservoir_params.beta_prime,
            restart=True)
        
        def transform_with_constant_rate(K_s, params, *args, **kwargs):
            return spn.gen_signal_slow_delayed_feedback_omegacons(K_s, params, reservoir_params.beta_prime)
    
        Output = RunSpnc(
            signal,
            1,                 
            len(signal),       
            reservoir_params.Nvirt,
            reservoir_params.m0,
            transform_with_constant_rate,
            reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )

    else:
        # 异质储层：使用 RunSpnc_heterogenous
        print("使用异质储层计算")
        
        # 设置储层设计
        manager = ReservoirMorphologyManager()

        # 生成 deltabeta_list
        deltabeta_list = manager.generate_deltabeta_list(config, reservoir_params.beta_prime)
        print(f"生成的deltabeta_list: {deltabeta_list}")
        print(f"deltabeta_list长度: {len(deltabeta_list)}")

        # 生成权重 
        if not weights:
            weights = [1.0/len(deltabeta_list)] * len(deltabeta_list)
        
        print(f"使用的权重: {weights}")
        print(f"权重长度: {len(weights)}")

        assert len(weights) == len(deltabeta_list), f"Weights count ({len(weights)}) should match deltabeta_list count ({len(deltabeta_list)})"

        # 生成 temp_params 和 res_params
        temp_params = {
            'beta_prime': reservoir_params.beta_prime,
            'beta_ref': reservoir_params.beta_prime,
        }

        res_params = {
            'm0': reservoir_params.m0,
            'h': reservoir_params.h,
            'deltabeta_list': deltabeta_list
        }
        
        print(f"temp_params: {temp_params}")
        print(f"res_params: {res_params}")

        # 运行储层
        print("开始运行异质储层...")
        Output = RunSpnc_heterogenous(
            signal, 
            1, 
            reservoir_params.Nvirt, 
            1, 
            temp_params, 
            res_params, 
            reservoir_params.params, 
            *weights,
            fixed_mask=True,
            seed_mask=1234)
    
    # calculate the MC
    print("计算MC...")
    MC = linear_MC(signal, Output, splits=[0.2, 0.6], delays=10)
    print(f"计算得到的MC值: {MC}")
    print("=== MC评估完成 ===\n")
    
    return {'MC': MC}



    


def evaluate_heterogeneous_KRandGR(reservoir_params: ReservoirParams, config: MorphologyConfig, weights: List[float], Nreadouts: int = 50, Nwash: int = 10, **kwargs):

    # 使用reservoir的Nvirt作为Nreadouts
    Nreadouts = reservoir_params.Nvirt
    
    # 生成KR和GR输入
    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=kwargs.get('seed', 1234))
    
    # 处理每个输入行
    outputs = []
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        
        # 判断储层类型
        if config.morph_type == 'uniform':
            # 均质储层：使用 RunSpnc
            spn = spnc_anisotropy(
                reservoir_params.h,
                reservoir_params.theta_H,
                reservoir_params.k_s_0,
                reservoir_params.phi,
                reservoir_params.beta_prime,
                restart=True)
            
            def transform_with_constant_rate(K_s, params, *args, **kwargs):
                return spn.gen_signal_slow_delayed_feedback_omegacons(
                    K_s, params, reservoir_params.beta_prime
                )
        
            output = RunSpnc(
                input_row,
                1,                 
                1,       
                reservoir_params.Nvirt,
                reservoir_params.m0,
                transform_with_constant_rate,
                reservoir_params.params,
                fixed_mask=True,
                seed_mask=1234
            )

        else:
            # 异质储层：使用 RunSpnc_heterogenous
            
            # 设置储层设计
            manager = ReservoirMorphologyManager()

            # 生成 deltabeta_list
            deltabeta_list = manager.generate_deltabeta_list(config, reservoir_params.beta_prime)

            # 生成权重
            if not weights:
                weights = [1.0/len(deltabeta_list)] * len(deltabeta_list)
            assert len(weights) == len(deltabeta_list), f"Weights count ({len(weights)}) should match deltabeta_list count ({len(deltabeta_list)})"

            # 生成 temp_params 和 res_params
            temp_params = {
                'beta_prime': reservoir_params.beta_prime,
                'beta_ref': reservoir_params.beta_prime,
            }

            res_params = {
                'm0': reservoir_params.m0,
                'h': reservoir_params.h,
                'deltabeta_list': deltabeta_list
            }

            # 运行储层
            output = RunSpnc_heterogenous(
                input_row, 
                1, 
                reservoir_params.Nvirt, 
                1, 
                temp_params, 
                res_params, 
                reservoir_params.params, 
                *weights,
                fixed_mask=True,
                seed_mask=1234)
        
        outputs.append(output)
    
    # 将输出堆叠为3D数组 [samples, time_steps, features]
    States = np.stack(outputs, axis=0)
    # States = States/np.amax(States)
    if kwargs.get('threshold') is not None:
        threshold = kwargs.get('threshold')
    else:
        threshold = 0.1
    
    # 计算KR和GR
    KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=threshold)

    CQ = KR - GR
    
    return {'KR': KR, 'GR': GR, 'CQ': CQ}


def evaluate_reservoir_metrics(reservoir_params: ReservoirParams, config: MorphologyConfig, **kwargs):
    """
    综合评估储层性能，包括MC、KR和GR
    
    Parameters:
    -----------
    reservoir_params : ReservoirParams
        储层参数
    config : MorphologyConfig
        形貌配置
    **kwargs : dict
        额外参数
        
    Returns:
    --------
    dict: {'MC': float, 'KR': float, 'GR': float, 'CQ': float}
    """

    if kwargs.get('weights') is not None:
        weights = kwargs.get('weights')
    else:
        weights = None

    mc_dict = evaluate_heterogeneous_MC(reservoir_params, config, weights, **kwargs)
    kgr_dict = evaluate_heterogeneous_KRandGR(reservoir_params, config, weights, **kwargs)
    
    # 合并结果
    results = {
        'MC': mc_dict.get('MC', 0.0),
        'KR': kgr_dict.get('KR', 0.0),
        'GR': kgr_dict.get('GR', 0.0),
        'CQ': kgr_dict.get('CQ', 0.0)
    }
    
    

    return results 
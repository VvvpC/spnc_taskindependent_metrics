@ -1,181 +0,0 @@
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
from reservoir_morphology_creator import MorphologyConfig, ReservoirMorphologyManager

# 导入SPNC相关模块
from spnc import spnc_anisotropy
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir

# 导入参数和评估函数
from formal_Parameter_Dynamics_Preformance import (
    ReservoirParams, 
    generate_signal, 
    linear_MC,
    gen_KR_GR_input,
    Evaluate_KR_GR,
    evaluate_MC,
    evaluate_KRandGR,
    RunSpnc
)

def RunSpnc_heterogenous(signal, Nin, Nout, Nvirt, m0, transform, params, **kwargs):
    snr = single_node_heterogenous_reservoir(
        Nin, Nout, Nvirt, m0, res=transform)
    fixed_mask = kwargs.get('fixed_mask', True)
    if fixed_mask==True:
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    # Run
    S,_ = snr.transform(signal,params)
    
    return S




def evaluate_heterogeneous_MC(reservoir_params: ReservoirParams, config: MorphologyConfig, signal_len: int = 550, **kwargs):
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
    # 生成测试信号
    signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
    
    # 判断是否为均质储层
    if config.morph_type == 'uniform':
        # 均质储层：使用 spnc_anisotropy 和 RunSpnc
        spn = spnc_anisotropy(
            reservoir_params.h,
            reservoir_params.theta_H,
            reservoir_params.k_s_0,
            reservoir_params.phi,
            reservoir_params.beta_prime,
            restart=True
        )
        
        # 创建transform函数
        def transform_func(K_s, params, *args, **kwargs):
            return spn.gen_signal_slow_delayed_feedback_omegacons(K_s, params)
        
        # 使用 RunSpnc 运行
        Output = RunSpnc(
            signal,
            1,                 
            len(signal),       
            reservoir_params.Nvirt,
            reservoir_params.m0,
            transform_func,
            reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
        
    else:
        # 异质储层：使用 ReservoirMorphologyManager 和 RunSpnc_heterogenous
        manager = ReservoirMorphologyManager()
        
        # 生成 deltabeta_list
        deltabeta_list = manager.generate_deltabeta_list(config, reservoir_params.beta_prime)
        
        # 创建异质储层参数
        temp_params = {
            'beta_prime': reservoir_params.beta_prime,
            'beta_ref': reservoir_params.beta_prime
        }
        
        res_params = {
            'h': reservoir_params.h,
            'm0': reservoir_params.m0,
            'deltabeta_list': deltabeta_list
        }
        
        # 创建异质储层
        reservoir = single_node_heterogenous_reservoir(
            Nin=1,
            Nvirt=reservoir_params.Nvirt, 
            Nout=1,  
            temp_params=temp_params,
            res_params=res_params,
            dilution=1.0,
            identity=False
        )
        
        # 使用 RunSpnc_heterogenous 运行
        Output = RunSpnc_heterogenous(
            signal,
            1,
            len(signal),
            reservoir_params.Nvirt,
            reservoir_params.m0,
            reservoir.transform,
            reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
    
    # 计算内存容量
    MC = linear_MC(signal, Output, splits=[0.2, 0.6], delays=10)
    
    return {'MC': MC}


def evaluate_heterogeneous_KRandGR(reservoir_params: ReservoirParams, config: MorphologyConfig, Nreadouts: int = 50, Nwash: int = 7, **kwargs):
    """
    评估异质储层的KR和GR
    
    Parameters:
    -----------
    reservoir_params : ReservoirParams
        储层参数
    config : MorphologyConfig
        形貌配置
    Nreadouts : int
        读出数量
    Nwash : int
        冲洗参数
    **kwargs : dict
        额外参数
        
    Returns:
    --------
    dict: {'KR': float, 'GR': float}
    """
    # 使用reservoir的Nvirt作为Nreadouts
    Nreadouts = reservoir_params.Nvirt
    
    # 生成KR和GR输入
    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=kwargs.get('seed', 1234))

    # 创建异质储层管理器
    manager = ReservoirMorphologyManager()
    
    # 创建异质储层
    reservoir = manager.create_reservoir(config, reservoir_params)
    
    # 生成权重
    weights = manager.generate_weights(reservoir, config)
    
    # 获取transform函数
    transform_func = manager.get_transform_function(reservoir, config)
    
    # 处理每个输入行（仅处理异质储层）
    outputs = []
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        
        # 异质储层使用 transform 方法，需要权重
        result = transform_func(input_row, reservoir_params.params, *weights)
        if isinstance(result, tuple):
            output = result[0]  # 取第一个元素（储层状态）
        else:
            output = result
        output = np.asarray(output)
        
        outputs.append(output)
    
    # 将输出堆叠为3D数组 [samples, time_steps, features]
    States = np.stack(outputs, axis=0)
    
    # 计算KR和GR
    KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=0.1)
    
    return {'KR': KR, 'GR': GR}


def evaluate_reservoir_performance(reservoir_params: ReservoirParams, config: MorphologyConfig, **kwargs):
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
    # 根据形貌类型选择评估方法
    if config.morph_type == 'homogeneous':
        # 均质储层使用标准评估
        mc_dict = evaluate_MC(reservoir_params, **kwargs)
        kgr_dict = evaluate_KRandGR(reservoir_params, **kwargs)
    else:
        # 异质储层使用特殊评估
        mc_dict = evaluate_heterogeneous_MC(reservoir_params, config, **kwargs)
        kgr_dict = evaluate_heterogeneous_KRandGR(reservoir_params, config, **kwargs)
    
    # 合并结果
    results = {
        'MC': mc_dict.get('MC', 0.0),
        'KR': kgr_dict.get('KR', 0.0),
        'GR': kgr_dict.get('GR', 0.0)
    }
    
    # 计算CQ
    results['CQ'] = results['KR'] - results['GR']
    
    return results 
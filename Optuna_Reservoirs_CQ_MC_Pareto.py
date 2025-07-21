"""
储层形貌CQ-MC Pareto优化 (Reservoir Morphology CQ-MC Pareto Optimization)
=====================================================================================

本脚本基于Optuna框架实现储层形貌设计的CQ和MC Pareto前沿搜索:
• 设计不同形貌的储层 (uniform, gradient, normal distribution, random)
• 搜索超参数空间中的CQ和MC Pareto前沿
• 使用NSGA-II采样器进行多目标优化

参考文件:
- Optuna_CQ_MC_Pareto.py: Optuna框架参考实现
- Reservoir_morphology_evaluation.py: MC和CQ评估方法
- Reservoirs_morphology_creator.py: 储层形貌创建模块

Author: Chen
Date: 2025-01-XX
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from contextlib import suppress
import optuna
from optuna.samplers import GPSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional

# 导入储层创造和评估模块
from Reservoirs_morphology_creator import MorphologyConfig, ReservoirMorphologyManager
from Reservoirs_morphology_evaluation import (
    evaluate_heterogeneous_MC,
    evaluate_heterogeneous_KRandGR,
    evaluate_reservoir_performance
)
from formal_Parameter_Dynamics_Preformance import ReservoirParams

# ──────────────────────────────────────────────────────────────────────────────
# 1. 超参数搜索空间定义
# ──────────────────────────────────────────────────────────────────────────────

# 储层基础参数搜索空间
RESERVOIR_HYPERSPACE = {
    "gamma": (0.01, 0.5),       
    "theta": (0.01, 0.6),       
    "m0": (0.001, 0.008),                   
    "beta_prime": (20, 50)
}

# 固定参数
FIXED_PARAMS = {
    "h": 0.4,
    "Nvirt": 200
}

# 固定的形貌参数（当morph_type不为uniform时使用）
FIXED_MORPHOLOGY_PARAMS = {
    "n_instances": 7,  # 异质储层实例数量
    "beta_range_delta": 3,  # beta变化范围：±3 around beta_prime
    "random_seed": 1234  # 随机种子
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. 目标函数 - 返回 (CQ, MC)
# ──────────────────────────────────────────────────────────────────────────────

def objective_reservoir_morphology(trial: optuna.Trial, morph_type: str = "uniform"):
    """
    储层形貌优化的目标函数，评估CQ和MC
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna试验对象
    morph_type : str
        储层形貌类型 ("uniform", "gradient", "normaldistribution", "random")
        
    Returns:
    --------
    tuple: (CQ, MC)
    """
    # 1. 采样储层基础参数
    gamma = trial.suggest_float("gamma", *RESERVOIR_HYPERSPACE["gamma"])
    theta = trial.suggest_float("theta", *RESERVOIR_HYPERSPACE["theta"])
    m0 = trial.suggest_float("m0", *RESERVOIR_HYPERSPACE["m0"])
    beta_prime = trial.suggest_float("beta_prime", *RESERVOIR_HYPERSPACE["beta_prime"])
    
    # 使用固定参数
    h = FIXED_PARAMS["h"]
    Nvirt = FIXED_PARAMS["Nvirt"]
    
    # 2. 储层形貌参数由函数参数指定
    # morph_type 已经作为函数参数传入
    
    # 3. 构建储层参数对象
    reservoir_params = ReservoirParams(
        h=h,
        m0=m0,
        Nvirt=Nvirt,
        beta_prime=beta_prime,
        # 设置物理参数默认值（这些参数在evaluation中需要）
        theta_H=90,  # 磁场角度
        k_s_0=0,     # 初始各向异性
        phi=45,      # 磁化角度
        params={
            "gamma": gamma,
            "theta": theta,
            "Nvirt": Nvirt,
        },
    )
    
    # 4. 构建形貌配置
    if morph_type == "uniform":
        config = MorphologyConfig(morph_type="uniform")
    else:
        # 对于非均质储层，使用固定参数
        n_instances = FIXED_MORPHOLOGY_PARAMS["n_instances"]
        
        # 基于当前trial的beta_prime动态计算beta_range
        beta_delta = FIXED_MORPHOLOGY_PARAMS["beta_range_delta"]
        beta_range = (beta_prime - beta_delta, beta_prime + beta_delta)
        
        # 设置随机种子
        random_seed = FIXED_MORPHOLOGY_PARAMS["random_seed"] if morph_type in ["normaldistribution", "random"] else None
        
        config = MorphologyConfig(
            morph_type=morph_type,
            beta_range=beta_range,
            n_instances=n_instances,
            random_seed=random_seed
        )
    
    try:
        # 5. 评估MC和CQ
        mc_dict = evaluate_heterogeneous_MC(reservoir_params, config, signal_len=550, seed=1234)
        kgr_dict = evaluate_heterogeneous_KRandGR(reservoir_params, config, Nwash=10, seed=1234)
        
        MC = float(mc_dict.get("MC", 0.0))
        KR = float(kgr_dict.get("KR", 0.0))
        GR = float(kgr_dict.get("GR", 0.0))
        CQ = KR - GR
        
        # 6. 早期剪枝低性能试验
        if MC < 0 or CQ < 0:
            raise optuna.exceptions.TrialPruned()
        
        # 添加试验属性用于后续分析
        trial.set_user_attr("morph_type", morph_type)
        trial.set_user_attr("KR", KR)
        trial.set_user_attr("GR", GR)
        
        return CQ, MC
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

# ──────────────────────────────────────────────────────────────────────────────
# 3. 研究设置
# ──────────────────────────────────────────────────────────────────────────────

def create_morphology_study(morph_type: str = "uniform"):
    """创建储层形貌优化研究"""
    suffix = 0
    storage = "sqlite:///db.sqlite3" 
    study_name = f"Reservoir_Morphology_CQ_MC_Pareto_{morph_type}"
    new_study_name = study_name
    
    # 检查研究是否已存在，如存在则添加后缀
    while True:
        try:
            optuna.load_study(study_name=new_study_name, storage=storage)
            suffix += 1
            new_study_name = f"{study_name}_{suffix}"
        except KeyError:
            print(f"for {morph_type} morphology, create new study: '{new_study_name}'")
            break
    
    # 使用GP采样器进行多目标优化
    sampler = GPSampler()
    
    study = optuna.create_study(
        sampler=sampler,
        directions=["maximize", "maximize"],  # 最大化CQ和MC
        storage=storage,
        study_name=new_study_name,
    )
    
    return study

# ──────────────────────────────────────────────────────────────────────────────
# 4. 运行研究
# ──────────────────────────────────────────────────────────────────────────────

def run_morphology_study(n_trials: int = 400, morph_type: str = "uniform"):
    """
    运行储层形貌CQ-MC Pareto优化研究
    
    Parameters:
    -----------
    n_trials : int
        试验数量
    timeout : int, optional
        超时时间（秒）
    morph_type : str
        储层形貌类型 ("uniform", "gradient", "normaldistribution", "random")
    """
    study = create_morphology_study(morph_type)
    
    print(f"Start the study, number of trials: {n_trials}")
    print(f"Morphology type: {morph_type}")

    
    study.optimize(
        lambda trial: objective_reservoir_morphology(trial, morph_type),
        n_trials=n_trials,
        catch=(ValueError, FloatingPointError),
    )
    
    print("\nPareto front (all non-dominated trials):")
    for t in study.best_trials:
        print('  Values: ', t.values)
        print('  Params:')
        for key, value in t.params.items():
            print(f'    {key}: {value}')
        print('-------------------')
    
    
 


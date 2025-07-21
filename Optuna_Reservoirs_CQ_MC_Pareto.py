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
    "n_instances": 5,  # 异质储层实例数量
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
    
    
 

# ──────────────────────────────────────────────────────────────────────────────
# 5. 主函数
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 6. 热力图绘制函数
# ──────────────────────────────────────────────────────────────────────────────

def plot_beta_gamma_heatmap(morph_type: str = "uniform", save_path: str = None):
    """
    绘制beta_prime和gamma的热力图
    
    Parameters:
    -----------
    morph_type : str
        储层形貌类型 ("uniform", "gradient", "normaldistribution", "random")
    save_path : str, optional
        保存路径，如果为None则显示图像
    """
    # 参数网格设置
    beta_prime_range = np.linspace(25, 35, 10)
    gamma_range = np.linspace(0.04, 0.06, 10)
    
    # 创建网格
    beta_grid, gamma_grid = np.meshgrid(beta_prime_range, gamma_range)
    
    # 初始化结果矩阵
    cq_matrix = np.zeros((10, 10))
    mc_matrix = np.zeros((10, 10))
    
    print(f"开始计算热力图数据，形貌类型: {morph_type}")
    
    # 固定其他参数
    theta = 0.3  # 使用中间值
    m0 = 0.004   # 使用中间值
    h = FIXED_PARAMS["h"]
    Nvirt = FIXED_PARAMS["Nvirt"]
    
    # 遍历参数网格
    for i, gamma in enumerate(gamma_range):
        for j, beta_prime in enumerate(beta_prime_range):
            try:
                # 构建储层参数对象
                reservoir_params = ReservoirParams(
                    h=h,
                    m0=m0,
                    Nvirt=Nvirt,
                    beta_prime=beta_prime,
                    theta_H=90,
                    k_s_0=0,
                    phi=45,
                    params={
                        "gamma": gamma,
                        "theta": theta,
                        "Nvirt": Nvirt,
                    },
                )
                
                # 构建形貌配置
                if morph_type == "uniform":
                    config = MorphologyConfig(morph_type="uniform")
                else:
                    beta_delta = FIXED_MORPHOLOGY_PARAMS["beta_range_delta"]
                    beta_range = (beta_prime - beta_delta, beta_prime + beta_delta)
                    random_seed = FIXED_MORPHOLOGY_PARAMS["random_seed"] if morph_type in ["normaldistribution", "random"] else None
                    
                    config = MorphologyConfig(
                        morph_type=morph_type,
                        beta_range=beta_range,
                        n_instances=FIXED_MORPHOLOGY_PARAMS["n_instances"],
                        random_seed=random_seed
                    )
                
                # 评估MC和CQ
                mc_dict = evaluate_heterogeneous_MC(reservoir_params, config, signal_len=550, seed=1234)
                kgr_dict = evaluate_heterogeneous_KRandGR(reservoir_params, config, Nwash=10, seed=1234)
                
                MC = float(mc_dict.get("MC", 0.0))
                KR = float(kgr_dict.get("KR", 0.0))
                GR = float(kgr_dict.get("GR", 0.0))
                CQ = KR - GR
                
                cq_matrix[i, j] = CQ
                mc_matrix[i, j] = MC
                
                print(f"完成 gamma={gamma:.3f}, beta_prime={beta_prime:.1f}, CQ={CQ:.3f}, MC={MC:.3f}")
                
            except Exception as e:
                print(f"计算失败 gamma={gamma:.3f}, beta_prime={beta_prime:.1f}: {e}")
                cq_matrix[i, j] = 0
                mc_matrix[i, j] = 0
    
    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CQ热力图
    im1 = ax1.imshow(cq_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title(f'CQ热力图 ({morph_type})', fontsize=14)
    ax1.set_xlabel('beta_prime', fontsize=12)
    ax1.set_ylabel('gamma', fontsize=12)
    
    # 设置刻度标签
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels([f'{x:.1f}' for x in beta_prime_range])
    ax1.set_yticklabels([f'{x:.3f}' for x in gamma_range])
    
    plt.colorbar(im1, ax=ax1, label='CQ')
    
    # MC热力图
    im2 = ax2.imshow(mc_matrix, cmap='plasma', aspect='auto', origin='lower')
    ax2.set_title(f'MC热力图 ({morph_type})', fontsize=14)
    ax2.set_xlabel('beta_prime', fontsize=12)
    ax2.set_ylabel('gamma', fontsize=12)
    
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels([f'{x:.1f}' for x in beta_prime_range])
    ax2.set_yticklabels([f'{x:.3f}' for x in gamma_range])
    
    plt.colorbar(im2, ax=ax2, label='MC')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
      import argparse

      parser = argparse.ArgumentParser(description="储层形貌CQ-MC Pareto优化")
      parser.add_argument("--trials", type=int, default=400, help="试验数量")
      parser.add_argument("--morph_type", type=str, default="uniform",
                          choices=["uniform", "gradient", "normaldistribution", "random"],
                          help="储层形貌类型")
      parser.add_argument("--heatmap", action="store_true", help="绘制beta_prime和gamma热力图")

      args = parser.parse_args()

      if args.heatmap:
          # 绘制热力图
          plot_beta_gamma_heatmap(morph_type=args.morph_type, save_path=f"heatmap_{args.morph_type}.png")
      else:
          # 运行优化研究
          study = run_morphology_study(n_trials=args.trials, morph_type="random")
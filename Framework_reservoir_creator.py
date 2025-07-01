"""
储层形貌创造模块 (Reservoir Morphology Creator)
===============================================

这个模块实现了对不同形貌储层的配置和创建：
1. MorphologyConfig - 储层形貌配置类
2. ReservoirMorphologyManager - 储层形貌管理器

Author: Chen
Date: 2025-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# 导入现有模块
from spnc import spnc_anisotropy
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir
from formal_Parameter_Dynamics_Preformance import ReservoirParams


@dataclass 
class MorphologyConfig:
    """储层形貌配置类"""
    morph_type: str  # 'homogeneous', 'gradient', 'random'
    
    # 异质储层相关参数
    beta_range: Optional[Tuple[float, float]] = None  # beta变化范围
    distribution_type: Optional[str] = None           # 'linear', 'exponential', 'random'
    random_seed: Optional[int] = None                 # 随机种子
    n_instances: Optional[int] = None                 # 异质储层中的实例数量（仅用于gradient和random）
    
    def __post_init__(self):
        if self.morph_type in ['gradient', 'random']:
            if self.beta_range is None:
                raise ValueError(f"{self.morph_type} morphology requires beta_range to be specified")
            if self.n_instances is None:
                self.n_instances = 100  # 默认实例数量


class ReservoirMorphologyManager:
    """储层形貌管理器"""
    
    def __init__(self):
        self.supported_morphologies = ['homogeneous', 'gradient', 'random']
    
    def generate_deltabeta_list(self, config: MorphologyConfig, base_beta: float) -> List[float]:
        """根据配置生成deltabeta列表"""
        if config.morph_type == 'homogeneous':
            return [0.0]  # 均质储层只有一个实例，delta=0
        
        elif config.morph_type == 'gradient':
            # 渐变分布
            if config.beta_range is None or config.n_instances is None:
                raise ValueError("Gradient morphology requires beta_range and n_instances")
            delta_min = config.beta_range[0] - base_beta
            delta_max = config.beta_range[1] - base_beta
            # 生成渐变分布的deltabeta列表，来控制储层的形貌部分
            return np.linspace(delta_min, delta_max, config.n_instances).tolist()
        
        elif config.morph_type == 'random':
            # 随机分布
            if config.beta_range is None or config.n_instances is None:
                raise ValueError("Random morphology requires beta_range and n_instances")
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            
            delta_min = config.beta_range[0] - base_beta
            delta_max = config.beta_range[1] - base_beta
            # 生成随机分布的deltabeta列表，来控制储层的形貌部分
            return np.random.uniform(delta_min, delta_max, config.n_instances).tolist()
        
        else:
            raise ValueError(f"Unsupported morphology type: {config.morph_type}")
    
    def create_reservoir(self, config: MorphologyConfig, reservoir_params: ReservoirParams):
        """创建指定形貌的储层"""
        if config.morph_type == 'homogeneous':
            # 创建均质储层 - 使用单个spnc_anisotropy实例
            return spnc_anisotropy(
                h=reservoir_params.h,
                theta_H=reservoir_params.theta_H,
                k_s=reservoir_params.k_s_0,
                phi=reservoir_params.phi,
                beta_prime=reservoir_params.beta_prime,
                restart=True
            )
        
        elif config.morph_type in ['gradient', 'random']:
            # 创建异质储层
            deltabeta_list = self.generate_deltabeta_list(config, reservoir_params.beta_prime)
            
            # 构建异质储层参数，与均质储层参数相同
            temp_params = {
                'beta_prime': reservoir_params.beta_prime,
                'beta_ref': reservoir_params.beta_prime  # 使用相同的参考温度
            }
            
            res_params = {
                'h': reservoir_params.h,
                'm0': reservoir_params.m0,
                'deltabeta_list': deltabeta_list
            }
            
            return single_node_heterogenous_reservoir(
                Nin=1,
                Nvirt=reservoir_params.Nvirt,  # 与均质储层保持一致
                Nout=1,   # 根据existing code pattern，对于非ML任务，Nout通常设为1或小值
                temp_params=temp_params,
                res_params=res_params,
                dilution=1.0,
                identity=False
            )
        
        else:
            raise ValueError(f"Unsupported morphology type: {config.morph_type}")
    
    def get_transform_function(self, reservoir, config: MorphologyConfig):
        """获取对应的transform函数"""
        if config.morph_type == 'homogeneous':
            return reservoir.get_signal_slow_delayed_feedback_heteroRes_sameinput
        else:
            # 直接返回异质储层的transform方法
            return reservoir.transform
    
    def generate_weights(self, reservoir, config: MorphologyConfig):
        """为异质储层生成均匀权重, 确保权重和严格等于1"""
        if config.morph_type == 'homogeneous':
            return []  # 均质储层不需要权重
        
        n_instances = len(reservoir.anisotropy_instances)
        
        # 对于渐变和随机形貌，所有实例使用相同的权重 = 1/实例数量
        weights = [1.0/n_instances] * n_instances

        return weights


# 便捷函数
def create_standard_morphology_configs(n_instances: int = 3, 
                                     beta_range: Tuple[float, float] = (20, 30)) -> List[MorphologyConfig]:
    """创建标准的三种形貌配置"""
    return [
        MorphologyConfig(
            morph_type='homogeneous'
        ),
        MorphologyConfig(
            morph_type='gradient',
            n_instances=n_instances,
            beta_range=beta_range,
            distribution_type='linear'
        ),
        MorphologyConfig(
            morph_type='random',
            n_instances=n_instances,
            beta_range=beta_range,
            random_seed=1234
        )
    ] 
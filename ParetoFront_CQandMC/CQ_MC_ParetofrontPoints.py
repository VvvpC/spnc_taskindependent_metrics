"""
CQ_MC_ParetofrontPoints.py
=========================

从Pareto前沿文件中提取参数，创建储层，并评估NARMA-10和TI46任务性能。

Author: Chen
Date: 2025-01-25
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import glob

# 导入储层相关模块
from spnc import spnc_anisotropy
from formal_Parameter_Dynamics_Preformance import ReservoirParams
from Reservoirs_morphology_creator import MorphologyConfig, ReservoirMorphologyManager

# 导入ML评估模块
from spnc_ml import spnc_narma10, spnc_spoken_digits

@dataclass
class ParetoPointParams:
    """Pareto点参数数据类"""
    trial_number: int
    gamma: float
    theta: float
    m0: float
    beta_prime: float
    cq_value: float
    mc_value: float

@dataclass 
class TaskResults:
    """任务评估结果数据类"""
    trial_number: int
    gamma: float
    theta: float
    m0: float
    beta_prime: float
    
    # NARMA-10结果
    narma10_nrmse: float
    narma10_y_test: np.ndarray
    narma10_pred: np.ndarray
    
    # TI46结果
    ti46_accuracy: float

class ParetoPointEvaluator:
    """Pareto点评估器"""
    
    def __init__(self):
        self.manager = ReservoirMorphologyManager()
        
    def load_pareto_csv(self, filename: str) -> List[ParetoPointParams]:
        """
        从CSV文件加载Pareto点参数
        支持使用filename搜索文件路径
        """
        # 搜索匹配的文件
        search_pattern = f"**/*{filename}*"
        matching_files = list(Path(".").glob(search_pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"未找到包含'{filename}'的文件")
        
        if len(matching_files) > 1:
            print(f"找到多个匹配文件：{matching_files}")
            print(f"使用第一个文件：{matching_files[0]}")
        
        csv_path = matching_files[0]
        print(f"读取文件：{csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        pareto_points = []
        for _, row in df.iterrows():
            # 解析values列（包含CQ和MC值）
            values_str = row['values'].strip('[]')
            cq_value, mc_value = map(float, values_str.split(', '))
            
            pareto_point = ParetoPointParams(
                trial_number=int(row['number']),
                gamma=float(row['param_gamma']),
                theta=float(row['param_theta']), 
                m0=float(row['param_m0']),
                beta_prime=float(row['param_beta_prime']),
                cq_value=cq_value,
                mc_value=mc_value
            )
            pareto_points.append(pareto_point)
            
        print(f"成功加载{len(pareto_points)}个Pareto点")
        return pareto_points
    
    def create_reservoir_params(self, pareto_point: ParetoPointParams) -> ReservoirParams:
        """根据Pareto点参数创建储层参数"""
        return ReservoirParams(
            gamma=pareto_point.gamma,
            theta_H=pareto_point.theta,
            m0=pareto_point.m0,
            beta_prime=pareto_point.beta_prime,
            h=0.4,  # 固定值
            k_s_0=1.0,  # 固定值
            phi=0.0,  # 固定值
            Nvirt=200,  # 固定值
            params={
                'name': f'trial_{pareto_point.trial_number}',
                'theta': pareto_point.theta,
                'gamma': pareto_point.gamma,
                'h': 0.4,
                'K_s': 1.0,
                'phi': 0.0
            }
        )
    
    def create_uniform_reservoir(self, reservoir_params: ReservoirParams):
        """创建均质储层（uniform类型）"""
        return spnc_anisotropy(
            h=reservoir_params.h,
            theta_H=reservoir_params.theta_H,
            k_s=reservoir_params.k_s_0,
            phi=reservoir_params.phi,
            beta_prime=reservoir_params.beta_prime,
            restart=True
        )
    
    def evaluate_narma10(self, reservoir, reservoir_params: ReservoirParams, 
                        Ntrain: int = 1000, Ntest: int = 1000) -> Tuple[float, np.ndarray, np.ndarray]:
        """评估NARMA-10任务"""
        
        # 定义变换函数
        def transform_with_constant_rate(K_s, params, *args, **kwargs):
            return reservoir.gen_signal_slow_delayed_feedback_omegacons(
                K_s, params, reservoir_params.beta_prime
            )
        
        # 运行NARMA-10任务
        y_test, pred = spnc_narma10(
            Ntrain=Ntrain,
            Ntest=Ntest,
            Nvirt=reservoir_params.Nvirt,
            m0=reservoir_params.m0,
            bias=True,
            transform=transform_with_constant_rate,
            params=reservoir_params.params,
            return_outputs=True,
            fixed_mask=True,
            seed_mask=1234,
            seed_NARMA=1234,
            seed_training=1234,
            spacer_NRMSE=0.001
        )
        
        # 计算NRMSE
        from utility import NRMSE
        nrmse = NRMSE(pred, y_test, spacer=0.001)
        
        return nrmse, y_test, pred
    
    def evaluate_ti46(self, reservoir, reservoir_params: ReservoirParams, 
                     speakers: Optional[List[str]] = None) -> float:
        """评估TI46任务"""
        
        # 定义变换函数
        def transform_with_constant_rate(K_s, params, *args, **kwargs):
            return reservoir.gen_signal_slow_delayed_feedback_omegacons(
                K_s, params, reservoir_params.beta_prime
            )
        
        # 运行TI46任务
        accuracy = spnc_spoken_digits(
            speakers=speakers,
            Nvirt=reservoir_params.Nvirt,
            m0=reservoir_params.m0,
            bias=True,
            transform=transform_with_constant_rate,
            params=reservoir_params.params,
            verbose=False,
            return_accuracy=True,
            fixed_mask=True
        )
        
        return accuracy
    
    def evaluate_single_point(self, pareto_point: ParetoPointParams, 
                            narma_config: Dict = None, ti46_config: Dict = None) -> TaskResults:
        """评估单个Pareto点"""
        
        # 设置默认配置
        if narma_config is None:
            narma_config = {'Ntrain': 1000, 'Ntest': 1000}
        if ti46_config is None:
            ti46_config = {'speakers': None}
        
        print(f"评估Trial {pareto_point.trial_number}: gamma={pareto_point.gamma:.4f}, "
              f"theta={pareto_point.theta:.4f}, m0={pareto_point.m0:.4f}, "
              f"beta_prime={pareto_point.beta_prime:.1f}")
        
        # 创建储层参数和储层
        reservoir_params = self.create_reservoir_params(pareto_point)
        reservoir = self.create_uniform_reservoir(reservoir_params)
        
        # 评估NARMA-10
        print("  评估NARMA-10...")
        narma10_nrmse, narma10_y_test, narma10_pred = self.evaluate_narma10(
            reservoir, reservoir_params, **narma_config
        )
        print(f"  NARMA-10 NRMSE: {narma10_nrmse:.4f}")
        
        # 评估TI46
        print("  评估TI46...")
        ti46_accuracy = self.evaluate_ti46(reservoir, reservoir_params, **ti46_config)
        print(f"  TI46 Accuracy: {ti46_accuracy:.4f}")
        
        return TaskResults(
            trial_number=pareto_point.trial_number,
            gamma=pareto_point.gamma,
            theta=pareto_point.theta,
            m0=pareto_point.m0,
            beta_prime=pareto_point.beta_prime,
            narma10_nrmse=narma10_nrmse,
            narma10_y_test=narma10_y_test,
            narma10_pred=narma10_pred,
            ti46_accuracy=ti46_accuracy
        )
    
    def evaluate_all_points(self, filename: str, 
                          narma_config: Dict = None, ti46_config: Dict = None,
                          output_filename: str = None) -> List[TaskResults]:
        """评估所有Pareto点"""
        
        # 加载Pareto点
        pareto_points = self.load_pareto_csv(filename)
        
        # 评估所有点
        all_results = []
        for i, point in enumerate(pareto_points):
            print(f"\n进度: {i+1}/{len(pareto_points)}")
            try:
                result = self.evaluate_single_point(point, narma_config, ti46_config)
                all_results.append(result)
            except Exception as e:
                print(f"  错误：评估Trial {point.trial_number}时出现异常: {e}")
                continue
        
        # 保存结果
        if output_filename is None:
            # 从输入文件名生成输出文件名
            base_name = Path(filename).stem
            output_filename = f"ParetoFront_TaskResults_{base_name}"
        
        self.save_results(all_results, output_filename)
        
        return all_results
    
    def save_results(self, results: List[TaskResults], output_filename: str):
        """保存评估结果"""
        
        output_dir = Path("ParetoFront_CQandMC")
        output_dir.mkdir(exist_ok=True)
        
        # 保存为pickle文件（包含所有数据）
        pickle_file = output_dir / f"{output_filename}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"完整结果已保存至: {pickle_file}")
        
        # 保存为CSV文件（汇总数据）
        csv_file = output_dir / f"{output_filename}.csv"
        summary_data = []
        for result in results:
            summary_data.append({
                'trial_number': result.trial_number,
                'gamma': result.gamma,
                'theta': result.theta,
                'm0': result.m0,
                'beta_prime': result.beta_prime,
                'narma10_nrmse': result.narma10_nrmse,
                'ti46_accuracy': result.ti46_accuracy
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        print(f"汇总结果已保存至: {csv_file}")
        
        # 保存详细的NARMA-10数据
        narma_data_file = output_dir / f"{output_filename}_narma10_detailed.pkl"
        narma_data = {}
        for result in results:
            narma_data[result.trial_number] = {
                'y_test': result.narma10_y_test,
                'pred': result.narma10_pred,
                'nrmse': result.narma10_nrmse
            }
        
        with open(narma_data_file, 'wb') as f:
            pickle.dump(narma_data, f)
        print(f"NARMA-10详细数据已保存至: {narma_data_file}")


def main():
    """主函数示例"""
    
    # 创建评估器
    evaluator = ParetoPointEvaluator()
    
    # 配置参数
    filename = "Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049_pareto.csv"
    
    # NARMA-10配置
    narma_config = {
        'Ntrain': 1000,
        'Ntest': 1000
    }
    
    # TI46配置
    ti46_config = {
        'speakers': None  # 使用所有说话者
    }
    
    # 评估所有点
    results = evaluator.evaluate_all_points(
        filename=filename,
        narma_config=narma_config,
        ti46_config=ti46_config,
        output_filename="uniform_pareto_task_results"
    )
    
    print(f"\n评估完成！共处理{len(results)}个Pareto点")


if __name__ == "__main__":
    main()
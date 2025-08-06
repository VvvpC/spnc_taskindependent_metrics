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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import glob

import sys
from pathlib import Path

# 添加上级目录到Python路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 直接导入所需模块
from spnc import spnc_anisotropy
from formal_Parameter_Dynamics_Preformance import ReservoirParams
from Reservoirs_morphology_creator import MorphologyConfig, ReservoirMorphologyManager
from spnc_ml import spnc_narma10, spnc_TI46
from formal_Parameter_Dynamics_Preformance import evaluate_KRandGR

def MSE(pred, desired):
    return np.mean(np.square(np.subtract(pred, desired)))

def NRMSE(pred, y_test, spacer=0.001):
    return np.sqrt(MSE(pred, y_test) / np.var(y_test))

def eva_narma10(reservoir_params: ReservoirParams, 
                        Ntrain: int = 2000, Ntest: int = 1000) -> Tuple[float, np.ndarray, np.ndarray]:
        """评估NARMA-10任务"""

        # 创建储层
        spn = spnc_anisotropy(
            h=reservoir_params.h,
            theta_H=reservoir_params.theta_H,
            k_s=reservoir_params.k_s_0,
            phi=reservoir_params.phi,
            beta_prime=reservoir_params.beta_prime,
            restart=True
        )

        transform = spn.gen_signal_slow_delayed_feedback
        
        # 运行NARMA-10任务
        (y_test, pred) = spnc_narma10(
            Ntrain,
            Ntest,
            reservoir_params.Nvirt,
            reservoir_params.m0,
            reservoir_params.bias,
            transform,
            reservoir_params.params,
            seed_NARMA=1234,
            fixed_mask=True,
            seed_mask=1234,
            return_outputs=True,
        )
        
        # 计算NRMSE
        nrmse = NRMSE(pred, y_test)
        
        return nrmse, y_test, pred
    
def eva_ti46(reservoir_params: ReservoirParams, 
                    speakers: Optional[List[str]] = None) -> float:
    """评估TI46任务"""
    
    # 创建储层
    spn = spnc_anisotropy(
        h=reservoir_params.h,
        theta_H=reservoir_params.theta_H,
        k_s=reservoir_params.k_s_0,
        phi=reservoir_params.phi,
        beta_prime=reservoir_params.beta_prime,
        restart=True
    )

    transform = spn.gen_signal_slow_delayed_feedback
    
    # 运行TI46任务
    accuracy = spnc_TI46(
        speakers,
        reservoir_params.Nvirt,
        reservoir_params.m0,
        reservoir_params.bias,
        transform,
        reservoir_params.params)
    
    return accuracy

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

    # KR和GR阈值
    kr_threshold: float
    gr_threshold: float

@dataclass
class ParameterSource:
    """参数来源配置类"""
    source_type: str  # 'csv' or 'dict'
    data: Union[str, Dict, List[Dict]]  # CSV文件名或参数字典/字典列表

class ParetoPointEvaluator:
    """Pareto点评估器"""
    
    def __init__(self):
        self.manager = ReservoirMorphologyManager()
    
    @staticmethod
    def validate_param_dict(params: Dict) -> Dict:
        """
        验证并标准化参数字典
        
        Args:
            params: 原始参数字典
            
        Returns:
            Dict: 验证后的参数字典
            
        Raises:
            ValueError: 参数验证失败时抛出
        """
        # 必需参数及其合理范围
        required_param_ranges = {
            'gamma': (0.001, 10.0),
            'theta': (0.001, 10.0), 
            'm0': (0.0001, 1),
            'beta_prime': (10.0, 100.0)
        }
        
        # 可选参数及其合理范围
        optional_param_ranges = {
            'cq_value': (0.1, 1000.0),
            'mc_value': (0.1, 100.0)
        }
        
        validated_params = {}
        
        # 验证必需参数
        for param_name, (min_val, max_val) in required_param_ranges.items():
            if param_name not in params:
                raise ValueError(f"缺少必需参数: {param_name}")
            
            value = float(params[param_name])
            
            if not (min_val <= value <= max_val):
                print(f"警告: 参数 {param_name}={value} 超出建议范围 [{min_val}, {max_val}]")
            
            validated_params[param_name] = value
        
        # 验证可选参数
        for param_name, (min_val, max_val) in optional_param_ranges.items():
            if param_name in params:
                value = float(params[param_name])
                
                if not (min_val <= value <= max_val):
                    print(f"警告: 参数 {param_name}={value} 超出建议范围 [{min_val}, {max_val}]")
                
                validated_params[param_name] = value
        
        # 处理其他可选参数
        validated_params['trial_number'] = params.get('trial_number', 0)
        
        return validated_params
        
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
        import ast
        for _, row in df.iterrows():
            values_data = row['values']
            if isinstance(values_data, str):
                # 自动把字符串"[176.0, 4.45859]"转成list
                values_data = ast.literal_eval(values_data)
            if isinstance(values_data, (list, tuple)):
                cq_value = float(values_data[0])
                mc_value = float(values_data[1])
            elif isinstance(values_data, float) or isinstance(values_data, int):
                cq_value = float(values_data)
                mc_value = float('nan')
            else:
                raise ValueError(f"未知的数据类型: {type(values_data)}")

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
        if len(pareto_points) > 0:
            n_show = min(2, len(pareto_points))
            print("前{}个点的参数:".format(n_show))
            for idx, pt in enumerate(pareto_points[:n_show]):
                print(f" trial_number={pt.trial_number}, CQ={pt.cq_value}, MC={pt.mc_value}, gamma={pt.gamma}, theta={pt.theta}, m0={pt.m0}, beta_prime={pt.beta_prime}")
        return pareto_points
    
    def load_pareto_dict(self, param_dict: Union[Dict, List[Dict]]) -> List[ParetoPointParams]:
        """
        从字典或字典列表加载Pareto点参数
        
        Args:
            param_dict: 单个参数字典或参数字典列表
                       每个字典必须包含: gamma, theta, m0, beta_prime
                       可选字段: cq_value, mc_value, trial_number (默认为索引)
        
        Returns:
            List[ParetoPointParams]: Pareto点参数列表
        """
        # 如果输入是单个字典，转换为列表
        if isinstance(param_dict, dict):
            param_list = [param_dict]
        elif isinstance(param_dict, list):
            param_list = param_dict
        else:
            raise TypeError("param_dict必须是字典或字典列表")
        
        pareto_points = []
        for i, params in enumerate(param_list):
            # 验证并标准化参数
            validated_params = self.validate_param_dict(params)
            
            # 创建ParetoPointParams对象
            pareto_point = ParetoPointParams(
                trial_number=validated_params.get('trial_number', i),  # 如果没有提供trial_number，使用索引
                gamma=validated_params['gamma'],
                theta=validated_params['theta'],
                m0=validated_params['m0'],
                beta_prime=validated_params['beta_prime'],
                cq_value=validated_params.get('cq_value'),  # 可选参数，可能为None
                mc_value=validated_params.get('mc_value')   # 可选参数，可能为None
            )
            pareto_points.append(pareto_point)
        
        print(f"成功从字典加载{len(pareto_points)}个点")

        return pareto_points
    
    def load_parameters(self, source: Union[str, Dict, List[Dict], ParameterSource]) -> List[ParetoPointParams]:
        """
        灵活的参数加载接口，支持多种输入格式
        
        Args:
            source: 参数来源，可以是：
                   - str: CSV文件名
                   - Dict: 单个参数字典
                   - List[Dict]: 参数字典列表
                   - ParameterSource: 参数源配置对象
        
        Returns:
            List[ParetoPointParams]: 加载的Pareto点参数列表
        """
        if isinstance(source, str):
            # 字符串输入，假设是CSV文件名
            return self.load_pareto_csv(source)
        
        elif isinstance(source, (dict, list)):
            # 字典或字典列表输入
            return self.load_pareto_dict(source)
        
        elif isinstance(source, ParameterSource):
            # ParameterSource对象输入
            if source.source_type == 'csv':
                return self.load_pareto_csv(source.data)
            elif source.source_type == 'dict':
                return self.load_pareto_dict(source.data)
            else:
                raise ValueError(f"不支持的参数源类型: {source.source_type}")
        
        else:
            raise TypeError(f"不支持的输入类型: {type(source)}")
    
    def create_reservoir_params(self, pareto_point: ParetoPointParams) -> ReservoirParams:
        """根据Pareto点参数创建储层参数"""
        return ReservoirParams(
            h=0.4, m0 = pareto_point.m0, Nvirt=200, beta_prime=pareto_point.beta_prime,
            params={'theta': pareto_point.theta, 'gamma': pareto_point.gamma, 'Nvirt': 200})

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
    
    def evaluate_narma10(self, reservoir_params: ReservoirParams, 
                        Ntrain: int = 2000, Ntest: int = 1000) -> Tuple[float, np.ndarray, np.ndarray]:
        """评估NARMA-10任务"""

        # 创建储层
        spn = spnc_anisotropy(
            h=reservoir_params.h,
            theta_H=reservoir_params.theta_H,
            k_s=reservoir_params.k_s_0,
            phi=reservoir_params.phi,
            beta_prime=reservoir_params.beta_prime,
            restart=True
        )

        transform = spn.gen_signal_slow_delayed_feedback
        
        # 运行NARMA-10任务
        (y_test, pred) = spnc_narma10(
            Ntrain,
            Ntest,
            reservoir_params.Nvirt,
            reservoir_params.m0,
            reservoir_params.bias,
            transform,
            reservoir_params.params,
            seed_NARMA=1234,
            fixed_mask=True,
            seed_mask=1234,
            return_outputs=True,
        )
        
        # 计算NRMSE
        nrmse = NRMSE(pred, y_test)
        
        return nrmse, y_test, pred
    
    def evaluate_ti46(self, reservoir_params: ReservoirParams, 
                     speakers: Optional[List[str]] = None) -> float:
        """评估TI46任务"""
        
        # 创建储层
        spn = spnc_anisotropy(
            h=reservoir_params.h,
            theta_H=reservoir_params.theta_H,
            k_s=reservoir_params.k_s_0,
            phi=reservoir_params.phi,
            beta_prime=reservoir_params.beta_prime,
            restart=True
        )

        transform = spn.gen_signal_slow_delayed_feedback
        
        # 运行TI46任务
        accuracy = spnc_TI46(
            speakers,
            reservoir_params.Nvirt,
            reservoir_params.m0,
            reservoir_params.bias,
            transform,
            reservoir_params.params)
        
        return accuracy
    
    def evaluate_KRandGR_threshold(self, reservoir_params: ReservoirParams, threshold: float = 0.1) -> float:
        
        results = evaluate_KRandGR(reservoir_params, threshold=threshold)
        return results['KR'], results['GR']
    
    def evaluate_single_point(self, pareto_point: ParetoPointParams, 
                            narma_config: Dict = None, ti46_config: Dict = None) -> TaskResults:
        """评估单个Pareto点"""
        
        # 设置默认配置
        if narma_config is None:
            narma_config = {'Ntrain': 2000, 'Ntest': 1000}
        if ti46_config is None:
            ti46_config = ['f1', 'f2', 'f3', 'f4', 'f5'] 
        
        print(f"评估Trial {pareto_point.trial_number}: gamma={pareto_point.gamma}, "
              f"theta={pareto_point.theta}, m0={pareto_point.m0}, "
              f"beta_prime={pareto_point.beta_prime}")
        
        # 创建储层参数和储层
        reservoir_params = self.create_reservoir_params(pareto_point)
        
        # 评估NARMA-10
        print("  评估NARMA-10...")
        narma10_nrmse, narma10_y_test, narma10_pred = self.evaluate_narma10(
        reservoir_params, **narma_config
        )
        print(f"  NARMA-10 NRMSE: {narma10_nrmse:.4f}")
        
        # 评估TI46
        print("  评估TI46...")
        ti46_accuracy = self.evaluate_ti46(reservoir_params, speakers=ti46_config)
        print(f"  TI46 Accuracy: {ti46_accuracy:.4f}")

        # 评估KR和GR阈值
        print("  评估KR和GR阈值...")
        kr, gr = self.evaluate_KRandGR_threshold(reservoir_params, threshold=0.1)
        print(f"  KR: {kr}, GR: {gr}")
        
        return TaskResults(
            trial_number=pareto_point.trial_number,
            gamma=pareto_point.gamma,
            theta=pareto_point.theta,
            m0=pareto_point.m0,
            beta_prime=pareto_point.beta_prime,
            narma10_nrmse=narma10_nrmse,
            narma10_y_test=narma10_y_test,
            narma10_pred=narma10_pred,
            ti46_accuracy=ti46_accuracy,
            kr_threshold=kr,
            gr_threshold=gr
        )
    
    def evaluate_all_points(self, source: Union[str, Dict, List[Dict], ParameterSource], 
                          narma_config: Dict = None, ti46_config: Dict = None,
                          output_filename: str = None, 
                          trial_numbers: Union[int, List[int]] = None) -> List[TaskResults]:
        """
        评估所有Pareto点或指定的特定trial
        
        Args:
            source: 参数来源，支持多种格式（CSV文件名、字典、字典列表等）
            narma_config: NARMA-10任务配置
            ti46_config: TI46任务配置  
            output_filename: 输出文件名
            trial_numbers: 指定要评估的trial编号，可以是单个数字或数字列表。如果为None则评估所有
        
        Returns:
            List[TaskResults]: 评估结果列表
        """
        
        # 使用灵活的参数加载接口
        pareto_points = self.load_parameters(source)
        
        # 过滤指定的trial
        if trial_numbers is not None:
            # 转换为列表格式
            if isinstance(trial_numbers, int):
                target_trials = [trial_numbers]
            else:
                target_trials = trial_numbers
            
            # 过滤出指定的trial
            filtered_points = [point for point in pareto_points 
                             if point.trial_number in target_trials]
            
            # 检查是否找到了所有指定的trial
            found_trials = [point.trial_number for point in filtered_points]
            missing_trials = [t for t in target_trials if t not in found_trials]
            
            if missing_trials:
                print(f"警告: 未找到以下trial编号: {missing_trials}")
            
            if not filtered_points:
                print(f"错误: 未找到任何指定的trial编号 {target_trials}")
                return []
            
            pareto_points = filtered_points
            print(f"将评估指定的 {len(pareto_points)} 个trial: {found_trials}")
        else:
            print(f"将评估所有 {len(pareto_points)} 个trial")
        
        # 评估所有点
        all_results = []
        for i, point in enumerate(pareto_points):
            print(f"\n进度: {i+1}/{len(pareto_points)} (Trial {point.trial_number})")
            try:
                result = self.evaluate_single_point(point, narma_config, ti46_config)
                all_results.append(result)
            except Exception as e:
                print(f"  错误：评估Trial {point.trial_number}时出现异常: {e}")
                continue
        
        # 保存结果
        if output_filename is None:
            # 根据输入类型生成输出文件名
            if isinstance(source, str):
                base_name = Path(source).stem
                if trial_numbers is not None:
                    trials_str = "_".join(map(str, target_trials)) if len(target_trials) <= 5 else f"{len(target_trials)}trials"
                    output_filename = f"ParetoFront_TaskResults_{base_name}_trials_{trials_str}"
                else:
                    output_filename = f"ParetoFront_TaskResults_{base_name}"
            else:
                if trial_numbers is not None:
                    trials_str = "_".join(map(str, target_trials)) if len(target_trials) <= 5 else f"{len(target_trials)}trials"
                    output_filename = f"ParetoFront_TaskResults_dict_trials_{trials_str}"
                else:
                    output_filename = f"ParetoFront_TaskResults_dict_input"
        
        self.save_results(all_results, output_filename)
        
        return all_results
    
    def evaluate_single_dict_point(self, param_dict: Dict, 
                                 narma_config: Dict = None, ti46_config: Dict = None) -> TaskResults:
        """
        便捷方法：直接从单个参数字典评估储层性能
        
        Args:
            param_dict: 包含储层参数的字典
            narma_config: NARMA-10任务配置
            ti46_config: TI46任务配置
        
        Returns:
            TaskResults: 评估结果
        """
        # 从字典创建ParetoPointParams
        pareto_points = self.load_pareto_dict(param_dict)
        
        # 评估单个点
        return self.evaluate_single_point(pareto_points[0], narma_config, ti46_config)
    
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
                'narma10_nrmse': getattr(result, 'narma10_nrmse', None),
                'ti46_accuracy': getattr(result, 'ti46_accuracy', None),
                'kr_threshold': getattr(result, 'kr_threshold', None),
                'gr_threshold': getattr(result, 'gr_threshold', None)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        print(f"汇总结果已保存至: {csv_file}")
        
        # 保存详细的NARMA-10数据（只保存有效数据）
        narma_data_file = output_dir / f"{output_filename}_narma10_detailed.pkl"
        narma_data = {}
        for result in results:
            # 只保存有NARMA-10结果的数据
            if (hasattr(result, 'narma10_y_test') and result.narma10_y_test is not None and
                hasattr(result, 'narma10_pred') and result.narma10_pred is not None and
                hasattr(result, 'narma10_nrmse') and result.narma10_nrmse is not None):
                narma_data[result.trial_number] = {
                    'y_test': result.narma10_y_test,
                    'pred': result.narma10_pred,
                    'nrmse': result.narma10_nrmse
                }
        
        with open(narma_data_file, 'wb') as f:
            pickle.dump(narma_data, f)
        print(f"NARMA-10详细数据已保存至: {narma_data_file} (共{len(narma_data)}个有效结果)")

def main(filename):
    """主函数示例，展示各种使用方法"""
    
    # 创建评估器
    evaluator = ParetoPointEvaluator()
    
    # 配置参数
    narma_config = {
        'Ntrain': 2000,
        'Ntest': 1000
    }
    
    ti46_config = ['f1', 'f2', 'f3', 'f4', 'f5']  # 使用所有说话者
    
    print("=" * 60)
    print("示例1: 从CSV文件评估（原有功能）")
    print("=" * 60)
    
    # filename = "Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049_pareto.csv"
    
    try:
        results_csv = evaluator.evaluate_all_points(
            source=filename,
            narma_config=narma_config,
            ti46_config=ti46_config,
            trial_numbers=[155,89,144,125,205,130],
            output_filename="uniform_chaos_task_results"
        )
        print(f"CSV评估完成！共处理{len(results_csv)}个Pareto点")
    except FileNotFoundError:
        print(f"未找到CSV文件，跳过CSV评估示例")
    
    
    # # 定义多个储层参数
    # param_list = [
    #     {
    #         'gamma': 0.10,
    #         'theta': 0.25,
    #         'm0': 0.002,
    #         'beta_prime': 25.0,
    #         'cq_value': 4.0,
    #         'mc_value': 6.5
    #     },
    #     {
    #         'gamma': 0.20,
    #         'theta': 0.35,
    #         'm0': 0.004,
    #         'beta_prime': 35.0,
    #         'cq_value': 6.0,
    #         'mc_value': 5.5
    #     }
    # ]
    
    # # 评估多个参数点
    # multi_results = evaluator.evaluate_all_points(
    #     source=param_list,
    #     narma_config=narma_config,
    #     ti46_config=ti46_config,
    #     output_filename="custom_dict_evaluation"
    # )
    
    # print(f"多点评估完成！共处理{len(multi_results)}个自定义参数点")
    


if __name__ == "__main__":
    main("Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049_trials.csv")
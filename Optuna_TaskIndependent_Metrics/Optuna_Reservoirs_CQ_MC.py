#!/usr/bin/env python3
"""
基于Pareto Front数据的储层形貌性能评估
=========================================

从ParetoFront_CQandMC/data/pareto_front.csv中提取每个Pareto点的参数，
对同一组参数创建三种不同形貌的储层（均质、渐变、随机），
并计算它们的MC和KRandGR性能指标。

Author: Chen  
Date: 2025-01-XX
"""

import numpy as np
import pandas as pd
import os
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
from reservoir_morphology_creator import (
    MorphologyConfig, 
    ReservoirMorphologyManager
)
from reservoir_evaluation import (
    evaluate_heterogeneous_MC,
    evaluate_heterogeneous_KRandGR
)
from formal_Parameter_Dynamics_Preformance import ReservoirParams, evaluate_MC, evaluate_KRandGR


class ReservoirsParetoEvaluator:
    """基于Pareto Front数据的储层形貌评估器"""
    
    def __init__(self, pareto_csv_path="ParetoFront_CQandMC/data/pareto_front.csv"):
        """初始化评估器"""
        self.pareto_csv_path = pareto_csv_path
        self.pareto_data = None
        self.morphology_manager = ReservoirMorphologyManager()
        self.results = []
        
        print("🚀 基于Pareto Front的储层形貌评估器已初始化")
        print(f"📁 Pareto数据路径: {self.pareto_csv_path}")
    
    def load_pareto_data(self):
        """加载Pareto Front数据"""
        try:
            self.pareto_data = pd.read_csv(self.pareto_csv_path)
            print(f"✅ 成功加载Pareto Front数据")
            print(f"📊 数据包含 {len(self.pareto_data)} 个参数组合")
            print(f"📋 列名: {list(self.pareto_data.columns)}")
            return True
        except Exception as e:
            print(f"❌ 加载Pareto Front数据失败: {e}")
            return False
    
    def extract_parameters(self, row_index):
        """从指定行提取参数"""
        if self.pareto_data is None:
            raise ValueError("请先加载Pareto Front数据")
        
        if row_index >= len(self.pareto_data):
            raise IndexError(f"行索引 {row_index} 超出数据范围 (0-{len(self.pareto_data)-1})")
        
        row = self.pareto_data.iloc[row_index]
        
        params = {
            'number': int(row['number']),
            'CQ': float(row['CQ']),
            'MC': float(row['MC']),
            'gamma': float(row['gamma']),
            'theta': float(row['theta']),
            'm0': float(row['m0']),
            'h': float(row['h']),
            'beta_prime': float(row['beta_prime']),
            'Nvirt': int(row['Nvirt'])
        }
        
        return params
    
    def create_morphology_configs(self, n_instances: int = 5) -> Dict[str, MorphologyConfig]:
        """创建三种形貌配置"""
        configs = {
            'homogeneous': MorphologyConfig(
                morph_type='homogeneous'
            ),
            'gradient': MorphologyConfig(
                morph_type='gradient',
                n_instances=n_instances,
                beta_range=(20, 40),  # 默认范围，会根据beta_prime调整
                distribution_type='linear'
            ),
            'random': MorphologyConfig(
                morph_type='random',
                n_instances=n_instances,
                beta_range=(20, 40),  # 默认范围，会根据beta_prime调整
                random_seed=1234
            )
        }
        return configs
    
    def adjust_beta_range_for_configs(self, configs: Dict[str, MorphologyConfig], 
                                    beta_prime: float, range_ratio: float = 0.2):
        """根据beta_prime调整异质储层的beta范围"""
        # 计算合理的beta范围（以beta_prime为中心，按比例扩展）
        range_span = beta_prime * range_ratio
        beta_min = max(1.0, beta_prime - range_span)  # 确保最小值大于0
        beta_max = beta_prime + range_span
        
        for morph_name, config in configs.items():
            if config.morph_type != 'homogeneous':
                config.beta_range = (beta_min, beta_max)
        
        return configs
    
    def create_reservoir_params_from_pareto(self, pareto_params: Dict) -> ReservoirParams:
        """从Pareto参数创建ReservoirParams对象"""
        return ReservoirParams(
            h=pareto_params['h'],
            m0=pareto_params['m0'],
            Nvirt=pareto_params['Nvirt'],
            beta_prime=pareto_params['beta_prime'],
            params={
                "gamma": pareto_params['gamma'],
                "theta": pareto_params['theta'],
                "Nvirt": pareto_params['Nvirt'],
                "delay_feedback": 0,
                "length_warmup": 0,
                "warmup_sample": 0,
                "voltage_noise": False,
                "johnson_noise": False,
                "thermal_noise": False,
            }
        )
    
    def evaluate_single_morphology(self, morph_name: str, config: MorphologyConfig, 
                                 reservoir_params: ReservoirParams, pareto_number: int) -> Dict:
        """评估单个形貌的储层性能"""
        print(f"    📊 评估 {morph_name} 形貌...")
        
        start_time = time.time()
        
        try:
            # 创建储层
            reservoir = self.morphology_manager.create_reservoir(config, reservoir_params)
            
            # 根据形貌类型选择评估函数
            if config.morph_type == 'homogeneous':
                # 均质储层使用标准评估函数
                mc_dict = evaluate_MC(reservoir_params)
                kgr_dict = evaluate_KRandGR(reservoir_params)
            else:
                # 异质储层使用专门的评估函数
                hetero_params = ReservoirParams(
                    h=reservoir_params.h,
                    m0=reservoir_params.m0,
                    Nvirt=reservoir_params.Nvirt,
                    beta_prime=reservoir_params.beta_prime,
                    params={
                        "gamma": reservoir_params.params["gamma"],
                        "theta": reservoir_params.params["theta"],
                        "Nvirt": reservoir_params.Nvirt,
                        "delay_feedback": 0,
                        "length_warmup": 0,
                        "warmup_sample": 0,
                    }
                )
                
                mc_dict = evaluate_heterogeneous_MC(hetero_params, config)
                kgr_dict = evaluate_heterogeneous_KRandGR(hetero_params, config, Nwash=7)
            
            # 提取指标
            MC = float(mc_dict.get("MC", 0.0))
            KR = float(kgr_dict.get("KR", 0.0))
            GR = float(kgr_dict.get("GR", 0.0))
            CQ = KR - GR
            
            calculation_time = time.time() - start_time
            
            print(f"      ✅ {morph_name}: MC={MC:.4f}, KR={KR:.1f}, GR={GR:.1f}, CQ={CQ:.4f}")
            
            # 构建结果字典
            result = {
                'pareto_number': pareto_number,
                'morphology': morph_name,
                'morphology_type': config.morph_type,
                'n_instances': getattr(config, 'n_instances', 1),
                'beta_range_min': config.beta_range[0] if hasattr(config, 'beta_range') and config.beta_range else reservoir_params.beta_prime,
                'beta_range_max': config.beta_range[1] if hasattr(config, 'beta_range') and config.beta_range else reservoir_params.beta_prime,
                'distribution_type': getattr(config, 'distribution_type', 'fixed'),
                'random_seed': getattr(config, 'random_seed', None),
                
                # Pareto参数
                'gamma': reservoir_params.params['gamma'],
                'theta': reservoir_params.params['theta'],
                'm0': reservoir_params.m0,
                'h': reservoir_params.h,
                'beta_prime': reservoir_params.beta_prime,
                'Nvirt': reservoir_params.Nvirt,
                
                # 性能指标
                'MC': MC,
                'KR': KR,
                'GR': GR,
                'CQ': CQ,
                
                # 元数据
                'calculation_time': calculation_time,
                'success': True,
                'error': None
            }
            
            return result
            
        except Exception as e:
            calculation_time = time.time() - start_time
            print(f"      ❌ {morph_name} 评估失败: {e}")
            
            # 构建失败结果字典
            result = {
                'pareto_number': pareto_number,
                'morphology': morph_name,
                'morphology_type': config.morph_type,
                'n_instances': getattr(config, 'n_instances', 1),
                'beta_range_min': config.beta_range[0] if hasattr(config, 'beta_range') and config.beta_range else reservoir_params.beta_prime,
                'beta_range_max': config.beta_range[1] if hasattr(config, 'beta_range') and config.beta_range else reservoir_params.beta_prime,
                'distribution_type': getattr(config, 'distribution_type', 'fixed'),
                'random_seed': getattr(config, 'random_seed', None),
                
                # Pareto参数
                'gamma': reservoir_params.params['gamma'],
                'theta': reservoir_params.params['theta'],
                'm0': reservoir_params.m0,
                'h': reservoir_params.h,
                'beta_prime': reservoir_params.beta_prime,
                'Nvirt': reservoir_params.Nvirt,
                
                # 性能指标（失败时设为0）
                'MC': 0.0,
                'KR': 0.0,
                'GR': 0.0,
                'CQ': 0.0,
                
                # 元数据
                'calculation_time': calculation_time,
                'success': False,
                'error': str(e)
            }
            
            return result
    
    def evaluate_pareto_point(self, pareto_index: int, n_instances: int = 5) -> List[Dict]:
        """评估单个Pareto点的所有形貌"""
        # 提取Pareto参数
        pareto_params = self.extract_parameters(pareto_index)
        pareto_number = pareto_params['number']
        
        print(f"\n🎯 评估Pareto点 #{pareto_number} (索引: {pareto_index})")
        print(f"    参数: gamma={pareto_params['gamma']}, theta={pareto_params['theta']}, "
              f"beta_prime={pareto_params['beta_prime']}, Nvirt={pareto_params['Nvirt']}, h={pareto_params['h']}, m0={pareto_params['m0']}")
        
        # 创建ReservoirParams对象
        reservoir_params = self.create_reservoir_params_from_pareto(pareto_params)
        
        # 创建形貌配置并调整beta范围
        configs = self.create_morphology_configs(n_instances)
        configs = self.adjust_beta_range_for_configs(configs, pareto_params['beta_prime'])
        
        # 评估每种形貌
        point_results = []
        for morph_name, config in configs.items():
            result = self.evaluate_single_morphology(morph_name, config, reservoir_params, pareto_number)
            point_results.append(result)
        
        return point_results
    
    def evaluate_all_pareto_points(self, start_index: int = 0, end_index: Optional[int] = None, 
                                 n_instances: int = 5) -> List[Dict]:
        """评估所有或指定范围的Pareto点"""
        if self.pareto_data is None:
            raise ValueError("请先加载Pareto Front数据")
        
        # 确定评估范围
        total_points = len(self.pareto_data)
        if end_index is None:
            end_index = total_points
        end_index = min(end_index, total_points)
        
        print(f"\n🚀 开始评估Pareto点 {start_index} 到 {end_index-1} (共 {end_index-start_index} 个点)")
        print(f"📊 每个点评估3种形貌，总共 {(end_index-start_index)*3} 次评估")
        
        all_results = []
        start_time = time.time()
        
        for i in range(start_index, end_index):
            try:
                print(f"\n📍 进度: [{i-start_index+1}/{end_index-start_index}]")
                point_results = self.evaluate_pareto_point(i, n_instances)
                all_results.extend(point_results)
                
                # 显示这个点的成功率
                success_count = sum(1 for r in point_results if r['success'])
                print(f"    ✅ 完成 {success_count}/3 个形貌评估")
                
            except Exception as e:
                print(f"    ❌ Pareto点 {i} 评估失败: {e}")
                continue
        
        total_time = time.time() - start_time
        successful_evaluations = sum(1 for r in all_results if r['success'])
        total_evaluations = len(all_results)
        
        print(f"\n🎉 评估完成!")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"📊 成功率: {successful_evaluations}/{total_evaluations} ({successful_evaluations/total_evaluations*100:.1f}%)")
        
        self.results = all_results
        return all_results
    
    def save_results(self, results: Optional[List[Dict]] = None, 
                    filename: Optional[str] = None) -> str:
        """保存结果到CSV文件"""
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("没有结果数据可保存")
        
        # 生成文件名
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Reservoirs_Pareto_Morphology_Evaluation_{timestamp}.csv"
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        
        print(f"\n💾 结果已保存到: {filename}")
        print(f"📊 保存了 {len(df)} 条记录")
        
        # 显示基本统计
        if len(df) > 0:
            successful_df = df[df['success'] == True]
            print(f"✅ 成功评估: {len(successful_df)} 条")
            
            if len(successful_df) > 0:
                print(f"📈 MC范围: {successful_df['MC'].min():.4f} - {successful_df['MC'].max():.4f}")
                print(f"📊 CQ范围: {successful_df['CQ'].min():.4f} - {successful_df['CQ'].max():.4f}")
                
                # 按形貌类型统计
                morph_stats = successful_df.groupby('morphology').agg({
                    'MC': ['count', 'mean', 'std'],
                    'CQ': ['mean', 'std']
                }).round(4)
                
                print(f"\n📋 各形貌统计:")
                print(morph_stats)
        
        return filename
    
    def print_summary_statistics(self, results: Optional[List[Dict]] = None):
        """打印结果摘要统计"""
        if results is None:
            results = self.results
        
        if not results:
            print("没有结果数据可分析")
            return
        
        df = pd.DataFrame(results)
        successful_df = df[df['success'] == True]
        
        print(f"\n" + "="*60)
        print("评估结果摘要统计")
        print("="*60)
        
        print(f"总评估次数: {len(df)}")
        print(f"成功评估: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)")
        
        if len(successful_df) > 0:
            print(f"\n性能指标统计:")
            print(f"MC (内存容量):")
            print(f"  平均值: {successful_df['MC'].mean():.4f}")
            print(f"  标准差: {successful_df['MC'].std():.4f}")
            print(f"  范围: {successful_df['MC'].min():.4f} - {successful_df['MC'].max():.4f}")
            
            print(f"CQ (计算质量):")
            print(f"  平均值: {successful_df['CQ'].mean():.4f}")
            print(f"  标准差: {successful_df['CQ'].std():.4f}")
            print(f"  范围: {successful_df['CQ'].min():.4f} - {successful_df['CQ'].max():.4f}")
            
            print(f"\n各形貌表现:")
            morph_summary = successful_df.groupby('morphology').agg({
                'MC': ['count', 'mean', 'std'],
                'CQ': ['mean', 'std'],
                'calculation_time': 'mean'
            }).round(4)
            
            print(morph_summary)


def run_full_evaluation(start_index: int = 0, end_index: Optional[int] = None, 
                       n_instances: int = 5):
    """运行完整的评估流程"""
    print("🚀 开始基于Pareto Front的储层形貌性能评估")
    
    # 创建评估器
    evaluator = ReservoirsParetoEvaluator()
    
    # 加载数据
    if not evaluator.load_pareto_data():
        return None
    
    # 运行评估
    results = evaluator.evaluate_all_pareto_points(start_index, end_index, n_instances)
    
    # 保存结果
    filename = evaluator.save_results(results)
    
    # 显示统计
    evaluator.print_summary_statistics(results)
    
    return evaluator, results, filename


def run_sample_evaluation(n_samples: int = 5, n_instances: int = 3):
    """运行样本评估（测试用）"""
    print(f"🔬 运行样本评估 (前{n_samples}个Pareto点)")
    
    return run_full_evaluation(start_index=0, end_index=n_samples, n_instances=n_instances)


def main():
    """主函数"""
    print("="*80)
    print("基于Pareto Front数据的储层形貌性能评估")
    print("="*80)
    
    # 用户选择运行模式
    print("\n请选择运行模式:")
    print("1. 样本评估 (前5个Pareto点，用于测试)")
    print("2. 完整评估 (所有Pareto点)")
    print("3. 自定义范围评估")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    try:
        if choice == '1':
            evaluator, results, filename = run_sample_evaluation()
        elif choice == '2':
            evaluator, results, filename = run_full_evaluation()
        elif choice == '3':
            start = int(input("起始索引: "))
            end = int(input("结束索引 (不包含): "))
            instances = int(input("异质储层实例数 (默认5): ") or "5")
            evaluator, results, filename = run_full_evaluation(start, end, instances)
        else:
            print("无效选择，运行样本评估")
            evaluator, results, filename = run_sample_evaluation()
        
        print(f"\n🎉 评估完成!")
        print(f"📁 结果文件: {filename}")
        
        return evaluator, results, filename
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断评估")
        return None, None, None
    except Exception as e:
        print(f"\n❌ 评估过程出现错误: {e}")
        return None, None, None


if __name__ == "__main__":
    evaluator, results, filename = main()
#!/usr/bin/env python3
"""
储层形貌CQ和MC计算测试脚本
==========================================

专门用于测试和对比三种储层形貌（均质、渐变、随机）的：
- CQ (Computational Quality): 计算质量 = KR - GR
- MC (Memory Capacity): 内存容量

Author: Chen
Date: 2025-01-XX
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
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


class MorphologyMetricsCalculator:
    """储层形貌性能指标计算器"""
    
    def __init__(self):
        self.manager = ReservoirMorphologyManager()
        self.results = {}
    
    def print_morphology_config(self, morph_name: str, config: MorphologyConfig):
        """打印形貌配置信息"""
        print(f"    📋 {morph_name} 储层配置:")
        print(f"      - 形貌类型: {config.morph_type}")
        
        if config.morph_type != 'homogeneous':
            print(f"      - 实例数量: {config.n_instances}")
            print(f"      - Beta范围: {config.beta_range}")
            print(f"      - 分布类型: {config.distribution_type}")
            if config.random_seed is not None:
                print(f"      - 随机种子: {config.random_seed}")
    
    def print_reservoir_details(self, morph_name: str, config: MorphologyConfig, 
                               reservoir_params: ReservoirParams, reservoir=None):
        """打印储层详细信息"""
        print(f"    🔧 {morph_name} 储层详细信息:")
        
        # 基础参数
        print(f"      储层参数:")
        print(f"        - h (外部磁场): {reservoir_params.h}")
        print(f"        - m0 (初始磁化): {reservoir_params.m0}")
        print(f"        - Nvirt (虚拟节点): {reservoir_params.Nvirt}")
        print(f"        - beta_prime (基准温度): {reservoir_params.beta_prime}")
        
        # 系统参数
        print(f"      系统参数:")
        for key, value in reservoir_params.params.items():
            if key in ['gamma', 'theta', 'delay_feedback']:
                print(f"        - {key}: {value}")
        
        # 形貌特定信息
        if config.morph_type == 'homogeneous':
            print(f"      形貌特性:")
            print(f"        - 储层类型: 单一均质储层")
            print(f"        - Beta值: {reservoir_params.beta_prime} (固定)")
        else:
            if reservoir is not None:
                print(f"      异质储层特性:")
                print(f"        - 实例数量: {len(reservoir.anisotropy_instances)}")
                
                # 生成并显示beta分布
                deltabeta_list = self.manager.generate_deltabeta_list(config, reservoir_params.beta_prime)
                actual_betas = [reservoir_params.beta_prime + delta for delta in deltabeta_list]
                
                print(f"        - Beta分布统计:")
                print(f"          * 最小值: {min(actual_betas):.3f}")
                print(f"          * 最大值: {max(actual_betas):.3f}")
                print(f"          * 平均值: {np.mean(actual_betas):.3f}")
                print(f"          * 标准差: {np.std(actual_betas):.3f}")
                
                # 显示前几个beta值作为示例
                print(f"        - Beta值示例 (前5个):")
                for i, beta in enumerate(actual_betas[:5]):
                    print(f"          * 实例{i+1}: {beta:.3f}")
                
                # 权重信息
                weights = self.manager.generate_weights(reservoir, config)
                print(f"        - 权重信息:")
                print(f"          * 权重数量: {len(weights)}")
                print(f"          * 权重和: {sum(weights):.6f}")
                print(f"          * 单个权重: {weights[0]:.6f} (均匀分布)")
    
    def print_evaluation_summary(self, morph_name: str, config: MorphologyConfig, 
                                reservoir_params: ReservoirParams, signals_info: Optional[Dict] = None):
        """打印评估设置摘要"""
        print(f"    📊 {morph_name} 评估设置:")
        
        if signals_info:
            print(f"      MC信号设置:")
            print(f"        - 信号长度: {signals_info.get('mc_signal_len', 550)}")
            print(f"        - 随机种子: {signals_info.get('mc_seed', 1234)}")
            print(f"        - 分割比例: {signals_info.get('mc_splits', [0.2, 0.6])}")
            print(f"        - 延迟数量: {signals_info.get('mc_delays', 10)}")
            
            print(f"      KR&GR设置:")
            print(f"        - 读出数量: {reservoir_params.Nvirt}")
            print(f"        - 冲洗参数: {signals_info.get('krgr_nwash', 7)}")
            print(f"        - 随机种子: {signals_info.get('krgr_seed', 1234)}")
            print(f"        - SVD阈值: {signals_info.get('krgr_threshold', 0.1)}")
        
    def create_test_configurations(self, n_instances: int = 5, beta_range: Tuple[float, float] = (20, 30)) -> Dict[str, MorphologyConfig]:
        """创建测试配置"""
        configs = {
            'homogeneous': MorphologyConfig(
                morph_type='homogeneous'
            ),
            'gradient': MorphologyConfig(
                morph_type='gradient',
                n_instances=n_instances,
                beta_range=beta_range,
                distribution_type='linear'
            ),
            'random': MorphologyConfig(
                morph_type='random',
                n_instances=n_instances,
                beta_range=beta_range,
                random_seed=1234
            )
        }
        return configs
    
    def create_test_parameters(self, gamma: float = 0.1, theta: float = 0.5, 
                             m0: float = 0.003, h: float = 0.4, 
                             Nvirt: int = 30, beta_prime: float = 25.0) -> ReservoirParams:
        """创建测试参数"""
        return ReservoirParams(
            h=h,
            m0=m0,
            Nvirt=Nvirt,
            beta_prime=beta_prime,
            params={
                "gamma": gamma,
                "theta": theta,
                "Nvirt": Nvirt,
                "delay_feedback": 0,
                "length_warmup": 0,
                "warmup_sample": 0,
                "voltage_noise": False,
                "johnson_noise": False,
                "thermal_noise": False,
            }
        )
    
    def calculate_single_morphology_metrics(self, morph_name: str, config: MorphologyConfig, 
                                         reservoir_params: ReservoirParams) -> Dict:
        """计算单个形貌的CQ和MC指标"""
        print(f"\n{'='*80}")
        print(f"计算 {morph_name.upper()} 形貌储层性能")
        print(f"{'='*80}")
        
        # 打印形貌配置
        self.print_morphology_config(morph_name, config)
        
        start_time = time.time()
        
        try:
            # 创建储层并打印详细信息
            print(f"\n  🔨 创建储层实例...")
            reservoir = self.manager.create_reservoir(config, reservoir_params)
            
            # 打印储层详细信息
            self.print_reservoir_details(morph_name, config, reservoir_params, reservoir)
            
            # 打印评估设置
            signals_info = {
                'mc_signal_len': 550,
                'mc_seed': 1234,
                'mc_splits': [0.2, 0.6],
                'mc_delays': 10,
                'krgr_nwash': 7,
                'krgr_seed': 1234,
                'krgr_threshold': 0.1
            }
            self.print_evaluation_summary(morph_name, config, reservoir_params, signals_info)
            
            print(f"\n  ⚡ 开始性能计算...")
            if config.morph_type == 'homogeneous':
                # 均质储层使用标准评估函数
                print("    📈 使用标准评估函数计算MC...")
                mc_dict = evaluate_MC(reservoir_params)
                print("    📊 使用标准评估函数计算KR&GR...")
                kgr_dict = evaluate_KRandGR(reservoir_params)
            else:
                # 异质储层使用专门的评估函数
                print("    📈 使用异质储层评估函数计算MC...")
                
                # 创建简化版参数用于异质储层评估
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
                print("    📊 使用异质储层评估函数计算KR&GR...")
                kgr_dict = evaluate_heterogeneous_KRandGR(hetero_params, config, Nwash=7)
            
            # 提取指标
            MC = float(mc_dict.get("MC", 0.0))
            KR = float(kgr_dict.get("KR", 0.0))
            GR = float(kgr_dict.get("GR", 0.0))
            CQ = KR - GR
            
            elapsed_time = time.time() - start_time
            
            # 打印详细结果
            print(f"\n  ✅ 计算成功完成!")
            print(f"  ⏱️  计算耗时: {elapsed_time:.2f} 秒")
            print(f"\n  📊 性能指标结果:")
            print(f"    🧠 内存容量 (MC): {MC:.6f}")
            print(f"    🔢 核等级 (KR):   {KR:.1f}")
            print(f"    📈 泛化等级 (GR): {GR:.1f}")
            print(f"    🎯 计算质量 (CQ): {CQ:.6f} (= KR - GR)")
            
            # 简单的性能评估
            print(f"\n  💡 性能评估:")
            if MC > 2.0:
                mc_rating = "优秀" if MC > 3.0 else "良好"
                print(f"    MC {mc_rating}: {MC:.3f} (记忆能力强)")
            else:
                print(f"    MC 一般: {MC:.3f} (记忆能力有限)")
            
            if CQ > 0.5:
                cq_rating = "优秀" if CQ > 1.0 else "良好"
                print(f"    CQ {cq_rating}: {CQ:.3f} (计算性能强)")
            elif CQ > 0:
                print(f"    CQ 一般: {CQ:.3f} (计算性能有限)")
            else:
                print(f"    CQ 较差: {CQ:.3f} (计算性能不佳)")
            
            result = {
                'morphology': morph_name,
                'config': config,
                'MC': MC,
                'KR': KR,
                'GR': GR,
                'CQ': CQ,
                'calculation_time': elapsed_time,
                'success': True,
                'error': None,
                'reservoir_info': {
                    'instance_count': getattr(reservoir, 'anisotropy_instances', None) and len(getattr(reservoir, 'anisotropy_instances', [])) or 1,
                    'beta_range': config.beta_range if config.morph_type != 'homogeneous' else None,
                    'base_beta': reservoir_params.beta_prime
                }
            }
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ✗ 计算失败: {e}")
            
            result = {
                'morphology': morph_name,
                'config': config,
                'MC': 0.0,
                'KR': 0.0,
                'GR': 0.0,
                'CQ': 0.0,
                'calculation_time': elapsed_time,
                'success': False,
                'error': str(e)
            }
            
            return result
    
    def print_test_overview(self, configs: Dict[str, MorphologyConfig], 
                           reservoir_params: ReservoirParams):
        """打印测试总览信息"""
        print("\n" + "🔬" + "="*78 + "🔬")
        print("                   储层形貌CQ和MC性能对比测试")
        print("🔬" + "="*78 + "🔬")
        
        print(f"\n🎯 测试概览:")
        print(f"  📊 形貌数量: {len(configs)} 种")
        print(f"  📋 形貌类型: {', '.join(configs.keys())}")
        
        print(f"\n⚙️  通用储层参数:")
        print(f"  🔧 基础参数:")
        print(f"    - h (外部磁场): {reservoir_params.h}")
        print(f"    - m0 (初始磁化): {reservoir_params.m0}")
        print(f"    - Nvirt (虚拟节点): {reservoir_params.Nvirt}")
        print(f"    - beta_prime (基准温度): {reservoir_params.beta_prime}")
        
        print(f"  🎛️  系统参数:")
        print(f"    - gamma (反馈增益): {reservoir_params.params['gamma']}")
        print(f"    - theta (阈值参数): {reservoir_params.params['theta']}")
        print(f"    - delay_feedback: {reservoir_params.params['delay_feedback']}")
        
        print(f"\n🔍 评估设置:")
        print(f"  📈 MC评估: 信号长度=550, 种子=1234, 分割=[0.2,0.6], 延迟=10")
        print(f"  📊 KR&GR评估: 冲洗参数=7, 种子=1234, SVD阈值=0.1")
        
        print(f"\n🏁 形貌配置详情:")
        for morph_name, config in configs.items():
            print(f"  📋 {morph_name}:")
            print(f"    - 类型: {config.morph_type}")
            if config.morph_type != 'homogeneous':
                print(f"    - 实例数: {config.n_instances}")
                print(f"    - Beta范围: {config.beta_range}")
                print(f"    - 分布: {config.distribution_type}")
                if config.random_seed:
                    print(f"    - 随机种子: {config.random_seed}")
    
    def calculate_all_morphologies(self, configs: Dict[str, MorphologyConfig], 
                                 reservoir_params: ReservoirParams) -> Dict[str, Dict]:
        """计算所有形貌的指标"""
        
        # 显示测试总览
        self.print_test_overview(configs, reservoir_params)
        
        print(f"\n🚀 开始执行性能计算...")
        
        results = {}
        
        for i, (morph_name, config) in enumerate(configs.items(), 1):
            print(f"\n📍 进度: [{i}/{len(configs)}]")
            result = self.calculate_single_morphology_metrics(morph_name, config, reservoir_params)
            results[morph_name] = result
        
        print(f"\n🎉 所有形貌计算完成!")
        
        self.results = results
        return results
    
    def display_comparison_table(self, results: Dict[str, Dict]):
        """显示对比表格"""
        print("\n" + "📊" + "="*78 + "📊")
        print("                         储层形貌性能对比表")
        print("📊" + "="*78 + "📊")
        
        # 创建详细的DataFrame
        data = []
        for morph_name, result in results.items():
            reservoir_info = result.get('reservoir_info', {})
            data.append({
                '形貌类型': morph_name,
                '实例数': reservoir_info.get('instance_count', 1),
                'Beta范围': str(reservoir_info.get('beta_range', f"固定({reservoir_info.get('base_beta', 'N/A')})")),
                'MC (内存容量)': f"{result['MC']:.4f}",
                'KR': f"{result['KR']:.1f}",
                'GR': f"{result['GR']:.1f}",
                'CQ (计算质量)': f"{result['CQ']:.4f}",
                '计算时间(秒)': f"{result['calculation_time']:.1f}",
                '状态': "✅成功" if result['success'] else "❌失败"
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # 添加额外的统计信息
        successful_results = [r for r in results.values() if r['success']]
        if successful_results:
            print(f"\n📈 快速统计:")
            print(f"  成功率: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
            print(f"  平均MC: {np.mean([r['MC'] for r in successful_results]):.4f}")
            print(f"  平均CQ: {np.mean([r['CQ'] for r in successful_results]):.4f}")
            print(f"  总计算时间: {sum([r['calculation_time'] for r in successful_results]):.1f} 秒")
    
    def analyze_morphology_differences(self, results: Dict[str, Dict]):
        """分析形貌差异"""
        print("\n" + "=" * 60)
        print("形貌性能差异分析")
        print("=" * 60)
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if len(successful_results) < 2:
            print("需要至少两个成功的结果才能进行对比分析")
            return
        
        # 找到最佳性能
        best_mc = max(successful_results.values(), key=lambda x: x['MC'])
        best_cq = max(successful_results.values(), key=lambda x: x['CQ'])
        
        print(f"最佳内存容量 (MC): {best_mc['morphology']} = {best_mc['MC']:.4f}")
        print(f"最佳计算质量 (CQ): {best_cq['morphology']} = {best_cq['CQ']:.4f}")
        
        # 计算相对性能
        print("\n相对性能分析 (以均质储层为基准):")
        if 'homogeneous' in successful_results:
            homo_mc = successful_results['homogeneous']['MC']
            homo_cq = successful_results['homogeneous']['CQ']
            
            for morph_name, result in successful_results.items():
                if morph_name != 'homogeneous':
                    mc_improvement = ((result['MC'] - homo_mc) / homo_mc * 100) if homo_mc > 0 else 0
                    cq_improvement = ((result['CQ'] - homo_cq) / homo_cq * 100) if homo_cq > 0 else 0
                    
                    print(f"  {morph_name:12s}: MC改善 {mc_improvement:+.1f}%, CQ改善 {cq_improvement:+.1f}%")
        
        # 性能排名
        print(f"\nMC性能排名:")
        mc_ranking = sorted(successful_results.items(), key=lambda x: x[1]['MC'], reverse=True)
        for i, (morph_name, result) in enumerate(mc_ranking):
            print(f"  {i+1}. {morph_name:12s}: {result['MC']:.4f}")
        
        print(f"\nCQ性能排名:")
        cq_ranking = sorted(successful_results.items(), key=lambda x: x[1]['CQ'], reverse=True)
        for i, (morph_name, result) in enumerate(cq_ranking):
            print(f"  {i+1}. {morph_name:12s}: {result['CQ']:.4f}")
    
    def plot_performance_comparison(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """绘制性能对比图"""
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if len(successful_results) < 2:
            print("数据不足，无法绘制对比图")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        morphologies = list(successful_results.keys())
        mc_values = [successful_results[m]['MC'] for m in morphologies]
        cq_values = [successful_results[m]['CQ'] for m in morphologies]
        
        # MC对比图
        bars1 = ax1.bar(morphologies, mc_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(morphologies)])
        ax1.set_title('内存容量 (MC) 对比')
        ax1.set_ylabel('MC值')
        ax1.set_ylim(0, max(mc_values) * 1.1)
        
        # 添加数值标签
        for bar, val in zip(bars1, mc_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mc_values)*0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # CQ对比图
        bars2 = ax2.bar(morphologies, cq_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(morphologies)])
        ax2.set_title('计算质量 (CQ) 对比')
        ax2.set_ylabel('CQ值 (KR - GR)')
        
        # 添加数值标签
        for bar, val in zip(bars2, cq_values):
            y_pos = bar.get_height() + (max(cq_values) - min(cq_values)) * 0.02 if val >= 0 else val - (max(cq_values) - min(cq_values)) * 0.02
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n对比图已保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Dict], filename: str = "morphology_cq_mc_results.csv"):
        """保存结果到CSV文件"""
        data = []
        for morph_name, result in results.items():
            data.append({
                'morphology': morph_name,
                'MC': result['MC'],
                'KR': result['KR'],
                'GR': result['GR'],
                'CQ': result['CQ'],
                'calculation_time': result['calculation_time'],
                'success': result['success'],
                'error': result['error'] if result['error'] else ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n结果已保存到: {filename}")
        
        return df


def run_standard_test():
    """运行标准测试"""
    print("开始运行储层形貌CQ和MC标准测试...")
    
    # 创建计算器
    calculator = MorphologyMetricsCalculator()
    
    # 创建测试配置
    configs = calculator.create_test_configurations(n_instances=4, beta_range=(37, 38))
    
    # 创建测试参数
    reservoir_params = calculator.create_test_parameters(
        gamma=0.015600089552382755,
        theta=0.5883604670826516, 
        m0=0.005884523604555656,
        h=0.3438491851897433,
        Nvirt=20,
        beta_prime=37.76687573530968
    )
    
    print(f"\n测试参数:")
    print(f"  gamma = {reservoir_params.params['gamma']}")
    print(f"  theta = {reservoir_params.params['theta']}")
    print(f"  m0 = {reservoir_params.m0}")
    print(f"  h = {reservoir_params.h}")
    print(f"  Nvirt = {reservoir_params.Nvirt}")
    print(f"  beta_prime = {reservoir_params.beta_prime}")
    
    # 计算所有形貌的指标
    results = calculator.calculate_all_morphologies(configs, reservoir_params)
    
    # 显示对比表格
    calculator.display_comparison_table(results)
    
    # 分析差异
    calculator.analyze_morphology_differences(results)
    
    # 绘制对比图
    calculator.plot_performance_comparison(results, "morphology_performance_comparison.png")
    
    # 保存结果
    df = calculator.save_results(results)
    
    return results, calculator


# def run_parameter_sensitivity_test():
#     """运行参数敏感性测试"""
#     print("\n" + "=" * 60)
#     print("参数敏感性测试")
#     print("=" * 60)
    
#     calculator = MorphologyMetricsCalculator()
#     configs = calculator.create_test_configurations(n_instances=3, beta_range=(22, 28))
    
#     # 测试不同gamma值
#     gamma_values = [0.05, 0.1, 0.2, 0.3]
    
#     all_results = {}
    
#     for gamma in gamma_values:
#         print(f"\n测试 gamma = {gamma}")
#         params = calculator.create_test_parameters(gamma=gamma, theta=0.4, Nvirt=20)
        
#         results = {}
#         for morph_name, config in configs.items():
#             result = calculator.calculate_single_morphology_metrics(morph_name, config, params)
#             results[morph_name] = result
        
#         all_results[f'gamma_{gamma}'] = results
    
#     # 显示敏感性分析结果
#     print(f"\n参数敏感性分析结果:")
#     for param_set, results in all_results.items():
#         print(f"\n{param_set}:")
#         for morph_name, result in results.items():
#             if result['success']:
#                 print(f"  {morph_name:12s}: MC={result['MC']:.3f}, CQ={result['CQ']:.3f}")
    
#     return all_results


def main():
    """主函数"""
    print("=" * 80)
    print("储层形貌CQ和MC计算测试")
    print("=" * 80)
    
    # 运行标准测试
    results, calculator = run_standard_test()
    
    # 显示成功的计算数量
    successful_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"\n" + "=" * 60)
    print(f"测试完成总结")
    print("=" * 60)
    print(f"成功计算: {successful_count}/{total_count}")
    print(f"成功率: {successful_count/total_count*100:.1f}%")
    
    if successful_count > 0:
        print("\n建议:")
        print("1. 检查对比表格和分析结果")
        print("2. 查看生成的对比图 'morphology_performance_comparison.png'")
        print("3. 查看保存的CSV结果文件")
        print("4. 根据结果选择最适合的储层形貌")
    
    # # 可选：运行参数敏感性测试
    # user_input = input("\n是否运行参数敏感性测试? (y/n): ").lower().strip()
    # if user_input == 'y':
    #     run_parameter_sensitivity_test()
    
    return results, calculator


if __name__ == "__main__":
    results, calculator = main() 
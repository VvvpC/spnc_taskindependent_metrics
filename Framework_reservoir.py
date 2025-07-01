#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pareto Front 全任务评估框架
========================================

从Pareto Front数据中提取参数，创建均质储层，
计算MC、KR、GR、CQ，执行NARMA10和TI46任务，
并保存所有数据和图像。

Author: Chen
Date: 2025-01-XX
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
from formal_Parameter_Dynamics_Preformance import ReservoirParams, evaluate_MC, evaluate_KRandGR, evaluate_NARMA10
import spnc_ml as ml
from spnc import spnc_anisotropy

def NRMSE(Y,Y_pred):
    var = np.var(Y)
    return np.sqrt(np.square(Y_pred-Y).mean()/var)


class ParetoFrontFramework:
    """Pareto Front全任务评估框架"""
    
    def __init__(self, pareto_csv_path="ParetoFront_CQandMC/data/pareto_front.csv"):
        """初始化框架"""
        self.pareto_csv_path = pareto_csv_path
        self.pareto_data = None
        self.results_base_dir = "results/ParetoFront_AllTI"
        
        # 确保结果目录存在
        os.makedirs(self.results_base_dir, exist_ok=True)
        
        print("🚀 Pareto Front全任务评估框架已初始化")
        print(f"📁 结果保存目录: {self.results_base_dir}")
    
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
    
    def create_homogeneous_reservoir(self, params):
        """创建均质储层"""
        reservoir_params = ReservoirParams(
            h=params['h'],
            m0=params['m0'],
            Nvirt=params['Nvirt'],
            beta_prime=params['beta_prime'],
            params={
                "gamma": params['gamma'],
                "theta": params['theta'],
                "Nvirt": params['Nvirt'],
                "delay_feedback": 0,
                "length_warmup": 0,
                "warmup_sample": 0,
                "voltage_noise": False,
                "johnson_noise": False,
                "thermal_noise": False,
            }
        )
        
        return reservoir_params
    
    def calculate_metrics(self, reservoir_params):
        """计算储层指标 (MC, KR, GR, CQ)"""
        print("  📊 计算储层指标...")
        
        # 计算MC
        print("    📈 计算MC...")
        mc_result = evaluate_MC(reservoir_params, signal_len=550, seed=1234)
        MC = mc_result['MC']
        
        # 计算KR和GR
        print("    📊 计算KR和GR...")
        kgr_result = evaluate_KRandGR(reservoir_params, Nwash=7)
        KR = kgr_result['KR']
        GR = kgr_result['GR']
        
        # 计算CQ
        CQ = KR - GR
        
        metrics = {
            'MC': MC,
            'KR': KR,
            'GR': GR,
            'CQ': CQ
        }
        
        print(f"    ✅ 指标计算完成: MC={MC:.3f}, KR={KR:.1f}, GR={GR:.1f}, CQ={CQ:.3f}")
        
        return metrics
    
    def execute_narma10_task(self, reservoir_params):
        """执行NARMA10任务"""
        print("  🎯 执行NARMA10任务...")
        
        try:
            # 创建SPNC实例
            spn = spnc_anisotropy(
                reservoir_params.h,
                reservoir_params.theta_H,
                reservoir_params.k_s_0,
                reservoir_params.phi,
                reservoir_params.beta_prime,
                restart=True
            )
            
            transform = spn.gen_signal_slow_delayed_feedback
            
            # 只调用一次，获取预测数据
            outputs = ml.spnc_narma10(
                Ntrain=2000,
                Ntest=1000,
                Nvirt=reservoir_params.Nvirt,
                m0=reservoir_params.m0,
                bias=reservoir_params.bias,
                transform=transform,
                params=reservoir_params.params,
                seed_NARMA=1234,
                fixed_mask=True,
                return_outputs=True  # 获取预测数据，不是NRMSE
            )
            
            if outputs is not None:
                y_test, pred = outputs
                
                # 在框架中计算NRMSE
                predNRMSE = NRMSE(y_test, pred)
                
                narma10_result = {
                    'NRMSE': predNRMSE,
                    'y_test': y_test,    # 有数据用于绘图
                    'pred': pred,        # 有数据用于绘图
                    'success': True,
                    'error': None
                }
            else:
                narma10_result = {
                    'NRMSE': np.nan,
                    'success': False,
                    'error': 'Failed to get outputs'
                }
            
            print(f"    ✅ NARMA10任务完成: NRMSE={narma10_result['NRMSE']:.4f}")
            
        except Exception as e:
            print(f"    ❌ NARMA10任务失败: {e}")
            narma10_result = {
                'NRMSE': np.nan,
                'success': False,
                'error': str(e)
            }
        
        return narma10_result
    
    def execute_ti46_task(self, reservoir_params):
        """执行TI46语音识别任务"""
        print("  🗣️  执行TI46任务...")
        
        try:
            # 创建SPNC实例
            spn = spnc_anisotropy(
                reservoir_params.h,
                reservoir_params.theta_H,
                reservoir_params.k_s_0,
                reservoir_params.phi,
                reservoir_params.beta_prime,
                restart=True
            )
            
            transform = spn.gen_signal_slow_delayed_feedback
            
            # 执行TI46任务，启用内部绘图
            accuracy = ml.spnc_spoken_digits(
                speakers=['f1', 'f2', 'f3', 'f4', 'f5'],  # 使用指定说话人
                Nvirt=reservoir_params.Nvirt,
                m0=reservoir_params.m0,
                bias=reservoir_params.bias,
                transform=transform,
                params=reservoir_params.params,
                verbose=True,  # 启用详细信息和内部绘图
                fixed_mask=True,
                return_accuracy=True
                #这里没有指定nfft
            )
            
            ti46_result = {
                'accuracy': accuracy,
                'success': True,
                'error': None
            }
            
            print(f"    ✅ TI46任务完成: Accuracy={accuracy:.4f}")
            
        except Exception as e:
            print(f"    ❌ TI46任务失败: {e}")
            ti46_result = {
                'accuracy': np.nan,
                'success': False,
                'error': str(e)
            }
        
        return ti46_result
    
    def save_results(self, params, metrics, narma10_result, ti46_result, result_dir):
        """保存结果数据"""
        # 创建完整的结果字典
        complete_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_params': params,
            'calculated_metrics': metrics,
            'narma10_results': narma10_result,
            'ti46_results': ti46_result,
            'framework_info': {
                'pareto_csv_path': self.pareto_csv_path,
                'results_dir': result_dir
            }
        }
        
        # 保存为pickle文件
        pickle_filename = f"ParetoFront_AllTask_CQ{metrics['CQ']:.1f}_MC{metrics['MC']:.1f}.pkl"
        pickle_path = os.path.join(result_dir, pickle_filename)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print(f"    💾 结果已保存: {pickle_filename}")
        
        # 也保存为CSV格式供快速查看
        csv_data = {
            'timestamp': [complete_results['timestamp']],
            'number': [params['number']],
            'original_CQ': [params['CQ']],
            'original_MC': [params['MC']],
            'gamma': [params['gamma']],
            'theta': [params['theta']],
            'm0': [params['m0']],
            'h': [params['h']],
            'beta_prime': [params['beta_prime']],
            'Nvirt': [params['Nvirt']],
            'calculated_MC': [metrics['MC']],
            'calculated_KR': [metrics['KR']],
            'calculated_GR': [metrics['GR']],
            'calculated_CQ': [metrics['CQ']],
            'NARMA10_NRMSE': [narma10_result['NRMSE']],
            'NARMA10_success': [narma10_result['success']],
            'TI46_accuracy': [ti46_result['accuracy']],
            'TI46_success': [ti46_result['success']]
        }
        
        csv_filename = f"ParetoFront_AllTask_CQ{metrics['CQ']:.1f}_MC{metrics['MC']:.1f}.csv"
        csv_path = os.path.join(result_dir, csv_filename)
        
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        print(f"    📄 CSV摘要已保存: {csv_filename}")
    
    def save_plots(self, params, metrics, narma10_result, ti46_result, result_dir):
        """保存图像"""
        base_filename = f"ParetoFront_AllTask_CQ{metrics['CQ']:.1f}_MC{metrics['MC']:.1f}"
        
        # 保存NARMA10图像
        if narma10_result['success'] and 'y_test' in narma10_result and 'pred' in narma10_result:
            plt.figure(figsize=(12, 5))
            
            # 绘制预测vs真实值
            plt.subplot(1, 2, 1)
            plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), 'k--', label='Perfect')
            plt.scatter(narma10_result['y_test'], narma10_result['pred'], alpha=0.6, s=10)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'NARMA10 Prediction\nNRMSE = {narma10_result["NRMSE"] if narma10_result['success'] else np.nan:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制时间序列
            plt.subplot(1, 2, 2)
            plot_len = min(200, len(narma10_result['y_test']))
            plt.plot(narma10_result['y_test'][:plot_len], 'b-', label='True', linewidth=1)
            plt.plot(narma10_result['pred'][:plot_len], 'r--', label='Predicted', linewidth=1)
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.title('Time Series Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            narma10_plot_path = os.path.join(result_dir, f"{base_filename}_NARMA10.png")
            plt.savefig(narma10_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    🖼️  NARMA10图像已保存: {base_filename}_NARMA10.png")
        
        # 创建性能摘要图
        plt.figure(figsize=(10, 6))
        
        # 创建一个性能指标摘要图
        metrics_names = ['MC', 'CQ', 'NARMA10\n(1-NRMSE)', 'TI46\nAccuracy']
        metrics_values = [
            metrics['MC'] / 10,  # 归一化MC
            max(0, metrics['CQ']) / 100 if metrics['CQ'] > 0 else 0,  # 归一化CQ
            max(0, 1 - narma10_result['NRMSE']) if narma10_result['success'] and not np.isnan(narma10_result['NRMSE']) else 0,
            ti46_result['accuracy'] if ti46_result['success'] and not np.isnan(ti46_result['accuracy']) else 0
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        
        # 添加数值标签
        original_values = [
            metrics['MC'], 
            metrics['CQ'], 
            narma10_result['NRMSE'] if narma10_result['success'] else np.nan,
            ti46_result['accuracy'] if ti46_result['success'] else np.nan
        ]
        
        for bar, val, original in zip(bars, metrics_values, original_values):
            if not np.isnan(original):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{original:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Normalized Performance')
        plt.title(f'Performance Summary - Pareto Point #{params["number"]}')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加参数信息
        param_text = f"γ={params['gamma']:.3f}, θ={params['theta']:.3f}\n"
        param_text += f"m₀={params['m0']:.4f}, h={params['h']:.3f}\n"
        param_text += f"β'={params['beta_prime']:.1f}, Nvirt={params['Nvirt']}"
        
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        ti46_plot_path = os.path.join(result_dir, f"{base_filename}_TI46.png")
        plt.savefig(ti46_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    🖼️  性能摘要图像已保存: {base_filename}_TI46.png")
    
    def process_single_pareto_point(self, row_index):
        """处理单个Pareto Point"""
        print(f"\n{'='*80}")
        print(f"处理Pareto Point #{row_index + 1}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 1. 提取参数
            print("🔧 1. 提取参数...")
            params = self.extract_parameters(row_index)
            print(f"  📋 参数: CQ={params['CQ']:.1f}, MC={params['MC']:.1f}, "
                  f"γ={params['gamma']}, θ={params['theta']}, m0={params['m0']}, h={params['h']}, β'={params['beta_prime']}, Nvirt={params['Nvirt']}")
            
            # 2. 创建均质储层
            print("🏗️  2. 创建均质储层...")
            reservoir_params = self.create_homogeneous_reservoir(params)
            print(f"  ✅ 储层创建成功: Nvirt={params['Nvirt']}, β'={params['beta_prime']}, h={params['h']}, m0={params['m0']}")
            
            # 3. 计算指标
            print("📊 3. 计算储层指标...")
            metrics = self.calculate_metrics(reservoir_params)
            
            # 4. 执行NARMA10任务
            print("🎯 4. 执行NARMA10任务...")
            narma10_result = self.execute_narma10_task(reservoir_params)
            
            # 5. 执行TI46任务
            print("🗣️  5. 执行TI46任务...")
            ti46_result = self.execute_ti46_task(reservoir_params)
            
            # 6. 创建结果目录
            result_dir_name = f"Point_{params['number']}_CQ{metrics['CQ']:.1f}_MC{metrics['MC']:.1f}"
            result_dir = os.path.join(self.results_base_dir, result_dir_name)
            os.makedirs(result_dir, exist_ok=True)
            
            # 7. 保存结果
            print("💾 6. 保存结果...")
            self.save_results(params, metrics, narma10_result, ti46_result, result_dir)
            
            # 8. 保存图像
            print("🖼️  7. 保存图像...")
            self.save_plots(params, metrics, narma10_result, ti46_result, result_dir)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n✅ Pareto Point #{row_index + 1} 处理完成!")
            print(f"⏱️  总耗时: {elapsed_time:.1f} 秒")
            print(f"📁 结果保存在: {result_dir}")
            
            return {
                'success': True,
                'params': params,
                'metrics': metrics,
                'narma10_result': narma10_result,
                'ti46_result': ti46_result,
                'result_dir': result_dir,
                'processing_time': elapsed_time
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n❌ Pareto Point #{row_index + 1} 处理失败!")
            print(f"💥 错误: {e}")
            print(f"⏱️  耗时: {elapsed_time:.1f} 秒")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': elapsed_time
            }
    
    def process_all_pareto_points(self, start_index=0, end_index=None):
        """处理所有Pareto Points"""
        if self.pareto_data is None:
            if not self.load_pareto_data():
                return
        
        if end_index is None:
            end_index = len(self.pareto_data) if self.pareto_data is not None else 0
        
        print(f"\n🚀 开始处理Pareto Front数据")
        print(f"📊 处理范围: {start_index} - {end_index-1} (共 {end_index-start_index} 个点)")
        print(f"📁 结果保存目录: {self.results_base_dir}")
        
        total_start_time = time.time()
        successful_count = 0
        failed_count = 0
        all_results = []
        
        for i in range(start_index, end_index):
            result = self.process_single_pareto_point(i)
            all_results.append(result)
            
            if result['success']:
                successful_count += 1
            else:
                failed_count += 1
            
            # 显示进度
            progress = (i - start_index + 1) / (end_index - start_index) * 100
            print(f"\n📈 进度: {progress:.1f}% ({i - start_index + 1}/{end_index - start_index})")
            print(f"✅ 成功: {successful_count} | ❌ 失败: {failed_count}")
        
        total_elapsed_time = time.time() - total_start_time
        
        # 保存总体摘要
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processed_range': [start_index, end_index],
            'total_processed': end_index - start_index,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'total_time': total_elapsed_time,
            'average_time_per_point': total_elapsed_time / (end_index - start_index),
            'all_results': all_results
        }
        
        summary_path = os.path.join(self.results_base_dir, 'processing_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"\n🎉 所有Pareto Points处理完成!")
        print(f"📊 总计处理: {end_index - start_index} 个点")
        print(f"✅ 成功: {successful_count} 个")
        print(f"❌ 失败: {failed_count} 个")
        print(f"⏱️  总耗时: {total_elapsed_time/3600:.1f} 小时")
        print(f"📄 摘要已保存: {summary_path}")


def main():
    """主函数"""
    print("🚀 Pareto Front全任务评估框架")
    print("=" * 50)
    
    # 创建框架实例
    framework = ParetoFrontFramework()
    
    # 加载数据
    if not framework.load_pareto_data():
        return
    
    # 询问用户要处理哪些点
    total_points = len(framework.pareto_data) if framework.pareto_data is not None else 0
    print(f"\n📊 可处理的Pareto Points: 0 - {total_points-1}")
    
    try:
        # 处理前几个点作为示例
        framework.process_all_pareto_points()
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")


if __name__ == "__main__":
    main() 
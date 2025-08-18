'''
这个代码的作用是根据up_pareto组的数据,来评估gamma和beta_prime对于CQ,MC,NARMA10,TI46的影响
'''

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
from pathlib import Path



# 添加上级目录到Python路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from formal_Parameter_Dynamics_Preformance import ReservoirParams, evaluate_KRandGR, evaluate_MC
from spnc import spnc_anisotropy

from ParetoFront_CQandMC.CQ_MC_ParetofrontPoints import eva_narma10, eva_ti46
def MSE(pred, desired):
    return np.mean(np.square(np.subtract(pred, desired)))

def NRMSE(pred, y_test, spacer=0.001):
    return np.sqrt(MSE(pred, y_test) / np.var(y_test))

class ReservoirParams:
    """
    Reservoir parameters configuration class
    """
    def __init__(self, gamma=0, beta_prime=20, **kwargs):
        # Reservoir parameters 
        self.h = 0.4
        self.theta_H = 90
        self.k_s_0 = 0
        self.phi = 45
        self.beta_prime = beta_prime

        # Network parameters 
        self.Nvirt = 200
        self.m0 = 0.008
        self.bias = True
        self.Nwarmup = 0
        self.verbose_repr = False

        self.params = {
            'theta': 0.156493839,
            'gamma': gamma,
            'delay_feedback': 0,
            'Nvirt': self.Nvirt,
            'length_warmup': self.Nwarmup,
            'warmup_sample': self.Nwarmup * self.Nvirt,
            'voltage_noise': False,
            'seed_voltage_noise': 1234,
            'delta_V': 0.1,
            'johnson_noise': False,
        }
        
        # 更新任何传入的额外参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.params:
                self.params[key] = value

class HeatmapEvaluator:
    """评估gamma和beta_prime对于CQ,MC,NARMA10,TI46影响的类"""
    
    def __init__(self):
        self.results = {}
        
    def create_reservoir_params(self, gamma, beta_prime):
        """创建储层参数"""
        return ReservoirParams(gamma=gamma, beta_prime=beta_prime)
    
    def evaluate_MC(self, reservoir_params: ReservoirParams):
        """评估记忆容量(MC)"""
        try:
            MC = evaluate_MC(reservoir_params)
            return MC['MC']
        except Exception as e:
            print(f"MC evaluation error: {e}")
            return np.nan
    
    def evaluate_CQ(self, reservoir_params: ReservoirParams):
        """评估计算质量(CQ) - 返回KR, GR"""
        try:
            results = evaluate_KRandGR(reservoir_params)
            CQ = results['KR']-results['GR']
            return CQ, results['KR'], results['GR']
        except Exception as e:
            print(f"CQ evaluation error: {e}")
            return np.nan, np.nan, np.nan
    
    def evaluate_NARMA10(self, reservoir_params: ReservoirParams):
        """评估NARMA-10任务性能"""
        try:
            nrmse, y_test, pred = eva_narma10(reservoir_params)
            return nrmse, y_test, pred
        except Exception as e:
            print(f"NARMA10 evaluation error: {e}")
            return np.nan, None, None
    
    def evaluate_TI46(self, reservoir_params: ReservoirParams):
        """评估TI46任务性能"""
        try:
            ti46_accuracy = eva_ti46(reservoir_params)
            error_rate = (1 - ti46_accuracy) * 100
            return error_rate
        except Exception as e:
            print(f"TI46 evaluation error: {e}")
            return np.nan
    
    def run_parameter_sweep(self, gamma_range, beta_prime_range, 
                           tasks=['MC', 'CQ', 'NARMA10', 'TI46'], 
                           save_results=True, results_dir="heatmap_results"):
        """
        执行参数扫描
        
        Parameters:
        -----------
        gamma_range : array-like
            gamma参数范围
        beta_prime_range : array-like
            beta_prime参数范围
        tasks : list
            要执行的任务列表，可选: ['MC', 'CQ', 'NARMA10', 'TI46']
            - 'MC': 记忆容量评估
            - 'CQ': 计算质量评估 (包含KR, GR)
            - 'NARMA10': NARMA-10任务性能评估
            - 'TI46': TI46任务性能评估
        save_results : bool
            是否保存结果
        results_dir : str
            结果保存目录
        """
        print(f"Starting parameter sweep...")
        print(f"Gamma range: {gamma_range}")
        print(f"Beta_prime range: {beta_prime_range}")
        print(f"Selected tasks: {tasks}")
        
        # 验证任务列表
        available_tasks = ['MC', 'CQ', 'NARMA10', 'TI46']
        invalid_tasks = [task for task in tasks if task not in available_tasks]
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}. Available tasks: {available_tasks}")
        
        # 创建结果目录
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # 初始化结果存储 - 只包含选定的任务
        results = {
            'gamma': [],
            'beta_prime': []
        }
        
        # 根据选择的任务添加相应的结果列
        if 'MC' in tasks:
            results['MC'] = []
        
        if 'CQ' in tasks:
            results['CQ'] = []
            results['KR'] = []
            results['GR'] = []
        
        if 'NARMA10' in tasks:
            results['NARMA10_NRMSE'] = []
        
        if 'TI46' in tasks:
            results['TI46_error_rate'] = []
        
        total_combinations = len(gamma_range) * len(beta_prime_range)
        pbar = tqdm(total=total_combinations, desc="Parameter sweep")
        
        for gamma in gamma_range:
            for beta_prime in beta_prime_range:
                # 创建储层参数
                reservoir_params = self.create_reservoir_params(gamma, beta_prime)
                # 打印所有参数
                print(f"Reservoir parameters: Nvirt={reservoir_params.Nvirt}, m0={reservoir_params.m0}, gamma={reservoir_params.params['gamma']}, beta_prime={reservoir_params.beta_prime}")
                
                # 存储基本参数
                results['gamma'].append(gamma)
                results['beta_prime'].append(beta_prime)
                
                # 根据选择的任务进行评估
                current_results = {}
                
                if 'MC' in tasks:
                    mc = self.evaluate_MC(reservoir_params)
                    print(f"MC: {mc}")
                    results['MC'].append(mc)
                    current_results['MC'] = mc
                
                if 'CQ' in tasks:
                    cq, kr, gr = self.evaluate_CQ(reservoir_params)
                    results['CQ'].append(cq)
                    results['KR'].append(kr)
                    results['GR'].append(gr)
                    current_results['CQ'] = cq
                
                if 'NARMA10' in tasks:
                    narma10_nrmse, _, _ = self.evaluate_NARMA10(reservoir_params)
                    results['NARMA10_NRMSE'].append(narma10_nrmse)
                    current_results['NARMA10'] = narma10_nrmse
                
                if 'TI46' in tasks:
                    ti46_error_rate = self.evaluate_TI46(reservoir_params)
                    results['TI46_error_rate'].append(ti46_error_rate)
                    current_results['TI46'] = ti46_error_rate
                
                # 更新进度条
                progress_info = {
                    'gamma': f'{gamma:.3f}',
                    'beta_prime': f'{beta_prime:.1f}'
                }
                progress_info.update(current_results)
                
                pbar.update(1)
                pbar.set_postfix(progress_info)
        
        pbar.close()
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        self.results = df_results
        
        # 保存结果
        if save_results:
            # 生成包含任务信息的文件名
            task_suffix = "_".join(tasks)
            results_file = os.path.join(results_dir, f'parameter_sweep_results_{task_suffix}.csv')
            df_results.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")
            
            # 保存为pickle文件
            pickle_file = os.path.join(results_dir, f'parameter_sweep_results_{task_suffix}.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(df_results, f)
            print(f"Results saved to {pickle_file}")
        
        # 打印结果摘要
        print(f"\nParameter sweep completed!")
        print(f"Total combinations: {len(df_results)}")
        print(f"Tasks evaluated: {tasks}")
        print(f"Result columns: {list(df_results.columns)}")
        
        return df_results


def create_heatmaps(filename, save_plots=True, plots_dir="heatmap_plots", file_format='csv'):
    """
    独立的热力图绘制函数
    
    Parameters:
    -----------
    filename : str
        数据文件路径 (支持 .csv 或 .pkl 文件)
    save_plots : bool
        是否保存图片
    plots_dir : str
        图片保存目录
    file_format : str
        文件格式 ('csv' 或 'pkl')
    """
    
    # 加载数据
    if file_format.lower() == 'csv' or filename.endswith('.csv'):
        try:
            results_df = pd.read_csv(filename)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None, None
    elif file_format.lower() == 'pkl' or filename.endswith('.pkl'):
        try:
            with open(filename, 'rb') as f:
                results_df = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None, None
    else:
        print(f"Unsupported file format. Use 'csv' or 'pkl'")
        return None, None
    
    print(f"Loaded data from {filename}")
    print(f"Data shape: {results_df.shape}")
    print(f"Available columns: {list(results_df.columns)}")
    
    # 创建保存目录
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
    
    # 检查必要的列是否存在
    required_cols = ['gamma', 'beta_prime']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None, None
    
    # 定义要绘制的指标
    available_metrics = [col for col in results_df.columns 
                        if col not in ['gamma', 'beta_prime']]
    
    # 根据可用指标数量调整子图布局
    n_metrics = len(available_metrics)
    if n_metrics <= 3:
        nrows, ncols = 1, n_metrics
        figsize = (6*n_metrics, 5)
    elif n_metrics <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        nrows = (n_metrics + 2) // 3
        ncols = 3
        figsize = (18, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else axes
    
    for i, metric in enumerate(available_metrics):
        if i >= len(axes):
            break
            
        try:
            # 创建透视表
            pivot_data = results_df.pivot(index='beta_prime', columns='gamma', values=metric)
            
            # 绘制热力图
            im = axes[i].imshow(pivot_data.values, cmap='viridis', aspect='auto')
            axes[i].set_title(f'{metric} Heatmap', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Gamma', fontsize=10)
            axes[i].set_ylabel('Beta Prime', fontsize=10)
            
            # 设置刻度标签
            axes[i].set_xticks(range(len(pivot_data.columns)))
            axes[i].set_xticklabels([f'{x:.3f}' for x in pivot_data.columns], 
                                   rotation=45, fontsize=8)
            axes[i].set_yticks(range(len(pivot_data.index)))
            axes[i].set_yticklabels([f'{y:.1f}' for y in pivot_data.index], 
                                   fontsize=8)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.ax.tick_params(labelsize=8)
            
            # 添加数值标注（如果数据点不太多）
            if len(pivot_data.columns) <= 10 and len(pivot_data.index) <= 10:
                for j in range(len(pivot_data.index)):
                    for k in range(len(pivot_data.columns)):
                        value = pivot_data.iloc[j, k]
                        if not np.isnan(value):
                            # 根据数值大小选择文字颜色
                            text_color = 'white' if value < pivot_data.values.mean() else 'black'
                            axes[i].text(k, j, f'{value:.3f}', 
                                       ha='center', va='center', 
                                       color=text_color, fontsize=7, fontweight='bold')
        
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
            axes[i].text(0.5, 0.5, f'Error plotting\n{metric}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # 移除多余的子图
    if n_metrics < len(axes):
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # 保存图片
    if save_plots:
        plot_file = os.path.join(plots_dir, 'parameter_heatmaps.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to {plot_file}")
        
        # 也保存为PDF格式
        pdf_file = os.path.join(plots_dir, 'parameter_heatmaps.pdf')
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"Heatmaps also saved to {pdf_file}")
    
    plt.show()
    
    return fig, axes



def main():
    """主函数 - 示例用法"""
    # 创建评估器
    evaluator = HeatmapEvaluator()
    
    # 定义参数范围
    gamma_range = np.linspace(0.04, 0.08, 10)  # 从0到0.5，6个点
    beta_prime_range = np.linspace(20, 50, 10)  # 从10到50，5个点
    
    print("开始参数扫描...")
    print(f"Gamma范围: {gamma_range}")
    print(f"Beta_prime范围: {beta_prime_range}")
    
    # 示例3: 评估所有任务（默认）
    print("\n=== 示例3: 评估所有任务 ===")
    results3 = evaluator.run_parameter_sweep(
        gamma_range, beta_prime_range
        # tasks参数默认为['MC', 'CQ', 'NARMA10', 'TI46']
    )
    
    print("\n分析完成！")
    return  results3


if __name__ == "__main__":
    results = main()





        

        






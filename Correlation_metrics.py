import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ParameterCorrelationAnalyzer:
    """
    参数相关性分析类，用于分析储层性能指标之间的相关性
    """
    
    def __init__(self, data_path: str = "results/ParetoFront_AllTI"):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.data = None
        self.correlation_results = {}
        
    def load_all_data(self) -> pd.DataFrame:
        """
        从所有子文件夹中加载CSV数据
        
        Returns:
            合并后的DataFrame
        """
        all_data = []
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        
        print(f"发现 {len(subfolders)} 个数据文件夹:")
        
        for folder in subfolders:
            folder_path = os.path.join(self.data_path, folder)
            # 查找CSV文件
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            
            if csv_files:
                csv_path = os.path.join(folder_path, csv_files[0])
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        all_data.append(df)
                        print(f"  ✓ {folder}: 加载了 {len(df)} 条记录")
                    else:
                        print(f"  ✗ {folder}: CSV文件为空")
                except Exception as e:
                    print(f"  ✗ {folder}: 加载失败 - {e}")
            else:
                print(f"  ✗ {folder}: 未找到CSV文件")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\n总共加载了 {len(self.data)} 条记录")
            return self.data
        else:
            raise ValueError("未能加载任何数据")
    
    def calculate_correlations(self, target_var: str = 'NARMA10_NRMSE', 
                             variables: Optional[List[str]] = None) -> Dict:
        """
        计算目标变量与其他变量的相关性
        
        Args:
            target_var: 目标变量名
            variables: 要分析的变量列表，如果为None则分析所有数值变量
            
        Returns:
            相关性结果字典
        """
        if self.data is None:
            self.load_all_data()
        
        assert self.data is not None, "数据加载失败"
        
        # 如果未指定变量，则选择所有数值变量
        if variables is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            variables = [col for col in numeric_columns if col != target_var]
        
        results = {}
        
        print(f"\n计算 {target_var} 与其他变量的相关性:")
        print("=" * 60)
        
        for var in variables:
            if var in self.data.columns:
                # 移除缺失值
                clean_data = self.data[[target_var, var]].dropna()
                
                if len(clean_data) > 1:
                    # 计算Pearson相关系数
                    pearson_result = pearsonr(clean_data[target_var], clean_data[var])
                    pearson_corr = pearson_result[0]
                    pearson_p = pearson_result[1]
                    
                    # 计算Spearman相关系数
                    spearman_result = spearmanr(clean_data[target_var], clean_data[var])
                    spearman_corr = spearman_result[0]
                    spearman_p = spearman_result[1]
                    
                    results[var] = {
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'sample_size': len(clean_data)
                    }
                    
                    # 打印结果
                    print(f"{var:20s}: Pearson={pearson_corr:6.3f} (p={pearson_p:.3f}), "
                          f"Spearman={spearman_corr:6.3f} (p={spearman_p:.3f}), n={len(clean_data)}")
        
        self.correlation_results[target_var] = results
        return results
    
    def analyze_parameter_correlation(self, param1: str, param2: str) -> Dict:
        """
        分析任意两个参数之间的相关性
        
        Args:
            param1: 第一个参数名
            param2: 第二个参数名
            
        Returns:
            详细的相关性分析结果
        """
        print("="*80)
        print(f"{param1} 与 {param2} 相关性专项分析")
        print("="*80)
        
        if self.data is None:
            self.load_all_data()
        
        assert self.data is not None, "数据加载失败"
        
        # 检查参数是否存在
        if param1 not in self.data.columns:
            print(f"错误：参数 '{param1}' 不存在于数据中")
            return {}
        
        if param2 not in self.data.columns:
            print(f"错误：参数 '{param2}' 不存在于数据中")
            return {}
        
        # 提取相关数据
        param_data = self.data[[param1, param2]].dropna()
        
        if len(param_data) == 0:
            print(f"警告：没有找到有效的{param1}和{param2}数据")
            return {}
        
        # 基本统计信息
        print(f"\n数据概览:")
        print(f"样本数量: {len(param_data)}")
        print(f"{param1}范围: {param_data[param1].min():.3f} - {param_data[param1].max():.3f} (均值: {param_data[param1].mean():.3f})")
        print(f"{param2}范围: {param_data[param2].min():.3f} - {param_data[param2].max():.3f} (均值: {param_data[param2].mean():.3f})")
        
        # 相关性分析
        pearson_result = pearsonr(param_data[param1], param_data[param2])
        pearson_corr = float(pearson_result.statistic)
        pearson_p = float(pearson_result.pvalue)
        
        spearman_result = spearmanr(param_data[param1], param_data[param2])
        spearman_corr = float(spearman_result.statistic)
        spearman_p = float(spearman_result.pvalue)
        
        results = {
            'param1': param1,
            'param2': param2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'sample_size': len(param_data),
            'data': param_data
        }
        
        print(f"\n相关性结果:")
        print(f"Pearson相关系数: {pearson_corr:.4f} (p值: {pearson_p:.4f})")
        print(f"Spearman相关系数: {spearman_corr:.4f} (p值: {spearman_p:.4f})")
        
        # 解释相关性强度
        abs_pearson = abs(pearson_corr)
        if abs_pearson >= 0.7:
            strength = "强相关"
        elif abs_pearson >= 0.5:
            strength = "中等相关"
        elif abs_pearson >= 0.3:
            strength = "弱相关"
        else:
            strength = "很弱或无相关"
        
        direction = "正相关" if pearson_corr > 0 else "负相关"
        print(f"\n相关性解释: {param1}与{param2}之间存在{direction}关系，强度为{strength}")
        
        return results
    
    def plot_correlation_matrix(self, variables: Optional[List[str]] = None, 
                               figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        绘制相关性矩阵热图
        
        Args:
            variables: 要包含的变量列表
            figsize: 图形大小
        """
        if self.data is None:
            self.load_all_data()
        
        # 选择数值变量
        if variables is None:
            numeric_data = self.data.select_dtypes(include=[np.number])
        else:
            numeric_data = self.data[variables]
        
        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 绘制热图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('参数相关性矩阵', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_scatter(self, param1: str, param2: str, 
                              color_by: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        绘制任意两个参数的散点图
        
        Args:
            param1: 第一个参数名
            param2: 第二个参数名
            color_by: 用于着色的第三个参数名（可选）
            figsize: 图形大小
        """
        if self.data is None:
            self.load_all_data()
        
        # 检查参数是否存在
        required_params = [param1, param2]
        if color_by:
            required_params.append(color_by)
        
        for param in required_params:
            if param not in self.data.columns:
                print(f"错误：参数 '{param}' 不存在于数据中")
                return
        
        # 提取数据
        param_data = self.data[required_params].dropna()
        
        if len(param_data) == 0:
            print(f"没有找到有效的{param1}和{param2}数据")
            return
        
        # 创建图形
        if color_by:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        
        # 散点图1：基本散点图
        ax1.scatter(param_data[param1], param_data[param2], 
                   alpha=0.7, s=100, c='blue', edgecolors='black', linewidth=0.5)
        ax1.set_xlabel(param1, fontsize=12)
        ax1.set_ylabel(param2, fontsize=12)
        ax1.set_title(f'{param2} vs {param1} Pearson correlation', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(param_data[param1], param_data[param2], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(param_data[param1].min(), param_data[param1].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        #在图上画出pearson相关系数
        pearson_corr = pearsonr(param_data[param1], param_data[param2])

        ax1.text(0.05, 0.95, f'Pearson coefficient: {pearson_corr[0]:.4f}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            
        
        # 散点图2：按第三个参数着色（如果提供）
        if color_by:
            scatter = ax2.scatter(param_data[param1], param_data[param2], 
                                c=param_data[color_by], cmap='viridis', 
                                alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            ax2.set_xlabel(param1, fontsize=12)
            ax2.set_ylabel(param2, fontsize=12)
            ax2.set_title(f'{param2} vs {param1} (按{color_by}着色)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label(color_by, fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # 计算并显示相关性
        correlation = param_data[param1].corr(param_data[param2])
        print(f"{param1} and {param2} Pearson correlation: {correlation:.4f}")
    
    def generate_correlation_report(self, target_var: str = 'NARMA10_NRMSE',
                                   param1: Optional[str] = None, param2: Optional[str] = None,
                                   output_file: str = "correlation_analysis_report.txt") -> None:
        """
        生成详细的相关性分析报告
        
        Args:
            target_var: 目标变量（用于全面分析）
            param1: 专项分析的第一个参数
            param2: 专项分析的第二个参数
            output_file: 输出文件名
        """
        if self.data is None:
            self.load_all_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("储层性能参数相关性分析报告\n")
            f.write("="*50 + "\n\n")
            
            # 数据概览
            f.write("1. 数据概览\n")
            f.write("-"*20 + "\n")
            f.write(f"总样本数: {len(self.data)}\n")
            f.write(f"数据列数: {len(self.data.columns)}\n\n")
            
            # 描述性统计
            f.write("2. 主要指标描述性统计\n")
            f.write("-"*30 + "\n")
            key_metrics = ['TI46_accuracy', 'calculated_KR', 'calculated_GR', 
                          'calculated_CQ', 'calculated_MC', 'NARMA10_NRMSE', 'Nvirt']
            
            for metric in key_metrics:
                if metric in self.data.columns:
                    data_clean = self.data[metric].dropna()
                    f.write(f"{metric}:\n")
                    f.write(f"  均值: {data_clean.mean():.4f}\n")
                    f.write(f"  标准差: {data_clean.std():.4f}\n")
                    f.write(f"  最小值: {data_clean.min():.4f}\n")
                    f.write(f"  最大值: {data_clean.max():.4f}\n\n")
            
            # 专项分析（如果提供了两个参数）
            if param1 and param2:
                f.write(f"3. {param1}与{param2}相关性专项分析\n")
                f.write("-"*30 + "\n")
                param_results = self.analyze_parameter_correlation(param1, param2)
                
                if param_results:
                    f.write(f"Pearson相关系数: {param_results['pearson_correlation']:.4f}\n")
                    f.write(f"Pearson p值: {param_results['pearson_p_value']:.4f}\n")
                    f.write(f"Spearman相关系数: {param_results['spearman_correlation']:.4f}\n")
                    f.write(f"Spearman p值: {param_results['spearman_p_value']:.4f}\n")
                    f.write(f"样本数: {param_results['sample_size']}\n\n")
            
            # 全面相关性分析
            f.write(f"4. {target_var}全面相关性分析\n")
            f.write("-"*20 + "\n")
            all_correlations = self.calculate_correlations(target_var)
            
            # 按相关性强度排序
            sorted_correlations = sorted(all_correlations.items(), 
                                       key=lambda x: abs(x[1]['pearson_correlation']), 
                                       reverse=True)
            
            for var, corr_data in sorted_correlations:
                f.write(f"{var}:\n")
                f.write(f"  Pearson: {corr_data['pearson_correlation']:.4f} (p={corr_data['pearson_p_value']:.4f})\n")
                f.write(f"  Spearman: {corr_data['spearman_correlation']:.4f} (p={corr_data['spearman_p_value']:.4f})\n\n")
        
        print(f"相关性分析报告已保存到: {output_file}")

    def get_available_parameters(self) -> List[str]:
        """
        获取所有可用的参数列表
        
        Returns:
            参数名列表
        """
        if self.data is None:
            self.load_all_data()
        
        return self.data.columns.tolist()

    def print_available_parameters(self) -> None:
        """
        打印所有可用的参数
        """
        params = self.get_available_parameters()
        print("\n可用参数列表:")
        print("="*40)
        for i, param in enumerate(params, 1):
            print(f"{i:2d}. {param}")


def main():
    """
    主函数：执行完整的相关性分析
    """
    # 创建分析器实例
    analyzer = ParameterCorrelationAnalyzer()
    analyzer.load_all_data()

    analyzer.data['KR_GR_ratio'] = analyzer.data['calculated_KR'] / analyzer.data['calculated_GR']
    analyzer.data['task_balance'] = analyzer.data['TI46_accuracy'] / analyzer.data['NARMA10_NRMSE']
    analyzer.data['MC_KR_ratio'] = analyzer.data['calculated_MC'] / analyzer.data['calculated_KR']
    analyzer.data['MC_GR_ratio'] = analyzer.data['calculated_MC'] / analyzer.data['calculated_GR']
    analyzer.data['MC_CQ_ratio'] = analyzer.data['calculated_MC'] / analyzer.data['calculated_CQ']
    analyzer.data['MC_density'] = analyzer.data['calculated_MC'] / analyzer.data['Nvirt']
    analyzer.data['CQ_density'] = analyzer.data['calculated_CQ'] / analyzer.data['Nvirt']
    analyzer.data['GR_density'] = analyzer.data['calculated_GR'] / analyzer.data['Nvirt']
    analyzer.data['KR_density'] = analyzer.data['calculated_KR'] / analyzer.data['Nvirt']





    try:
        # 加载数据
        print("正在加载数据...")
        # analyzer.load_all_data()
        
        # 显示可用参数
        analyzer.print_available_parameters()
        
        # # 示例1：分析TI46与KR的相关性
        # print("\n示例1：分析TI46与KR的相关性...")
        # ti46_cq_results = analyzer.analyze_parameter_correlation('TI46_accuracy', 'calculated_CQ')
        
        # # 示例2：分析NARMA10与MC的相关性
        # print("\n示例2：分析NARMA10与MC的相关性...")
        # narma_mc_results = analyzer.analyze_parameter_correlation('NARMA10_NRMSE', 'calculated_MC')
        
        # # 计算NARMA10与所有变量的相关性
        # print("\n计算NARMA10与所有变量的相关性...")
        # all_correlations = analyzer.calculate_correlations('NARMA10_NRMSE')
        
        # 绘制图表
        print("\n生成可视化图表...")
        analyzer.plot_parameter_scatter('calculated_MC', 'calculated_CQ')
        
        # # 生成报告
        # print("\n生成分析报告...")
        # analyzer.generate_correlation_report('NARMA10_NRMSE', 'TI46_accuracy', 'calculated_KR')
        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

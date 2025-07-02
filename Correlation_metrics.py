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
    
    def __init__(self, data_path: str = "results/ParetoFront_"):
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
        

        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

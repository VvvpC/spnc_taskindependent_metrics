import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        pearson_corr = float(pearson_result[0])
        pearson_p = float(pearson_result[1])
        
        spearman_result = spearmanr(param_data[param1], param_data[param2])
        spearman_corr = float(spearman_result[0])
        spearman_p = float(spearman_result[1])
        
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
    
    def perform_pca_analysis(self, variables: Optional[List[str]] = None, 
                           n_components: int = 2) -> Dict:
        """
        执行主成分分析
        
        Args:
            variables: 要分析的变量列表，如果为None则使用所有数值变量
            n_components: 主成分数量
            
        Returns:
            PCA分析结果字典
        """
        if self.data is None:
            self.load_all_data()
        
        # 选择变量
        if variables is None:
            numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
        else:
            numeric_data = self.data[variables].dropna()
        
        if len(numeric_data) == 0:
            raise ValueError("没有可用于PCA分析的数据")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # 计算特征重要性
        feature_importance = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numeric_data.columns
        )
        
        results = {
            'pca_data': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'feature_importance': feature_importance,
            'original_features': numeric_data.columns.tolist(),
            'scaler': scaler,
            'pca_model': pca
        }
        
        print(f"PCA分析完成:")
        print(f"主成分数量: {n_components}")
        print(f"解释方差比例: {pca.explained_variance_ratio_}")
        print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")
        
        return results
    
    def plot_pca_scatter(self, pca_results: Dict, color_by: Optional[str] = None) -> None:
        """
        绘制PCA散点图
        
        Args:
            pca_results: PCA分析结果
            color_by: 用于着色的变量名
        """
        pca_data = pca_results['pca_data']
        explained_var = pca_results['explained_variance_ratio']
        
        plt.figure(figsize=(10, 8))
        
        if color_by and color_by in self.data.columns:
            color_data = self.data[color_by].iloc[:len(pca_data)]
            scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                c=color_data, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=color_by)
        else:
            plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
        
        plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        plt.title('PCA Analysis - First Two Principal Components')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_pca_feature_importance(self, pca_results: Dict) -> None:
        """
        绘制PCA特征重要性图
        
        Args:
            pca_results: PCA分析结果
        """
        feature_importance = pca_results['feature_importance']
        n_components = feature_importance.shape[1]
        
        fig, axes = plt.subplots(1, n_components, figsize=(15, 6))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            pc_name = f'PC{i+1}'
            importance = feature_importance[pc_name].abs().sort_values(ascending=True)
            
            ax = axes[i] if n_components > 1 else axes[0]
            ax.barh(range(len(importance)), importance.values)
            ax.set_yticks(range(len(importance)))
            ax.set_yticklabels(importance.index)
            ax.set_xlabel('Absolute Loading')
            ax.set_title(f'{pc_name} Feature Importance')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, variables: Optional[List[str]] = None, 
                                figsize: tuple = (12, 10)) -> None:
        """
        绘制相关性热图
        
        Args:
            variables: 要包含的变量列表
            figsize: 图形大小
        """
        if self.data is None:
            self.load_all_data()
        
        if variables is None:
            numeric_data = self.data.select_dtypes(include=[np.number])
        else:
            numeric_data = self.data[variables]
        
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('参数相关性矩阵热图', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_bar(self, target_var: str, variables: Optional[List[str]] = None,
                           top_n: int = 10) -> None:
        """
        绘制相关性条形图
        
        Args:
            target_var: 目标变量
            variables: 要分析的变量列表
            top_n: 显示前N个相关性最强的变量
        """
        correlations = self.calculate_correlations(target_var, variables)
        
        # 提取相关系数并排序
        corr_data = [(var, data['pearson_correlation']) 
                    for var, data in correlations.items()]
        corr_data.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if top_n:
            corr_data = corr_data[:top_n]
        
        vars_list, corr_values = zip(*corr_data)
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(vars_list)), corr_values, color=colors, alpha=0.7)
        plt.yticks(range(len(vars_list)), vars_list)
        plt.xlabel('Pearson相关系数')
        plt.title(f'{target_var} 相关性排序图')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            plt.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_matrix(self, variables: List[str], target_var: Optional[str] = None) -> None:
        """
        绘制散点图矩阵
        
        Args:
            variables: 要分析的变量列表
            target_var: 用于着色的目标变量
        """
        if self.data is None:
            self.load_all_data()
        
        data_subset = self.data[variables].dropna()
        
        if target_var and target_var in self.data.columns:
            color_data = self.data[target_var].iloc[:len(data_subset)]
            pd.plotting.scatter_matrix(data_subset, c=color_data, 
                                     figsize=(15, 15), alpha=0.7, cmap='viridis')
        else:
            pd.plotting.scatter_matrix(data_subset, figsize=(15, 15), alpha=0.7)
        
        plt.suptitle('变量散点图矩阵', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_pair_correlations(self, param1: str, param2: str, 
                             color_by: Optional[str] = None) -> None:
        """
        绘制参数对相关性图（含多种视图）
        
        Args:
            param1: 第一个参数名
            param2: 第二个参数名
            color_by: 用于着色的第三个参数名
        """
        if self.data is None:
            self.load_all_data()
        
        required_params = [param1, param2]
        if color_by:
            required_params.append(color_by)
        
        for param in required_params:
            if param not in self.data.columns:
                print(f"错误：参数 '{param}' 不存在于数据中")
                return
        
        param_data = self.data[required_params].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 基本散点图
        if color_by:
            scatter = axes[0,0].scatter(param_data[param1], param_data[param2], 
                                      c=param_data[color_by], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=axes[0,0], label=color_by)
        else:
            axes[0,0].scatter(param_data[param1], param_data[param2], alpha=0.7)
        
        # 添加趋势线
        z = np.polyfit(param_data[param1], param_data[param2], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(param_data[param1].min(), param_data[param1].max(), 100)
        axes[0,0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # 添加相关系数
        corr = param_data[param1].corr(param_data[param2])
        axes[0,0].text(0.05, 0.95, f'r = {corr:.4f}', transform=axes[0,0].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0,0].set_xlabel(param1)
        axes[0,0].set_ylabel(param2)
        axes[0,0].set_title('散点图 + 趋势线')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 六边形分布图
        axes[0,1].hexbin(param_data[param1], param_data[param2], gridsize=20, cmap='Blues')
        axes[0,1].set_xlabel(param1)
        axes[0,1].set_ylabel(param2)
        axes[0,1].set_title('六边形密度图')
        
        # 3. 参数1的分布
        axes[1,0].hist(param_data[param1], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_xlabel(param1)
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title(f'{param1} 分布')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 参数2的分布
        axes[1,1].hist(param_data[param2], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1,1].set_xlabel(param2)
        axes[1,1].set_ylabel('频次')
        axes[1,1].set_title(f'{param2} 分布')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # def plot_correlation_network(self, variables: Optional[List[str]] = None, 
    #                            threshold: float = 0.3) -> None:
    #     """
    #     绘制相关性网络图（需要networkx库）
        
    #     Args:
    #         variables: 要分析的变量列表
    #         threshold: 相关性阈值，只显示超过此值的连接
    #     """
    #     try:
    #         import networkx as nx
    #     except ImportError:
    #         print("需要安装networkx库: pip install networkx")
    #         return
        
    #     if self.data is None:
    #         self.load_all_data()
        
    #     if variables is None:
    #         numeric_data = self.data.select_dtypes(include=[np.number])
    #     else:
    #         numeric_data = self.data[variables]
        
    #     corr_matrix = numeric_data.corr()
        
    #     # 创建网络图
    #     G = nx.Graph()
        
    #     # 添加节点
    #     for var in corr_matrix.columns:
    #         G.add_node(var)
        
    #     # 添加边（超过阈值的相关性）
    #     for i, var1 in enumerate(corr_matrix.columns):
    #         for j, var2 in enumerate(corr_matrix.columns):
    #             if i < j:  # 避免重复
    #                 corr_val = abs(corr_matrix.iloc[i, j])
    #                 if corr_val >= threshold:
    #                     G.add_edge(var1, var2, weight=corr_val)
        
    #     plt.figure(figsize=(12, 10))
    #     pos = nx.spring_layout(G, k=1, iterations=50)
        
    #     # 绘制节点
    #     nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
    #                           node_size=1500, alpha=0.7)
        
    #     # 绘制边，粗细表示相关性强度
    #     edges = G.edges()
    #     weights = [G[u][v]['weight'] * 3 for u, v in edges]
    #     nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6)
        
    #     # 绘制标签
    #     nx.draw_networkx_labels(G, pos, font_size=10)
        
    #     plt.title(f'相关性网络图 (阈值: {threshold})')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()


def run_correlation_analysis():
    """
    执行相关性分析和可视化
    """
    # 创建分析器实例
    analyzer = ParameterCorrelationAnalyzer()
    analyzer.load_all_data()

    # 优雅地创建派生特征
    derived_features = {
        'KR_GR_ratio': ('calculated_KR', 'calculated_GR'),
        'task_balance': ('TI46_accuracy', 'NARMA10_NRMSE'),
        'MC_KR_ratio': ('calculated_MC', 'calculated_KR'),
        'MC_GR_ratio': ('calculated_MC', 'calculated_GR'),
        'MC_CQ_ratio': ('calculated_MC', 'calculated_CQ'),
        'MC_density': ('calculated_MC', 'Nvirt'),
        'CQ_density': ('calculated_CQ', 'Nvirt'),
        'GR_density': ('calculated_GR', 'Nvirt'),
        'KR_density': ('calculated_KR', 'Nvirt'),
    }
    
    for feature_name, (num, den) in derived_features.items():
        if num in analyzer.data.columns and den in analyzer.data.columns:
            analyzer.data[feature_name] = analyzer.data[num] / analyzer.data[den]

    try:
        print("=" * 60)
        print("相关性分析可视化")
        print("=" * 60)
        
        # 定义要分析的变量
        key_variables = ['calculated_KR', 'calculated_GR', 'calculated_CQ', 'calculated_MC', 
                        'TI46_accuracy', 'NARMA10_NRMSE', 'Nvirt']
        
        # 1. 相关性热图
        print("\n1. 生成相关性热图...")
        analyzer.plot_correlation_heatmap(variables=key_variables)
        
        # 2. 相关性条形图
        print("\n2. 生成相关性条形图...")
        analyzer.plot_correlation_bar('NARMA10_NRMSE', variables=key_variables, top_n=8)
        
        # 3. 散点图矩阵
        print("\n3. 生成散点图矩阵...")
        matrix_vars = ['calculated_KR', 'calculated_MC', 'TI46_accuracy', 'NARMA10_NRMSE']
        analyzer.plot_scatter_matrix(matrix_vars, target_var='NARMA10_NRMSE')
        
        # 4. 参数对详细分析
        print("\n4. 生成参数对详细分析图...")
        analyzer.plot_pair_correlations('calculated_MC', 'calculated_CQ', color_by='NARMA10_NRMSE')
        
        # 5. 相关性网络图（可选，需要networkx）
        # print("\n5. 生成相关性网络图...")
        # analyzer.plot_correlation_network(variables=key_variables, threshold=0.3)
        
        print("\n相关性分析完成！")
        return analyzer
        
    except Exception as e:
        print(f"相关性分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_pca_analysis():
    """
    执行PCA分析和可视化
    """
    # 创建分析器实例
    analyzer = ParameterCorrelationAnalyzer()
    analyzer.load_all_data()

    # 优雅地创建派生特征
    derived_features = {
        'KR_GR_ratio': ('calculated_KR', 'calculated_GR'),
        'task_balance': ('TI46_accuracy', 'NARMA10_NRMSE'),
        'MC_KR_ratio': ('calculated_MC', 'calculated_KR'),
        'MC_GR_ratio': ('calculated_MC', 'calculated_GR'),
        'MC_CQ_ratio': ('calculated_MC', 'calculated_CQ'),
        'MC_density': ('calculated_MC', 'Nvirt'),
        'CQ_density': ('calculated_CQ', 'Nvirt'),
        'GR_density': ('calculated_GR', 'Nvirt'),
        'KR_density': ('calculated_KR', 'Nvirt'),
    }
    
    for feature_name, (num, den) in derived_features.items():
        if num in analyzer.data.columns and den in analyzer.data.columns:
            analyzer.data[feature_name] = analyzer.data[num] / analyzer.data[den]

    try:
        print("=" * 60)
        print("主成分分析(PCA)可视化")
        print("=" * 60)
        
        # 定义核心任务相关变量
        task_vars = ['calculated_KR', 'calculated_GR', 'calculated_CQ', 'calculated_MC', 
                    'TI46_accuracy', 'NARMA10_NRMSE']
        
        # 检查变量是否存在
        available_vars = [var for var in task_vars if var in analyzer.data.columns]
        if len(available_vars) < 2:
            print("可用变量不足，无法进行PCA分析")
            return analyzer, None
            
        print(f"\n任务相关变量PCA分析...")
        print(f"分析变量: {available_vars}")
        
        # 执行PCA分析
        pca_results = analyzer.perform_pca_analysis(variables=available_vars, n_components=2)
        
        # 生成PCA可视化
        print("生成PCA散点图...")
        analyzer.plot_pca_scatter(pca_results, color_by='NARMA10_NRMSE')
        
        print("生成特征重要性图...")
        analyzer.plot_pca_feature_importance(pca_results)
        
        print("\nPCA分析完成！")
        return analyzer, pca_results
        
    except Exception as e:
        print(f"PCA分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """
    主函数：选择执行相关性分析或PCA分析
    """
    print("储层参数分析工具")
    print("1. 相关性分析")
    print("2. PCA分析")
    print("3. 全面分析（两者都执行）")
    
    choice = input("\n请选择分析类型 (1/2/3): ").strip()
    
    if choice == "1":
        run_correlation_analysis()
    elif choice == "2":
        run_pca_analysis()
    elif choice == "3":
        print("执行全面分析...")
        run_correlation_analysis()
        print("\n" + "="*60)
        run_pca_analysis()
    else:
        print("无效选择，执行全面分析...")
        run_correlation_analysis()
        print("\n" + "="*60)
        run_pca_analysis()


if __name__ == "__main__":
    main()

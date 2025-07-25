"""
Optuna Study数据保存工具使用示例
=====================================

展示如何使用optuna_study_saver模块保存和载入已完成的study数据，
以及如何进行后续的数据分析和可视化。

Author: Chen
Date: 2025-01-XX
"""

import os
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from optuna_study_saver import OptunaStudySaver, save_study, load_study

def get_first_available_study_name():
    """获取第一个可用的study名称"""
    storage = "sqlite:///db.sqlite3"
    try:
        study_summaries = optuna.get_all_study_summaries(storage=storage)
        if not study_summaries:
            raise ValueError("没有找到任何study，请先运行优化获得数据。")
        return study_summaries[0].study_name
    except Exception as e:
        raise ValueError(f"无法获取study列表: {e}")

def example_save_existing_study(study_name: str = None):
    """示例1: 保存现有的study数据"""
    print("=" * 60)
    print("示例1: 保存现有的study数据")
    print("=" * 60)
    
    # 检查是否提供了study名称
    if study_name is None:
        raise ValueError("必须提供study_name参数来指定要保存的study！")
    
    # 连接到现有的SQLite数据库
    storage = "sqlite:///db.sqlite3"
    
    try:
        # 获取所有可用的study
        study_summaries = optuna.get_all_study_summaries(storage=storage)
        
        if not study_summaries:
            print("没有找到任何study，请先运行优化获得数据。")
            return
        
        # 显示可用的study
        available_names = [summary.study_name for summary in study_summaries]
        print("可用的studies:")
        for i, name in enumerate(available_names):
            print(f"  {i+1}. {name}")
        
        # 检查指定的study是否存在
        if study_name not in available_names:
            raise ValueError(f"Study '{study_name}' 不存在！可用的studies: {available_names}")
        
        # 载入指定的study
        print(f"\n载入study: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        print(f"Study信息:")
        print(f"  试验总数: {len(study.trials)}")
        print(f"  完成试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        if hasattr(study, 'best_trials'):
            print(f"  Pareto最优解数: {len(study.best_trials)}")
        
        # 保存study数据
        print(f"\n保存study数据...")
        saved_files = save_study(
            study=study,
            save_dir="saved_studies",
            filename_prefix=study_name,
            formats=["json", "pickle", "csv"]
        )
        
        return saved_files
        
    except Exception as e:
        print(f"错误: {e}")
        return None

def example_load_and_analyze():
    """示例2: 载入并分析保存的数据"""
    print("\n" + "=" * 60)
    print("示例2: 载入并分析保存的数据")
    print("=" * 60)
    
    # 首先保存一个study（如果还没有的话）
    try:
        study_name = get_first_available_study_name()
        saved_files = example_save_existing_study(study_name)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    if not saved_files or "json" not in saved_files:
        print("没有可用的保存文件进行分析。")
        return
    
    # 载入JSON数据
    json_file = saved_files["json"]
    print(f"\n从文件载入数据: {json_file}")
    
    saver = OptunaStudySaver()
    study_data = saver.load_study_data(json_file)
    
    # 打印数据摘要
    print("\n数据摘要:")
    saver.print_summary(study_data)
    
    # 转换为DataFrame进行分析
    print("\n转换为DataFrame进行详细分析...")
    trials_df = saver.get_trials_dataframe(study_data)
    pareto_df = saver.get_pareto_dataframe(study_data)
    
    if not trials_df.empty:
        print(f"\nTrials DataFrame形状: {trials_df.shape}")
        print("列名:")
        for col in trials_df.columns:
            print(f"  - {col}")
        
        # 显示完成的试验统计
        completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
        if not completed_trials.empty:
            print(f"\n完成试验统计:")
            print(f"  完成试验数: {len(completed_trials)}")
            
            # 分析目标函数值
            if 'values' in completed_trials.columns:
                values_list = completed_trials['values'].tolist()
                if values_list and isinstance(values_list[0], list):
                    cq_values = [v[0] for v in values_list if len(v) > 0]
                    mc_values = [v[1] for v in values_list if len(v) > 1]
                    
                    print(f"  CQ范围: {min(cq_values):.4f} - {max(cq_values):.4f}")
                    print(f"  MC范围: {min(mc_values):.4f} - {max(mc_values):.4f}")
    
    if not pareto_df.empty:
        print(f"\nPareto前沿DataFrame形状: {pareto_df.shape}")
        print("Pareto最优解:")
        for i, row in pareto_df.iterrows():
            values = row['values']
            print(f"  解{i+1}: CQ={values[0]:.4f}, MC={values[1]:.4f}")
    
    return study_data, trials_df, pareto_df

def example_data_filtering():
    """示例3: 数据筛选和查询"""
    print("\n" + "=" * 60)
    print("示例3: 数据筛选和查询")
    print("=" * 60)
    
    # 获取数据
    result = example_load_and_analyze()
    if result is None:
        return
    
    study_data, trials_df, pareto_df = result
    
    saver = OptunaStudySaver()
    
    # 筛选完成的试验
    print("\n筛选完成的试验:")
    completed = saver.filter_trials(study_data, state="COMPLETE")
    print(f"  完成试验数: {len(completed)}")
    
    # 筛选特定形貌类型
    if 'attr_morph_type' in trials_df.columns:
        morph_types = trials_df['attr_morph_type'].unique()
        print(f"\n可用的形貌类型: {morph_types}")
        
        for morph_type in morph_types:
            if pd.notna(morph_type):
                filtered = saver.filter_trials(study_data, state="COMPLETE", morph_type=morph_type)
                print(f"  {morph_type}形貌试验数: {len(filtered)}")
    
    # 筛选高性能解
    print("\n筛选高性能解:")
    high_performance = saver.filter_trials(study_data, state="COMPLETE", min_cq=0.1, min_mc=0.5)
    print(f"  CQ>0.1且MC>0.5的试验数: {len(high_performance)}")

def example_visualization():
    """示例4: 数据可视化"""
    print("\n" + "=" * 60)
    print("示例4: 数据可视化")
    print("=" * 60)
    
    # 获取数据
    result = example_load_and_analyze()
    if result is None:
        return
    
    study_data, trials_df, pareto_df = result
    
    # 创建可视化
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Optuna Study数据分析', fontsize=16)
    
    # 提取完成的试验数据
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if not completed_trials.empty and 'values' in completed_trials.columns:
        values_list = completed_trials['values'].tolist()
        if values_list and isinstance(values_list[0], list):
            cq_values = [v[0] for v in values_list if len(v) > 0]
            mc_values = [v[1] for v in values_list if len(v) > 1]
            
            # 1. CQ-MC散点图
            axes[0, 0].scatter(cq_values, mc_values, alpha=0.6, s=30)
            if not pareto_df.empty:
                pareto_cq = [v[0] for v in pareto_df['values']]
                pareto_mc = [v[1] for v in pareto_df['values']]
                axes[0, 0].scatter(pareto_cq, pareto_mc, color='red', s=50, 
                                 label='Pareto Front', marker='*')
                axes[0, 0].legend()
            axes[0, 0].set_xlabel('CQ')
            axes[0, 0].set_ylabel('MC')
            axes[0, 0].set_title('CQ vs MC散点图')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. CQ分布直方图
            axes[0, 1].hist(cq_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('CQ')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].set_title('CQ值分布')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. MC分布直方图
            axes[1, 0].hist(mc_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('MC')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('MC值分布')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 试验收敛图
            trial_numbers = completed_trials['number'].tolist()
            axes[1, 1].plot(trial_numbers, cq_values, 'o-', alpha=0.6, markersize=3, label='CQ')
            axes[1, 1].plot(trial_numbers, mc_values, 's-', alpha=0.6, markersize=3, label='MC')
            axes[1, 1].set_xlabel('Trial Number')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('优化收敛过程')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    viz_dir = "saved_studies/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(os.path.join(viz_dir, "study_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"\n可视化图片已保存到: {os.path.join(viz_dir, 'study_analysis.png')}")
    
    plt.show()

def example_csv_analysis():
    """示例5: 使用CSV文件进行分析"""
    print("\n" + "=" * 60)
    print("示例5: 使用CSV文件进行分析")
    print("=" * 60)
    
    # 首先保存数据
    try:
        study_name = get_first_available_study_name()
        saved_files = example_save_existing_study(study_name)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    if not saved_files or "trials_csv" not in saved_files:
        print("没有可用的CSV文件。")
        return
    
    # 载入CSV数据
    trials_csv = saved_files["trials_csv"]
    print(f"\n从CSV文件载入数据: {trials_csv}")
    
    df = pd.read_csv(trials_csv)
    print(f"数据形状: {df.shape}")
    
    # 基本统计
    print("\n基本统计信息:")
    if 'values' in df.columns:
        # 解析values列（如果是字符串格式）
        import ast
        try:
            df['CQ'] = df['values'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0])
            df['MC'] = df['values'].apply(lambda x: ast.literal_eval(x)[1] if isinstance(x, str) else x[1])
            
            print(f"  CQ统计: 均值={df['CQ'].mean():.4f}, 标准差={df['CQ'].std():.4f}")
            print(f"  MC统计: 均值={df['MC'].mean():.4f}, 标准差={df['MC'].std():.4f}")
            
            # 找到最佳平衡点
            df['combined_score'] = df['CQ'] + df['MC']  # 简单相加作为综合得分
            best_idx = df['combined_score'].idxmax()
            print(f"\n最佳综合得分试验:")
            print(f"  Trial {df.loc[best_idx, 'number']}: CQ={df.loc[best_idx, 'CQ']:.4f}, MC={df.loc[best_idx, 'MC']:.4f}")
            
        except Exception as e:
            print(f"解析values列时出错: {e}")

# if __name__ == "__main__":
#     """运行所有示例"""
#     print("Optuna Study数据保存工具使用示例")
#     print("=" * 60)
    
#     # 确保保存目录存在
#     os.makedirs("saved_studies", exist_ok=True)
    
#     try:
#         # 获取第一个可用的study进行演示
#         study_name = get_first_available_study_name()
#         print(f"使用study进行演示: {study_name}")
        
#         # 运行各个示例
#         example_save_existing_study(study_name)
#         example_load_and_analyze()
#         example_data_filtering()
#         example_visualization()
#         example_csv_analysis()
        
#         print("\n" + "=" * 60)
#         print("所有示例运行完成！")
#         print("查看saved_studies目录获取保存的数据和可视化结果。")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"运行示例时出错: {e}")
#         print("请确保已经运行过Optuna优化并生成了db.sqlite3文件。")

# 直接使用示例
def save_specific_study_example():
    """
    直接保存指定study的示例
    使用方法：
    - 替换 'your_study_name' 为实际的study名称
    - 运行此函数即可保存该study的数据
    """
    
    # 指定要保存的study名称
    target_study_name = "Reservoir_Morphology_CQ_MC_Pareto_uniform_2"  # 替换为你的study名称
    
    try:
        # 保存指定的study
        saved_files = example_save_existing_study(target_study_name)
        
        if saved_files:
            print(f"\n成功保存study '{target_study_name}' 的数据！")
            print("保存的文件:")
            for format_type, filepath in saved_files.items():
                print(f"  {format_type}: {filepath}")
        else:
            print(f"保存study '{target_study_name}' 失败。")
            
    except ValueError as e:
        print(f"错误: {e}")
        print("\\n可用的study列表:")
        try:
            storage = "sqlite:///db.sqlite3"
            study_summaries = optuna.get_all_study_summaries(storage=storage)
            for i, summary in enumerate(study_summaries, 1):
                print(f"  {i}. {summary.study_name}")
        except Exception:
            print("  无法获取study列表")

# 取消注释以下行来直接运行保存指定study的示例
save_specific_study_example()
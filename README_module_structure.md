# 储层形貌比较框架 - 模块化结构

## 概述

原始的 `formal_reservoirs_framework.py` 文件已经被拆分为四个独立的模块，以提高代码的可维护性和模块化程度。

## 新的模块结构

### 1. `reservoir_morphology_creator.py`
**功能**: 储层创造和配置管理
- `MorphologyConfig`: 储层形貌配置类
- `ReservoirMorphologyManager`: 储层形貌管理器
- `create_standard_morphology_configs()`: 创建标准形貌配置的便捷函数

**主要用途**:
- 定义和配置不同类型的储层形貌（均质、渐变、随机）
- 管理储层实例的创建
- 生成权重和变换函数

### 2. `reservoir_evaluation.py`
**功能**: 储层性能评估
- `evaluate_heterogeneous_MC()`: 评估异质储层内存容量
- `evaluate_heterogeneous_KRandGR()`: 评估异质储层KR和GR
- `evaluate_reservoir_performance()`: 综合评估储层性能

**主要用途**:
- 计算储层的内存容量 (MC)
- 计算储层的核等级 (KR) 和广义等级 (GR)
- 计算计算质量 (CQ = KR - GR)

### 3. `optuna_optimization.py`
**功能**: 基于Optuna的超参数优化
- `SingleMorphologyObjective`: 单形貌储层目标函数
- `SingleMorphologyParetoOptimizer`: 单形貌储层Pareto前沿优化器
- `run_single_morphology_optimization()`: 单形貌优化的便捷函数

**主要用途**:
- 对单个形貌储层进行超参数优化
- 生成Pareto前沿
- 保存和分析优化结果

### 4. `reservoir_comparison_framework.py`
**功能**: 多形貌储层比较的主集成框架
- `run_sequential_morphology_comparison()`: 串行比较多种形貌储层
- `create_comparison_summary()`: 创建比较总结数据
- `load_comparison_results()`: 加载比较结果
- `analyze_comparison_results()`: 分析比较结果

**主要用途**:
- 串行运行多种形貌的储层优化
- 整合和比较不同形貌的结果
- 数据分析和可视化准备

### 5. `formal_reservoirs_framework.py` (简化版)
**功能**: 向后兼容接口
- 导入所有新模块的主要功能
- 提供向后兼容的函数名
- 使用指南和示例

## 模块间依赖关系

```
reservoir_morphology_creator.py
    ↓
reservoir_evaluation.py
    ↓
optuna_optimization.py
    ↓
reservoir_comparison_framework.py
    ↓
formal_reservoirs_framework.py (legacy interface)
```

## 使用方法

### 方法 1: 使用集成框架（推荐）

```python
from reservoir_comparison_framework import run_sequential_morphology_comparison

# 运行完整的多形貌比较
all_results, comparison_data = run_sequential_morphology_comparison(
    n_trials=100,
    study_name_prefix="MyComparison"
)
```

### 方法 2: 单形貌优化

```python
from optuna_optimization import run_single_morphology_optimization
from reservoir_morphology_creator import MorphologyConfig

# 创建形貌配置
config = MorphologyConfig(morph_type='gradient', n_instances=5, beta_range=(20, 30))

# 运行优化
study, optimizer, results = run_single_morphology_optimization(config, n_trials=200)
```

### 方法 3: 手动控制

```python
from reservoir_morphology_creator import MorphologyConfig, ReservoirMorphologyManager
from reservoir_evaluation import evaluate_reservoir_performance
from formal_Parameter_Dynamics_Preformance import ReservoirParams

# 创建储层参数
reservoir_params = ReservoirParams(h=0.4, m0=0.003, Nvirt=30, beta_prime=25)

# 创建形貌配置
config = MorphologyConfig(morph_type='homogeneous')

# 评估性能
performance = evaluate_reservoir_performance(reservoir_params, config)
print(f"MC: {performance['MC']}, CQ: {performance['CQ']}")
```

## 向后兼容性

为了保持向后兼容性，原始的 `formal_reservoirs_framework.py` 文件仍然存在，但现在只是一个导入和转发接口。现有代码应该能够继续工作，但建议迁移到新的模块化结构。

## 优势

1. **模块化**: 每个文件专注于特定功能
2. **可维护性**: 更容易定位和修改特定功能
3. **可扩展性**: 更容易添加新的储层类型或评估方法
4. **可测试性**: 可以独立测试每个模块
5. **代码复用**: 可以在不同项目中重用特定模块

## 示例文件

- `example_usage.py`: 包含各种使用方法的完整示例

## 注意事项

1. 确保所有依赖模块在同一目录下或在Python路径中
2. 新的模块结构需要 `formal_Parameter_Dynamics_Preformance.py` 模块
3. 结果文件格式与原版本保持兼容
4. 所有原有的功能都已保留，只是重新组织

## 迁移指南

如果您之前使用的是原始的 `formal_reservoirs_framework.py`：

1. **无需更改**: 如果只是导入和使用主要函数，代码应该继续工作
2. **建议迁移**: 为了更好的性能和可维护性，建议使用新的模块化接口
3. **新功能**: 可以使用新的便捷函数简化常见任务

## 联系信息

如有问题或建议，请联系：Chen (作者) 
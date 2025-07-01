# Pareto Front全任务评估框架

这个框架能够从Pareto Front数据中提取参数，创建均质储层，计算所有性能指标，执行NARMA10和TI46任务，并保存所有数据和图像。

## 功能特性

✅ **参数提取**: 从Pareto Front CSV文件中自动提取参数  
✅ **储层创建**: 根据提取的参数创建均质储层  
✅ **指标计算**: 计算MC (Memory Capacity)、KR、GR、CQ (Computational Quality)  
✅ **NARMA10任务**: 执行非线性自回归移动平均任务  
✅ **TI46任务**: 执行语音识别任务  
✅ **数据保存**: 保存所有结果为pickle和CSV格式  
✅ **图像生成**: 自动生成性能对比图和NARMA10预测图  

## 文件结构

```
📁 results/ParetoFront_AllTI/
├── Point_97_CQ5.0_MC6.1/
│   ├── ParetoFront_AllTI_CQ5.0_MC6.1.pkl    # 完整结果数据
│   ├── ParetoFront_AllTI_CQ5.0_MC6.1.csv    # 结果摘要
│   ├── ParetoFront_AllTI_CQ5.0_MC6.1_NARMA10.png  # NARMA10图像
│   └── ParetoFront_AllTI_CQ5.0_MC6.1_TI46.png     # 性能摘要图
├── Point_269_CQ4.0_MC6.2/
│   └── ...
└── processing_summary.pkl  # 处理过程总结
```

## 快速开始

### 1. 测试模式（推荐首次使用）

```bash
python run_pareto_framework.py --test
```

这将处理前3个Pareto Points作为测试。

### 2. 处理指定范围

```bash
# 处理前5个点
python run_pareto_framework.py --start 0 --end 5

# 处理第10-20个点
python run_pareto_framework.py --start 10 --end 20
```

### 3. 处理所有点

```bash
python run_pareto_framework.py --all
```

⚠️ **注意**: 处理所有点可能需要数小时时间！

### 4. 使用自定义CSV文件

```bash
python run_pareto_framework.py --csv your_pareto_data.csv --test
```

## 输出结果说明

### 数据文件

#### Pickle文件 (`.pkl`)
包含完整的结果数据：
- `timestamp`: 处理时间戳
- `original_params`: 原始Pareto Front参数
- `calculated_metrics`: 计算得到的指标 (MC, KR, GR, CQ)
- `narma10_results`: NARMA10任务结果 (NRMSE, 预测数据)
- `ti46_results`: TI46任务结果 (accuracy)

#### CSV文件 (`.csv`)
便于快速查看的摘要数据，包含所有关键指标。

### 图像文件

#### NARMA10图像 (`_NARMA10.png`)
- 左图: 预测值 vs 真实值散点图
- 右图: 时间序列对比图

#### 性能摘要图 (`_TI46.png`)
- 显示MC、CQ、NARMA10、TI46的归一化性能
- 包含参数信息标注

## 程序流程

对于每个Pareto Point，框架会执行以下步骤：

1. **🔧 提取参数**: 从CSV文件中提取gamma, theta, m0, h, beta_prime, Nvirt
2. **🏗️ 创建储层**: 根据参数创建均质储层
3. **📊 计算指标**: 计算MC, KR, GR, CQ
4. **🎯 执行NARMA10**: 运行非线性自回归移动平均任务
5. **🗣️ 执行TI46**: 运行语音识别任务
6. **💾 保存结果**: 保存数据为pickle和CSV格式
7. **🖼️ 保存图像**: 生成并保存性能图像

## 性能指标说明

- **MC (Memory Capacity)**: 储层的记忆容量，越高越好
- **KR (Kernel Rank)**: 核等级
- **GR (Generalization Rank)**: 泛化等级  
- **CQ (Computational Quality)**: 计算质量 = KR - GR，越高越好
- **NRMSE**: 归一化均方根误差，越低越好
- **Accuracy**: TI46语音识别准确率，越高越好

## 错误处理

框架具有完善的错误处理机制：
- 如果某个任务失败，会记录错误信息但继续处理其他任务
- 所有错误都会保存在结果文件中
- 处理过程中可以随时按Ctrl+C中断

## 处理时间估算

- 每个Pareto Point大约需要 **2-5分钟**
- 11个点总共大约需要 **30-60分钟**
- 具体时间取决于：
  - 计算机性能
  - 储层复杂度 (Nvirt大小)
  - 网络条件 (如需下载TI46数据)

## 注意事项

1. **首次运行TI46任务**时可能需要下载语音数据，请确保网络连接正常
2. **内存使用**: 大型储层 (Nvirt > 500) 可能消耗较多内存
3. **磁盘空间**: 每个点约占用5-10MB存储空间
4. **依赖库**: 确保已安装所有必要的Python库

## 示例运行输出

```
🚀 Pareto Front全任务评估框架
==================================================
✅ 成功加载Pareto Front数据
📊 数据包含 11 个参数组合
📋 列名: ['number', 'CQ', 'MC', 'gamma', 'theta', 'm0', 'h', 'beta_prime', 'Nvirt']

================================================================================
处理Pareto Point #1
================================================================================
🔧 1. 提取参数...
  📋 参数: CQ=5.0, MC=6.1, γ=0.054, θ=0.084
🏗️ 2. 创建均质储层...
  ✅ 储层创建成功: Nvirt=378, β'=47.9
📊 3. 计算储层指标...
    📈 计算MC...
    📊 计算KR和GR...
    ✅ 指标计算完成: MC=6.118, KR=162.0, GR=157.0, CQ=5.0
🎯 4. 执行NARMA10任务...
    ✅ NARMA10任务完成: NRMSE=0.3456
🗣️ 5. 执行TI46任务...
    ✅ TI46任务完成: Accuracy=0.8234
💾 6. 保存结果...
    💾 结果已保存: ParetoFront_AllTI_CQ5.0_MC6.1.pkl
    📄 CSV摘要已保存: ParetoFront_AllTI_CQ5.0_MC6.1.csv
🖼️ 7. 保存图像...
    🖼️ NARMA10图像已保存: ParetoFront_AllTI_CQ5.0_MC6.1_NARMA10.png
    🖼️ 性能摘要图像已保存: ParetoFront_AllTI_CQ5.0_MC6.1_TI46.png

✅ Pareto Point #1 处理完成!
⏱️ 总耗时: 243.2 秒
📁 结果保存在: results/ParetoFront_AllTI/Point_97_CQ5.0_MC6.1
```

## 故障排除

### 常见问题

1. **ImportError**: 确保所有依赖库已正确安装
2. **FileNotFoundError**: 检查Pareto Front CSV文件路径是否正确
3. **内存错误**: 尝试处理较少的点或使用较小的Nvirt参数
4. **TI46数据下载失败**: 检查网络连接，或手动下载TI46数据集

### 获取帮助

```bash
python run_pareto_framework.py --help
```

## 开发者信息

- **Author**: Chen
- **Date**: 2025-01-XX
- **Framework**: pareto_framework_complete.py
- **Runner**: run_pareto_framework.py

## 更新日志

- v1.0: 初始版本，支持完整的Pareto Front分析流程 
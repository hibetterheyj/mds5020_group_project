# SVM and KNN Model Implementation ｜ SVM和KNN模型实现

## 中文说明

### 项目概述

本文件夹包含完整的机器学习流水线，用于银行营销分类任务，实现了支持向量机(SVM)和K最近邻(KNN)两种算法。流水线包括数据加载、预处理、超参数调优、模型训练、预测和结果可视化。

使用GridSearchCV重新实现了超参数调优功能，大幅提高了搜索效率，并优化了可视化方式以更好地展示参数组合的影响。

### 文件结构

```text
├── archive/                      # 存档文件夹，用于存储历史版本或不再使用的文件
├── data_loader.py                # 加载和准备训练/测试数据集
├── eda.ipynb                     # 探索性数据分析jupyter笔记本
├── eda.py                        # 探索性数据分析脚本
├── hyperparameter_tuner.py       # 执行超参数调优和可视化，支持SVM和KNN
├── main_kernel_svm.py            # 运行核SVM流水线的主脚本
├── main_knn.py                   # 运行KNN流水线的主脚本
├── main_linear_svm.py            # 运行线性SVM流水线的主脚本
├── model_kernel_svm.py           # 核SVM模型实现（SVC）
├── model_knn.py                  # KNN模型实现与交叉验证
├── model_linear_svm.py           # 线性SVM模型实现（LinearSVC）
├── preprocessor.py               # 处理和清洗数据（填充、标准化、编码）
├── README.md                     # 项目说明文档
├── res/                          # 结果存储目录，包含实验生成的结果文件
├── results_handler.py            # 管理预测结果保存和报告生成
└── test_hyperparameter_tuner.py  # 超参数调优器测试脚本
```

### 主要组件说明

- **res/目录**：专门用于存储实验生成的各种结果文件，包括超参数调优结果JSON文件、可视化图表等
- **test_hyperparameter_tuner.py**：测试脚本，用于验证超参数调优器的功能正确性
- **archive/目录**：用于存档不再使用但可能需要保留的历史文件，保持项目结构整洁

### 使用方法

1. **环境要求**：Python 3.13.5，所需库：pandas, numpy, scikit-learn, matplotlib, seaborn
2. **运行模型**：
   - KNN模型：`python main_knn.py`
   - 核SVM模型：`python main_kernel_svm.py`
   - 线性SVM模型：`python main_linear_svm.py`
3. **自定义**：可在对应主脚本中修改数据路径和输出路径
4. **测试超参数调优**：执行 `python test_hyperparameter_tuner.py` 验证GridSearchCV功能

### 模型特性与优化

**核心优化**：
- 使用`GridSearchCV`重写了超参数调优功能，支持完全并行化的参数搜索
- 新增`HyperparameterTuner.tune_parameters_with_gridsearch`方法，设置`n_jobs=-2`利用多核心处理
- 优化了可视化方式，提高了参数调优结果的可读性
- 所有模型统一使用5-fold cross-validation进行稳健评估，主要评价指标为ROC AUC

**1. KNN模型**：
   - 在`model_knn.py`中实现，使用`main_knn.py`运行
   - 优化的可视化：x轴为n_neighbors参数从小到大，不同颜色/线型表示weights和p参数的不同组合
   - 支持weights（'uniform'/'distance'）和p（1/2）参数的组合调优

**2. 核SVM (SVC)**：
   - 在`model_kernel_svm.py`中实现，使用`main_kernel_svm.py`运行
   - 支持使用各种核函数（RBF、Sigmoid）的非线性分类
   - 优化的可视化：x轴为C参数从小到大（log scale），不同颜色/线型表示其他参数组合
   - 支持kernel、gamma、C参数的组合调优

**3. 线性SVM (LinearSVC)**：
   - 在`model_linear_svm.py`中实现，使用`main_linear_svm.py`运行
   - 针对线性分类任务优化，计算效率更高
   - 优化的可视化：x轴为C参数从小到大（log scale），不同颜色/线型表示penalty和loss参数的不同组合

## English Documentation

### Project Overview

This folder contains complete machine learning pipelines for the bank marketing classification task, implemented using both Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) algorithms. The pipelines include data loading, preprocessing, hyperparameter tuning, model training, prediction, and result visualization.

Reimplemented hyperparameter tuning using GridSearchCV for significantly improved search efficiency and optimized visualization methods to better demonstrate the impact of parameter combinations.

### File Structure

```text
├── archive/                      # Archive folder for historical versions or unused files
├── data_loader.py                # Handles loading and preparing training/test datasets
├── eda.ipynb                     # Jupyter notebook for interactive data analysis
├── eda.py                        # Exploratory data analysis script
├── hyperparameter_tuner.py       # Performs hyperparameter tuning and visualization for both SVM and KNN
├── main_kernel_svm.py            # Main script to run the Kernel SVM pipeline
├── main_knn.py                   # Main script to run the KNN pipeline
├── main_linear_svm.py            # Main script to run the Linear SVM pipeline
├── model_kernel_svm.py           # Kernel SVM model implementation (SVC)
├── model_knn.py                  # KNN model implementation with cross-validation
├── model_linear_svm.py           # Linear SVM model implementation (LinearSVC)
├── preprocessor.py               # Processes and cleans data (imputation, scaling, encoding)
├── README.md                     # Project documentation
├── res/                          # Results storage directory containing experiment results
├── results_handler.py            # Manages saving predictions and generating reports
└── test_hyperparameter_tuner.py  # Test script for hyperparameter tuner
```

### Key Components Description

- **res/ directory**: Dedicated for storing various experiment result files, including hyperparameter tuning result JSON files, visualization charts, etc.
- **test_hyperparameter_tuner.py**: Test script used to verify the functionality correctness of the hyperparameter tuner
- **archive/ directory**: Used for archiving historical files that are no longer in use but might need to be preserved, maintaining a clean project structure

### Usage

1. **Prerequisites**: Python 3.13.5, Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
2. **Running Models**:
   - KNN Model: `python main_knn.py`
   - Kernel SVM Model: `python main_kernel_svm.py`
   - Linear SVM Model: `python main_linear_svm.py`
3. **Customization**: Modify data and output paths in the respective main scripts as needed
4. **Testing Hyperparameter Tuning**: Execute `python test_hyperparameter_tuner.py` to verify GridSearchCV functionality

### Model Features and Optimizations

**Core Optimizations**：
- Reimplemented hyperparameter tuning using `GridSearchCV` for fully parallelized parameter search
- Added new `HyperparameterTuner.tune_parameters_with_gridsearch` method with `n_jobs=-2` to utilize multiple cores
- Optimized visualization methods for improved readability of tuning results
- All models use 5-fold cross-validation for robust evaluation with ROC AUC as the main metric

**1. KNN Model**：
   - Implemented in `model_knn.py`, run with `main_knn.py`
   - Optimized visualization: x-axis shows n_neighbors from smallest to largest, with different colors/line types representing different combinations of weights and p parameters
   - Supports combined tuning of weights ('uniform'/'distance') and p (1/2) parameters

**2. Kernel SVM (SVC)**：
   - Implemented in `model_kernel_svm.py`, run with `main_kernel_svm.py`
   - Supports non-linear classification using various kernels (RBF, Sigmoid)
   - Optimized visualization: x-axis shows C parameter from smallest to largest (log scale), with different colors/line types representing other parameter combinations
   - Supports combined tuning of kernel, gamma, and C parameters

**3. Linear SVM (LinearSVC)**：
   - Implemented in `model_linear_svm.py`, run with `main_linear_svm.py`
   - Optimized for linear classification tasks with higher computational efficiency
   - Optimized visualization: x-axis shows C parameter from smallest to largest (log scale), with different colors/line types representing different combinations of penalty and loss parameters

### 注意事项

- 建议使用优化后的 `tune_parameters_with_gridsearch` 方法代替旧的 `tune_hyperparameters` 方法，性能显著提升。新方法利用并行处理将超参数调优速度提升数倍至十几倍。
- 可视化方法已更新，支持参数轴和对数刻度，提高了调优结果的可解释性。
- GridSearchCV 中的 `n_jobs=-2` 参数确保使用除一个核心外的所有 CPU 核心进行并行处理，这有助于避免系统过载，同时仍能获得最佳性能。
- 在大型数据集上运行模型时，确保有足够的内存，尤其是在使用包含多个参数组合的 GridSearchCV 时。

### Notes

- It is recommended to use the optimized `tune_parameters_with_gridsearch` method instead of the older `tune_hyperparameters` method for significantly better performance. The new method leverages parallel processing to speed up hyperparameter tuning by several to dozens of times.
- The visualization methods have been updated to support parameter axes and logarithmic scales, improving the interpretation of tuning results.
- The `n_jobs=-2` parameter in GridSearchCV ensures that all CPU cores except one are used for parallel processing, which helps to avoid overloading the system while still achieving optimal performance.
- When running the model on large datasets, ensure that sufficient memory is available, especially when using GridSearchCV with multiple parameter combinations.

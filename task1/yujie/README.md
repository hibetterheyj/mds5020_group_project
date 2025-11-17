# SVM and KNN Model Implementation ｜ SVM和KNN模型实现

## 中文说明

### 项目概述

本文件夹包含完整的机器学习流水线，用于银行营销分类任务，实现了支持向量机(SVM)和K最近邻(KNN)两种算法。流水线包括数据加载、预处理、超参数调优、模型训练、预测和结果可视化。

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
├── res/                          # 结果存储目录，包含实验生成的结果文件（如linear_svm_tuning_results.json）
├── results_handler.py            # 管理预测结果保存和报告生成
└── test_hyperparameter_tuner.py  # 超参数调优器测试脚本，用于验证hyperparameter_tuner.py的功能
```

### 主要组件说明

- **res/目录**：专门用于存储实验生成的各种结果文件，包括超参数调优结果JSON文件、可视化图表等。
- **test_hyperparameter_tuner.py**：测试脚本，用于验证超参数调优器的功能正确性，确保不同算法的超参数调优过程能够正常工作。
- **archive/目录**：用于存档不再使用但可能需要保留的历史文件，保持项目结构的整洁。

### 使用方法

1. **环境要求**：Python 3.13.5，所需库：pandas, numpy, scikit-learn, matplotlib, seaborn
2. **运行KNN模型**：执行主脚本 `python main_knn.py`
3. **运行核SVM模型**：执行主脚本 `python main_kernel_svm.py`
4. **运行线性SVM模型**：执行主脚本 `python main_linear_svm.py`
5. **自定义**：可在对应主脚本中修改数据路径和输出路径

### SVM模型特性

1. **核SVM (SVC)**：
   - 在`model_kernel_svm.py`中实现，使用`main_kernel_svm.py`运行
   - 支持使用各种核函数（RBF、Sigmoid）的非线性分类
   - 实现了对gamma、C和kernel参数的超参数调优
   - 超参数调优结果可视化

2. **线性SVM (LinearSVC)**：
   - 在`model_linear_svm.py`中实现，使用`main_linear_svm.py`运行
   - 针对线性分类任务优化，计算效率更高
   - 实现了对penalty、loss和C参数的超参数调优

## English Documentation

### Project Overview

This folder contains complete machine learning pipelines for the bank marketing classification task, implemented using both Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) algorithms. The pipelines include data loading, preprocessing, hyperparameter tuning, model training, prediction, and result visualization.

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
├── res/                          # Results storage directory containing experiment results (e.g., linear_svm_tuning_results.json)
├── results_handler.py            # Manages saving predictions and generating reports
└── test_hyperparameter_tuner.py  # Test script for hyperparameter tuner to verify its functionality
```

### Key Components Description

- **res/ directory**: Dedicated for storing various experiment result files, including hyperparameter tuning result JSON files, visualization charts, etc.
- **test_hyperparameter_tuner.py**: Test script used to verify the functionality correctness of the hyperparameter tuner, ensuring that hyperparameter tuning processes for different algorithms work properly.
- **archive/ directory**: Used for archiving historical files that are no longer in use but might need to be preserved, maintaining a clean project structure.

### Usage

1. **Prerequisites**: Python 3.13.5, Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
2. **Running the KNN Model**: Execute the main script `python main_knn.py`
3. **Running the Kernel SVM Model**: Execute the main script `python main_kernel_svm.py`
4. **Running the Linear SVM Model**: Execute the main script `python main_linear_svm.py`
5. **Customization**: Modify data and output paths in the respective main scripts as needed

### SVM Model Features

1. **Kernel SVM (SVC)**：
   - Implemented in `model_kernel_svm.py`, run with `main_kernel_svm.py`
   - Supports non-linear classification using various kernels (RBF, Sigmoid)
   - Implements hyperparameter tuning for gamma, C, and kernel parameters
   - Visualization of hyperparameter tuning results

2. **Linear SVM (LinearSVC)**：
   - Implemented in `model_linear_svm.py`, run with `main_linear_svm.py`
   - Optimized for linear classification tasks with higher computational efficiency
   - Implements hyperparameter tuning for penalty, loss, and C parameters

### Performance Metrics

- Area Under the Receiver Operating Characteristic Curve (ROC AUC)
- 5-fold cross-validation for robust evaluation

### Notes

- Both models use stratified k-fold cross-validation
- The pipeline includes robust error handling throughout the process
- Modular design allows for easy extension with new models while maintaining code consistency

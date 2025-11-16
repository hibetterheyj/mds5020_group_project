# SVM and KNN Model Implementation ｜ SVM和KNN模型实现

## 中文说明

### 项目概述

本文件夹包含完整的机器学习流水线，用于银行营销分类任务，实现了支持向量机(SVM)和K最近邻(KNN)两种算法。流水线包括数据加载、预处理、超参数调优、模型训练、预测和结果可视化。

### 文件结构

```text
├── data_loader.py            # 加载和准备训练/测试数据集
├── preprocessor.py           # 处理和清洗数据（填充、标准化、编码）
├── hyperparameter_tuner.py   # 执行超参数调优和可视化，支持SVM和KNN
├── model_knn.py              # KNN模型实现与交叉验证
├── model_svm.py              # SVM模型实现，支持SVC和LinearSVC
├── main_knn.py               # 运行KNN流水线的主脚本
├── main_svm.py               # 运行SVM流水线的主脚本
├── results_handler.py        # 管理预测结果保存和报告生成
├── eda.py                    # 探索性数据分析脚本
├── eda.ipynb                 # 探索性数据分析jupyter笔记本
├── knn_hyperparameter_tuning.png    # KNN超参数调优结果可视化
├── svm_hyperparameter_tuning.png    # SVM超参数调优结果可视化
├── knn_hyperparameter_tuning_data.json  # KNN超参数调优原始数据
└── svm_hyperparameter_tuning_data.json  # SVM超参数调优原始数据
```

### 使用方法

1. **环境要求**：Python 3.13.5，所需库：pandas, numpy, scikit-learn, matplotlib, seaborn
2. **运行KNN模型**：执行主脚本 `python main_knn.py`
3. **运行SVM模型**：执行主脚本 `python main_svm.py`
4. **自定义**：可在对应主脚本中修改数据路径和输出路径
5. **切换SVM类型**：在`main_svm.py`中可通过修改`model_type`参数切换SVC和LinearSVC

### SVM模型特性

1. **SVC (支持向量分类器)**：
   - 支持使用各种核函数的非线性分类
   - 实现了对gamma、C和kernel参数的超参数调优
   - 超参数调优结果可视化

2. **LinearSVC**：
   - 针对线性分类任务优化
   - 实现了对penalty、loss和C参数的超参数调优

## English Documentation

### Project Overview

This folder contains complete machine learning pipelines for the bank marketing classification task, implemented using both Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) algorithms. The pipelines include data loading, preprocessing, hyperparameter tuning, model training, prediction, and result visualization.

### File Structure

```text
├── data_loader.py            # Handles loading and preparing training/test datasets
├── preprocessor.py           # Processes and cleans data (imputation, scaling, encoding)
├── hyperparameter_tuner.py   # Performs hyperparameter tuning and visualization for both SVM and KNN
├── model_knn.py              # KNN model implementation with cross-validation
├── model_svm.py              # SVM model implementation with SVC and LinearSVC support
├── results_handler.py        # Manages saving predictions and generating reports
├── main_knn.py               # Main script to run the KNN pipeline
├── main_svm.py               # Main script to run the SVM pipeline
├── eda.py                    # Exploratory data analysis script
├── eda.ipynb                 # Jupyter notebook for interactive data analysis
├── knn_hyperparameter_tuning.png    # Visualization of KNN hyperparameter tuning results
├── svm_hyperparameter_tuning.png    # Visualization of SVM hyperparameter tuning results
├── knn_hyperparameter_tuning_data.json  # Raw data from KNN hyperparameter tuning
└── svm_hyperparameter_tuning_data.json  # Raw data from SVM hyperparameter tuning
```

### Usage

1. **Prerequisites**: Python 3.13.5, Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
2. **Running the KNN Model**: Execute the main script `python main_knn.py`
3. **Running the SVM Model**: Execute the main script `python main_svm.py`
4. **Customization**: Modify data and output paths in the respective main scripts as needed
5. **Switching SVM Type**: In `main_svm.py`, you can switch between SVC and LinearSVC by modifying the `model_type` parameter

### SVM Model Features

1. **SVC (Support Vector Classifier)**:
   - Supports non-linear classification using various kernels
   - Implements hyperparameter tuning for gamma, C, and kernel parameters
   - Visualization of hyperparameter tuning results

2. **LinearSVC**:
   - Optimized for linear classification tasks
   - Implements hyperparameter tuning for penalty, loss, and C parameters

### Performance Metrics

- Area Under the Receiver Operating Characteristic Curve (ROC AUC)
- 5-fold cross-validation for robust evaluation

### Notes

- Both models use stratified k-fold cross-validation
- The pipeline includes robust error handling throughout the process
- Modular design allows for easy extension with new models while maintaining code consistency

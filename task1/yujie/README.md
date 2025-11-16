# KNN分类模型 ｜ KNN Classification Model

## 中文说明

### 项目概述

本文件夹包含一个完整的机器学习流水线，用于银行营销分类任务，实现了K最近邻(KNN)算法。流水线包括数据加载、预处理、超参数调优、模型训练、预测和结果可视化。

### 文件结构

```text
├── data_loader.py            # 加载和准备训练/测试数据集
├── preprocessor.py           # 处理和清洗数据（填充、标准化、编码）
├── hyperparameter_tuner.py   # 执行超参数调优和可视化
├── knn_model.py              # KNN模型实现与交叉验证
├── results_handler.py        # 管理预测结果保存和报告生成
├── knn_main.py               # 运行整个流水线的主脚本
├── eda.py                    # 探索性数据分析脚本
├── eda.ipynb                 # 探索性数据分析jupyter笔记本
├── knn_hyperparameter_tuning.png    # 超参数调优结果可视化
└── knn_hyperparameter_tuning_data.json  # 超参数调优原始数据
```

### 使用方法

1. **环境要求**：Python 3.13.5，所需库：pandas, numpy, scikit-learn, matplotlib
2. **运行模型**：执行主脚本 `python knn_main.py`
3. **自定义**：可在`knn_main.py`中修改数据路径和输出路径

## English Documentation

### Project Overview

This folder contains a complete machine learning pipeline for the bank marketing classification task, implemented using a K-Nearest Neighbors (KNN) algorithm. The pipeline includes data loading, preprocessing, hyperparameter tuning, model training, prediction, and result visualization.

### File Structure

```text
├── data_loader.py            # Handles loading and preparing training/test datasets
├── preprocessor.py           # Processes and cleans data (imputation, scaling, encoding)
├── hyperparameter_tuner.py   # Performs hyperparameter tuning and visualization
├── knn_model.py              # KNN model implementation with cross-validation
├── results_handler.py        # Manages saving predictions and generating reports
├── knn_main.py               # Main script to run the entire pipeline
├── eda.py                    # Exploratory data analysis script
├── eda.ipynb                 # Jupyter notebook for interactive data analysis
├── knn_hyperparameter_tuning.png    # Visualization of hyperparameter tuning results
└── knn_hyperparameter_tuning_data.json  # Raw data from hyperparameter tuning
```

### Usage

1. **Prerequisites**: Python 3.13.5, Required libraries: pandas, numpy, scikit-learn, matplotlib
2. **Running the Model**: Execute the main script `python knn_main.py`
3. **Customization**: Modify data and output paths in `knn_main.py` as needed

### Performance Metrics

- Area Under the Receiver Operating Characteristic Curve (ROC AUC)
- 5-fold cross-validation for robust evaluation

### Notes

- The model uses stratified k-fold cross-validation
- Hyperparameters tuned include: number of neighbors (k), weighting strategy, and distance metric
- The pipeline includes robust error handling throughout the process

import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional, Any
from model_knn import KNNModel
from model_kernel_svm import KernelSVMModel
from model_linear_svm import LinearSVMModel


# Create a simple test dataset
def create_test_data(n_samples: int = 100, n_features: int = 10, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # 创建一个简单的线性可分问题
    weights = np.random.randn(n_features)
    y = (X.dot(weights) > 0).astype(int)
    return X, y

# Test KNN model hyperparameter tuning and CSV export


def test_knn_model() -> None:
    print("===== Testing KNN Model =====")
    X, y = create_test_data()

    # 创建KNN模型实例
    knn_model = KNNModel()

    # 运行参数调优 (使用GridSearchCV)
    csv_path = "../tests/hyperparameter_tuning_data.csv"
    print("\n测试KNN模型使用GridSearchCV:")
    knn_model.tune_hyperparameters(X, y, cv_folds=3, save_results=True,
                                   export_format='csv', results_file_path=csv_path)

    if os.path.exists(csv_path):
        print(f"\nCSV file generated successfully: {csv_path}")
        # 读取并显示CSV文件的前几行
        df = pd.read_csv(csv_path)
        print(f"CSV文件形状: {df.shape}")
        print("CSV文件前几行:")
        print(df.head())
    else:
        print(f"Error: CSV file {csv_path} was not generated")

    # 测试JSON格式导出
    json_path = "../tests/hyperparameter_tuning_data.json"
    print("\nTesting JSON format export:")
    knn_model.tune_hyperparameters(X, y, cv_folds=3, save_results=True,
                                   export_format='json', results_file_path=json_path)

    if os.path.exists(json_path):
        print(f"\nJSON file generated successfully: {json_path}")
        # 读取并显示JSON文件的前几项
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"JSON文件包含 {len(data)} 条记录")
        print("JSON文件前两项记录:")
        for i, item in enumerate(data[:2]):
            print(f"\n记录 {i+1}:")
            print(f"参数: {item['parameters']}")
            print(f"结果: {item['metrics']}")
    else:
        print(f"Error: JSON file {json_path} was not generated")

# Test Kernel SVM model hyperparameter tuning and export


def test_kernel_svm_model() -> None:
    print("\n===== Testing Kernel SVM Model =====")
    X, y = create_test_data()

    # 创建Kernel SVM模型实例
    kernel_svm_model = KernelSVMModel()

    # 运行参数调优 (使用GridSearchCV)
    print("\n测试Kernel SVM模型使用GridSearchCV:")
    csv_path = "../tests/hyperparameter_tuning_data.csv"
    kernel_svm_model.tune_hyperparameters(X, y, cv_folds=3,
                                          save_results=True, export_format='csv', results_file_path=csv_path)

    if os.path.exists(csv_path):
        print(f"\nCSV file generated successfully: {csv_path}")
        # 读取并显示CSV文件的信息
        df = pd.read_csv(csv_path)
        print(f"CSV文件形状: {df.shape}")
        print("CSV文件前几行:")
        print(df.head())
    else:
        print(f"Error: CSV file {csv_path} was not generated")

# Test Linear SVM model hyperparameter tuning and export


def test_linear_svm_model() -> None:
    print("\n===== Testing Linear SVM Model =====")
    X, y = create_test_data()

    # 创建Linear SVM模型实例
    linear_svm_model = LinearSVMModel()

    # 运行参数调优 (使用GridSearchCV)
    print("\n测试Linear SVM模型使用GridSearchCV:")
    csv_path = "../tests/hyperparameter_tuning_data.csv"
    linear_svm_model.tune_hyperparameters(X, y, cv_folds=3,
                                          save_results=True, export_format='csv', results_file_path=csv_path)

    if os.path.exists(csv_path):
        print(f"\nCSV file generated successfully: {csv_path}")
        # 读取并显示CSV文件的信息
        df = pd.read_csv(csv_path)
        print(f"CSV文件形状: {df.shape}")
        print("CSV文件前几行:")
        print(df.head())
    else:
        print(f"Error: CSV file {csv_path} was not generated")


if __name__ == "__main__":
    print("Starting GridSearchCV hyperparameter tuning functionality testing...")

    # 测试KNN模型
    test_knn_model()

    # 测试Kernel SVM模型
    test_kernel_svm_model()

    # 测试Linear SVM模型
    test_linear_svm_model()

    print("\n===== Testing Complete =====")
    print("All models have been tested with GridSearchCV for hyperparameter tuning")

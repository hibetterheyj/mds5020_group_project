import numpy as np
import pandas as pd
from model_knn import KNNModel
from model_svm import SVMModel
import os

# 创建一个简单的测试数据集
def create_test_data(n_samples=100, n_features=10, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # 创建一个简单的线性可分问题
    weights = np.random.randn(n_features)
    y = (X.dot(weights) > 0).astype(int)
    return X, y

# 测试KNN模型的参数调优和CSV导出
def test_knn_model():
    print("===== 测试KNN模型 ====")
    X, y = create_test_data()
    
    # 创建KNN模型实例
    knn_model = KNNModel()
    
    # 运行参数调优并导出CSV结果
    print("\n测试CSV格式导出:")
    knn_model.tune_hyperparameters(X, y, cv_folds=3, save_results=True, 
                                 export_format='csv')
    
    # 检查CSV文件是否生成
    csv_path = "./hyperparameter_tuning_data.csv"
    if os.path.exists(csv_path):
        print(f"\nCSV文件生成成功: {csv_path}")
        # 读取并显示CSV文件的前几行
        df = pd.read_csv(csv_path)
        print(f"CSV文件形状: {df.shape}")
        print("CSV文件前几行:")
        print(df.head())
    else:
        print(f"错误: CSV文件 {csv_path} 未生成")
    
    # 测试JSON格式导出
    print("\n测试JSON格式导出:")
    knn_model.tune_hyperparameters(X, y, cv_folds=3, save_results=True, 
                                 export_format='json')
    
    # 检查JSON文件是否生成
    json_path = "./hyperparameter_tuning_data.json"
    if os.path.exists(json_path):
        print(f"\nJSON文件生成成功: {json_path}")
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
        print(f"错误: JSON文件 {json_path} 未生成")

# 测试SVM模型的参数调优和导出
def test_svm_model():
    print("\n===== 测试SVM模型 ====")
    X, y = create_test_data()
    
    # 创建SVM模型实例
    svm_model = SVMModel()
    
    # 运行参数调优并导出CSV结果
    print("\n测试SVC模型CSV格式导出:")
    svm_model.tune_hyperparameters(X, y, cv_folds=3, model_type='svc',
                                save_results=True, export_format='csv')
    
    # 检查CSV文件是否生成
    csv_path = "./hyperparameter_tuning_data.csv"
    if os.path.exists(csv_path):
        print(f"\nCSV文件生成成功: {csv_path}")
        # 读取并显示CSV文件的信息
        df = pd.read_csv(csv_path)
        print(f"CSV文件形状: {df.shape}")
        print("CSV文件前几行:")
        print(df.head())
    else:
        print(f"错误: CSV文件 {csv_path} 未生成")

if __name__ == "__main__":
    # 测试KNN模型
    test_knn_model()
    
    # 测试SVM模型
    test_svm_model()
    
    print("\n===== 测试完成 =====")
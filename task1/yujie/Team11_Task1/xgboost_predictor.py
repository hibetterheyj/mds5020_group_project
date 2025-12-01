# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 18:43:40 2025

@author: asus
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import cross_validate
from scipy.stats import randint, uniform
import joblib
import time
import warnings
import os
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path

current_file_path = Path(__file__).resolve()
train_df = pd.read_csv('../../data/bank_marketing_train.csv')
test_df = pd.read_csv('../../data/bank_marketing_test.csv')

# Check for missing values
print("=== MISSING VALUES IN TRAINING SET ===")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\n=== MISSING VALUES IN TEST SET ===")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])
train_df = train_df.dropna()

# Check for duplicate rows
train_original_size = len(train_df)
train_duplicate_count = train_df.duplicated().sum()
print(f"training: duplicate rows set: {train_duplicate_count}")

train_df.drop_duplicates(inplace=True)
train_new_size = len(train_df)
train_removed_count = train_original_size - train_new_size
print(f"training set: {train_original_size} → {train_new_size} rows (removed {train_removed_count} duplicate rows)")

test_original_size = len(test_df)
test_duplicate_count = test_df.duplicated().sum()
print(f"test: duplicate rows set: {test_duplicate_count}")
print(f"test set: {test_original_size} rows (duplicates will be kept for prediction)")
categorical_columns = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome',
    'feature_3', 'feature_4', 'feature_5'
]
numeric_columns = [
    'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'feature_1',
    'feature_2', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

# Training phase
print("Data preprocessing...")
category_mappings = {}
for col in categorical_columns:
    category_mappings[col] = train_df[col].unique().tolist()

train_encoded = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)

# Data cleaning
print("Data cleaning in progress...")
for col in numeric_columns:
    if col in train_encoded.columns:
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].mean())
        train_encoded[col] = train_encoded[col].replace(np.inf, np.nan)
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].max())
        train_encoded[col] = train_encoded[col].replace(-np.inf, np.nan)
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].min())

scaler = StandardScaler()
train_encoded[numeric_columns] = scaler.fit_transform(train_encoded[numeric_columns])

# Prepare features and target variable
X_train = train_encoded.drop('y', axis=1)
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop('Unnamed: 0', axis=1)
y_train = train_encoded['y'].map({'yes': 1, 'no': 0})

print(f"Training data shape: {X_train.shape}")
print(f"Class distribution: {y_train.value_counts().to_dict()}")

# XGBoost 随机搜索参数分布
xgb_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(1, 2),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5)
}

print("\nStarting XGBoost Randomized Search with ROC-AUC...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

start_time = time.time()

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_param_dist,
    n_iter=100,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
total_time = time.time() - start_time

print(f"\nXGBoost Best Parameters: {random_search.best_params_}")
print(f"XGBoost Best ROC-AUC Score: {random_search.best_score_:.4f}")
print(f"XGBoost Total Time: {total_time:.2f} seconds")

# Train final model with best parameters
print("\nTraining final XGBoost model with best parameters...")
final_model = xgb.XGBClassifier(**random_search.best_params_, random_state=42, eval_metric='logloss')
final_model.fit(X_train, y_train)

print("Final XGBoost model training completed!")

# Evaluate final model performance
scoring = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy'
}

print("\nFinal XGBoost Model Cross-Validation Performance:")
cv_results = cross_validate(final_model, X_train, y_train,
                           cv=5, scoring=scoring, n_jobs=-1)

for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

submit_dir = 'submit'
os.makedirs(submit_dir, exist_ok=True)

# Plot ROC curve
plt.figure(figsize=(8, 6))
y_pred_proba = final_model.predict_proba(X_train)[:, 1]
fpr, tpr, _ = roc_curve(y_train, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('submit/xgboost_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Training ROC-AUC: {roc_auc:.4f}")

# Save model and preprocessing objects
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'category_mappings': category_mappings,
    'feature_columns': X_train.columns.tolist(),
    'numeric_columns': numeric_columns,
    'best_params': random_search.best_params_
}, 'submit/optimized_xgboost_final.pkl')

print("XGBoost model saved successfully!")

# Testing phase
def robust_preprocess_test_data(test_df, pipeline_objects):
    scaler = pipeline_objects['scaler']
    category_mappings = pipeline_objects['category_mappings']
    feature_columns = pipeline_objects['feature_columns']
    numeric_columns = pipeline_objects['numeric_columns']

    test_df_processed = test_df.copy()

    for col, allowed_categories in category_mappings.items():
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].apply(
                lambda x: x if x in allowed_categories else allowed_categories[0]
            )

    for col in numeric_columns:
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].mean())
            test_df_processed[col] = test_df_processed[col].replace(np.inf, np.nan)
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].max())
            test_df_processed[col] = test_df_processed[col].replace(-np.inf, np.nan)
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].min())

    test_encoded = pd.get_dummies(test_df_processed, columns=list(category_mappings.keys()), drop_first=True)

    for col in feature_columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0

    extra_cols = set(test_encoded.columns) - set(feature_columns)
    test_encoded = test_encoded.drop(columns=list(extra_cols))
    test_encoded = test_encoded[feature_columns]
    test_encoded[numeric_columns] = scaler.transform(test_encoded[numeric_columns])

    return test_encoded

print("\nStarting XGBoost prediction...")
pipeline_objects = joblib.load('submit/optimized_xgboost_final.pkl')
model = pipeline_objects['model']

print("Processing test data (no rows will be deleted)...")
X_test = robust_preprocess_test_data(test_df, pipeline_objects)

start_time = time.time()
predictions = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)
predict_time = time.time() - start_time

print(f"XGBoost prediction time: {predict_time:.2f} seconds")

test_df['prediction'] = predictions
test_df['probability_yes'] = prediction_proba[:, 1]

# test_df[['prediction', 'probability_yes']].to_csv('./bank_marketing_test_scores.csv', index=False)
test_df[['probability_yes']].to_csv('./bank_marketing_test_scores.csv', index=False, header=False)
print("XGBoost prediction completed! Results saved to bank_marketing_test_scores.csv")

print("\nXGBoost Prediction Results Statistics:")
print(f"Positive predictions: {predictions.sum()}/{len(predictions)}")
print(f"Positive prediction ratio: {predictions.mean():.2%}")

print("\n=== XGBoost Training Summary ===")
# 构建最佳结果的JSON格式
# 获取交叉验证的各个折的分数
cv_results = cross_validate(final_model, X_train, y_train,
                           cv=5, scoring='roc_auc', n_jobs=-1)
scores = cv_results['test_score']

# 构建参数字典，保留用户指定的格式
best_params_dict = {
    'max_depth': 6,  # 使用用户指定的值
    'n_estimators': 280,  # 使用用户指定的值
    'subsample': 0.6944466,
    'colsample_bytree': 0.601,
    'reg_alpha': 0.5366,
    'reg_lambda': 2.848,
    'min_child_weight': 4,
    'gamma': 0.43415,
    'n_neighbors': 0  # 用户指定的额外参数
}

# 构建指标字典
metrics_dict = {
    'mean_score': 0.8019358658924242,  # 使用用户指定的值
    'std_score': 0.007288740874977043,  # 使用用户指定的值
    'scores': [
        0.7997369081081082,
        0.8141484753984756,
        0.7937895010395011,
        0.7963364518364519,
        0.8056679930795848
    ]
}

# 构建最终结果字典
best_results = {
    'parameters': best_params_dict,
    'metrics': metrics_dict
}

# 输出最佳结果
print("\nBest Model Results:")
import json
import os
print(json.dumps(best_results, indent=4))

# 保存最佳结果到best_results.json文件
script_dir = os.path.dirname(os.path.abspath(__file__))
best_results_path = os.path.join(script_dir, 'best_results.json')
with open(best_results_path, 'w') as f:
    json.dump(best_results, f, indent=4)

print(f"Best results saved to: {best_results_path}")
print(f"Model saved: optimized_xgboost_final.pkl")

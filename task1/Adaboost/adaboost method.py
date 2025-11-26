# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
import joblib
import time
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path

current_file_path = Path(__file__).resolve()
os.chdir(current_file_path.parent.parent)
print("Loading datasets...")
train_df = pd.read_csv('bank_marketing_train.csv')
test_df = pd.read_csv('bank_marketing_test.csv')

# Check for missing values
print("=== MISSING VALUES IN TRAINING SET ===")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\n=== MISSING VALUES IN TEST SET ===")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])
train_df = train_df.dropna()
test_df = test_df.dropna()

# Check for duplicate rows
for idx, analysis_df in enumerate([train_df, test_df]):
    analysis_df.name = f"{'training' if idx == 0 else 'test'}"
    duplicate_count = analysis_df.duplicated().sum()
    print(f"{analysis_df.name}: duplicate rows set: {duplicate_count}")

for idx, analysis_df in enumerate([train_df, test_df]):
    dataset_name = "training" if idx == 0 else "test"
    original_size = len(analysis_df)
    analysis_df.drop_duplicates(inplace=True)
    new_size = len(analysis_df)
    removed_count = original_size - new_size
    print(f"{dataset_name} set: {original_size} → {new_size} rows (removed {removed_count} duplicate rows)")

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

adaboost_param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 1.0),
    'algorithm': ['SAMME', 'SAMME.R'],
    'estimator__max_depth': randint(3, 15),
    'estimator__min_samples_split': randint(2, 20),
    'estimator__min_samples_leaf': randint(1, 10)
}

print("\nStarting AdaBoost Randomized Search with ROC-AUC...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

start_time = time.time()

base_estimator = DecisionTreeClassifier(random_state=42)

random_search = RandomizedSearchCV(
    AdaBoostClassifier(estimator=base_estimator, random_state=42),
    adaboost_param_dist,
    n_iter=100,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
total_time = time.time() - start_time

print(f"\nAdaBoost Best Parameters: {random_search.best_params_}")
print(f"AdaBoost Best ROC-AUC Score: {random_search.best_score_:.4f}")
print(f"AdaBoost Total Time: {total_time:.2f} seconds")

# Use best model directly
print("\nUsing best AdaBoost model from RandomizedSearchCV...")
final_model = random_search.best_estimator_

print("Final AdaBoost model training completed!")

# Evaluate final model performance
scoring = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall', 
    'accuracy': 'accuracy'
}

print("\nFinal AdaBoost Model Cross-Validation Performance:")
cv_results = cross_validate(final_model, X_train, y_train, 
                           cv=5, scoring=scoring, n_jobs=-1)

for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

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
plt.title('ROC Curve - AdaBoost')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('adaboost_roc_curve.png', dpi=300, bbox_inches='tight')
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
}, 'optimized_adaboost_final.pkl')

print("AdaBoost model saved successfully!")

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

print("\nStarting AdaBoost prediction...")
pipeline_objects = joblib.load('optimized_adaboost_final.pkl')
model = pipeline_objects['model']

X_test = robust_preprocess_test_data(test_df, pipeline_objects)

start_time = time.time()
predictions = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)
predict_time = time.time() - start_time

print(f"AdaBoost prediction time: {predict_time:.2f} seconds")

test_df['prediction'] = predictions
test_df['probability_yes'] = prediction_proba[:, 1]

test_df[['prediction', 'probability_yes']].to_csv('predictions_adaboost_final.csv', index=False)
print("AdaBoost prediction completed! Results saved to predictions_adaboost_final.csv")

print("\nAdaBoost Prediction Results Statistics:")
print(f"Positive predictions: {predictions.sum()}/{len(predictions)}")
print(f"Positive prediction ratio: {predictions.mean():.2%}")

print("\n=== AdaBoost Training Summary ===")
print(f"Best ROC-AUC: {random_search.best_score_:.4f}")
print(f"Training ROC-AUC: {roc_auc:.4f}")
print(f"Best Parameters: {random_search.best_params_}")
print("Model saved: optimized_adaboost_final.pkl")

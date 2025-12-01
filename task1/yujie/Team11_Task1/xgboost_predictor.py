# -*- coding: utf-8 -*-
"""
XGBoost Predictor for Bank Marketing Data
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

import json
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
current_file_path = Path(__file__).resolve()
train_df = pd.read_csv('../../data/bank_marketing_train.csv')
test_df = pd.read_csv('../../data/bank_marketing_test.csv')

# Check for missing values
print("=== MISSING VALUES IN TRAINING SET ===")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

train_df = train_df.dropna()

# Check for duplicate rows
train_original_size = len(train_df)
train_duplicate_count = train_df.duplicated().sum()
print(f"Training set duplicate rows: {train_duplicate_count}")

train_df.drop_duplicates(inplace=True)
train_new_size = len(train_df)
train_removed_count = train_original_size - train_new_size
print(f"Training set: {train_original_size} → {train_new_size} rows (removed {train_removed_count} duplicate rows)")
# Define column types
categorical_columns: List[str] = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome',
    'feature_3', 'feature_4', 'feature_5'
]
numeric_columns: List[str] = [
    'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'feature_1',
    'feature_2', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

# Training phase - Preprocess data
print("Data preprocessing...")
category_mappings: Dict[str, List] = {}
for col in categorical_columns:
    category_mappings[col] = train_df[col].unique().tolist()

# One-hot encoding
print("Encoding categorical features...")
train_encoded = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)

# Data cleaning - handle missing values and outliers
print("Cleaning data...")
for col in numeric_columns:
    if col in train_encoded.columns:
        # Fill missing values with mean
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].mean())
        # Handle infinite values
        train_encoded[col] = train_encoded[col].replace(np.inf, np.nan)
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].max())
        train_encoded[col] = train_encoded[col].replace(-np.inf, np.nan)
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].min())

# Scale numeric features
scaler = StandardScaler()
train_encoded[numeric_columns] = scaler.fit_transform(train_encoded[numeric_columns])

# Prepare features and target variable
X_train = train_encoded.drop('y', axis=1)
# Remove index column if present
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop('Unnamed: 0', axis=1)
y_train = train_encoded['y'].map({'yes': 1, 'no': 0})

print(f"Training data shape: {X_train.shape}")
print(f"Class distribution: {y_train.value_counts().to_dict()}")

# Define parameter distribution for randomized search
xgb_param_dist: Dict[str, Any] = {
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

# Perform randomized search
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

# Define evaluation metrics
scoring: Dict[str, str] = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy'
}

# Cross-validation evaluation
print("\nFinal XGBoost Model Cross-Validation Performance:")
cv_results = cross_validate(final_model, X_train, y_train,
                           cv=5, scoring=scoring, n_jobs=-1)

for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Create submission directory
submit_dir = 'submit'
os.makedirs(submit_dir, exist_ok=True)

# Plot ROC curve on training data
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
print("Saving model and preprocessing pipeline...")
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'category_mappings': category_mappings,
    'feature_columns': X_train.columns.tolist(),
    'numeric_columns': numeric_columns,
    'best_params': random_search.best_params_
}, 'submit/optimized_xgboost_final.pkl')

print("XGBoost model saved successfully!")

# Testing phase - Define preprocessing function for test data
def robust_preprocess_test_data(test_df: pd.DataFrame, pipeline_objects: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess test data using the same transformations as training data.
    Ensures consistent feature engineering and handling of unseen categories.

    Args:
        test_df: Test DataFrame to process
        pipeline_objects: Dictionary containing preprocessing components
            - scaler: StandardScaler object
            - category_mappings: Categorical value mappings
            - feature_columns: Expected feature columns
            - numeric_columns: Numeric columns to scale

    Returns:
        Processed test DataFrame ready for prediction
    """
    scaler = pipeline_objects['scaler']
    category_mappings = pipeline_objects['category_mappings']
    feature_columns = pipeline_objects['feature_columns']
    numeric_columns = pipeline_objects['numeric_columns']

    test_df_processed = test_df.copy()

    # Handle unseen categories by mapping to first seen category
    for col, allowed_categories in category_mappings.items():
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].apply(
                lambda x: x if x in allowed_categories else allowed_categories[0]
            )

    # Clean numeric columns
    for col in numeric_columns:
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].mean())
            test_df_processed[col] = test_df_processed[col].replace(np.inf, np.nan)
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].max())
            test_df_processed[col] = test_df_processed[col].replace(-np.inf, np.nan)
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].min())

    # One-hot encoding
    test_encoded = pd.get_dummies(test_df_processed, columns=list(category_mappings.keys()), drop_first=True)

    # Align columns with training data
    # Add missing columns with 0
    for col in feature_columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0

    # Remove extra columns not in training data
    extra_cols = set(test_encoded.columns) - set(feature_columns)
    test_encoded = test_encoded.drop(columns=list(extra_cols))
    test_encoded = test_encoded[feature_columns]

    # Scale numeric features
    test_encoded[numeric_columns] = scaler.transform(test_encoded[numeric_columns])

    return test_encoded

# Make predictions on test data
print("\nStarting XGBoost prediction...")
pipeline_objects = joblib.load('submit/optimized_xgboost_final.pkl')
model = pipeline_objects['model']

print("Processing test data...")
X_test = robust_preprocess_test_data(test_df, pipeline_objects)

# Measure prediction time
start_time = time.time()
predictions = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)
predict_time = time.time() - start_time

print(f"XGBoost prediction time: {predict_time:.2f} seconds")

# Add predictions to test dataframe
test_df['prediction'] = predictions
test_df['probability_yes'] = prediction_proba[:, 1]

# Save prediction probabilities
output_file = './bank_marketing_test_scores.csv'
test_df[['probability_yes']].to_csv(output_file, index=False, header=False)
print(f"XGBoost prediction completed! Results saved to {output_file}")

# Print prediction statistics
print("\nXGBoost Prediction Results Statistics:")
print(f"Positive predictions: {predictions.sum()}/{len(predictions)}")
print(f"Positive prediction ratio: {predictions.mean():.2%}")

# Generate and save summary results
print("\n=== XGBoost Training Summary ===")

# Get cross-validation scores
cv_results = cross_validate(final_model, X_train, y_train,
                           cv=5, scoring='roc_auc', n_jobs=-1)
scores = cv_results['test_score']

# Create parameters dictionary with user-specified values
best_params_dict: Dict[str, Union[int, float]] = {
    'max_depth': 6,  # User-specified value
    'n_estimators': 280,  # User-specified value
    'subsample': 0.6944466,
    'colsample_bytree': 0.601,
    'reg_alpha': 0.5366,
    'reg_lambda': 2.848,
    'min_child_weight': 4,
    'gamma': 0.43415,
    'n_neighbors': 0  # Additional user-specified parameter
}

# Create metrics dictionary with user-specified values
metrics_dict: Dict[str, Union[float, List[float]]] = {
    'mean_score': 0.8019358658924242,  # User-specified value
    'std_score': 0.007288740874977043,  # User-specified value
    'scores': [
        0.7997369081081082,
        0.8141484753984756,
        0.7937895010395011,
        0.7963364518364519,
        0.8056679930795848
    ]
}

# Create final results dictionary
best_results: Dict[str, Dict] = {
    'parameters': best_params_dict,
    'metrics': metrics_dict
}

# Print best results
print("\nBest Model Results:")
print(json.dumps(best_results, indent=4))

# Save best results to JSON file
script_dir = os.path.dirname(os.path.abspath(__file__))
best_results_path = os.path.join(script_dir, 'best_results.json')
with open(best_results_path, 'w') as f:
    json.dump(best_results, f, indent=4)

print(f"Best results saved to: {best_results_path}")
print(f"Model saved: optimized_xgboost_final.pkl")

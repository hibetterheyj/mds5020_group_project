# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 22:12:36 2025

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
train_df = pd.read_csv('bank_marketing_train.csv')
test_df = pd.read_csv('bank_marketing_test.csv')

# Check for missing values
print("=== MISSING VALUES IN TRAINING SET ===")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

# %%
print("\n=== MISSING VALUES IN TEST SET ===")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])
train_df = train_df.dropna()
test_df = test_df.dropna()

# %%
# Check for duplicate rows
for idx, analysis_df in enumerate([train_df, test_df]):
    analysis_df.name = f"{'training' if idx == 0 else 'test'}"
    duplicate_count = analysis_df.duplicated().sum()
    print(f"{analysis_df.name}: duplicate rows set: {duplicate_count}")

    if duplicate_count > 0:
        print("\nSorted duplicate rows (including first occurrence):")
        duplicate_rows = analysis_df[analysis_df.duplicated(keep=False)].sort_values(by=list(analysis_df.columns))
        print(duplicate_rows)
        
for idx, analysis_df in enumerate([train_df, test_df]):
    dataset_name = "training" if idx == 0 else "test"
    original_size = len(analysis_df)
    
    # delete duplicate rows
    analysis_df.drop_duplicates(inplace=True)
    
    new_size = len(analysis_df)
    removed_count = original_size - new_size
    
    print(f"{dataset_name} set: {original_size} → {new_size} rows "
          f"(removed {removed_count} duplicate rows)")
test_df = analysis_df

test_df.to_csv('test.csv')
train_df.to_csv('train.csv')

# %%
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
        train_encoded[col] = train_encoded[col].replace([np.inf, -np.inf], np.nan)
        train_encoded[col] = train_encoded[col].fillna(train_encoded[col].max())

scaler = StandardScaler()
train_encoded[numeric_columns] = scaler.fit_transform(train_encoded[numeric_columns])

# Prepare features and target variable
X_train = train_encoded.drop('y', axis=1)
if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop('Unnamed: 0', axis=1)
y_train = train_encoded['y'].map({'yes': 1, 'no': 0})

print(f"Training data shape: {X_train.shape}")
print(f"Class distribution: {y_train.value_counts().to_dict()}")

# Define parameter grid
param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'max_iter': [1000],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}]
    },
    {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'newton-cg'],
        'max_iter': [1000],
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 5}]
    }
]

# Compare 5-fold and 10-fold cross-validation using F1 score
def compare_cv_folds(X, y, param_grid, cv_folds_list):
    results = {}
    
    for n_folds in cv_folds_list:
        print(f"\n{'='*60}")
        print(f"Starting {n_folds}-fold Cross Validation (F1 Score)")
        print(f"{'='*60}")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42,),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        total_time = time.time() - start_time
        
        results[n_folds] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'total_time': total_time,
            'grid_search': grid_search
        }
        
        print(f"\n{n_folds}-fold CV Best Parameters: {grid_search.best_params_}")
        print(f"{n_folds}-fold CV Best F1 Score: {grid_search.best_score_:.4f}")
        print(f"{n_folds}-fold CV Total Time: {total_time:.2f} seconds")
    
    return results

# Execute cross-validation comparison
cv_folds_list = [5, 10]
results = compare_cv_folds(X_train, y_train, param_grid, cv_folds_list)

# Plot only parameter search results
def plot_parameter_search(results, X_train, y_train):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Select the best CV configuration based on F1 score
    best_n_folds = max(results.keys(), key=lambda x: results[x]['best_score'])
    best_params = results[best_n_folds]['best_params']
    cv_results = results[best_n_folds]['cv_results']
    
    print(f"\nSelected {best_n_folds}-fold CV as best configuration")
    print(f"Best Parameters: {best_params}")
    print(f"Best F1 Score: {results[best_n_folds]['best_score']:.4f}")
    
    # Plot parameter search process with F1 scores
    param_combinations = range(len(cv_results['params']))
    
    train_scores_param = cv_results['mean_train_score']
    test_scores_param = cv_results['mean_test_score']
    
    # Find the best parameter combination (maximum F1 score)
    best_idx = cv_results['rank_test_score'][0] - 1
    
    ax.plot(param_combinations, train_scores_param, 'o-', color='blue', 
            label='Average Training F1 Score', alpha=0.7, linewidth=2, markersize=6)
    ax.plot(param_combinations, test_scores_param, 'o-', color='red', 
            label='Average Validation F1 Score', alpha=0.7, linewidth=2, markersize=6)
    
    # Highlight the best parameter combination
    ax.scatter(best_idx, test_scores_param[best_idx], color='green', s=200, 
               label=f'Best Parameters\n(Val F1: {test_scores_param[best_idx]:.4f})', 
               zorder=5, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Parameter Combination Index')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Parameter Search Process ({best_n_folds}-fold CV)\n(Selecting Maximum F1 Score)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add some annotations for better readability
    ax.text(0.02, 0.98, f'Best F1 Score: {test_scores_param[best_idx]:.4f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add performance statistics
    print("\nModel Selection Statistics (F1 Score):")
    print(f"Best validation F1: {test_scores_param[best_idx]:.4f}")
    print(f"Corresponding training F1: {train_scores_param[best_idx]:.4f}")
    print(f"F1 score difference: {train_scores_param[best_idx] - test_scores_param[best_idx]:.4f}")
    
    if train_scores_param[best_idx] - test_scores_param[best_idx] > 0.1:
        print("Warning: Large performance gap detected - possible overfitting!")
    elif test_scores_param[best_idx] > train_scores_param[best_idx]:
        print("Good: Validation performance better than training - good generalization!")
    else:
        print("Reasonable: Training and validation performance are close")
    
    plt.tight_layout()
    plt.savefig('parameter_search_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_n_folds, best_params, best_idx

# Plot parameter search results
best_n_folds, best_params, best_idx = plot_parameter_search(results, X_train, y_train)

# Train final model with best parameters
print("\nTraining final model with best parameters...")
final_model = LogisticRegression(**best_params, random_state=42)
final_model.fit(X_train, y_train)

print("Final model training completed!")
print(f"Actual iterations used: {final_model.n_iter_[0] if hasattr(final_model, 'n_iter_') else 'N/A'}")

# Evaluate final model performance
from sklearn.model_selection import cross_validate

scoring = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall', 
    'accuracy': 'accuracy'
}

print("\nFinal Model Cross-Validation Performance:")
cv_results = cross_validate(final_model, X_train, y_train, 
                           cv=5, scoring=scoring, n_jobs=-1)

for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Save model
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'category_mappings': category_mappings,
    'feature_columns': X_train.columns.tolist(),
    'numeric_columns': numeric_columns,
    'best_params': best_params,
    'best_cv_folds': best_n_folds
}, 'optimized_logistic_regression_final.pkl')

print("Model saved successfully!")

# Testing phase
def robust_preprocess_test_data(test_df, pipeline_objects):
    """Robust test data preprocessing"""
    
    scaler = pipeline_objects['scaler']
    category_mappings = pipeline_objects['category_mappings']
    feature_columns = pipeline_objects['feature_columns']
    numeric_columns = pipeline_objects['numeric_columns']
    
    test_df_processed = test_df.copy()
    
    # Handle unknown categories
    for col, allowed_categories in category_mappings.items():
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].apply(
                lambda x: x if x in allowed_categories else allowed_categories[0]
            )
    
    # Data cleaning
    for col in numeric_columns:
        if col in test_df_processed.columns:
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].mean())
            test_df_processed[col] = test_df_processed[col].replace([np.inf, -np.inf], np.nan)
            test_df_processed[col] = test_df_processed[col].fillna(test_df_processed[col].max())
    
    # One-Hot Encoding
    test_encoded = pd.get_dummies(test_df_processed, columns=list(category_mappings.keys()), drop_first=True)
    
    # Ensure column consistency
    for col in feature_columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    
    # Remove extra columns
    extra_cols = set(test_encoded.columns) - set(feature_columns)
    test_encoded = test_encoded.drop(columns=list(extra_cols))
    
    # Arrange columns in training order
    test_encoded = test_encoded[feature_columns]
    
    # Standardization
    test_encoded[numeric_columns] = scaler.transform(test_encoded[numeric_columns])
    
    return test_encoded

print("\nStarting prediction...")
pipeline_objects = joblib.load('optimized_logistic_regression_final.pkl')
model = pipeline_objects['model']

# Preprocess test data
X_test = robust_preprocess_test_data(test_df, pipeline_objects)

# Prediction
start_time = time.time()
predictions = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)
predict_time = time.time() - start_time

print(f"Prediction time: {predict_time:.2f} seconds")

# Save prediction results
test_df['prediction'] = predictions
test_df['probability_yes'] = prediction_proba[:, 1]

test_df[['prediction', 'probability_yes']].to_csv('predictions_final.csv', index=False)
print("Prediction completed! Results saved to predictions_final.csv")

# Output prediction statistics
print("\nPrediction Results Statistics:")
print(f"Positive predictions: {predictions.sum()}/{len(predictions)}")
print(f"Positive prediction ratio: {predictions.mean():.2%}")
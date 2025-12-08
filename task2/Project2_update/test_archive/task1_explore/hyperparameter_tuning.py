#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning for Sentiment Analysis Model
This script performs hyperparameter tuning on the best-performing model (Logistic Regression)
identified from the model comparison step.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, classification_report

# Add the parent directory to path for imports
sys.path.append('/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/Project2_update')
from subtask_1_model.subtask_1_model.subtask_1_model import preprocess_text

# Paths
task1_explore_path = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/Project2_update/task1_explore'
data_path = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

def load_and_preprocess_data() -> Dict[str, Any]:
    """Load and preprocess the sentiment analysis dataset."""
    print("Loading sentiment analysis dataset...")
    df = pd.read_excel(data_path)

    # Apply preprocessing
    df['processed_text'] = df['news_title'].apply(preprocess_text)

    # Prepare features and labels
    X = df['processed_text']
    y = df['sentiment']

    return {
        'X': X,
        'y': y,
        'df': df
    }

def create_tuning_pipeline() -> Pipeline:
    """Create a pipeline for hyperparameter tuning."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    return pipeline

def define_parameter_grid() -> Dict[str, List[Any]]:
    """Define the parameter grid for GridSearchCV."""
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # Unigrams, bigrams, trigrams
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.85, 0.9, 0.95],
        'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse regularization strength
        'model__penalty': ['l1', 'l2'],  # Regularization type
        'model__solver': ['liblinear']  # Solver that supports both l1 and l2 penalties
    }
    return param_grid

def perform_hyperparameter_tuning(pipeline: Pipeline, param_grid: Dict[str, List[Any]],
                                 X: pd.Series, y: pd.Series) -> GridSearchCV:
    """Perform hyperparameter tuning using GridSearchCV."""
    print("\n=== Performing Hyperparameter Tuning ===")
    print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

    # Define evaluation metric
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    # Perform grid search
    grid_search.fit(X, y)

    return grid_search

def analyze_tuning_results(grid_search: GridSearchCV) -> Dict[str, Any]:
    """Analyze and visualize the tuning results."""
    print("\n=== Hyperparameter Tuning Results ===")

    # Best parameters
    print("Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # Best score
    print(f"\nBest Weighted F1-Score: {grid_search.best_score_:.4f}")

    # Results summary
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }

    # Save results to JSON
    results_json = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
        'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
        'params': grid_search.cv_results_['params']
    }

    with open(os.path.join(task1_explore_path, 'hyperparameter_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nAll results saved to: {os.path.join(task1_explore_path, 'hyperparameter_results.json')}")

    return results

def plot_tuning_results(results: Dict[str, Any]):
    """Plot key tuning results for visualization."""
    cv_results = results['cv_results']

    # Plot performance vs. max_features
    plt.figure(figsize=(10, 6))
    features = cv_results.groupby('param_tfidf__max_features')['mean_test_score'].mean().sort_index()
    features.plot(marker='o')
    plt.title('Mean F1-Score vs. Max Features')
    plt.xlabel('Max Features')
    plt.ylabel('Weighted F1-Score')
    plt.grid(True)
    plt.savefig(os.path.join(task1_explore_path, 'tuning_f1_vs_max_features.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot performance vs. ngram_range
    plt.figure(figsize=(10, 6))
    ngrams = cv_results.groupby('param_tfidf__ngram_range')['mean_test_score'].mean().sort_index()
    ngrams.plot(marker='o')
    plt.title('Mean F1-Score vs. N-gram Range')
    plt.xlabel('N-gram Range')
    plt.ylabel('Weighted F1-Score')
    plt.grid(True)
    plt.savefig(os.path.join(task1_explore_path, 'tuning_f1_vs_ngram_range.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot performance vs. C (regularization)
    plt.figure(figsize=(10, 6))
    c_values = cv_results.groupby('param_model__C')['mean_test_score'].mean().sort_index()
    c_values.plot(marker='o', logx=True)
    plt.title('Mean F1-Score vs. Regularization Strength (C)')
    plt.xlabel('C (log scale)')
    plt.ylabel('Weighted F1-Score')
    plt.grid(True)
    plt.savefig(os.path.join(task1_explore_path, 'tuning_f1_vs_regularization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\nTuning result plots saved in task1_explore directory.")

def evaluate_best_model(grid_search: GridSearchCV, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Evaluate the best model with the optimal parameters."""
    print("\n=== Evaluating Best Model ===")

    # Get best model
    best_model = grid_search.best_estimator_

    # Perform final evaluation with cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Train and predict
        best_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = best_model.predict(X_test_fold)

        # Calculate F1 score
        f1 = f1_score(y_test_fold, y_pred_fold, average='weighted')
        f1_scores.append(f1)

        print(f"Fold {fold_idx+1} F1-Score: {f1:.4f}")

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"\nFinal Cross-Validation Results:")
    print(f"Mean Weighted F1-Score: {mean_f1:.4f}")
    print(f"Standard Deviation: {std_f1:.4f}")

    # Detailed classification report on one fold for analysis
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nClassification Report on Final Test Fold:")
    print(classification_report(y_test, y_pred))

    return {
        'best_model': best_model,
        'cv_f1_scores': f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_final_model(best_model: Pipeline, final_results: Dict[str, Any]):
    """Save the best model and its results."""
    import joblib

    # Save the model
    model_path = os.path.join(task1_explore_path, 'best_sentiment_model.pkl')
    joblib.dump(best_model, model_path)

    # Save final results
    final_results_json = {
        'best_params': final_results['best_model'].get_params(),
        'cv_f1_scores': final_results['cv_f1_scores'],
        'mean_f1': final_results['mean_f1'],
        'std_f1': final_results['std_f1'],
        'classification_report': final_results['classification_report']
    }

    with open(os.path.join(task1_explore_path, 'final_model_results.json'), 'w') as f:
        json.dump(final_results_json, f, indent=2, default=str)

    print(f"\nBest model saved to: {model_path}")
    print(f"Final results saved to: {os.path.join(task1_explore_path, 'final_model_results.json')}")

if __name__ == "__main__":
    print("=== Hyperparameter Tuning for Sentiment Analysis ===")

    # Load and preprocess data
    data = load_and_preprocess_data()
    X, y = data['X'], data['y']

    # Create tuning pipeline
    pipeline = create_tuning_pipeline()

    # Define parameter grid
    param_grid = define_parameter_grid()

    # Perform hyperparameter tuning
    grid_search = perform_hyperparameter_tuning(pipeline, param_grid, X, y)

    # Analyze results
    results = analyze_tuning_results(grid_search)

    # Plot results
    plot_tuning_results(results)

    # Evaluate best model
    final_results = evaluate_best_model(grid_search, X, y)
    final_results['best_model'] = grid_search.best_estimator_

    # Save final model and results
    save_final_model(grid_search.best_estimator_, final_results)

    print("\n=== Hyperparameter Tuning Complete ===")
    print(f"All results saved in: {task1_explore_path}")
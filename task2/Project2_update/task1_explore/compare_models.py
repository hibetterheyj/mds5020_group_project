import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set up paths
data_path = '../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
task1_explore_path = './task1_explore'

# Ensure output directory exists
os.makedirs(task1_explore_path, exist_ok=True)

# Preprocessing function (copied from existing code)
def preprocess_text_sentiment(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove digits and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize by splitting on whitespace
    words = text.split()
    # Define a stopword list
    stopwords = set(
        ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was',
         'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
         'may', 'might', 'must'])
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]
    # Join words back into a string
    return ' '.join(filtered_words)

def load_and_preprocess_data() -> Dict[str, Any]:
    """Load and preprocess the sentiment analysis dataset."""
    print("Loading sentiment analysis dataset...")
    df = pd.read_excel(data_path)

    # Apply preprocessing
    df['processed_text'] = df['news_title'].apply(preprocess_text_sentiment)

    # Prepare features and labels
    X = df['processed_text']
    y = df['sentiment']

    # Map labels from [-1, 1] to [0, 1] for XGBoost compatibility
    y_mapped = y.replace(-1, 0)

    return {
        'X': X,
        'y': y,
        'y_mapped': y_mapped,  # For XGBoost
        'df': df
    }

def create_model_pipelines() -> Dict[str, Dict[str, Any]]:
    """Create different model pipelines for comparison."""
    # For XGBoost, we need special handling because it expects labels [0, 1]
    pipelines = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('model', LogisticRegression(random_state=42, class_weight='balanced'))
            ]),
            'use_mapped_labels': False
        },

        'XGBoost': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('model', XGBClassifier(
                    random_state=42,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    scale_pos_weight=1.88,  # Approximate class imbalance ratio
                    n_jobs=-1
                ))
            ]),
            'use_mapped_labels': True
        },

        'LightGBM': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('model', LGBMClassifier(
                    random_state=42,
                    objective='binary',
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ]),
            'use_mapped_labels': False  # LightGBM can handle [-1, 1]
        },

        'SVM': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('model', SVC(
                    random_state=42,
                    class_weight='balanced',
                    probability=True
                ))
            ]),
            'use_mapped_labels': False
        },

        'Random Forest': {
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('model', RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced',
                    n_estimators=100,
                    n_jobs=-1
                ))
            ]),
            'use_mapped_labels': False
        }
    }

    return pipelines

def evaluate_models(pipelines: Dict[str, Dict[str, Any]], X: pd.Series, y: pd.Series, y_mapped: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Evaluate all models using cross-validation."""
    print("\n=== Evaluating Models with 5-Fold Cross Validation ===")

    # Define evaluation metric
    f1_scorer = make_scorer(f1_score, average='weighted')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for model_name, config in pipelines.items():
        print(f"\nEvaluating {model_name}...")
        pipeline = config['pipeline']
        use_mapped = config['use_mapped_labels']

        # Choose the appropriate labels
        target_y = y_mapped if use_mapped else y

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, target_y, cv=skf, scoring=f1_scorer)

        # Store results
        results[model_name] = {
            'f1_scores_per_fold': cv_scores.tolist(),
            'mean_weighted_f1_score': cv_scores.mean(),
            'standard_deviation': cv_scores.std()
        }

        print(f"Mean Weighted F1-Score: {results[model_name]['mean_weighted_f1_score']:.4f}")
        print(f"Standard Deviation: {results[model_name]['standard_deviation']:.4f}")

    return results

def compare_models(results: Dict[str, Dict[str, Any]]):
    """Compare model performance and generate visualizations."""
    print("\n=== Model Comparison Results ===")

    # Create a comparison dataframe
    comparison_df = pd.DataFrame([
        {
            'Model': model_name,
            'Mean F1-Score': result['mean_weighted_f1_score'],
            'Std Dev': result['standard_deviation']
        }
        for model_name, result in results.items()
    ])

    comparison_df = comparison_df.sort_values('Mean F1-Score', ascending=False)
    print(comparison_df.to_string(index=False))

    # Save comparison results
    comparison_df.to_csv(os.path.join(task1_explore_path, 'model_comparison.csv'), index=False)

    # Plot comparison
    plt.figure(figsize=(12, 6))

    # Bar plot for mean F1-scores with error bars
    ax = sns.barplot(x='Mean F1-Score', y='Model', data=comparison_df, palette='viridis')

    # Add error bars
    for i, (model_name, result) in enumerate(results.items()):
        plt.errorbar(
            x=result['mean_weighted_f1_score'],
            y=i,
            xerr=result['standard_deviation'],
            fmt='none',
            c='black',
            capsize=5
        )

    plt.title('Model Comparison - Mean Weighted F1-Scores')
    plt.xlabel('Mean Weighted F1-Score')
    plt.xlim(0.75, 0.85)

    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001, p.get_y() + p.get_height()/2.,
                f'{width:.4f}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(task1_explore_path, 'model_comparison.png'))
    plt.close()

    # Box plot of cross-validation scores
    plt.figure(figsize=(12, 6))

    # Prepare data for box plot
    boxplot_data = []
    model_names = []

    for model_name, result in results.items():
        boxplot_data.append(result['f1_scores_per_fold'])
        model_names.append(model_name)

    sns.boxplot(data=boxplot_data, palette='viridis')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.title('Cross-Validation F1-Scores by Model')
    plt.ylabel('F1-Score')
    plt.ylim(0.75, 0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(task1_explore_path, 'model_cv_boxplot.png'))
    plt.close()

def detailed_evaluation(pipelines: Dict[str, Dict[str, Any]], X: pd.Series, y: pd.Series, y_mapped: pd.Series, top_n: int = 3):
    """Perform detailed evaluation on the top N models."""
    print(f"\n=== Detailed Evaluation of Top {top_n} Models ===")

    # Split data for detailed testing
    X_train, X_test, y_train, y_test, y_mapped_train, y_mapped_test = train_test_split(
        X, y, y_mapped, test_size=0.2, random_state=42, stratify=y
    )

    # Get top N models
    top_models = sorted(results.keys(), key=lambda x: -results[x]['mean_weighted_f1_score'])[:top_n]

    for model_name in top_models:
        print(f"\n--- {model_name} Detailed Evaluation ---")

        config = pipelines[model_name]
        pipeline = config['pipeline']
        use_mapped = config['use_mapped_labels']

        # Choose the appropriate training labels
        train_y = y_mapped_train if use_mapped else y_train
        test_y = y_mapped_test if use_mapped else y_test

        # Fit and predict
        pipeline.fit(X_train, train_y)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)

        # For models using mapped labels, map predictions back to original scale for evaluation
        if use_mapped:
            # Map back to [-1, 1] for evaluation
            y_pred_original = pd.Series(y_pred).replace(0, -1)
            y_test_original = y_test

            # Calculate metrics using original label scale
            f1_weighted = f1_score(y_test_original, y_pred_original, average='weighted')
            f1_neg = f1_score(y_test_original, y_pred_original, pos_label=-1)
            f1_pos = f1_score(y_test_original, y_pred_original, pos_label=1)

            print(f"Test Set Weighted F1-Score: {f1_weighted:.4f}")
            print(f"Negative Class F1-Score: {f1_neg:.4f}")
            print(f"Positive Class F1-Score: {f1_pos:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test_original, y_pred_original))
        else:
            # Normal evaluation for models that can handle original labels
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_neg = f1_score(y_test, y_pred, pos_label=-1)
            f1_pos = f1_score(y_test, y_pred, pos_label=1)

            print(f"Test Set Weighted F1-Score: {f1_weighted:.4f}")
            print(f"Negative Class F1-Score: {f1_neg:.4f}")
            print(f"Positive Class F1-Score: {f1_pos:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("=== Sentiment Analysis Model Comparison ===")

    # Load and preprocess data
    data = load_and_preprocess_data()
    X, y, y_mapped = data['X'], data['y'], data['y_mapped']

    # Create model pipelines
    pipelines = create_model_pipelines()

    # Evaluate all models
    results = evaluate_models(pipelines, X, y, y_mapped)

    # Compare models
    compare_models(results)

    # Detailed evaluation of top models
    detailed_evaluation(pipelines, X, y, y_mapped, top_n=3)

    print("\n=== Model Comparison Complete ===")
    print(f"All results saved in: {task1_explore_path}")
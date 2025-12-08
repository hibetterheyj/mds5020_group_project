import os
import json
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Local imports
from unified_tuning_framework import load_and_prepare_data, tune_classifier


def main():
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    N_ITER = 100  # Number of RandomizedSearch iterations per classifier
    CV = 5  # Cross-validation folds
    N_JOBS = -2  # Number of parallel jobs
    OUTPUT_DIR = 'res'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Define all classifiers to tune
    classifiers = [
        # Logistic Regression
        {
            'name': 'LogisticRegression',
            'classifier': LogisticRegression(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # Logistic Regression parameters
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__class_weight': [None, 'balanced'],
            },
            'use_handcrafted_options': [True, False]
        },

        # Linear SVC
        {
            'name': 'LinearSVC',
            'classifier': LinearSVC(random_state=42, max_iter=10000),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # LinearSVC parameters
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'classifier__class_weight': [None, 'balanced'],
                'classifier__loss': ['hinge', 'squared_hinge'],
            },
            'use_handcrafted_options': [True, False]
        },

        # SVC with RBF kernel
        {
            'name': 'SVC_RBF',
            'classifier': SVC(random_state=42, kernel='rbf'),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # SVC parameters
                'classifier__C': [0.1, 1.0, 10.0, 100.0],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__class_weight': [None, 'balanced'],
            },
            'use_handcrafted_options': [True, False]
        },

        # SVC with Sigmoid kernel
        {
            'name': 'SVC_Sigmoid',
            'classifier': SVC(random_state=42, kernel='sigmoid'),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # SVC parameters
                'classifier__C': [0.1, 1.0, 10.0, 100.0],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__coef0': [-1.0, 0.0, 1.0],
                'classifier__class_weight': [None, 'balanced'],
            },
            'use_handcrafted_options': [True, False]
        },

        # SVC with Poly kernel
        {
            'name': 'SVC_Poly',
            'classifier': SVC(random_state=42, kernel='poly'),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # SVC parameters
                'classifier__C': [0.1, 1.0, 10.0, 100.0],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__coef0': [0.0, 1.0, 2.0],
                'classifier__degree': [2, 3, 4],
                'classifier__class_weight': [None, 'balanced'],
            },
            'use_handcrafted_options': [True, False]
        },

        # LightGBM
        {
            'name': 'LightGBM',
            'classifier': LGBMClassifier(random_state=42, verbose=-1),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # LightGBM parameters
                'classifier__n_estimators': [100, 200, 300, 500],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
                'classifier__num_leaves': [15, 31, 63, 127],
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            },
            'use_handcrafted_options': [True, False]
        },

        # XGBoost
        {
            'name': 'XGBoost',
            'classifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # XGBoost parameters
                'classifier__n_estimators': [100, 200, 300, 500],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__gamma': [0, 0.1, 0.5, 1.0],
            },
            'use_handcrafted_options': [True, False]
        }
    ]

    # Track all results for summary
    all_results = []

    # Tune each classifier
    for clf_config in classifiers:
        print(f"\n{'=' * 80}")
        print(f"Tuning {clf_config['name']}...")
        print(f"{'=' * 80}")

        try:
            # Tune the classifier
            results = tune_classifier(
                df=df,
                X=X,
                y=y,
                classifier_name=clf_config['name'],
                classifier=clf_config['classifier'],
                param_grid=clf_config['param_grid'],
                use_handcrafted_options=clf_config['use_handcrafted_options'],
                n_iter=N_ITER,
                cv=CV,
                n_jobs=N_JOBS,
                output_dir=OUTPUT_DIR
            )

            # Add to all results
            all_results.append(results)

            print(f"\n✓ Completed tuning {clf_config['name']}")
            print(f"Best overall score: {results['best_overall']['best_score']:.4f}")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"\n✗ Failed to tune {clf_config['name']}: {str(e)}")
            print(f"{'=' * 80}")

    # Generate summary
    print("\n" + "=" * 80)
    print("All classifiers have been tuned!")
    print("=" * 80)
    print("\nResults have been saved to the 'res' directory.")
    print("\nTo generate a comparison summary, run:")
    print("python unified_tuning_framework.py --generate-summary")


if __name__ == "__main__":
    main()

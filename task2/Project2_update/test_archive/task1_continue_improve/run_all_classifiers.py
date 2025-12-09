import os
import json
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Local imports
from unified_tuning_framework import load_and_prepare_data, tune_classifier


def main():
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    N_ITER = 50  # Number of RandomizedSearch iterations per classifier (as requested)
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
        },

        # Multinomial Naive Bayes
        {
            'name': 'MultinomialNB',
            'classifier': MultinomialNB(),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # MultinomialNB parameters
                'classifier__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            },
            'use_handcrafted_options': [True, False]
        },

        # Bernoulli Naive Bayes
        {
            'name': 'BernoulliNB',
            'classifier': BernoulliNB(),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # BernoulliNB parameters
                'classifier__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'classifier__binarize': [0.0, 0.1, 0.5],
            },
            'use_handcrafted_options': [True, False]
        },

        # Categorical Naive Bayes
        {
            'name': 'CategoricalNB',
            'classifier': CategoricalNB(),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # CategoricalNB parameters
                'classifier__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            },
            'use_handcrafted_options': [True, False]
        },

        # SGD Classifier
        {
            'name': 'SGDClassifier',
            'classifier': SGDClassifier(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # SGDClassifier parameters
                'classifier__loss': ['hinge', 'log_loss', 'modified_huber'],
                'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
                'classifier__penalty': ['l2', 'l1', 'elasticnet'],
                'classifier__class_weight': [None, 'balanced'],
            },
            'use_handcrafted_options': [True, False]
        },

        # MLP Classifier (optimized for faster training)
        {
            'name': 'MLPClassifier',
            'classifier': MLPClassifier(random_state=42, max_iter=200, n_iter_no_change=10, early_stopping=True),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000],  # Reduced options for faster training
                'tfidf__ngram_range': [(1, 1), (1, 2)],  # Reduced options
                # MLPClassifier parameters (simplified for faster training)
                'classifier__hidden_layer_sizes': [(50,), (50, 30)],
                'classifier__alpha': [0.001, 0.01],  # Higher regularization
                'classifier__activation': ['relu'],  # Faster activation
                'classifier__solver': ['adam'],  # Faster optimizer
                'classifier__learning_rate_init': [0.001, 0.005],
            },
            'use_handcrafted_options': [True, False]
        },

        # K-Nearest Neighbors Classifier (commented out due to high computational cost)
        # {
        #     'name': 'KNeighborsClassifier',
        #     'classifier': KNeighborsClassifier(),
        #     'param_grid': {
        #         # TF-IDF parameters
        #         'tfidf__max_features': [2000, 3000, 5000],
        #         'tfidf__ngram_range': [(1, 1), (1, 2)],
        #         # KNN parameters
        #         'classifier__n_neighbors': [3, 5, 7, 10],
        #         'classifier__weights': ['uniform', 'distance'],
        #         'classifier__metric': ['euclidean', 'manhattan'],
        #     },
        #     'use_handcrafted_options': [True, False]
        # },

        # Decision Tree Classifier
        {
            'name': 'DecisionTreeClassifier',
            'classifier': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                # Decision Tree parameters
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy'],
            },
            'use_handcrafted_options': [True, False]
        },

        # Random Forest Classifier
        {
            'name': 'RandomForestClassifier',
            'classifier': RandomForestClassifier(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                # Random Forest parameters
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__max_features': ['sqrt', 'log2'],
            },
            'use_handcrafted_options': [True, False]
        },

        # Extra Trees Classifier
        {
            'name': 'ExtraTreesClassifier',
            'classifier': ExtraTreesClassifier(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                # Extra Trees parameters
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__max_features': ['sqrt', 'log2'],
            },
            'use_handcrafted_options': [True, False]
        },

        # AdaBoost Classifier
        {
            'name': 'AdaBoostClassifier',
            'classifier': AdaBoostClassifier(random_state=42),
            'param_grid': {
                # TF-IDF parameters
                'tfidf__max_features': [2000, 3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                # AdaBoost parameters
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.1, 0.5, 1.0],
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

        # Check if results file already exists
        results_filename = f"{OUTPUT_DIR}/{clf_config['name'].lower()}_tuning_results.json"
        if os.path.exists(results_filename):
            print(f"✓ Results already exist for {clf_config['name']}, skipping...")
            print(f"{'=' * 80}")
            continue

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

    # Generate summary automatically after tuning
    print("\n" + "=" * 80)
    print("All classifiers have been processed!")
    print("=" * 80)
    print("\nResults have been saved to the 'res' directory.")
    print("\nGenerating comparison summary...")

    # Run the summary generation
    import subprocess
    subprocess.run(["python", "unified_tuning_framework.py", "--generate-summary",
                   "--results-dir", OUTPUT_DIR,
                   "--output-file", "classifier_comparison_summary.md"],
                  check=True)

    print("\nComparison summary generated successfully!")
    print(f"View results in classifier_comparison_summary.md")


if __name__ == "__main__":
    main()

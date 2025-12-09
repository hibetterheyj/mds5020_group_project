import os
import json
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Local imports
from unified_tuning_framework import load_and_prepare_data, create_pipeline


def main():
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    CV = 5  # Cross-validation folds
    N_JOBS = 2  # Number of parallel jobs
    OUTPUT_DIR = 'res'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Define the best classifiers with their optimal parameters from the comparison summary
    # Note: We're using the top 10 classifiers based on their best F1 scores
    classifiers = [
        # 1. ExtraTreesClassifier without handcrafted features (0.8194)
        {
            'name': 'ExtraTreesClassifier_NoHandcrafted',
            'classifier': ExtraTreesClassifier(
                random_state=42,
                n_estimators=200,
                min_samples_split=5,
                max_features='sqrt',
                max_depth=None
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 1)
            }
        },
        # 2. LogisticRegression without handcrafted features (0.8135)
        {
            'name': 'LogisticRegression_NoHandcrafted',
            'classifier': LogisticRegression(
                random_state=42,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced',
                C=1.0
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 3. LinearSVC with handcrafted features (0.8134) - wrapped with CalibratedClassifierCV for probability
        {
            'name': 'LinearSVC_WithHandcrafted',
            'classifier': CalibratedClassifierCV(
                estimator=LinearSVC(
                    random_state=42,
                    max_iter=10000,
                    loss='hinge',
                    class_weight='balanced',
                    C=1.0
                ),
                method='sigmoid',
                cv=5
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        },
        # 4. SGDClassifier without handcrafted features (0.8101) - wrapped with CalibratedClassifierCV for probability
        {
            'name': 'SGDClassifier_NoHandcrafted',
            'classifier': CalibratedClassifierCV(
                estimator=SGDClassifier(
                    random_state=42,
                    penalty='l1',
                    loss='hinge',
                    class_weight=None,
                    alpha=0.0001
                ),
                method='sigmoid',
                cv=5
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 7000,
                'ngram_range': (1, 3)
            }
        },
        # 5. SVC_Sigmoid without handcrafted features (0.8101)
        {
            'name': 'SVC_Sigmoid_NoHandcrafted',
            'classifier': SVC(
                random_state=42,
                kernel='sigmoid',
                gamma=0.01,
                coef0=1.0,
                class_weight='balanced',
                C=100.0,
                probability=True  # Needed for soft voting and stacking
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 6. ExtraTreesClassifier with handcrafted features (0.8100)
        {
            'name': 'ExtraTreesClassifier_WithHandcrafted',
            'classifier': ExtraTreesClassifier(
                random_state=42,
                n_estimators=100,
                min_samples_split=2,
                max_features='sqrt',
                max_depth=None
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 7. MLPClassifier with handcrafted features (0.8095)
        {
            'name': 'MLPClassifier_WithHandcrafted',
            'classifier': MLPClassifier(
                random_state=42,
                max_iter=200,
                n_iter_no_change=10,
                early_stopping=True,
                hidden_layer_sizes=(50,),
                alpha=0.001,
                activation='relu',
                solver='adam',
                learning_rate_init=0.005
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 2)
            }
        },
        # 8. SVC_Poly with handcrafted features (0.8082)
        {
            'name': 'SVC_Poly_WithHandcrafted',
            'classifier': SVC(
                random_state=42,
                kernel='poly',
                gamma=0.001,
                degree=3,
                coef0=2.0,
                class_weight='balanced',
                C=100.0,
                probability=True  # Needed for soft voting and stacking
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        },
        # 9. MultinomialNB with handcrafted features (0.8060)
        {
            'name': 'MultinomialNB_WithHandcrafted',
            'classifier': MultinomialNB(alpha=0.1),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 3)
            }
        },
        # 10. BernoulliNB with handcrafted features (0.8059)
        {
            'name': 'BernoulliNB_WithHandcrafted',
            'classifier': BernoulliNB(binarize=0.0, alpha=1.0),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        }
    ]

    # Create pipelines for each classifier with their best parameters
    print("Creating classifier pipelines with best parameters...")
    estimators = []
    for clf_config in classifiers:
        print(f"- Creating pipeline for {clf_config['name']}...")

        # Create text pipeline with best TF-IDF parameters
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import FunctionTransformer
        from unified_tuning_framework import (
            TextPreprocessorTransformer,
            get_text_column,
            get_handcrafted_features,
            create_text_pipeline,
            create_feature_union,
            SparseToDenseTransformer
        )

        text_pipeline = Pipeline([
            ('selector', FunctionTransformer(get_text_column, validate=False)),
            ('preprocessor', TextPreprocessorTransformer(
                use_stemming=False,
                use_lemmatization=True
            )),
            ('tfidf', TfidfVectorizer(**clf_config['tfidf_params']))
        ])

        # Create complete pipeline with or without handcrafted features
        pipeline = create_pipeline(
            clf_config['classifier'],
            use_handcrafted=clf_config['use_handcrafted'],
            text_pipeline=text_pipeline
        )

        estimators.append((clf_config['name'], pipeline))

    # Create ensemble classifiers
    print("\nCreating ensemble classifiers...")
    ensemble_classifiers = [
        # Stacking Classifier
        {
            'name': 'StackingClassifier',
            'classifier': StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=CV,
                n_jobs=N_JOBS,
                passthrough=False
            )
        },
        # Voting Classifier with hard voting
        {
            'name': 'VotingClassifier_Hard',
            'classifier': VotingClassifier(
                estimators=estimators,
                voting='hard',
                n_jobs=N_JOBS
            )
        },
        # Voting Classifier with soft voting
        {
            'name': 'VotingClassifier_Soft',
            'classifier': VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=N_JOBS
            )
        }
    ]

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Track all results for summary
    all_results = []

    # Evaluate each ensemble classifier
    for ensemble_config in ensemble_classifiers:
        print(f"\n{'=' * 80}")
        print(f"Evaluating {ensemble_config['name']}...")
        print(f"{'=' * 80}")

        # Check if results file already exists
        results_filename = f"{OUTPUT_DIR}/{ensemble_config['name'].lower()}_results.json"
        if os.path.exists(results_filename):
            print(f"✓ Results already exist for {ensemble_config['name']}, skipping...")
            print(f"{'=' * 80}")
            continue

        try:
            # Split features based on whether handcrafted features are needed
            # For ensemble models, we need to use the full dataframe with all features
            # because some base estimators may require handcrafted features
            X_features = df.copy()

            # Perform cross-validation manually for better control
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import f1_score

            skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
            cv_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
                print(f"  - Fold {fold_idx + 1}/{CV}...")

                X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train the ensemble classifier
                ensemble_config['classifier'].fit(X_train, y_train)

                # Predict and evaluate
                y_pred = ensemble_config['classifier'].predict(X_val)
                fold_score = f1_score(y_val, y_pred, average='weighted')
                cv_scores.append(fold_score)

                print(f"    Fold {fold_idx + 1} F1 score: {fold_score:.4f}")

            # Calculate average score
            avg_score = sum(cv_scores) / len(cv_scores)

            print(f"\n  ✓ Completed {ensemble_config['name']}")
            print(f"    Average weighted F1 score: {avg_score:.4f}")
            print(f"    All fold scores: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"{'=' * 80}")

            # Create results dictionary
            results = {
                'classifier': ensemble_config['name'],
                'base_estimators': [name for name, _ in estimators],
                'cv_folds': CV,
                'cv_scores': cv_scores,
                'average_score': float(avg_score),
                'n_jobs': N_JOBS,
                'datetime': json.loads(json.dumps({'now': None}))['now']  # For serialization
            }

            # Save results to JSON file
            with open(results_filename, 'w') as f:
                json.dump(results, f, default=str, indent=2)

            print(f"  ✓ Results saved to: {results_filename}")

            # Add to all results
            all_results.append(results)

        except Exception as e:
            print(f"\n✗ Failed to evaluate {ensemble_config['name']}: {str(e)}")
            print(f"{'=' * 80}")

    # Generate summary
    print("\n" + "=" * 80)
    print("All ensemble classifiers have been processed!")
    print("=" * 80)
    print("Results have been saved to the 'res' directory.")

    # Print summary
    print("\nEnsemble Classifier Performance Summary:")
    print("-" * 60)
    for result in all_results:
        print(f"{result['classifier']}: {result['average_score']:.4f}")


if __name__ == "__main__":
    main()

import os
import json
import warnings
from itertools import combinations
from tqdm import tqdm

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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings('ignore')

# Local imports
from unified_tuning_framework import (
    load_and_prepare_data,
    create_pipeline,
    TextPreprocessorTransformer,
    get_text_column
)

def generate_classifier_combinations(classifiers, k_values):
    """Generate all combinations of classifiers for given k values

    Args:
        classifiers: List of classifier configurations
        k_values: List of integers representing the number of classifiers in each combination

    Returns:
        Dictionary with k as key and list of combinations as value
    """
    combinations_dict = {}
    for k in k_values:
        if k < 1 or k > len(classifiers):
            print(f"Warning: Invalid k value {k}, skipping...")
            continue

        print(f"Generating all combinations of {k} classifiers...")
        # Generate all combinations of k classifiers
        all_combinations = list(combinations(classifiers, k))
        print(f"  Found {len(all_combinations)} combinations for k={k}")
        combinations_dict[k] = all_combinations

    return combinations_dict

def create_estimators_from_combination(classifier_combination):
    """Create estimators list from a classifier combination

    Args:
        classifier_combination: Tuple of classifier configurations

    Returns:
        List of estimators (name, pipeline) for ensemble classifiers
    """
    estimators = []
    for clf_config in classifier_combination:
        # Create text pipeline with best TF-IDF parameters
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

    return estimators

def load_existing_results(results_file):
    """Load existing results from JSON file

    Args:
        results_file: Path to results file

    Returns:
        Dictionary of existing results with combination signature as key
    """
    existing_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                file_content = f.read().strip()

                if not file_content:
                    return existing_results

                # Parse JSON content
                results = json.loads(file_content)

                # Handle case where results is a list of dictionaries
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and 'base_estimators' in result:
                            # Create a signature for the combination
                            combination_signature = tuple(sorted(result['base_estimators']))
                            existing_results[combination_signature] = result
                        else:
                            print(f"Warning: Invalid result format in {results_file}, skipping...")
                # Handle case where results is a single dictionary (legacy format)
                elif isinstance(results, dict) and 'base_estimators' in results:
                    combination_signature = tuple(sorted(results['base_estimators']))
                    existing_results[combination_signature] = results
                else:
                    print(f"Warning: Unexpected results format in {results_file}, skipping...")

        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing {results_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Unexpected error loading {results_file}: {str(e)}")
    return existing_results

def save_results(results_file, new_results):
    """Save new results to JSON file, appending if file exists

    Args:
        results_file: Path to results file
        new_results: List of new results to save
    """
    # Load existing results
    existing_results = load_existing_results(results_file)

    # Combine existing and new results, overwriting existing entries with same signature
    for result in new_results:
        combination_signature = tuple(sorted(result['base_estimators']))
        existing_results[combination_signature] = result

    # Convert back to list and save
    all_results = list(existing_results.values())
    with open(results_file, 'w') as f:
        json.dump(all_results, f, default=str, indent=2)

def main():
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    CV = 5  # Cross-validation folds
    N_JOBS = 2  # Number of parallel jobs
    OUTPUT_DIR = 'res'

    # Define the k values to test (number of classifiers in each combination)
    K_VALUES = [7, 9]  # Can be extended to [3, 5, 7, 9] later

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data ONCE
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

    # Generate classifier combinations
    print("\nGenerating classifier combinations...")
    classifier_combinations = generate_classifier_combinations(classifiers, K_VALUES)

    # Load existing results to avoid duplicates
    voting_results_file = f"{OUTPUT_DIR}/votingclassifier_hard_results.json"
    stacking_results_file = f"{OUTPUT_DIR}/stackingclassifier_results.json"

    existing_voting_results = load_existing_results(voting_results_file)
    existing_stacking_results = load_existing_results(stacking_results_file)

    # Track all results
    all_voting_results = []
    all_stacking_results = []

    # Evaluate each combination
    for k, combinations_list in classifier_combinations.items():
        print(f"\n{'=' * 80}")
        print(f"Testing combinations with {k} classifiers")
        print(f"{'=' * 80}")

        # Create progress bar
        total_combinations = len(combinations_list)
        with tqdm(total=total_combinations, desc=f"Processing {k}-classifier combinations") as pbar:
            for combo_idx, classifier_combination in enumerate(combinations_list):
                # Create estimators for this combination
                estimators = create_estimators_from_combination(classifier_combination)
                estimator_names = [name for name, _ in estimators]

                # Create combination signature for result checking
                combo_signature = tuple(sorted(estimator_names))

                # Check if this combination has already been tested
                if combo_signature in existing_voting_results:
                    print(f"  Skipping VotingClassifier_Hard for combination {combo_idx + 1}/{total_combinations} (already exists)")
                    skip_voting = True
                else:
                    skip_voting = False

                if combo_signature in existing_stacking_results:
                    print(f"  Skipping StackingClassifier for combination {combo_idx + 1}/{total_combinations} (already exists)")
                    skip_stacking = True
                else:
                    skip_stacking = False

                if skip_voting and skip_stacking:
                    pbar.update(1)
                    continue

                print(f"\n  Testing combination {combo_idx + 1}/{total_combinations}:")
                print(f"    Classifiers: {', '.join(estimator_names)}")

                # Create and evaluate Voting Classifier with hard voting
                if not skip_voting:
                    try:
                        print(f"    Creating VotingClassifier_Hard...")
                        voting_clf = VotingClassifier(
                            estimators=estimators,
                            voting='hard',
                            n_jobs=N_JOBS
                        )

                        print(f"    Evaluating VotingClassifier_Hard...")
                        # Split features based on whether handcrafted features are needed
                        X_features = df.copy()

                        # Perform cross-validation manually for better control
                        from sklearn.model_selection import StratifiedKFold
                        from sklearn.metrics import f1_score

                        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
                        cv_scores = []

                        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
                            X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                            # Train the classifier
                            voting_clf.fit(X_train, y_train)

                            # Predict and evaluate
                            y_pred = voting_clf.predict(X_val)
                            fold_score = f1_score(y_val, y_pred, average='weighted')
                            cv_scores.append(fold_score)

                        # Calculate average score
                        avg_score = sum(cv_scores) / len(cv_scores)

                        print(f"    ✓ VotingClassifier_Hard - Average F1 score: {avg_score:.4f}")

                        # Create results dictionary
                        voting_results = {
                            'classifier': 'VotingClassifier_Hard',
                            'base_estimators': estimator_names,
                            'num_estimators': k,
                            'cv_folds': CV,
                            'cv_scores': cv_scores,
                            'average_score': float(avg_score),
                            'n_jobs': N_JOBS,
                            'datetime': json.loads(json.dumps({'now': None}))['now']  # For serialization
                        }

                        # Save results to all results list
                        all_voting_results.append(voting_results)

                    except Exception as e:
                        print(f"    ✗ Failed to evaluate VotingClassifier_Hard: {str(e)}")

                # Create and evaluate Stacking Classifier
                if not skip_stacking:
                    try:
                        print(f"    Creating StackingClassifier...")
                        stacking_clf = StackingClassifier(
                            estimators=estimators,
                            final_estimator=LogisticRegression(random_state=42),
                            cv=CV,
                            n_jobs=N_JOBS,
                            passthrough=False
                        )

                        print(f"    Evaluating StackingClassifier...")
                        # Split features based on whether handcrafted features are needed
                        X_features = df.copy()

                        # Perform cross-validation manually for better control
                        from sklearn.model_selection import StratifiedKFold
                        from sklearn.metrics import f1_score

                        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
                        cv_scores = []

                        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
                            X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                            # Train the classifier
                            stacking_clf.fit(X_train, y_train)

                            # Predict and evaluate
                            y_pred = stacking_clf.predict(X_val)
                            fold_score = f1_score(y_val, y_pred, average='weighted')
                            cv_scores.append(fold_score)

                        # Calculate average score
                        avg_score = sum(cv_scores) / len(cv_scores)

                        print(f"    ✓ StackingClassifier - Average F1 score: {avg_score:.4f}")

                        # Create results dictionary
                        stacking_results = {
                            'classifier': 'StackingClassifier',
                            'base_estimators': estimator_names,
                            'num_estimators': k,
                            'final_estimator': 'LogisticRegression',
                            'cv_folds': CV,
                            'cv_scores': cv_scores,
                            'average_score': float(avg_score),
                            'n_jobs': N_JOBS,
                            'datetime': json.loads(json.dumps({'now': None}))['now']  # For serialization
                        }

                        # Save results to all results list
                        all_stacking_results.append(stacking_results)

                    except Exception as e:
                        print(f"    ✗ Failed to evaluate StackingClassifier: {str(e)}")

                # Update progress bar
                pbar.update(1)

    # Save all results
    if all_voting_results:
        print(f"\nSaving VotingClassifier results to {voting_results_file}...")
        save_results(voting_results_file, all_voting_results)

    if all_stacking_results:
        print(f"Saving StackingClassifier results to {stacking_results_file}...")
        save_results(stacking_results_file, all_stacking_results)

    # Generate summary
    print(f"\n{'=' * 80}")
    print("All combinations have been processed!")
    print(f"{'=' * 80}")
    print("Results have been saved to the 'res' directory.")

    # Print summary
    print(f"\nEnsemble Classifier Performance Summary:")
    print(f"- Total VotingClassifier_Hard combinations processed: {len(all_voting_results)}")
    print(f"- Total StackingClassifier combinations processed: {len(all_stacking_results)}")


if __name__ == "__main__":
    main()
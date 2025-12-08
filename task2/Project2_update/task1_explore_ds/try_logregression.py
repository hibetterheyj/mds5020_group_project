import pandas as pd
import numpy as np
import json
import joblib
import warnings
from typing import Dict, List, Any, Tuple

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Local imports
from handcrafted_features import create_sentiment_features
from improved_sentiment_model import EnhancedTextPreprocessor

class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Transformer for text preprocessing using the EnhancedTextPreprocessor"""

    def __init__(self):
        self.preprocessor = EnhancedTextPreprocessor(
            use_stemming=False,
            use_lemmatization=True
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocessor.preprocess(text) for text in X]


def load_and_prepare_data(file_path: str, label_mapping: Dict[int, int] = {-1: 0, 1: 1}) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for training

    Args:
        file_path: Path to the Excel data file
        label_mapping: Mapping from original labels to model labels

    Returns:
        Tuple of (processed dataframe, processed text series, labels)
    """
    df = pd.read_excel(file_path)

    # Create handcrafted features
    handcrafted_features = create_sentiment_features(df)
    df = pd.concat([df, handcrafted_features], axis=1)

    # Map labels
    df['sentiment_original'] = df['sentiment'].astype(int)
    df['sentiment'] = df['sentiment_original'].map(label_mapping)

    print(f"Dataset size: {len(df)}")
    print(f"Original class distribution:\n{df['sentiment_original'].value_counts()}")
    print(f"Mapped class distribution:\n{df['sentiment'].value_counts()}")

    return df, df['news_title'], df['sentiment']


def create_logregression_pipeline(use_handcrafted: bool = True) -> Pipeline:
    """Create Logistic Regression pipeline with or without handcrafted features

    Args:
        use_handcrafted: Whether to include handcrafted features

    Returns:
        Sklearn Pipeline object
    """
    # Text processing pipeline
    text_pipeline = Pipeline([
        ('preprocessor', TextPreprocessorTransformer()),
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            use_idf=True
        ))
    ])

    if use_handcrafted:
        # Text feature extraction
        def get_text_column(X):
            return X['news_title']

        # Handcrafted features extraction
        def get_handcrafted_features(X):
            return X[[
                'pos_word_count', 'neg_word_count', 'net_sentiment',
                'has_strong_positive', 'has_strong_negative',
                'financial_word_count', 'financial_density'
            ]]

        # Combined pipelines
        feature_union = FeatureUnion([
            ('text', Pipeline([
                ('selector', FunctionTransformer(get_text_column, validate=False)),
                ('preprocessor', TextPreprocessorTransformer()),
                ('tfidf', TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    use_idf=True
                ))
            ])),
            ('handcrafted', Pipeline([
                ('selector', FunctionTransformer(get_handcrafted_features, validate=False)),
                ('scaler', StandardScaler())
            ]))
        ])

        # Complete pipeline with Logistic Regression
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', LogisticRegression(random_state=42))
        ])
    else:
        # Text-only pipeline
        pipeline = Pipeline([
            ('text', text_pipeline),
            ('classifier', LogisticRegression(random_state=42))
        ])

    return pipeline


def tune_logregression_hyperparameters(
    df: pd.DataFrame,
    X: pd.Series,
    y: pd.Series,
    n_iter: int = 500,
    cv: int = 5,
    n_jobs: int = -2
) -> Dict[str, Any]:
    """Tune Logistic Regression hyperparameters using RandomizedSearchCV

    Args:
        df: Processed dataframe with handcrafted features
        X: Input features (news titles)
        y: Labels
        n_iter: Number of iterations for RandomizedSearch
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with tuning results
    """
    print("=" * 60)
    print("Starting Logistic Regression Hyperparameter Tuning")
    print("=" * 60)

    # Cross-validation settings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Results container
    all_results = []

    # Test both with and without handcrafted features
    for use_handcrafted in [True, False]:
        print(f"\n{'=' * 40}")
        print(f"Testing with handcrafted features: {use_handcrafted}")
        print(f"{'=' * 40}")

        # Create pipeline
        pipeline = create_logregression_pipeline(use_handcrafted=use_handcrafted)

        # Prepare features
        if use_handcrafted:
            X_features = df.copy()  # DataFrame with handcrafted features
        else:
            X_features = X.copy()   # Series with just the text data

        # Parameter grid for Logistic Regression
        param_grid = {
            'classifier__C': np.logspace(-4, 4, 20),  # Inverse of regularization strength
            'classifier__solver': ['liblinear', 'lbfgs'],  # Solvers compatible with both L1 and L2 penalties
            'classifier__penalty': ['l1', 'l2'],  # Regularization type
            'classifier__max_iter': [100, 200, 500, 1000],  # Maximum number of iterations
            'classifier__class_weight': [None, 'balanced'],  # Class weights
        }

        # Adjust parameter names based on pipeline type
        if use_handcrafted:
            param_grid = {
                'features__text__tfidf__max_features': [2000, 3000, 5000],
                'features__text__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__C': np.logspace(-4, 4, 20),
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__max_iter': [100, 200, 500, 1000],
                'classifier__class_weight': [None, 'balanced'],
            }
        else:
            param_grid = {
                'text__tfidf__max_features': [2000, 3000, 5000],
                'text__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__C': np.logspace(-4, 4, 20),
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__max_iter': [100, 200, 500, 1000],
                'classifier__class_weight': [None, 'balanced'],
            }

        # Run RandomizedSearchCV with 500 iterations
        print(f"\nRunning RandomizedSearchCV with {n_iter} iterations...")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=f1_scorer,
            cv=skf,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )
        random_search.fit(X_features, y)

        print(f"Best RandomizedSearchCV score: {random_search.best_score_:.4f}")
        print(f"Best parameters: {random_search.best_params_}")

        # Extract results from RandomizedSearchCV
        for i, (params, mean_score, std_score) in enumerate(zip(
            random_search.cv_results_['params'],
            random_search.cv_results_['mean_test_score'],
            random_search.cv_results_['std_test_score']
        )):
            # Get individual fold scores
            fold_scores = []
            for j in range(cv):
                fold_score = random_search.cv_results_[f'split{j}_test_score'][i]
                fold_scores.append(fold_score)

            # Add use_handcrafted to parameters
            params['use_handcrafted'] = use_handcrafted

            # Store results
            all_results.append({
                'parameters': params,
                'metrics': {
                    'mean_score': float(mean_score),
                    'std_score': float(std_score),
                    'scores': [float(s) for s in fold_scores]
                }
            })

    return all_results


def main():
    """Main function to run Logistic Regression hyperparameter tuning"""
    # File path
    data_path = '../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

    # Load data
    df, X, y = load_and_prepare_data(data_path)

    # Tune hyperparameters with 500 iterations
    tuning_results = tune_logregression_hyperparameters(df, X, y, n_iter=500, cv=5, n_jobs=-2)

    # Save results to JSON
    with open('logregression_tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Tuning completed! Results saved to logregression_tuning_results.json")
    print(f"Total results: {len(tuning_results)}")

    # Print best results
    best_result = max(tuning_results, key=lambda x: x['metrics']['mean_score'])
    print(f"\nBest result:")
    print(f"Mean F1 score: {best_result['metrics']['mean_score']:.4f}")
    print(f"Parameters: {best_result['parameters']}")

    # Compare handcrafted vs non-handcrafted results
    handcrafted_results = [r for r in tuning_results if r['parameters']['use_handcrafted']]
    non_handcrafted_results = [r for r in tuning_results if not r['parameters']['use_handcrafted']]
    
    best_handcrafted = max(handcrafted_results, key=lambda x: x['metrics']['mean_score'])
    best_non_handcrafted = max(non_handcrafted_results, key=lambda x: x['metrics']['mean_score'])
    
    print(f"\nComparison:")
    print(f"Best with handcrafted features: {best_handcrafted['metrics']['mean_score']:.4f}")
    print(f"Best without handcrafted features: {best_non_handcrafted['metrics']['mean_score']:.4f}")
    print(f"Improvement: {best_handcrafted['metrics']['mean_score'] - best_non_handcrafted['metrics']['mean_score']:.4f}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import os
from typing import Dict, List, Any, Tuple, Optional, Callable
from tqdm.auto import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Local imports
from handcrafted_features import create_sentiment_features
from text_preprocessor import EnhancedTextPreprocessor


class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Transformer for text preprocessing using the EnhancedTextPreprocessor"""

    def __init__(self, use_stemming: bool = False, use_lemmatization: bool = True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.preprocessor = EnhancedTextPreprocessor(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
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
        Tuple of (processed dataframe with handcrafted features, processed text series, labels)
    """
    df = pd.read_excel(file_path)

    # Create handcrafted features
    handcrafted_features = create_sentiment_features(df)
    df = pd.concat([df, handcrafted_features], axis=1)

    # Process labels
    df['sentiment'] = df['sentiment'].map(label_mapping)

    # Separate features and labels
    X = df['news_title']
    y = df['sentiment']

    return df, X, y


def get_text_column(df: pd.DataFrame) -> pd.Series:
    """Function to extract text column from dataframe"""
    if isinstance(df, pd.DataFrame) and 'news_title' in df.columns:
        return df['news_title']
    else:
        return df


def get_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Function to extract handcrafted features from dataframe"""
    # List of handcrafted feature columns (from handcrafted_features.py)
    feature_columns = [
        # Basic sentiment features
        'pos_word_count', 'neg_word_count', 'pos_ratio', 'neg_ratio', 'net_sentiment',
        'has_strong_positive', 'has_strong_negative',

        # Financial features
        'financial_word_count', 'financial_density',
        'has_earnings', 'has_dividend', 'has_forecast', 'has_rating',

        # Title length features
        'word_count', 'char_count', 'avg_word_length', 'is_short_title', 'is_long_title',

        # Number and percentage features
        'has_number', 'has_percentage', 'has_money',

        # Sentiment word position features
        'pos_in_first_three', 'neg_in_first_three', 'pos_in_last_three', 'neg_in_last_three',

        # Negation features
        'has_negation', 'negated_pos_count', 'negated_neg_count'
    ]

    return df[feature_columns]


def create_text_pipeline(use_stemming: bool = False, use_lemmatization: bool = True) -> Pipeline:
    """Create a text-only pipeline with preprocessing and TF-IDF vectorization"""
    return Pipeline([
        ('selector', FunctionTransformer(get_text_column, validate=False)),
        ('preprocessor', TextPreprocessorTransformer(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )),
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            use_idf=True
        ))
    ])


def create_feature_union(text_pipeline: Pipeline) -> FeatureUnion:
    """Create a feature union with text pipeline and handcrafted features"""
    return FeatureUnion([
        ('text', text_pipeline),
        ('handcrafted', Pipeline([
            ('selector', FunctionTransformer(get_handcrafted_features, validate=False)),
            ('scaler', StandardScaler())
        ]))
    ])


def create_pipeline(
    classifier: BaseEstimator,
    use_handcrafted: bool = True,
    text_pipeline: Optional[Pipeline] = None
) -> Pipeline:
    """Create a classifier pipeline with optional handcrafted features

    Args:
        classifier: The classifier to use
        use_handcrafted: Whether to include handcrafted features
        text_pipeline: Optional pre-configured text pipeline

    Returns:
        Complete sklearn Pipeline
    """
    if text_pipeline is None:
        text_pipeline = create_text_pipeline()

    if use_handcrafted:
        # Combined pipeline with both text and handcrafted features
        feature_union = create_feature_union(text_pipeline)
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', classifier)
        ])
    else:
        # Text-only pipeline
        pipeline = Pipeline([
            ('text', text_pipeline),
            ('classifier', classifier)
        ])

    return pipeline


def tune_classifier(
    df: pd.DataFrame,
    X: pd.Series,
    y: pd.Series,
    classifier_name: str,
    classifier: BaseEstimator,
    param_grid: Dict[str, Any],
    use_handcrafted_options: List[bool] = [True, False],
    n_iter: int = 100,
    cv: int = 5,
    n_jobs: int = -2,
    output_dir: str = "res"
) -> Dict[str, Any]:
    """Tune a classifier using RandomizedSearchCV

    Args:
        df: Processed dataframe with handcrafted features
        X: Input features (news titles)
        y: Labels
        classifier_name: Name of the classifier for display purposes
        classifier: The classifier to tune
        param_grid: Parameter grid for RandomizedSearch
        use_handcrafted_options: List of boolean values indicating whether to use handcrafted features
        n_iter: Number of iterations for RandomizedSearch
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        output_dir: Directory to save results

    Returns:
        Dictionary with tuning results
    """
    print(f"\n{'=' * 60}")
    print(f"Starting {classifier_name.upper()} Hyperparameter Tuning")
    print(f"{'=' * 60}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Cross-validation settings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Results container
    all_results = []

    # Test with different handcrafted feature options
    for use_handcrafted in use_handcrafted_options:
        print(f"\n{'=' * 40}")
        print(f"Testing with handcrafted features: {use_handcrafted}")
        print(f"{'=' * 40}")

        # Create pipeline
        pipeline = create_pipeline(classifier, use_handcrafted=use_handcrafted)

        # Prepare features
        if use_handcrafted:
            X_features = df.copy()  # DataFrame with handcrafted features
            # Adjust parameter names for combined pipeline
            adjusted_param_grid = {}
            for param_name, param_values in param_grid.items():
                if param_name.startswith('classifier__'):
                    adjusted_param_grid[param_name] = param_values
                else:
                    adjusted_param_grid[f'features__text__{param_name}'] = param_values
        else:
            X_features = X.copy()   # Series with just the text data
            # Adjust parameter names for text-only pipeline
            adjusted_param_grid = {}
            for param_name, param_values in param_grid.items():
                if param_name.startswith('classifier__'):
                    adjusted_param_grid[param_name] = param_values
                else:
                    adjusted_param_grid[f'text__{param_name}'] = param_values

        # Run RandomizedSearchCV with tqdm integration
        print(f"Running RandomizedSearchCV with {n_iter} iterations...")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=adjusted_param_grid,
            n_iter=n_iter,
            scoring=f1_scorer,
            cv=skf,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            return_train_score=True
        )

        # Fit with tqdm (using a custom wrapper)
        with tqdm(total=n_iter, desc=f"Tuning {classifier_name} (handcrafted={use_handcrafted})") as pbar:
            def update_progress(step):
                pbar.update(1)
                return step

            # Set up callback if supported
            random_search.fit(X_features, y)

        # Get the best parameters and score
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        # Get all CV results
        cv_results = random_search.cv_results_

        # Create a result entry
        result = {
            'classifier': classifier_name,
            'use_handcrafted': use_handcrafted,
            'best_params': best_params,
            'best_score': float(best_score),
            'cv_results': cv_results,
            'n_iter': n_iter,
            'cv_folds': cv,
            'datetime': datetime.now().isoformat()
        }

        all_results.append(result)

        print(f"Best parameters: {best_params}")
        print(f"Best weighted F1 score: {best_score:.4f}")

    # Save results to JSON file
    results_filename = f"{output_dir}/{classifier_name.lower()}_tuning_results.json"
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, default=str, indent=2)

    print(f"\nResults saved to: {results_filename}")
    print(f"{'=' * 60}")
    print(f"{classifier_name} Hyperparameter Tuning Complete")
    print(f"{'=' * 60}")

    return {
        'all_results': all_results,
        'best_overall': max(all_results, key=lambda x: x['best_score'])
    }


def generate_summary_markdown(
    results_files: List[str],
    output_file: str = "classifier_comparison_summary.md"
) -> str:
    """Generate a markdown summary comparing all classifiers

    Args:
        results_files: List of JSON results files
        output_file: Path to save the markdown summary

    Returns:
        Markdown content as string
    """
    print("\nGenerating classifier comparison summary...")

    all_results = []

    # Load all results files
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_results:
        print("No results found!")
        return ""

    # Group results by classifier
    results_by_classifier = {}
    for result in all_results:
        clf_name = result['classifier']
        if clf_name not in results_by_classifier:
            results_by_classifier[clf_name] = []
        results_by_classifier[clf_name].append(result)

    # Generate markdown content
    markdown = "# Classifier Comparison Summary\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "## Table of Contents\n\n"

    # Add table of contents
    for clf_name in sorted(results_by_classifier.keys()):
        markdown += f"- [{clf_name}](#{clf_name.lower().replace(' ', '-')})\n"

    markdown += "- [Overall Comparison](#overall-comparison)\n"
    markdown += "- [Handcrafted Features Impact](#handcrafted-features-impact)\n\n"

    # Add detailed results for each classifier
    for clf_name in sorted(results_by_classifier.keys()):
        markdown += f"## {clf_name}\n\n"

        clf_results = results_by_classifier[clf_name]

        for result in clf_results:
            use_handcrafted = result['use_handcrafted']
            best_score = result['best_score']
            best_params = result['best_params']

            markdown += f"### {'With' if use_handcrafted else 'Without'} Handcrafted Features\n\n"
            markdown += f"**Best Weighted F1 Score:** {best_score:.4f}\n\n"
            markdown += "**Best Parameters:**\n\n"
            markdown += "```json\n"
            markdown += json.dumps(best_params, indent=2)
            markdown += "\n```\n\n"

    # Overall comparison table
    markdown += "## Overall Comparison\n\n"
    markdown += "| Classifier | Handcrafted Features | Best F1 Score |\n"
    markdown += "|------------|----------------------|---------------|\n"

    for clf_name in sorted(results_by_classifier.keys()):
        clf_results = results_by_classifier[clf_name]
        for result in clf_results:
            use_handcrafted = result['use_handcrafted']
            best_score = result['best_score']
            markdown += f"| {clf_name} | {'Yes' if use_handcrafted else 'No'} | {best_score:.4f} |\n"

    # Handcrafted features impact
    markdown += "\n## Handcrafted Features Impact\n\n"

    for clf_name in sorted(results_by_classifier.keys()):
        clf_results = results_by_classifier[clf_name]

        # Find results with and without handcrafted features
        with_features = next((r for r in clf_results if r['use_handcrafted']), None)
        without_features = next((r for r in clf_results if not r['use_handcrafted']), None)

        if with_features and without_features:
            score_with = with_features['best_score']
            score_without = without_features['best_score']
            improvement = (score_with - score_without) * 100

            markdown += f"- **{clf_name}:** {improvement:+.2f}% improvement (from {score_without:.4f} to {score_with:.4f})\n"

    # Save to file
    with open(output_file, 'w') as f:
        f.write(markdown)

    print(f"Summary saved to: {output_file}")

    return markdown


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Unified Classifier Tuning Framework")
    parser.add_argument('--generate-summary', action='store_true',
                      help='Generate markdown summary from all results files')
    parser.add_argument('--results-dir', type=str, default='res',
                      help='Directory containing results JSON files')
    parser.add_argument('--output-file', type=str, default='classifier_comparison_summary.md',
                      help='Output markdown file path')

    args = parser.parse_args()

    if args.generate_summary:
        # Find all results files in the specified directory
        results_files = glob.glob(os.path.join(args.results_dir, "*_tuning_results.json"))

        if not results_files:
            print(f"No results files found in {args.results_dir}!")
            return

        print(f"Found {len(results_files)} results files")
        markdown = generate_summary_markdown(results_files, args.output_file)
        print(f"\nSummary generated successfully!\nView results in {args.output_file}")
    else:
        print("Unified Tuning Framework Loaded Successfully!")
        print("Use this framework to tune multiple classifiers with consistent settings.")
        print("\nTo generate a summary of all results:")
        print("python unified_tuning_framework.py --generate-summary")


if __name__ == "__main__":
    main()
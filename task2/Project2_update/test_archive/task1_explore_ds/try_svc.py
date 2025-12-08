import pandas as pd
import numpy as np
import json
import joblib
import warnings
from typing import Dict, List, Any, Tuple, Optional
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
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


def create_svc_pipeline(svc_type: str = 'linearsvc', kernel: Optional[str] = None, use_handcrafted: bool = True) -> Pipeline:
    """Create SVC pipeline with or without handcrafted features

    Args:
        svc_type: Type of SVC ('linearsvc' or 'svc')
        kernel: Kernel type for SVC (RBF, Sigmoid, Poly); ignored for LinearSVC
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

        # Create classifier based on type
        if svc_type == 'linearsvc':
            classifier = LinearSVC(random_state=42)
        else:  # svc
            classifier = SVC(kernel=kernel, random_state=42)

        # Complete pipeline
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', classifier)
        ])
    else:
        # Text-only pipeline
        if svc_type == 'linearsvc':
            classifier = LinearSVC(random_state=42)
        else:  # svc
            classifier = SVC(kernel=kernel, random_state=42)

        pipeline = Pipeline([
            ('text', text_pipeline),
            ('classifier', classifier)
        ])

    return pipeline


def run_svc_tuning(
    df: pd.DataFrame,
    X: pd.Series,
    y: pd.Series,
    svc_type: str = 'linearsvc',
    kernel: Optional[str] = None,
    n_iter: int = 200,
    cv: int = 5,
    n_jobs: int = -2
) -> Dict[str, Any]:
    """Run hyperparameter tuning for a specific SVC type

    Args:
        df: Processed dataframe with handcrafted features
        X: Input features (news titles)
        y: Labels
        svc_type: Type of SVC ('linearsvc' or 'svc')
        kernel: Kernel type for SVC; ignored for LinearSVC
        n_iter: Number of iterations for RandomizedSearch
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with tuning results
    """
    print(f"\n{'=' * 60}")
    print(f"Starting {svc_type.upper()}{' (' + kernel.upper() + ')' if kernel else ''} Hyperparameter Tuning")
    print(f"{'=' * 60}")

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
        pipeline = create_svc_pipeline(svc_type=svc_type, kernel=kernel, use_handcrafted=use_handcrafted)

        # Prepare features
        if use_handcrafted:
            X_features = df.copy()  # DataFrame with handcrafted features
        else:
            X_features = X.copy()   # Series with just the text data

        # Base parameter grid
        base_param_grid = {
            'classifier__class_weight': [None, 'balanced'],
        }

        # Add TF-IDF parameters based on pipeline structure
        if use_handcrafted:
            base_param_grid.update({
                'features__text__tfidf__max_features': [1000, 3000, 5000, 7000],
                'features__text__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
            })
        else:
            base_param_grid.update({
                'text__tfidf__max_features': [1000, 3000, 5000, 7000],
                'text__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
            })

        # Add SVC-specific parameters
        if svc_type == 'linearsvc':
            param_grid = base_param_grid.copy()
            param_grid.update({
                'classifier__C': np.logspace(-4, 4, 20),
                'classifier__penalty': ['l1', 'l2'],
                'classifier__dual': [False],  # For larger datasets
                'classifier__max_iter': [1000, 2000, 5000],
            })
        else:  # SVC with different kernels
            param_grid = base_param_grid.copy()
            param_grid.update({
                'classifier__C': np.logspace(-3, 3, 15),
                'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-4, 2, 10)),
            })

            # Add kernel-specific parameters
            if kernel == 'poly':
                param_grid.update({
                    'classifier__degree': [2, 3, 4, 5],
                    'classifier__coef0': np.linspace(0, 1, 5),
                })
            elif kernel == 'sigmoid':
                param_grid.update({
                    'classifier__coef0': np.linspace(-1, 1, 10),
                })

        # Run RandomizedSearchCV with verbose output to show progress
        print(f"Running RandomizedSearchCV with {n_iter} iterations...")

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=f1_scorer,
            cv=skf,
            n_jobs=n_jobs,
            verbose=2,  # Use verbose=2 for detailed progress updates
            random_state=42
        )

        # Fit the model
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

            # Add metadata to parameters
            params['use_handcrafted'] = use_handcrafted
            params['svc_type'] = svc_type
            if kernel:
                params['kernel'] = kernel

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
    """Main function to run SVC hyperparameter tuning for different kernels"""
    # File path
    data_path = '../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

    # Load data
    df, X, y = load_and_prepare_data(data_path)

    # SVC configurations to tune
    svc_configs = [
        {'name': 'linear_svc', 'svc_type': 'linearsvc', 'kernel': None},
        {'name': 'svc_rbf', 'svc_type': 'svc', 'kernel': 'rbf'},
        {'name': 'svc_sigmoid', 'svc_type': 'svc', 'kernel': 'sigmoid'},
        {'name': 'svc_poly', 'svc_type': 'svc', 'kernel': 'poly'},
    ]

    # Store all results
    all_config_results = {}

    # Run tuning for each configuration
    for config in svc_configs:
        print(f"\n{'=' * 80}")
        print(f"TUNING: {config['name'].upper()}")
        print(f"{'=' * 80}")

        results = run_svc_tuning(
            df=df,
            X=X,
            y=y,
            svc_type=config['svc_type'],
            kernel=config['kernel'],
            n_iter=200,
            cv=5,
            n_jobs=-2
        )

        # Save results to JSON
        output_file = f"{config['name']}_tuning_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")
        all_config_results[config['name']] = results

    # Generate summary markdown
    generate_summary_markdown(all_config_results)


def generate_summary_markdown(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Generate a markdown summary of all SVC tuning results"""
    markdown_content = "# SVC Hyperparameter Tuning Summary\n\n"
    markdown_content += "## Experiment Setup\n\n"
    markdown_content += "- **Baseline**: Logistic Regression with TF-IDF features (0.8024 F1 score)\n"
    markdown_content += "- **Tuning Method**: RandomizedSearchCV with 200 iterations per configuration\n"
    markdown_content += "- **Cross-Validation**: 5-fold StratifiedKFold\n"
    markdown_content += "- **Evaluation Metric**: Weighted F1 Score\n\n"
    markdown_content += "## Key Parameters Tuned\n\n"
    markdown_content += "1. **TF-IDF Parameters**:\n"
    markdown_content += "   - `max_features`: Number of top features to keep\n"
    markdown_content += "   - `ngram_range`: Range of n-grams to consider\n\n"
    markdown_content += "2. **SVC Parameters**:\n"
    markdown_content += "   - `C`: Regularization parameter\n"
    markdown_content += "   - `class_weight`: Class weighting strategy\n"
    markdown_content += "   - `kernel`: Kernel type (for non-linear SVC)\n"
    markdown_content += "   - `gamma`: Kernel coefficient\n"
    markdown_content += "   - `degree`: Degree of polynomial kernel\n"
    markdown_content += "   - `coef0`: Independent term in kernel function\n\n"
    markdown_content += "3. **Feature Configuration**:\n"
    markdown_content += "   - `use_handcrafted`: Whether to include handcrafted features\n\n"
    markdown_content += "## Results Overview\n\n"
    markdown_content += "| Classifier | Best F1 Score | Use Handcrafted |\n"
    markdown_content += "|------------|---------------|-----------------|\n"

    # Find best result for each classifier
    best_results = {}
    for config_name, results in all_results.items():
        best_result = max(results, key=lambda x: x['metrics']['mean_score'])
        best_results[config_name] = best_result
        markdown_content += f"| {config_name.replace('_', ' ').title()} | {best_result['metrics']['mean_score']:.4f} | {best_result['parameters']['use_handcrafted']} |\n"

    markdown_content += "\n## Detailed Best Results\n\n"

    for config_name, best_result in best_results.items():
        markdown_content += f"### {config_name.replace('_', ' ').title()}\n\n"
        markdown_content += "#### Best Parameters\n"
        markdown_content += "```json\n"
        markdown_content += json.dumps(best_result['parameters'], indent=2)
        markdown_content += "\n```\n\n"
        markdown_content += "#### Performance Metrics\n"
        markdown_content += f"- Mean F1 Score: {best_result['metrics']['mean_score']:.4f}\n"
        markdown_content += f"- Standard Deviation: {best_result['metrics']['std_score']:.4f}\n"
        markdown_content += f"- Fold Scores: {[round(score, 4) for score in best_result['metrics']['scores']]}\n\n"

    markdown_content += "## Comparison with Baseline\n\n"
    baseline = 0.8024
    for config_name, best_result in best_results.items():
        improvement = best_result['metrics']['mean_score'] - baseline
        markdown_content += f"- {config_name.replace('_', ' ').title()}: {improvement:+.4f} F1 score over baseline\n"

    # Write to file
    with open('svc_tuning_summary.md', 'w') as f:
        f.write(markdown_content)

    print(f"\nSummary markdown generated: svc_tuning_summary.md")


if __name__ == "__main__":
    main()

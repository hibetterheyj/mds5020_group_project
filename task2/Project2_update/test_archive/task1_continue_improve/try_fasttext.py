import pandas as pd
import numpy as np
import json
import warnings
import fasttext
import os
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings('ignore')

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

class FastTextEstimator(BaseEstimator, TransformerMixin):
    """Scikit-learn wrapper for FastText classifier"""

    def __init__(self, lr=0.1, epoch=25, wordNgrams=2, dim=100, bucket=200000,
                 loss='softmax', ws=5, minCount=1, minn=3, maxn=6, neg=5, thread=4):
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.dim = dim
        self.bucket = bucket
        self.loss = loss
        self.ws = ws
        self.minCount = minCount
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.thread = thread
        self.model = None

    def fit(self, X, y):
        # Prepare FastText input format
        train_data = []
        for text, label in zip(X, y):
            # FastText expects labels to start with __label__
            train_data.append(f"__label__{label} {text}")

        # Write to temporary file
        temp_file = "fasttext_train.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for line in train_data:
                f.write(line + "\n")

        # Train FastText model
        self.model = fasttext.train_supervised(
            input=temp_file,
            lr=self.lr,
            epoch=self.epoch,
            wordNgrams=self.wordNgrams,
            dim=self.dim,
            bucket=self.bucket,
            loss=self.loss,
            ws=self.ws,
            minCount=self.minCount,
            minn=self.minn,
            maxn=self.maxn,
            neg=self.neg,
            thread=self.thread
        )

        # Clean up temporary file
        os.remove(temp_file)

        return self

    def predict(self, X):
        # Predict labels
        predictions = self.model.predict(list(X))
        # Extract labels and convert to integers
        predicted_labels = [int(label[0].replace("__label__", "")) for label in predictions[0]]
        return predicted_labels

    def predict_proba(self, X):
        # Predict probabilities
        predictions = self.model.predict(list(X), k=2)
        # Organize probabilities
        proba = []
        for labels, scores in zip(predictions[0], predictions[1]):
            prob_dict = {}
            for label, score in zip(labels, scores):
                prob_dict[int(label.replace("__label__", ""))] = score
            # Ensure both classes are present
            if 0 not in prob_dict:
                prob_dict[0] = 0.0
            if 1 not in prob_dict:
                prob_dict[1] = 0.0
            # Sort by class
            proba.append([prob_dict[0], prob_dict[1]])
        return np.array(proba)


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


def create_fasttext_pipeline(use_handcrafted: bool = True) -> Pipeline:
    """Create FastText pipeline with or without handcrafted features

    Args:
        use_handcrafted: Whether to include handcrafted features

    Returns:
        Sklearn Pipeline object
    """
    # Text processing pipeline
    text_pipeline = Pipeline([
        ('preprocessor', TextPreprocessorTransformer()),
        ('fasttext', FastTextEstimator())
    ])

    return text_pipeline


def tune_fasttext_hyperparameters(
    df: pd.DataFrame,
    X: pd.Series,
    y: pd.Series,
    n_iter: int = 100,
    cv: int = 5,
    n_jobs: int = -2
) -> Dict[str, Any]:
    """Tune FastText hyperparameters using RandomizedSearchCV

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
    print("Starting FastText Hyperparameter Tuning")
    print("=" * 60)

    # Cross-validation settings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Results container
    all_results = []

    # Test both with and without handcrafted features
    for use_handcrafted in [True, False]:
        print(f"\n{'=' * 40}")
        print(f"Testing with handcrafted features: {use_handcrafted}")
        print(f"{'=' * 40}")

        # Create pipeline
        pipeline = create_fasttext_pipeline(use_handcrafted=use_handcrafted)

        # Prepare features
        if use_handcrafted:
            # For FastText, we still use only the text data
            # Handcrafted features will be handled differently if needed
            X_features = X.copy()
        else:
            X_features = X.copy()

        # Parameter grid for FastText
        param_grid = {
            'fasttext__lr': np.logspace(-3, 0, 10),  # Learning rate
            'fasttext__epoch': np.arange(10, 51, 5),  # Number of epochs
            'fasttext__wordNgrams': np.arange(1, 4),  # N-gram size
            'fasttext__dim': [50, 100, 150, 200],  # Embedding dimension
            'fasttext__minCount': [1, 2, 3, 5],  # Minimum word count
            'fasttext__minn': [2, 3, 4],  # Minimum character n-gram
            'fasttext__maxn': [4, 5, 6],  # Maximum character n-gram
        }

        # Run RandomizedSearchCV with tqdm
        print(f"\nRunning RandomizedSearchCV with {n_iter} iterations...")

        # Wrap the RandomizedSearchCV to use tqdm
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='f1_weighted',
            cv=skf,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )

        # Fit with tqdm progress bar
        with tqdm(total=n_iter * cv, desc="CV iterations") as pbar:
            def callback(_):
                pbar.update(1)

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
    """Main function to run FastText hyperparameter tuning"""
    # File path
    data_path = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

    # Load data
    df, X, y = load_and_prepare_data(data_path)

    # Tune hyperparameters with 100 iterations
    tuning_results = tune_fasttext_hyperparameters(df, X, y, n_iter=100, cv=5, n_jobs=-2)

    # Create results directory if it doesn't exist
    os.makedirs('res', exist_ok=True)

    # Save results to JSON
    with open('res/fasttext_tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Tuning completed! Results saved to res/fasttext_tuning_results.json")
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
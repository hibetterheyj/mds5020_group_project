import pandas as pd
import numpy as np
import warnings
import os
import json
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch
from datasets import Dataset

# Transformers imports
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging
)

warnings.filterwarnings('ignore')

# Set logging level to avoid excessive output
logging.set_verbosity_error()

class DistilRobertaSentimentClassifier:
    def __init__(self, model_name: str = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                 num_labels: int = 3,
                 download_path: str = './results/distilroberta'):
        """Initialize DistilRoberta sentiment classifier

        Args:
            model_name: Name of the pre-trained DistilRoberta model
            num_labels: Number of output labels (3 for the pre-trained model: negative, neutral, positive)
            download_path: Path to download and save the model
        """
        # Create download directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_path)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
            cache_dir=download_path
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize input examples

        Args:
            examples: Dictionary of examples with 'text' key

        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(examples['text'], padding='max_length', truncation=True)

    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              epochs: int = 3, batch_size: int = 16) -> None:
        """Train method (not implemented as per requirements)
        This method is kept for compatibility but will not be used.
        """
        pass

    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels for input texts

        Args:
            texts: List of input texts

        Returns:
            List of predicted labels (-1 for negative, 1 for positive)
        """
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})

        # Tokenize dataset
        tokenized = dataset.map(self.tokenize_function, batched=True)
        tokenized.set_format('torch', columns=['input_ids', 'attention_mask'])

        # Initialize Trainer for prediction
        trainer = Trainer(model=self.model)

        # Predict
        predictions = trainer.predict(tokenized)

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

        # Ignore neutral class (index 1) and normalize probabilities
        # Only consider negative (0) and positive (2)
        negative_probs = probabilities[:, 0]
        positive_probs = probabilities[:, 2]
        normalized_denominator = negative_probs + positive_probs

        # Handle case where denominator is 0 (should be rare)
        normalized_denominator = np.where(normalized_denominator == 0, 1e-10, normalized_denominator)

        normalized_neg = negative_probs / normalized_denominator
        normalized_pos = positive_probs / normalized_denominator

        # Convert to binary labels: -1 for negative, 1 for positive
        predicted_labels = []
        for neg_prob, pos_prob in zip(normalized_neg, normalized_pos):
            if neg_prob > pos_prob:
                predicted_labels.append(-1)  # negative
            else:
                predicted_labels.append(1)   # positive

        return predicted_labels

def load_and_prepare_data(file_path: str, label_mapping: Dict[int, int] = {-1: -1, 1: 1}) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for training

    Args:
        file_path: Path to the Excel data file
        label_mapping: Mapping from original labels to model labels

    Returns:
        Tuple of (processed dataframe, processed text series, labels)
    """
    df = pd.read_excel(file_path)

    # Convert sentiment to integer
    df['sentiment'] = df['sentiment'].astype(int)

    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")

    return df, df['news_title'], df['sentiment']

def run_distilroberta_experiment(X: pd.Series, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
    """Run DistilRoberta experiment with 5-fold cross-validation

    Args:
        X: Input features (news titles)
        y: Labels
        cv: Number of cross-validation folds

    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("Running DistilRoberta Baseline Experiment")
    print("=" * 60)
    print("Using pre-trained model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    print("Performing inference only (no training)")
    print("=" * 60)

    # Cross-validation settings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Results container
    all_results = []

    # Initialize classifier once (model will be downloaded once)
    classifier = DistilRobertaSentimentClassifier()

    # Run cross-validation
    fold_scores = []
    with tqdm(total=cv, desc="5-fold CV") as pbar:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'=' * 40}")
            print(f"Fold {fold_idx + 1}/{cv}")
            print(f"{'=' * 40}")

            # Split data
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Predict and evaluate
            print("Evaluating model...")
            y_pred = classifier.predict(X_val.tolist())

            # Calculate F1 score
            f1 = f1_score(y_val, y_pred, average='weighted')
            fold_scores.append(f1)

            print(f"Fold {fold_idx + 1} F1 score: {f1:.4f}")

            pbar.update(1)
            pbar.set_postfix({"Fold": fold_idx + 1, "F1": f"{f1:.4f}"})

    # Calculate statistics
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    # Store results
    result = {
        "experiment_name": "DistilRoberta_Baseline",
        "parameters": {
            "model_name": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "download_path": "./results/distilroberta",
            "inference_only": True
        },
        "metrics": {
            "mean_weighted_f1": mean_f1,
            "std_weighted_f1": std_f1,
            "fold_scores": fold_scores
        }
    }
    all_results.append(result)

    # Print final results
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean Weighted F1-score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"All fold scores: {[f'{score:.4f}' for score in fold_scores]}")

    return all_results

def main():
    """Main function to run DistilRoberta baseline experiment"""
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    CV = 5  # Cross-validation folds
    OUTPUT_DIR = 'res'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Run DistilRoberta experiment
    results = run_distilroberta_experiment(X, y, cv=CV)

    # Save results to JSON file
    results_filename = f"{OUTPUT_DIR}/distilroberta_baseline_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, default=str, indent=2)

    print(f"\nResults saved to: {results_filename}")

if __name__ == "__main__":
    main()
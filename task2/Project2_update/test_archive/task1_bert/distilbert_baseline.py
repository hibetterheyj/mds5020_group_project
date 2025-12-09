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
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

warnings.filterwarnings('ignore')

class DistilBertSentimentClassifier:
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_labels: int = 2):
        """Initialize DistilBert sentiment classifier

        Args:
            model_name: Name of the pre-trained DistilBert model
            num_labels: Number of output labels
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

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
        """Train the DistilBert model

        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts
            val_labels: List of validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'label': train_labels
        })

        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'label': val_labels
        })

        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)

        # Set format for PyTorch
        tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=False,  # Disable mixed precision to save memory
            gradient_accumulation_steps=1,
        )

        # Define compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1 = f1_score(labels, predictions, average='weighted')
            return {"f1": f1}

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels for input texts

        Args:
            texts: List of input texts

        Returns:
            List of predicted labels
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

        # Extract predicted labels
        predicted_labels = np.argmax(predictions.predictions, axis=-1).tolist()

        return predicted_labels

def load_and_prepare_data(file_path: str, label_mapping: Dict[int, int] = {-1: 0, 1: 1}) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for training

    Args:
        file_path: Path to the Excel data file
        label_mapping: Mapping from original labels to model labels

    Returns:
        Tuple of (processed dataframe, processed text series, labels)
    """
    df = pd.read_excel(file_path)

    # Map labels
    df['sentiment_original'] = df['sentiment'].astype(int)
    df['sentiment'] = df['sentiment_original'].map(label_mapping)

    print(f"Dataset size: {len(df)}")
    print(f"Original class distribution:\n{df['sentiment_original'].value_counts()}")
    print(f"Mapped class distribution:\n{df['sentiment'].value_counts()}")

    return df, df['news_title'], df['sentiment']

def run_distilbert_experiment(X: pd.Series, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
    """Run DistilBert experiment with 5-fold cross-validation

    Args:
        X: Input features (news titles)
        y: Labels
        cv: Number of cross-validation folds

    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("Running DistilBert Baseline Experiment")
    print("=" * 60)

    # Cross-validation settings
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Results container
    all_results = []

    # Run cross-validation
    fold_scores = []
    with tqdm(total=cv, desc="5-fold CV") as pbar:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'=' * 40}")
            print(f"Fold {fold_idx + 1}/{cv}")
            print(f"{'=' * 40}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Initialize classifier
            classifier = DistilBertSentimentClassifier()

            # Train model
            print("Training DistilBert model...")
            classifier.train(
                train_texts=X_train.tolist(),
                train_labels=y_train.tolist(),
                val_texts=X_val.tolist(),
                val_labels=y_val.tolist(),
                epochs=3,
                batch_size=16
            )

            # Predict and evaluate
            print("Evaluating model...")
            y_pred = classifier.predict(X_val.tolist())
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
        "experiment_name": "DistilBert_Baseline",
        "parameters": {
            "model_name": "distilbert-base-uncased",
            "epochs": 3,
            "batch_size": 16
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
    """Main function to run DistilBert baseline experiment"""
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    CV = 5  # Cross-validation folds
    OUTPUT_DIR = 'res'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Run DistilBert experiment
    results = run_distilbert_experiment(X, y, cv=CV)

    # Save results to JSON file
    results_filename = f"{OUTPUT_DIR}/distilbert_baseline_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, default=str, indent=2)

    print(f"\nResults saved to: {results_filename}")

if __name__ == "__main__":
    main()
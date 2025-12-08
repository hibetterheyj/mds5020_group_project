import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Tuple, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np

# Set up paths
data_path = '../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
task1_explore_path = './task1_explore'

# Ensure output directory exists
os.makedirs(task1_explore_path, exist_ok=True)

# Preprocessing function (copied from existing code)
def preprocess_text_sentiment(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove digits and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize by splitting on whitespace
    words = text.split()
    # Define a stopword list
    stopwords = set(
        ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was',
         'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
         'may', 'might', 'must'])
    # Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]
    # Join words back into a string
    return ' '.join(filtered_words)

def load_and_explore_data() -> pd.DataFrame:
    """Load and perform initial EDA on the sentiment analysis dataset."""
    print("Loading sentiment analysis dataset...")
    df = pd.read_excel(data_path)

    print("\n=== Dataset Overview ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Sentiment Distribution ===")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)

    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(os.path.join(task1_explore_path, 'sentiment_distribution.png'))
    plt.close()

    # Analyze text length
    df['text_length'] = df['news_title'].apply(lambda x: len(str(x)))
    df['word_count'] = df['news_title'].apply(lambda x: len(str(x).split()))

    print("\n=== Text Length Analysis ===")
    print(f"Average text length: {df['text_length'].mean():.2f} characters")
    print(f"Average word count: {df['word_count'].mean():.2f} words")
    print(f"Min text length: {df['text_length'].min()} characters")
    print(f"Max text length: {df['text_length'].max()} characters")

    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['text_length'], bins=50)
    plt.title('Text Length Distribution (Characters)')
    plt.xlabel('Number of Characters')

    plt.subplot(1, 2, 2)
    sns.histplot(df['word_count'], bins=50)
    plt.title('Text Length Distribution (Words)')
    plt.xlabel('Number of Words')

    plt.tight_layout()
    plt.savefig(os.path.join(task1_explore_path, 'text_length_distribution.png'))
    plt.close()

    return df

def analyze_existing_model(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Analyze the existing sentiment analysis model and identify misclassifications."""
    print("\n=== Analyzing Existing Model ===")

    # Apply preprocessing
    df['processed_text'] = df['news_title'].apply(preprocess_text_sentiment)

    # Split data
    X = df['processed_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline (same as existing model)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('model', LogisticRegression(random_state=42, class_weight='balanced'))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\nTest Set Results:")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(task1_explore_path, 'confusion_matrix.png'))
    plt.close()

    # Analyze misclassifications
    test_df = df.loc[indices_test].copy()
    test_df['predicted_sentiment'] = y_pred
    test_df['predicted_probability'] = y_pred_proba.max(axis=1)

    misclassified = test_df[test_df['sentiment'] != test_df['predicted_sentiment']]
    print(f"\nMisclassified samples: {len(misclassified)} out of {len(test_df)} ({len(misclassified)/len(test_df)*100:.2f}%)")

    # Save misclassified examples
    misclassified.to_csv(os.path.join(task1_explore_path, 'misclassified_samples.csv'), index=False, encoding='utf-8')
    print(f"\nMisclassified samples saved to: {os.path.join(task1_explore_path, 'misclassified_samples.csv')}")

    # Cross-validation (same as existing code)
    f1_scorer = make_scorer(f1_score, average='weighted')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=f1_scorer)

    results = {
        "f1_scores_per_fold": cv_scores.tolist(),
        "mean_weighted_f1_score": cv_scores.mean(),
        "standard_deviation": cv_scores.std()
    }

    print(f"\n5-Fold Cross Validation Results:")
    print(f"Mean Weighted F1-Score: {results['mean_weighted_f1_score']:.4f}")
    print(f"Standard Deviation: {results['standard_deviation']:.4f}")

    return results, misclassified

def analyze_misclassified_samples(misclassified: pd.DataFrame):
    """Analyze patterns in misclassified samples."""
    print("\n=== Analyzing Misclassified Samples ===")

    # Distribution of misclassified sentiments
    print("\nMisclassified Sentiment Distribution:")
    print(misclassified['sentiment'].value_counts())

    # Example misclassified samples
    print("\nExample Misclassified Samples (Top 10):")
    for idx, row in misclassified.head(10).iterrows():
        print(f"Actual: {row['sentiment']}, Predicted: {row['predicted_sentiment']}, Prob: {row['predicted_probability']:.4f}")
        print(f"Text: {row['news_title']}")
        print(f"Processed: {row['processed_text']}")
        print()

    # Analyze text length for misclassified samples
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(misclassified['text_length'], bins=30)
    plt.title('Text Length of Misclassified Samples (Characters)')
    plt.xlabel('Number of Characters')

    plt.subplot(1, 2, 2)
    sns.histplot(misclassified['word_count'], bins=30)
    plt.title('Word Count of Misclassified Samples')
    plt.xlabel('Number of Words')

    plt.tight_layout()
    plt.savefig(os.path.join(task1_explore_path, 'misclassified_text_length.png'))
    plt.close()

if __name__ == "__main__":
    print("=== Sentiment Analysis Model and Data Exploration ===")

    # Load and explore data
    df = load_and_explore_data()

    # Analyze existing model and get misclassifications
    model_results, misclassified = analyze_existing_model(df)

    # Analyze misclassified samples
    analyze_misclassified_samples(misclassified)

    print("\n=== Exploration Complete ===")
    print(f"All results saved in: {task1_explore_path}")
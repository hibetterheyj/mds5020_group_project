import re
import json
import os
from typing import List, Dict, Any, Tuple

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline

# Preprocessing function for sentiment analysis
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

# Preprocessing function for topic classification
def preprocess_text_topic(text: str) -> str:
    """Preprocess Chinese text: remove punctuation and tokenize using jieba."""
    # Clean and remove punctuation (keep Chinese, numbers, and letters)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text))

    # Tokenize Chinese text using jieba (precise mode)
    words = jieba.cut(text)

    # Filter out single-character words
    filtered_words = [word for word in words if len(word.strip()) > 1]

    return ' '.join(filtered_words)

# Cross-validation for sentiment analysis model
def cross_validation_sentiment() -> Dict[str, Any]:
    # Load sentiment analysis data
    data_path = '../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    df = pd.read_excel(data_path)

    # Apply text preprocessing to the news_title column
    df['processed_text'] = df['news_title'].apply(preprocess_text_sentiment)

    # Prepare features (X) and labels (y)
    X = df['processed_text']
    y = df['sentiment']

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    # Initialize logistic regression model with balanced class weights
    model = LogisticRegression(random_state=42, class_weight='balanced')

    # Define weighted F1 scorer for cross-validation
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Perform 5-fold cross-validation with stratified sampling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring=f1_scorer)

    # Calculate mean and standard deviation of F1 scores
    mean_f1 = cv_scores.mean()
    std_f1 = cv_scores.std()

    # Print cross-validation results
    print("=== Sentiment Analysis Model 5-fold Cross Validation ===")
    print(f"F1-Scores per fold: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean Weighted F1-Score: {mean_f1:.4f}")
    print(f"Standard Deviation: {std_f1:.4f}")

    # Return all results as a dictionary
    return {
        "f1_scores_per_fold": cv_scores.tolist(),
        "mean_weighted_f1_score": mean_f1,
        "standard_deviation": std_f1
    }

# Cross-validation for topic classification model
def cross_validation_topic() -> Dict[str, Any]:
    # Load topic classification data
    data_path = '../data/Subtask2-topic_classification/training_news-topic.xlsx'
    df = pd.read_excel(data_path)

    # Apply text preprocessing to the news_title column
    df['processed_text'] = df['news_title'].fillna('').apply(preprocess_text_topic)

    # Prepare features (X) and labels (y)
    X = df['processed_text']
    y = df['topic']

    # Build model pipeline with TF-IDF and SVM
    text_clf_svm = Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            min_df=5,
            norm='l2',
            encoding='utf-8',
            ngram_range=(1, 2)
        )),
        ('clf', SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=42
        )),
    ])

    # Define weighted F1 scorer for cross-validation (multi-class)
    f1_scorer = make_scorer(f1_score, average='weighted', pos_label=None)

    # Perform 5-fold cross-validation with stratified sampling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(text_clf_svm, X, y, cv=skf, scoring=f1_scorer, n_jobs=-1)

    # Calculate mean and standard deviation of F1 scores
    mean_f1 = cv_scores.mean()
    std_f1 = cv_scores.std()

    # Print cross-validation results
    print("\n=== Topic Classification Model 5-fold Cross Validation ===")
    print(f"F1-Scores per fold: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean Weighted F1-Score: {mean_f1:.4f}")
    print(f"Standard Deviation: {std_f1:.4f}")

    # Return all results as a dictionary
    return {
        "f1_scores_per_fold": cv_scores.tolist(),
        "mean_weighted_f1_score": mean_f1,
        "standard_deviation": std_f1
    }

if __name__ == "__main__":
    print("=== Model Cross Validation Results ===")

    # Run sentiment analysis cross-validation
    print("\nRunning sentiment analysis model cross-validation...")
    sentiment_results = cross_validation_sentiment()

    # Run topic classification cross-validation
    print("\nRunning topic classification model cross-validation...")
    topic_results = cross_validation_topic()

    # Print final summary
    print("\n=== Final Results Summary ===")
    print(f"Sentiment Analysis Model Mean Weighted F1-Score: {sentiment_results['mean_weighted_f1_score']:.4f}")
    print(f"Topic Classification Model Mean Weighted F1-Score: {topic_results['mean_weighted_f1_score']:.4f}")

    # Prepare results for JSON output
    all_results = {
        "sentiment_analysis": sentiment_results,
        "topic_classification": topic_results,
        "cross_validation_configuration": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # Save results to JSON file
    output_file = 'test_cross_validation.json'
    output_path = os.path.join(os.getcwd(), output_file)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4, default=str)

    print(f"\nResults saved to: {output_path}")

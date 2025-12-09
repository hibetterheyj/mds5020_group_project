import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer


def preprocess_text(text):
    """Preprocess text by lowercasing, removing non-alphabetic characters, and removing stopwords."""
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


def main():
    # Load the dataset
    data_path = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/Subtask1-sentiment_analysis/training_news-sentiment.csv'
    df = pd.read_csv(data_path)
    
    print("Data loaded successfully. Shape:", df.shape)
    print("Sentiment distribution:", df['sentiment'].value_counts().to_dict())
    
    # Apply text preprocessing to the 'news_title' column
    df['processed_text'] = df['news_title'].apply(preprocess_text)
    print("Text preprocessing completed.")
    
    # Convert labels from -1/1 to 0/1
    df['sentiment'] = df['sentiment'].map({-1: 0, 1: 1})
    print("Labels transformed to 0/1 format.")
    
    # Prepare features (X) and labels (y)
    X = df['processed_text']
    y = df['sentiment']
    
    # Feature extraction using TF-IDF with optimal parameters from tuning
    vectorizer = TfidfVectorizer(
        max_features=3000,  # Optimal from tuning
        ngram_range=(1, 1)  # Optimal from tuning
    )
    X_tfidf = vectorizer.fit_transform(X)
    print("TF-IDF feature extraction completed. Shape:", X_tfidf.shape)
    
    # Initialize logistic regression model with optimal parameters from tuning
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',  # Optimal from tuning
        C=1.0,  # Optimal from tuning
        penalty='l2',  # Optimal from tuning
        solver='liblinear'  # Optimal from tuning
    )
    
    # Define weighted F1 scorer for cross-validation
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    # Perform 5-fold cross-validation and calculate mean weighted F1-score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring=f1_scorer)
    mean_f1 = cv_scores.mean()
    print(f"5-Fold Cross Validation Mean Weighted F1-Score: {mean_f1:.4f}")
    print(f"Cross Validation Scores: {cv_scores}")
    
    # Train the model on the entire dataset
    model.fit(X_tfidf, y)
    print("Model training completed.")
    
    # Save the trained model and vectorizer for integration with Flask app
    save_path = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/Project2_update/subtask_1_model/subtask_1_model/model.joblib'
    joblib.dump({'model': model, 'vectorizer': vectorizer}, save_path)
    print(f"Model and vectorizer saved to: {save_path}")


if __name__ == "__main__":
    main()

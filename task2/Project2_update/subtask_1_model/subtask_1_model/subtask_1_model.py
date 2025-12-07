import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import joblib

# # Load the dataset
# df = pd.read_excel('dataset.xlsx')


# Preprocess text data: remove stopwords, digits, and punctuation
def preprocess_text(text):
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


# # Apply text preprocessing to the 'news_title' column
# df['processed_text'] = df['news_title'].apply(preprocess_text)

# # Convert labels from string to integer if needed
# df['sentiment'] = df['sentiment'].astype(int)

# # Prepare features (X) and labels (y)
# X = df['processed_text']
# y = df['sentiment']

# # Feature extraction using TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000)
# X_tfidf = vectorizer.fit_transform(X)

# # Initialize logistic regression model
# model = LogisticRegression(random_state=42, class_weight='balanced')

# # Define weighted F1 scorer for cross-validation
# f1_scorer = make_scorer(f1_score, average='weighted')

# # Perform 5-fold cross-validation and calculate mean weighted F1-score
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring=f1_scorer)
# mean_f1 = cv_scores.mean()
# print(f"5-Fold Cross Validation Mean Weighted F1-Score: {mean_f1:.4f}")

# # Train the model on the entire dataset
# model.fit(X_tfidf, y)

# # Save the trained model and vectorizer for future use
# joblib.dump({'model': model, 'vectorizer': vectorizer}, 'model.joblib')


# Function to predict sentiment and probability for new titles
def predict_sentiment(model_loaded, vectorizer_loaded, title):

    # Preprocess the input title
    processed_title = preprocess_text(title)

    # Transform the title using the saved vectorizer
    title_tfidf = vectorizer_loaded.transform([processed_title])

    # Predict sentiment class
    prediction = model_loaded.predict(title_tfidf)[0]

    # Get probability for each class
    probabilities = model_loaded.predict_proba(title_tfidf)[0]

    # Create probability mapping for both classes
    prob_dict = dict(zip(model_loaded.classes_, probabilities))

    return prediction, prob_dict


# # Example usage of the prediction function
# example_headline = "Excellent Performance beyond Expectations"
# sentiment, probability = predict_sentiment(example_headline)
# print(f"Predicted sentiment: {sentiment}")
# print(f"Probability distribution: {probability}")
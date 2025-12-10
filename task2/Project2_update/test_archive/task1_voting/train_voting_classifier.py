import os
import json
import joblib
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# Local imports
from utils import (
    load_and_prepare_data,
    create_pipeline,
    TextPreprocessorTransformer,
    get_text_column
)

def main():
    # Configuration
    DATA_PATH = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'  # Updated path
    CV = 5  # Cross-validation folds
    N_JOBS = 2  # Number of parallel jobs
    MODEL_DIR = './model'  # Updated path
    RESULTS_FILE = './model/voting_classifier_results.json'  # Updated path

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data...")
    df, X, y = load_and_prepare_data(DATA_PATH)

    # Define the best classifiers with their optimal parameters
    # Based on the JSON configuration provided
    classifiers = [
        # 1. ExtraTreesClassifier without handcrafted features
        {
            'name': 'ExtraTreesClassifier_NoHandcrafted',
            'classifier': ExtraTreesClassifier(
                random_state=42,
                n_estimators=200,
                min_samples_split=5,
                max_features='sqrt',
                max_depth=None
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 1)
            }
        },
        # 2. LogisticRegression without handcrafted features
        {
            'name': 'LogisticRegression_NoHandcrafted',
            'classifier': LogisticRegression(
                random_state=42,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced',
                C=1.0
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 3. SGDClassifier without handcrafted features
        {
            'name': 'SGDClassifier_NoHandcrafted',
            'classifier': CalibratedClassifierCV(
                estimator=SGDClassifier(
                    random_state=42,
                    penalty='l1',
                    loss='hinge',
                    class_weight=None,
                    alpha=0.0001
                ),
                method='sigmoid',
                cv=5
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 7000,
                'ngram_range': (1, 3)
            }
        },
        # 4. SVC_Sigmoid without handcrafted features
        {
            'name': 'SVC_Sigmoid_NoHandcrafted',
            'classifier': SVC(
                random_state=42,
                kernel='sigmoid',
                gamma=0.01,
                coef0=1.0,
                class_weight='balanced',
                C=100.0,
                probability=True
            ),
            'use_handcrafted': False,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 5. ExtraTreesClassifier with handcrafted features
        {
            'name': 'ExtraTreesClassifier_WithHandcrafted',
            'classifier': ExtraTreesClassifier(
                random_state=42,
                n_estimators=100,
                min_samples_split=2,
                max_features='sqrt',
                max_depth=None
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 1)
            }
        },
        # 6. MLPClassifier with handcrafted features
        {
            'name': 'MLPClassifier_WithHandcrafted',
            'classifier': MLPClassifier(
                random_state=42,
                max_iter=200,
                n_iter_no_change=10,
                early_stopping=True,
                hidden_layer_sizes=(50,),
                alpha=0.001,
                activation='relu',
                solver='adam',
                learning_rate_init=0.005
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 3000,
                'ngram_range': (1, 2)
            }
        },
        # 7. SVC_Poly with handcrafted features
        {
            'name': 'SVC_Poly_WithHandcrafted',
            'classifier': SVC(
                random_state=42,
                kernel='poly',
                gamma=0.001,
                degree=3,
                coef0=2.0,
                class_weight='balanced',
                C=100.0,
                probability=True
            ),
            'use_handcrafted': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        }
    ]

    # Create estimators for VotingClassifier
    print("Creating classifier pipelines...")
    estimators = []
    for clf_config in classifiers:
        # Create text pipeline with best TF-IDF parameters
        text_pipeline = Pipeline([
            ('selector', FunctionTransformer(get_text_column, validate=False)),
            ('preprocessor', TextPreprocessorTransformer(
                use_stemming=False,
                use_lemmatization=True
            )),
            ('tfidf', TfidfVectorizer(**clf_config['tfidf_params']))
        ])

        # Create complete pipeline with or without handcrafted features
        pipeline = create_pipeline(
            clf_config['classifier'],
            use_handcrafted=clf_config['use_handcrafted'],
            text_pipeline=text_pipeline
        )

        estimators.append((clf_config['name'], pipeline))

    # Create Voting Classifier with hard voting
    print("Creating VotingClassifier with hard voting...")
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='hard',
        n_jobs=N_JOBS
    )

    # Perform cross-validation
    print("Performing 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the classifier
        voting_clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = voting_clf.predict(X_val)
        fold_score = f1_score(y_val, y_pred, average='weighted')
        cv_scores.append(fold_score)
        print(f"  Fold {fold_idx+1}/{CV}: F1 score = {fold_score:.4f}")

    # Calculate average score
    average_score = sum(cv_scores) / len(cv_scores)
    print(f"\nAverage F1 score: {average_score:.4f}")

    # Train the final model on all data
    print("Training final model on all data...")
    voting_clf.fit(df, y)

    # Save the model
    model_path = os.path.join(MODEL_DIR, 'voting_classifier.pkl')
    joblib.dump(voting_clf, model_path)
    print(f"Model saved to {model_path}")

    # Save results to JSON
    results = {
        'classifier': 'VotingClassifier_Hard',
        'base_estimators': [name for name, _ in estimators],
        'num_estimators': len(estimators),
        'cv_folds': CV,
        'cv_scores': cv_scores,
        'average_score': float(average_score),
        'n_jobs': N_JOBS,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_FILE}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
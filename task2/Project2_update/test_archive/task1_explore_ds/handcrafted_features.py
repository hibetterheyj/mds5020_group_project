import pandas as pd
import numpy as np
import re

def create_sentiment_features(df):
    """Create handcrafted features based on EDA findings
    
    Args:
        df: DataFrame with 'news_title' column
        
    Returns:
        DataFrame with handcrafted features
    """
    
    # Positive and negative words based on EDA findings
    positive_words = [
        'profit', 'rise', 'increase', 'growth', 'higher', 'gain',
        'win', 'success', 'improve', 'boost', 'surge', 'soar',
        'strong', 'positive', 'beat', 'exceed'
    ]
    
    negative_words = [
        'loss', 'fall', 'decrease', 'drop', 'cut', 'lower',
        'decline', 'weak', 'negative', 'miss', 'fail', 'warn',
        'drop', 'hit', 'lose', 'fell', 'decrease', 'downgrade'
    ]
    
    # Financial specific words
    financial_words = ['eur', 'net', 'sale', 'share', 'price', 'dividend', 'million']
    
    # Create features
    features = []
    
    for text in df['news_title']:
        text_lower = str(text).lower()
        
        # 1. Positive and negative word counts
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # 2. Sentiment word ratios
        total_sentiment_words = pos_count + neg_count
        pos_ratio = pos_count / (total_sentiment_words + 1e-10)
        neg_ratio = neg_count / (total_sentiment_words + 1e-10)
        
        # 3. Net sentiment score
        net_sentiment = pos_count - neg_count
        
        # 4. Strong sentiment indicators
        has_strong_positive = any(word in text_lower for word in ['soar', 'surge', 'beat'])
        has_strong_negative = any(word in text_lower for word in ['plunge', 'crash', 'collapse', 'slump'])
        
        # 5. Financial word density
        financial_count = sum(1 for word in financial_words if word in text_lower)
        word_count = len(text_lower.split())
        financial_density = financial_count / (word_count + 1e-10)
        
        features.append([
            pos_count, neg_count, pos_ratio, neg_ratio, net_sentiment,
            int(has_strong_positive), int(has_strong_negative),
            financial_count, financial_density
        ])
    
    feature_names = [
        'pos_word_count', 'neg_word_count', 'pos_ratio', 'neg_ratio', 'net_sentiment',
        'has_strong_positive', 'has_strong_negative',
        'financial_word_count', 'financial_density'
    ]
    
    return pd.DataFrame(features, columns=feature_names)
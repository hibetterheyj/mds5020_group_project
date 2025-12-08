import pandas as pd
import numpy as np
import re

def create_sentiment_features(df):
    """Create handcrafted features based on EDA findings and financial text analysis

    Args:
        df: DataFrame with 'news_title' column

    Returns:
        DataFrame with handcrafted features
    """

    # Positive and negative words based on EDA findings and financial domain knowledge
    positive_words = [
        'profit', 'rise', 'increase', 'growth', 'higher', 'gain',
        'win', 'success', 'improve', 'boost', 'surge', 'soar',
        'strong', 'positive', 'beat', 'exceed', 'expand', 'advance',
        'climb', 'jump', 'leap', 'outperform', 'upgrade', 'strengthen'
    ]

    negative_words = [
        'loss', 'fall', 'decrease', 'drop', 'cut', 'lower',
        'decline', 'weak', 'negative', 'miss', 'fail', 'warn',
        'lose', 'fell', 'downgrade', 'plunge', 'crash', 'collapse',
        'slump', 'dip', 'slide', 'tumble', 'underperform', 'weaken',
        'reduce', 'shrink', 'contract', 'downturn', 'downgrade'
    ]

    # Financial specific words (expanded)
    financial_words = ['eur', 'net', 'sale', 'share', 'price', 'dividend', 'million',
                      'billion', 'quarter', 'year', 'month', 'revenue', 'income',
                      'earnings', 'profitability', 'margin', 'asset', 'liability',
                      'equity', 'debt', 'cash', 'flow', 'market', 'value', 'cap',
                      'index', 'benchmark', 'forecast', 'guidance', 'target',
                      'rating', 'analyst', 'investor', 'holder', 'trading']

    # Negation words
    negation_words = ['not', 'no', 'never', 'none', 'nor', 'neither', 'cannot',
                     'n\'t', 'without', 'hardly', 'scarcely', 'barely']

    # Create features
    features = []

    for text in df['news_title']:
        text_lower = str(text).lower()
        words = text_lower.split()
        word_count = len(words)
        char_count = len(text_lower)

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
        has_strong_positive = any(word in text_lower for word in ['soar', 'surge', 'beat', 'outperform'])
        has_strong_negative = any(word in text_lower for word in ['plunge', 'crash', 'collapse', 'slump', 'underperform'])

        # 5. Financial word density and event indicators
        financial_count = sum(1 for word in financial_words if word in text_lower)
        word_count = len(text_lower.split())
        financial_density = financial_count / (word_count + 1e-10)

        # 6. Financial event indicators
        has_earnings = 'earning' in text_lower or 'profit' in text_lower or 'revenue' in text_lower
        has_dividend = 'dividend' in text_lower or 'div' in text_lower
        has_forecast = 'forecast' in text_lower or 'guidance' in text_lower or 'target' in text_lower
        has_rating = 'rating' in text_lower or 'upgrade' in text_lower or 'downgrade' in text_lower
        # 6. Title length features
        avg_word_length = char_count / (word_count + 1e-10) if word_count > 0 else 0
        is_short_title = word_count < 10
        is_long_title = word_count > 20

        # 7. Number and percentage features
        has_number = bool(re.search(r'\b\d+\b', text_lower))
        has_percentage = bool(re.search(r'\b\d+%\b|\b\d+\s*percent\b', text_lower))
        has_money = bool(re.search(r'\$\d+|\d+\s*eur|\d+\s*usd|\d+\s*gbp', text_lower))

        # 8. Sentiment word position features
        first_three_words = set(words[:3] if len(words) >= 3 else words)
        last_three_words = set(words[-3:] if len(words) >= 3 else words)

        pos_in_first_three = sum(1 for word in positive_words if word in first_three_words)
        neg_in_first_three = sum(1 for word in negative_words if word in first_three_words)
        pos_in_last_three = sum(1 for word in positive_words if word in last_three_words)
        neg_in_last_three = sum(1 for word in negative_words if word in last_three_words)

        # 9. Negation handling
        has_negation = any(word in words for word in negation_words)
        negated_pos_count = 0
        negated_neg_count = 0

        for i, word in enumerate(words):
            if word in negation_words and i < len(words) - 1:
                next_word = words[i+1]
                if next_word in positive_words:
                    negated_pos_count += 1
                elif next_word in negative_words:
                    negated_neg_count += 1

        # 10. Financial event indicators
        has_earnings = 'earning' in text_lower or 'profit' in text_lower or 'revenue' in text_lower
        has_dividend = 'dividend' in text_lower or 'div' in text_lower
        has_forecast = 'forecast' in text_lower or 'guidance' in text_lower or 'target' in text_lower
        has_rating = 'rating' in text_lower or 'upgrade' in text_lower or 'downgrade' in text_lower

        features.append([
            # Basic sentiment features
            pos_count, neg_count, pos_ratio, neg_ratio, net_sentiment,
            int(has_strong_positive), int(has_strong_negative),

            # Financial features
            financial_count, financial_density,
            int(has_earnings), int(has_dividend), int(has_forecast), int(has_rating),

            # Title length features
            word_count, char_count, avg_word_length, int(is_short_title), int(is_long_title),

            # Number and percentage features
            int(has_number), int(has_percentage), int(has_money),

            # Sentiment word position features
            pos_in_first_three, neg_in_first_three, pos_in_last_three, neg_in_last_three,

            # Negation features
            int(has_negation), negated_pos_count, negated_neg_count
        ])

    feature_names = [
        # Basic sentiment features
        'pos_word_count', 'neg_word_count', 'pos_ratio', 'neg_ratio', 'net_sentiment',
        'has_strong_positive', 'has_strong_negative',

        # Financial features
        'financial_word_count', 'financial_density',
        'has_earnings', 'has_dividend', 'has_forecast', 'has_rating',

        # Title length features
        'word_count', 'char_count', 'avg_word_length', 'is_short_title', 'is_long_title',

        # Number and percentage features
        'has_number', 'has_percentage', 'has_money',

        # Sentiment word position features
        'pos_in_first_three', 'neg_in_first_three', 'pos_in_last_three', 'neg_in_last_three',

        # Negation features
        'has_negation', 'negated_pos_count', 'negated_neg_count'
    ]

    return pd.DataFrame(features, columns=feature_names)
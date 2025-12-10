import pandas as pd
import numpy as np
import re
import os
import warnings
from typing import Dict, List, Any, Tuple, Optional, Callable

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.pipeline import Pipeline, FeatureUnion

# Import NLTK modules with lazy loading
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tokenize import word_tokenize
    
    # Check if NLTK data is available
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        
except ImportError:
    # If NLTK is not available, install it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tokenize import word_tokenize


class EnhancedTextPreprocessor:
    """增强的文本预处理器 - 修复数值稳定性问题并优化金融文本处理"""

    def __init__(self, use_stemming=True, use_lemmatization=True, preserve_negation=True, expand_ngrams=False):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.preserve_negation = preserve_negation
        self.expand_ngrams = expand_ngrams

        # 初始化NLTK工具
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if use_stemming else None

        # 创建停用词列表，保留否定词和重要情感词
        self.stop_words = set(stopwords.words('english'))
        if preserve_negation:
            # 移除否定词
            negation_words = {'not', 'no', 'never', 'none', 'nor', 'neither', 'cannot', "n't"}
            self.stop_words = self.stop_words - negation_words

            # 保留重要的情感词和强化词
            sentiment_words = {
                'very', 'too', 'so', 'more', 'less', 'most', 'least',
                'much', 'only', 'just', 'extremely', 'absolutely', 'highly',
                'greatly', 'significantly', 'substantially', 'considerably',
                'moderately', 'slightly'
            }
            self.stop_words = self.stop_words - sentiment_words

        # 扩展的停用词列表，添加金融领域常见停用词
        financial_stopwords = {
            'said', 'also', 'one', 'two', 'three', 'first', 'second', 'third',
            'year', 'years', 'percent', 'pct', 'update', 'according', 'says',
            'including', 'us', 'vs', 'via', 'inc', 'ltd', 'corp', 'co', 'group', 'plc'
        }
        self.stop_words = self.stop_words.union(financial_stopwords)

        # 金融缩写及其扩展
        self.financial_abbreviations = {
            'ebitda': 'earnings before interest taxes depreciation and amortization',
            'p/e': 'price to earnings',
            'eps': 'earnings per share',
            'roi': 'return on investment',
            'roa': 'return on assets',
            'roe': 'return on equity',
            'gross margin': 'gross margin',
            'net margin': 'net margin',
            'cagr': 'compound annual growth rate',
            'yoy': 'year over year',
            'qoq': 'quarter over quarter',
            'cap': 'capitalization',
            'guidance': 'guidance',
            'forecast': 'forecast'
        }

    def preprocess(self, text, max_words=100):
        """预处理单个文本 - 优化版本"""
        # 转换为字符串
        text = str(text)
        if not text.strip():
            return ""

        # 转换为小写
        text = text.lower()

        # 处理特殊表达
        text = self._handle_special_expressions(text)

        # 处理金融术语和缩写
        text = self._handle_financial_terms(text)

        # 使用NLTK的word_tokenize进行分词
        tokens = word_tokenize(text)

        # 限制最大单词数
        if len(tokens) > max_words:
            tokens = tokens[:max_words]

        # 处理单词
        processed_words = []
        skip_next = False
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            if token.isalpha():
                if token not in self.stop_words:
                    # 应用词形还原
                    if self.lemmatizer:
                        token = self.lemmatizer.lemmatize(token)
                        token = self.lemmatizer.lemmatize(token, pos='v')  # 动词
                        token = self.lemmatizer.lemmatize(token, pos='a')  # 形容词
                    # 应用词干提取
                    if self.stemmer:
                        token = self.stemmer.stem(token)
                    processed_words.append(token)
            elif re.match(r'\b\d+(?:\.\d+)?\b', token):
                # 保留数字（包括小数）
                processed_words.append(token)
            elif re.match(r'[\$%]', token):
                # 保留金融符号
                processed_words.append(token)
            elif self.preserve_negation and token in {
                'not', 'no', 'never', 'none', 'nor', 'neither', 'cannot', "n't",
                'without', 'hardly', 'scarcely', 'barely'
            } and i < len(tokens) - 1:
                # 处理否定词，与下一个词组合
                next_token = tokens[i + 1]
                if next_token.isalpha():
                    combined = f"{next_token}_negated"
                    processed_words.append(combined)
                    skip_next = True
                else:
                    processed_words.append(token)

        return ' '.join(processed_words)

    def _handle_special_expressions(self, text):
        """处理特殊表达，包括否定、金融符号和数字"""
        # 处理否定
        if self.preserve_negation:
            # 将否定词与下一个词组合
            text = re.sub(
                r'\b(?:not|no|never|none|nor|neither|cannot|n\'t)\s+(\w+)\b',
                r'\1_negated', text
            )

        # 处理金融符号和数字
        text = re.sub(r'\$\s*(\d+(?:\.\d+)?)', r'\1 dollars', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        text = re.sub(r'\b(\d+)\s*million\b', r'\1000000', text)
        text = re.sub(r'\b(\d+)\s*billion\b', r'\1000000000', text)

        # 处理缩写
        contraction_mapping = {
            "n't": ' not',
            "'re": ' are',
            "'s": ' is',
            "'d": ' would',
            "'ll": ' will',
            "'t": ' not',
            "'ve": ' have',
            "'m": ' am'
        }
        for contraction, expansion in contraction_mapping.items():
            text = re.sub(re.escape(contraction), expansion, text)

        # 清理特殊字符，保留字母、数字和基本标点
        text = re.sub(r'[^\w\s\.\,\!\?\-\+\%\$\&]', ' ', text)

        # 移除额外空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _handle_financial_terms(self, text):
        """处理金融术语和缩写"""
        # 扩展金融缩写
        for abbreviation, expansion in self.financial_abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbreviation) + r'\b', expansion, text)

        # 对金融短语进行分组，保持短语完整性
        financial_phrases = [
            'stock price', 'earnings report', 'dividend yield', 'market capitalization',
            'gross domestic product', 'interest rate', 'exchange rate', 'inflation rate',
            'unemployment rate', 'consumer price index', 'producer price index'
        ]

        for phrase in financial_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in text:
                # 用下划线替换空格，保持短语在一起
                text = text.replace(phrase_lower, phrase_lower.replace(' ', '_'))

        return text

    def preprocess_batch(self, texts):
        """批量预处理文本"""
        return [self.preprocess(text) for text in texts]


class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Transformer for text preprocessing using the EnhancedTextPreprocessor"""

    def __init__(self, use_stemming: bool = False, use_lemmatization: bool = True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.preprocessor = EnhancedTextPreprocessor(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocessor.preprocess(text) for text in X]


class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert sparse matrices to dense numpy arrays"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if hasattr(X, 'toarray') else X


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


def load_and_prepare_data(file_path: str, label_mapping: Dict[int, int] = {-1: 0, 1: 1}) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for training

    Args:
        file_path: Path to the Excel data file
        label_mapping: Mapping from original labels to model labels

    Returns:
        Tuple of (processed dataframe with handcrafted features, processed text series, labels)
    """
    df = pd.read_excel(file_path)

    # Create handcrafted features
    handcrafted_features = create_sentiment_features(df)
    df = pd.concat([df, handcrafted_features], axis=1)

    # Process labels
    df['sentiment'] = df['sentiment'].map(label_mapping)

    # Separate features and labels
    X = df['news_title']
    y = df['sentiment']

    return df, X, y


def get_text_column(df: pd.DataFrame) -> pd.Series:
    """Function to extract text column from dataframe"""
    if isinstance(df, pd.DataFrame) and 'news_title' in df.columns:
        return df['news_title']
    else:
        return df


def get_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Function to extract handcrafted features from dataframe"""
    # List of handcrafted feature columns
    feature_columns = [
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

    return df[feature_columns]


def create_text_pipeline(use_stemming: bool = False, use_lemmatization: bool = True) -> Pipeline:
    """Create a text-only pipeline with preprocessing and TF-IDF vectorization"""
    return Pipeline([
        ('selector', FunctionTransformer(get_text_column, validate=False)),
        ('preprocessor', TextPreprocessorTransformer(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )),
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            use_idf=True
        ))
    ])


def create_feature_union(text_pipeline: Pipeline, classifier: BaseEstimator) -> FeatureUnion:
    """Create a feature union with text pipeline and handcrafted features

    Args:
        text_pipeline: Text processing pipeline
        classifier: Classifier to determine scaling approach (Naive Bayes needs non-negative scaling)
    """

    # Use MinMaxScaler for Naive Bayes (requires non-negative inputs), otherwise StandardScaler
    is_naive_bayes = isinstance(classifier, (MultinomialNB, BernoulliNB, CategoricalNB))
    scaler = MinMaxScaler() if is_naive_bayes else StandardScaler()

    return FeatureUnion([
        ('text', text_pipeline),
        ('handcrafted', Pipeline([
            ('selector', FunctionTransformer(get_handcrafted_features, validate=False)),
            ('scaler', scaler)
        ]))
    ])


def create_pipeline(
    classifier: BaseEstimator,
    use_handcrafted: bool = True,
    text_pipeline: Optional[Pipeline] = None
) -> Pipeline:
    """Create a classifier pipeline with optional handcrafted features

    Args:
        classifier: The classifier to use
        use_handcrafted: Whether to include handcrafted features
        text_pipeline: Optional pre-configured text pipeline

    Returns:
        Complete sklearn Pipeline
    """
    if text_pipeline is None:
        text_pipeline = create_text_pipeline()

    if use_handcrafted:
        # Combined pipeline with both text and handcrafted features
        # Pass classifier to determine scaling approach for handcrafted features
        feature_union = create_feature_union(text_pipeline, classifier)

        # Check if classifier needs dense data
        needs_dense = isinstance(classifier, CategoricalNB)

        pipeline_steps = [
            ('features', feature_union)
        ]

        # Add sparse to dense transformer if needed
        if needs_dense:
            pipeline_steps.append(('to_dense', SparseToDenseTransformer()))

        pipeline_steps.append(('classifier', classifier))

        pipeline = Pipeline(pipeline_steps)
    else:
        # Text-only pipeline
        needs_dense = isinstance(classifier, CategoricalNB)

        pipeline_steps = [
            ('text', text_pipeline)
        ]

        # Add sparse to dense transformer if needed
        if needs_dense:
            pipeline_steps.append(('to_dense', SparseToDenseTransformer()))

        pipeline_steps.append(('classifier', classifier))

        pipeline = Pipeline(pipeline_steps)

    return pipeline
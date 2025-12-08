# [file name]: text_preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('tokenizers/punkt')
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')


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
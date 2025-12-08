# [file name]: text_preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')


class EnhancedTextPreprocessor:
    """增强的文本预处理器 - 修复数值稳定性问题"""

    def __init__(self, use_stemming=True, use_lemmatization=True):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if use_stemming else None

        # 扩展的停用词列表，但保留否定词和重要情感词
        self.extended_stopwords = self.stopwords.union({
            'said', 'also', 'one', 'two', 'three', 'first', 'second', 'third',
            'year', 'years', 'percent', 'pct', 'update', 'according', 'says',
            'including', 'us', 'vs', 'via', 'inc', 'ltd', 'corp', 'co', 'group', 'plc'
        })

        # 不从停用词中移除否定词和重要情感词
        important_words = {'not', 'no', 'never', 'none', 'nor', 'neither', 'cannot',
                          'very', 'too', 'so', 'more', 'less', 'most', 'least',
                          'good', 'bad', 'better', 'worse', 'best', 'worst'}
        self.extended_stopwords = self.extended_stopwords - important_words

    def preprocess(self, text, max_words=100):
        """预处理单个文本 - 修复版本"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # 转换为小写
        text = text.lower()

        # 处理金融和商业相关的特殊表达 - 更保守的方式
        text = re.sub(r'\b(\d+)\s*percent\b', r'\1percent', text)
        text = re.sub(r'\b(\d+)\s*pct\b', r'\1percent', text)
        text = re.sub(r'\$(\d+(?:\.\d+)?)\s*million\b', r'dollar\1m', text, flags=re.IGNORECASE)
        text = re.sub(r'\$(\d+(?:\.\d+)?)\s*billion\b', r'dollar\1b', text, flags=re.IGNORECASE)

        # 更保守的字符清理 - 保留基本标点
        text = re.sub(r'[^\w\s\.\,\!\?\-\+\%\$\&]', ' ', text)

        # 分割单词
        words = text.split()

        # 限制最大单词数以防止过长的文本
        if len(words) > max_words:
            words = words[:max_words]

        # 处理单词
        processed_words = []
        for word in words:
            # 跳过太短的单词（但保留重要的小词）
            if len(word) < 2 and word not in {'no', 'up', 'in', 'on', 'at', 'to', 'by'}:
                continue

            # 跳过停用词
            if word in self.extended_stopwords:
                continue

            # 应用词形还原
            if self.lemmatizer:
                word = self.lemmatizer.lemmatize(word, pos='n')  # 名词
                word = self.lemmatizer.lemmatize(word, pos='v')  # 动词
                word = self.lemmatizer.lemmatize(word, pos='a')  # 形容词

            # 应用词干提取
            if self.stemmer:
                word = self.stemmer.stem(word)

            processed_words.append(word)

        return ' '.join(processed_words)
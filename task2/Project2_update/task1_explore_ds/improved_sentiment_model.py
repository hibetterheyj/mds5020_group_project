# [file name]: improved_sentiment_model_fixed.py
import pandas as pd
import numpy as np
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

# 尝试导入XGBoost和LightGBM，但设置为可选
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost 不可用")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 不可用")

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

def load_and_prepare_data(file_path, label_mapping={-1: 0, 1: 1}):
    """加载和准备数据 - 修复标签问题"""
    df = pd.read_excel(file_path)

    # 初始化预处理器 - 更保守的设置
    preprocessor = EnhancedTextPreprocessor(
        use_stemming=False,  # 词干提取可能导致信息丢失
        use_lemmatization=True
    )

    # 预处理文本
    df['processed_text'] = df['news_title'].apply(preprocessor.preprocess)

    # 确保标签为整数并映射到[0, 1]
    df['sentiment_original'] = df['sentiment'].astype(int)

    if label_mapping:
        df['sentiment'] = df['sentiment_original'].map(label_mapping)
    else:
        df['sentiment'] = df['sentiment_original']

    print(f"数据集大小: {len(df)}")
    print(f"原始类别分布:\n{df['sentiment_original'].value_counts()}")
    print(f"映射后类别分布:\n{df['sentiment'].value_counts()}")

    return df

def benchmark_models(X, y, cv=5, use_smote=False):
    """基准测试多个模型 - 修复版本"""
    print("=" * 60)
    print("开始模型基准测试")
    print("=" * 60)

    # 定义要测试的模型
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            solver='saga',  # 更稳定的solver
            C=0.5,
            penalty='l2'
        ),
        'Linear SVC': LinearSVC(
            random_state=42,
            class_weight='balanced',
            max_iter=5000,
            C=0.1
        ),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=100,
            max_depth=20,
            min_samples_split=10
        ),
    }

    # 可选添加XGBoost和LightGBM
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            n_jobs=-1
        )

    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            n_jobs=-1
        )

    # 定义TF-IDF向量化器 - 更保守的参数
    vectorizer = TfidfVectorizer(
        max_features=3000,  # 减少特征数量以提高稳定性
        ngram_range=(1, 2),
        min_df=3,  # 增加最小文档频率
        max_df=0.9,
        sublinear_tf=True,
        norm='l2'  # 使用L2规范化
    )

    # 转换为TF-IDF特征
    X_tfidf = vectorizer.fit_transform(X)

    # 检查特征矩阵
    print(f"特征矩阵形状: {X_tfidf.shape}")
    print(f"特征矩阵密度: {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.4f}")

    # 交叉验证设置
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')

    # 存储结果
    results = {}

    for name, model in models.items():
        print(f"\n正在训练 {name}...")

        # 使用交叉验证评估
        try:
            cv_scores = cross_val_score(
                model, X_tfidf, y,
                cv=skf,
                scoring=f1_scorer,
                n_jobs=-1,
                error_score='raise'
            )
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            results[name] = {
                'mean_f1': mean_score,
                'std_f1': std_score,
                'scores': cv_scores.tolist()
            }

            print(f"  {name}:")
            print(f"    平均F1分数: {mean_score:.4f} (±{std_score:.4f})")
            print(f"    各折分数: {[f'{s:.4f}' for s in cv_scores]}")

        except Exception as e:
            print(f"  {name} 训练失败: {str(e)[:200]}...")
            results[name] = {
                'mean_f1': 0,
                'std_f1': 0,
                'scores': [],
                'error': str(e)
            }

    # 打印结果总结
    print("\n" + "=" * 60)
    print("基准测试结果总结")
    print("=" * 60)

    results_df = pd.DataFrame([
        {
            'Model': name,
            'Mean F1': res['mean_f1'],
            'Std F1': res['std_f1'],
            'Status': '成功' if 'error' not in res else '失败'
        }
        for name, res in results.items()
    ]).sort_values('Mean F1', ascending=False)

    print(results_df.to_string(index=False))

    return results, vectorizer

def optimize_best_model(X, y, model_type='logistic'):
    """优化最佳模型 - 简化版本"""
    print(f"\n" + "=" * 60)
    print(f"优化 {model_type} 模型")
    print("=" * 60)

    if model_type == 'logistic':
        # 创建管道
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=2000,
                solver='saga'
            ))
        ])

        # 简化的参数网格
        param_grid = {
            'tfidf__max_features': [2000, 3000, 4000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [2, 3, 5],
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'clf__C': [0.1, 0.5, 1.0, 5.0],
            'clf__penalty': ['l2', 'l1']
        }
    elif model_type == 'svm' and 'svm' in model_type:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(
                random_state=42,
                class_weight='balanced',
                max_iter=5000
            ))
        ])

        param_grid = {
            'tfidf__max_features': [2000, 3000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.01, 0.1, 1.0]
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 使用网格搜索（减少参数组合以加快速度）
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    print(f"正在进行网格搜索优化...")
    grid_search.fit(X, y)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def train_ensemble_model(X, y):
    """训练集成模型"""
    print("\n" + "=" * 60)
    print("训练集成模型")
    print("=" * 60)

    from sklearn.ensemble import VotingClassifier, StackingClassifier

    # 创建基学习器
    estimators = [
        ('lr', LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            C=0.5,
            solver='liblinear'
        )),
        ('svm', LinearSVC(
            random_state=42,
            class_weight='balanced',
            max_iter=5000,
            C=0.1
        )),
        ('rf', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=100,
            max_depth=15
        ))
    ]

    # 使用投票分类器
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )

    # 创建管道
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ('clf', voting_clf)
    ])

    # 交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')

    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=f1_scorer, n_jobs=-1)

    print(f"5折交叉验证F1分数:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  第{i}折: {score:.4f}")
    print(f"  平均分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 在完整数据集上训练
    pipeline.fit(X, y)

    return pipeline

def train_final_model(X, y, model_type='logistic'):
    """训练最终模型 - 修复版本"""
    print("\n" + "=" * 60)
    print(f"训练最终模型 ({model_type})")
    print("=" * 60)

    if model_type == 'logistic':
        # 使用经过验证的参数
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=2000,
                C=0.5,
                solver='saga',
                penalty='l2'
            ))
        ])
    elif model_type == 'ensemble':
        return train_ensemble_model(X, y)
    elif model_type == 'svm':
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9
            )),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(
                random_state=42,
                class_weight='balanced',
                max_iter=5000,
                C=0.1
            ))
        ])
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')

    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=f1_scorer, n_jobs=-1)

    print(f"5折交叉验证F1分数:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  第{i}折: {score:.4f}")
    print(f"  平均分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 在完整数据集上训练
    pipeline.fit(X, y)

    return pipeline

def predict_sentiment_new(model_pipeline, title, label_mapping={0: -1, 1: 1}):
    """预测新标题的情感 - 修复版本"""
    preprocessor = EnhancedTextPreprocessor(
        use_stemming=False,
        use_lemmatization=True
    )
    processed_title = preprocessor.preprocess(title)

    # 预测
    prediction = model_pipeline.predict([processed_title])[0]

    # 获取概率
    if hasattr(model_pipeline, 'predict_proba'):
        probabilities = model_pipeline.predict_proba([processed_title])[0]
        prob_dict = dict(zip(model_pipeline.classes_, probabilities))
    elif hasattr(model_pipeline.named_steps['clf'], 'predict_proba'):
        probabilities = model_pipeline.predict_proba([processed_title])[0]
        prob_dict = dict(zip(model_pipeline.named_steps['clf'].classes_, probabilities))
    else:
        # 对于SVM等没有概率的模型，使用决策函数
        decision_values = model_pipeline.decision_function([processed_title])[0]
        prob_dict = {f"class_{i}": val for i, val in enumerate(decision_values)}

    # 如果需要，映射回原始标签
    if label_mapping and prediction in label_mapping:
        original_prediction = label_mapping[prediction]
    else:
        original_prediction = prediction

    return original_prediction, prob_dict

def analyze_model_errors(model_pipeline, X, y, sample_size=20):
    """分析模型错误 - 新增功能"""
    print("\n" + "=" * 60)
    print("模型错误分析")
    print("=" * 60)

    # 预测
    y_pred = model_pipeline.predict(X)

    # 计算指标
    from sklearn.metrics import classification_report, confusion_matrix
    print("分类报告:")
    print(classification_report(y, y_pred))

    # 识别错误样本
    errors_mask = y != y_pred
    error_indices = np.where(errors_mask)[0]

    if len(error_indices) > 0:
        print(f"\n总共 {len(error_indices)} 个错误样本 ({len(error_indices)/len(y)*100:.1f}%)")

        # 随机选择一些错误样本进行分析
        np.random.seed(42)
        sample_indices = np.random.choice(
            error_indices,
            size=min(sample_size, len(error_indices)),
            replace=False
        )

        print(f"\n随机选择的 {len(sample_indices)} 个错误样本:")
        for idx in sample_indices:
            original_text = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            true_label = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            pred_label = y_pred[idx]

            print(f"\n样本 {idx}:")
            print(f"  文本: {original_text[:150]}...")
            print(f"  真实标签: {true_label}, 预测标签: {pred_label}")

def main():
    # 文件路径 - 请根据实际情况修改
    data_path = '../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

    print("开始改进情感分析模型 (修复版本)...")

    # 1. 加载和准备数据 - 映射标签到[0, 1]
    label_mapping = {-1: 0, 1: 1}
    df = load_and_prepare_data(data_path, label_mapping)
    X = df['processed_text']
    y = df['sentiment']

    # 检查预处理后的文本
    print(f"\n预处理后的文本示例:")
    for i in range(min(3, len(X))):
        print(f"  原始: {df['news_title'].iloc[i][:80]}...")
        print(f"  处理后: {X.iloc[i][:80]}...")
        print()

    # 2. 基准测试多个模型
    benchmark_results, vectorizer = benchmark_models(X, y, cv=5)

    # 3. 根据基准测试结果选择最佳模型进行优化
    best_model_name = max(benchmark_results.items(), key=lambda x: x[1]['mean_f1'])[0]
    print(f"\n最佳模型: {best_model_name}")

    # 4. 优化最佳模型
    if 'Logistic' in best_model_name:
        best_model, best_params = optimize_best_model(X, y, model_type='logistic')
    elif 'SVC' in best_model_name:
        best_model, best_params = optimize_best_model(X, y, model_type='svm')
    else:
        print(f"跳过优化，直接训练最终模型")
        best_model = None

    # 5. 训练最终模型（可以选择集成模型）
    print("\n选择并训练最终模型...")

    # 根据基准测试，逻辑回归表现最好，我们训练一个逻辑回归模型
    final_model = train_final_model(X, y, model_type='logistic')

    # 6. 分析模型错误
    analyze_model_errors(final_model, X, y, sample_size=10)

    # 7. 保存模型
    model_info = {
        'model': final_model,
        'preprocessor': EnhancedTextPreprocessor(),
        'vectorizer': final_model.named_steps['tfidf'],
        'model_type': 'logistic',
        'label_mapping': label_mapping
    }

    joblib.dump(model_info, 'improved_sentiment_model_fixed.joblib')
    print(f"\n模型已保存到: improved_sentiment_model_fixed.joblib")

    # 8. 测试预测（映射回原始标签）
    print("\n" + "=" * 60)
    print("测试预测功能")
    print("=" * 60)

    test_titles = [
        "Excellent Performance beyond Expectations",
        "Company reports significant losses in Q3",
        "Stable growth maintained throughout the year",
        "Market crashes amid global economic concerns",
        "Innovative product launch boosts company shares",
        "Profit fell by 30% due to market conditions",
        "Sales increased by 15% in the last quarter"
    ]

    for title in test_titles:
        prediction, probabilities = predict_sentiment_new(
            final_model,
            title,
            label_mapping={v: k for k, v in label_mapping.items()}
        )
        print(f"\n标题: {title}")
        print(f"预测情感: {prediction}")
        print(f"概率分布: {probabilities}")

    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)

    # 打印性能总结
    print("\n性能总结:")
    print(f"- 数据集大小: {len(df)}")
    print(f"- 类别分布: 正面({label_mapping[1]}): {sum(y == 1)}, 负面({label_mapping[-1]}): {sum(y == 0)}")
    print(f"- 最佳基准模型: {best_model_name}")
    print(f"- 最终模型: 逻辑回归")

if __name__ == "__main__":
    main()

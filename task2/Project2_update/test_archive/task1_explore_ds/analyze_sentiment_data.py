# [file name]: analyze_sentiment_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查nltk数据是否存在，不存在则下载
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    print("NLTK数据已存在，跳过下载")
except LookupError:
    print("正在下载NLTK数据...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK数据下载完成")

# 辅助函数：将numpy数据类型转换为Python原生类型
def convert_numpy_types(obj):
    """Recursively convert numpy data types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_and_analyze_data(file_path):
    """加载并分析数据集"""
    df = pd.read_excel(file_path)

    print("=" * 50)
    print("数据集基本信息")
    print("=" * 50)
    print(f"数据集形状: {df.shape}")
    print(f"\n前5行数据:")
    print(df.head())
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n数据类型:")
    print(df.dtypes)

    # 检查缺失值
    print(f"\n缺失值统计:")
    print(df.isnull().sum())

    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"\n重复行数: {duplicates}")
    if duplicates > 0:
        print("重复值示例:")
        print(df[df.duplicated(keep=False)].head())

    # 收集数据集信息用于JSON输出
    dataset_info = {
        "shape": {
            "rows": df.shape[0],
            "columns": df.shape[1]
        },
        "columns": df.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": duplicates
    }

    return df, dataset_info

def analyze_class_distribution(df, label_col='sentiment'):
    """分析类别分布"""
    print("\n" + "=" * 50)
    print("类别分布分析")
    print("=" * 50)

    # 统计各类别数量
    class_counts = df[label_col].value_counts()
    print(f"各类别数量:\n{class_counts}")

    # 计算比例
    class_ratios = df[label_col].value_counts(normalize=True)
    print(f"\n各类别比例:\n{class_ratios}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 柱状图
    ax1 = axes[0]
    bars = ax1.bar(class_counts.index.astype(str), class_counts.values)
    ax1.set_xlabel('情感类别')
    ax1.set_ylabel('样本数量')
    ax1.set_title('情感类别分布')

    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    # 饼图
    ax2 = axes[1]
    ax2.pie(class_counts.values, labels=class_counts.index.astype(str),
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('情感类别比例')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 收集类别分布信息用于JSON输出
    class_distribution_info = {
        "counts": class_counts.to_dict(),
        "ratios": {str(key): float(value) for key, value in class_ratios.to_dict().items()}
    }

    return class_counts, class_distribution_info

def analyze_text_length(df, text_col='news_title'):
    """分析文本长度分布"""
    print("\n" + "=" * 50)
    print("文本长度分析")
    print("=" * 50)

    # 计算文本长度（按字符）
    df['char_length'] = df[text_col].astype(str).apply(len)

    # 计算文本长度（按单词）
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(str(x).split()))

    print(f"字符长度统计:")
    print(df['char_length'].describe())

    print(f"\n单词数量统计:")
    print(df['word_count'].describe())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 字符长度分布
    ax1 = axes[0, 0]
    ax1.hist(df['char_length'], bins=50, alpha=0.7, color='skyblue')
    ax1.set_xlabel('字符长度')
    ax1.set_ylabel('频率')
    ax1.set_title('文本字符长度分布')
    ax1.axvline(df['char_length'].median(), color='red', linestyle='--',
               label=f'中位数: {df["char_length"].median():.1f}')
    ax1.legend()

    # 按类别的字符长度
    ax2 = axes[0, 1]
    for sentiment in sorted(df['sentiment'].unique()):
        subset = df[df['sentiment'] == sentiment]
        ax2.hist(subset['char_length'], bins=30, alpha=0.5,
                label=f'类别 {sentiment}')
    ax2.set_xlabel('字符长度')
    ax2.set_ylabel('频率')
    ax2.set_title('按类别的文本字符长度分布')
    ax2.legend()

    # 单词数量分布
    ax3 = axes[1, 0]
    ax3.hist(df['word_count'], bins=30, alpha=0.7, color='lightcoral')
    ax3.set_xlabel('单词数量')
    ax3.set_ylabel('频率')
    ax3.set_title('文本单词数量分布')
    ax3.axvline(df['word_count'].median(), color='red', linestyle='--',
               label=f'中位数: {df["word_count"].median():.1f}')
    ax3.legend()

    # 箱线图
    ax4 = axes[1, 1]
    box_data = [df[df['sentiment'] == s]['word_count'] for s in sorted(df['sentiment'].unique())]
    ax4.boxplot(box_data, labels=sorted(df['sentiment'].unique()))
    ax4.set_xlabel('情感类别')
    ax4.set_ylabel('单词数量')
    ax4.set_title('各类别单词数量分布箱线图')

    plt.tight_layout()
    plt.savefig('text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 收集文本长度统计信息用于JSON输出
    text_length_info = {
        "char_length": df['char_length'].describe().to_dict(),
        "word_count": df['word_count'].describe().to_dict(),
        "by_sentiment": {}
    }

    # 添加按情感类别的统计
    for sentiment in sorted(df['sentiment'].unique()):
        subset = df[df['sentiment'] == sentiment]
        text_length_info["by_sentiment"][str(sentiment)] = {
            "char_length": subset['char_length'].describe().to_dict(),
            "word_count": subset['word_count'].describe().to_dict()
        }

    return df, text_length_info

def enhanced_preprocess_text(text):
    """增强的文本预处理函数"""
    if not isinstance(text, str):
        return ""

    text = str(text).lower()

    # 移除特殊字符和数字，但保留有用的符号如$、%、+
    text = re.sub(r'[^a-zA-Z\s\$\+\%\.]', ' ', text)

    # 分割单词
    words = text.split()

    # 扩展停用词列表
    extended_stopwords = set(stopwords.words('english') + [
        'said', 'would', 'could', 'also', 'one', 'two', 'three',
        'first', 'second', 'third', 'new', 'company', 'companies',
        'year', 'years', 'percent', 'pct', 'update', 'according',
        'said', 'says', 'including', 'like', 'us', 'vs', 'via'
    ])

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    filtered_words = []
    for word in words:
        if word not in extended_stopwords and len(word) > 2:
            # 尝试词形还原（名词和动词）
            lemma = lemmatizer.lemmatize(word, pos='n')  # 名词
            lemma = lemmatizer.lemmatize(lemma, pos='v')  # 动词
            filtered_words.append(lemma)

    return ' '.join(filtered_words)

def analyze_most_common_words(df, text_col='news_title', label_col='sentiment'):
    """分析最常见的单词"""
    print("\n" + "=" * 50)
    print("高频词汇分析")
    print("=" * 50)

    # 预处理文本
    df['processed_text'] = df[text_col].apply(enhanced_preprocess_text)

    # 分离正负样本
    positive_texts = ' '.join(df[df[label_col] == 1]['processed_text'])
    negative_texts = ' '.join(df[df[label_col] == -1]['processed_text'])
    neutral_texts = ' '.join(df[df[label_col] == 0]['processed_text']) if 0 in df[label_col].values else ""

    def get_top_words(text, n=20):
        words = text.split()
        return Counter(words).most_common(n)

    # 收集各类别的高频词信息
    most_common_words_info = {}

    # 获取各类别的高频词
    if positive_texts:
        positive_top = get_top_words(positive_texts, 20)
        most_common_words_info['1'] = [{"word": word, "count": count} for word, count in positive_top]
        print(f"正面样本({label_col}=1)前20高频词:")
        for word, count in positive_top:
            print(f"  {word}: {count}")

    if negative_texts:
        negative_top = get_top_words(negative_texts, 20)
        most_common_words_info['-1'] = [{"word": word, "count": count} for word, count in negative_top]
        print(f"\n负面样本({label_col}=-1)前20高频词:")
        for word, count in negative_top:
            print(f"  {word}: {count}")

    if neutral_texts:
        neutral_top = get_top_words(neutral_texts, 20)
        most_common_words_info['0'] = [{"word": word, "count": count} for word, count in neutral_top]
        print(f"\n中性样本({label_col}=0)前20高频词:")
        for word, count in neutral_top:
            print(f"  {word}: {count}")

    # 生成词云
    fig, axes = plt.subplots(1, 3 if neutral_texts else 2, figsize=(15, 5))

    idx = 0
    if positive_texts:
        wc = WordCloud(width=400, height=300, background_color='white').generate(positive_texts)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'正面样本词云 ({label_col}=1)')
        axes[idx].axis('off')
        idx += 1

    if negative_texts:
        wc = WordCloud(width=400, height=300, background_color='white').generate(negative_texts)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'负面样本词云 ({label_col}=-1)')
        axes[idx].axis('off')
        idx += 1

    if neutral_texts:
        wc = WordCloud(width=400, height=300, background_color='white').generate(neutral_texts)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'中性样本词云 ({label_col}=0)')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df, most_common_words_info

def analyze_model_performance(df):
    """分析模型性能，识别错误分类样本"""
    print("\n" + "=" * 50)
    print("模型错误分析")
    print("=" * 50)

    # 使用现有模型进行预测
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # 准备数据
    df['processed_text'] = df['news_title'].apply(enhanced_preprocess_text)
    X = df['processed_text']
    y = df['sentiment']

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 特征提取
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # 加入bigram
        min_df=2,  # 最小文档频率
        max_df=0.95  # 最大文档频率
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 训练模型
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        C=0.5  # 调整正则化强度
    )

    model.fit(X_train_tfidf, y_train)

    # 预测
    y_pred = model.predict(X_test_tfidf)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"加权F1分数: {f1:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 识别错误分类的样本
    test_df = pd.DataFrame({
        'text': X_test,
        'true_label': y_test,
        'pred_label': y_pred
    })

    test_df['correct'] = test_df['true_label'] == test_df['pred_label']
    errors = test_df[~test_df['correct']]

    print(f"\n错误分类样本数量: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

    if len(errors) > 0:
        print("\n错误分类样本示例:")
        for idx, row in errors.head(10).iterrows():
            print(f"\n文本: {row['text'][:100]}...")
            print(f"真实标签: {row['true_label']}, 预测标签: {row['pred_label']}")

    # 存储模型性能指标
    model_performance_info = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "labels": [int(label) for label in sorted(y.unique())],
            "matrix": cm.tolist()
        },
        "error_count": int(len(errors)),
        "total_test_samples": int(len(y_test)),
        "error_rate": float(len(errors)/len(y_test)*100)
    }

    return model, vectorizer, errors, model_performance_info

def analyze_feature_importance(model, vectorizer, n=20):
    """分析特征重要性"""
    print("\n" + "=" * 50)
    print("特征重要性分析")
    print("=" * 50)

    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()

    # 存储特征重要性信息
    feature_importance_info = {}

    # 对于逻辑回归，使用系数的绝对值作为重要性
    if hasattr(model, 'coef_'):
        # 检查模型是否为多分类
        if len(model.coef_.shape) > 1:
            # 多分类情况
            for i, class_label in enumerate(model.classes_):
                print(f"\n类别 {class_label} 最重要的特征:")
                # 确保索引不越界
                if i < model.coef_.shape[0]:
                    coef = model.coef_[i]
                    indices = np.argsort(np.abs(coef))[-n:]

                    # 存储当前类别的特征重要性
                    feature_importance_info[str(class_label)] = []
                    for idx in reversed(indices):
                        importance_value = float(coef[idx])
                        print(f"  {feature_names[idx]}: {importance_value:.4f}")
                        feature_importance_info[str(class_label)].append({
                            "feature": feature_names[idx],
                            "importance": importance_value
                        })
        else:
            # 二分类情况
            print(f"\n最重要的特征:")
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[-n:]

            feature_importance_info["overall"] = []
            for idx in reversed(indices):
                importance_value = float(coef[idx])
                print(f"  {feature_names[idx]}: {importance_value:.4f}")
                feature_importance_info["overall"].append({
                    "feature": feature_names[idx],
                    "importance": importance_value
                })

    return feature_importance_info

def main():
    # 文件路径 - 请根据实际情况修改
    data_path = '../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'  # 或者 '../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'

    print("开始数据分析...")

    # 1. 加载数据
    df, dataset_info = load_and_analyze_data(data_path)

    # 2. 分析类别分布
    class_counts, class_distribution_info = analyze_class_distribution(df)

    # 3. 分析文本长度
    df, text_length_info = analyze_text_length(df)

    # 4. 分析高频词汇
    df, most_common_words_info = analyze_most_common_words(df)

    # 5. 分析模型性能
    model, vectorizer, errors, model_performance_info = analyze_model_performance(df)

    # 6. 分析特征重要性
    feature_importance_info = analyze_feature_importance(model, vectorizer)

    # 7. 编译所有EDA结果到一个字典
    eda_results = {
        "dataset_info": dataset_info,
        "class_distribution": class_distribution_info,
        "text_length_analysis": text_length_info,
        "most_common_words": most_common_words_info,
        "model_performance": model_performance_info,
        "feature_importance": feature_importance_info,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    # 转换所有numpy数据类型为Python原生类型
    eda_results_serializable = convert_numpy_types(eda_results)

    # 保存EDA结果到JSON文件
    json_output_path = 'eda_results.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(eda_results_serializable, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("数据分析完成！")
    print("=" * 50)

    # 保存处理后的数据
    df.to_csv('processed_sentiment_data.csv', index=False)
    print(f"\n处理后的数据已保存到: processed_sentiment_data.csv")
    print(f"EDA分析结果已保存到: {json_output_path}")

if __name__ == "__main__":
    main()
import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.model_selection import cross_validate


# --- 1. 定义预处理函数 ---
def preprocess_text(text):
    """
    中文文本预处理：去除标点符号，进行分词。
    """
    # 1. 清理和去除标点符号（保留中文、数字和字母）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text))

    # 2. 中文分词 (jieba)
    # 使用精确模式分词
    words = jieba.cut(text)

    # 3. 过滤掉可能的单字符和停用词（此处未加载外部停用词表，只做基本过滤）
    filtered_words = [word for word in words if len(word.strip()) > 1]

    return ' '.join(filtered_words)


# # --- 2. 加载数据 ---
# # 假设文件名为 training_news-topic.xlsx
# file_path = "training_news-topic.xlsx"
# try:
#     df = pd.read_excel(file_path)
# except FileNotFoundError:
#     print(f"错误：未找到文件 {file_path}，请确保文件路径正确。")
#     exit()

# # 检查数据列名，确保与提供的格式一致
# if 'news_title' not in df.columns or 'topic' not in df.columns:
#     print("错误：数据文件中未找到 'news_title' 或 'topic' 列。")
#     exit()

# # 特征（X）和目标变量（y）
# X = df['news_title'].fillna('')  # 填充缺失值以防报错
# y = df['topic']

# # --- 3. 文本预处理（分词） ---
# print("正在进行中文分词和预处理...")
# # 对新闻标题进行分词处理
# X_processed = X.apply(preprocess_text)
# print("分词完成。")

# # --- 4. 划分训练集和保留测试集 ---
# # 划分出 100 个样本作为保留测试集 (test_size=100)
# TEST_SIZE = 100
# RANDOM_STATE = 42

# X_train, X_test, y_train, y_test = train_test_split(
#     X_processed,
#     y,
#     test_size=TEST_SIZE,
#     random_state=RANDOM_STATE,
#     stratify=y  # 使用分层抽样，确保测试集中的主题分布与总数据集相似
# )

# print(f"\n数据划分完成：")
# print(f"训练集大小: {len(X_train)}")
# print(f"保留测试集大小: {len(X_test)}")


# # --- 5. 构建模型管道 (Pipeline) ---
# text_clf_svm = Pipeline([
#     ('tfidf', TfidfVectorizer(
#         sublinear_tf=True,
#         min_df=5,
#         norm='l2',
#         encoding='utf-8',
#         ngram_range=(1, 2)
#     )),
#     ('clf', SVC(
#         kernel='linear',
#         C=1.0,
#         probability=True,
#         random_state=RANDOM_STATE
#     )),
# ])

# # --- 6. 5-折交叉验证 (CV) 训练和评估（只在训练集上执行） ---
# print("\n--- 6. 5-折交叉验证评估 (在训练集上) ---")
# scoring = ['f1_weighted']  # 项目要求的加权F1-score

# cv_results = cross_validate(
#     text_clf_svm,
#     X_train, # 使用划分后的训练集
#     y_train,
#     cv=5,
#     scoring=scoring,
#     return_train_score=False,
#     n_jobs=-1
# )

# # 计算并打印所需的平均加权 F1-score
# mean_weighted_f1 = cv_results['test_f1_weighted'].mean()
# print(f"5-Fold CV 加权 F1-score 结果: {cv_results['test_f1_weighted']}")
# print(f"平均加权 F1-score（项目提交所需）: {mean_weighted_f1:.4f}")

# # --- 7. 在**整个训练集**上训练最终模型 ---
# print("\n--- 7. 训练最终模型 (在整个训练集上) ---")
# final_model = text_clf_svm.fit(X_train, y_train) # 使用 X_train 训练
# print("最终模型训练完成。")


# # --- 8. 在保留测试集上测试模型并报告得分 ---
# print("\n--- 8. 在保留测试集上测试模型并报告得分 ---")

# # 预测保留测试集的标签
# y_pred = final_model.predict(X_test)

# # 计算加权 F1-score
# test_weighted_f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"保留测试集加权 F1-score: {test_weighted_f1:.4f}")

# # 打印完整的分类报告
# print("\n保留测试集分类报告:\n")
# # zero_division='warn' 是默认值，当某个类别的召回率/精确率为零时发出警告。
# print(classification_report(y_test, y_pred, zero_division=0))


# # --- 9. 保存管道模型 (为部署做准备) ---
# model_filename = 'topic_classification_svm_pipeline_split.joblib'
# dump(final_model, model_filename)
# print(f"\n模型管道已保存到: {model_filename}")
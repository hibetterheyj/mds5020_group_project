关键发现和针对性改进建议：
1. 数据特性发现：
正面样本(1): 2805个 (65.2%)

负面样本(-1): 1495个 (34.8%)

存在类别不平衡，但不算严重

2. 重要特征发现：
正面关键词: profit, rise, increase, growth, higher
负面关键词: loss, fall, decrease, drop, cut, lower

```
# [file name]: targeted_improvements.py
import pandas as pd
import numpy as np
from collections import Counter

def create_sentiment_features(df):
    """基于EDA发现的模式创建手工特征"""

    # 正面词汇列表（基于EDA发现）
    positive_words = [
        'profit', 'rise', 'increase', 'growth', 'higher', 'gain',
        'win', 'success', 'improve', 'boost', 'surge', 'soar',
        'strong', 'positive', 'beat', 'exceed'
    ]

    # 负面词汇列表（基于EDA发现）
    negative_words = [
        'loss', 'fall', 'decrease', 'drop', 'cut', 'lower',
        'decline', 'weak', 'negative', 'miss', 'fail', 'warn',
        'drop', 'hit', 'lose', 'fell', 'decrease', 'downgrade'
    ]

    # 金融特定词汇
    financial_words = ['eur', 'net', 'sale', 'share', 'price', 'dividend', 'million']

    # 创建特征
    features = []

    for text in df['news_title']:
        text_lower = str(text).lower()

        # 1. 正面词汇计数
        pos_count = sum(1 for word in positive_words if word in text_lower)

        # 2. 负面词汇计数
        neg_count = sum(1 for word in negative_words if word in text_lower)

        # 3. 情感词汇比例
        total_sentiment_words = pos_count + neg_count
        pos_ratio = pos_count / (total_sentiment_words + 1e-10)
        neg_ratio = neg_count / (total_sentiment_words + 1e-10)

        # 4. 净情感得分
        net_sentiment = pos_count - neg_count

        # 5. 是否包含强烈情感词
        has_strong_positive = any(word in text_lower for word in ['soar', 'surge', 'beat'])
        has_strong_negative = any(word in text_lower for word in ['plunge', 'crash', 'collapse', 'slump'])

        # 6. 金融词汇密度
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

def analyze_misclassifications(df, y_true, y_pred):
    """分析错误分类样本的模式"""

    errors = df[y_true != y_pred].copy()
    errors['error_type'] = np.where(
        (y_true == 1) & (y_pred == 0), 'FN',  # 假阴性：正面预测为负面
        np.where((y_true == 0) & (y_pred == 1), 'FP', 'Other')  # 假阳性：负面预测为正面
    )

    print(f"\n错误分析:")
    print(f"总错误数: {len(errors)}")
    print(f"假阴性(FN): {sum(errors['error_type'] == 'FN')}")
    print(f"假阳性(FP): {sum(errors['error_type'] == 'FP')}")

    # 分析假阴性的共同特征
    fn_samples = errors[errors['error_type'] == 'FN']
    if len(fn_samples) > 0:
        print(f"\n假阴性样本分析 ({len(fn_samples)} 个):")

        # 检查这些样本是否包含负面词汇
        negative_words = ['loss', 'fall', 'cut', 'decline', 'drop']
        fn_with_neg_words = fn_samples['news_title'].apply(
            lambda x: sum(word in str(x).lower() for word in negative_words)
        )
        print(f"包含负面词汇的假阴性样本: {sum(fn_with_neg_words > 0)}")

    return errors

def create_enhanced_model(X_text, X_features, y):
    """创建增强模型，结合文本和手工特征"""
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # 文本特征管道
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9
        ))
    ])

    # 手工特征管道
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # 结合两种特征
    preprocessor = ColumnTransformer([
        ('text', text_pipeline, 'news_title'),
        ('features', feature_pipeline, list(X_features.columns))
    ])

    # 完整管道
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            C=0.5
        ))
    ])

    return pipeline
```

## 总结建议：

1. **解决数值稳定性问题**：
   - 使用更保守的TF-IDF参数
   - 增加`min_df`减少稀疏特征
   - 使用`norm='l2'`规范化
2. **处理标签问题**：
   - 将标签映射为[0, 1]供XGBoost使用
   - 预测时映射回原始标签[-1, 1]
3. **基于EDA的改进**：
   - 利用发现的正面/负面关键词创建手工特征
   - 重点处理假阴性和假阳性样本
   - 考虑使用集成方法结合多种特征
4. **简单有效的方法**：
   - 逻辑回归已经表现不错(0.8072)
   - 可以尝试添加手工特征进一步提升
   - 考虑使用集成学习或投票分类器

---

```

## improved_sentiment_model.py

============================================================
基准测试结果总结
============================================================
                  Model  Mean F1   Std F1 Status
             Linear SVC 0.803617 0.007069     成功
    Logistic Regression 0.801237 0.009137     成功
          Random Forest 0.778065 0.011516     成功
Multinomial Naive Bayes 0.773990 0.010258     成功
                XGBoost 0.745074 0.011213     成功
               LightGBM 0.737397 0.015263     成功

最佳参数: {'clf__C': 0.01, 'tfidf__max_features': 3000, 'tfidf__ngram_range': (1, 1)}
最佳交叉验证分数: 0.7412

选择并训练最终模型...

============================================================
训练最终模型 (logistic)
============================================================
5折交叉验证F1分数:
  第1折: 0.8001
  第2折: 0.8103
  第3折: 0.7995
  第4折: 0.8070
  第5折: 0.7850
  平均分数: 0.8004 (±0.0087)

============================================================
模型错误分析
============================================================
分类报告:
              precision    recall  f1-score   support

           0       0.80      0.89      0.84      1495
           1       0.94      0.88      0.91      2805

    accuracy                           0.89      4300
   macro avg       0.87      0.89      0.88      4300
weighted avg       0.89      0.89      0.89      4300


总共 492 个错误样本 (11.4%)

随机选择的 10 个错误样本:

样本 742:
  文本: water utility severn trent up save forecast, fy profit fall...
  真实标签: 1, 预测标签: 0

样本 460:
  文本: sarawak oil palm 3q earnings double high palm product price...
  真实标签: 0, 预测标签: 1

样本 2484:
  文本: pavilion reit see flattish rental reversion pavilion mall...
  真实标签: 0, 预测标签: 1

样本 2590:
  文本: capitaland report 18.8% low 1q earnings s$319.1m absence one-off gain...
  真实标签: 1, 预测标签: 0

样本 559:
  文本: malaysia april jobless rate stay 3.4% m-o-m, unemployed person 0.2%...
  真实标签: 0, 预测标签: 1

样本 1784:
  文本: klci track global markets, tick high gain see limit...
  真实标签: 0, 预测标签: 1

样本 291:
  文本: klci expect stay 1,720-level bear remain control...
  真实标签: 1, 预测标签: 0

样本 1100:
  文本: ceo optimistic global economic outlook 2018, threat growth remain...
  真实标签: 0, 预测标签: 1

样本 4031:
  文本: perstima 2q net profit 73.48% low margin sale volume...
  真实标签: 0, 预测标签: 1

样本 2877:
  文本: retail sale set close 4% 5% growth...
  真实标签: 0, 预测标签: 1

模型已保存到: improved_sentiment_model_fixed.joblib

============================================================
测试预测功能
============================================================

标题: Excellent Performance beyond Expectations
预测情感: 1
概率分布: {np.int64(0): np.float64(0.43222626260098884), np.int64(1): np.float64(0.5677737373990112)}

标题: Company reports significant losses in Q3
预测情感: -1
概率分布: {np.int64(0): np.float64(0.6356980952742479), np.int64(1): np.float64(0.364301904725752)}

标题: Stable growth maintained throughout the year
预测情感: 1
概率分布: {np.int64(0): np.float64(0.2306258519206681), np.int64(1): np.float64(0.7693741480793319)}

标题: Market crashes amid global economic concerns
预测情感: -1
概率分布: {np.int64(0): np.float64(0.5926193063935153), np.int64(1): np.float64(0.40738069360648466)}

标题: Innovative product launch boosts company shares
预测情感: 1
概率分布: {np.int64(0): np.float64(0.1563496760898797), np.int64(1): np.float64(0.8436503239101203)}

标题: Profit fell by 30% due to market conditions
预测情感: -1
概率分布: {np.int64(0): np.float64(0.7815254370342724), np.int64(1): np.float64(0.21847456296572765)}

标题: Sales increased by 15% in the last quarter
预测情感: 1
概率分布: {np.int64(0): np.float64(0.40767007858376414), np.int64(1): np.float64(0.5923299214162359)}

============================================================
模型训练完成！
============================================================

性能总结:
- 数据集大小: 4300
- 类别分布: 正面(1): 2805, 负面(0): 1495
- 最佳基准模型: Linear SVC
- 最终模型: 逻辑回归
```

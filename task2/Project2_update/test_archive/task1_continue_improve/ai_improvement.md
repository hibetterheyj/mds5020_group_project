# 首次提问

请帮我分析task2-subtask1，目前我是采用传统机器学习方法，尝试了log regression，svm，lighgbm。xgbboost等等，并且进行EDA结果如下（不过有些像长度特征感觉不应该直接考虑在情感分析内，以下只是训练集的特征，测试集未公开，所以考虑可能存在风险）：
```
{
  "dataset_info": {
    "shape": {
      "rows": 4300,
      "columns": 3
    },
    "columns": [
      "doc_id",
      "news_title",
      "sentiment"
    ],
    "data_types": {
      "doc_id": "int64",
      "news_title": "object",
      "sentiment": "int64"
    },
    "missing_values": {
      "doc_id": 0,
      "news_title": 0,
      "sentiment": 0
    },
    "duplicates": 0
  },
  "class_distribution": {
    "counts": {
      "1": 2805,
      "-1": 1495
    },
    "ratios": {
      "1": 0.6523255813953488,
      "-1": 0.34767441860465115
    }
  },
  "text_length_analysis": {
    "char_length": {
      "count": 4300.0,
      "mean": 88.87116279069767,
      "std": 44.62339534369326,
      "min": 24.0,
      "25%": 63.0,
      "50%": 73.0,
      "75%": 93.0,
      "max": 298.0
    },
    "word_count": {
      "count": 4300.0,
      "mean": 15.684418604651162,
      "std": 8.391956017968402,
      "min": 4.0,
      "25%": 11.0,
      "50%": 12.0,
      "75%": 17.0,
      "max": 57.0
    },
    "by_sentiment": {
      "-1": {
        "char_length": {
          "count": 1495.0,
          "mean": 84.81939799331104,
          "std": 40.34658447942548,
          "min": 27.0,
          "25%": 62.0,
          "50%": 71.0,
          "75%": 88.0,
          "max": 296.0
        },
        "word_count": {
          "count": 1495.0,
          "mean": 15.132441471571907,
          "std": 7.891066262942571,
          "min": 4.0,
          "25%": 11.0,
          "50%": 12.0,
          "75%": 16.0,
          "max": 56.0
        }
      },
      "1": {
        "char_length": {
          "count": 2805.0,
          "mean": 91.03065953654189,
          "std": 46.606470698957544,
          "min": 24.0,
          "25%": 63.0,
          "50%": 74.0,
          "75%": 97.0,
          "max": 298.0
        },
        "word_count": {
          "count": 2805.0,
          "mean": 15.97860962566845,
          "std": 8.634038330641449,
          "min": 4.0,
          "25%": 11.0,
          "50%": 12.0,
          "75%": 17.0,
          "max": 57.0
        }
      }
    }
  },
  "most_common_words": {
    "1": [
      {
        "word": "profit",
        "count": 429
      },
      {
        "word": "eur",
        "count": 383
      },
      {
        "word": "net",
        "count": 326
      },
      {
        "word": "sale",
        "count": 226
      },
      {
        "word": "rise",
        "count": 186
      },
      {
        "word": "share",
        "count": 152
      },
      {
        "word": "group",
        "count": 130
      },
      {
        "word": "finnish",
        "count": 129
      },
      {
        "word": "market",
        "count": 120
      },
      {
        "word": "increase",
        "count": 120
      },
      {
        "word": "growth",
        "count": 117
      },
      {
        "word": "million",
        "count": 114
      },
      {
        "word": "price",
        "count": 107
      },
      {
        "word": "higher",
        "count": 105
      },
      {
        "word": "dividend",
        "count": 104
      },
      {
        "word": "malaysia",
        "count": 102
      },
      {
        "word": "bank",
        "count": 97
      },
      {
        "word": "business",
        "count": 97
      },
      {
        "word": "sen",
        "count": 96
      },
      {
        "word": "service",
        "count": 95
      }
    ],
    "-1": [
      {
        "word": "eur",
        "count": 265
      },
      {
        "word": "profit",
        "count": 193
      },
      {
        "word": "net",
        "count": 145
      },
      {
        "word": "loss",
        "count": 123
      },
      {
        "word": "sale",
        "count": 115
      },
      {
        "word": "fall",
        "count": 108
      },
      {
        "word": "share",
        "count": 101
      },
      {
        "word": "price",
        "count": 92
      },
      {
        "word": "finnish",
        "count": 78
      },
      {
        "word": "lower",
        "count": 74
      },
      {
        "word": "cut",
        "count": 69
      },
      {
        "word": "operate",
        "count": 67
      },
      {
        "word": "market",
        "count": 66
      },
      {
        "word": "million",
        "count": 63
      },
      {
        "word": "quarter",
        "count": 62
      },
      {
        "word": "bank",
        "count": 62
      },
      {
        "word": "period",
        "count": 60
      },
      {
        "word": "report",
        "count": 60
      },
      {
        "word": "oil",
        "count": 60
      },
      {
        "word": "earnings",
        "count": 55
      }
    ]
  },
  "model_performance": {
    "accuracy": 0.8290697674418605,
    "precision": 0.8284463896496227,
    "recall": 0.8290697674418605,
    "f1_score": 0.8287283218267724,
    "confusion_matrix": {
      "labels": [
        -1,
        1
      ],
      "matrix": [
        [
          223,
          76
        ],
        [
          71,
          490
        ]
      ]
    },
    "error_count": 147,
    "total_test_samples": 860,
    "error_rate": 17.093023255813954
  },
  "feature_importance": {
    "-1": [
      {
        "feature": "fall",
        "importance": -3.048946836732839
      },
      {
        "feature": "rise",
        "importance": 2.503947743907569
      },
      {
        "feature": "loss",
        "importance": -2.2955607319187292
      },
      {
        "feature": "decrease",
        "importance": -2.131491050134111
      },
      {
        "feature": "drop",
        "importance": -2.024642634223156
      },
      {
        "feature": "cut",
        "importance": -1.7003126705933516
      },
      {
        "feature": "increase",
        "importance": 1.6726924724968086
      },
      {
        "feature": "fell",
        "importance": -1.6545517808087469
      },
      {
        "feature": "lose",
        "importance": -1.5541497836675666
      },
      {
        "feature": "hit",
        "importance": -1.4849270101048484
      },
      {
        "feature": "warn",
        "importance": -1.4744309976081524
      },
      {
        "feature": "launch",
        "importance": 1.3402353433229819
      },
      {
        "feature": "win",
        "importance": 1.2944389513810919
      },
      {
        "feature": "claim",
        "importance": -1.2914598094679026
      },
      {
        "feature": "weak",
        "importance": -1.2569444240502967
      },
      {
        "feature": "strong",
        "importance": 1.2567946938021826
      },
      {
        "feature": "due",
        "importance": -1.2317450406174888
      },
      {
        "feature": "decrease eur",
        "importance": -1.2300032961745864
      },
      {
        "feature": "downgrade",
        "importance": -1.222157117729609
      },
      {
        "feature": "lower",
        "importance": -1.1880730878312413
      }
    ]
  },
  "timestamp": "2025-12-07T23:44:26.162844"
}
我参考结果设计了相关的利用手工特征的函数handcrafted_features.py
```
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
```
但是测试下来，提升效果不明显，总体效果还是很差最高只有80.9%，相比不使用handcrafted提升效果微乎其微而且会不会存在过拟合？各个方法目前最好结果如下：

### LinearSVC

- **Best F1 Score**: 0.8095
- **Standard Deviation**: 0.0105
- **Use Handcrafted Features**: True
- **Best Parameters**:
  - `tfidf__ngram_range`: [1, 3]
  - `tfidf__max_features`: 7000
  - `penalty`: l2
  - `max_iter`: 1000
  - `dual`: False
  - `class_weight`: balanced
  - `C`: 0.23357214690901212
  - `use_handcrafted`: True

### SVC with RBF Kernel

- **Best F1 Score**: 0.8097
- **Standard Deviation**: 0.0136
- **Use Handcrafted Features**: True
- **Best Parameters**:
  - `tfidf__ngram_range`: [1, 2]
  - `tfidf__max_features`: 7000
  - `gamma`: 0.01
  - `class_weight`: None
  - `C`: 51.794746792312125
  - `use_handcrafted`: True

### SVC with Sigmoid Kernel

- **Best F1 Score**: 0.8087
- **Standard Deviation**: 0.0141
- **Use Handcrafted Features**: True
- **Best Parameters**:
  - `tfidf__ngram_range`: [1, 3]
  - `tfidf__max_features`: 3000
  - `gamma`: 0.00046415888336127773
  - `coef0`: 0.33333333333333326
  - `class_weight`: balanced
  - `C`: 1000.0
  - `use_handcrafted`: True

## Cross-Classifier Comparison

- **LinearSVC**: 0.8095
- **SVC with RBF Kernel**: 0.8097
- **SVC with Sigmoid Kernel**: 0.8087

### Best Overall Model (So Far)

- **Classifier**: SVC with RBF Kernel
- **F1 Score**: 0.8097
- **Parameters**:
  - `tfidf__ngram_range`: [1, 2]
  - `tfidf__max_features`: 7000
  - `gamma`: 0.01
  - `class_weight`: None
  - `C`: 51.794746792312125
  - `use_handcrafted`: True

## Impact of Handcrafted Features

- **LinearSVC**:
  - With handcrafted features: 0.8095
  - Without handcrafted features: 0.8076
  - Difference: +0.0019
- **SVC with RBF Kernel**:
  - With handcrafted features: 0.8097
  - Without handcrafted features: 0.8061
  - Difference: +0.0036
- **SVC with Sigmoid Kernel**:
  - With handcrafted features: 0.8087
  - Without handcrafted features: 0.8072
  - Difference: +0.0015


{
    "parameters": {
      "classifier__learning_rate": 0.07,
      "classifier__max_depth": 6,
      "classifier__n_estimators": 400,
      "classifier__num_leaves": 63,
      "features__text__tfidf__max_features": 2000,
      "use_handcrafted": true
    },
    "metrics": {
      "mean_score": 0.7735244311837117,
      "std_score": 0.015138431499928402,
      "scores": [
        0.7764985149455326,
        0.7700918971434032,
        0.7572322517630763,
        0.800866904087357,
        0.7629325879791902
      ]
    }
  }

{
   "parameters": {
   "text__tfidf__ngram_range": [
      1,
      2
   ],
   "text__tfidf__max_features": 3000,
   "classifier__solver": "liblinear",
   "classifier__penalty": "l1",
   "classifier__max_iter": 200,
   "classifier__class_weight": null,
   "classifier__C": 4.281332398719396,
   "use_handcrafted": false
   },
   "metrics": {
   "mean_score": 0.8092571021501211,
   "std_score": 0.006799347083290248,
   "scores": [
      0.8102830610122308,
      0.7990576094977683,
      0.807036836728562,
      0.8202565696745638,
      0.8096514338374804
   ]
   }
}


除了现在这些库scikit-learn,xgboost,lightgbm有没有什么轻量的适合API部署的并且满足（maximum runtime memory of container should not exceed 900 MB (without accessing GPU or Internet). The image size should not exceed 4 GB）的方法推荐呢？因为后续测评不仅希望有比较高的weighted F1-score，也有小部分考虑Prediction time （5%）得分


```

# 豆包

https://www.doubao.com/chat/32961012901834754

# deepseek

https://chat.deepseek.com/a/chat/s/98c5fc0e-521b-4243-81d1-786c03ebb112

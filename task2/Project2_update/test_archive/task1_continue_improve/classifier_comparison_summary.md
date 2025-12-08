# Classifier Comparison Summary

Generated on: 2025-12-08 18:57:21

## Table of Contents

- [LightGBM](#lightgbm)
- [LinearSVC](#linearsvc)
- [LogisticRegression](#logisticregression)
- [SVC_Poly](#svc_poly)
- [SVC_RBF](#svc_rbf)
- [SVC_Sigmoid](#svc_sigmoid)
- [XGBoost](#xgboost)
- [Overall Comparison](#overall-comparison)
- [Handcrafted Features Impact](#handcrafted-features-impact)

## LightGBM

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7756

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    3
  ],
  "features__text__tfidf__max_features": 7000,
  "classifier__subsample": 1.0,
  "classifier__num_leaves": 127,
  "classifier__n_estimators": 500,
  "classifier__max_depth": 10,
  "classifier__learning_rate": 0.05,
  "classifier__colsample_bytree": 0.8
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7597

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    2
  ],
  "text__tfidf__max_features": 7000,
  "classifier__subsample": 1.0,
  "classifier__num_leaves": 15,
  "classifier__n_estimators": 300,
  "classifier__max_depth": 7,
  "classifier__learning_rate": 0.1,
  "classifier__colsample_bytree": 1.0
}
```

## LinearSVC

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8134

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__loss": "hinge",
  "classifier__class_weight": "balanced",
  "classifier__C": 1.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8090

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    3
  ],
  "text__tfidf__max_features": 7000,
  "classifier__loss": "squared_hinge",
  "classifier__class_weight": "balanced",
  "classifier__C": 0.1
}
```

## LogisticRegression

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8067

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__solver": "liblinear",
  "classifier__penalty": "l1",
  "classifier__class_weight": "balanced",
  "classifier__C": 10.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8135

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 3000,
  "classifier__solver": "liblinear",
  "classifier__penalty": "l2",
  "classifier__class_weight": "balanced",
  "classifier__C": 1.0
}
```

## SVC_Poly

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8082

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__gamma": 0.001,
  "classifier__degree": 3,
  "classifier__coef0": 2.0,
  "classifier__class_weight": "balanced",
  "classifier__C": 100.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8071

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 7000,
  "classifier__gamma": 0.1,
  "classifier__degree": 2,
  "classifier__coef0": 2.0,
  "classifier__class_weight": "balanced",
  "classifier__C": 1.0
}
```

## SVC_RBF

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8045

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__gamma": 0.01,
  "classifier__class_weight": "balanced",
  "classifier__C": 100.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8036

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 2000,
  "classifier__gamma": "scale",
  "classifier__class_weight": "balanced",
  "classifier__C": 1.0
}
```

## SVC_Sigmoid

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7984

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 7000,
  "classifier__gamma": 0.01,
  "classifier__coef0": -1.0,
  "classifier__class_weight": "balanced",
  "classifier__C": 100.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8101

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 3000,
  "classifier__gamma": 0.01,
  "classifier__coef0": 1.0,
  "classifier__class_weight": "balanced",
  "classifier__C": 100.0
}
```

## XGBoost

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7976

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 7000,
  "classifier__subsample": 1.0,
  "classifier__n_estimators": 300,
  "classifier__max_depth": 10,
  "classifier__learning_rate": 0.2,
  "classifier__gamma": 0.5,
  "classifier__colsample_bytree": 0.6
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7910

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 3000,
  "classifier__subsample": 1.0,
  "classifier__n_estimators": 500,
  "classifier__max_depth": 7,
  "classifier__learning_rate": 0.2,
  "classifier__gamma": 1.0,
  "classifier__colsample_bytree": 1.0
}
```

## Overall Comparison

| Classifier | Handcrafted Features | Best F1 Score |
|------------|----------------------|---------------|
| LightGBM | Yes | 0.7756 |
| LightGBM | No | 0.7597 |
| LinearSVC | Yes | 0.8134 |
| LinearSVC | No | 0.8090 |
| LogisticRegression | Yes | 0.8067 |
| LogisticRegression | No | 0.8135 |
| SVC_Poly | Yes | 0.8082 |
| SVC_Poly | No | 0.8071 |
| SVC_RBF | Yes | 0.8045 |
| SVC_RBF | No | 0.8036 |
| SVC_Sigmoid | Yes | 0.7984 |
| SVC_Sigmoid | No | 0.8101 |
| XGBoost | Yes | 0.7976 |
| XGBoost | No | 0.7910 |

## Handcrafted Features Impact

- **LightGBM:** +1.59% improvement (from 0.7597 to 0.7756)
- **LinearSVC:** +0.44% improvement (from 0.8090 to 0.8134)
- **LogisticRegression:** -0.68% improvement (from 0.8135 to 0.8067)
- **SVC_Poly:** +0.11% improvement (from 0.8071 to 0.8082)
- **SVC_RBF:** +0.10% improvement (from 0.8036 to 0.8045)
- **SVC_Sigmoid:** -1.17% improvement (from 0.8101 to 0.7984)
- **XGBoost:** +0.66% improvement (from 0.7910 to 0.7976)

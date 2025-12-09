# Classifier Comparison Summary

Generated on: 2025-12-09 03:33:50

## Table of Contents

- [AdaBoostClassifier](#adaboostclassifier)
- [BernoulliNB](#bernoullinb)
- [CategoricalNB](#categoricalnb)
- [DecisionTreeClassifier](#decisiontreeclassifier)
- [ExtraTreesClassifier](#extratreesclassifier)
- [LightGBM](#lightgbm)
- [LinearSVC](#linearsvc)
- [LogisticRegression](#logisticregression)
- [MLPClassifier](#mlpclassifier)
- [MultinomialNB](#multinomialnb)
- [RandomForestClassifier](#randomforestclassifier)
- [SGDClassifier](#sgdclassifier)
- [SVC_Poly](#svc_poly)
- [SVC_RBF](#svc_rbf)
- [SVC_Sigmoid](#svc_sigmoid)
- [XGBoost](#xgboost)
- [Overall Comparison](#overall-comparison)
- [Handcrafted Features Impact](#handcrafted-features-impact)

## AdaBoostClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7382

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__n_estimators": 200,
  "classifier__learning_rate": 1.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.6752

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 5000,
  "classifier__n_estimators": 200,
  "classifier__learning_rate": 1.0
}
```

## BernoulliNB

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8059

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__binarize": 0.0,
  "classifier__alpha": 1.0
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7965

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 2000,
  "classifier__binarize": 0.1,
  "classifier__alpha": 1.0
}
```

## CategoricalNB

### With Handcrafted Features

**Best Weighted F1 Score:** nan

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 2000,
  "classifier__alpha": 0.001
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** nan

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 2000,
  "classifier__alpha": 0.001
}
```

## DecisionTreeClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7614

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 3000,
  "classifier__min_samples_split": 5,
  "classifier__min_samples_leaf": 1,
  "classifier__max_depth": 15,
  "classifier__criterion": "gini"
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7584

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 5000,
  "classifier__min_samples_split": 2,
  "classifier__min_samples_leaf": 1,
  "classifier__max_depth": null,
  "classifier__criterion": "entropy"
}
```

## ExtraTreesClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8100

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 3000,
  "classifier__n_estimators": 100,
  "classifier__min_samples_split": 2,
  "classifier__max_features": "sqrt",
  "classifier__max_depth": null
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8194

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 5000,
  "classifier__n_estimators": 200,
  "classifier__min_samples_split": 5,
  "classifier__max_features": "sqrt",
  "classifier__max_depth": null
}
```

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

## MLPClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8095

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 3000,
  "classifier__solver": "adam",
  "classifier__learning_rate_init": 0.005,
  "classifier__hidden_layer_sizes": [
    50
  ],
  "classifier__alpha": 0.001,
  "classifier__activation": "relu"
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7943

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 3000,
  "classifier__solver": "adam",
  "classifier__learning_rate_init": 0.005,
  "classifier__hidden_layer_sizes": [
    50
  ],
  "classifier__alpha": 0.01,
  "classifier__activation": "relu"
}
```

## MultinomialNB

### With Handcrafted Features

**Best Weighted F1 Score:** 0.8060

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    3
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__alpha": 0.1
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.7799

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    3
  ],
  "text__tfidf__max_features": 3000,
  "classifier__alpha": 0.1
}
```

## RandomForestClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7965

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    1
  ],
  "features__text__tfidf__max_features": 2000,
  "classifier__n_estimators": 50,
  "classifier__min_samples_split": 2,
  "classifier__max_features": "sqrt",
  "classifier__max_depth": null
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8034

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    1
  ],
  "text__tfidf__max_features": 5000,
  "classifier__n_estimators": 200,
  "classifier__min_samples_split": 5,
  "classifier__max_features": "sqrt",
  "classifier__max_depth": null
}
```

## SGDClassifier

### With Handcrafted Features

**Best Weighted F1 Score:** 0.7959

**Best Parameters:**

```json
{
  "features__text__tfidf__ngram_range": [
    1,
    2
  ],
  "features__text__tfidf__max_features": 5000,
  "classifier__penalty": "elasticnet",
  "classifier__loss": "hinge",
  "classifier__class_weight": "balanced",
  "classifier__alpha": 0.0001
}
```

### Without Handcrafted Features

**Best Weighted F1 Score:** 0.8101

**Best Parameters:**

```json
{
  "text__tfidf__ngram_range": [
    1,
    3
  ],
  "text__tfidf__max_features": 7000,
  "classifier__penalty": "l1",
  "classifier__loss": "hinge",
  "classifier__class_weight": null,
  "classifier__alpha": 0.0001
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
| AdaBoostClassifier | Yes | 0.7382 |
| AdaBoostClassifier | No | 0.6752 |
| BernoulliNB | Yes | 0.8059 |
| BernoulliNB | No | 0.7965 |
| CategoricalNB | Yes | nan |
| CategoricalNB | No | nan |
| DecisionTreeClassifier | Yes | 0.7614 |
| DecisionTreeClassifier | No | 0.7584 |
| ExtraTreesClassifier | Yes | 0.8100 |
| ExtraTreesClassifier | No | 0.8194 |
| LightGBM | Yes | 0.7756 |
| LightGBM | No | 0.7597 |
| LinearSVC | Yes | 0.8134 |
| LinearSVC | No | 0.8090 |
| LogisticRegression | Yes | 0.8067 |
| LogisticRegression | No | 0.8135 |
| MLPClassifier | Yes | 0.8095 |
| MLPClassifier | No | 0.7943 |
| MultinomialNB | Yes | 0.8060 |
| MultinomialNB | No | 0.7799 |
| RandomForestClassifier | Yes | 0.7965 |
| RandomForestClassifier | No | 0.8034 |
| SGDClassifier | Yes | 0.7959 |
| SGDClassifier | No | 0.8101 |
| SVC_Poly | Yes | 0.8082 |
| SVC_Poly | No | 0.8071 |
| SVC_RBF | Yes | 0.8045 |
| SVC_RBF | No | 0.8036 |
| SVC_Sigmoid | Yes | 0.7984 |
| SVC_Sigmoid | No | 0.8101 |
| XGBoost | Yes | 0.7976 |
| XGBoost | No | 0.7910 |

## Handcrafted Features Impact

- **AdaBoostClassifier:** +6.30% improvement (from 0.6752 to 0.7382)
- **BernoulliNB:** +0.94% improvement (from 0.7965 to 0.8059)
- **CategoricalNB:** +nan% improvement (from nan to nan)
- **DecisionTreeClassifier:** +0.30% improvement (from 0.7584 to 0.7614)
- **ExtraTreesClassifier:** -0.94% improvement (from 0.8194 to 0.8100)
- **LightGBM:** +1.59% improvement (from 0.7597 to 0.7756)
- **LinearSVC:** +0.44% improvement (from 0.8090 to 0.8134)
- **LogisticRegression:** -0.68% improvement (from 0.8135 to 0.8067)
- **MLPClassifier:** +1.53% improvement (from 0.7943 to 0.8095)
- **MultinomialNB:** +2.61% improvement (from 0.7799 to 0.8060)
- **RandomForestClassifier:** -0.69% improvement (from 0.8034 to 0.7965)
- **SGDClassifier:** -1.41% improvement (from 0.8101 to 0.7959)
- **SVC_Poly:** +0.11% improvement (from 0.8071 to 0.8082)
- **SVC_RBF:** +0.10% improvement (from 0.8036 to 0.8045)
- **SVC_Sigmoid:** -1.17% improvement (from 0.8101 to 0.7984)
- **XGBoost:** +0.66% improvement (from 0.7910 to 0.7976)

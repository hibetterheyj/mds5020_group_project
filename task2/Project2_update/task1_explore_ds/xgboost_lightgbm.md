# XGBoost and LightGBM Performance Comparison

- lightgbm

```
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
```

- xgboost

```
{
    "parameters": {
      "classifier__learning_rate": 0.17,
      "classifier__max_depth": 6,
      "classifier__n_estimators": 300,
      "features__text__tfidf__max_features": 2000,
      "use_handcrafted": true
    },
    "metrics": {
      "mean_score": 0.7879848599829378,
      "std_score": 0.00966403164275438,
      "scores": [
        0.7895455436751775,
        0.7856591236757977,
        0.7808424747696358,
        0.8056652919406737,
        0.7782118658534045
      ]
    }
  }
```


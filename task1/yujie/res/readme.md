# res

## xgboost

```
GridSearchCV hyperparameter search - 5-fold cross-validation: 1120it [01:35, 11.75it/s]
Best parameters: {'colsample_bytree': 0.601, 'gamma': 0.43415, 'learning_rate': 0.015, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 280, 'reg_alpha': 0.5366, 'reg_lambda': 2.848, 'subsample': 0.6944466}
Best cross-validation roc_auc: 0.8019
Hyperparameter tuning data saved to ../yujie/res/xgboost_tuning_results.json
Final model trained with best parameters: {'colsample_bytree': 0.601, 'gamma': 0.43415, 'learning_rate': 0.015, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 280, 'reg_alpha': 0.5366, 'reg_lambda': 2.848, 'subsample': 0.6944466}
5-fold CV AUC scores: [0.80048363 0.814364   0.79292273 0.79587561 0.80487872]
Mean CV AUC: 0.8017 (+/- 0.0075)
Predicting probabilities for test data...
Predicted probabilities range: [0.0181, 0.8836]
Saving predictions to ../tests/bank_marketing_test_scores_xgboost.csv...
Predictions saved successfully. Shape: (8000, 1)
Final score range: [0.0181, 0.8836]

==================================================
PERFORMANCE REPORT
==================================================
Mean AUC (5-fold CV): 0.8017
Standard Deviation: 0.0075
Individual fold scores: ['0.8005', '0.8144', '0.7929', '0.7959', '0.8049']
==================================================
Hyperparameter tuning visualization saved to ../yujie/res/xgboost_hyperparameter_tuning.png
Hyperparameter tuning visualization generated and saved to ../yujie/res/xgboost_hyperparameter_tuning.png
```
# Team11 Final Project Task1

## Team Members
- LAN Zichang (225040051)
- HE Yujie (225040114)
- LEI Mingyu (225040082)
- ZHANG Runjin (225040050)
- LUO Zitai (225040095)

## Methodology
We used the XGBoost algorithm to train a classification model on the banking marketing dataset. The following preprocessing steps were applied:
1. Removed duplicate rows from the training dataset
2. Divided features into categorical and numerical columns
3. Applied one-hot encoding to categorical columns
4. Applied StandardScaler normalization to numerical columns

## Performance Results

### 5-Fold Cross-Validation Performance
Based on the best_results.json file, the model achieved the following performance metrics:
- Mean ROC-AUC Score: 0.8019
- Standard Deviation of ROC-AUC: 0.0073
- Individual Fold Scores:
  - Fold 1: 0.7997
  - Fold 2: 0.8141
  - Fold 3: 0.7938
  - Fold 4: 0.7963
  - Fold 5: 0.8057

### Best Hyperparameters
```
{
  "max_depth": 6,
  "n_estimators": 280,
  "subsample": 0.6944,
  "colsample_bytree": 0.601,
  "reg_alpha": 0.5366,
  "reg_lambda": 2.848,
  "min_child_weight": 4,
  "gamma": 0.4342
}
```

### Test Dataset Results
The trained model was applied to the test dataset, and the prediction scores are saved in bank_marketing_test_scores.csv. This file contains 8000 probability scores for the test samples.

## File Descriptions
- **bank_marketing_test_scores.csv**: Contains prediction scores for the test dataset (8000 rows of probability scores)
- **best_results.json**: Contains the best hyperparameters and 5-fold CV performance metrics
- **README.md**: This documentation file
- **submit/**: Directory containing submission files
  - **optimized_xgboost_final.pkl**: Serialized trained XGBoost model
  - **xgboost_roc_curve.png**: Visualization of the model's ROC curve
- **xgboost_predictor.py**: Python script used for making predictions with the trained model
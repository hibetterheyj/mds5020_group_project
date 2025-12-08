# Logistic Regression Hyperparameter Tuning Summary

## Experiment Setup

- **Baseline Model**: Logistic Regression with TF-IDF features (max_features=5000)
- **Baseline Performance**: Mean Weighted F1 Score = 0.8024
- **Tuning Method**: RandomizedSearchCV with 500 iterations per configuration
- **Cross-Validation**: 5-fold StratifiedKFold
- **Evaluation Metric**: Weighted F1 Score

## Key Parameters Tuned

1. **TF-IDF Parameters**:
   - `max_features`: Number of top features to keep
   - `ngram_range`: Range of n-grams to consider

2. **Logistic Regression Parameters**:
   - `C`: Inverse of regularization strength
   - `penalty`: Regularization type (L1 or L2)
   - `solver`: Optimization algorithm
   - `max_iter`: Maximum number of iterations
   - `class_weight`: Class weighting strategy

3. **Feature Configuration**:
   - `use_handcrafted`: Whether to include handcrafted features

## Results

### Overall Best Result
- **Mean F1 Score**: 0.8093 (Improvement: +0.0069 over baseline)
- **Parameters**:

```json
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
```

### Comparison: Handcrafted vs. Non-Handcrafted Features

| Configuration | Best Mean F1 Score | Improvement Over Baseline |
|---------------|--------------------|---------------------------|
| With Handcrafted Features | 0.8075 | +0.0051 |
| Without Handcrafted Features | 0.8093 | +0.0069 |
| **Difference** | **-0.0017** | |

## Insights

1. **Performance Improvement**: The tuned model achieved a mean F1 score of 0.8093, which exceeds the baseline of 0.8024 by approximately 0.7%.

2. **Handcrafted Features Impact**:
   - Contrary to expectation, the model without handcrafted features performed slightly better than with handcrafted features.
   - This suggests that the handcrafted features might not be adding significant value beyond what the TF-IDF features already capture.
   - Possible reasons: redundancy between TF-IDF and handcrafted features, or the handcrafted features introducing noise.

3. **Optimal Parameters**:
   - **TF-IDF**: max_features=3000 and ngram_range=(1, 2) performed better than higher max_features settings.
   - **Logistic Regression**: L1 penalty (sparse solution) with C=4.2813 and no class weighting yielded the best results.
   - **Solver**: liblinear solver worked well with L1 penalty.

4. **Class Imbalance Handling**:
   - Interestingly, the best model did not use class weighting (`class_weight=None`), despite the dataset showing imbalance (2805 positive, 1495 negative examples).

## Recommendations

1. **Model Selection**: The tuned Logistic Regression model without handcrafted features is recommended for deployment due to its slightly better performance and simpler architecture.

2. **Parameter Focus**: For future tuning, focus on:
   - Further optimizing TF-IDF parameters (ngram_range, max_features)
   - Exploring different text preprocessing techniques
   - Testing alternative feature extraction methods (e.g., word embeddings)

3. **Handcrafted Features**: Consider:
   - Re-evaluating the handcrafted features to ensure they provide unique information
   - Testing feature selection methods to identify the most valuable handcrafted features
   - Exploring different combinations of handcrafted and TF-IDF features

## Files Generated

- `logregression_tuning_results.json`: Full tuning results with all parameter combinations and their performance metrics
- `logregression_tuning_summary.md`: This summary report

## Conclusion

The hyperparameter tuning experiment successfully improved the Logistic Regression model performance beyond the baseline, achieving a mean weighted F1 score of 0.8093. Interestingly, the inclusion of handcrafted features did not improve the model performance compared to using only TF-IDF features. This suggests that TF-IDF alone is sufficient for capturing the relevant information in the text data for this sentiment analysis task.

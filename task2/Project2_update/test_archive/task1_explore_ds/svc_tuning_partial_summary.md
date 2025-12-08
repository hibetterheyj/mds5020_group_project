# SVC Hyperparameter Tuning - Partial Summary

**Note:** Poly kernel tuning is still in progress.

## Experiment Setup

- **Baseline**: Logistic Regression with TF-IDF features (0.8024 F1 score)
- **Tuning Method**: RandomizedSearchCV with 200 iterations per configuration
- **Cross-Validation**: 5-fold StratifiedKFold
- **Evaluation Metric**: Weighted F1 Score

## Key Parameters Tuned

1. **TF-IDF Parameters**:
   - `max_features`: Number of top features to keep
   - `ngram_range`: Range of n-grams to consider

2. **SVC Parameters**:
   - `C`: Regularization parameter
   - `class_weight`: Class weighting strategy
   - `kernel`: Kernel type (for non-linear SVC)
   - `gamma`: Kernel coefficient
   - `coef0`: Independent term in kernel function

3. **Feature Configuration**:
   - `use_handcrafted`: Whether to include handcrafted features

## Best Results by Classifier

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

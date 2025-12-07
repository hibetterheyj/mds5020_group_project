# Sentiment Analysis Model Improvement Report

## 1. Project Overview

This report analyzes the sentiment analysis task performance and provides improvement recommendations. The original model achieved an average weighted F1-score of **0.8024** across 5-fold cross-validation. Our goal was to:
- Perform EDA on the training dataset
- Analyze correctly and incorrectly classified samples
- Test alternative models (XGBoost, LightGBM, SVM, Random Forest)
- Tune hyperparameters of the best-performing model
- Provide actionable improvement recommendations

## 2. Data Analysis Summary

### 2.1 Dataset Overview
- **Dataset**: News title sentiment analysis
- **Size**: 4300+ samples
- **Labels**: Binary sentiment (-1 = negative, 1 = positive)
- **Class Distribution**: Positive class dominance (~65% positive, ~35% negative)

### 2.2 Key Features
- **Text Length**: Average 12-15 words per news title
- **Vocabulary Size**: ~5000 unique words after preprocessing
- **Class Imbalance**: Requires balanced class weights in models

### 2.3 Preprocessing Steps
- Lowercasing
- Punctuation and digit removal
- Stopword removal
- Tokenization and filtering

## 3. Model Performance Comparison

### 3.1 Baseline Performance
- **Model**: Logistic Regression
- **Original F1-Score**: 0.8024
- **Standard Deviation**: 0.0151

### 3.2 Alternative Model Comparison
We evaluated five different models with identical TF-IDF feature extraction:

| Model | Mean F1-Score | Standard Deviation |
|-------|---------------|--------------------|
| Logistic Regression | **0.8024** | 0.0151 |
| SVM | 0.7985 | 0.0128 |
| Random Forest | 0.7712 | 0.0143 |
| XGBoost | N/A | N/A |
| LightGBM | N/A | N/A |

**Note**: XGBoost and LightGBM encountered technical issues with the label format (-1/1 vs 0/1) despite attempted fixes.

## 4. Error Analysis Findings

### 4.1 Misclassified Samples Analysis
- **Total Misclassified**: 154 samples (out of 4300+)
- **Confusion Matrix Trends**:
  - More negative samples misclassified as positive (~60% of errors)
  - Common issues: ambiguous language, sarcasm, domain-specific terminology

### 4.2 Error Patterns
1. **Ambiguous Phrases**: Titles with neutral language that lean towards one sentiment
2. **Domain Specificity**: Financial/economic terms that carry nuanced sentiment
3. **Sarcasm/Irony**: Hard to detect without context
4. **Short Texts**: Very short titles with limited context

## 5. Hyperparameter Tuning Results

### 5.1 Tuning Parameters
We optimized the following parameters for Logistic Regression:
- TF-IDF max features
- N-gram range
- Minimum/maximum document frequency
- Regularization strength (C)
- Penalty type (L1/L2)

### 5.2 Best Parameters Found
```
{
  "model__C": 10.0,
  "model__penalty": "l2",
  "model__solver": "liblinear",
  "tfidf__max_df": 0.85,
  "tfidf__max_features": 10000,
  "tfidf__min_df": 1,
  "tfidf__ngram_range": (1, 1)
}
```

### 5.3 Tuning Results
- **Improved F1-Score**: 0.8040 (+0.0016 improvement)
- **Standard Deviation**: 0.0112 (reduced variability)

## 6. Improvement Recommendations

### 6.1 Immediate Actions (Low Effort, High Impact)

#### 6.1.1 Parameter Optimization
- **Implement tuned parameters**: Use the optimized hyperparameters found in our tuning process
- **Key changes**:
  - Increase TF-IDF max features from 5000 to 10000
  - Set C=10.0 for stronger regularization
  - Use max_df=0.85 to filter more frequent terms

#### 6.1.2 Preprocessing Enhancements
- **Expand stopword list**: Add domain-specific stopwords (e.g., financial terms with neutral sentiment)
- **Preserve negation context**: Avoid splitting "not good" into separate tokens
- **Lemmatization**: Reduce words to their base form (e.g., "running" → "run")

### 6.2 Medium Effort Actions

#### 6.2.1 Advanced Feature Engineering
- **N-gram expansion**: Try (1,2) and (1,3) n-grams to capture phrase-level sentiment
- **Sentiment lexicons**: Integrate pre-trained sentiment lexicons (VADER, TextBlob)
- **Word embeddings**: Consider pre-trained word embeddings (Word2Vec, GloVe) instead of TF-IDF

#### 6.2.2 Class Imbalance Handling
- **Oversampling minority class**: Use SMOTE (Synthetic Minority Oversampling Technique)
- **Class weight adjustment**: Fine-tune class weights beyond 'balanced'
- **Ensemble methods**: Try AdaBoost or Gradient Boosting with class weights

### 6.3 Long Term Actions (Higher Effort, Potential for Significant Improvement)

#### 6.3.1 Model Architecture Exploration
- **Neural networks**: Try LSTM/GRU networks for sequence modeling
- **Transformers**: Explore pre-trained models like BERT, RoBERTa for better contextual understanding
- **Ensemble models**: Combine multiple classifiers for improved robustness

#### 6.3.2 Data Augmentation
- **Text generation**: Use synonym replacement, back-translation for data augmentation
- **Active learning**: Identify difficult samples and acquire more labeled data
- **Domain adaptation**: Fine-tune on financial news domain if possible

#### 6.3.3 Error Analysis Integration
- **Categorize errors**: Develop a taxonomy of common error types
- **Focus training**: Weight misclassified samples more heavily during training
- **Post-processing rules**: Add simple rules to fix common error patterns

## 7. Implementation Plan

### 7.1 Phase 1: Quick Wins (1-2 days)
1. Implement the tuned hyperparameters
2. Enhance preprocessing with negation handling and lemmatization
3. Re-evaluate model performance

### 7.2 Phase 2: Advanced Features (3-5 days)
1. Add n-gram features (1,2) and (1,3)
2. Integrate sentiment lexicons
3. Experiment with SMOTE for class imbalance

### 7.3 Phase 3: Advanced Models (1-2 weeks)
1. Implement Word2Vec/GloVe embeddings
2. Test LSTM/GRU models
3. Explore BERT/RoBERTa fine-tuning

## 8. Expected Outcomes

By implementing the recommended improvements, we expect:
- **F1-Score Improvement**: Target 0.83-0.85 with Phase 1 & 2 implementations
- **Reduced Variability**: More consistent performance across cross-validation folds
- **Better Minority Class Performance**: Improved F1-score for negative sentiment class

## 9. Conclusion

The current sentiment analysis model shows reasonable performance (0.8024 F1-score), but there are several actionable ways to improve it:

1. **Parameter tuning** provided modest improvement (0.8040 F1-score)
2. **Alternative models** like SVM show comparable performance but no significant improvement
3. **Preprocessing enhancements** and **feature engineering** offer the most immediate potential gains
4. **Advanced models** (transformers) could provide the biggest performance leap with higher implementation effort

We recommend starting with the quick wins (parameter optimization and preprocessing enhancements) before moving to more complex solutions.

---

**Report Generated**: 2024-05-07
**Author**: MDS5020 Group Project Team
**Location**: `task1_explore/SENTIMENT_ANALYSIS_IMPROVEMENT_REPORT.md`
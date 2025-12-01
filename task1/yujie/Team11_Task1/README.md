This is the Final Project Task1 file of Team11.
The teammates in Team11 are: Zichang Lan 225040051  He Yujie 225040114  Mingyu LEI 225040082    Runjin Zhang 225040050  Luo Zitai 225040095.


We use XGBoost method to train the banking training dataset.
We first drop the duplicate rows of the training dataset and divide columns into 2 different groups: Categorical columns and Numerical columns
For Categorical columns, we use one-hot encoding and for Numerical columns we use Standardscalar.

After 5-fold CV, we get:
{
    Best ROC-AUC: 0.8014
    Training ROC-AUC: 0.8203
    Best Parameters:{'colsample_bytree': 0.8404460046972835, 'gamma': 0.35403628889802274, 'learning_rate': 0.016175348288740735, 'max_depth': 4, 'min_child_weight': 8, 'n_estimators': 393, 'reg_alpha': 0.0007787658410143283, 'reg_lambda': 2.984423118582435, 'subsample': 0.8469926038510867}
}
Model saved in: optimized_xgboost_final.pkl
ROC curve saved in: xgboost_roc_curve.png

Then we use this model to predict. The result file is saved in predictions_xgboost_final.csv, and we list predicted labels and the corresponding scores.
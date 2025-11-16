import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

class KNNModel:
    """KNN classifier with hyperparameter tuning and cross-validation"""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.cv_auc_scores = None

    def tune_hyperparameters(self, X_train, y_train, cv_folds=5):
        """Find best k parameter using cross-validation"""
        print("Tuning KNN hyperparameters...")

        # Define parameter grid for k
        k_values = [3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        best_score = 0
        best_k = 5

        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            cv_scores = cross_val_score(knn, X_train, y_train,
                                      cv=cv, scoring='roc_auc', n_jobs=-1)
            mean_score = np.mean(cv_scores)

            print(f"k={k}: Mean AUC = {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        self.best_params = {'n_neighbors': best_k}
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation AUC: {best_score:.4f}")

        return best_k

    def train(self, X_train, y_train, cv_folds=5):
        """Train KNN model with best parameters"""
        print("Training KNN model...")

        # Find best k
        best_k = self.tune_hyperparameters(X_train, y_train, cv_folds)

        # Train final model with best parameters
        self.model = KNeighborsClassifier(
            n_neighbors=best_k,
            weights='distance'  # Use distance-weighted voting
        )
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                           cv=cv, scoring='roc_auc', n_jobs=-1)

        print(f"Final model trained with k={best_k}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

        return self.model

    def predict_proba(self, X_test):
        """Predict probability scores for positive class"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Predicting probabilities for test data...")
        # Get probabilities for positive class (y='yes')
        positive_proba = self.model.predict_proba(X_test)[:, 1]

        # Ensure probabilities are between 0 and 1
        positive_proba = np.clip(positive_proba, 0, 1)

        print(f"Predicted probabilities range: [{positive_proba.min():.4f}, {positive_proba.max():.4f}]")
        return positive_proba

    def get_cv_performance(self):
        """Return cross-validation performance metrics"""
        if self.cv_auc_scores is None:
            raise ValueError("No cross-validation results available")

        return {
            'mean_auc': np.mean(self.cv_auc_scores),
            'std_auc': np.std(self.cv_auc_scores),
            'all_scores': self.cv_auc_scores.tolist()
        }
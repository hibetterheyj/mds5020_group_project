import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from hyperparameter_tuner import HyperparameterTuner

class LinearSVMModel:
    """LinearSVC classifier with hyperparameter tuning and cross-validation

    Supports different penalties (l1, l2) and loss functions.
    Optimized for linear classification tasks.
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.cv_auc_scores = None

    def tune_hyperparameters(self, X_train, y_train, cv_folds=5,
                           save_results=False, results_file_path=None, export_format='csv'):
        """Find best hyperparameters using cross-validation

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            save_results: Whether to save tuning results
            results_file_path: Path to save results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Best hyperparameters
        """
        # Initialize hyperparameter tuner
        self.tuner = HyperparameterTuner()

        # Define valid parameter combinations for LinearSVC
        # For l1 penalty, must use dual=False
        # For hinge loss with l2 penalty, must use dual=True
        param_grid = {
            'penalty': ['l2', 'l1'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
            'dual': [True]  # For l2 penalty with hinge loss, dual must be True
        }

        # Tune hyperparameters using GridSearchCV for better efficiency
        print("Tuning LinearSVC hyperparameters with GridSearchCV...")
        best_params, best_score = self.tuner.tune_parameters_with_gridsearch(
            X_train=X_train,
            y_train=y_train,
            model_constructor=LinearSVC,
            param_grid=param_grid,
            cv_folds=cv_folds,
            scoring='roc_auc',
            n_jobs=-2,  # Use all but one CPU cores
            save_results=save_results,
            results_file_path=results_file_path,
            export_format=export_format
        )

        # Store best parameters
        self.best_params = best_params

        return best_params

    def train(self, X_train, y_train, cv_folds=5,
             save_results=True, results_file_path=None, export_format='csv'):
        """Train LinearSVC model with best parameters

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            save_results: Whether to save tuning results (default: True)
            results_file_path: Path to save results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Trained model
        """
        print("Training LinearSVC model...")

        # Set default results file path if not provided
        if save_results and results_file_path is None:
            results_file_path = f"linear_svm_tuning_results.{export_format}"
            print(f"Using default results path: {results_file_path}")

        # Find best parameters
        best_params = self.tune_hyperparameters(X_train, y_train, cv_folds,
                                             save_results=save_results,
                                             results_file_path=results_file_path,
                                             export_format=export_format)

        # Ensure parameter combinations are valid
        # For l2 penalty with hinge loss, dual must be True
        # We've already handled this in the parameter grid

        # Train final model with best parameters
        self.model = LinearSVC(**best_params)
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # For LinearSVC, we need to use decision_function for scoring
        def custom_scorer(estimator, X, y):
            from sklearn.metrics import roc_auc_score
            decision_scores = estimator.decision_function(X)
            return roc_auc_score(y, decision_scores)

        self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                           cv=cv, scoring=custom_scorer, n_jobs=-1)

        print(f"Final model trained with parameters: {best_params}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

        return self.model

    def predict_proba(self, X_test):
        """Predict probability scores for positive class

        For LinearSVC, we use the decision function output scaled to [0, 1]
        using sigmoid function.

        Args:
            X_test: Test features

        Returns:
            Probability scores for the positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Predicting scores for test data...")

        # For LinearSVC, use decision function and scale to [0, 1] using sigmoid
        decision_scores = self.model.decision_function(X_test)
        positive_proba = 1 / (1 + np.exp(-decision_scores))

        # Ensure scores are between 0 and 1
        positive_proba = np.clip(positive_proba, 0, 1)

        print(f"Predicted scores range: [{positive_proba.min():.4f}, {positive_proba.max():.4f}]")
        return positive_proba

    def get_cv_performance(self):
        """Return cross-validation performance metrics

        Returns:
            Dictionary with mean AUC, std AUC, and individual scores
        """
        if self.cv_auc_scores is None:
            raise ValueError("No cross-validation results available")

        return {
            'mean_auc': np.mean(self.cv_auc_scores),
            'std_auc': np.std(self.cv_auc_scores),
            'all_scores': self.cv_auc_scores.tolist()
        }

    def visualize_hyperparameter_tuning(self, output_path=None, save_results=False,
                                       results_output_path=None, export_format='csv'):
        """Visualize the hyperparameter tuning results using the HyperparameterTuner

        Args:
            output_path: Path to save the visualization
            save_results: Whether to save the results
            results_output_path: Path to save the results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Matplotlib figure object
        """
        if not hasattr(self, 'tuner'):
            raise ValueError("No tuning results available. Run tune_hyperparameters first.")

        # Use the tuner's visualization method
        return self.tuner.visualize_tuning_results(
            output_path=output_path,
            save_results=save_results,
            results_output_path=results_output_path,
            export_format=export_format,
            metric_name='AUC'
        )
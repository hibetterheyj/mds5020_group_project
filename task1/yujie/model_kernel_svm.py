import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from hyperparameter_tuner import HyperparameterTuner

class KernelSVMModel:
    """SVC classifier with hyperparameter tuning and cross-validation

    Supports RBF and Sigmoid kernels for non-linear classification.
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

        # Define parameter grid for SVC with RBF and Sigmoid kernels
        param_grid = {
            'kernel': ['rbf', 'sigmoid'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'probability': [True]  # Enable probability estimates
        }

        # Tune hyperparameters
        print("Tuning SVC hyperparameters...")
        best_params, best_score = self.tuner.tune_parameters(
            X_train=X_train,
            y_train=y_train,
            model_constructor=SVC,
            param_grid=param_grid,
            cv_folds=cv_folds,
            scoring='roc_auc',
            save_results=save_results,
            results_file_path=results_file_path,
            export_format=export_format
        )

        # Store best parameters
        self.best_params = best_params

        return best_params

    def train(self, X_train, y_train, cv_folds=5,
             save_results=False, results_file_path=None, export_format='csv'):
        """Train SVC model with best parameters

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            save_results: Whether to save tuning results
            results_file_path: Path to save results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Trained model
        """
        print("Training SVC model...")

        # Find best parameters
        best_params = self.tune_hyperparameters(X_train, y_train, cv_folds,
                                             save_results=save_results,
                                             results_file_path=results_file_path,
                                             export_format=export_format)

        # Train final model with best parameters
        self.model = SVC(**best_params)
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                           cv=cv, scoring='roc_auc', n_jobs=-1)

        print(f"Final model trained with parameters: {best_params}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

        return self.model

    def predict_proba(self, X_test):
        """Predict probability scores for positive class

        Args:
            X_test: Test features

        Returns:
            Probability scores for the positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Predicting scores for test data...")

        # For SVC, use predict_proba directly
        positive_proba = self.model.predict_proba(X_test)[:, 1]

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
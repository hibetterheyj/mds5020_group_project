import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from hyperparameter_tuner import HyperparameterTuner

class SVMModel:
    """SVM classifier with hyperparameter tuning and cross-validation

    Supports both SVC with different kernels and LinearSVC with different penalties.
    """

    def __init__(self):
        self.model = None
        self.best_params = None
        self.cv_auc_scores = None
        self.model_type = None  # 'svc' or 'linear_svc'

    def tune_hyperparameters(self, X_train, y_train, cv_folds=5, model_type='svc',
                           save_results=False, results_file_path=None, export_format='csv'):
        """Find best hyperparameters using cross-validation

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            model_type: Type of SVM model ('svc' or 'linear_svc')
            save_results: Whether to save tuning results
            results_file_path: Path to save results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Best hyperparameters
        """
        # Initialize hyperparameter tuner
        self.tuner = HyperparameterTuner()
        self.model_type = model_type

        # Define parameter grid based on model type
        if model_type == 'svc':
            # SVC with RBF and Sigmoid kernels
            param_grid = {
                'kernel': ['rbf', 'sigmoid'],
                # 'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                # 'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'probability': [True]  # Enable probability estimates
            }
            # Best parameters: {'kernel': 'rbf', 'C': 1000, 'gamma':  1e-05, 'probability': True, 'random_state': 42} Best cross-validation roc_auc: 0.7669
            model_constructor = SVC
        elif model_type == 'linear_svc':
            # LinearSVC with different penalties and loss functions
            param_grid = {
                'penalty': ['l1', 'l2'],
                'loss': ['hinge', 'squared_hinge'],
                # 'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'dual': [False]  # Use primal form for better performance with n_samples > n_features
            }
            model_constructor = LinearSVC
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'svc' or 'linear_svc'")

        # Tune hyperparameters
        print(f"Tuning {model_type.upper()} hyperparameters...")
        best_params, best_score = self.tuner.tune_parameters(
            X_train=X_train,
            y_train=y_train,
            model_constructor=model_constructor,
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

    def train(self, X_train, y_train, cv_folds=5, model_type='svc',
             save_results=False, results_file_path=None, export_format='csv'):
        """Train SVM model with best parameters

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            model_type: Type of SVM model ('svc' or 'linear_svc')
            save_results: Whether to save tuning results
            results_file_path: Path to save results file
            export_format: Format to export results ('csv' or 'json')

        Returns:
            Trained model
        """
        print(f"Training {model_type.upper()} model...")

        # Find best parameters
        best_params = self.tune_hyperparameters(X_train, y_train, cv_folds, model_type,
                                             save_results=save_results,
                                             results_file_path=results_file_path,
                                             export_format=export_format)

        # Create appropriate model constructor
        if model_type == 'svc':
            model_constructor = SVC
        elif model_type == 'linear_svc':
            model_constructor = LinearSVC
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'svc' or 'linear_svc'")

        # Train final model with best parameters
        # For LinearSVC, ensure dual is set appropriately
        if model_type == 'linear_svc' and best_params.get('penalty') == 'l1':
            best_params['dual'] = False

        # random_state is already added in hyperparameter_tuner.py, avoid duplication
        self.model = model_constructor(**best_params)
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # For LinearSVC, we need to use decision_function for scoring
        if model_type == 'linear_svc':
            def custom_scorer(estimator, X, y):
                from sklearn.metrics import roc_auc_score
                decision_scores = estimator.decision_function(X)
                return roc_auc_score(y, decision_scores)

            self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                               cv=cv, scoring=custom_scorer, n_jobs=-1)
        else:
            self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                               cv=cv, scoring='roc_auc', n_jobs=-1)

        print(f"Final model trained with parameters: {best_params}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

        return self.model

    def predict_proba(self, X_test):
        """Predict probability scores for positive class

        For LinearSVC, we use the decision function output scaled to [0, 1]
        For SVC, we use the built-in predict_proba method

        Args:
            X_test: Test features

        Returns:
            Probability scores for the positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Predicting scores for test data...")

        if self.model_type == 'linear_svc':
            # For LinearSVC, use decision function and scale to [0, 1]
            decision_scores = self.model.decision_function(X_test)
            # Apply sigmoid to convert to probability-like scores
            positive_proba = 1 / (1 + np.exp(-decision_scores))
        else:
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
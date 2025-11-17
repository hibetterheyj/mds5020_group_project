import numpy as np
from typing import Dict, Optional, List, Any, Union, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from hyperparameter_tuner import HyperparameterTuner


class KNNModel:
    """KNN classifier with hyperparameter tuning and cross-validation"""

    def __init__(self) -> None:
        self.model: Optional[KNeighborsClassifier] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_auc_scores: Optional[np.ndarray] = None

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5,
                             save_results: bool = False, results_file_path: Optional[str] = None,
                             export_format: str = 'csv') -> Tuple[int, str, int]:
        """Find best k parameter, weights parameter, and p parameter using GridSearchCV for improved efficiency"""
        # Initialize hyperparameter tuner
        self.tuner = HyperparameterTuner()

        # Define parameter grid
        param_grid = {
            # List of k values to try
            'k_values': [i for i in range(3, 40, 2)],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1: Manhattan distance, p=2: Euclidean distance
        }

        # Tune hyperparameters using GridSearchCV for better efficiency
        print("Tuning KNN hyperparameters with GridSearchCV...")
        best_params, best_score = self.tuner.tune_parameters_with_gridsearch(
            X_train=X_train,
            y_train=y_train,
            model_constructor=KNeighborsClassifier,
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

        # Extract individual parameters for backward compatibility
        best_k = best_params.get('n_neighbors', 5)
        best_weights = best_params.get('weights', 'distance')
        best_p = best_params.get('p', 2)

        return best_k, best_weights, best_p

    def train(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5,
              save_results: bool = True, results_file_path: Optional[str] = None,
              export_format: str = 'csv') -> KNeighborsClassifier:
        """Train KNN model with best parameters"""
        print("Training KNN model...")

        # Set default results file path if not provided
        if save_results and results_file_path is None:
            results_file_path = f"knn_tuning_results.{export_format}"
            print(f"Using default results path: {results_file_path}")

        # Find best k, weights, and p
        best_k, best_weights, best_p = self.tune_hyperparameters(X_train, y_train, cv_folds,
                                                                 save_results=save_results,
                                                                 results_file_path=results_file_path,
                                                                 export_format=export_format)

        # Train final model with best parameters
        self.model = KNeighborsClassifier(
            n_neighbors=best_k,
            weights=best_weights,
            p=best_p  # Use best p from tuning
        )
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                             cv=cv, scoring='roc_auc', n_jobs=-1)

        print(
            f"Final model trained with k={best_k}, weights={best_weights}, p={best_p}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(
            f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

        return self.model

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict probability scores for positive class"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("Predicting probabilities for test data...")
        # Get probabilities for positive class (y='yes')
        positive_proba = self.model.predict_proba(X_test)[:, 1]

        # Ensure probabilities are between 0 and 1
        positive_proba = np.clip(positive_proba, 0, 1)

        print(
            f"Predicted probabilities range: [{positive_proba.min():.4f}, {positive_proba.max():.4f}]")
        return positive_proba

    def get_cv_performance(self) -> Dict[str, Union[float, List[float]]]:
        """Return cross-validation performance metrics"""
        if self.cv_auc_scores is None:
            raise ValueError("No cross-validation results available")

        return {
            'mean_auc': np.mean(self.cv_auc_scores),
            'std_auc': np.std(self.cv_auc_scores),
            'all_scores': self.cv_auc_scores.tolist()
        }

    def visualize_hyperparameter_tuning(self, output_path: Optional[str] = None, save_results: bool = False,
                                        results_output_path: Optional[str] = None, export_format: str = 'csv') -> Any:
        """Visualize the hyperparameter tuning results using the HyperparameterTuner"""
        if not hasattr(self, 'tuner'):
            raise ValueError(
                "No tuning results available. Run tune_hyperparameters first.")

        # Use the tuner's visualization method
        return self.tuner.visualize_tuning_results(
            output_path=output_path,
            save_results=save_results,
            results_output_path=results_output_path,
            export_format=export_format,
            metric_name='AUC'
        )

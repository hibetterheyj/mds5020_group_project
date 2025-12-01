import numpy as np
from typing import Dict, Optional, List, Any, Union, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from hyperparameter_tuner import HyperparameterTuner


class XGBoostModel:
    """XGBoost classifier with hyperparameter tuning and cross-validation"""

    def __init__(self) -> None:
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_auc_scores: Optional[np.ndarray] = None
        self.tuner: Optional[HyperparameterTuner] = None

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5,
                             save_results: bool = False, results_file_path: Optional[str] = None,
                             export_format: str = 'csv') -> Dict[str, Any]:
        """Find best XGBoost parameters using GridSearchCV for improved efficiency"""
        # Initialize hyperparameter tuner
        self.tuner = HyperparameterTuner()

        # Define parameter grid for XGBoost with optimized ranges
        # Focus on tuning learning_rate, max_depth, and n_estimators
        # Other parameters are set to optimal values from previous runs
        param_grid = {
            # Learning rate from 0.010 to 0.040 with 0.005 increments (9 values)
            'learning_rate': [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040],
            # Max depth values 5-8 (4 values)
            'max_depth': [5, 6, 7, 8],
            # Number of estimators from 280 to 350 with 10 increments (8 values)
            'n_estimators': [280, 290, 300, 310, 320, 330, 340, 350],
            # Fixed parameters based on optimal values from previous tuning
            'subsample': [0.6944466],
            'colsample_bytree': [0.601],
            'reg_alpha': [0.5366],
            'reg_lambda': [2.8480],
            'min_child_weight': [4],
            'gamma': [0.434150]
        }

        # Tune hyperparameters using GridSearchCV
        print("Tuning XGBoost hyperparameters with GridSearchCV...")
        best_params, best_score = self.tuner.tune_parameters_with_gridsearch(
            X_train=X_train,
            y_train=y_train,
            model_constructor=xgb.XGBClassifier,
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = 5,
              save_results: bool = True, results_file_path: Optional[str] = None,
              export_format: str = 'csv') -> xgb.XGBClassifier:
        """Train XGBoost model with best parameters"""
        print("Training XGBoost model...")

        # Set default results file path if not provided
        if save_results and results_file_path is None:
            results_file_path = f"xgboost_tuning_results.{export_format}"
            print(f"Using default results path: {results_file_path}")

        # Find best parameters
        self.best_params = self.tune_hyperparameters(X_train, y_train, cv_folds,
                                                    save_results=save_results,
                                                    results_file_path=results_file_path,
                                                    export_format=export_format)

        # Train final model with best parameters
        self.model = xgb.XGBClassifier(
            **self.best_params,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)

        # Calculate cross-validation scores for reporting
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.cv_auc_scores = cross_val_score(self.model, X_train, y_train,
                                             cv=cv, scoring='roc_auc', n_jobs=-1)

        print(f"Final model trained with best parameters: {self.best_params}")
        print(f"5-fold CV AUC scores: {self.cv_auc_scores}")
        print(f"Mean CV AUC: {np.mean(self.cv_auc_scores):.4f} (+/- {np.std(self.cv_auc_scores):.4f})")

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

        print(f"Predicted probabilities range: [{positive_proba.min():.4f}, {positive_proba.max():.4f}]")
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
        if not hasattr(self, 'tuner') or self.tuner is None:
            raise ValueError("No tuning results available. Run tune_hyperparameters first.")

        # Use the tuner's visualization method
        return self.tuner.visualize_tuning_results(
            output_path=output_path,
            save_results=save_results,
            results_output_path=results_output_path,
            export_format=export_format,
            metric_name='AUC'
        )
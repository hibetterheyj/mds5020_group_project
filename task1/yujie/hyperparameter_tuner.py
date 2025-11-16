from typing import Dict, List, Tuple, Any, Optional, Callable
from itertools import product

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold

class HyperparameterTuner:
    """Generic hyperparameter tuning and visualization utility class"""

    def __init__(self):
        self.tuning_results = {}
        self.best_params = None
        self.best_score = 0
        self.model_constructor = None

    def tune_parameters(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       model_constructor: Callable,
                       param_grid: Dict[str, List[Any]],
                       cv_folds: int = 5,
                       scoring: str = 'roc_auc',
                       n_jobs: int = -1,
                       random_state: int = 42) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using cross-validation

        Args:
            X_train: Training features
            y_train: Training labels
            model_constructor: Function that constructs the model with given parameters
            param_grid: Dictionary of parameter names and their possible values
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to use
            n_jobs: Number of jobs to run in parallel
            random_state: Random state for reproducibility

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Tuning hyperparameters with {cv_folds}-fold cross-validation...")

        # Store model constructor for later use
        self.model_constructor = model_constructor

        # Reset results
        self.tuning_results = {}
        self.best_params = None
        self.best_score = 0

        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Special handling for k_values to ensure it stays as a list
        # Separate k_values from other parameters if present
        k_values = None
        other_params = {}
        
        for name, values in param_grid.items():
            if name == 'k_values':
                k_values = values  # Save k_values as a whole list
            else:
                other_params[name] = values
        
        # Generate combinations only for other parameters
        other_param_names = list(other_params.keys())
        other_param_values_list = [other_params[name] for name in other_param_names]
        
        # Generate all possible parameter combinations for other parameters
        for param_values in product(*other_param_values_list):
            # Create parameter dictionary for this combination
            params = dict(zip(other_param_names, param_values))

            # Create a readable key for this parameter combination
            param_key = ", ".join([f"{name}={value}" for name, value in params.items()])

            # Initialize results storage for this parameter combination
            self.tuning_results[param_key] = {}
            
            # Use provided k_values or default to None (single iteration)
            current_k_values = k_values if k_values is not None else [None]

            # Iterate over k values if applicable
            for k in current_k_values:
                # Create model parameters for this iteration
                model_params = params.copy()
                if k is not None:
                    model_params['n_neighbors'] = k

                try:
                    # Create model with current parameters
                    model = model_constructor(**model_params)

                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train,
                                              cv=cv, scoring=scoring, n_jobs=n_jobs)

                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)

                    # Store results
                    result_key = k if k is not None else 'single'
                    self.tuning_results[param_key][result_key] = {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'scores': cv_scores
                    }

                    # Log progress
                    params_str = ", ".join([f"{name}={value}" for name, value in model_params.items()])
                    print(f"{params_str}: Mean {scoring} = {mean_score:.4f} (+/- {std_score:.4f})")

                    # Update best parameters if current score is better
                    if mean_score > self.best_score:
                        self.best_score = mean_score
                        self.best_params = model_params.copy()

                except Exception as e:
                    print(f"Error with parameters {model_params}: {str(e)}")

        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation {scoring}: {self.best_score:.4f}")

        return self.best_params, self.best_score

    def visualize_tuning_results(self,
                               output_path: Optional[str] = None,
                               save_json: bool = True,
                               json_output_path: Optional[str] = None,
                               metric_name: str = 'AUC') -> plt.Figure:
        """
        Visualize hyperparameter tuning results

        Args:
            output_path: Path to save the visualization
            save_json: Whether to save the results as JSON
            json_output_path: Path to save the JSON results
            metric_name: Name of the metric being visualized

        Returns:
            Matplotlib figure object
        """
        if not self.tuning_results:
            raise ValueError("No tuning results available. Run tune_parameters first.")

        # Save tuning data as JSON if requested
        if save_json:
            self._save_tuning_results(json_output_path)

        # Create visualization
        fig = self._create_visualization(metric_name)

        # Save figure if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter tuning visualization saved to {output_path}")

        plt.tight_layout()
        return fig

    def _save_tuning_results(self, json_output_path: Optional[str] = None) -> None:
        """Save tuning results as JSON"""
        if json_output_path is None:
            json_output_path = "./hyperparameter_tuning_data.json"

        # Prepare data for JSON serialization (convert numpy types)
        json_serializable_results = {}
        for param_key, k_results in self.tuning_results.items():
            json_serializable_results[param_key] = {}
            for k, metrics in k_results.items():
                json_serializable_results[param_key][str(k)] = {
                    'mean_score': float(metrics['mean_score']),
                    'std_score': float(metrics['std_score']),
                    'scores': [float(score) for score in metrics['scores']]
                }

        # Save to JSON file
        with open(json_output_path, 'w') as json_file:
            json.dump(json_serializable_results, json_file, indent=2)
        print(f"Hyperparameter tuning data saved to {json_output_path}")

    def _create_visualization(self, metric_name: str) -> plt.Figure:
        """Create the tuning results visualization"""
        plt.figure(figsize=(12, 6))

        # Define styles for different parameter combinations
        styles = self._get_visualization_styles()

        # Plot results for each parameter combination
        for i, (param_key, k_results) in enumerate(self.tuning_results.items()):
            # Check if we have multiple k values (for KNN-like models)
            has_multiple_ks = len(k_results) > 1 and all(k is not None for k in k_results.keys())

            if has_multiple_ks:
                # For KNN-like models, plot k vs score
                k_values = list(k_results.keys())
                mean_scores = [k_results[k]['mean_score'] for k in k_values]
                std_scores = [k_results[k]['std_score'] for k in k_values]

                # Get style for this parameter combination
                style = styles.get(param_key, {
                    'color': f'C{i}',
                    'marker': ['o', 's', '^', 'D', 'x'][i % 5],
                    'linestyle': ['-', '--', '-.', ':'][i % 4]
                })

                plt.errorbar(k_values, mean_scores, yerr=std_scores,
                            marker=style['marker'],
                            linestyle=style['linestyle'],
                            color=style['color'],
                            label=param_key)

        # Highlight best parameter combination if available
        if self.best_params:
            self._highlight_best_parameters()

        # Add plot details
        plt.title(f'Hyperparameter Tuning Results')
        plt.xlabel('Number of Neighbors (k)' if any('n_neighbors' in params for params in self.tuning_results.values()) else 'Parameter Value')
        plt.ylabel(f'Mean {metric_name} Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        return plt.gcf()

    def _get_visualization_styles(self) -> Dict[str, Dict[str, str]]:
        """Define visualization styles for common parameter combinations"""
        # Default styles for common parameter combinations
        default_styles = {
            'weights=uniform, p=1': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
            'weights=uniform, p=2': {'color': 'blue', 'marker': 's', 'linestyle': '--'},
            'weights=distance, p=1': {'color': 'green', 'marker': '^', 'linestyle': '-'},
            'weights=distance, p=2': {'color': 'green', 'marker': 'D', 'linestyle': '--'}
        }
        return default_styles

    def _highlight_best_parameters(self) -> None:
        """Highlight the best parameter combination on the plot"""
        # Find the parameter combination and k value for the best score
        best_param_key = None
        best_k = None

        # Create a flat dictionary to find the best combination
        for param_key, k_results in self.tuning_results.items():
            for k, metrics in k_results.items():
                if metrics['mean_score'] == self.best_score:
                    best_param_key = param_key
                    best_k = k
                    break
            if best_param_key:
                break

        if best_param_key and best_k is not None:
            best_score = self.tuning_results[best_param_key][best_k]['mean_score']
            plt.scatter([best_k], [best_score], color='red', s=150, marker='*',
                        label=f'Best: k={best_k}, {best_param_key}')

    def get_best_model(self) -> Any:
        """Return a model instance with the best parameters"""
        if not self.model_constructor or not self.best_params:
            raise ValueError("No best parameters available. Run tune_parameters first.")

        return self.model_constructor(**self.best_params)
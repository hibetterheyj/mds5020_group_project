from typing import Dict, List, Tuple, Any, Optional, Callable
from itertools import product
import inspect

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from tqdm_joblib import tqdm_joblib  # 桥接 joblib 和 tqdm，提供进度条支持

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
                       random_state: int = 42,
                       save_results: bool = False,
                       results_file_path: Optional[str] = None,
                       export_format: str = 'csv') -> Tuple[Dict[str, Any], float]:
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

        # Check if we need special handling for k_values (KNN-specific)
        has_k_values = 'k_values' in param_grid

        if has_k_values:
            # KNN-specific handling with k_values
            k_values = param_grid.pop('k_values')
            param_names = list(param_grid.keys())
            param_values_list = [param_grid[name] for name in param_names]

            # Generate all possible parameter combinations
            for param_values in product(*param_values_list):
                params = dict(zip(param_names, param_values))
                param_key = ", ".join([f"{name}={value}" for name, value in params.items()])
                self.tuning_results[param_key] = {}

                for k in k_values:
                    model_params = params.copy()
                    model_params['n_neighbors'] = k
                    self._evaluate_model(X_train, y_train, model_constructor, model_params,
                                        cv, scoring, n_jobs, param_key, k)
        else:
            # General handling for all parameters (works for SVC, LinearSVC, etc.)
            param_names = list(param_grid.keys())
            param_values_list = [param_grid[name] for name in param_names]

            # Generate all possible parameter combinations
            for param_values in product(*param_values_list):
                model_params = dict(zip(param_names, param_values))
                param_key = ", ".join([f"{name}={value}" for name, value in model_params.items()])
                self.tuning_results[param_key] = {}

                # Evaluate with this parameter combination
                self._evaluate_model(X_train, y_train, model_constructor, model_params,
                                    cv, scoring, n_jobs, param_key, 'single')

        # Restore k_values in param_grid for future use if it was present
        if has_k_values:
            param_grid['k_values'] = k_values

        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation {scoring}: {self.best_score:.4f}")

        # Save results if requested
        if save_results:
            if export_format.lower() == 'csv':
                self._save_tuning_results_to_csv(results_file_path)
            else:
                self._save_tuning_results(results_file_path)

        return self.best_params, self.best_score

    def _evaluate_model(self, X_train, y_train, model_constructor, model_params,
                       cv, scoring, n_jobs, param_key, result_key):
        """
        Helper method to evaluate a model with given parameters and store results

        Args:
            X_train: Training features
            y_train: Training labels
            model_constructor: Model constructor function
            model_params: Model parameters
            cv: Cross-validation strategy
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            param_key: Key for parameter combination
            result_key: Key for this specific result
        """
        try:
            # Create model with current parameters
            # Add random_state for reproducibility if supported by the model
            if 'random_state' not in model_params and hasattr(model_constructor, '__name__'):
                if model_constructor.__name__ in ['SVC', 'LinearSVC']:
                    model_params['random_state'] = 42

            model = model_constructor(**model_params)

            # Handle special case for LinearSVC with custom scoring
            if hasattr(model_constructor, '__name__') and model_constructor.__name__ == 'LinearSVC':
                from sklearn.metrics import roc_auc_score

                def custom_scorer(estimator, X, y):
                    decision_scores = estimator.decision_function(X)
                    return roc_auc_score(y, decision_scores)

                cv_scores = cross_val_score(model, X_train, y_train,
                                          cv=cv, scoring=custom_scorer, n_jobs=n_jobs)
            else:
                # Standard cross-validation
                cv_scores = cross_val_score(model, X_train, y_train,
                                          cv=cv, scoring=scoring, n_jobs=n_jobs)

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            # Store results
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

    def visualize_tuning_results(self,
                               output_path: Optional[str] = None,
                               save_results: bool = False,
                               results_output_path: Optional[str] = None,
                               export_format: str = 'csv',
                               metric_name: str = 'AUC') -> plt.Figure:
        """
        Visualize hyperparameter tuning results

        Args:
            output_path: Path to save the visualization
            save_results: Whether to save the results
            results_output_path: Path to save the results file
            export_format: Format to export results ('csv' or 'json')
            metric_name: Name of the metric being visualized

        Returns:
            Matplotlib figure object
        """
        if not self.tuning_results:
            raise ValueError("No tuning results available. Run tune_parameters first.")

        # Save tuning data if requested
        if save_results:
            if export_format.lower() == 'csv':
                self._save_tuning_results_to_csv(results_output_path)
            else:
                self._save_tuning_results(results_output_path)

        # Create visualization
        fig = self._create_visualization(metric_name)

        # Save figure if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter tuning visualization saved to {output_path}")

        plt.tight_layout()
        return fig

    def _save_tuning_results(self, json_output_path: Optional[str] = None) -> None:
        """Save tuning results as JSON with improved structure"""
        if json_output_path is None:
            json_output_path = "./hyperparameter_tuning_data.json"

        # Prepare data in the requested format
        results_list = []
        for param_key, sub_results in self.tuning_results.items():
            for sub_key, metrics in sub_results.items():
                # Create a complete parameter dictionary
                params_dict = {}
                # Parse the param_key string to extract parameter values
                for param in param_key.split(', '):
                    if '=' in param:
                        name, value = param.split('=')
                        # Convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Keep as string if conversion fails
                            pass
                        params_dict[name] = value

                # For KNN, add n_neighbors from sub_key
                if isinstance(sub_key, (int, float)) or (isinstance(sub_key, str) and sub_key.isdigit()):
                    params_dict['n_neighbors'] = int(sub_key)

                # Create a result entry with separate params and metrics
                result_entry = {
                    'parameters': params_dict,
                    'metrics': {
                        'mean_score': float(metrics['mean_score']),
                        'std_score': float(metrics['std_score']),
                        'scores': [float(score) for score in metrics['scores']]
                    }
                }
                results_list.append(result_entry)

        # Save to JSON file
        with open(json_output_path, 'w') as json_file:
            json.dump(results_list, json_file, indent=2)
        print(f"Hyperparameter tuning data saved to {json_output_path}")

    def _save_tuning_results_to_csv(self, csv_output_path: Optional[str] = None) -> None:
        """Save tuning results as CSV for easy pandas analysis"""
        if csv_output_path is None:
            csv_output_path = "./hyperparameter_tuning_data.csv"

        # Prepare data for CSV
        data = []
        for param_key, sub_results in self.tuning_results.items():
            for sub_key, metrics in sub_results.items():
                # Start with metrics
                row = {
                    'mean_score': float(metrics['mean_score']),
                    'std_score': float(metrics['std_score'])
                }

                # Add individual fold scores
                for i, score in enumerate(metrics['scores']):
                    row[f'fold_{i+1}_score'] = float(score)

                # Parse the param_key string to extract parameter values
                for param in param_key.split(', '):
                    if '=' in param:
                        name, value = param.split('=')
                        # Convert to appropriate type
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Keep as string if conversion fails
                            pass
                        row[f'param_{name}'] = value

                # For KNN, add n_neighbors from sub_key
                if isinstance(sub_key, (int, float)) or (isinstance(sub_key, str) and sub_key.isdigit()):
                    row['param_n_neighbors'] = int(sub_key)

                data.append(row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_output_path, index=False)
        print(f"Hyperparameter tuning data saved to {csv_output_path}")
        print(f"Data shape: {df.shape}")

    def _create_visualization(self, metric_name: str) -> plt.Figure:
        """Create the tuning results visualization"""
        plt.figure(figsize=(12, 6))

        # Define styles for different parameter combinations
        styles = self._get_visualization_styles()

        # Determine if this is KNN, SVC, or LinearSVC based on parameter keys
        model_type = self._determine_model_type()

        # Plot results based on model type
        if model_type == 'knn':
            self._plot_knn_results(styles, metric_name)
        elif model_type in ['svc', 'linear_svc']:
            self._plot_svm_results(styles, metric_name, model_type)
        else:
            # Default plotting for any other model type
            self._plot_default_results(styles, metric_name, model_type)

        # Highlight best parameter combination if available
        if self.best_params:
            self._highlight_best_parameters()

        # Add plot details
        plt.title(f'Hyperparameter Tuning Results - {model_type.upper()}')

        # Set appropriate x-label based on model type
        if model_type == 'knn':
            plt.xlabel('Number of Neighbors (k)')
        elif model_type in ['svc', 'linear_svc']:
            plt.xlabel('C Value (log scale)')
            plt.xscale('log')
        else:
            plt.xlabel('Parameter Value')

        plt.ylabel(f'Mean {metric_name} Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        return plt.gcf()

    def _determine_model_type(self) -> str:
        """
        Determine the model type based on parameter keys

        Returns:
            Model type as string ('knn', 'svc', 'linear_svc', or 'other')
        """
        # Check for KNN specific parameters
        if any('weights=' in key or 'p=' in key for key in self.tuning_results.keys()):
            return 'knn'

        # Check for SVC specific parameters
        if any('kernel=' in key for key in self.tuning_results.keys()):
            return 'svc'

        # Check for LinearSVC specific parameters
        if any('penalty=' in key or 'loss=' in key for key in self.tuning_results.keys()):
            return 'linear_svc'

        return 'other'

    def _plot_knn_results(self, styles: Dict[str, Dict[str, str]], metric_name: str) -> None:
        """
        Plot results specifically for KNN models
        x轴为k_values（n_neighbors）从小到大，不同的weights和p组合以不同线型、颜色或标记展示
        """
        for i, (param_key, k_results) in enumerate(self.tuning_results.items()):
            # Check if we have multiple k values
            has_multiple_ks = len(k_results) > 1 and all(k is not None for k in k_results.keys())

            if has_multiple_ks:
                # For KNN-like models, plot k vs score
                # 确保k_values从小到大排序
                k_values = sorted(list(k_results.keys()))
                mean_scores = [k_results[k]['mean_score'] for k in k_values]
                std_scores = [k_results[k]['std_score'] for k in k_values]

                # 获取样式，优先使用预定义样式
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

    def _plot_svm_results(self, styles: Dict[str, Dict[str, str]], metric_name: str, model_type: str) -> None:
        """
        Plot results specifically for SVM models (SVC or LinearSVC)
        x轴为C值从小到大（log scale），不同参数组合以不同线型、颜色或标记展示
        """
        # 直接使用self.tuning_results中的键作为不同参数组合的标识
        for i, (param_key, results) in enumerate(self.tuning_results.items()):
            # 提取C值和对应的分数
            x_values = []
            mean_scores = []
            std_scores = []

            for c_value, metrics in results.items():
                x_values.append(c_value)
                mean_scores.append(metrics['mean_score'])
                std_scores.append(metrics['std_score'])

            # 确保C值从小到大排序
            sorted_pairs = sorted(zip(x_values, mean_scores, std_scores))
            x_values, mean_scores, std_scores = zip(*sorted_pairs)

            # 获取样式，优先使用预定义样式
            style = styles.get(param_key, {
                'color': f'C{i}',
                'marker': ['o', 's', '^', 'D', 'x'][i % 5],
                'linestyle': ['-', '--', '-.', ':'][i % 4]
            })

            plt.errorbar(x_values, mean_scores, yerr=std_scores,
                        marker=style['marker'],
                        linestyle=style['linestyle'],
                        color=style['color'],
                        label=param_key)

    def _plot_default_results(self, styles: Dict[str, Dict[str, str]], metric_name: str, model_type: str) -> None:
        """
        Default plotting for any other model type
        """
        for i, (param_key, results) in enumerate(self.tuning_results.items()):
            # Extract values and scores
            x_values = []
            mean_scores = []
            std_scores = []

            for result_key, metrics in results.items():
                if result_key != 'single':
                    x_values.append(result_key)
                    mean_scores.append(metrics['mean_score'])
                    std_scores.append(metrics['std_score'])

            if x_values:  # Only plot if we have data
                # Get style
                style = styles.get(param_key, {
                    'color': f'C{i}',
                    'marker': ['o', 's', '^', 'D', 'x'][i % 5],
                    'linestyle': ['-', '--', '-.', ':'][i % 4]
                })

                plt.errorbar(x_values, mean_scores, yerr=std_scores,
                            marker=style['marker'],
                            linestyle=style['linestyle'],
                            color=style['color'],
                            label=param_key)

    def _highlight_best_parameters(self) -> None:
        """
        Highlight the best parameter combination on the plot
        """
        # Find the parameter combination and key for the best score
        best_param_key = None
        best_key = None

        # Create a flat dictionary to find the best combination
        for param_key, results in self.tuning_results.items():
            for key, metrics in results.items():
                if metrics['mean_score'] == self.best_score:
                    best_param_key = param_key
                    best_key = key
                    break
            if best_param_key:
                break

        if best_param_key and best_key is not None:
            best_score = self.tuning_results[best_param_key][best_key]['mean_score']

            # 确定要高亮显示的x值
            if isinstance(best_key, (int, float)):
                # 对于KNN，best_key是k值
                plt.scatter([best_key], [best_score], color='red', s=150, marker='*',
                           label=f'Best: k={best_key}, {best_param_key}')
            else:
                # 对于SVM，从best_params中获取C值
                if 'C' in self.best_params:
                    plt.scatter([self.best_params['C']], [best_score], color='red', s=150, marker='*',
                               label=f'Best: C={self.best_params["C"]}, {best_param_key}')

    def _get_visualization_styles(self) -> Dict[str, Dict[str, str]]:
        """Define visualization styles for common parameter combinations"""
        # Default styles for common parameter combinations
        default_styles = {
            # KNN styles
            'weights=uniform, p=1': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
            'weights=uniform, p=2': {'color': 'blue', 'marker': 's', 'linestyle': '--'},
            'weights=distance, p=1': {'color': 'green', 'marker': '^', 'linestyle': '-'},
            'weights=distance, p=2': {'color': 'green', 'marker': 'D', 'linestyle': '--'},
            # SVC styles with gamma variations - using different shades for different gamma values
            'kernel=rbf, gamma=0.00001': {'color': 'darkred', 'marker': 'o', 'linestyle': '-'},
            'kernel=rbf, gamma=0.0001': {'color': 'red', 'marker': 'o', 'linestyle': '--'},
            'kernel=rbf, gamma=0.001': {'color': 'lightcoral', 'marker': 'o', 'linestyle': '-.'},
            'kernel=rbf, gamma=0.01': {'color': 'salmon', 'marker': 'o', 'linestyle': ':'},
            'kernel=rbf, gamma=0.1': {'color': 'indianred', 'marker': 'o', 'linestyle': '-'},
            'kernel=sigmoid, gamma=0.00001': {'color': 'darkpurple', 'marker': 's', 'linestyle': '-'},
            'kernel=sigmoid, gamma=0.0001': {'color': 'purple', 'marker': 's', 'linestyle': '--'},
            'kernel=sigmoid, gamma=0.001': {'color': 'violet', 'marker': 's', 'linestyle': '-.'},
            'kernel=sigmoid, gamma=0.01': {'color': 'plum', 'marker': 's', 'linestyle': ':'},
            'kernel=sigmoid, gamma=0.1': {'color': 'orchid', 'marker': 's', 'linestyle': '-'},
            # Fallback SVC styles
            'kernel=rbf': {'color': 'red', 'marker': 'o', 'linestyle': '-'},
            'kernel=sigmoid': {'color': 'purple', 'marker': 's', 'linestyle': '--'},
            # LinearSVC styles
            'penalty=l1, loss=hinge': {'color': 'orange', 'marker': '^', 'linestyle': '-'},
            'penalty=l1, loss=squared_hinge': {'color': 'orange', 'marker': 'D', 'linestyle': '--'},
            'penalty=l2, loss=hinge': {'color': 'brown', 'marker': 'x', 'linestyle': '-'},
            'penalty=l2, loss=squared_hinge': {'color': 'brown', 'marker': '+', 'linestyle': '--'}
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

    def tune_parameters_with_gridsearch(self,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      model_constructor: Callable,
                                      param_grid: Dict[str, List[Any]],
                                      cv_folds: int = 5,
                                      scoring: str = 'roc_auc',
                                      n_jobs: int = -2,
                                      random_state: int = 42,
                                      save_results: bool = False,
                                      results_file_path: Optional[str] = None,
                                      export_format: str = 'csv') -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using GridSearchCV for improved efficiency

        Args:
            X_train: Training features
            y_train: Training labels
            model_constructor: Function that constructs the model with given parameters
            param_grid: Dictionary of parameter names and their possible values
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to use
            n_jobs: Number of jobs to run in parallel (-2 for using all but one CPU)
            random_state: Random state for reproducibility
            save_results: Whether to save the results
            results_file_path: Path to save results
            export_format: Format to export results

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Tuning hyperparameters with GridSearchCV and {cv_folds}-fold cross-validation...")

        # Store model constructor for later use
        self.model_constructor = model_constructor

        # Reset results
        self.tuning_results = {}
        self.best_params = None
        self.best_score = 0

        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Check if we need special handling for k_values (KNN-specific)
        has_k_values = 'k_values' in param_grid

        # Prepare parameter grid for GridSearchCV
        gs_param_grid = param_grid.copy()

        # Handle KNN-specific parameter naming
        if has_k_values:
            gs_param_grid['n_neighbors'] = gs_param_grid.pop('k_values')

        # Create base model with random_state if supported
        try:
            # Use inspect to check if the constructor accepts random_state parameter
            sig = inspect.signature(model_constructor.__init__)
            if 'random_state' in sig.parameters:
                base_model = model_constructor(random_state=random_state)
            else:
                base_model = model_constructor()
        except (ValueError, TypeError):
            # Fallback if inspect fails
            try:
                base_model = model_constructor(random_state=random_state)
            except TypeError:
                base_model = model_constructor()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=gs_param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=True
        )

        # Handle special case for LinearSVC with custom scoring
        if hasattr(model_constructor, '__name__') and model_constructor.__name__ == 'LinearSVC':
            from sklearn.metrics import roc_auc_score

            def custom_scorer(estimator, X, y):
                decision_scores = estimator.decision_function(X)
                return roc_auc_score(y, decision_scores)

            grid_search.scoring = custom_scorer

        # 计算总参数组合数量用于进度条
        total_combinations = 1
        for param_values in gs_param_grid.values():
            total_combinations *= len(param_values)
        
        # 使用tqdm_joblib添加进度条支持
        with tqdm_joblib(desc=f"GridSearchCV 超参数搜索 - {cv_folds}折交叉验证", 
                         total=total_combinations) as progress_bar:
            grid_search.fit(X_train, y_train)

        # Get best parameters and score
        self.best_params = grid_search.best_params_.copy()
        self.best_score = grid_search.best_score_

        # Format results to be compatible with existing visualization methods
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Organize results in the format expected by visualization methods
        if has_k_values:
            # KNN-specific organization
            # Group by parameters other than n_neighbors
            knn_param_names = [param for param in gs_param_grid.keys() if param != 'n_neighbors']

            for idx, row in results_df.iterrows():
                # Create parameter key excluding k_values
                param_values = {}
                for param in knn_param_names:
                    if param in row['params']:
                        param_values[param] = row['params'][param]

                param_key = ", ".join([f"{name}={value}" for name, value in param_values.items()])

                if param_key not in self.tuning_results:
                    self.tuning_results[param_key] = {}

                # Add k_value and its score
                k_value = row['params']['n_neighbors']
                self.tuning_results[param_key][k_value] = {
                    'mean_score': row['mean_test_score'],
                    'std_score': row['std_test_score'],
                    'scores': [row[f'split{i}_test_score'] for i in range(cv_folds)]
                }
        else:
            # SVC/LinearSVC organization
            # For SVC/LinearSVC, we need to identify the main parameter (C) and group by other parameters
            main_param = 'C' if 'C' in gs_param_grid else list(gs_param_grid.keys())[0]
            other_params = [param for param in gs_param_grid.keys() if param != main_param]

            for idx, row in results_df.iterrows():
                # Create parameter key excluding main parameter
                param_values = {}
                for param in other_params:
                    if param in row['params']:
                        param_values[param] = row['params'][param]

                param_key = ", ".join([f"{name}={value}" for name, value in param_values.items()])

                if param_key not in self.tuning_results:
                    self.tuning_results[param_key] = {}

                # Add main parameter and its score
                main_param_value = row['params'][main_param]
                self.tuning_results[param_key][main_param_value] = {
                    'mean_score': row['mean_test_score'],
                    'std_score': row['std_test_score'],
                    'scores': [row[f'split{i}_test_score'] for i in range(cv_folds)]
                }

        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation {scoring}: {self.best_score:.4f}")

        # Save results if requested
        if save_results:
            if export_format.lower() == 'csv':
                self._save_tuning_results_to_csv(results_file_path)
            else:
                self._save_tuning_results(results_file_path)

        return self.best_params, self.best_score
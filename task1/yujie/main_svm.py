import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from preprocessor import RobustDataPreprocessor as DataPreprocessor
from model_svm import SVMModel
from results_handler import ResultsHandler

def main():
    """Main function to execute the SVM modeling pipeline

    This pipeline follows the same structure as the KNN pipeline but uses SVM models.
    It supports both SVC and LinearSVC models with different hyperparameters.
    """
    print("Bank Marketing Classification - SVM Model")
    print("=" * 50)

    # Initialize paths
    train_path = "../data/bank_marketing_train.csv"
    test_path = "../data/bank_marketing_test.csv"
    output_path = "../tests/bank_marketing_test_scores_svm.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 1. Load data
        data_loader = DataLoader(train_path, test_path)
        train_data, test_data = data_loader.load_data()
        X_train, y_train, X_test = data_loader.prepare_features(train_data, test_data)

        # 2. Preprocess data
        preprocessor = DataPreprocessor()
        X_train_processed = preprocessor.preprocess_train(X_train)
        X_test_processed = preprocessor.preprocess_test(X_test)

        # 3. Train SVM model
        # Uncomment one of the following lines to choose between SVC and LinearSVC

        # Option 1: Use SVC with RBF and Sigmoid kernels
        svm_model = SVMModel()
        svm_model.train(X_train_processed, y_train, cv_folds=5, model_type='svc')

        # Option 2: Use LinearSVC with different penalties
        # svm_model = SVMModel()
        # svm_model.train(X_train_processed, y_train, cv_folds=5, model_type='linear_svc')

        # 4. Generate predictions
        predictions = svm_model.predict_proba(X_test_processed)

        # 5. Save results
        results_handler = ResultsHandler()
        results_handler.save_predictions(predictions, output_path)

        # 6. Generate performance report
        cv_results = svm_model.get_cv_performance()
        results_handler.generate_performance_report(cv_results)

        # 7. Generate and save hyperparameter tuning visualization
        try:
            visualization_path = "../yujie/svm_hyperparameter_tuning.png"
            json_path = "../yujie/svm_hyperparameter_tuning_data.json"
            svm_model.visualize_hyperparameter_tuning(visualization_path, results_output_path=json_path)
            print(f"Hyperparameter tuning visualization generated and saved to {visualization_path}")
        except Exception as viz_error:
            print(f"Warning: Could not generate visualization: {str(viz_error)}")

        print(f"\nPipeline completed successfully!")
        print(f"Output file: {output_path}")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
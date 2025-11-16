import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from preprocessor import RobustDataPreprocessor as DataPreprocessor
from knn_model import KNNModel
from results_handler import ResultsHandler

def main():
    """Main function to execute the KNN modeling pipeline"""
    print("Bank Marketing Classification - KNN Model")
    print("=" * 50)

    # Initialize paths
    train_path = "../data/bank_marketing_train.csv"
    test_path = "../data/bank_marketing_test.csv"
    output_path = "../tests/bank_marketing_test_scores_knn.csv"

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

        # 3. Train KNN model
        knn_model = KNNModel()
        knn_model.train(X_train_processed, y_train, cv_folds=5)

        # 4. Generate predictions
        predictions = knn_model.predict_proba(X_test_processed)

        # 5. Save results
        results_handler = ResultsHandler()
        results_handler.save_predictions(predictions, output_path)

        # 6. Generate performance report
        cv_results = knn_model.get_cv_performance()
        results_handler.generate_performance_report(cv_results)

        # 7. Generate and save hyperparameter tuning visualization
        try:
            visualization_path = "../yujie/knn_hyperparameter_tuning.png"
            json_path = "../yujie/knn_hyperparameter_tuning_data.json"
            knn_model.visualize_hyperparameter_tuning(visualization_path, json_output_path=json_path)
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
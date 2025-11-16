import pandas as pd
import numpy as np

class ResultsHandler:
    """Handle saving and formatting prediction results"""

    def __init__(self):
        pass

    def save_predictions(self, predictions, output_path):
        """Save prediction scores to CSV file"""
        print(f"Saving predictions to {output_path}...")

        # Create DataFrame with predictions
        results_df = pd.DataFrame({
            'ranking_score': predictions
        })

        # Ensure scores are between 0 and 1
        results_df['ranking_score'] = np.clip(results_df['ranking_score'], 0, 1)

        # Save to CSV
        results_df.to_csv(output_path, index=False)

        print(f"Predictions saved successfully. Shape: {results_df.shape}")
        print(f"Final score range: [{results_df['ranking_score'].min():.4f}, {results_df['ranking_score'].max():.4f}]")

        return results_df

    def generate_performance_report(self, cv_results):
        """Generate performance report for submission"""
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"Mean AUC (5-fold CV): {cv_results['mean_auc']:.4f}")
        print(f"Standard Deviation: {cv_results['std_auc']:.4f}")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in cv_results['all_scores']]}")
        print("="*50)
import csv
import json
import requests
import time
import os
from datetime import datetime
import numpy as np

# Configuration
LOGREG_API_URL = 'http://localhost:5724/predict_sentiment'
BERT_API_URL = 'http://localhost:5725/predict_sentiment'
# Use CSV file extension instead of XLSX since we're reading with csv module
DATA_FILE = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.csv'
NUM_SAMPLES = 500

# Create timestamped directory
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'./res/logreg_bert_comparison_{current_time}'
os.makedirs(results_dir, exist_ok=True)

def load_test_data(num_samples=NUM_SAMPLES):
    """Load the first num_samples (default: 500) samples from the training data"""
    test_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as csvfile:
        # Read and clean the header
        header = csvfile.readline().strip().split(',')
        # Remove any potential BOM or invisible characters
        header = [h.strip('"\ufeff') for h in header]

        reader = csv.DictReader(csvfile, fieldnames=header)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            try:
                test_data.append({
                    'doc_id': row[header[0]],
                    'news_title': row[header[1]],
                    'sentiment': int(row[header[2]])
                })
            except (ValueError, KeyError) as e:
                print(f"Error parsing row {i+1}: {e}")
                continue
    return test_data

def test_api(api_url, test_data, model_name):
    """Test an API with the given test data and return predictions and timing"""
    predictions = []
    total_time = 0

    for sample in test_data:
        start_time = time.time()

        try:
            response = requests.post(api_url, json={'news_text': sample['news_title']}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                pred_time = time.time() - start_time
                total_time += pred_time

                predictions.append({
                    'doc_id': sample['doc_id'],
                    'news_title': sample['news_title'],
                    'actual': sample['sentiment'],
                    'predicted': int(result['sentiment']),
                    'probability': float(result['probability']),
                    'prediction_time': pred_time
                })
            else:
                print(f"Error for doc_id {sample['doc_id']}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Exception for doc_id {sample['doc_id']}: {e}")

    return predictions, total_time

def calculate_accuracy(predictions):
    """Calculate accuracy from predictions"""
    correct = 0
    for pred in predictions:
        if pred['actual'] == pred['predicted']:
            correct += 1
    return correct / len(predictions) if predictions else 0

def save_predictions(predictions, filename):
    """Save predictions to a CSV file"""
    if not predictions:
        return

    keys = predictions[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(predictions)

def save_comparison(logreg_preds, bert_preds, logreg_time, bert_time):
    """Save comparison results to JSON"""
    logreg_accuracy = calculate_accuracy(logreg_preds)
    bert_accuracy = calculate_accuracy(bert_preds)

    # Calculate average prediction times
    logreg_avg_time = logreg_time / len(logreg_preds) if logreg_preds else 0
    bert_avg_time = bert_time / len(bert_preds) if bert_preds else 0

    # Calculate prediction time distributions
    logreg_times = [p['prediction_time'] for p in logreg_preds]
    bert_times = [p['prediction_time'] for p in bert_preds]

    comparison = {
        'timestamp': current_time,
        'num_samples': NUM_SAMPLES,
        'logreg': {
            'accuracy': logreg_accuracy,
            'total_prediction_time': logreg_time,
            'average_prediction_time': logreg_avg_time,
            'min_prediction_time': min(logreg_times) if logreg_times else 0,
            'max_prediction_time': max(logreg_times) if logreg_times else 0,
            'median_prediction_time': np.median(logreg_times) if logreg_times else 0
        },
        'bert': {
            'accuracy': bert_accuracy,
            'total_prediction_time': bert_time,
            'average_prediction_time': bert_avg_time,
            'min_prediction_time': min(bert_times) if bert_times else 0,
            'max_prediction_time': max(bert_times) if bert_times else 0,
            'median_prediction_time': np.median(bert_times) if bert_times else 0
        },
        'accuracy_difference': bert_accuracy - logreg_accuracy,
        'time_ratio_bert_vs_logreg': bert_avg_time / logreg_avg_time if logreg_avg_time > 0 else float('inf')
    }

    with open(os.path.join(results_dir, 'comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    return comparison

def evaluate_model_size():
    """Evaluate the BERT model size and check if it meets requirements"""
    model_dir = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/Project2_update/test_archive/task1_bert/results/checkpoint-645'

    # Calculate total size
    total_size = 0
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)

    # Convert to MB and GB
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)

    requirements = {
        'max_file_size_gb': 4,
        'max_memory_mb': 900
    }

    evaluation = {
        'model_directory': model_dir,
        'total_size_mb': round(size_mb, 2),
        'total_size_gb': round(size_gb, 2),
        'meets_file_size_requirement': size_gb <= requirements['max_file_size_gb'],
        'max_allowed_file_size_gb': requirements['max_file_size_gb'],
        'max_allowed_memory_mb': requirements['max_memory_mb'],
        'notes': 'Memory usage will depend on runtime environment and implementation details'
    }

    with open(os.path.join(results_dir, 'model_size_evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    return evaluation

def main():
    print("Loading test data...")
    test_data = load_test_data(NUM_SAMPLES)
    print(f"Loaded {len(test_data)} samples")

    print("\nTesting Logistic Regression API...")
    logreg_preds, logreg_total_time = test_api(LOGREG_API_URL, test_data, 'Logistic Regression')
    print(f"Logistic Regression: {len(logreg_preds)}/{len(test_data)} predictions successful")

    print("\nTesting BERT API...")
    bert_preds, bert_total_time = test_api(BERT_API_URL, test_data, 'BERT')
    print(f"BERT: {len(bert_preds)}/{len(test_data)} predictions successful")

    print("\nSaving predictions...")
    save_predictions(logreg_preds, os.path.join(results_dir, 'logreg_predictions.csv'))
    save_predictions(bert_preds, os.path.join(results_dir, 'bert_predictions.csv'))

    print("\nCalculating comparison...")
    comparison = save_comparison(logreg_preds, bert_preds, logreg_total_time, bert_total_time)

    print("\nEvaluating model size...")
    model_evaluation = evaluate_model_size()

    print("\n=== Comparison Results ===")
    print(f"Logistic Regression Accuracy: {comparison['logreg']['accuracy']:.4f}")
    print(f"BERT Accuracy: {comparison['bert']['accuracy']:.4f}")
    print(f"Accuracy Difference (BERT - LogReg): {comparison['accuracy_difference']:.4f}")

    print(f"\nLogistic Regression Avg Prediction Time: {comparison['logreg']['average_prediction_time']:.4f} seconds")
    print(f"BERT Avg Prediction Time: {comparison['bert']['average_prediction_time']:.4f} seconds")
    print(f"Time Ratio (BERT vs LogReg): {comparison['time_ratio_bert_vs_logreg']:.2f}x")

    print("\n=== Model Size Evaluation ===")
    print(f"BERT Model Size: {model_evaluation['total_size_gb']:.2f} GB")
    print(f"Meets File Size Requirement (<= 4 GB): {model_evaluation['meets_file_size_requirement']}")
    print(f"Note: Runtime memory requirement (<= 900 MB) cannot be fully verified without actual runtime testing")

    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()
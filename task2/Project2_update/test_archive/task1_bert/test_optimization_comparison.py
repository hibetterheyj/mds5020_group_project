import csv
import json
import requests
import time
import os
import psutil
import numpy as np
from datetime import datetime
import subprocess
import signal
import threading

# Configuration
DATA_FILE = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.csv'
NUM_SAMPLES = 100  # Reduce samples for faster testing
BASE_PORT = 5725
BERT_APP_SCRIPT = 'app_optimized.py'

# Create timestamped directory
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'./res/optimization_comparison_{current_time}'
os.makedirs(results_dir, exist_ok=True)

# Test configurations to evaluate
optimization_configs = [
    # (optimization_method, sequence_length)
    ('baseline', 512),      # Original configuration
    ('baseline', 256),      # Shorter sequence length
    ('baseline', 128),      # Even shorter sequence length
    ('torchscript', 512),   # TorchScript optimization
    ('torchscript', 256),   # TorchScript + shorter sequence
    ('dynamic_quant', 512), # Dynamic quantization
    ('dynamic_quant', 256), # Dynamic quantization + shorter sequence
]

def load_test_data(num_samples=NUM_SAMPLES):
    """Load test data from CSV"""
    test_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as csvfile:
        header = csvfile.readline().strip().split(',')
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

def start_bert_server(optimization_method, seq_length, port):
    """Start BERT server with specific optimization parameters"""
    env = os.environ.copy()
    env['OPTIMIZATION_METHOD'] = optimization_method
    env['REDUCE_SEQ_LENGTH'] = str(seq_length)
    env['PORT'] = str(port)
    
    process = subprocess.Popen(
        ['python', BERT_APP_SCRIPT],
        env=env,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if server is running
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        if response.status_code == 200:
            print(f"✓ Server started successfully on port {port} with {optimization_method}, seq_len={seq_length}")
            return process
    except requests.exceptions.RequestException:
        pass
    
    print(f"✗ Failed to start server on port {port}")
    process.kill()
    return None

def stop_bert_server(process):
    """Stop the BERT server process"""
    if process:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped")

def test_api(api_url, test_data, model_name):
    """Test API and return predictions with timing data"""
    predictions = []
    total_time = 0
    
    for sample in test_data:
        start_time = time.time()
        
        try:
            response = requests.post(api_url, json={'title': sample['news_title']}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                pred_time = time.time() - start_time
                total_time += pred_time
                
                predictions.append({
                    'doc_id': sample['doc_id'],
                    'news_title': sample['news_title'],
                    'actual': sample['sentiment'],
                    'predicted': result['prediction'],
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
    if not predictions:
        return 0.0
    correct = sum(1 for pred in predictions if pred['actual'] == pred['predicted'])
    return correct / len(predictions)

def measure_memory_usage(process):
    """Measure memory usage of the BERT server process"""
    try:
        p = psutil.Process(process.pid)
        memory_info = p.memory_info()
        return memory_info.rss / (1024 * 1024)  # Return in MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return -1

def save_predictions(predictions, filename):
    """Save predictions to CSV file"""
    if not predictions:
        return
    
    keys = predictions[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(predictions)

def main():
    print(f"Optimization Comparison Test Suite")
    print(f"Testing {len(optimization_configs)} configurations")
    print(f"Data samples: {NUM_SAMPLES}")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 60)
    
    # Load test data once
    print("Loading test data...")
    test_data = load_test_data(NUM_SAMPLES)
    print(f"Loaded {len(test_data)} samples")
    print("=" * 60)
    
    # Store all results
    all_results = []
    
    for i, (optimization_method, seq_length) in enumerate(optimization_configs):
        print(f"\nTest {i+1}/{len(optimization_configs)}:")
        print(f"Configuration: {optimization_method}, Sequence Length: {seq_length}")
        print("-" * 40)
        
        # Start server with current configuration
        port = BASE_PORT + i
        process = start_bert_server(optimization_method, seq_length, port)
        
        if not process:
            print(f"Skipping configuration: {optimization_method}, seq_len={seq_length}")
            continue
        
        try:
            # Measure memory usage
            memory_usage = measure_memory_usage(process)
            
            # Test the API
            api_url = f'http://localhost:{port}/predict_sentiment'
            model_name = f'{optimization_method}_seq{seq_length}'
            
            print(f"Testing API at {api_url}...")
            predictions, total_time = test_api(api_url, test_data, model_name)
            
            if not predictions:
                print("No predictions received")
                continue
            
            # Calculate metrics
            accuracy = calculate_accuracy(predictions)
            avg_pred_time = total_time / len(predictions)
            pred_times = [p['prediction_time'] for p in predictions]
            
            # Save individual predictions
            pred_filename = os.path.join(results_dir, f'{model_name}_predictions.csv')
            save_predictions(predictions, pred_filename)
            
            # Store result
            result = {
                'configuration': {
                    'optimization_method': optimization_method,
                    'sequence_length': seq_length
                },
                'performance': {
                    'accuracy': accuracy,
                    'total_prediction_time': total_time,
                    'average_prediction_time': avg_pred_time,
                    'min_prediction_time': min(pred_times),
                    'max_prediction_time': max(pred_times),
                    'median_prediction_time': np.median(pred_times),
                    'memory_usage_mb': memory_usage,
                    'samples_processed': len(predictions),
                    'samples_failed': NUM_SAMPLES - len(predictions)
                },
                'timestamp': current_time
            }
            
            all_results.append(result)
            
            # Print results
            print(f"Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Avg Prediction Time: {avg_pred_time:.4f}s")
            print(f"  Memory Usage: {memory_usage:.2f} MB")
            print(f"  Predictions saved to: {pred_filename}")
            
        finally:
            # Stop the server
            stop_bert_server(process)
    
    # Save all results to JSON
    results_filename = os.path.join(results_dir, 'optimization_results.json')
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Config':<35} {'Accuracy':<10} {'Avg Time (s)':<12} {'Memory (MB)':<12} {'Within Memory?':<12}")
    print("-" * 81)
    
    for result in all_results:
        config = result['configuration']
        perf = result['performance']
        config_name = f"{config['optimization_method']}, seq_len={config['sequence_length']}"
        within_memory = perf['memory_usage_mb'] <= 900 if perf['memory_usage_mb'] > 0 else 'N/A'
        
        print(f"{config_name:<35} {perf['accuracy']:<10.4f} {perf['average_prediction_time']:<12.4f} {perf['memory_usage_mb']:<12.2f} {within_memory:<12}")
    
    print("-" * 81)
    print(f"All results saved to: {results_filename}")
    print("Test completed!")

if __name__ == "__main__":
    main()

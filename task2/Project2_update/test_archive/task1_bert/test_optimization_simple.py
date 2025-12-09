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

# Configuration
DATA_FILE = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.csv'
NUM_SAMPLES = 100  # Number of samples to test
BERT_APP_SCRIPT = 'app_optimized.py'

# Create timestamped directory
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'./res/optimization_comparison_simple_{current_time}'
os.makedirs(results_dir, exist_ok=True)

# Test configurations
test_configs = [
    # (optimization_method, sequence_length, description)
    ('baseline', 512, 'Baseline model with default sequence length'),
    ('baseline', 256, 'Baseline model with reduced sequence length (256)'),
    ('baseline', 128, 'Baseline model with reduced sequence length (128)'),
    ('dynamic_quant', 512, 'Dynamic quantization with default sequence length'),
    ('dynamic_quant', 256, 'Dynamic quantization with reduced sequence length (256)'),
    ('dynamic_quant', 128, 'Dynamic quantization with reduced sequence length (128)'),
]

def load_test_data(num_samples):
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

def start_server(optimization_method, seq_length, port=5725):
    """Start the BERT server with specified configuration"""
    env = os.environ.copy()
    env['OPTIMIZATION_METHOD'] = optimization_method
    env['REDUCE_SEQ_LENGTH'] = str(seq_length)
    env['PORT'] = str(port)

    print(f"Starting server: {optimization_method}, seq_len={seq_length}...")
    process = subprocess.Popen(
        ['python', BERT_APP_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to start
    time.sleep(8)

    # Check if server is running
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        if response.status_code == 200:
            print("✓ Server started successfully")
            return process
        else:
            print(f"✗ Server returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to connect to server: {e}")

    # Print server error if available
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"Server error: {stderr_output[:500]}...")

    process.kill()
    return None

def stop_server(process):
    """Stop the server process"""
    if process:
        print("Stopping server...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
            print("✓ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("! Server didn't stop gracefully, killing it")
            process.kill()
            process.wait()
    time.sleep(3)  # Additional delay to ensure port is released

def test_server(port, test_data):
    """Test the server with provided test data"""
    print(f"Testing server on port {port} with {len(test_data)} samples...")

    predictions = []
    total_time = 0
    errors = 0

    for i, sample in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(test_data)}...")

        start_time = time.time()

        try:
            response = requests.post(
                f'http://localhost:{port}/predict_sentiment',
                json={'title': sample['news_title']},
                timeout=10
            )

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
                print(f"Error sample {i+1}: HTTP {response.status_code}")
                errors += 1
        except Exception as e:
            print(f"Exception sample {i+1}: {e}")
            errors += 1

    print(f"Test completed: {len(predictions)} successful, {errors} errors")
    return predictions, total_time

def calculate_metrics(predictions, total_time):
    """Calculate performance metrics"""
    if not predictions:
        return {}

    # Accuracy
    correct = sum(1 for p in predictions if p['actual'] == p['predicted'])
    accuracy = correct / len(predictions)

    # Timing metrics
    pred_times = [p['prediction_time'] for p in predictions]
    avg_time = total_time / len(predictions)

    return {
        'accuracy': accuracy,
        'total_time': total_time,
        'average_time': avg_time,
        'min_time': min(pred_times),
        'max_time': max(pred_times),
        'median_time': np.median(pred_times),
        'samples': len(predictions)
    }

def save_results(results):
    """Save all results to a JSON file"""
    with open(os.path.join(results_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def save_predictions(predictions, filename):
    """Save predictions to a CSV file"""
    if not predictions:
        return

    keys = predictions[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(predictions)

def main():
    print("=" * 60)
    print("BERT Optimization Comparison Test")
    print("=" * 60)
    print(f"Testing {len(test_configs)} configurations")
    print(f"Using {NUM_SAMPLES} data samples")
    print(f"Results directory: {results_dir}")
    print("=" * 60)

    # Load test data once
    print("Loading test data...")
    test_data = load_test_data(NUM_SAMPLES)
    print(f"Loaded {len(test_data)} samples")
    print("=" * 60)

    all_results = []
    port = 5725

    for i, (method, seq_len, desc) in enumerate(test_configs):
        print(f"\nTest {i+1}/{len(test_configs)}:")
        print(f"Configuration: {desc}")
        print(f"Details: method={method}, seq_len={seq_len}")
        print("-" * 40)

        # Start server
        process = start_server(method, seq_len, port)

        if not process:
            print(f"❌ Skipping configuration due to server startup failure")
            continue

        try:
            # Measure memory usage
            time.sleep(2)
            mem_usage = -1
            try:
                p = psutil.Process(process.pid)
                mem_usage = p.memory_info().rss / (1024 * 1024)  # Convert to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("⚠️  Could not measure memory usage")

            # Test server
            predictions, total_time = test_server(port, test_data)

            if not predictions:
                print("❌ No predictions received")
                continue

            # Calculate metrics
            metrics = calculate_metrics(predictions, total_time)

            # Save predictions
            pred_filename = os.path.join(results_dir, f"{method}_seq{seq_len}_predictions.csv")
            save_predictions(predictions, pred_filename)

            # Compile result
            result = {
                'config': {
                    'method': method,
                    'seq_len': seq_len,
                    'description': desc
                },
                'metrics': {
                    'accuracy': metrics['accuracy'],
                    'avg_pred_time': metrics['average_time'],
                    'total_time': metrics['total_time'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time'],
                    'median_time': metrics['median_time'],
                    'memory_mb': mem_usage,
                    'samples_processed': metrics['samples']
                },
                'timestamp': current_time
            }

            all_results.append(result)

            # Print summary
            print("\n📊 Test Results:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Avg Prediction Time: {metrics['average_time']:.4f} seconds")
            print(f"   Memory Usage: {mem_usage:.2f} MB")
            print(f"   Total Time: {metrics['total_time']:.2f} seconds")
            print(f"   Predictions saved to: {pred_filename}")
            print(f"   ✅ Configuration completed successfully")

        finally:
            # Ensure server is stopped
            stop_server(process)

    # Save all results
    if all_results:
        save_results(all_results)

        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Config':<25} {'Accuracy':<10} {'Avg Time (s)':<15} {'Memory (MB)':<12}")
        print("-" * 60)

        for result in all_results:
            config_name = f"{result['config']['method']}_seq{result['config']['seq_len']}"
            accuracy = result['metrics']['accuracy']
            avg_time = result['metrics']['avg_pred_time']
            memory = result['metrics']['memory_mb']
            print(f"{config_name:<25} {accuracy:<10.4f} {avg_time:<15.4f} {memory:<12.2f}")

        print("\n" + "=" * 60)
        print(f"All results saved to: {results_dir}")

        # Identify best configurations
        if all_results:
            # Best accuracy
            best_accuracy = max(all_results, key=lambda x: x['metrics']['accuracy'])
            # Fastest prediction time
            fastest_pred = min(all_results, key=lambda x: x['metrics']['avg_pred_time'])
            # Best balance (accuracy/time ratio)
            best_balance = max(all_results, key=lambda x: x['metrics']['accuracy'] / x['metrics']['avg_pred_time'])

            print("\n🏆 TOP PERFORMERS:")
            print(f"   Best Accuracy: {best_accuracy['config']['method']}_seq{best_accuracy['config']['seq_len']} ({best_accuracy['metrics']['accuracy']:.4f})")
            print(f"   Fastest Predictions: {fastest_pred['config']['method']}_seq{fastest_pred['config']['seq_len']} ({fastest_pred['metrics']['avg_pred_time']:.4f}s)")
            print(f"   Best Balance (Accuracy/Time): {best_balance['config']['method']}_seq{best_balance['config']['seq_len']} ({best_balance['metrics']['accuracy']:.4f}/{best_balance['metrics']['avg_pred_time']:.4f}s)")
    else:
        print("\n❌ No successful configurations tested")

    print("\n" + "=" * 60)
    print("Test completed")

if __name__ == "__main__":
    main()

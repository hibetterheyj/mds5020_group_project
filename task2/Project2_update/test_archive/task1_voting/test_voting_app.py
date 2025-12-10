from typing import List, Dict, Any

import json
import requests
import pandas as pd

# Configuration
TEST_PORT = 5726  # Port number for the voting classifier app

def load_test_data() -> List[Dict[str, Any]]:
    """
    Load the first 10 entries from the sentiment analysis training dataset.

    Returns:
        List[Dict[str, Any]]: List of test data records.
    """
    # Load sentiment analysis data
    sentiment_data_path = '../../../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    df_sentiment = pd.read_excel(sentiment_data_path)
    return df_sentiment.head(10).to_dict('records')


def test_sentiment_prediction(news_text: str) -> Dict[str, Any]:
    """
    Test the sentiment prediction endpoint and return the result.

    Args:
        news_text (str): News title text to predict sentiment for.

    Returns:
        Dict[str, Any]: Test result containing success status and prediction.
    """
    url = f"http://localhost:{TEST_PORT}/predict_sentiment"
    payload = {"text": news_text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return {
            'success': True,
            'prediction': result['sentiment'],
            'confidence': result['confidence']
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request error: {e}"
        }
    except KeyError as e:
        return {
            'success': False,
            'error': f"Invalid response format: {e}"
        }


def test_health_check() -> Dict[str, Any]:
    """
    Test the health check endpoint.

    Returns:
        Dict[str, Any]: Test result containing success status and health info.
    """
    url = f"http://localhost:{TEST_PORT}/health"

    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        return {
            'success': True,
            'status': result['status'],
            'model_loaded': result['model_loaded']
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request error: {e}"
        }


def test_model_info() -> Dict[str, Any]:
    """
    Test the model info endpoint.

    Returns:
        Dict[str, Any]: Test result containing success status and model info.
    """
    url = f"http://localhost:{TEST_PORT}/info"

    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        return {
            'success': True,
            'model_type': result['model_type'],
            'loaded': result['loaded'],
            'base_estimators': result.get('base_estimators', [])
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Request error: {e}"
        }


def main():
    """
    Main function to run all tests for the voting classifier app.
    """
    print("Testing Voting Classifier Flask API Endpoints")
    print("=" * 70)

    # Test health check first
    print("\n1. Testing Health Check Endpoint")
    print("-" * 70)
    health_result = test_health_check()
    if health_result['success']:
        print(f"   Status: {health_result['status']}")
        print(f"   Model loaded: {health_result['model_loaded']}")
        print("   Health check: ✓ PASSED")
    else:
        print(f"   Health check: ✗ FAILED - {health_result['error']}")
        return  # Exit if server is not running

    # Test model info
    print("\n2. Testing Model Info Endpoint")
    print("-" * 70)
    info_result = test_model_info()
    if info_result['success']:
        print(f"   Model type: {info_result['model_type']}")
        print(f"   Model loaded: {info_result['loaded']}")
        print(f"   Base estimators: {len(info_result['base_estimators'])}")
        for estimator in info_result['base_estimators']:
            print(f"      - {estimator}")
        print("   Model info: ✓ PASSED")
    else:
        print(f"   Model info: ✗ FAILED - {info_result['error']}")

    # Load test data
    test_data = load_test_data()

    # Test sentiment analysis endpoint
    print("\n3. Testing Sentiment Prediction Endpoint")
    print("=" * 70)
    print(f"{'Doc ID':<8} {'Title':<50} {'Actual':<8} {'Predicted':<10} {'Confidence':<12} {'Match':<8}")
    print("-" * 70)

    sentiment_correct = 0
    for item in test_data:
        result = test_sentiment_prediction(item['news_title'])
        if result['success']:
            actual = item['sentiment']
            predicted = result['prediction']
            match = "✓" if actual == predicted else "✗"
            if match == "✓":
                sentiment_correct += 1

            print(
                f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {actual:<8} {predicted:<10} {result['confidence']:<12.4f} {match:<8}")
        else:
            print(
                f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {'ERROR':<30} {result['error']}")

    sentiment_accuracy = sentiment_correct / len(test_data) * 100
    print("-" * 70)
    print(
        f"Sentiment Analysis Accuracy: {sentiment_accuracy:.2f}% ({sentiment_correct}/{len(test_data)})")

    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Test Summary")
    print("=" * 70)
    print(f"Health Check: {'✓ PASSED' if health_result['success'] else '✗ FAILED'}")
    print(f"Model Info: {'✓ PASSED' if info_result['success'] else '✗ FAILED'}")
    print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
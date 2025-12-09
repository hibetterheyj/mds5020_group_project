from typing import List, Dict, Any

import json
import requests
import pandas as pd

# Load first 10 entries from both training datasets
def load_test_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the first 10 entries from both sentiment analysis and topic classification training datasets.
    """
    # Load sentiment analysis data
    sentiment_data_path = '../data/Subtask1-sentiment_analysis/training_news-sentiment.xlsx'
    df_sentiment = pd.read_excel(sentiment_data_path)
    sentiment_test_data = df_sentiment.head(10).to_dict('records')

    # Load topic classification data
    topic_data_path = '../data/Subtask2-topic_classification/training_news-topic.xlsx'
    df_topic = pd.read_excel(topic_data_path)
    topic_test_data = df_topic.head(10).to_dict('records')

    return {
        'sentiment': sentiment_test_data,
        'topic': topic_test_data
    }


def test_sentiment_prediction(news_text: str) -> Dict[str, Any]:
    """
    Test the sentiment prediction endpoint and return the result.
    """
    url = "http://localhost:5724/predict_sentiment"
    payload = {"news_text": news_text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return {
            'success': True,
            'prediction': result['sentiment'],
            'probability': result['probability']
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


def test_topic_prediction(news_text: str) -> Dict[str, Any]:
    """
    Test the topic classification endpoint and return the result.
    """
    url = "http://localhost:5724/predict_topic"
    payload = {"news_text": news_text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return {
            'success': True,
            'topic': result['topic'],
            'probability': result['probability']
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


def main():
    """
    Main function to run all tests with the first 10 entries from training datasets.
    """
    print("Testing Flask API Endpoints with Training Data")
    print("=" * 70)

    # Load test data
    test_data = load_test_data()

    # Test sentiment analysis endpoint
    print("\n" + "=" * 70)
    print("Sentiment Analysis Endpoint Test Results")
    print("=" * 70)
    print(f"{'Doc ID':<8} {'Title':<50} {'Actual':<8} {'Predicted':<10} {'Probability':<12} {'Match':<8}")
    print("-" * 70)

    sentiment_correct = 0
    for item in test_data['sentiment']:
        result = test_sentiment_prediction(item['news_title'])
        if result['success']:
            actual = item['sentiment']
            predicted = result['prediction']
            match = "✓" if actual == predicted else "✗"
            if match == "✓":
                sentiment_correct += 1

            print(
                f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {actual:<8} {predicted:<10} {result['probability']:<12} {match:<8}")
        else:
            print(
                f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {'ERROR':<40} {result['error']}")

    sentiment_accuracy = sentiment_correct / len(test_data['sentiment']) * 100
    print("-" * 70)
    print(
        f"Sentiment Analysis Accuracy: {sentiment_accuracy:.2f}% ({sentiment_correct}/{len(test_data['sentiment'])})")

    # Test topic classification endpoint
    print("\n" + "=" * 70)
    print("Topic Classification Endpoint Test Results")
    print("=" * 70)
    print(f"{'Doc ID':<8} {'Title':<50} {'Actual Topic':<20} {'Predicted Topic':<15} {'Probability':<12} {'Match':<8}")
    print("-" * 70)

    # Create a reverse topic dictionary for display purposes
    topic_dictionary = {
        '上市保荐书': '1',
        '保荐/核查意见': '2',
        '公司章程': '3',
        '公司章程修订': '4',
        '关联交易': '5',
        '分配方案决议公告': '6',
        '分配方案实施': '7',
        '分配预案': '8',
        '半年度报告全文': '9',
        '发行保荐书': '10',
        '年度报告全文': '11',
        '年度报告摘要': '12',
        '独立董事候选人声明': '13',
        '独立董事提名人声明': '14',
        '独立董事述职报告': '15',
        '股东大会决议公告': '16',
        '诉讼仲裁': '17',
        '高管人员任职变动': '18'
    }
    reverse_topic_dict = {v: k for k, v in topic_dictionary.items()}

    topic_correct = 0
    for item in test_data['topic']:
        result = test_topic_prediction(item['news_title'])
        if result['success']:
            actual_topic = item['topic']
            actual_topic_id = topic_dictionary.get(actual_topic, actual_topic)
            predicted_topic_id = result['topic']
            match = "✓" if actual_topic_id == predicted_topic_id else "✗"
            if match == "✓":
                topic_correct += 1

            print(f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {actual_topic:<20} {predicted_topic_id:<15} {result['probability']:<12} {match:<8}")
        else:
            print(
                f"{item['doc_id']:<8} {item['news_title'][:45]:<50} {'ERROR':<40} {result['error']}")

    topic_accuracy = topic_correct / len(test_data['topic']) * 100
    print("-" * 70)
    print(
        f"Topic Classification Accuracy: {topic_accuracy:.2f}% ({topic_correct}/{len(test_data['topic'])})")

    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Test Summary")
    print("=" * 70)
    print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.2f}%")
    print(f"Topic Classification Accuracy: {topic_accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

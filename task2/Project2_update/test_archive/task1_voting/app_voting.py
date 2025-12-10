import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Local imports from utils.py
from utils import TextPreprocessorTransformer, create_sentiment_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voting_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Model path and configuration
MODEL_PATH = './model/voting_classifier.pkl'  # Updated path
PORT = 5726

# Global model variable
model = None
model_loaded = False


def load_model(model_path: str) -> Any:
    """
    Load the pre-trained model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Any: Loaded model object.
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_input(text: str) -> pd.DataFrame:
    """
    Preprocess the input text into the format expected by the model.

    Args:
        text (str): Input text to preprocess.

    Returns:
        pd.DataFrame: DataFrame with the input text and required columns.
    """
    # Create a DataFrame with the required columns
    # The model expects a DataFrame with at least 'news_title' column
    # Handcrafted features will be generated automatically by the pipeline
    df = pd.DataFrame({'news_title': [text]})
    return df


def predict_sentiment(text: str) -> Tuple[int, float]:
    """
    Predict sentiment for the given text.

    Args:
        text (str): Input text to analyze.

    Returns:
        Tuple[int, float]: Predicted sentiment (-1, 0, or 1) and confidence score.
    """
    global model, model_loaded

    if not model_loaded:
        model = load_model(MODEL_PATH)
        model_loaded = True

    try:
        # Preprocess the input - only provide the news_title column
        # The model's internal pipeline handles text preprocessing and feature generation
        input_df = preprocess_input(text)

        # Generate handcrafted features
        handcrafted_features = create_sentiment_features(input_df)

        # Merge handcrafted features with original DataFrame (keeping news_title)
        input_df = pd.concat([input_df, handcrafted_features], axis=1)

        # Predict sentiment
        prediction = model.predict(input_df)
        sentiment = int(prediction[0])

        # Map from model output [0, 1] to expected [-1, 1]
        sentiment = sentiment if sentiment == 1 else -1

        # Confidence: Since model uses hard voting, we can't get probabilities
        # Return 0.0 as confidence indicator
        confidence = 0.0

        logger.info(f"Prediction: text='{text[:50]}...', sentiment={sentiment}, confidence={confidence:.4f}")
        return sentiment, confidence

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_endpoint():
    """
    API endpoint for sentiment prediction.
    Expects JSON input with 'text' field.
    Returns JSON with 'sentiment' and 'confidence' fields.
    """
    try:
        # Get input data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid input. Please provide a "text" field in the request body.'}), 400

        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Invalid text input. Please provide a non-empty string.'}), 400

        # Predict sentiment
        sentiment, confidence = predict_sentiment(text)

        # Return results
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in predict_sentiment_endpoint: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/info', methods=['GET'])
def model_info():
    """
    Endpoint to get model information.
    """
    global model

    info = {
        'model_type': 'VotingClassifier_Hard',
        'loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    }

    if model_loaded and hasattr(model, 'named_estimators_'):
        info['base_estimators'] = list(model.named_estimators_.keys())

    return jsonify(info), 200


if __name__ == "__main__":
    # Load the model on startup
    try:
        model = load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        model_loaded = False

    # Start the server
    logger.info(f"Starting Flask server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
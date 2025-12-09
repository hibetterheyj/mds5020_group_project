from flask import Flask, request
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoConfig
)
import time
import os

app = Flask(__name__)
app.json.ensure_ascii = False
app.config['JSON_AS_ASCII'] = False

# Load the saved model checkpoint
CHECKPOINT_PATH = './results/checkpoint-645'

# Optimization parameters (can be set via environment variables)
OPTIMIZATION_METHOD = os.environ.get('OPTIMIZATION_METHOD', 'baseline')  # baseline, torchscript, dynamic_quant, static_quant
REDUCE_SEQ_LENGTH = int(os.environ.get('REDUCE_SEQ_LENGTH', 512))  # Reduce sequence length for faster processing


def load_model_with_optimization(optimization_method='baseline', seq_length=512):
    """Load and optimize the DistilBert model based on the specified method"""
    print(f"Loading model with {optimization_method} optimization...")

    # Load tokenizer with custom sequence length
    tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT_PATH)

    # Load base model with original config
    model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_PATH)

    # Set to CPU as GPU is not available
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Apply optimization based on method
    if optimization_method == 'dynamic_quant':
        # Set quantization engine - required for dynamic quantization
        torch.backends.quantized.engine = 'qnnpack'  # Use qnnpack for CPU

        # Apply dynamic quantization
        try:
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # Apply quantization to linear layers
                dtype=torch.qint8  # Quantize to int8
            )
            print("✓ Dynamic quantization applied successfully")
        except Exception as e:
            print(f"✗ Dynamic quantization failed: {e}")
            print("Continuing with baseline model")

    print(f"Model loaded with {optimization_method} optimization, sequence length: {seq_length}")
    return tokenizer, model, device


# Load model during app initialization
tokenizer, model, device = load_model_with_optimization(OPTIMIZATION_METHOD, REDUCE_SEQ_LENGTH)


def predict_sentiment_bert(text, tokenizer, model, device, seq_length=512):
    """Predict sentiment using the BERT model"""
    # Tokenize input with specified sequence length
    inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=seq_length)

    # Get input tensors
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        # Simple unified approach for all model types
        outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return prediction, probabilities


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    news_text = request.json.get('news_text')
    if news_text:
        prediction, probabilities = predict_sentiment_bert(news_text, tokenizer, model, device, REDUCE_SEQ_LENGTH)
        # Map 0 to -1 for negative sentiment to match dataset format
        mapped_prediction = -1 if prediction == 0 else 1
        probability_for_prediction = probabilities[0] if prediction == 0 else probabilities[1]
        return {
            'sentiment': str(mapped_prediction),
            'probability': f"{probability_for_prediction:.4f}"
        }

    return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "optimization": OPTIMIZATION_METHOD, "seq_length": REDUCE_SEQ_LENGTH}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5725))
    app.run(host='0.0.0.0', port=port)

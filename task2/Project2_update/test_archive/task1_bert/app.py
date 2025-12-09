from flask import Flask, request
import numpy as np
import torch
from datasets import Dataset

# Transformers imports
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

app = Flask(__name__)
app.json.ensure_ascii = False
app.config['JSON_AS_ASCII'] = False

# Load the saved model checkpoint
CHECKPOINT_PATH = './results/checkpoint-645'

def load_model():
    """Load the saved DistilBert model from checkpoint"""
    tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set to evaluation mode
    return tokenizer, model, device

# Load model during app initialization
tokenizer, model, device = load_model()

def tokenize_function(examples, tokenizer):
    """Tokenize input examples"""
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def predict_sentiment_bert(text, tokenizer, model, device):
    """Predict sentiment using the BERT model"""
    # Direct tokenization without using Dataset
    inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

    # Get input tensors
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.tolist()[0]

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    news_text = request.json.get('news_text')
    if news_text:
        prediction, probabilities = predict_sentiment_bert(news_text, tokenizer, model, device)
        # Map 0 to -1 for negative sentiment to match dataset format
        mapped_prediction = -1 if prediction == 0 else 1
        probability_for_prediction = probabilities[0] if prediction == 0 else probabilities[1]
        return {
            'sentiment': str(mapped_prediction),
            'probability': f"{probability_for_prediction:.4f}"
        }

    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5725)
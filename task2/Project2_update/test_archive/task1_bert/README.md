# BERT Sentiment Analysis API

This API uses a fine-tuned DistilBERT model for sentiment analysis of financial news titles.

## Files

- `app.py`: The Flask API application
- `requirements.txt`: Dependencies required to run the API

## Model

The API uses a fine-tuned DistilBERT model from the checkpoint located at:
`./results/checkpoint-645`

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server:

   ```bash
   python app.py
   ```

The server will start on port 5724.

## API Endpoints

### POST /predict_sentiment

Predicts the sentiment of a financial news title.

#### Request

```json
{
  "title": "Company reports strong quarterly earnings"
}
```

#### Response

```json
{
  "prediction": 1,
  "probability": "0.9947"
}
```

- `prediction`: Sentiment label (-1 = negative, 1 = positive)
- `probability`: Confidence score for the prediction

## Testing

You can test the API using curl:

```bash
curl -X POST http://localhost:5724/predict_sentiment -H 'Content-Type: application/json' -d '{"title": "Company reports strong quarterly earnings"}'
```

## Results

The model achieved a mean weighted F1-score of 0.8712 with 5-fold cross-validation, as shown in `res/distilbert_baseline_results.json`.

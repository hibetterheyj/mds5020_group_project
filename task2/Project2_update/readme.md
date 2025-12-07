# Final Project Task 2

## Deliveries

### 1) Docker Image URL in ACR
crpi-9vcwd7dr6qcdahci.cn-shenzhen.personal.cr.aliyuncs.com/goldfishz/data_mining:v1.0

### 2) 5-fold CV Weighted F1-scores on Training Data
- **Sentiment Analysis**: 0.8024127741384737 (standard deviation: 0.015061383790324021)
- **Topic Classification**: 0.982653106555011 (standard deviation: 0.001730632240072608)

## Code Structure

```
.
├── app.py                      # Main application file
├── dockerfile                  # Docker configuration for containerization
├── readme.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
├── subtask_1_model             # Sentiment analysis model
│   └── subtask_1_model
│       ├── F1-score.jpg        # Model performance visualization
│       ├── model.joblib        # Trained sentiment analysis model
│       └── subtask_1_model.py  # Model implementation
├── subtask_2_model             # Topic classification model
│   └── topic_model
│       └── Proj                # Topic model implementation
├── test_app.py                 # Application tests
├── test_cross_validation.json  # Cross-validation results
└── test_cross_validation.py    # Cross-validation test script
```

## Components

1. **app.py**: The main entry point of the application, likely handling API endpoints or user interactions.

2. **dockerfile**: Contains instructions for building a Docker image to run the application in a containerized environment.

3. **requirements.txt**: Lists all Python libraries and dependencies required to run the project.

4. **subtask_1_model/**: Houses the sentiment analysis model implementation, including:
   - Trained model file (`model.joblib`)
   - Model source code (`subtask_1_model.py`)
   - Performance visualization (`F1-score.jpg`)

5. **subtask_2_model/**: Contains the topic classification model implementation in the `topic_model/Proj` directory.

6. **test_app.py**: Tests for the main application functionality.

7. **test_cross_validation.json**: JSON file containing cross-validation results for both models.

8. **test_cross_validation.py**: Script used to perform cross-validation testing.

## Usage

To run the application, refer to the Docker instructions or install dependencies from `requirements.txt` and execute `app.py`.

To run the tests, execute the appropriate test script:
- `python test_app.py` for application tests
- `python test_cross_validation.py` for cross-validation tests

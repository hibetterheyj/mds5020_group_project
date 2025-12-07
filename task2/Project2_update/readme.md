# Final Project Task 2

This repository contains the implementation for Task 2 of the final project, which includes two main subtasks: sentiment analysis and topic classification.

## ACR Image URL
crpi-9vcwd7dr6qcdahci.cn-shenzhen.personal.cr.aliyuncs.com/goldfishz/data_mining:v1.0

## Code Structure

```
.
├── app.py                  # Main application file
├── dockerfile              # Docker configuration for containerization
├── readme.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
├── subtask_1_model         # Sentiment analysis model
│   └── subtask_1_model
│       ├── __pycache__     # Python cache files
│       ├── F1-score.jpg    # Model performance visualization
│       ├── model.joblib    # Trained sentiment analysis model
│       └── subtask_1_model.py  # Model implementation
└── subtask_2_model         # Topic classification model
    └── topic_model
        └── Proj            # Topic model implementation
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

## Usage

To run the application, refer to the Docker instructions or install dependencies from `requirements.txt` and execute `app.py`.

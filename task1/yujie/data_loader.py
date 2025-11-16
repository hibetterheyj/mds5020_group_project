import pandas as pd
import numpy as np

class DataLoader:
    """Load and prepare training and test datasets"""

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        """Load training and test datasets"""
        print("Loading training and test data...")

        # Load training data
        train_data = pd.read_csv(self.train_path)
        print(f"Training data shape: {train_data.shape}")

        # Load test data
        test_data = pd.read_csv(self.test_path)
        print(f"Test data shape: {test_data.shape}")

        return train_data, test_data

    def prepare_features(self, train_data, test_data):
        """Prepare features for modeling without creating new features"""
        print("Preparing features for modeling...")

        # Separate features and target
        X_train = train_data.drop('y', axis=1)
        y_train = train_data['y']
        X_test = test_data.copy()

        # Convert target to binary
        y_train_binary = (y_train == 'yes').astype(int)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Target distribution: {y_train_binary.value_counts(normalize=True)}")

        return X_train, y_train_binary, X_test
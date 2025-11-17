from typing import Tuple
import pandas as pd
import numpy as np
from pandas import DataFrame


class DataLoader:
    """Load and prepare training and test datasets"""

    def __init__(self, train_path: str, test_path: str) -> None:
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[DataFrame, DataFrame]:
        """Load training and test datasets from CSV files."""
        print("Loading training and test data...")

        train_data = pd.read_csv(self.train_path)
        print(f"Training data shape: {train_data.shape}")

        test_data = pd.read_csv(self.test_path)
        print(f"Test data shape: {test_data.shape}")

        return train_data, test_data

    def prepare_features(self, train_data: DataFrame, test_data: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Prepare features for modeling by separating features and target."""
        print("Preparing features for modeling...")

        # Separate features and target from training data
        X_train = train_data.drop('y', axis=1)
        y_train = train_data['y']
        X_test = test_data.copy()

        # Convert target variable to binary representation
        y_train_binary = (y_train == 'yes').astype(int)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(
            f"Target distribution: {y_train_binary.value_counts(normalize=True)}")

        return X_train, y_train_binary, X_test

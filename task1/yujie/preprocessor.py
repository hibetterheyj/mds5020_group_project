import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class RobustDataPreprocessor:
    """Handle data preprocessing with robust error handling"""

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoders: Dict[str,
                                  Union[LabelEncoder, Dict[str, int]]] = {}
        self.imputer: SimpleImputer = SimpleImputer(strategy='median')
        self.numerical_features: Optional[List[str]] = None
        self.categorical_features: Optional[List[str]] = None

    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical features"""
        numerical_features = X.select_dtypes(
            include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=['object']).columns.tolist()

        print(
            f"Numerical features ({len(numerical_features)}): {numerical_features}")
        print(
            f"Categorical features ({len(categorical_features)}): {categorical_features}")

        return numerical_features, categorical_features

    def clean_data_quality_issues(self, df: pd.DataFrame, numerical_features: List[str],
                                  categorical_features: List[str]) -> pd.DataFrame:
        """Handle various data quality issues including infinite values"""
        print("Cleaning data quality issues...")

        df_clean = df.copy()
        issues_found = 0

        # Handle infinite values in numerical features
        for feature in numerical_features:
            if feature in df_clean.columns:
                # Count and replace infinite values
                inf_mask = np.isinf(df_clean[feature])
                inf_count = inf_mask.sum()

                if inf_count > 0:
                    print(f"Found {inf_count} infinite values in {feature}")
                    # Replace with NaN for imputation
                    df_clean[feature] = df_clean[feature].replace(
                        [np.inf, -np.inf], np.nan)
                    issues_found += inf_count

        # Handle 'unknown' values in categorical features
        for feature in categorical_features:
            if feature in df_clean.columns:
                # Count 'unknown' values
                unknown_count = (df_clean[feature] == 'unknown').sum()
                if unknown_count > 0:
                    print(
                        f"Found {unknown_count} 'unknown' values in {feature}")
                    # Keep as 'unknown' - they will be encoded as a separate category

        if issues_found == 0:
            print("No data quality issues found")
        else:
            print(f"Total data quality issues handled: {issues_found}")

        return df_clean

    def preprocess_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data with robust error handling"""
        print("Preprocessing training data...")

        X_train_processed = X_train.copy()
        self.numerical_features, self.categorical_features = self.identify_feature_types(
            X_train_processed)

        # Clean data quality issues
        X_train_processed = self.clean_data_quality_issues(
            X_train_processed, self.numerical_features, self.categorical_features
        )

        # Handle numerical features
        if self.numerical_features:
            print("Processing numerical features...")
            # Impute missing values (including those from cleaned infinite values)
            X_train_processed[self.numerical_features] = self.imputer.fit_transform(
                X_train_processed[self.numerical_features]
            )
            # Scale numerical features
            X_train_processed[self.numerical_features] = self.scaler.fit_transform(
                X_train_processed[self.numerical_features]
            )

        # Handle categorical features
        if self.categorical_features:
            print("Processing categorical features...")
            for feature in self.categorical_features:
                self.label_encoders[feature] = LabelEncoder()
                # Fit and transform categorical features
                try:
                    X_train_processed[feature] = self.label_encoders[feature].fit_transform(
                        X_train_processed[feature].astype(str)
                    )
                except Exception as e:
                    print(f"Error encoding {feature}: {e}")
                    # Fallback: simple integer encoding
                    unique_vals = X_train_processed[feature].astype(
                        str).unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    X_train_processed[feature] = X_train_processed[feature].map(
                        mapping)
                    self.label_encoders[feature] = mapping

        print(f"Processed training data shape: {X_train_processed.shape}")
        return X_train_processed

    def preprocess_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Preprocess test data using fitted transformers"""
        print("Preprocessing test data...")

        X_test_processed = X_test.copy()

        # Clean data quality issues
        X_test_processed = self.clean_data_quality_issues(
            X_test_processed, self.numerical_features, self.categorical_features
        )

        # Handle numerical features
        if self.numerical_features:
            X_test_processed[self.numerical_features] = self.imputer.transform(
                X_test_processed[self.numerical_features]
            )
            X_test_processed[self.numerical_features] = self.scaler.transform(
                X_test_processed[self.numerical_features]
            )

        # Handle categorical features
        if self.categorical_features:
            for feature in self.categorical_features:
                try:
                    if hasattr(self.label_encoders[feature], 'transform'):
                        # sklearn LabelEncoder
                        X_test_processed[feature] = self.label_encoders[feature].transform(
                            X_test_processed[feature].astype(str)
                        )
                    else:
                        # Fallback mapping
                        mapping = self.label_encoders[feature]
                        X_test_processed[feature] = X_test_processed[feature].astype(
                            str).map(mapping)
                        # Fill unmapped values with 0 (most common category)
                        X_test_processed[feature] = X_test_processed[feature].fillna(
                            0)
                except Exception as e:
                    print(f"Error transforming {feature}: {e}")
                    # Default to 0 for all values
                    X_test_processed[feature] = 0

        print(f"Processed test data shape: {X_test_processed.shape}")
        return X_test_processed

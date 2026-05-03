# This file prepares the dataset before model training.

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# These columns are used as input data for the model.
FEATURE_COLUMNS = [
    "Age",
    "Education",
    "Skills",
    "Interests",
    "Recommendation_Score",
]
TARGET_COLUMN = "Recommended_Career"
CATEGORICAL_COLUMNS = ["Education", "Skills", "Interests"]


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn raw data into model input X and target y."""
    # Make sure the dataset has all columns we need.
    _validate_columns(data)

    # X contains the input columns, and y contains the answer column.
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    encoders = {}

    # Change text columns into numbers so the model can use them.
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].fillna("Unknown").astype(str))

    # Change the target labels into numbers too.
    target_encoder = LabelEncoder()
    y = pd.Series(
        target_encoder.fit_transform(y.fillna("Unknown").astype(str)),
        name=TARGET_COLUMN,
        index=data.index,
    )
    encoders["target"] = target_encoder
    return X, y, encoders


def _validate_columns(data: pd.DataFrame) -> None:
    """Check that the dataset has all required columns."""
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))

    # Show a clear error if any needed column is missing.
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

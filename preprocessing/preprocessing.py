# This file prepares the dataset before model training.

# pandas is used for DataFrame and Series operations.
import pandas as pd
# LabelEncoder changes text labels into numeric values.
from sklearn.preprocessing import LabelEncoder


# FEATURE_COLUMNS lists the input columns used by the model.
FEATURE_COLUMNS = [
    "Age",
    "Education",
    "Skills",
    "Interests",
    "Recommendation_Score",
]
# TARGET_COLUMN stores the name of the output column to predict.
TARGET_COLUMN = "Recommended_Career"
# CATEGORICAL_COLUMNS stores the text-based input columns that need encoding.
CATEGORICAL_COLUMNS = ["Education", "Skills", "Interests"]


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Turn raw data into model input X, target y, and saved encoders."""
    # Check that all required columns are present in the dataset.
    _validate_columns(data)

    # X stores the input features used for training.
    X = data[FEATURE_COLUMNS].copy()
    # y stores the target values that the model must predict.
    y = data[TARGET_COLUMN].copy()

    # encoders stores fitted encoders for possible later use.
    encoders = {}

    # Loop through each text input column that needs to become numeric.
    for column in CATEGORICAL_COLUMNS:
        # Create a fresh encoder for the current column.
        encoder = LabelEncoder()
        # Fill missing values, convert to text, and encode into numbers.
        X[column] = encoder.fit_transform(X[column].fillna("Unknown").astype(str))

    # Create a separate encoder for the target labels.
    target_encoder = LabelEncoder()
    # Encode the target column and keep the original index and column name.
    y = pd.Series(
        target_encoder.fit_transform(y.fillna("Unknown").astype(str)),
        name=TARGET_COLUMN,
        index=data.index,
    )
    # Store the target encoder in the encoder dictionary.
    encoders["target"] = target_encoder
    # Return the processed features, processed target, and saved encoders.
    return X, y, encoders


def _validate_columns(data: pd.DataFrame) -> None:
    """Check that the dataset has all required columns."""
    # Combine the feature columns and target column into one required set.
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    # Find which required columns are missing from the dataset.
    missing_columns = sorted(required_columns.difference(data.columns))

    # Stop the program early if any important column is missing.
    if missing_columns:
        # Join missing column names into one readable message.
        missing = ", ".join(missing_columns)
        # Raise a clear error so the user knows what is wrong.
        raise ValueError(f"Missing required columns: {missing}")

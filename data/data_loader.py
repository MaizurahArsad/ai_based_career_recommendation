# This file loads the dataset from the data folder.

# Path is used to build a safe file path to the CSV dataset.
from pathlib import Path

# pandas is used to read the CSV file into a DataFrame.
import pandas as pd


# DATA_PATH stores the full path to the dataset file inside the data folder.
DATA_PATH = Path(__file__).resolve().parent / "AI-based Career Recommendation System (2).csv"


def load_data():
    """Load the career dataset and return it as a pandas DataFrame."""
    # Check whether the dataset file exists before trying to read it.
    if not DATA_PATH.exists():
        # Raise an error with the file path if the dataset is missing.
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    # Read the CSV file and store the result in a pandas DataFrame.
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Print a short message to confirm that the dataset was loaded.
    print("Dataset loaded successfully!")
    # Print the shape so the user can see the number of rows and columns.
    print(df.shape)

    # Return the loaded DataFrame to the caller.
    return df

# This file loads the dataset from the data folder.

from pathlib import Path

import pandas as pd


# This is the path to the CSV file.
DATA_PATH = Path(__file__).resolve().parent / "AI-based Career Recommendation System (2).csv"


def load_data():
    """Load the career dataset and return it as a pandas DataFrame."""
    # Stop the program if the CSV file is missing.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    #return pd.read_csv(DATA_PATH)
    # Read the CSV file into a table.
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    print("Dataset loaded successfully!")
    print(df.shape)

    return df

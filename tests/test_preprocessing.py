# This file tests if preprocessing creates X and y.

from data.data_loader import load_data
from preprocessing.preprocessing import preprocess_data


def test_preprocess_data():
    """Check that preprocessing creates X and y."""
    df = load_data()
    X, y, _ = preprocess_data(df)

    # Check that X and y are created.
    assert X is not None
    assert y is not None

    # Check that X and y have rows.
    assert X.shape[0] > 0
    assert y.shape[0] > 0

    # Check that X has columns.
    assert X.shape[1] > 0

    # X and y should have the same number of rows.
    assert X.shape[0] == y.shape[0]

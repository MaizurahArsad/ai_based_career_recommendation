# This file tests whether preprocessing creates usable X and y data.

# load_data is imported so the test can use the real dataset.
from data.data_loader import load_data
# preprocess_data is imported so the test can check the preprocessing step.
from preprocessing.preprocessing import preprocess_data


def test_preprocess_data():
    """Check that preprocessing creates X and y."""
    # Load the original dataset.
    df = load_data()
    # Preprocess the dataset into features and target.
    X, y, _ = preprocess_data(df)

    # Check that the feature DataFrame exists.
    assert X is not None
    # Check that the target Series exists.
    assert y is not None

    # Check that the feature data has rows.
    assert X.shape[0] > 0
    # Check that the target data has rows.
    assert y.shape[0] > 0

    # Check that the feature data has columns.
    assert X.shape[1] > 0

    # Check that features and target have the same number of rows.
    assert X.shape[0] == y.shape[0]

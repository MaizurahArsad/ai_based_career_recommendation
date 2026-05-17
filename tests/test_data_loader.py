# This file tests whether the dataset can be loaded correctly.

# load_data is imported so the test can call the real data loader.
from data.data_loader import load_data


def test_load_data():
    """Check that the dataset loads and has data."""
    # Load the dataset using the project data loader.
    df = load_data()

    # Check that a DataFrame object was returned.
    assert df is not None

    # Check that the DataFrame is not empty.
    assert not df.empty

    # Check that the DataFrame has at least one row.
    assert df.shape[0] > 0
    # Check that the DataFrame has at least one column.
    assert df.shape[1] > 0

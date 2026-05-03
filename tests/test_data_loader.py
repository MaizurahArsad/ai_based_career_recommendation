# This file tests if the dataset can be loaded.

from data.data_loader import load_data


def test_load_data():
    """Check that the dataset loads and has data."""
    df = load_data()

    # The dataframe should load successfully.
    assert df is not None

    # The dataframe should not be empty.
    assert not df.empty

    # The dataframe should have at least one row and one column.
    assert df.shape[0] > 0
    assert df.shape[1] > 0
